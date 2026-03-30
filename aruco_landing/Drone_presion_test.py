#!/usr/bin/env python3

import os
import csv
import math
from datetime import datetime

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped


class DronePresionTest(Node):

    def __init__(self):
        super().__init__('drone_presion_test')

        # -----------------------------
        # Parameters / tuning
        # -----------------------------
        self.takeoff_alt = 1.0
        self.post_takeoff_settle_sec = 5.0   # let it settle after takeoff
        self.capture_after_sec = 2.0         # after entering hover phase, wait this long before locking ref
        self.hover_test_duration_sec = 5.0   # hold/log duration
        self.log_rate_hz = 20.0              # logging / setpoint publish rate

        # -----------------------------
        # State
        # -----------------------------
        self.state = State()
        self.current_pose = None
        self.connected = False
        self.armed = False
        self.mode = ""

        self.hover_ref = None
        self.land_start_pose = None

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            10
        )

        self.setpoint_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            10
        )

        self.setmode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        while not self.setmode_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for /mavros/set_mode ...")

        while not self.arming_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for /mavros/cmd/arming ...")

        while not self.takeoff_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for /mavros/cmd/takeoff ...")

        # -----------------------------
        # Logging folder / files
        # -----------------------------
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.expanduser(f'~/uav/ros2_ws/src/aruco_landing/aruco_landing/precision_test_logs/{timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)

        self.hover_log_path = os.path.join(self.log_dir, 'hover_precision_test.csv')
        self.land_log_path = os.path.join(self.log_dir, 'landing_precision_test.csv')

        self.hover_file = open(self.hover_log_path, 'w', newline='')
        self.hover_writer = csv.writer(self.hover_file)
        self.hover_writer.writerow([
            't_sec',
            'ref_x', 'ref_y', 'ref_z',
            'x', 'y', 'z',
            'dx', 'dy', 'dz',
            'xy_error', 'xyz_error'
        ])

        self.land_file = open(self.land_log_path, 'w', newline='')
        self.land_writer = csv.writer(self.land_file)
        self.land_writer.writerow([
            't_sec',
            'land_start_x', 'land_start_y', 'land_start_z',
            'x', 'y', 'z',
            'dx_from_land_start', 'dy_from_land_start', 'dz_from_land_start',
            'xy_error_from_land_start', 'xyz_error_from_land_start',
            'mode', 'armed'
        ])

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------
    def state_callback(self, msg):
        self.state = msg
        self.connected = msg.connected
        self.armed = msg.armed
        self.mode = msg.mode

    def pose_callback(self, msg):
        self.current_pose = msg

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def wait_for_connection_and_pose(self):
        while rclpy.ok() and (not self.connected or self.current_pose is None):
            if not self.connected:
                self.get_logger().info("Waiting for FCU connection...")
            if self.current_pose is None:
                self.get_logger().info("Waiting for local pose...")
            rclpy.spin_once(self, timeout_sec=0.2)

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.setmode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is not None and result.mode_sent:
            self.get_logger().info(f"Mode changed to {mode}")
            return True

        self.get_logger().error(f"Failed to change mode to {mode}")
        return False

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is not None and result.success:
            self.get_logger().info("Drone armed successfully")
            return True

        self.get_logger().error("Failed to arm drone")
        return False

    def takeoff(self, altitude):
        req = CommandTOL.Request()
        req.min_pitch = 0.0
        req.yaw = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.altitude = altitude

        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is not None and result.success:
            self.get_logger().info(f"Takeoff command sent: {altitude:.2f} m")
            return True

        self.get_logger().error("Failed to send takeoff command")
        return False

    def copy_current_pose(self):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x = self.current_pose.pose.position.x
        p.pose.position.y = self.current_pose.pose.position.y
        p.pose.position.z = self.current_pose.pose.position.z
        p.pose.orientation = self.current_pose.pose.orientation
        return p

    def publish_pose_setpoint(self, pose_msg):
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_pub.publish(pose_msg)

    def sleep_with_spin(self, duration_sec):
        start = self.get_clock().now().nanoseconds / 1e9
        while rclpy.ok():
            now = self.get_clock().now().nanoseconds / 1e9
            if (now - start) >= duration_sec:
                break
            rclpy.spin_once(self, timeout_sec=0.02)

    def error_metrics(self, ref_pose, cur_pose):
        dx = cur_pose.pose.position.x - ref_pose.pose.position.x
        dy = cur_pose.pose.position.y - ref_pose.pose.position.y
        dz = cur_pose.pose.position.z - ref_pose.pose.position.z
        xy_error = math.sqrt(dx * dx + dy * dy)
        xyz_error = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dx, dy, dz, xy_error, xyz_error

    def run_hover_precision_test(self):
        self.get_logger().info("Starting hover precision test...")

        # Let vehicle settle after takeoff
        self.get_logger().info(f"Settling for {self.post_takeoff_settle_sec:.1f} s after takeoff...")
        self.sleep_with_spin(self.post_takeoff_settle_sec)

        # Wait a bit more before capturing the reference
        self.get_logger().info(f"Waiting additional {self.capture_after_sec:.1f} s before locking hover reference...")
        self.sleep_with_spin(self.capture_after_sec)

        # Capture reference hover point
        self.hover_ref = self.copy_current_pose()
        self.get_logger().info(
            f"Hover reference locked at "
            f"x={self.hover_ref.pose.position.x:.3f}, "
            f"y={self.hover_ref.pose.position.y:.3f}, "
            f"z={self.hover_ref.pose.position.z:.3f}"
        )

        dt = 1.0 / self.log_rate_hz
        start = self.get_clock().now().nanoseconds / 1e9

        while rclpy.ok():
            now = self.get_clock().now().nanoseconds / 1e9
            t = now - start
            if t >= self.hover_test_duration_sec:
                break

            self.publish_pose_setpoint(self.hover_ref)
            rclpy.spin_once(self, timeout_sec=dt)

            if self.current_pose is None:
                continue

            dx, dy, dz, xy_error, xyz_error = self.error_metrics(self.hover_ref, self.current_pose)

            self.hover_writer.writerow([
                f"{t:.4f}",
                f"{self.hover_ref.pose.position.x:.6f}",
                f"{self.hover_ref.pose.position.y:.6f}",
                f"{self.hover_ref.pose.position.z:.6f}",
                f"{self.current_pose.pose.position.x:.6f}",
                f"{self.current_pose.pose.position.y:.6f}",
                f"{self.current_pose.pose.position.z:.6f}",
                f"{dx:.6f}",
                f"{dy:.6f}",
                f"{dz:.6f}",
                f"{xy_error:.6f}",
                f"{xyz_error:.6f}",
            ])

            self.get_logger().info(
                f"[HOVER] t={t:.2f}s | "
                f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f} | "
                f"xy={xy_error:.3f} m"
            )

        self.hover_file.flush()
        self.get_logger().info("Hover precision test complete")

    def run_landing_precision_log(self):
        if self.current_pose is None:
            self.get_logger().error("No current pose available before landing")
            return False

        self.land_start_pose = self.copy_current_pose()
        self.get_logger().info(
            f"Landing start pose recorded at "
            f"x={self.land_start_pose.pose.position.x:.3f}, "
            f"y={self.land_start_pose.pose.position.y:.3f}, "
            f"z={self.land_start_pose.pose.position.z:.3f}"
        )

        if not self.set_mode("LAND"):
            return False

        start = self.get_clock().now().nanoseconds / 1e9
        dt = 1.0 / self.log_rate_hz

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=dt)

            if self.current_pose is None:
                continue

            now = self.get_clock().now().nanoseconds / 1e9
            t = now - start

            dx, dy, dz, xy_error, xyz_error = self.error_metrics(self.land_start_pose, self.current_pose)

            self.land_writer.writerow([
                f"{t:.4f}",
                f"{self.land_start_pose.pose.position.x:.6f}",
                f"{self.land_start_pose.pose.position.y:.6f}",
                f"{self.land_start_pose.pose.position.z:.6f}",
                f"{self.current_pose.pose.position.x:.6f}",
                f"{self.current_pose.pose.position.y:.6f}",
                f"{self.current_pose.pose.position.z:.6f}",
                f"{dx:.6f}",
                f"{dy:.6f}",
                f"{dz:.6f}",
                f"{xy_error:.6f}",
                f"{xyz_error:.6f}",
                self.mode,
                self.armed
            ])

            self.get_logger().info(
                f"[LAND] t={t:.2f}s | "
                f"dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f} | "
                f"xy={xy_error:.3f} m | mode={self.mode}"
            )

            if not self.armed:
                break

        self.land_file.flush()
        self.get_logger().info("Landing log complete")
        return True

    def close_logs(self):
        try:
            self.hover_file.close()
        except Exception:
            pass
        try:
            self.land_file.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = DronePresionTest()

    try:
        node.wait_for_connection_and_pose()

        if not node.set_mode("GUIDED"):
            return

        if not node.armed:
            if not node.arm():
                return

        if not node.takeoff(node.takeoff_alt):
            return

        node.run_hover_precision_test()
        node.run_landing_precision_log()

        node.get_logger().info(f"Hover log saved to: {node.hover_log_path}")
        node.get_logger().info(f"Landing log saved to: {node.land_log_path}")

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")

    finally:
        node.close_logs()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
