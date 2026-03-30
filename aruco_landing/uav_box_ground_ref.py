#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL


class UAVBoxGroundRef(Node):
    def __init__(self):
        super().__init__('uav_box_ground_ref')

        self.current_state = State()
        self.current_pose = None

        # Ground/start reference captured BEFORE takeoff
        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

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
            pose_qos
        )

        self.target_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            10
        )

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def wait_for_connection(self):
        while rclpy.ok() and not self.current_state.connected:
            self.get_logger().info('Waiting for FCU connection...')
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info('FCU connected')

    def wait_for_pose(self):
        while rclpy.ok() and self.current_pose is None:
            self.get_logger().info('Waiting for pose...')
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info('Pose received')

    def wait_with_spin(self, duration):
        start = time.time()
        while rclpy.ok() and (time.time() - start < duration):
            rclpy.spin_once(self, timeout_sec=0.05)

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return (0.0, 0.0, qz, qw)

    def angle_diff(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def set_mode(self, mode):
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None:
            self.get_logger().info(f'Set mode {mode} sent')
        return result

    def arm(self):
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        req = CommandBool.Request()
        req.value = True
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None:
            self.get_logger().info(f'Arming result: success={result.success}, result={result.result}')
        return result

    def takeoff(self, altitude):
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for takeoff service...')
        req = CommandTOL.Request()
        req.min_pitch = 0.0
        req.yaw = float(self.home_yaw if self.home_yaw is not None else 0.0)
        req.latitude = 0.0
        req.longitude = 0.0
        req.altitude = float(altitude)
        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None:
            self.get_logger().info(f'Takeoff result: success={result.success}, result={result.result}')
        return result

    def publish_target_pose(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        qx, qy, qz, qw = self.yaw_to_quaternion(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.target_pub.publish(msg)

    def capture_ground_reference(self, sample_time=2.0):
        self.get_logger().info('Capturing ground reference before takeoff... keep drone still')

        xs, ys, zs, yaws = [], [], [], []
        start = time.time()

        while rclpy.ok() and (time.time() - start < sample_time):
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_pose is None:
                continue

            xs.append(self.current_pose.pose.position.x)
            ys.append(self.current_pose.pose.position.y)
            zs.append(self.current_pose.pose.position.z)
            yaws.append(self.quaternion_to_yaw(self.current_pose.pose.orientation))
            time.sleep(0.05)

        if len(xs) < 5:
            self.get_logger().error('Not enough pose samples to capture ground reference')
            return False

        self.home_x = sum(xs) / len(xs)
        self.home_y = sum(ys) / len(ys)
        self.home_z = sum(zs) / len(zs)

        # circular mean for yaw
        mean_sin = sum(math.sin(y) for y in yaws) / len(yaws)
        mean_cos = sum(math.cos(y) for y in yaws) / len(yaws)
        self.home_yaw = math.atan2(mean_sin, mean_cos)

        self.get_logger().info(
            f'Ground reference locked | x={self.home_x:.2f}, y={self.home_y:.2f}, '
            f'z={self.home_z:.2f}, yaw={math.degrees(self.home_yaw):.1f} deg'
        )
        return True

    def stream_home_ground_setpoint(self, duration=2.0):
        if None in (self.home_x, self.home_y, self.home_z, self.home_yaw):
            return

        start = time.time()
        while rclpy.ok() and (time.time() - start < duration):
            self.publish_target_pose(self.home_x, self.home_y, self.home_z, self.home_yaw)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

    def mission_to_world(self, forward_m, left_m):
        dx_world = forward_m * math.cos(self.home_yaw) - left_m * math.sin(self.home_yaw)
        dy_world = forward_m * math.sin(self.home_yaw) + left_m * math.cos(self.home_yaw)
        return dx_world, dy_world

    def move_to_target_world(self, target_x, target_y, target_z, target_yaw,
                             hover_time=3.0, tolerance_xy=0.05, tolerance_z=0.10,
                             tolerance_yaw_deg=5.0, timeout=25.0):

        self.get_logger().info(
            f'Target -> x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, '
            f'yaw={math.degrees(target_yaw):.1f} deg'
        )

        start_time = time.time()
        tolerance_yaw = math.radians(tolerance_yaw_deg)

        while rclpy.ok():
            self.publish_target_pose(target_x, target_y, target_z, target_yaw)
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_pose is None:
                time.sleep(0.05)
                continue

            curr_x = self.current_pose.pose.position.x
            curr_y = self.current_pose.pose.position.y
            curr_z = self.current_pose.pose.position.z
            curr_yaw = self.quaternion_to_yaw(self.current_pose.pose.orientation)

            err_xy = math.sqrt((target_x - curr_x) ** 2 + (target_y - curr_y) ** 2)
            err_z = abs(target_z - curr_z)
            err_yaw = abs(self.angle_diff(target_yaw, curr_yaw))

            self.get_logger().info(
                f'Current x={curr_x:.2f}, y={curr_y:.2f}, z={curr_z:.2f} | '
                f'err_xy={err_xy:.2f}, err_z={err_z:.2f}, '
                f'err_yaw_deg={math.degrees(err_yaw):.1f}'
            )

            if err_xy <= tolerance_xy and err_z <= tolerance_z and err_yaw <= tolerance_yaw:
                self.get_logger().info('Target reached')
                hover_start = time.time()
                while rclpy.ok() and (time.time() - hover_start < hover_time):
                    self.publish_target_pose(target_x, target_y, target_z, target_yaw)
                    rclpy.spin_once(self, timeout_sec=0.05)
                    time.sleep(0.05)
                return True

            if time.time() - start_time > timeout:
                self.get_logger().warn('Move timeout')
                return False

            time.sleep(0.05)

        return False

    def move_to_mission_point(self, forward_m, left_m, altitude_m,
                              hover_time=3.0, tolerance_xy=0.05, timeout=25.0):
        dx_world, dy_world = self.mission_to_world(forward_m, left_m)
        target_x = self.home_x + dx_world
        target_y = self.home_y + dy_world
        target_z = self.home_z + altitude_m

        self.get_logger().info(
            f'Mission point | forward={forward_m:.2f}, left={left_m:.2f}'
        )

        return self.move_to_target_world(
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            target_yaw=self.home_yaw,
            hover_time=hover_time,
            tolerance_xy=tolerance_xy,
            timeout=timeout
        )


def main(args=None):
    rclpy.init(args=args)
    drone = UAVBoxGroundRef()

    try:
        drone.wait_for_connection()
        drone.wait_for_pose()

        # Capture x/y/z/yaw on the ground BEFORE takeoff
        if not drone.capture_ground_reference(sample_time=2.0):
            return

        # Send initial grounded setpoints using the captured yaw reference
        drone.stream_home_ground_setpoint(duration=2.0)

        drone.set_mode('GUIDED')
        drone.wait_with_spin(1.0)

        drone.arm()
        drone.wait_with_spin(1.0)

        # Takeoff to 1m above ground reference
        takeoff_alt = 1.0
        drone.takeoff(takeoff_alt)
        drone.wait_with_spin(5.0)

        # Force hold directly above home point at commanded yaw
        success = drone.move_to_target_world(
            target_x=drone.home_x,
            target_y=drone.home_y,
            target_z=drone.home_z + takeoff_alt,
            target_yaw=drone.home_yaw,
            hover_time=3.0,
            tolerance_xy=0.05,
            timeout=20.0
        )

        if not success:
            drone.get_logger().warn('Failed to stabilize above home after takeoff')

        # 1m x 1m box using ground reference and locked yaw
        box_points = [
            (1.0,  0.0),   # forward 1m
            (1.0, -1.0),   # right 1m
            (0.0, -1.0),   # back 1m
            (0.0,  0.0),   # left back to home
        ]

        for i, (fwd, left) in enumerate(box_points, start=1):
            drone.get_logger().info(f'Starting leg {i}')
            success = drone.move_to_mission_point(
                forward_m=fwd,
                left_m=left,
                altitude_m=takeoff_alt,
                hover_time=3.0,
                tolerance_xy=0.05,
                timeout=25.0
            )
            if not success:
                drone.get_logger().warn(f'Leg {i} failed or timed out')
                break

        # Re-center above exact home before landing
        drone.get_logger().info('Returning to exact home before landing...')
        drone.move_to_target_world(
            target_x=drone.home_x,
            target_y=drone.home_y,
            target_z=drone.home_z + takeoff_alt,
            target_yaw=drone.home_yaw,
            hover_time=3.0,
            tolerance_xy=0.05,
            timeout=20.0
        )

        drone.set_mode('LAND')
        drone.get_logger().info('Landing...')

    finally:
        drone.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
