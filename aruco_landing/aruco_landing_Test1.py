#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from std_msgs.msg import Bool


class UAVBoxArUcoIntercept(Node):
    def __init__(self):
        super().__init__('uav_box_aruco_intercept')

        self.current_state = State()
        self.current_pose = None
        self.aruco_detected = False
        self.aruco_pose = None

        # Ground/start reference captured BEFORE takeoff
        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, pose_qos)

        # Guidance listeners
        self.aruco_sub = self.create_subscription(Bool, '/aruco/detected', self.aruco_callback, 10)
        self.aruco_pose_sub = self.create_subscription(Vector3Stamped, '/aruco/pose_raw', self.aruco_pose_callback, 10)

        # Publishers
        self.target_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        # Services
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def aruco_callback(self, msg):
        self.aruco_detected = msg.data

    def aruco_pose_callback(self, msg):
        self.aruco_pose = msg.vector

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return (0.0, 0.0, qz, qw)

    def set_mode(self, mode):
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')
        req = SetMode.Request(custom_mode=mode)
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def publish_target_pose(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = self.yaw_to_quaternion(yaw)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.target_pub.publish(msg)

    def send_velocity(self, vx, vy):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = 0.0
        self.vel_pub.publish(msg)

    def move_to_target_world(self, target_x, target_y, target_z, target_yaw,
                             tolerance_xy=0.05, timeout=25.0):
        start_time = time.time()
        while rclpy.ok():
            if self.aruco_detected:
                return "DETECTED"

            self.publish_target_pose(target_x, target_y, target_z, target_yaw)
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_pose:
                curr = self.current_pose.pose.position
                if math.sqrt((target_x - curr.x) ** 2 + (target_y - curr.y) ** 2) <= tolerance_xy:
                    return "SUCCESS"

            if time.time() - start_time > timeout:
                return "TIMEOUT"

        return "EXIT"

    def align_sequential(self, kp=0.4, target_tol=0.04):
        """Sequential alignment with corrected sign based on flight test."""

        # Phase 1: Forward/Backward
        self.get_logger().info("Mission Status: Aligning X (Forward/Back)...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.aruco_pose:
                error_x = self.aruco_pose.y

                if abs(error_x) < target_tol:
                    self.send_velocity(0.0, 0.0)
                    break

                vx = kp * error_x
                vx = max(min(vx, 0.08), -0.08)
                self.send_velocity(vx, 0.0)

        # Phase 2: Left/Right
        self.get_logger().info("Mission Status: Aligning Y (Left/Right)...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.aruco_pose:
                error_y = self.aruco_pose.x

                if abs(error_y) < target_tol:
                    self.send_velocity(0.0, 0.0)
                    break

                vy = kp * error_y
                vy = max(min(vy, 0.08), -0.08)
                self.send_velocity(0.0, vy)


def main(args=None):
    rclpy.init(args=args)
    drone = UAVBoxArUcoIntercept()

    try:
        while rclpy.ok() and not drone.current_state.connected:
            rclpy.spin_once(drone, timeout_sec=0.2)

        # Capture reference
        drone.get_logger().info('Capturing ground reference...')
        xs, ys, zs, yaws = [], [], [], []
        start = time.time()
        while rclpy.ok() and (time.time() - start < 2.0):
            rclpy.spin_once(drone, timeout_sec=0.05)
            if drone.current_pose:
                p = drone.current_pose.pose
                xs.append(p.position.x)
                ys.append(p.position.y)
                zs.append(p.position.z)
                yaws.append(drone.quaternion_to_yaw(p.orientation))

        drone.home_x = sum(xs) / len(xs)
        drone.home_y = sum(ys) / len(ys)
        drone.home_z = sum(zs) / len(zs)
        drone.home_yaw = math.atan2(sum(math.sin(y) for y in yaws),
                                    sum(math.cos(y) for y in yaws))

        # Takeoff
        drone.set_mode('GUIDED')
        drone.arm_client.call_async(CommandBool.Request(value=True))
        takeoff_alt = 1.0
        drone.takeoff_client.call_async(
            CommandTOL.Request(altitude=takeoff_alt, yaw=float(drone.home_yaw))
        )
        time.sleep(5.0)

        # Box search
        box_points = [(1.0, 0.0), (1.0, -1.0), (0.0, -1.0), (0.0, 0.0)]
        found_marker = False

        for i, (fwd, left) in enumerate(box_points, start=1):
            drone.get_logger().info(f"Mission Status: Starting Leg {i}")
            dx = fwd * math.cos(drone.home_yaw) - left * math.sin(drone.home_yaw)
            dy = fwd * math.sin(drone.home_yaw) + left * math.cos(drone.home_yaw)

            status = drone.move_to_target_world(
                drone.home_x + dx,
                drone.home_y + dy,
                drone.home_z + takeoff_alt,
                drone.home_yaw
            )

            if status == "DETECTED":
                found_marker = True
                drone.get_logger().info("Mission Status: Marker Detected.")
                break

        if found_marker:
            drone.get_logger().info("Mission Status: Sequential Alignment Starting (4cm Tol).")
            time.sleep(1.0)
            drone.align_sequential(target_tol=0.04)
        else:
            drone.get_logger().info("Mission Status: Box Complete. Returning Home.")
            drone.move_to_target_world(
                drone.home_x,
                drone.home_y,
                drone.home_z + takeoff_alt,
                drone.home_yaw
            )

        drone.get_logger().info("Mission Status: Landing sequence active.")
        drone.set_mode('LAND')

    finally:
        drone.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
