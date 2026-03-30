#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from std_msgs.msg import Bool # Added for detection listener

class UAVBoxArUcoIntercept(Node):
    def __init__(self):
        super().__init__('uav_box_aruco_intercept')

        self.current_state = State()
        self.current_pose = None
        self.aruco_detected = False # New state variable

        # Ground/start reference captured BEFORE takeoff
        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, pose_qos)
        
        # New: Listen to the guidance node
        self.aruco_sub = self.create_subscription(Bool, '/aruco/detected', self.aruco_callback, 10)

        self.target_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def aruco_callback(self, msg):
        """Updates the detection flag based on the guidance node."""
        self.aruco_detected = msg.data

    # --- Standard Math Helpers from Ground Ref Code ---
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

    # --- Service Wrappers ---
    def set_mode(self, mode):
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_mode service...')
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def arm(self):
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for arming service...')
        req = CommandBool.Request(value=True)
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def takeoff(self, altitude):
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for takeoff service...')
        req = CommandTOL.Request()
        req.yaw = float(self.home_yaw if self.home_yaw is not None else 0.0)
        req.altitude = float(altitude)
        future = self.takeoff_client.call_async(req)
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
        msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        self.target_pub.publish(msg)

    def capture_ground_reference(self, sample_time=2.0):
        self.get_logger().info('Capturing ground reference...')
        xs, ys, zs, yaws = [], [], [], []
        start = time.time()
        while rclpy.ok() and (time.time() - start < sample_time):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_pose:
                xs.append(self.current_pose.pose.position.x)
                ys.append(self.current_pose.pose.position.y)
                zs.append(self.current_pose.pose.position.z)
                yaws.append(self.quaternion_to_yaw(self.current_pose.pose.orientation))
        if len(xs) < 5: return False
        self.home_x, self.home_y, self.home_z = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
        self.home_yaw = math.atan2(sum(math.sin(y) for y in yaws), sum(math.cos(y) for y in yaws))
        return True

    def move_to_target_world(self, target_x, target_y, target_z, target_yaw,
                             tolerance_xy=0.05, timeout=25.0):
        """Original movement logic with added Detection Check."""
        start_time = time.time()
        while rclpy.ok():
            # NEW: Stop and Return status if marker is seen
            if self.aruco_detected:
                self.get_logger().warn('!!! ARUCO DETECTED - ABORTING SEARCH !!!')
                return "DETECTED"

            self.publish_target_pose(target_x, target_y, target_z, target_yaw)
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_pose:
                curr = self.current_pose.pose.position
                err_xy = math.sqrt((target_x - curr.x) ** 2 + (target_y - curr.y) ** 2)
                if err_xy <= tolerance_xy:
                    return "SUCCESS"

            if time.time() - start_time > timeout:
                return "TIMEOUT"
        return "EXIT"

def main(args=None):
    rclpy.init(args=args)
    drone = UAVBoxArUcoIntercept()

    try:
        # 1. Connection and Ground Ref
        while rclpy.ok() and not drone.current_state.connected:
            rclpy.spin_once(drone, timeout_sec=0.2)
        if not drone.capture_ground_reference(): return

        # 2. Takeoff sequence
        drone.set_mode('GUIDED')
        drone.arm()
        takeoff_alt = 1.0
        drone.takeoff(takeoff_alt)
        time.sleep(5.0)

        # 3. Mission Box Points
        box_points = [(1.0, 0.0), (1.0, -1.0), (0.0, -1.0), (0.0, 0.0)]
        found_marker = False

        for fwd, left in box_points:
            # Convert mission to world coordinates based on locked home_yaw
            dx = fwd * math.cos(drone.home_yaw) - left * math.sin(drone.home_yaw)
            dy = fwd * math.sin(drone.home_yaw) + left * math.cos(drone.home_yaw)
            
            # Execute move with intercept check
            status = drone.move_to_target_world(
                drone.home_x + dx, drone.home_y + dy, drone.home_z + takeoff_alt, drone.home_yaw
            )

            if status == "DETECTED":
                found_marker = True
                break
        
        # 4. End of Mission Behavior
        if found_marker:
            drone.get_logger().info("Stopping search. Hovering for 2s before landing...")
            time.sleep(2.0)
        else:
            drone.get_logger().info("Box complete (No marker). Returning home...")
            drone.move_to_target_world(drone.home_x, drone.home_y, drone.home_z + takeoff_alt, drone.home_yaw)

        # 5. Land
        drone.set_mode('LAND')
        drone.get_logger().info('Landing sequence active.')

    finally:
        drone.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
