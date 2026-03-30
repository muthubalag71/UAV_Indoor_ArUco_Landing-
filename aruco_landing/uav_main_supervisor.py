#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool, Int32


class UAVMainSupervisor(Node):
    def __init__(self):
        super().__init__('uav_main_supervisor')

        # Parameters
        self.declare_parameter('takeoff_altitude_m', 1.5)
        self.declare_parameter('phase_hover_sec', 3.0)
        self.declare_parameter('ground_ref_sample_time_sec', 2.0)
        self.declare_parameter('prestream_sec', 3.0)

        self.declare_parameter('search_tolerance_xy_m', 0.15)
        self.declare_parameter('search_tolerance_z_m', 0.2)
        self.declare_parameter('search_tolerance_yaw_deg', 10.0)

        self.declare_parameter('yaw_kp', 1.2)
        self.declare_parameter('yaw_rate_max', 0.8)

        self.takeoff_altitude_m = float(self.get_parameter('takeoff_altitude_m').value)
        self.phase_hover_sec = float(self.get_parameter('phase_hover_sec').value)
        self.ground_ref_sample_time_sec = float(self.get_parameter('ground_ref_sample_time_sec').value)
        self.prestream_sec = float(self.get_parameter('prestream_sec').value)

        self.search_tolerance_xy_m = float(self.get_parameter('search_tolerance_xy_m').value)
        self.search_tolerance_z_m = float(self.get_parameter('search_tolerance_z_m').value)
        self.search_tolerance_yaw_deg = float(self.get_parameter('search_tolerance_yaw_deg').value)

        self.yaw_kp = float(self.get_parameter('yaw_kp').value)
        self.yaw_rate_max = float(self.get_parameter('yaw_rate_max').value)

        # State
        self.current_state = State()
        self.current_pose = None

        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        self.search_target = None
        self.search_complete = False
        self.search_leg_index = -1

        self.aruco_detected = False
        self.aruco_cmd = TwistStamped()

        self.phase = 'INIT'
        self.phase_start_time = time.time()
        self.last_log_time = 0.0
        self.last_request_time = 0.0

        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, pose_qos)
        self.create_subscription(PoseStamped, '/search/target_pose', self.search_target_cb, 10)
        self.create_subscription(Bool, '/search/complete', self.search_complete_cb, 10)
        self.create_subscription(Int32, '/search/leg_index', self.search_leg_cb, 10)
        self.create_subscription(Bool, '/aruco/detected', self.aruco_detected_cb, 10)
        self.create_subscription(TwistStamped, '/aruco/cmd_vel_body', self.aruco_cmd_cb, 10)

        # Publishers
        self.pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.search_pause_pub = self.create_publisher(Bool, '/search/pause', 10)
        self.search_enable_pub = self.create_publisher(Bool, '/search/enable', 10)
        self.home_ref_pub = self.create_publisher(PoseStamped, '/mission/home_ref', 10)

        # Services (Asynchronous)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.timer = self.create_timer(0.05, self.timer_cb)
        self.get_logger().info('UAVMainSupervisor initialized (Non-blocking mode)')

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg

    def search_target_cb(self, msg):
        self.search_target = msg

    def search_complete_cb(self, msg):
        self.search_complete = msg.data

    def search_leg_cb(self, msg):
        self.search_leg_index = msg.data

    def aruco_detected_cb(self, msg):
        self.aruco_detected = msg.data

    def aruco_cmd_cb(self, msg):
        self.aruco_cmd = msg

    def set_phase(self, name):
        if self.phase != name:
            self.phase = name
            self.phase_start_time = time.time()
            self.get_logger().info(f'PHASE -> {name}')

    def phase_elapsed(self):
        return time.time() - self.phase_start_time

    def quaternion_to_yaw(self, q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def yaw_to_quaternion(self, yaw):
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def angle_diff(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def publish_position_target(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = self.yaw_to_quaternion(yaw)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qx, qy, qz, qw
        self.pos_pub.publish(msg)

    def publish_velocity_hold_yaw(self, vx, vy, vz=0.0):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        if self.current_pose and self.home_yaw is not None:
            curr_yaw = self.quaternion_to_yaw(self.current_pose.pose.orientation)
            yaw_err = self.angle_diff(self.home_yaw, curr_yaw)
            msg.angular.z = float(self.clamp(self.yaw_kp * yaw_err, -self.yaw_rate_max, self.yaw_rate_max))
        self.vel_pub.publish(msg)

    def request_mode(self, mode):
        if time.time() - self.last_request_time > 2.0:
            req = SetMode.Request()
            req.custom_mode = mode
            self.mode_client.call_async(req)
            self.last_request_time = time.time()
            self.get_logger().info(f'Requesting mode: {mode}')

    def request_arm(self, arm_bool):
        if time.time() - self.last_request_time > 2.0:
            req = CommandBool.Request()
            req.value = arm_bool
            self.arm_client.call_async(req)
            self.last_request_time = time.time()
            self.get_logger().info(f'Requesting Arm: {arm_bool}')

    def capture_ground_reference(self):
        if self.current_pose is None: return False
        self.home_x = self.current_pose.pose.position.x
        self.home_y = self.current_pose.pose.position.y
        self.home_z = self.current_pose.pose.position.z
        self.home_yaw = self.quaternion_to_yaw(self.current_pose.pose.orientation)
        self.get_logger().info(f'Home Locked: x={self.home_x:.2f}, z={self.home_z:.2f}')
        return True

    def timer_cb(self):
        # 1. INIT Phase: Wait for Connectivity
        if self.phase == 'INIT':
            if not self.current_state.connected or self.current_pose is None:
                if time.time() - self.last_log_time > 5.0:
                    self.get_logger().info('Waiting for FCU/Pose...')
                    self.last_log_time = time.time()
                return
            if self.capture_ground_reference():
                self.set_phase('PRESTREAM')
            return

        # 2. PRESTREAM: Send setpoints BEFORE switching mode
        if self.phase == 'PRESTREAM':
            self.publish_position_target(self.home_x, self.home_y, self.home_z, self.home_yaw)
            if self.phase_elapsed() > self.prestream_sec:
                if self.current_state.mode != 'GUIDED':
                    self.request_mode('GUIDED')
                else:
                    self.set_phase('ARMING')
            return

        # 3. ARMING: Arm the motors
        if self.phase == 'ARMING':
            self.publish_position_target(self.home_x, self.home_y, self.home_z, self.home_yaw)
            if not self.current_state.armed:
                self.request_arm(True)
            else:
                self.set_phase('TAKEOFF_CLIMB')
            return

        # 4. TAKEOFF_CLIMB: Increase Z
        if self.phase == 'TAKEOFF_CLIMB':
            target_z = self.home_z + self.takeoff_altitude_m
            self.publish_position_target(self.home_x, self.home_y, target_z, self.home_yaw)
            
            if self.current_pose and abs(self.current_pose.pose.position.z - target_z) < self.search_tolerance_z_m:
                self.set_phase('TAKEOFF_HOVER')
            return

        # 5. TAKEOFF_HOVER: Stabilize
        if self.phase == 'TAKEOFF_HOVER':
            self.publish_position_target(self.home_x, self.home_y, self.home_z + self.takeoff_altitude_m, self.home_yaw)
            if self.phase_elapsed() > self.phase_hover_sec:
                msg = Bool(); msg.data = True; self.search_enable_pub.publish(msg)
                msg_p = Bool(); msg_p.data = False; self.search_pause_pub.publish(msg_p)
                self.set_phase('SEARCH')
            return

        # 6. SEARCH: Follow external search targets
        if self.phase == 'SEARCH':
            if self.aruco_detected:
                msg = Bool(); msg.data = True; self.search_pause_pub.publish(msg)
                self.set_phase('ARUCO')
            elif self.search_target:
                self.pos_pub.publish(self.search_target)
            else:
                self.publish_position_target(self.home_x, self.home_y, self.home_z + self.takeoff_altitude_m, self.home_yaw)
            return

        # 7. ARUCO: Velocity Control based on detection
        if self.phase == 'ARUCO':
            if not self.aruco_detected:
                self.set_phase('SEARCH') # Lost target, go back to search
                return
            self.publish_velocity_hold_yaw(self.aruco_cmd.twist.linear.x, self.aruco_cmd.twist.linear.y)
            return

def main(args=None):
    rclpy.init(args=args)
    node = UAVMainSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
