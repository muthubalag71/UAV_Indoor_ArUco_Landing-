#!/usr/bin/env python3

import math
import time
import csv
import os
import sys
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from std_msgs.msg import Bool, String


class ArucoAxisCalibration(Node):
    def __init__(self):
        super().__init__('aruco_axis_calibration')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('takeoff_alt', 1.0)
        self.declare_parameter('arrival_tol_xy', 0.15)
        self.declare_parameter('arrival_tol_z', 0.20)
        self.declare_parameter('search_hold_sec', 1.0)

        self.declare_parameter('step_distance_m', 0.30)
        self.declare_parameter('settle_time_sec', 1.5)
        self.declare_parameter('pose_fresh_timeout_sec', 0.8)

        self.declare_parameter('use_accepted_gate', True)
        self.declare_parameter('log_csv_path', os.path.expanduser('~/uav_axis_calibration_log.csv'))

        self.takeoff_alt = float(self.get_parameter('takeoff_alt').value)
        self.arrival_tol_xy = float(self.get_parameter('arrival_tol_xy').value)
        self.arrival_tol_z = float(self.get_parameter('arrival_tol_z').value)
        self.search_hold_sec = float(self.get_parameter('search_hold_sec').value)

        self.step_distance_m = float(self.get_parameter('step_distance_m').value)
        self.settle_time_sec = float(self.get_parameter('settle_time_sec').value)
        self.pose_fresh_timeout_sec = float(self.get_parameter('pose_fresh_timeout_sec').value)

        self.use_accepted_gate = bool(self.get_parameter('use_accepted_gate').value)
        self.log_csv_path = str(self.get_parameter('log_csv_path').value)

        # -----------------------------
        # State
        # -----------------------------
        self.current_state = State()
        self.current_pose = None

        self.aruco_detected = False
        self.aruco_accepted = False
        self.guidance_phase = 'SEARCH'

        self.pose_hold = None
        self.pose_hold_time = 0.0

        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_yaw = 0.0

        self.search_points_body = [
            (1.0, 0.0),
            (1.0, -1.0),
            (0.0, -1.0),
            (0.0, 0.0),
        ]
        self.search_index = 0
        self.search_target_world = None
        self.search_target_start_time = None

        self.mission_state = 'WAIT_CONNECTION'
        self.detect_time = None
        self.hover_start_time = None

        self.axis_sequence = deque(['X', 'Y'])
        self.test_count = 0
        self.current_test = None
        self.pending_prompt_printed = False
        self.step_exec_start = None
        self.step_target_world = None

        # stdin confirmation queue
        self.input_queue = deque()
        self.input_thread = threading.Thread(target=self._stdin_worker, daemon=True)
        self.input_thread.start()

        # -----------------------------
        # ROS Interfaces
        # -----------------------------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos)

        self.create_subscription(Bool, '/aruco/detected', self.aruco_detected_cb, qos)
        self.create_subscription(Bool, '/aruco/accepted', self.aruco_accepted_cb, qos)
        self.create_subscription(Vector3Stamped, '/aruco/pose_hold', self.aruco_pose_hold_cb, qos)
        self.create_subscription(String, '/aruco/guidance_phase', self.aruco_phase_cb, qos)

        self.target_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for arming service...')
        while not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for set_mode service...')
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for takeoff service...')

        # -----------------------------
        # CSV log init
        # -----------------------------
        self._init_csv()

        # -----------------------------
        # Timer
        # -----------------------------
        self.timer = self.create_timer(0.1, self.step)

        self.get_logger().info('ArucoAxisCalibration started')
        self.get_logger().info(f'CSV log: {self.log_csv_path}')

    # =========================================================
    # Subscribers
    # =========================================================
    def state_cb(self, msg: State):
        self.current_state = msg

    def pose_cb(self, msg: PoseStamped):
        self.current_pose = msg

    def aruco_detected_cb(self, msg: Bool):
        self.aruco_detected = bool(msg.data)

    def aruco_accepted_cb(self, msg: Bool):
        self.aruco_accepted = bool(msg.data)

    def aruco_pose_hold_cb(self, msg: Vector3Stamped):
        self.pose_hold = msg
        self.pose_hold_time = time.time()

    def aruco_phase_cb(self, msg: String):
        self.guidance_phase = msg.data

    # =========================================================
    # Utilities
    # =========================================================
    def _stdin_worker(self):
        while True:
            try:
                line = sys.stdin.readline()
                if line == '':
                    time.sleep(0.1)
                    continue
                self.input_queue.append(line.strip().lower())
            except Exception:
                time.sleep(0.1)

    def _init_csv(self):
        need_header = not os.path.exists(self.log_csv_path)
        with open(self.log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow([
                    'timestamp',
                    'test_id',
                    'camera_axis',
                    'camera_error_before_x',
                    'camera_error_before_y',
                    'camera_error_before_z',
                    'body_move_forward_m',
                    'body_move_left_m',
                    'target_world_x',
                    'target_world_y',
                    'camera_error_after_x',
                    'camera_error_after_y',
                    'camera_error_after_z',
                    'delta_x',
                    'delta_y',
                    'primary_improved',
                    'cross_axis_changed',
                    'result_text'
                ])

    def log_test(self, row):
        with open(self.log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def now_sec(self):
        return time.time()

    def fresh_pose_hold_available(self):
        return self.pose_hold is not None and (self.now_sec() - self.pose_hold_time) <= self.pose_fresh_timeout_sec

    def get_pose_hold_xyz(self):
        if not self.fresh_pose_hold_available():
            return None
        return (
            float(self.pose_hold.vector.x),
            float(self.pose_hold.vector.y),
            float(self.pose_hold.vector.z),
        )

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_target_pose(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        half = yaw * 0.5
        msg.pose.orientation.w = math.cos(half)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(half)
        self.target_pub.publish(msg)

    def publish_hold_current(self):
        if self.current_pose is None:
            return
        p = self.current_pose.pose.position
        self.publish_target_pose(p.x, p.y, p.z, self.home_yaw)

    def world_from_body_step(self, body_forward_m, body_left_m):
        """
        Convert desired body-frame step into local/world XY target using locked home yaw.
        """
        if self.current_pose is None:
            return None

        p = self.current_pose.pose.position

        dx = body_forward_m * math.cos(self.home_yaw) - body_left_m * math.sin(self.home_yaw)
        dy = body_forward_m * math.sin(self.home_yaw) + body_left_m * math.cos(self.home_yaw)

        return (
            p.x + dx,
            p.y + dy,
            p.z
        )

    def distance_to_target(self, tx, ty, tz):
        if self.current_pose is None:
            return None
        p = self.current_pose.pose.position
        ex = tx - p.x
        ey = ty - p.y
        ez = tz - p.z
        return ex, ey, ez, math.sqrt(ex * ex + ey * ey), abs(ez)

    def reached_target(self, tx, ty, tz):
        d = self.distance_to_target(tx, ty, tz)
        if d is None:
            return False
        _, _, _, dxy, dz = d
        return dxy <= self.arrival_tol_xy and dz <= self.arrival_tol_z

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        ok = future.result() is not None and future.result().mode_sent
        self.get_logger().info(f'Set mode {mode}: {"OK" if ok else "FAILED"}')
        return ok

    def arm(self, arm_it=True):
        req = CommandBool.Request()
        req.value = arm_it
        future = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        ok = future.result() is not None and future.result().success
        self.get_logger().info(f'Arm {arm_it}: {"OK" if ok else "FAILED"}')
        return ok

    def takeoff(self, alt):
        req = CommandTOL.Request()
        req.altitude = float(alt)
        future = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        ok = future.result() is not None and future.result().success
        self.get_logger().info(f'Takeoff {alt:.2f}m: {"OK" if ok else "FAILED"}')
        return ok

    def capture_home_reference(self):
        if self.current_pose is None:
            return False
        p = self.current_pose.pose.position
        q = self.current_pose.pose.orientation
        self.home_x = p.x
        self.home_y = p.y
        self.home_z = p.z
        self.home_yaw = self.yaw_from_quaternion(q)
        self.get_logger().info(
            f'Home captured: x={self.home_x:.2f} y={self.home_y:.2f} z={self.home_z:.2f} '
            f'yaw={math.degrees(self.home_yaw):.1f} deg'
        )
        return True

    def next_search_target_world(self):
        if self.search_index >= len(self.search_points_body):
            self.search_index = 0

        fwd, left = self.search_points_body[self.search_index]
        dx = fwd * math.cos(self.home_yaw) - left * math.sin(self.home_yaw)
        dy = fwd * math.sin(self.home_yaw) + left * math.cos(self.home_yaw)

        return (
            self.home_x + dx,
            self.home_y + dy,
            self.takeoff_alt
        )

    def aruco_ready_for_interrupt(self):
        if self.use_accepted_gate:
            return self.aruco_detected and self.aruco_accepted
        return self.aruco_detected

    # =========================================================
    # Calibration logic
    # =========================================================
    def build_next_test(self):
        pose = self.get_pose_hold_xyz()
        if pose is None:
            self.get_logger().warn('No fresh /aruco/pose_hold yet')
            return None

        x_m, y_m, z_m = pose
        axis = self.axis_sequence[0]
        self.axis_sequence.rotate(-1)

        step = self.step_distance_m

        if axis == 'X':
            # trying to reduce camera x error
            body_forward = 0.0
            body_left = -step if x_m > 0.0 else step

            motion_text = 'left' if body_left > 0.0 else 'right'
            explain = (
                f'I am trying to reduce CAMERA X error.\n'
                f'Current camera pose: x={x_m:+.3f}, y={y_m:+.3f}, z={z_m:.3f}\n'
                f'Proposed body move: {step:.2f} m {motion_text}\n'
                f'Expectation: |camera x| should decrease after the move.'
            )
        else:
            # trying to reduce camera y error
            body_forward = -step if y_m > 0.0 else step
            body_left = 0.0

            motion_text = 'backward' if body_forward < 0.0 else 'forward'
            explain = (
                f'I am trying to reduce CAMERA Y error.\n'
                f'Current camera pose: x={x_m:+.3f}, y={y_m:+.3f}, z={z_m:.3f}\n'
                f'Proposed body move: {step:.2f} m {motion_text}\n'
                f'Expectation: |camera y| should decrease after the move.'
            )

        tgt = self.world_from_body_step(body_forward, body_left)
        if tgt is None:
            return None

        tx, ty, tz = tgt

        self.test_count += 1
        return {
            'test_id': self.test_count,
            'axis': axis,
            'before_x': x_m,
            'before_y': y_m,
            'before_z': z_m,
            'body_forward': body_forward,
            'body_left': body_left,
            'target_x': tx,
            'target_y': ty,
            'target_z': tz,
            'prompt_text': explain,
        }

    def evaluate_test(self, test, after_x, after_y, after_z):
        axis = test['axis']
        before_x = test['before_x']
        before_y = test['before_y']

        delta_x = after_x - before_x
        delta_y = after_y - before_y

        if axis == 'X':
            primary_before = abs(before_x)
            primary_after = abs(after_x)
            cross_before = abs(before_y)
            cross_after = abs(after_y)
        else:
            primary_before = abs(before_y)
            primary_after = abs(after_y)
            cross_before = abs(before_x)
            cross_after = abs(after_x)

        primary_improved = primary_after < primary_before
        cross_axis_changed = abs(cross_after - cross_before) > 0.05

        if primary_improved and not cross_axis_changed:
            result_text = f'GREEN: {axis} axis looks calibrated'
        elif primary_improved and cross_axis_changed:
            result_text = f'YELLOW: {axis} improved but cross-coupling observed'
        else:
            result_text = f'RED: {axis} did not improve; likely reversed or wrong mapping'

        return {
            'delta_x': delta_x,
            'delta_y': delta_y,
            'primary_improved': primary_improved,
            'cross_axis_changed': cross_axis_changed,
            'result_text': result_text,
        }

    # =========================================================
    # State machine
    # =========================================================
    def step(self):
        # Always keep publishing hold / targets when needed
        if self.mission_state == 'WAIT_CONNECTION':
            if self.current_state.connected:
                self.get_logger().info('FCU connected')
                self.mission_state = 'CAPTURE_HOME'
            return

        if self.mission_state == 'CAPTURE_HOME':
            if self.capture_home_reference():
                self.mission_state = 'SET_GUIDED'
            return

        if self.mission_state == 'SET_GUIDED':
            self.set_mode('GUIDED')
            self.mission_state = 'ARMING'
            return

        if self.mission_state == 'ARMING':
            self.arm(True)
            self.mission_state = 'TAKEOFF'
            return

        if self.mission_state == 'TAKEOFF':
            self.takeoff(self.takeoff_alt)
            self.hover_start_time = self.now_sec()
            self.mission_state = 'POST_TAKEOFF_HOLD'
            return

        if self.mission_state == 'POST_TAKEOFF_HOLD':
            self.publish_target_pose(self.home_x, self.home_y, self.takeoff_alt, self.home_yaw)
            if self.now_sec() - self.hover_start_time >= 5.0:
                self.search_target_world = self.next_search_target_world()
                self.search_target_start_time = self.now_sec()
                self.mission_state = 'SEARCH'
            return

        if self.mission_state == 'SEARCH':
            if self.aruco_ready_for_interrupt():
                self.get_logger().warn('ARUCO DETECTED -> STOP SEARCH -> HOVER')
                self.detect_time = self.now_sec()
                self.mission_state = 'DETECTION_HOVER'
                return

            tx, ty, tz = self.search_target_world
            self.publish_target_pose(tx, ty, tz, self.home_yaw)

            if self.reached_target(tx, ty, tz):
                if self.now_sec() - self.search_target_start_time >= self.search_hold_sec:
                    self.search_index += 1
                    self.search_target_world = self.next_search_target_world()
                    self.search_target_start_time = self.now_sec()
            return

        if self.mission_state == 'DETECTION_HOVER':
            self.publish_hold_current()
            if self.now_sec() - self.detect_time >= 2.0:
                self.current_test = self.build_next_test()
                self.pending_prompt_printed = False
                self.mission_state = 'CALIBRATION_PROMPT'
            return

        if self.mission_state == 'CALIBRATION_PROMPT':
            self.publish_hold_current()

            if self.current_test is None:
                self.current_test = self.build_next_test()
                if self.current_test is None:
                    return

            if not self.pending_prompt_printed:
                self.get_logger().info('========================================')
                for line in self.current_test['prompt_text'].split('\n'):
                    self.get_logger().info(line)
                self.get_logger().info('Type YES then Enter, or just press Enter to execute.')
                self.get_logger().info('Type SKIP then Enter to skip this step.')
                self.get_logger().info('Type DONE then Enter to finish calibration.')
                self.get_logger().info('========================================')
                self.pending_prompt_printed = True

            if len(self.input_queue) == 0:
                return

            cmd = self.input_queue.popleft()

            if cmd in ['', 'yes', 'y']:
                self.step_target_world = (
                    self.current_test['target_x'],
                    self.current_test['target_y'],
                    self.current_test['target_z'],
                )
                self.step_exec_start = self.now_sec()
                self.mission_state = 'EXECUTING_STEP'
                self.get_logger().warn(
                    f'Executing test #{self.current_test["test_id"]}: '
                    f'forward={self.current_test["body_forward"]:+.2f} m, '
                    f'left={self.current_test["body_left"]:+.2f} m'
                )
            elif cmd == 'skip':
                self.get_logger().warn('Skipping current test.')
                self.current_test = self.build_next_test()
                self.pending_prompt_printed = False
            elif cmd == 'done':
                self.get_logger().info('Calibration session finished by user.')
                self.mission_state = 'DONE'
            else:
                self.get_logger().warn(f'Unknown input "{cmd}". Use Enter/yes, skip, or done.')
            return

        if self.mission_state == 'EXECUTING_STEP':
            tx, ty, tz = self.step_target_world
            self.publish_target_pose(tx, ty, tz, self.home_yaw)

            if self.reached_target(tx, ty, tz):
                self.hover_start_time = self.now_sec()
                self.mission_state = 'SETTLING_AFTER_STEP'
            return

        if self.mission_state == 'SETTLING_AFTER_STEP':
            tx, ty, tz = self.step_target_world
            self.publish_target_pose(tx, ty, tz, self.home_yaw)

            if self.now_sec() - self.hover_start_time < self.settle_time_sec:
                return

            pose = self.get_pose_hold_xyz()
            if pose is None:
                self.get_logger().warn('No fresh pose after step; cannot evaluate yet.')
                return

            after_x, after_y, after_z = pose
            verdict = self.evaluate_test(self.current_test, after_x, after_y, after_z)

            self.get_logger().info('------------- STEP RESULT -------------')
            self.get_logger().info(
                f'Test #{self.current_test["test_id"]} axis={self.current_test["axis"]} '
                f'before: x={self.current_test["before_x"]:+.3f}, y={self.current_test["before_y"]:+.3f}, z={self.current_test["before_z"]:.3f}'
            )
            self.get_logger().info(
                f'after : x={after_x:+.3f}, y={after_y:+.3f}, z={after_z:.3f}'
            )
            self.get_logger().info(
                f'delta : dx={verdict["delta_x"]:+.3f}, dy={verdict["delta_y"]:+.3f}'
            )
            self.get_logger().info(verdict['result_text'])
            self.get_logger().info('---------------------------------------')

            self.log_test([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                self.current_test['test_id'],
                self.current_test['axis'],
                self.current_test['before_x'],
                self.current_test['before_y'],
                self.current_test['before_z'],
                self.current_test['body_forward'],
                self.current_test['body_left'],
                self.current_test['target_x'],
                self.current_test['target_y'],
                after_x,
                after_y,
                after_z,
                verdict['delta_x'],
                verdict['delta_y'],
                verdict['primary_improved'],
                verdict['cross_axis_changed'],
                verdict['result_text'],
            ])

            self.current_test = self.build_next_test()
            self.pending_prompt_printed = False
            self.mission_state = 'CALIBRATION_PROMPT'
            return

        if self.mission_state == 'DONE':
            self.publish_hold_current()
            return


def main(args=None):
    rclpy.init(args=args)
    node = ArucoAxisCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
