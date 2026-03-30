#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, String


class ArucoAxisTestNode(Node):
    def __init__(self):
        super().__init__('aruco_axis_test_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('pulse_speed_mps', 0.12)
        self.declare_parameter('pulse_duration_sec', 2.0)
        self.declare_parameter('pause_duration_sec', 3.0)
        self.declare_parameter('startup_wait_sec', 18.0)

        self.pulse_speed_mps = float(self.get_parameter('pulse_speed_mps').value)
        self.pulse_duration_sec = float(self.get_parameter('pulse_duration_sec').value)
        self.pause_duration_sec = float(self.get_parameter('pause_duration_sec').value)
        self.startup_wait_sec = float(self.get_parameter('startup_wait_sec').value)

        # Publishers
        self.pub_detected = self.create_publisher(Bool, '/aruco/detected', 10)
        self.pub_pose_valid = self.create_publisher(Bool, '/aruco/pose_valid', 10)
        self.pub_accepted = self.create_publisher(Bool, '/aruco/accepted', 10)
        self.pub_land_ready = self.create_publisher(Bool, '/aruco/land_ready', 10)
        self.pub_phase = self.create_publisher(String, '/aruco/guidance_phase', 10)
        self.pub_cmd = self.create_publisher(TwistStamped, '/aruco/cmd_vel_body', 10)

        self.steps = [
            ('announce', 0.0, 0.0, 2.0, 'Trigger detect/accept and wait'),
            ('pulse',  +self.pulse_speed_mps, 0.0, self.pulse_duration_sec, '+vx test'),
            ('pause',  0.0, 0.0, self.pause_duration_sec, 'Pause after +vx'),
            ('pulse',  -self.pulse_speed_mps, 0.0, self.pulse_duration_sec, '-vx test'),
            ('pause',  0.0, 0.0, self.pause_duration_sec, 'Pause after -vx'),
            ('pulse',  0.0, +self.pulse_speed_mps, self.pulse_duration_sec, '+vy test'),
            ('pause',  0.0, 0.0, self.pause_duration_sec, 'Pause after +vy'),
            ('pulse',  0.0, -self.pulse_speed_mps, self.pulse_duration_sec, '-vy test'),
            ('pause',  0.0, 0.0, self.pause_duration_sec, 'Pause after -vy'),
        ]

        self.start_time = time.time()
        self.sequence_started = False
        self.current_step_index = 0
        self.step_start_time = None
        self.finished = False
        self.last_logged_step = -1

        self.timer = self.create_timer(0.05, self.timer_cb)

        self.get_logger().info('ArucoAxisTestNode ready')
        self.get_logger().info(
            f'Test settings | speed={self.pulse_speed_mps:.2f} m/s '
            f'pulse={self.pulse_duration_sec:.2f}s pause={self.pause_duration_sec:.2f}s '
            f'startup_wait={self.startup_wait_sec:.1f}s'
        )

    def publish_bool(self, pub, value: bool):
        msg = Bool()
        msg.data = value
        pub.publish(msg)

    def publish_phase(self, text: str):
        msg = String()
        msg.data = text
        self.pub_phase.publish(msg)

    def publish_cmd(self, vx: float, vy: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.pub_cmd.publish(msg)

    def publish_active_aruco_state(self):
        self.publish_bool(self.pub_detected, True)
        self.publish_bool(self.pub_pose_valid, True)
        self.publish_bool(self.pub_accepted, True)
        self.publish_bool(self.pub_land_ready, False)
        self.publish_phase('GUIDE')

    def publish_idle_aruco_state(self):
        self.publish_bool(self.pub_detected, False)
        self.publish_bool(self.pub_pose_valid, False)
        self.publish_bool(self.pub_accepted, False)
        self.publish_bool(self.pub_land_ready, False)
        self.publish_phase('SEARCH')
        self.publish_cmd(0.0, 0.0)

    def timer_cb(self):
        if self.finished:
            self.publish_idle_aruco_state()
            return

        elapsed = time.time() - self.start_time

        if not self.sequence_started:
            if elapsed < self.startup_wait_sec:
                self.publish_idle_aruco_state()
                return

            self.sequence_started = True
            self.current_step_index = 0
            self.step_start_time = time.time()
            self.get_logger().info('Starting axis test sequence')
            return

        self.publish_active_aruco_state()

        if self.current_step_index >= len(self.steps):
            self.get_logger().info('Axis test complete. Returning to idle state.')
            self.finished = True
            self.publish_idle_aruco_state()
            return

        step_type, vx, vy, duration, label = self.steps[self.current_step_index]
        step_elapsed = time.time() - self.step_start_time

        if self.current_step_index != self.last_logged_step:
            self.get_logger().info(f'Step {self.current_step_index}: {label}')
            self.last_logged_step = self.current_step_index

        if step_type == 'announce':
            self.publish_cmd(0.0, 0.0)
        elif step_type == 'pulse':
            self.publish_cmd(vx, vy)
        elif step_type == 'pause':
            self.publish_cmd(0.0, 0.0)
        else:
            self.publish_cmd(0.0, 0.0)

        if step_elapsed >= duration:
            self.current_step_index += 1
            self.step_start_time = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoAxisTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_idle_aruco_state()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
