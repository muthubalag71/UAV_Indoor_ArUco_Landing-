#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32


class BoxSearchNode(Node):
    def __init__(self):
        super().__init__('box_search_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('box_size_m', 1.0)
        self.declare_parameter('altitude_m', 1.0)
        self.declare_parameter('corner_hover_sec', 3.0)

        self.box_size_m = float(self.get_parameter('box_size_m').value)
        self.altitude_m = float(self.get_parameter('altitude_m').value)
        self.corner_hover_sec = float(self.get_parameter('corner_hover_sec').value)

        # -----------------------------
        # State
        # -----------------------------
        self.have_home_ref = False
        self.enabled = False
        self.paused = True

        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_yaw = 0.0

        self.targets = []
        self.leg_index = 0
        self.search_complete = False

        self.waiting_for_reach = True
        self.advance_after_time = None
        self.last_reached = False

        # -----------------------------
        # Topics
        # -----------------------------
        self.create_subscription(PoseStamped, '/mission/home_ref', self.home_ref_cb, 10)
        self.create_subscription(Bool, '/search/enable', self.enable_cb, 10)
        self.create_subscription(Bool, '/search/pause', self.pause_cb, 10)
        self.create_subscription(Bool, '/search/target_reached', self.target_reached_cb, 10)

        self.target_pub = self.create_publisher(PoseStamped, '/search/target_pose', 10)
        self.active_pub = self.create_publisher(Bool, '/search/active', 10)
        self.complete_pub = self.create_publisher(Bool, '/search/complete', 10)
        self.leg_pub = self.create_publisher(Int32, '/search/leg_index', 10)

        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('BoxSearchNode ready')

    # =========================================================
    # Helpers
    # =========================================================
    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return (0.0, 0.0, qz, qw)

    def mission_to_world(self, forward_m, left_m):
        dx_world = forward_m * math.cos(self.home_yaw) - left_m * math.sin(self.home_yaw)
        dy_world = forward_m * math.sin(self.home_yaw) + left_m * math.cos(self.home_yaw)
        return dx_world, dy_world

    def build_targets(self):
        b = self.box_size_m
        alt = self.altitude_m

        # Same box pattern as your current code:
        # (forward, left)
        mission_points = [
            (1.0 * b,  0.0 * b),   # forward
            (1.0 * b, -1.0 * b),   # right
            (0.0 * b, -1.0 * b),   # back
            (0.0 * b,  0.0 * b),   # return to home
        ]

        self.targets = []

        for forward_m, left_m in mission_points:
            dx_world, dy_world = self.mission_to_world(forward_m, left_m)

            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.pose.position.x = self.home_x + dx_world
            msg.pose.position.y = self.home_y + dy_world
            msg.pose.position.z = self.home_z + alt

            qx, qy, qz, qw = self.yaw_to_quaternion(self.home_yaw)
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw

            self.targets.append(msg)

        self.leg_index = 0
        self.search_complete = False
        self.waiting_for_reach = True
        self.advance_after_time = None

        self.get_logger().info(
            f'Built {len(self.targets)} search legs | home=({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f}) '
            f'yaw={math.degrees(self.home_yaw):.1f} deg'
        )

    def publish_status(self):
        active = Bool()
        active.data = self.enabled and self.have_home_ref and not self.search_complete
        self.active_pub.publish(active)

        complete = Bool()
        complete.data = self.search_complete
        self.complete_pub.publish(complete)

        leg = Int32()
        leg.data = self.leg_index if not self.search_complete else -1
        self.leg_pub.publish(leg)

    # =========================================================
    # Callbacks
    # =========================================================
    def home_ref_cb(self, msg):
        self.home_x = msg.pose.position.x
        self.home_y = msg.pose.position.y
        self.home_z = msg.pose.position.z
        self.home_yaw = self.quaternion_to_yaw(msg.pose.orientation)
        self.have_home_ref = True
        self.build_targets()

    def enable_cb(self, msg):
        self.enabled = msg.data
        if self.enabled:
            self.get_logger().info('Search enabled')
        else:
            self.get_logger().info('Search disabled')

    def pause_cb(self, msg):
        was_paused = self.paused
        self.paused = msg.data
        if self.paused and not was_paused:
            self.get_logger().info(f'Search paused at leg {self.leg_index}')
        elif (not self.paused) and was_paused:
            self.get_logger().info(f'Search resumed at leg {self.leg_index}')

    def target_reached_cb(self, msg):
        if msg.data and not self.last_reached:
            if (
                self.enabled and
                self.have_home_ref and
                (not self.paused) and
                (not self.search_complete) and
                self.waiting_for_reach
            ):
                self.waiting_for_reach = False
                self.advance_after_time = time.time() + self.corner_hover_sec
                self.get_logger().info(f'Leg {self.leg_index} reached, hovering before next leg')

        self.last_reached = msg.data

    # =========================================================
    # Timer
    # =========================================================
    def timer_cb(self):
        self.publish_status()

        if not self.enabled:
            return
        if not self.have_home_ref:
            return
        if self.search_complete:
            return
        if self.leg_index < 0 or self.leg_index >= len(self.targets):
            return

        # Always publish the current target, even if paused.
        # That freezes the search target and avoids jumps on resume.
        target = self.targets[self.leg_index]
        target.header.stamp = self.get_clock().now().to_msg()
        self.target_pub.publish(target)

        # Do not advance while paused.
        if self.paused:
            return

        if self.advance_after_time is not None and time.time() >= self.advance_after_time:
            self.leg_index += 1
            self.advance_after_time = None

            if self.leg_index >= len(self.targets):
                self.search_complete = True
                self.get_logger().info('Box search complete')
            else:
                self.waiting_for_reach = True
                self.get_logger().info(f'Advancing to leg {self.leg_index}')


def main(args=None):
    rclpy.init(args=args)
    node = BoxSearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
