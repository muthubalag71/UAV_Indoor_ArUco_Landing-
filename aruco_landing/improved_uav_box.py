#!/usr/bin/env python3

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL


class UAVBoxIndoorV2(Node):
    def __init__(self):
        super().__init__('uav_box_indoor_v2')

        # ----------------------------
        # Current state
        # ----------------------------
        self.current_state = State()
        self.current_pose = None

        # Optional upward rangefinder hook
        # Set self.ceiling_distance_m from another subscriber if you know the topic
        self.ceiling_distance_m = None

        # Ground/start reference captured BEFORE takeoff
        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        # ----------------------------
        # Parameters
        # ----------------------------
        self.takeoff_alt_m = 1.0
        self.leg_size_m = 1.0
        self.substep_m = 0.20

        self.tolerance_xy_m = 0.05
        self.tolerance_z_m = 0.08
        self.tolerance_yaw_deg = 4.0

        self.settle_time_s = 0.8
        self.hover_time_s = 2.0
        self.timeout_per_target_s = 20.0
        self.no_progress_timeout_s = 3.0

        self.min_progress_m = 0.015
        self.pose_sample_dt = 0.05

        # Ground stability acceptance
        self.max_ground_spread_xy_m = 0.04
        self.max_ground_spread_z_m = 0.05
        self.max_ground_spread_yaw_deg = 4.0

        # Optional ceiling safety
        self.enable_ceiling_safety = False
        self.min_ceiling_clearance_m = 0.50

        # ----------------------------
        # QoS
        # ----------------------------
        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # ----------------------------
        # Subscribers / publisher
        # ----------------------------
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

        # ----------------------------
        # Services
        # ----------------------------
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    # =========================================================
    # ROS callbacks
    # =========================================================
    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    # =========================================================
    # Utility
    # =========================================================
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
            time.sleep(0.02)

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

    def get_current_xyz_yaw(self):
        if self.current_pose is None:
            return None
        p = self.current_pose.pose.position
        q = self.current_pose.pose.orientation
        yaw = self.quaternion_to_yaw(q)
        return p.x, p.y, p.z, yaw

    def ceiling_is_safe(self):
        if not self.enable_ceiling_safety:
            return True
        if self.ceiling_distance_m is None:
            return True
        return self.ceiling_distance_m >= self.min_ceiling_clearance_m

    # =========================================================
    # MAVROS services
    # =========================================================
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
            self.get_logger().info(
                f'Arming result: success={result.success}, result={result.result}'
            )
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
            self.get_logger().info(
                f'Takeoff result: success={result.success}, result={result.result}'
            )
        return result

    # =========================================================
    # Setpoint publishing
    # =========================================================
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

    def stream_setpoint(self, x, y, z, yaw, duration=2.0):
        start = time.time()
        while rclpy.ok() and (time.time() - start < duration):
            self.publish_target_pose(x, y, z, yaw)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(self.pose_sample_dt)

    # =========================================================
    # Ground reference + stability
    # =========================================================
    def circular_mean(self, angles):
        mean_sin = sum(math.sin(a) for a in angles) / len(angles)
        mean_cos = sum(math.cos(a) for a in angles) / len(angles)
        return math.atan2(mean_sin, mean_cos)

    def circular_spread_deg(self, angles):
        if not angles:
            return 999.0
        mean = self.circular_mean(angles)
        diffs = [abs(math.degrees(self.angle_diff(a, mean))) for a in angles]
        return max(diffs) if diffs else 999.0

    def capture_ground_reference(self, sample_time=2.0):
        self.get_logger().info('Capturing ground reference before takeoff... keep drone still')

        xs, ys, zs, yaws = [], [], [], []
        start = time.time()

        while rclpy.ok() and (time.time() - start < sample_time):
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_pose is None:
                continue

            x, y, z, yaw = self.get_current_xyz_yaw()
            xs.append(x)
            ys.append(y)
            zs.append(z)
            yaws.append(yaw)
            time.sleep(self.pose_sample_dt)

        if len(xs) < 10:
            self.get_logger().error('Not enough pose samples to capture ground reference')
            return False

        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        spread_xy = math.hypot(spread_x, spread_y)
        spread_z = max(zs) - min(zs)
        spread_yaw_deg = self.circular_spread_deg(yaws)

        self.get_logger().info(
            f'Ground stability | spread_x={spread_x:.3f} m, '
            f'spread_y={spread_y:.3f} m, '
            f'spread_xy={spread_xy:.3f} m, '
            f'spread_z={spread_z:.3f} m, '
            f'spread_yaw={spread_yaw_deg:.2f} deg'
        )

        if spread_xy > self.max_ground_spread_xy_m:
            self.get_logger().error('Ground pose too unstable in XY. Abort.')
            return False
        if spread_z > self.max_ground_spread_z_m:
            self.get_logger().error('Ground pose too unstable in Z. Abort.')
            return False
        if spread_yaw_deg > self.max_ground_spread_yaw_deg:
            self.get_logger().error('Ground yaw too unstable. Abort.')
            return False

        self.home_x = sum(xs) / len(xs)
        self.home_y = sum(ys) / len(ys)
        self.home_z = sum(zs) / len(zs)
        self.home_yaw = self.circular_mean(yaws)

        self.get_logger().info(
            f'Ground reference locked | x={self.home_x:.2f}, y={self.home_y:.2f}, '
            f'z={self.home_z:.2f}, yaw={math.degrees(self.home_yaw):.1f} deg'
        )
        return True

    # =========================================================
    # Mission frame conversion
    # =========================================================
    def mission_to_world(self, forward_m, left_m):
        dx_world = forward_m * math.cos(self.home_yaw) - left_m * math.sin(self.home_yaw)
        dy_world = forward_m * math.sin(self.home_yaw) + left_m * math.cos(self.home_yaw)
        return dx_world, dy_world

    def mission_point_to_world(self, forward_m, left_m, altitude_m):
        dx_world, dy_world = self.mission_to_world(forward_m, left_m)
        target_x = self.home_x + dx_world
        target_y = self.home_y + dy_world
        target_z = self.home_z + altitude_m
        return target_x, target_y, target_z

    # =========================================================
    # Motion logic
    # =========================================================
    def move_to_target_world(
        self,
        target_x,
        target_y,
        target_z,
        target_yaw,
        hover_time=2.0,
        tolerance_xy=0.05,
        tolerance_z=0.08,
        tolerance_yaw_deg=4.0,
        timeout=20.0,
        settle_time=0.8
    ):
        self.get_logger().info(
            f'Target -> x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}, '
            f'yaw={math.degrees(target_yaw):.1f} deg'
        )

        tolerance_yaw = math.radians(tolerance_yaw_deg)
        start_time = time.time()
        settle_start = None

        best_err_xy = float('inf')
        last_progress_time = time.time()

        recent_positions = deque(maxlen=8)

        while rclpy.ok():
            if not self.ceiling_is_safe():
                self.get_logger().warn('Ceiling clearance unsafe. Holding current target.')
                self.publish_target_pose(target_x, target_y, target_z, target_yaw)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(self.pose_sample_dt)
                continue

            self.publish_target_pose(target_x, target_y, target_z, target_yaw)
            rclpy.spin_once(self, timeout_sec=0.05)

            state_now = self.get_current_xyz_yaw()
            if state_now is None:
                time.sleep(self.pose_sample_dt)
                continue

            curr_x, curr_y, curr_z, curr_yaw = state_now
            err_xy = math.hypot(target_x - curr_x, target_y - curr_y)
            err_z = abs(target_z - curr_z)
            err_yaw = abs(self.angle_diff(target_yaw, curr_yaw))

            recent_positions.append((curr_x, curr_y, curr_z, time.time()))

            # Progress watchdog
            if err_xy < (best_err_xy - self.min_progress_m):
                best_err_xy = err_xy
                last_progress_time = time.time()

            # Drift/stillness estimate while inside tolerance
            motion_span_xy = 999.0
            motion_span_z = 999.0
            if len(recent_positions) >= 4:
                xs = [p[0] for p in recent_positions]
                ys = [p[1] for p in recent_positions]
                zs = [p[2] for p in recent_positions]
                motion_span_xy = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                motion_span_z = max(zs) - min(zs)

            self.get_logger().info(
                f'Curr x={curr_x:.2f}, y={curr_y:.2f}, z={curr_z:.2f} | '
                f'err_xy={err_xy:.3f}, err_z={err_z:.3f}, '
                f'err_yaw_deg={math.degrees(err_yaw):.2f}, '
                f'motion_span_xy={motion_span_xy:.3f}'
            )

            in_tol = (
                err_xy <= tolerance_xy and
                err_z <= tolerance_z and
                err_yaw <= tolerance_yaw
            )

            still_enough = (
                motion_span_xy <= 0.03 and
                motion_span_z <= 0.04
            )

            if in_tol and still_enough:
                if settle_start is None:
                    settle_start = time.time()
                    self.get_logger().info('Inside tolerance and settling...')
                elif (time.time() - settle_start) >= settle_time:
                    self.get_logger().info('Target settled')

                    hover_start = time.time()
                    while rclpy.ok() and (time.time() - hover_start < hover_time):
                        self.publish_target_pose(target_x, target_y, target_z, target_yaw)
                        rclpy.spin_once(self, timeout_sec=0.05)
                        time.sleep(self.pose_sample_dt)

                    return True
            else:
                settle_start = None

            if (time.time() - last_progress_time) > self.no_progress_timeout_s:
                self.get_logger().warn('No progress watchdog triggered')
                return False

            if (time.time() - start_time) > timeout:
                self.get_logger().warn('Move timeout')
                return False

            time.sleep(self.pose_sample_dt)

        return False

    def move_to_target_segmented(
        self,
        target_x,
        target_y,
        target_z,
        target_yaw,
        substep_m=0.20
    ):
        state_now = self.get_current_xyz_yaw()
        if state_now is None:
            self.get_logger().error('No current pose available')
            return False

        curr_x, curr_y, curr_z, _ = state_now

        dx = target_x - curr_x
        dy = target_y - curr_y
        dz = target_z - curr_z

        distance_xyz = math.sqrt(dx * dx + dy * dy + dz * dz)
        steps = max(1, int(math.ceil(distance_xyz / substep_m)))

        self.get_logger().info(f'Segmented move with {steps} step(s)')

        for i in range(1, steps + 1):
            alpha = i / steps
            ix = curr_x + alpha * dx
            iy = curr_y + alpha * dy
            iz = curr_z + alpha * dz

            self.get_logger().info(f'  Substep {i}/{steps}')
            ok = self.move_to_target_world(
                target_x=ix,
                target_y=iy,
                target_z=iz,
                target_yaw=target_yaw,
                hover_time=0.3,
                tolerance_xy=self.tolerance_xy_m,
                tolerance_z=self.tolerance_z_m,
                tolerance_yaw_deg=self.tolerance_yaw_deg,
                timeout=self.timeout_per_target_s,
                settle_time=self.settle_time_s
            )
            if not ok:
                self.get_logger().warn(f'Substep {i}/{steps} failed')
                return False

        return True

    def move_to_mission_point(self, forward_m, left_m, altitude_m):
        target_x, target_y, target_z = self.mission_point_to_world(
            forward_m, left_m, altitude_m
        )

        self.get_logger().info(
            f'Mission point | forward={forward_m:.2f}, left={left_m:.2f}'
        )

        return self.move_to_target_segmented(
            target_x=target_x,
            target_y=target_y,
            target_z=target_z,
            target_yaw=self.home_yaw,
            substep_m=self.substep_m
        )

    # =========================================================
    # Mission
    # =========================================================
    def run_mission(self):
        self.wait_for_connection()
        self.wait_for_pose()

        if not self.capture_ground_reference(sample_time=2.0):
            return

        # Grounded pre-stream for GUIDED setpoint readiness
        self.stream_setpoint(
            self.home_x, self.home_y, self.home_z, self.home_yaw, duration=2.0
        )

        self.set_mode('GUIDED')
        self.wait_with_spin(1.0)

        self.arm()
        self.wait_with_spin(1.0)

        self.takeoff(self.takeoff_alt_m)
        self.wait_with_spin(5.0)

        # Recenter and settle after takeoff
        self.get_logger().info('Stabilizing above home after takeoff...')
        ok = self.move_to_target_segmented(
            target_x=self.home_x,
            target_y=self.home_y,
            target_z=self.home_z + self.takeoff_alt_m,
            target_yaw=self.home_yaw,
            substep_m=self.substep_m
        )

        if not ok:
            self.get_logger().warn('Failed to stabilize above home after takeoff')

        # 1 m x 1 m box in mission frame
        box_points = [
            (1.0,  0.0),   # forward
            (1.0, -1.0),   # right
            (0.0, -1.0),   # back
            (0.0,  0.0),   # left back home
        ]

        for i, (fwd, left) in enumerate(box_points, start=1):
            self.get_logger().info(f'========== LEG {i} ==========')
            ok = self.move_to_mission_point(
                forward_m=fwd,
                left_m=left,
                altitude_m=self.takeoff_alt_m
            )
            if not ok:
                self.get_logger().warn(f'Leg {i} failed or stalled')
                break

        self.get_logger().info('Returning to exact home before landing...')
        self.move_to_target_segmented(
            target_x=self.home_x,
            target_y=self.home_y,
            target_z=self.home_z + self.takeoff_alt_m,
            target_yaw=self.home_yaw,
            substep_m=self.substep_m
        )

        self.set_mode('LAND')
        self.get_logger().info('Landing...')


def main(args=None):
    rclpy.init(args=args)
    drone = UAVBoxIndoorV2()

    try:
        drone.run_mission()
    finally:
        drone.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
