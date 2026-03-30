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
        self.aruco_pose_time = None
        self.aruco_detect_time = None

        # Ground/start reference captured BEFORE takeoff
        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, pose_qos)

        self.aruco_sub = self.create_subscription(Bool, '/aruco/detected', self.aruco_callback, 10)
        self.aruco_pose_sub = self.create_subscription(Vector3Stamped, '/aruco/pose_raw', self.aruco_pose_callback, 10)

        self.target_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

    def state_callback(self, msg):
        self.current_state = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def aruco_callback(self, msg):
        self.aruco_detected = msg.data
        if msg.data:
            self.aruco_detect_time = time.time()

    def aruco_pose_callback(self, msg):
        self.aruco_pose = msg.vector
        self.aruco_pose_time = time.time()

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return (0.0, 0.0, qz, qw)

    def wait_for_pose(self, timeout=5.0):
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_pose is not None:
                return True
        return False

    def call_service_blocking(self, client, request, service_name, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'Service not available: {service_name}')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError(f'Service call timed out: {service_name}')
        result = future.result()
        if result is None:
            raise RuntimeError(f'Service call returned no result: {service_name}')
        return result

    def set_mode(self, mode, verify_timeout=5.0):
        result = self.call_service_blocking(
            self.mode_client,
            SetMode.Request(custom_mode=mode),
            '/mavros/set_mode',
            timeout=5.0,
        )
        accepted = getattr(result, 'mode_sent', True)
        if not accepted:
            raise RuntimeError(f'Autopilot rejected mode change to {mode}')

        start = time.time()
        while rclpy.ok() and (time.time() - start) < verify_timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if getattr(self.current_state, 'mode', '') == mode:
                return True
        raise RuntimeError(f'Mode did not become {mode} within {verify_timeout:.1f}s')

    def arm(self, verify_timeout=5.0):
        result = self.call_service_blocking(
            self.arm_client,
            CommandBool.Request(value=True),
            '/mavros/cmd/arming',
            timeout=5.0,
        )
        accepted = getattr(result, 'success', True)
        if not accepted:
            raise RuntimeError('Autopilot rejected arming request')

        start = time.time()
        while rclpy.ok() and (time.time() - start) < verify_timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if getattr(self.current_state, 'armed', False):
                return True
        raise RuntimeError(f'Vehicle did not arm within {verify_timeout:.1f}s')

    def send_takeoff_request(self, relative_alt, yaw):
        result = self.call_service_blocking(
            self.takeoff_client,
            CommandTOL.Request(altitude=float(relative_alt), yaw=float(yaw)),
            '/mavros/cmd/takeoff',
            timeout=8.0,
        )
        accepted = getattr(result, 'success', True)
        if not accepted:
            raise RuntimeError('Autopilot rejected takeoff request')
        return True

    def publish_target_pose(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        _, _, qz, qw = self.yaw_to_quaternion(yaw)
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

    def hold_position_world(self, x, y, z, yaw, hold_time=1.0, publish_rate_hz=15.0):
        dt = 1.0 / max(publish_rate_hz, 1.0)
        start = time.time()
        while rclpy.ok() and (time.time() - start) < hold_time:
            self.publish_target_pose(x, y, z, yaw)
            t0 = time.time()
            rclpy.spin_once(self, timeout_sec=dt)
            remaining = dt - (time.time() - t0)
            if remaining > 0.0:
                time.sleep(remaining)

    def stop_and_hold_current_hover(self, hold_time=1.0, publish_rate_hz=15.0):
        if self.current_pose is None:
            return
        p = self.current_pose.pose.position
        yaw = self.quaternion_to_yaw(self.current_pose.pose.orientation)
        self.send_velocity(0.0, 0.0)
        self.hold_position_world(p.x, p.y, p.z, yaw, hold_time=hold_time, publish_rate_hz=publish_rate_hz)

    def hold_nominal_hover(self, z_target, yaw, hold_time=1.0, publish_rate_hz=15.0):
        """Loose hover around the nominal takeoff point/altitude. No tight acceptance gate."""
        self.send_velocity(0.0, 0.0)
        self.hold_position_world(
            self.home_x,
            self.home_y,
            z_target,
            yaw,
            hold_time=hold_time,
            publish_rate_hz=publish_rate_hz,
        )

    def has_fresh_aruco_measurement(self, max_age=0.25):
        if not self.aruco_detected:
            return False
        if self.aruco_pose is None or self.aruco_pose_time is None:
            return False
        return (time.time() - self.aruco_pose_time) <= max_age

    def get_aruco_body_errors(self, max_measurement_age=0.25):
        if not self.has_fresh_aruco_measurement(max_age=max_measurement_age):
            return None
        # Preserve original tested sign convention.
        error_x = float(self.aruco_pose.x)
        error_y = float(self.aruco_pose.y)
        return error_x, error_y

    def move_to_target_world(self, target_x, target_y, target_z, target_yaw,
                             tolerance_xy=0.20, timeout=25.0,
                             require_fresh_aruco=True, aruco_freshness=0.25):
        start_time = time.time()
        while rclpy.ok():
            if require_fresh_aruco and self.has_fresh_aruco_measurement(max_age=aruco_freshness):
                return 'DETECTED'

            self.publish_target_pose(target_x, target_y, target_z, target_yaw)
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_pose is not None:
                curr = self.current_pose.pose.position
                if math.sqrt((target_x - curr.x) ** 2 + (target_y - curr.y) ** 2) <= tolerance_xy:
                    return 'SUCCESS'

            if time.time() - start_time > timeout:
                return 'TIMEOUT'

        return 'EXIT'

    def align_xy_and_hold_before_land(
        self,
        kp=0.4,
        target_tol_xy=0.04,
        target_tol_r=0.055,
        hold_time=2.0,
        cmd_rate_hz=15.0,
        max_speed_xy=0.08,
        filter_alpha=0.25,
        max_delta_per_cycle=0.02,
        overall_timeout=30.0,
        marker_loss_timeout=0.75,
        max_measurement_age=0.25,
        pre_align_hover_time=1.0,
        pre_land_hover_time=1.0,
    ):
        self.get_logger().info('Mission Status: Simultaneous XY alignment starting.')

        self.stop_and_hold_current_hover(hold_time=pre_align_hover_time, publish_rate_hz=cmd_rate_hz)

        dt = 1.0 / max(cmd_rate_hz, 1.0)
        start_time = time.time()
        in_tol_since = None
        vx_cmd = 0.0
        vy_cmd = 0.0

        while rclpy.ok():
            loop_start = time.time()
            rclpy.spin_once(self, timeout_sec=dt)
            now = time.time()

            if (now - start_time) > overall_timeout:
                self.send_velocity(0.0, 0.0)
                self.stop_and_hold_current_hover(hold_time=0.5, publish_rate_hz=cmd_rate_hz)
                return 'TIMEOUT'

            errors = self.get_aruco_body_errors(max_measurement_age=max_measurement_age)
            if errors is None:
                stale_for = float('inf')
                if self.aruco_pose_time is not None:
                    stale_for = now - self.aruco_pose_time

                self.send_velocity(0.0, 0.0)
                in_tol_since = None

                if stale_for > marker_loss_timeout:
                    self.stop_and_hold_current_hover(hold_time=0.5, publish_rate_hz=cmd_rate_hz)
                    return 'MARKER_LOST'

                remaining = dt - (time.time() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)
                continue

            error_x, error_y = errors
            radial_error = math.sqrt(error_x ** 2 + error_y ** 2)

            vx_raw = kp * error_x
            vy_raw = kp * error_y

            speed_mag = math.sqrt(vx_raw ** 2 + vy_raw ** 2)
            if speed_mag > max_speed_xy and speed_mag > 1e-6:
                scale = max_speed_xy / speed_mag
                vx_raw *= scale
                vy_raw *= scale

            vx_filt = filter_alpha * vx_raw + (1.0 - filter_alpha) * vx_cmd
            vy_filt = filter_alpha * vy_raw + (1.0 - filter_alpha) * vy_cmd

            dvx = max(min(vx_filt - vx_cmd, max_delta_per_cycle), -max_delta_per_cycle)
            dvy = max(min(vy_filt - vy_cmd, max_delta_per_cycle), -max_delta_per_cycle)
            vx_cmd += dvx
            vy_cmd += dvy

            in_xy_tol = abs(error_x) < target_tol_xy and abs(error_y) < target_tol_xy
            in_r_tol = radial_error < target_tol_r

            if in_xy_tol and in_r_tol:
                self.send_velocity(0.0, 0.0)
                if in_tol_since is None:
                    in_tol_since = now
                    self.get_logger().info('Mission Status: In tolerance, starting hold timer.')
                elif (now - in_tol_since) >= hold_time:
                    self.stop_and_hold_current_hover(hold_time=pre_land_hover_time, publish_rate_hz=cmd_rate_hz)
                    final_errors = self.get_aruco_body_errors(max_measurement_age=max_measurement_age)
                    if final_errors is None:
                        return 'MARKER_LOST'
                    final_ex, final_ey = final_errors
                    final_r = math.sqrt(final_ex ** 2 + final_ey ** 2)
                    if abs(final_ex) < target_tol_xy and abs(final_ey) < target_tol_xy and final_r < target_tol_r:
                        self.get_logger().info('Mission Status: Alignment stable after pre-LAND hover. Ready to LAND.')
                        return 'ALIGNED'
                    self.get_logger().warn('Mission Status: Drifted out of tolerance during pre-LAND hover. Resuming alignment.')
                    in_tol_since = None
            else:
                in_tol_since = None
                self.send_velocity(vx_cmd, vy_cmd)

            remaining = dt - (time.time() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

        self.send_velocity(0.0, 0.0)
        return 'EXIT'


def main(args=None):
    rclpy.init(args=args)
    drone = UAVBoxArUcoIntercept()

    try:
        while rclpy.ok() and not drone.current_state.connected:
            rclpy.spin_once(drone, timeout_sec=0.2)

        if not drone.wait_for_pose(timeout=5.0):
            raise RuntimeError('No local pose received from /mavros/local_position/pose')

        drone.get_logger().info('Capturing ground reference...')
        xs, ys, zs, yaws = [], [], [], []
        start = time.time()
        while rclpy.ok() and (time.time() - start < 2.0):
            rclpy.spin_once(drone, timeout_sec=0.05)
            if drone.current_pose is not None:
                p = drone.current_pose.pose
                xs.append(p.position.x)
                ys.append(p.position.y)
                zs.append(p.position.z)
                yaws.append(drone.quaternion_to_yaw(p.orientation))

        if not xs:
            raise RuntimeError('No local pose received while capturing home reference.')

        drone.home_x = sum(xs) / len(xs)
        drone.home_y = sum(ys) / len(ys)
        drone.home_z = sum(zs) / len(zs)
        drone.home_yaw = math.atan2(sum(math.sin(y) for y in yaws), sum(math.cos(y) for y in yaws))

        takeoff_alt = 1.0
        nominal_takeoff_z = drone.home_z + takeoff_alt

        # Loose non-ArUco behavior: verify mode/arming and takeoff command acceptance,
        # but do not block mission start on a tight altitude gate.
        drone.set_mode('GUIDED')
        drone.arm()
        drone.send_takeoff_request(takeoff_alt, yaw=float(drone.home_yaw))

        # Old-style practical takeoff transition: timed wait + loose commanded hover.
        drone.get_logger().info('Mission Status: Waiting for takeoff climb (loose 1 m logic).')
        time.sleep(5.0)
        drone.hold_nominal_hover(
            z_target=nominal_takeoff_z,
            yaw=drone.home_yaw,
            hold_time=1.0,
            publish_rate_hz=15.0,
        )

        box_points = [(1.0, 0.0), (1.0, -1.0), (0.0, -1.0), (0.0, 0.0)]
        found_marker = False

        for i, (fwd, left) in enumerate(box_points, start=1):
            drone.get_logger().info(f'Mission Status: Starting Leg {i}')
            dx = fwd * math.cos(drone.home_yaw) - left * math.sin(drone.home_yaw)
            dy = fwd * math.sin(drone.home_yaw) + left * math.cos(drone.home_yaw)

            status = drone.move_to_target_world(
                drone.home_x + dx,
                drone.home_y + dy,
                nominal_takeoff_z,
                drone.home_yaw,
                tolerance_xy=0.20,
                timeout=25.0,
                require_fresh_aruco=True,
                aruco_freshness=0.25,
            )

            if status == 'DETECTED':
                found_marker = True
                drone.get_logger().info('Mission Status: Fresh marker detected.')
                break

        if found_marker:
            align_status = drone.align_xy_and_hold_before_land(
                kp=0.4,
                target_tol_xy=0.04,
                target_tol_r=0.055,
                hold_time=2.0,
                cmd_rate_hz=15.0,
                max_speed_xy=0.08,
                filter_alpha=0.25,
                max_delta_per_cycle=0.02,
                overall_timeout=30.0,
                marker_loss_timeout=0.75,
                max_measurement_age=0.25,
                pre_align_hover_time=1.0,
                pre_land_hover_time=1.0,
            )

            if align_status == 'ALIGNED':
                drone.get_logger().info('Mission Status: Landing sequence active from 1 m hover.')
                drone.set_mode('LAND')
            else:
                drone.get_logger().warn(
                    f'Mission Status: Visual alignment failed ({align_status}). Returning home before LAND.'
                )
                drone.move_to_target_world(
                    drone.home_x,
                    drone.home_y,
                    nominal_takeoff_z,
                    drone.home_yaw,
                    tolerance_xy=0.20,
                    timeout=25.0,
                    require_fresh_aruco=False,
                )
                drone.hold_nominal_hover(
                    z_target=nominal_takeoff_z,
                    yaw=drone.home_yaw,
                    hold_time=1.0,
                    publish_rate_hz=15.0,
                )
                drone.get_logger().info('Mission Status: Landing sequence active.')
                drone.set_mode('LAND')
        else:
            drone.get_logger().info('Mission Status: Box complete. Returning home.')
            drone.move_to_target_world(
                drone.home_x,
                drone.home_y,
                nominal_takeoff_z,
                drone.home_yaw,
                tolerance_xy=0.20,
                timeout=25.0,
                require_fresh_aruco=False,
            )
            drone.hold_nominal_hover(
                z_target=nominal_takeoff_z,
                yaw=drone.home_yaw,
                hold_time=1.0,
                publish_rate_hz=15.0,
            )
            drone.get_logger().info('Mission Status: Landing sequence active.')
            drone.set_mode('LAND')

    finally:
        drone.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

