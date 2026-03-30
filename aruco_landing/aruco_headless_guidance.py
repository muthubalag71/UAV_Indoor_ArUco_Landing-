#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped, TwistStamped
from std_msgs.msg import Bool, Float32, String


class ArucoHeadlessGuidance(Node):
    def __init__(self):
        super().__init__('aruco_headless_guidance')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('marker_id', -1)
        self.declare_parameter('marker_size_m', 0.10)

        # Acceptance / hold
        self.declare_parameter('accept_radial_pct', 18.0)
        self.declare_parameter('accept_confirm_sec', 0.5)
        self.declare_parameter('pose_hold_timeout_sec', 0.60)

        # Guidance gains / limits
        self.declare_parameter('kp_x', 0.8)
        self.declare_parameter('kp_y', 0.8)
        self.declare_parameter('max_vx', 0.25)
        self.declare_parameter('max_vy', 0.25)
        self.declare_parameter('deadband_m', 0.01)

        # Mapping controls for tomorrow's axis fix
        self.declare_parameter('swap_xy_for_body', False)
        self.declare_parameter('invert_vx', False)
        self.declare_parameter('invert_vy', False)

        # Land-ready logic
        self.declare_parameter('land_radius_m', 0.50)
        self.declare_parameter('land_hold_sec', 1.0)

        image_topic = str(self.get_parameter('image_topic').value)
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)

        self.accept_radial_pct = float(self.get_parameter('accept_radial_pct').value)
        self.accept_confirm_sec = float(self.get_parameter('accept_confirm_sec').value)
        self.pose_hold_timeout_sec = float(self.get_parameter('pose_hold_timeout_sec').value)

        self.kp_x = float(self.get_parameter('kp_x').value)
        self.kp_y = float(self.get_parameter('kp_y').value)
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.deadband_m = float(self.get_parameter('deadband_m').value)

        self.swap_xy_for_body = bool(self.get_parameter('swap_xy_for_body').value)
        self.invert_vx = bool(self.get_parameter('invert_vx').value)
        self.invert_vy = bool(self.get_parameter('invert_vy').value)

        self.land_radius_m = float(self.get_parameter('land_radius_m').value)
        self.land_hold_sec = float(self.get_parameter('land_hold_sec').value)

        # -----------------------------
        # Camera calibration
        # Replace with your own if updated
        # -----------------------------
        self.camera_matrix = np.array([
            [966.37143189, 0.0, 631.07007401],
            [0.0, 966.99413182, 338.23834988],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.array([
            0.07158245,
            -0.20095973,
            -0.00939044,
            -0.00098631,
            0.17590348
        ], dtype=np.float64)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector_params = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, image_topic, self.image_callback, qos)

        # -----------------------------
        # Publishers
        # -----------------------------
        self.pub_detected = self.create_publisher(Bool, '/aruco/detected', 10)
        self.pub_accepted = self.create_publisher(Bool, '/aruco/accepted', 10)
        self.pub_pose_valid = self.create_publisher(Bool, '/aruco/pose_valid', 10)

        self.pub_pose_raw = self.create_publisher(Vector3Stamped, '/aruco/pose_raw', 10)
        self.pub_pose_hold = self.create_publisher(Vector3Stamped, '/aruco/pose_hold', 10)

        self.pub_yaw_raw_deg = self.create_publisher(Float32, '/aruco/yaw_raw_deg', 10)
        self.pub_yaw_hold_deg = self.create_publisher(Float32, '/aruco/yaw_hold_deg', 10)

        self.pub_cmd_vel_body = self.create_publisher(TwistStamped, '/aruco/cmd_vel_body', 10)
        self.pub_land_ready = self.create_publisher(Bool, '/aruco/land_ready', 10)
        self.pub_phase = self.create_publisher(String, '/aruco/guidance_phase', 10)

        self.pub_radial_error_m = self.create_publisher(Float32, '/aruco/radial_error_m', 10)
        self.pub_radial_error_pct = self.create_publisher(Float32, '/aruco/radial_error_pct', 10)

        # -----------------------------
        # State / memory
        # -----------------------------
        self.last_valid_time = 0.0
        self.have_valid_estimate = False

        self.last_x_m = 0.0
        self.last_y_m = 0.0
        self.last_z_m = 0.0
        self.last_yaw_deg = 0.0

        self.accept_candidate_since = None
        self.land_candidate_since = None

        self.last_phase = 'SEARCH'
        self.last_log_time = 0.0

        self.get_logger().info(f'Subscribed to {image_topic}')
        self.get_logger().info('Publishing headless ArUco guidance topics')

    # =========================================================
    # Helpers
    # =========================================================
    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def extract_yaw_deg_from_rvec(self, rvec) -> float:
        rot_mat, _ = cv2.Rodrigues(rvec)
        yaw_rad = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        return math.degrees(yaw_rad)

    def radial_error_m(self, x_m: float, y_m: float) -> float:
        return math.sqrt(x_m * x_m + y_m * y_m)

    def radial_error_pct(self, x_m: float, y_m: float, z_m: float) -> float:
        z_safe = max(abs(z_m), 1e-6)
        return self.radial_error_m(x_m, y_m) / z_safe * 100.0

    def publish_bool(self, pub, value: bool):
        msg = Bool()
        msg.data = value
        pub.publish(msg)

    def publish_float(self, pub, value: float):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def publish_string(self, pub, value: str):
        msg = String()
        msg.data = value
        pub.publish(msg)

    def publish_pose(self, pub, header, x_m: float, y_m: float, z_m: float):
        msg = Vector3Stamped()
        msg.header = header
        msg.vector.x = float(x_m)
        msg.vector.y = float(y_m)
        msg.vector.z = float(z_m)
        pub.publish(msg)

    def publish_cmd_vel_body(self, header, x_m: float, y_m: float):
        """
        Provisional body-frame velocity suggestion from ArUco pose.
        Main supervisor is still the only node that commands the FC.

        Tomorrow, if the axis mapping is wrong, edit only these mapping params:
        - swap_xy_for_body
        - invert_vx
        - invert_vy
        """
        raw_vx = 0.0
        raw_vy = 0.0

        # Default idea:
        # reduce y_m with body x
        # reduce x_m with body y
        if abs(y_m) > self.deadband_m:
            raw_vx = -self.kp_x * y_m

        if abs(x_m) > self.deadband_m:
            raw_vy = -self.kp_y * x_m

        if self.swap_xy_for_body:
            raw_vx, raw_vy = raw_vy, raw_vx

        if self.invert_vx:
            raw_vx = -raw_vx

        if self.invert_vy:
            raw_vy = -raw_vy

        vx = self.clamp(raw_vx, -self.max_vx, self.max_vx)
        vy = self.clamp(raw_vy, -self.max_vy, self.max_vy)

        msg = TwistStamped()
        msg.header = header
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.pub_cmd_vel_body.publish(msg)

    # =========================================================
    # Main callback
    # =========================================================
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.aruco_dict,
            parameters=self.detector_params
        )

        detected = False
        valid_now = False
        chosen_index = None

        if ids is not None and len(ids) > 0:
            ids = ids.flatten()

            if self.marker_id == -1:
                chosen_index = 0
            else:
                for i, mid in enumerate(ids):
                    if int(mid) == self.marker_id:
                        chosen_index = i
                        break

        if chosen_index is not None:
            detected = True
            marker_corners = corners[chosen_index].astype(np.float32)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners],
                self.marker_size_m,
                self.camera_matrix,
                self.dist_coeffs
            )

            rvec = rvecs[0][0]
            tvec = tvecs[0][0]

            x_m = float(tvec[0])
            y_m = float(tvec[1])
            z_m = float(tvec[2])
            yaw_deg = float(self.extract_yaw_deg_from_rvec(rvec))

            self.last_x_m = x_m
            self.last_y_m = y_m
            self.last_z_m = z_m
            self.last_yaw_deg = yaw_deg
            self.last_valid_time = time.time()
            self.have_valid_estimate = True
            valid_now = True

            self.publish_pose(self.pub_pose_raw, msg.header, x_m, y_m, z_m)
            self.publish_float(self.pub_yaw_raw_deg, yaw_deg)

        using_hold = False
        if not valid_now and self.have_valid_estimate:
            if (time.time() - self.last_valid_time) <= self.pose_hold_timeout_sec:
                using_hold = True
            else:
                self.have_valid_estimate = False

        estimate_available = valid_now or using_hold

        self.publish_bool(self.pub_detected, detected)
        self.publish_bool(self.pub_pose_valid, estimate_available)

        accepted = False
        land_ready = False
        phase = 'SEARCH'

        if estimate_available:
            x_m = self.last_x_m
            y_m = self.last_y_m
            z_m = self.last_z_m
            yaw_deg = self.last_yaw_deg

            radial_m = self.radial_error_m(x_m, y_m)
            radial_pct = self.radial_error_pct(x_m, y_m, z_m)

            self.publish_pose(self.pub_pose_hold, msg.header, x_m, y_m, z_m)
            self.publish_float(self.pub_yaw_hold_deg, yaw_deg)
            self.publish_float(self.pub_radial_error_m, radial_m)
            self.publish_float(self.pub_radial_error_pct, radial_pct)

            if radial_pct <= self.accept_radial_pct:
                if self.accept_candidate_since is None:
                    self.accept_candidate_since = time.time()
                elif (time.time() - self.accept_candidate_since) >= self.accept_confirm_sec:
                    accepted = True
            else:
                self.accept_candidate_since = None

            if accepted:
                phase = 'GUIDE'
                self.publish_cmd_vel_body(msg.header, x_m, y_m)
            else:
                self.publish_cmd_vel_body(msg.header, 0.0, 0.0)

            if accepted and radial_m <= self.land_radius_m:
                if self.land_candidate_since is None:
                    self.land_candidate_since = time.time()
                elif (time.time() - self.land_candidate_since) >= self.land_hold_sec:
                    land_ready = True
                    phase = 'LAND_READY'
            else:
                self.land_candidate_since = None

        else:
            self.accept_candidate_since = None
            self.land_candidate_since = None
            self.publish_cmd_vel_body(msg.header, 0.0, 0.0)

        self.publish_bool(self.pub_accepted, accepted)
        self.publish_bool(self.pub_land_ready, land_ready)
        self.publish_string(self.pub_phase, phase)

        now = time.time()
        if now - self.last_log_time > 0.5 or phase != self.last_phase:
            if estimate_available:
                radial_m = self.radial_error_m(self.last_x_m, self.last_y_m)
                radial_pct = self.radial_error_pct(self.last_x_m, self.last_y_m, self.last_z_m)
                self.get_logger().info(
                    f'phase={phase} detected={detected} accepted={accepted} land_ready={land_ready} '
                    f'x={self.last_x_m:+.3f} y={self.last_y_m:+.3f} z={self.last_z_m:.3f} '
                    f'yaw={self.last_yaw_deg:+.2f}deg radial={radial_m:.3f}m ({radial_pct:.1f}%)'
                )
            else:
                self.get_logger().info(
                    f'phase={phase} detected={detected} accepted={accepted} land_ready={land_ready} no_estimate'
                )
            self.last_log_time = now
            self.last_phase = phase


def main(args=None):
    rclpy.init(args=args)
    node = ArucoHeadlessGuidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
