#!/usr/bin/env python3

import math
import threading
import time
from enum import Enum

import cv2
import numpy as np
from flask import Flask, Response

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge


class DemoState(Enum):
    SEARCH = 0
    HOVER_CONFIRM_1 = 1
    MOVE_FIXED_FRAME = 2
    HOVER_CONFIRM_2 = 3
    YAW_ALIGN = 4
    FINE_POINTING = 5
    LAND_HOLD = 6
    LAND = 7


class DemoDetectionAndMovement(Node):
    def __init__(self):
        super().__init__('demo_detection_and_movement')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('marker_id', -1)
        self.declare_parameter('marker_size_m', 0.10)
        self.declare_parameter('web_host', '0.0.0.0')
        self.declare_parameter('web_port', 8090)

        image_topic = str(self.get_parameter('image_topic').value)
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)
        self.web_host = str(self.get_parameter('web_host').value)
        self.web_port = int(self.get_parameter('web_port').value)

        self.bridge = CvBridge()
        self.offset_pub = self.create_publisher(Vector3Stamped, '/aruco/demo_offset', 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos
        )

        # Camera calibration used for metric pose estimation
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

        self.latest_jpeg = None
        self.frame_lock = threading.Lock()

        self.state = DemoState.SEARCH
        self.state_start_time = time.time()

        # Timers
        self.hover_confirm_1_sec = 2.0
        self.hover_confirm_2_sec = 2.0
        self.land_hold_sec = 1.5

        # Detection / fallback
        self.last_valid_time = 0.0
        self.pose_hold_timeout_sec = 0.60

        # Latest valid estimates
        self.have_valid_estimate = False
        self.last_x_m = 0.0
        self.last_y_m = 0.0
        self.last_z_m = 0.0
        self.last_yaw_deg = 0.0

        # Logic thresholds (pose-based, not video-region-based)
        # Position % = abs(axis_error) / z * 100
        # Radial %   = sqrt(x^2+y^2) / z * 100
        self.enter_move_pct = 20.0
        self.enter_yaw_pct = 8.0
        self.yaw_aligned_deg = 5.0
        self.land_pos_pct = 1.0
        self.land_yaw_deg = 1.0

        self.app = Flask(__name__)
        self.setup_routes()

        self.web_thread = threading.Thread(target=self.run_web_server, daemon=True)
        self.web_thread.start()

        self.get_logger().info(f'Subscribed to {image_topic} with QoS depth=1 BEST_EFFORT')
        self.get_logger().info(f'Web stream at http://<PI_IP>:{self.web_port}/')
        self.get_logger().info('Demo logic uses ArUco pose/yaw estimates, video is only a representation')

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return f"""
            <html>
                <head><title>Demo Detection And Movement</title></head>
                <body style="font-family: Arial; background:#111; color:#eee; text-align:center;">
                    <h2>Demo Detection And Movement</h2>
                    <img src="/video_feed" width="960">
                    <p>ROS topic: /aruco/demo_offset</p>
                    <p>Pose/yaw-driven state logic. Video is display only.</p>
                </body>
            </html>
            """

        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self.generate_mjpeg(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

    def run_web_server(self):
        self.app.run(
            host=self.web_host,
            port=self.web_port,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    def generate_mjpeg(self):
        while True:
            with self.frame_lock:
                frame = self.latest_jpeg

            if frame is None:
                time.sleep(0.03)
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
            time.sleep(0.03)

    def reset_to_search(self):
        self.state = DemoState.SEARCH
        self.state_start_time = time.time()

    def elapsed_in_state(self):
        return time.time() - self.state_start_time

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            self.get_logger().info(f'STATE -> {self.state.name}')

    def get_phase_text(self):
        if self.state == DemoState.SEARCH:
            return 'PHASE: SEARCH'
        if self.state == DemoState.HOVER_CONFIRM_1:
            remain = max(0.0, self.hover_confirm_1_sec - self.elapsed_in_state())
            return f'PHASE: HOVER CONFIRM 1 ({remain:.1f}s)'
        if self.state == DemoState.MOVE_FIXED_FRAME:
            return 'PHASE: MOVE FIXED FRAME'
        if self.state == DemoState.HOVER_CONFIRM_2:
            remain = max(0.0, self.hover_confirm_2_sec - self.elapsed_in_state())
            return f'PHASE: HOVER CONFIRM 2 ({remain:.1f}s)'
        if self.state == DemoState.YAW_ALIGN:
            return 'PHASE: YAW ALIGN'
        if self.state == DemoState.FINE_POINTING:
            return 'PHASE: FINE POINTING'
        if self.state == DemoState.LAND_HOLD:
            remain = max(0.0, self.land_hold_sec - self.elapsed_in_state())
            return f'PHASE: LAND HOLD ({remain:.1f}s)'
        if self.state == DemoState.LAND:
            return 'PHASE: LAND'
        return 'PHASE: UNKNOWN'

    def draw_overlay(self, img, x_pct, y_pct, yaw_pct, phase_text, valid_now):
        h, w = img.shape[:2]
        cx = int(w / 2)
        cy = int(h / 2)

        cv2.line(img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)

        if valid_now:
            color = (0, 255, 0)
            valid_text = 'ESTIMATE: LIVE'
        elif self.have_valid_estimate:
            color = (0, 200, 255)
            valid_text = 'ESTIMATE: HOLDING LAST'
        else:
            color = (0, 0, 255)
            valid_text = 'ESTIMATE: NONE'

        line1 = f'X ERR: {x_pct:5.2f}%   Y ERR: {y_pct:5.2f}%   YAW ERR: {yaw_pct:5.2f}%'

        cv2.rectangle(img, (12, 12), (w - 12, 98), (20, 20, 20), -1)
        cv2.rectangle(img, (12, 12), (w - 12, 98), color, 2)

        cv2.putText(img, line1, (28, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(img, phase_text, (28, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(img, valid_text, (w - 260, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    def extract_yaw_deg_from_rvec(self, rvec):
        rot_mat, _ = cv2.Rodrigues(rvec)
        yaw_rad = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        return math.degrees(yaw_rad)

    def compute_error_percentages(self, x_m, y_m, z_m, yaw_deg):
        z_safe = max(abs(z_m), 1e-6)

        x_pct = abs(x_m) / z_safe * 100.0
        y_pct = abs(y_m) / z_safe * 100.0
        radial_pct = math.sqrt(x_m * x_m + y_m * y_m) / z_safe * 100.0
        yaw_pct = abs(yaw_deg) / 90.0 * 100.0

        return x_pct, y_pct, radial_pct, yaw_pct

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated = frame.copy()

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.aruco_dict,
            parameters=self.detector_params
        )

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

            cv2.aruco.drawDetectedMarkers(
                annotated,
                [marker_corners],
                np.array([[ids[chosen_index]]], dtype=np.int32)
            )

            cv2.drawFrameAxes(
                annotated,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                self.marker_size_m * 0.5
            )

        # Use latest valid numbers for short fallback
        using_held_estimate = False
        if not valid_now and self.have_valid_estimate:
            if (time.time() - self.last_valid_time) <= self.pose_hold_timeout_sec:
                using_held_estimate = True
            else:
                self.have_valid_estimate = False

        x_pct = 0.0
        y_pct = 0.0
        yaw_pct = 0.0
        radial_pct = 999.0
        yaw_abs_deg = 999.0

        if valid_now or using_held_estimate:
            x_pct, y_pct, radial_pct, yaw_pct = self.compute_error_percentages(
                self.last_x_m,
                self.last_y_m,
                self.last_z_m,
                self.last_yaw_deg
            )
            yaw_abs_deg = abs(self.last_yaw_deg)

            out = Vector3Stamped()
            out.header = msg.header
            out.vector.x = self.last_x_m
            out.vector.y = self.last_y_m
            out.vector.z = self.last_z_m
            self.offset_pub.publish(out)

        # -----------------------------
        # State machine driven by pose/yaw estimates
        # -----------------------------
        estimate_available = valid_now or using_held_estimate

        if self.state == DemoState.SEARCH:
            if estimate_available and radial_pct <= self.enter_move_pct:
                self.change_state(DemoState.HOVER_CONFIRM_1)

        elif self.state == DemoState.HOVER_CONFIRM_1:
            if not estimate_available or radial_pct > self.enter_move_pct:
                self.reset_to_search()
            elif self.elapsed_in_state() >= self.hover_confirm_1_sec:
                self.change_state(DemoState.MOVE_FIXED_FRAME)

        elif self.state == DemoState.MOVE_FIXED_FRAME:
            if not estimate_available:
                self.reset_to_search()
            elif radial_pct <= self.enter_yaw_pct:
                self.change_state(DemoState.HOVER_CONFIRM_2)

        elif self.state == DemoState.HOVER_CONFIRM_2:
            if not estimate_available:
                self.reset_to_search()
            elif radial_pct > self.enter_yaw_pct:
                self.change_state(DemoState.MOVE_FIXED_FRAME)
            elif self.elapsed_in_state() >= self.hover_confirm_2_sec:
                self.change_state(DemoState.YAW_ALIGN)

        elif self.state == DemoState.YAW_ALIGN:
            if not estimate_available:
                self.reset_to_search()
            elif radial_pct > self.enter_yaw_pct:
                self.change_state(DemoState.MOVE_FIXED_FRAME)
            elif yaw_abs_deg <= self.yaw_aligned_deg:
                self.change_state(DemoState.FINE_POINTING)

        elif self.state == DemoState.FINE_POINTING:
            if not estimate_available:
                self.reset_to_search()
            elif radial_pct > self.enter_yaw_pct:
                self.change_state(DemoState.MOVE_FIXED_FRAME)
            elif radial_pct <= self.land_pos_pct and yaw_abs_deg <= self.land_yaw_deg:
                self.change_state(DemoState.LAND_HOLD)

        elif self.state == DemoState.LAND_HOLD:
            if not estimate_available:
                self.reset_to_search()
            elif radial_pct > self.enter_yaw_pct:
                self.change_state(DemoState.MOVE_FIXED_FRAME)
            elif not (radial_pct <= self.land_pos_pct and yaw_abs_deg <= self.land_yaw_deg):
                self.change_state(DemoState.FINE_POINTING)
            elif self.elapsed_in_state() >= self.land_hold_sec:
                self.change_state(DemoState.LAND)

        elif self.state == DemoState.LAND:
            pass

        phase_text = self.get_phase_text()
        self.draw_overlay(annotated, x_pct, y_pct, yaw_pct, phase_text, valid_now)

        ok, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self.frame_lock:
                self.latest_jpeg = jpg.tobytes()


def main(args=None):
    rclpy.init(args=args)
    node = DemoDetectionAndMovement()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
