#!/usr/bin/env python3

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge


class ArucoWebPose(Node):
    def __init__(self):
        super().__init__('aruco_web_pose')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('marker_id', -1)
        self.declare_parameter('marker_size_m', 0.10)
        self.declare_parameter('web_host', '0.0.0.0')
        self.declare_parameter('web_port', 8080)

        image_topic = str(self.get_parameter('image_topic').value)
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.marker_size_m = float(self.get_parameter('marker_size_m').value)
        self.web_host = str(self.get_parameter('web_host').value)
        self.web_port = int(self.get_parameter('web_port').value)

        self.bridge = CvBridge()
        self.offset_pub = self.create_publisher(Vector3Stamped, '/aruco/offset', 10)

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

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
        self.last_log_time = 0.0

        self.app = Flask(__name__)
        self.setup_routes()

        self.web_thread = threading.Thread(target=self.run_web_server, daemon=True)
        self.web_thread.start()

        self.get_logger().info(f'Subscribed to {image_topic}')
        self.get_logger().info(f'Web stream at http://<PI_IP>:{self.web_port}/')
        self.get_logger().info('Publishing offsets on /aruco/offset')

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return """
            <html>
                <head><title>ArUco Web Stream</title></head>
                <body style="font-family: Arial; background:#111; color:#eee; text-align:center;">
                    <h2>ArUco Detection Stream</h2>
                    <img src="/video_feed" width="960">
                    <p>ROS topic: /aruco/offset</p>
                    <p>x = right offset (m), y = down offset (m), z = forward distance (m)</p>
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
        self.app.run(host=self.web_host, port=self.web_port, threaded=True, debug=False, use_reloader=False)

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

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated = frame.copy()

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.aruco_dict,
            parameters=self.detector_params
        )

        img_h, img_w = frame.shape[:2]
        cx_img = img_w / 2.0
        cy_img = img_h / 2.0

        if ids is not None and len(ids) > 0:
            ids = ids.flatten()

            chosen_index = None
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

                pts = marker_corners[0]
                mx = float(np.mean(pts[:, 0]))
                my = float(np.mean(pts[:, 1]))

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

                cv2.circle(annotated, (int(cx_img), int(cy_img)), 6, (0, 255, 255), -1)
                cv2.circle(annotated, (int(mx), int(my)), 6, (0, 255, 0), -1)

                cv2.putText(annotated, f'ID: {int(ids[chosen_index])}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, f'X offset: {x_m:+.3f} m', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, f'Y offset: {y_m:+.3f} m', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, f'Z dist:   {z_m:.3f} m', (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                out = Vector3Stamped()
                out.header = msg.header
                out.vector.x = x_m
                out.vector.y = y_m
                out.vector.z = z_m
                self.offset_pub.publish(out)

                now = time.time()
                if now - self.last_log_time > 0.5:
                    self.get_logger().info(
                        f'ID={int(ids[chosen_index])} X={x_m:+.3f} Y={y_m:+.3f} Z={z_m:.3f}'
                    )
                    self.last_log_time = now
            else:
                cv2.putText(annotated, 'Wanted marker ID not found', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(annotated, 'No ArUco detected', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ok, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self.frame_lock:
                self.latest_jpeg = jpg.tobytes()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoWebPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
