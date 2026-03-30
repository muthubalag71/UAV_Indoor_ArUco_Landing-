#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher720(Node):

    def __init__(self):
        super().__init__('camera_pub_720')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('topic_name', '/camera/image_raw')

        camera_index = int(self.get_parameter('camera_index').value)
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = float(self.get_parameter('fps').value)
        topic_name = str(self.get_parameter('topic_name').value)

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()

        # Force V4L2 for USB webcam
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f"Camera index {camera_index} not found")
            raise RuntimeError("Camera failed")

        # Logitech webcams usually behave better with MJPG at higher resolutions
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.get_logger().info(
            f"Camera publisher started | requested={width}x{height}@{fps:.1f} | "
            f"actual={actual_width}x{actual_height}@{actual_fps:.1f}"
        )

        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.get_logger().warning("Frame capture failed")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.publisher_.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher720()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
