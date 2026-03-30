#!/usr/bin/env python3

import math
import socket
import struct
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class UdpCamSender(Node):
    def __init__(self):
        super().__init__('udp_cam_sender')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('dest_ip', '172.20.10.4')   # CHANGE THIS
        self.declare_parameter('dest_port', 5005)            # separate from MAVLink
        self.declare_parameter('send_width', 640)
        self.declare_parameter('send_height', 360)
        self.declare_parameter('jpeg_quality', 55)
        self.declare_parameter('max_payload', 1200)
        self.declare_parameter('send_fps', 20.0)

        image_topic = str(self.get_parameter('image_topic').value)
        self.dest_ip = str(self.get_parameter('dest_ip').value)
        self.dest_port = int(self.get_parameter('dest_port').value)
        self.send_width = int(self.get_parameter('send_width').value)
        self.send_height = int(self.get_parameter('send_height').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.max_payload = int(self.get_parameter('max_payload').value)
        self.send_period = 1.0 / max(float(self.get_parameter('send_fps').value), 1.0)

        self.bridge = CvBridge()

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

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1_000_000)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp_ns = 0
        self.frame_id = 0
        self.last_send_time = 0.0

        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.sender_thread.start()

        self.get_logger().info(
            f'UDP sender streaming to {self.dest_ip}:{self.dest_port} '
            f'at {self.send_width}x{self.send_height}, JPEG q={self.jpeg_quality}'
        )
        self.get_logger().info('QoS = BEST_EFFORT, depth=1, latest-frame-only')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return

        with self.lock:
            self.latest_frame = frame
            self.latest_stamp_ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)

    def sender_loop(self):
        while rclpy.ok():
            now = time.time()
            if (now - self.last_send_time) < self.send_period:
                time.sleep(0.001)
                continue

            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                stamp_ns = self.latest_stamp_ns

            if frame is None:
                time.sleep(0.005)
                continue

            resized = cv2.resize(
                frame,
                (self.send_width, self.send_height),
                interpolation=cv2.INTER_AREA
            )

            ok, enc = cv2.imencode(
                '.jpg',
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                continue

            data = enc.tobytes()
            self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF

            total_chunks = math.ceil(len(data) / self.max_payload)
            if total_chunks > 65535:
                self.get_logger().warning('Frame too large; skipping')
                continue

            for chunk_idx in range(total_chunks):
                start = chunk_idx * self.max_payload
                end = start + self.max_payload
                payload = data[start:end]

                # Header:
                # uint32 frame_id
                # uint16 total_chunks
                # uint16 chunk_idx
                # uint32 jpeg_size
                # uint16 width
                # uint16 height
                # uint64 stamp_ns
                header = struct.pack(
                    '!IHHIHHQ',
                    self.frame_id,
                    total_chunks,
                    chunk_idx,
                    len(data),
                    self.send_width,
                    self.send_height,
                    stamp_ns
                )

                packet = header + payload
                try:
                    self.sock.sendto(packet, (self.dest_ip, self.dest_port))
                except Exception as e:
                    self.get_logger().warning(f'UDP send failed: {e}')
                    break

            self.last_send_time = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = UdpCamSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
