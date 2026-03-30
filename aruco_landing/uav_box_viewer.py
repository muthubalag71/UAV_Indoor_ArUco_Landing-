#!/usr/bin/env python3

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>UAV Box Viewer</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #111;
      color: #eee;
      text-align: center;
    }
    h2 {
      margin: 16px 0 8px 0;
    }
    #info {
      margin: 8px 0 12px 0;
      font-size: 16px;
    }
    canvas {
      background: #1b1b1b;
      border: 1px solid #444;
      margin-bottom: 16px;
    }
  </style>
</head>
<body>
  <h2>UAV Box Viewer</h2>
  <div id="info">Waiting for pose...</div>
  <canvas id="plot" width="900" height="700"></canvas>

  <script>
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const info = document.getElementById("info");

    const W = canvas.width;
    const H = canvas.height;
    const origin = { x: W / 2, y: H / 2 };
    const scale = 180; // px per meter

    function worldToScreen(x, y) {
      return {
        x: origin.x + x * scale,
        y: origin.y - y * scale
      };
    }

    function drawGrid() {
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;

      for (let xm = -5; xm <= 5; xm++) {
        const x = origin.x + xm * scale;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }

      for (let ym = -5; ym <= 5; ym++) {
        const y = origin.y + ym * scale;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      ctx.strokeStyle = "#666";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, origin.y);
      ctx.lineTo(W, origin.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(origin.x, 0);
      ctx.lineTo(origin.x, H);
      ctx.stroke();
    }

    function drawReferenceBox() {
      ctx.strokeStyle = "#4fb3ff";
      ctx.lineWidth = 3;

      const pts = [
        worldToScreen(0, 0),
        worldToScreen(1, 0),
        worldToScreen(1, -1),
        worldToScreen(0, -1),
        worldToScreen(0, 0)
      ];

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.stroke();

      ctx.fillStyle = "#00ff66";
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, 6, 0, 2 * Math.PI);
      ctx.fill();
    }

    function drawPath(path) {
      if (!path || path.length < 2) return;

      ctx.strokeStyle = "#ffd54f";
      ctx.lineWidth = 2;
      ctx.beginPath();

      const p0 = worldToScreen(path[0][0], path[0][1]);
      ctx.moveTo(p0.x, p0.y);

      for (let i = 1; i < path.length; i++) {
        const p = worldToScreen(path[i][0], path[i][1]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }

    function drawDrone(x, y) {
      const p = worldToScreen(x, y);
      ctx.fillStyle = "#ff5a5a";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
      ctx.fill();
    }

    async function update() {
      try {
        const res = await fetch("/pose");
        const data = await res.json();

        ctx.clearRect(0, 0, W, H);
        drawGrid();
        drawReferenceBox();

        if (data.have_pose) {
          drawPath(data.path_xy_rel);
          drawDrone(data.rel_x, data.rel_y);

          info.textContent =
            `x=${data.rel_x.toFixed(2)} m   y=${data.rel_y.toFixed(2)} m   z=${data.z.toFixed(2)} m   ` +
            `home_locked=${data.home_locked}`;
        } else {
          info.textContent = "Waiting for pose...";
        }
      } catch (err) {
        info.textContent = "Browser fetch error / server not ready";
      }
    }

    setInterval(update, 100);
    update();
  </script>
</body>
</html>
"""


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.have_pose = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.home_x = None
        self.home_y = None
        self.path_xy_rel = []

    def update_pose(self, x, y, z):
        with self.lock:
            self.current_x = x
            self.current_y = y
            self.current_z = z
            self.have_pose = True

            if self.home_x is None:
                self.home_x = x
                self.home_y = y

            rel_x = x - self.home_x
            rel_y = y - self.home_y

            self.path_xy_rel.append([rel_x, rel_y])
            if len(self.path_xy_rel) > 5000:
                self.path_xy_rel.pop(0)

    def snapshot(self):
        with self.lock:
            rel_x = 0.0
            rel_y = 0.0
            home_locked = self.home_x is not None and self.home_y is not None

            if home_locked:
                rel_x = self.current_x - self.home_x
                rel_y = self.current_y - self.home_y

            return {
                "have_pose": self.have_pose,
                "x": self.current_x,
                "y": self.current_y,
                "z": self.current_z,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "home_locked": home_locked,
                "path_xy_rel": list(self.path_xy_rel),
            }


class PoseViewerNode(Node):
    def __init__(self, shared_state):
        super().__init__('uav_box_viewer')
        self.shared_state = shared_state

        pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            pose_qos
        )

    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.shared_state.update_pose(x, y, z)


def make_handler(shared_state):
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))

            elif self.path == "/pose":
                data = shared_state.snapshot()
                body = json.dumps(data).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return

    return ViewerHandler


def ros_spin_thread(node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)


def main(args=None):
    rclpy.init(args=args)

    shared_state = SharedState()
    node = PoseViewerNode(shared_state)

    spin_thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()

    host = "0.0.0.0"
    port = 8080
    server = ThreadingHTTPServer((host, port), make_handler(shared_state))

    node.get_logger().info(f"UAV browser viewer running at http://0.0.0.0:{port}")
    node.get_logger().info("Open from your laptop browser using: http://<PI_IP>:8080")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
