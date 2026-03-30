#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist, Vector3Stamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL


# ============================================================
# Global calibration results for reuse in later codes
# ============================================================
CALIBRATION = {
    "home_yaw_rad": None,
    "g_x_vx": None,
    "g_y_vx": None,
    "g_x_vy": None,
    "g_y_vy": None,
    "cross_vx": None,
    "cross_vy": None,
    "vx_primary_axis": None,
    "vy_primary_axis": None,
}


class AxisPulseCoupling(Node):
    def __init__(self):
        super().__init__('axis_pulse_coupling')

        # ---------------- Parameters ----------------
        self.takeoff_alt = 1.0

        # Safe pulse settings
        self.body_pulse_speed = 0.12      # m/s
        self.body_pulse_duration = 1.2    # s  -> about 0.144 m intended displacement
        self.stop_brake_time = 1.0        # s of continuous zero command
        self.hover_before_test = 2.0      # s
        self.hover_between_tests = 2.0    # s

        # Safety caps
        self.max_radius_from_home = 1.0   # m hard cap
        self.min_safe_alt = 0.6           # m
        self.max_safe_alt = 1.5           # m
        self.pose_hold_timeout = 0.7      # s
        self.max_yaw_error_deg = 20.0     # deg hard cap

        # Yaw hold
        self.yaw_kp = 1.2
        self.max_yaw_rate = 0.6           # rad/s hard cap

        # State
        self.current_state = State()
        self.current_pose = None
        self.pose_hold = None
        self.pose_hold_stamp = 0.0

        self.connected = False
        self.armed = False
        self.mode = ""

        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        # MAVROS-friendly QoS
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ROS interfaces
        self.create_subscription(State, '/mavros/state', self.state_cb, mavros_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.local_pose_cb, mavros_qos)
        self.create_subscription(Vector3Stamped, '/aruco/pose_hold', self.pose_hold_cb, mavros_qos)

        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        self.setmode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        while not self.setmode_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for set mode service...")
        while not self.arming_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for arming service...")
        while not self.takeoff_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for takeoff service...")

    # =========================================================
    # Callbacks
    # =========================================================
    def state_cb(self, msg):
        self.current_state = msg
        self.connected = msg.connected
        self.armed = msg.armed
        self.mode = msg.mode

    def local_pose_cb(self, msg):
        self.current_pose = msg

    def pose_hold_cb(self, msg):
        self.pose_hold = msg
        self.pose_hold_stamp = time.time()

    # =========================================================
    # Helpers
    # =========================================================
    def yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def wrap_pi(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def get_current_yaw(self):
        if self.current_pose is None:
            return None
        return self.yaw_from_quat(self.current_pose.pose.orientation)

    def get_xy_alt(self):
        if self.current_pose is None:
            return None
        p = self.current_pose.pose.position
        return p.x, p.y, p.z

    def get_pose_hold_xyz(self):
        if self.pose_hold is None:
            return None
        if (time.time() - self.pose_hold_stamp) > self.pose_hold_timeout:
            return None
        return (
            float(self.pose_hold.vector.x),
            float(self.pose_hold.vector.y),
            float(self.pose_hold.vector.z),
        )

    def safety_ok(self):
        xyz = self.get_xy_alt()
        if xyz is None or self.home_x is None:
            return False, "No local pose or home pose"

        x, y, z = xyz
        dx = x - self.home_x
        dy = y - self.home_y
        radius = math.sqrt(dx * dx + dy * dy)

        if radius > self.max_radius_from_home:
            return False, f"Exceeded home radius cap: {radius:.2f} m"

        if z < self.min_safe_alt or z > self.max_safe_alt:
            return False, f"Altitude outside safe band: {z:.2f} m"

        cyaw = self.get_current_yaw()
        if cyaw is None or self.home_yaw is None:
            return False, "No yaw available"

        yaw_err = abs(self.wrap_pi(self.home_yaw - cyaw))
        if math.degrees(yaw_err) > self.max_yaw_error_deg:
            return False, f"Yaw error too large: {math.degrees(yaw_err):.1f} deg"

        return True, "OK"

    def zero_cmd(self):
        self.vel_pub.publish(Twist())

    def zero_for(self, duration_sec):
        end_t = time.time() + duration_sec
        while rclpy.ok() and time.time() < end_t:
            self.zero_cmd()
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.05)

    def publish_velocity_local_with_yaw_hold(self, vx_local, vy_local):
        msg = Twist()

        cyaw = self.get_current_yaw()
        if cyaw is None or self.home_yaw is None:
            yaw_rate_cmd = 0.0
        else:
            yaw_err = self.wrap_pi(self.home_yaw - cyaw)
            yaw_rate_cmd = self.yaw_kp * yaw_err
            yaw_rate_cmd = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate_cmd))

        msg.linear.x = float(vx_local)
        msg.linear.y = float(vy_local)
        msg.linear.z = 0.0
        msg.angular.z = float(yaw_rate_cmd)
        self.vel_pub.publish(msg)

    def body_to_local_velocity(self, vx_body, vy_body):
        """
        Desired motion in body frame relative to locked home yaw,
        converted to local-frame velocity command.
        """
        psi = self.home_yaw
        vx_local = vx_body * math.cos(psi) - vy_body * math.sin(psi)
        vy_local = vx_body * math.sin(psi) + vy_body * math.cos(psi)
        return vx_local, vy_local

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        fut = self.setmode_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() and fut.result().mode_sent:
            self.get_logger().info(f"Mode changed to {mode}")
            return True
        self.get_logger().error(f"Failed to change mode to {mode}")
        return False

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        fut = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() and fut.result().success:
            self.get_logger().info("Drone armed successfully")
            return True
        self.get_logger().error("Failed to arm")
        return False

    def takeoff(self, alt=1.0):
        req = CommandTOL.Request()
        req.min_pitch = 0.0
        req.yaw = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.altitude = float(alt)
        fut = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() and fut.result().success:
            self.get_logger().info(f"Takeoff command sent to {alt:.2f} m")
            return True
        self.get_logger().error("Takeoff failed")
        return False

    def capture_home(self):
        xyz = self.get_xy_alt()
        cyaw = self.get_current_yaw()
        if xyz is None or cyaw is None:
            return False

        self.home_x, self.home_y, self.home_z = xyz
        self.home_yaw = cyaw
        CALIBRATION["home_yaw_rad"] = cyaw

        self.get_logger().info(
            f"Home locked: x={self.home_x:.2f}, y={self.home_y:.2f}, z={self.home_z:.2f}, "
            f"yaw={math.degrees(self.home_yaw):.1f} deg"
        )
        return True

    def wait_for_alt_band(self, target_alt, timeout=12.0):
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            xyz = self.get_xy_alt()
            if xyz is None:
                continue
            _, _, z = xyz
            self.zero_cmd()
            if abs(z - target_alt) < 0.20:
                self.get_logger().info(f"Reached takeoff altitude band: z={z:.2f}")
                return True
            time.sleep(0.05)
        return False

    def hold_hover(self, duration_sec):
        self.get_logger().info(f"Holding hover for {duration_sec:.1f}s")
        end_t = time.time() + duration_sec
        while rclpy.ok() and time.time() < end_t:
            ok, msg = self.safety_ok()
            if not ok:
                self.get_logger().error(f"Safety abort during hover: {msg}")
                return False
            self.publish_velocity_local_with_yaw_hold(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.05)
        return True

    def run_body_pulse(self, label, vx_body, vy_body):
        pose_before = self.get_pose_hold_xyz()
        if pose_before is None:
            self.get_logger().error(f"{label}: No fresh /aruco/pose_hold before pulse")
            return None

        x0, y0, z0 = pose_before
        vx_local, vy_local = self.body_to_local_velocity(vx_body, vy_body)

        self.get_logger().info("======================================")
        self.get_logger().info(f"{label} starting")
        self.get_logger().info(
            f"Desired body pulse: vx_body={vx_body:+.3f} m/s, vy_body={vy_body:+.3f} m/s"
        )
        self.get_logger().info(
            f"Published local velocity: vx_local={vx_local:+.3f}, vy_local={vy_local:+.3f}"
        )
        self.get_logger().info(
            f"Pose before: x={x0:+.4f}, y={y0:+.4f}, z={z0:.4f}"
        )

        end_t = time.time() + self.body_pulse_duration
        while rclpy.ok() and time.time() < end_t:
            ok, msg = self.safety_ok()
            if not ok:
                self.get_logger().error(f"{label}: Safety abort during pulse: {msg}")
                self.zero_for(self.stop_brake_time)
                return None

            self.publish_velocity_local_with_yaw_hold(vx_local, vy_local)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.05)

        self.get_logger().info(f"{label}: Brake/settle")
        self.zero_for(self.stop_brake_time)

        pose_after = self.get_pose_hold_xyz()
        if pose_after is None:
            self.get_logger().error(f"{label}: No fresh /aruco/pose_hold after pulse")
            return None

        x1, y1, z1 = pose_after
        dx = x1 - x0
        dy = y1 - y0

        cmd_mag = max(abs(vx_body), abs(vy_body))
        gx = dx / cmd_mag if cmd_mag > 1e-6 else 0.0
        gy = dy / cmd_mag if cmd_mag > 1e-6 else 0.0

        result = {
            "label": label,
            "before": (x0, y0, z0),
            "after": (x1, y1, z1),
            "dx": dx,
            "dy": dy,
            "g_x": gx,
            "g_y": gy,
        }

        self.get_logger().info(
            f"Pose after : x={x1:+.4f}, y={y1:+.4f}, z={z1:.4f}"
        )
        self.get_logger().info(
            f"Delta      : dx={dx:+.4f}, dy={dy:+.4f}"
        )
        self.get_logger().info("======================================")
        return result

    def classify_and_store(self, result_vx, result_vy):
        if abs(result_vx["dx"]) >= abs(result_vx["dy"]):
            primary_vx = "cam_x"
            cross_vx = abs(result_vx["dy"] / result_vx["dx"]) if abs(result_vx["dx"]) > 1e-6 else float("inf")
        else:
            primary_vx = "cam_y"
            cross_vx = abs(result_vx["dx"] / result_vx["dy"]) if abs(result_vx["dy"]) > 1e-6 else float("inf")

        if abs(result_vy["dx"]) >= abs(result_vy["dy"]):
            primary_vy = "cam_x"
            cross_vy = abs(result_vy["dy"] / result_vy["dx"]) if abs(result_vy["dx"]) > 1e-6 else float("inf")
        else:
            primary_vy = "cam_y"
            cross_vy = abs(result_vy["dx"] / result_vy["dy"]) if abs(result_vy["dy"]) > 1e-6 else float("inf")

        CALIBRATION["g_x_vx"] = result_vx["g_x"]
        CALIBRATION["g_y_vx"] = result_vx["g_y"]
        CALIBRATION["g_x_vy"] = result_vy["g_x"]
        CALIBRATION["g_y_vy"] = result_vy["g_y"]
        CALIBRATION["cross_vx"] = cross_vx
        CALIBRATION["cross_vy"] = cross_vy
        CALIBRATION["vx_primary_axis"] = primary_vx
        CALIBRATION["vy_primary_axis"] = primary_vy

        self.get_logger().info("========= CALIBRATION SUMMARY =========")
        self.get_logger().info(f"+Vx primary axis: {primary_vx}, cross-coupling={cross_vx:.3f}")
        self.get_logger().info(f"+Vy primary axis: {primary_vy}, cross-coupling={cross_vy:.3f}")
        self.get_logger().info(f"g_x_vx = {CALIBRATION['g_x_vx']:+.5f}")
        self.get_logger().info(f"g_y_vx = {CALIBRATION['g_y_vx']:+.5f}")
        self.get_logger().info(f"g_x_vy = {CALIBRATION['g_x_vy']:+.5f}")
        self.get_logger().info(f"g_y_vy = {CALIBRATION['g_y_vy']:+.5f}")
        self.get_logger().info("=======================================")

    def land(self):
        return self.set_mode("LAND")


def main(args=None):
    rclpy.init(args=args)
    node = AxisPulseCoupling()

    try:
        while rclpy.ok() and not node.connected:
            node.get_logger().info("Waiting for MAVROS FCU connection...")
            rclpy.spin_once(node, timeout_sec=0.5)

        start_wait = time.time()
        while rclpy.ok() and node.current_pose is None:
            if time.time() - start_wait > 15.0:
                node.get_logger().error("No local pose received from /mavros/local_position/pose")
                return
            node.get_logger().info("Waiting for local pose...")
            rclpy.spin_once(node, timeout_sec=0.2)

        if not node.capture_home():
            node.get_logger().error("Failed to capture home pose/yaw")
            return

        if not node.set_mode("GUIDED"):
            return

        if not node.armed:
            if not node.arm():
                return

        if not node.takeoff(node.takeoff_alt):
            return

        if not node.wait_for_alt_band(node.takeoff_alt, timeout=12.0):
            node.get_logger().error("Did not reach takeoff altitude band safely")
            node.land()
            return

        if not node.hold_hover(node.hover_before_test):
            node.land()
            return

        start_pose_wait = time.time()
        while rclpy.ok() and node.get_pose_hold_xyz() is None:
            if time.time() - start_pose_wait > 5.0:
                node.get_logger().error("No fresh /aruco/pose_hold available")
                node.land()
                return
            node.publish_velocity_local_with_yaw_hold(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)

        result_vx = node.run_body_pulse(
            label="+Vx body pulse",
            vx_body=node.body_pulse_speed,
            vy_body=0.0
        )
        if result_vx is None:
            node.land()
            return

        if not node.hold_hover(node.hover_between_tests):
            node.land()
            return

        result_vy = node.run_body_pulse(
            label="+Vy body pulse",
            vx_body=0.0,
            vy_body=node.body_pulse_speed
        )
        if result_vy is None:
            node.land()
            return

        node.classify_and_store(result_vx, result_vy)

        node.get_logger().info("Landing...")
        node.land()
        rclpy.spin(node)

    except KeyboardInterrupt:
        try:
            node.get_logger().warn("KeyboardInterrupt -> zero command")
            node.zero_for(0.5)
        except Exception:
            pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
