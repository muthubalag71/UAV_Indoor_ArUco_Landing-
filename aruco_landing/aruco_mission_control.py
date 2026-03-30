#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from mavros_msgs.msg import State
from geometry_msgs.msg import Twist, Vector3Stamped, PoseStamped
from std_msgs.msg import Bool
import threading
import time
import math

class ArUcoX500Final(Node):
    def __init__(self):
        super().__init__('aruco_x500_final')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publisher
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        
        # Subscribers
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos)
        self.create_subscription(Bool, '/aruco/accepted', self.aruco_cb, 10)
        self.create_subscription(Vector3Stamped, '/aruco/pose_hold', self.aruco_pose_cb, 10)

        # Clients
        self.setmode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # State
        self.connected = False
        self.curr_yaw = 0.0
        self.initial_yaw = None
        self.is_aruco_accepted = False
        self.aruco_offset = None
        self.lock_start_time = None

    def state_cb(self, msg):
        self.connected = msg.connected

    def pose_cb(self, msg):
        q = msg.pose.orientation
        self.curr_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        if self.connected and self.initial_yaw is None:
            self.initial_yaw = self.curr_yaw
            self.get_logger().info(f"Yaw Locked: {math.degrees(self.initial_yaw):.1f} deg")

    def aruco_cb(self, msg):
        self.is_aruco_accepted = msg.data
        if self.is_aruco_accepted:
            if self.lock_start_time is None:
                self.lock_start_time = time.time()
        else:
            self.lock_start_time = None

    def aruco_pose_cb(self, msg):
        self.aruco_offset = msg.vector

    def send_pulse(self, target_vx, target_vy, duration=1.2):
        """
        Final Mapping Correction:
        W (Forward) -> Drone Forward (+X)
        S (Backward) -> Drone Backward (-X)
        A (Left) -> Drone Left (+Y)
        D (Right) -> Drone Right (-Y)
        """
        self.get_logger().info("Executing Step...")
        start = time.time()
        while (time.time() - start) < duration:
            msg = Twist()
            # Flipped signs from the previous test to correct inversion
            msg.linear.x = float(target_vx) 
            msg.linear.y = float(target_vy)
            
            if self.initial_yaw is not None:
                yaw_err = (self.initial_yaw - self.curr_yaw + math.pi) % (2 * math.pi) - math.pi
                msg.angular.z = 1.5 * yaw_err 
                
            self.vel_pub.publish(msg)
            time.sleep(0.05)
        
        self.vel_pub.publish(Twist())

def mission_logic(node):
    while not node.connected or node.initial_yaw is None:
        time.sleep(1.0)
    
    node.setmode_client.call_async(SetMode.Request(custom_mode="GUIDED"))
    time.sleep(1)
    node.arming_client.call_async(CommandBool.Request(value=True))
    time.sleep(1)
    node.takeoff_client.call_async(CommandTOL.Request(altitude=1.2))
    
    node.get_logger().info("Taking off...")
    time.sleep(10.0)

    node.get_logger().info("Waiting for ArUco (3s)...")
    while rclpy.ok():
        if node.lock_start_time and (time.time() - node.lock_start_time >= 3.0):
            break
        node.send_pulse(0.0, 0.0, duration=0.1)
    
    while rclpy.ok():
        print("\n" + "="*40)
        if node.aruco_offset:
            print(f"CAMERA DATA -> X: {node.aruco_offset.x:.2f} | Y: {node.aruco_offset.y:.2f}")
        
        print("W: Forward | S: Backward | A: Left | D: Right | LAND: Land")
        cmd = input("Command: ").strip().upper()

        if cmd == "W": 
            node.send_pulse(0.25, 0.0)   # Drone Forward (+X)
        elif cmd == "S": 
            node.send_pulse(-0.25, 0.0)  # Drone Backward (-X)
        elif cmd == "A": 
            node.send_pulse(0.0, 0.25)   # Drone Left (+Y)
        elif cmd == "D": 
            node.send_pulse(0.0, -0.25)  # Drone Right (-Y)
        elif cmd == "LAND":
            node.setmode_client.call_async(SetMode.Request(custom_mode="LAND"))
            break
        else:
            node.send_pulse(0.0, 0.0, 0.1)

def main():
    rclpy.init()
    node = ArUcoX500Final()
    t = threading.Thread(target=mission_logic, args=(node,), daemon=True)
    t.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vel_pub.publish(Twist())
        rclpy.shutdown()

if __name__ == '__main__':
    main()
