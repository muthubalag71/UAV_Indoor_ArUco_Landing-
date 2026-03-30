#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from mavros_msgs.msg import State
from geometry_msgs.msg import TwistStamped # <--- Added for movement
import sys
import time

class X500Mavros(Node):

    def __init__(self):
        super().__init__('x500mavros')

        self.create_subscription(State, '/mavros/state', self.state_callback, 10)

        # Clients
        self.setmode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # Velocity Publisher
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        self.connected = False
        self.armed = False
        self.mode = ""
        self.myModeSetting = ""
        self.myArmSetting = False

    def state_callback(self, msg):
        self.armed = msg.armed
        self.mode = msg.mode
        self.connected = msg.connected
        # ... (Your existing safety checks) ...

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        async_call = self.setmode_client.call_async(req)
        rclpy.spin_until_future_complete(self, async_call)
        if async_call.result().mode_sent:
            self.myModeSetting = mode
            return True
        return False

    def arm(self):
        req = CommandBool.Request()
        req.value = True
        async_call = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, async_call)
        if async_call.result().success:
            self.myArmSetting = True
            return True
        return False

    def takeoff(self):
        req = CommandTOL.Request()
        req.altitude = 1.0 # Your existing 1m altitude
        async_call = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, async_call)
        return async_call.result().success

    def move_body_velocity(self, vx, vy, duration):
        """ Sends a body-frame velocity for a set duration while holding yaw """
        start_time = time.time()
        while (time.time() - start_time) < duration:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link" # Commands relative to drone nose
            
            msg.twist.linear.x = float(vx)
            msg.twist.linear.y = float(vy)
            msg.twist.linear.z = 0.0 # Maintain altitude
            
            # Locked Yaw: Leaving angular.z at 0.0 keeps the current heading
            msg.twist.angular.z = 0.0 
            
            self.vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Stop movement after duration
        self.stop()

    def stop(self):
        msg = TwistStamped()
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        self.vel_pub.publish(msg)

def ask_permission(prompt):
    user_input = input(f"\n[COMMANDER]: {prompt} (type 'yes' to proceed): ").lower()
    if user_input == 'yes':
        return True
    print("Mission aborted by commander.")
    return False

def main(args=None):
    rclpy.init(args=args)
    drone_node = X500Mavros()

    # 1. Connection & Takeoff (Original Logic)
    while not drone_node.connected:
        rclpy.spin_once(drone_node, timeout_sec=1)

    drone_node.set_mode("GUIDED")
    if not drone_node.armed: drone_node.arm()
    drone_node.takeoff()

    # Wait for takeoff to stabilize
    time.sleep(8) 
    drone_node.get_logger().info("Hovering at 1m. Ready for flight test.")

    # 2. Sequential Mission Legs
    # LEG 1: Positive Vx
    if ask_permission("Move +Vx (Forward 0.4m/s for 2s)?"):
        drone_node.move_body_velocity(0.4, 0.0, 2.0)
    else: return

    # LEG 2: Negative Vx
    if ask_permission("Move -Vx (Backward 0.4m/s for 2s)?"):
        drone_node.move_body_velocity(-0.4, 0.0, 2.0)
    else: return

    # LEG 3: Positive Vy
    if ask_permission("Move +Vy (Right 0.4m/s for 2s)?"):
        drone_node.move_body_velocity(0.0, 0.4, 2.0)
    else: return

    # LEG 4: Negative Vy
    if ask_permission("Move -Vy (Left 0.4m/s for 2s)?"):
        drone_node.move_body_velocity(0.0, -0.4, 2.0)
    else: return

    # 3. Final Landing
    if ask_permission("All legs complete. Proceed to LAND?"):
        drone_node.set_mode("LAND")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
