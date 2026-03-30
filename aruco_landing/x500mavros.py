#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from mavros_msgs.msg import State
import sys


class X500Mavros(Node):

    def __init__(self):
        super().__init__('x500mavros')

        self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )

        self.setmode_client = self.create_client(SetMode, '/mavros/set_mode')
        while not self.setmode_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for set mode service...")

        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        while not self.arming_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for arming service...")

        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        while not self.takeoff_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for takeoff service...")

        self.connected = False
        self.armed = False
        self.mode = ""
        self.alarmed = False

        self.myModeSetting = ""
        self.myArmSetting = False


    def state_callback(self, msg):
        self.armed = msg.armed
        self.mode = msg.mode
        self.connected = msg.connected

        if (self.myArmSetting) and (not self.armed) and (self.myModeSetting.upper() != "LAND"):
            self.get_logger().warn("RC disarm detected, program terminated!")
            rclpy.shutdown()
            sys.exit(0)

        if (self.myModeSetting != "") and            (self.myModeSetting.upper() != "LAND") and            (self.mode == "LAND") and            (self.armed):
            self.get_logger().warn("RC landing request detected, program terminated!")
            rclpy.shutdown()
            sys.exit(0)


    def set_mode(self, mode):

        req = SetMode.Request()
        req.custom_mode = mode

        async_call = self.setmode_client.call_async(req)

        rclpy.spin_until_future_complete(self, async_call)

        if async_call.result().mode_sent:
            self.get_logger().info(f"Mode changed to {mode}")
            self.myModeSetting = mode
            return True

        else:
            self.get_logger().error("Failed to change mode!")
            return False


    def arm(self):

        req = CommandBool.Request()
        req.value = True

        async_call = self.arming_client.call_async(req)

        rclpy.spin_until_future_complete(self, async_call)

        if async_call.result().success:
            self.get_logger().info("Drone armed successfully!")
            self.myArmSetting = True
            return True

        else:
            self.get_logger().error("Failed to arm the drone!")
            return False


    def takeoff(self):

        req = CommandTOL.Request()

        req.min_pitch = 0.0
        req.yaw = 0.0
        req.latitude = 0.0
        req.longitude = 0.0
        req.altitude = 1.0

        async_call = self.takeoff_client.call_async(req)

        rclpy.spin_until_future_complete(self, async_call)

        if async_call.result().success:
            self.get_logger().info('Takeoff command sent OK!')
            return True
        else:
            self.get_logger().error('Failed to call takeoff service!')
            return False


    def start_timer(self, x):
        self.alarmed = False
        self.create_timer(x, self.timer_callback)


    def timer_callback(self):
        self.alarmed = True


def main(args=None):

    rclpy.init(args=args)

    drone_node = X500Mavros()

    while not drone_node.connected:
        drone_node.get_logger().info("MavLink not Connected...")
        rclpy.spin_once(drone_node, timeout_sec=1)

    if not drone_node.set_mode("GUIDED"):
        rclpy.shutdown()
        exit()

    if not drone_node.armed:
        if not drone_node.arm():
            rclpy.shutdown()
            exit()

    if not drone_node.takeoff():
        rclpy.shutdown()
        exit()

    drone_node.start_timer(10.0)

    while not drone_node.alarmed:
        drone_node.get_logger().info("Lift off...")
        rclpy.spin_once(drone_node, timeout_sec=1)

    if not drone_node.set_mode("LAND"):
        rclpy.shutdown()
        exit()

    rclpy.spin(drone_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
