#!/usr/bin/env python3

import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class TimeTrialsDriver(Node):
    def __init__(self):
        super().__init__('driver_node')
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.move = Twist()

        self.score_pub = self.create_publisher(String, '/score_tracker', 1)

        self.iterationN = 0
        self.speed = 0.5

        print("init finished")

    

    def callback(self, data):
        if self.iterationN == 0:
            # first msg should start timer with id=0
            message = String(data='Team10,paxxword,0,BEGIN')
            self.score_pub.publish(msg=message)

        if self.iterationN == 100:
            # after 100th iteration, stop timer with id=-1
            message = String(data='Team10,paxxword,-1,END')
            self.score_pub.publish(msg=message)
            self.speed = 0.0
        
        self.move.linear.x = self.speed
        self.cmd_vel_pub.publish(msg=self.move)
        print("msg published; N = " + str(self.iterationN) + "; spd = " + str(self.speed))

        self.iterationN += 1

def main(args=None):
    print("go_forward is running")
    rclpy.init(args=args)
    driver = TimeTrialsDriver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        print("goodnight")
    finally:
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()