#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SimpleForward(Node):
    def __init__(self):
        super().__init__('simple_forward')

        ## Publisher to send Twist messages to the robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        ## Timer to publish movement commands at 10 Hz (every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('Simple Forward Node started. Driving straight...')

    def timer_callback(self):
        twist = Twist()
        
        ## Set forward speed (adjust this value depending on your robot, e.g., 0.2 to 0.5)
        twist.linear.x = 0.3  
        
        ## Keep rotation zero so it goes straight
        twist.angular.z = 0.0 

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleForward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()