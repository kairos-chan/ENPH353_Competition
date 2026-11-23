import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        self.bridge = CvBridge()

        # Camera image subscription
        self.image_sub = self.create_subscription(Image,'/camera/image_raw', self.callback1, 1)

        # Velocity command publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Processed camera image publishers
        self.gray_pub = self.create_publisher(Image, '/camera/grayscale', 1)
        self.threshold_pub = self.create_publisher(Image, '/camera/threshold', 1)
        self.debugimage_pub = self.create_publisher(Image, '/camera/debugimage', 1)

        self.get_logger().info('Line follower node started.')

    def region_of_interest(self, frame, vertices):
        """Keep only the lower trapezoidal/rectangular area of the image."""
        y_start = vertices[0][1]
        roi = np.zeros_like(frame)
        roi[y_start:, :] = frame[y_start:, :]
        return roi

    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def callback1(self, msg):
        # Convert the ROS Image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        self.get_logger().info('callback called')

        height, width = frame.shape[:2]

        vertices = [
            (0, int(height * (2 / 3))),
            (width, int(height * (2 / 3))),
            (width, height),
            (0, height)
        ]

        roi = self.region_of_interest(frame, vertices)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        reversethreshold = cv2.bitwise_not(threshold)

        # Publish the grayscale and thresholded images
        self.gray_pub.publish(self.bridge.cv2_to_imgmsg(gray, encoding='mono8'))
        self.threshold_pub.publish(self.bridge.cv2_to_imgmsg(reversethreshold, encoding='mono8'))

        moments = cv2.moments(reversethreshold)
        twist = Twist()
        debug = frame.copy()

        mid = width // 2
        left_m = cv2.moments(reversethreshold[:, :mid], binaryImage=True)
        right_m = cv2.moments(reversethreshold[:, mid:], binaryImage=True)

        dot_y = (vertices[0][1] + height) // 2

        if left_m['m00'] > 0 and right_m['m00'] > 0:
            left_cx = left_m['m10'] / left_m['m00']
            right_cx = mid + right_m['m10'] / right_m['m00']
            cx = int((left_cx + right_cx) / 2)

            # Yellow dots on each line, red dot on the lane centre
            cv2.circle(debug, (int(left_cx), dot_y), 6, (255, 255, 0), -1)
            cv2.circle(debug, (int(right_cx), dot_y), 6, (255, 255, 0), -1)
            cv2.circle(debug, (cx, dot_y), 8, (255, 0, 0), -1)

            norm_error = (cx - width / 2.0) / (width / 2.0)
            twist.linear.x = max(0.05, 0.5 - 0.40 * abs(norm_error))
            twist.angular.z = -5.0 * norm_error
        elif left_m['m00'] > 0:
            # Only the left line visible -> turn a little right
            twist.linear.x = 0.1
            twist.angular.z = -2

        elif right_m['m00'] > 0:
            # Only the right line visible -> turn a little left
            twist.linear.x = 0.1
            twist.angular.z = 2

        else:
            # No lines visible -> stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.debugimage_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='rgb8'))
        twist.linear.x = float(twist.linear.x)
        twist.angular.z = float(twist.angular.z)
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()