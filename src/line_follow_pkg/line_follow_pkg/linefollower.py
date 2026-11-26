import rclpy
import cv2
import numpy as np
 
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
 
from line_follow_pkg.red_line_detection import detect_red_in_right_third
from line_follow_pkg.pedestrian_detection import (
    PedestrianTracker, blue_mask, find_pedestrian, draw_debug)
 
# States
GO = 'GO'        # normal line following
STOP = 'STOP'    # just saw the red line: brake and settle
WAIT = 'WAIT'    # stopped, watching the pedestrian until they leave the frame
 
STOP_SETTLE_SEC = 1.0      # how long to stay in STOP before starting to WAIT
WAIT_TIMEOUT_SEC = 15.0    # safety: go anyway if no pedestrian was ever seen/exited (set None to disable)
RED_CLEAR_FRAMES = 15      # frames without red before the red line can stop us again
BOOST_SEC = 2.0            # how long to drive faster after the wait is over
BOOST_FACTOR = 1.8         # forward speed multiplier during the boost (1.0 = no boost)
 
 
class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
 
        self.bridge = CvBridge()
 
        # True when there is red in the right third of the front camera view
        self.red_right = False
 
        # State machine
        self.state = GO
        self.state_start = self.get_clock().now()
        self.ignore_red = False     # True after leaving WAIT, until the red line is behind us
        self.red_clear_count = 0
        self.boost_start = None     # set when leaving WAIT; speed boost lasts BOOST_SEC
        self.tracker = PedestrianTracker()
 
        # Camera image subscription
        self.image_sub = self.create_subscription(Image,'/line_camera/image_raw', self.callback1, 1)
        self.front_sub = self.create_subscription(Image,'/camera/image_raw', self.front_callback, 1)
 
        # Velocity command publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
 
        # Processed camera image publishers
        self.gray_pub = self.create_publisher(Image, '/line_camera/grayscale', 1)
        self.threshold_pub = self.create_publisher(Image, '/line_camera/threshold', 1)
        self.debugimage_pub = self.create_publisher(Image, '/line_camera/debugimage', 1)
 
        # Pedestrian detection views (front camera)
        self.ped_mask_pub = self.create_publisher(Image, '/camera/pedestrian_mask', 1)
        self.ped_debug_pub = self.create_publisher(Image, '/camera/pedestrian_debug', 1)
 
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
 
    def set_state(self, new_state):
        self.get_logger().info(f'State: {self.state} -> {new_state}')
        self.state = new_state
        self.state_start = self.get_clock().now()
 
    def seconds_in_state(self):
        return (self.get_clock().now() - self.state_start).nanoseconds / 1e9
 
    def front_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.red_right = detect_red_in_right_third(frame)
 
        # Pedestrian detection runs every frame so you can always see it in the debug views
        ped_mask = blue_mask(frame)
        ped_det = find_pedestrian(ped_mask)
        ped_debug = draw_debug(frame, ped_mask, ped_det, label=f'state: {self.state}')
        self.ped_mask_pub.publish(self.bridge.cv2_to_imgmsg(ped_mask, encoding='mono8'))
        self.ped_debug_pub.publish(self.bridge.cv2_to_imgmsg(ped_debug, encoding='rgb8'))
 
        if self.state == GO:
            if self.ignore_red:
                # After crossing: wait until the red line has left the view before re-arming
                if self.red_right:
                    self.red_clear_count = 0
                else:
                    self.red_clear_count += 1
                    if self.red_clear_count >= RED_CLEAR_FRAMES:
                        self.ignore_red = False
            elif self.red_right:
                self.set_state(STOP)
 
        elif self.state == STOP:
            if self.seconds_in_state() >= STOP_SETTLE_SEC:
                self.tracker.reset()
                self.set_state(WAIT)
 
        elif self.state == WAIT:
            exit_side = self.tracker.update(ped_det, frame.shape[1])
            if exit_side is not None:
                self.get_logger().info(f'Pedestrian left the frame to the {exit_side}.')
                self.leave_wait()
            elif WAIT_TIMEOUT_SEC is not None and self.seconds_in_state() >= WAIT_TIMEOUT_SEC:
                self.get_logger().warn('Wait timeout reached, going anyway.')
                self.leave_wait()
 
    def leave_wait(self):
        self.ignore_red = True
        self.red_clear_count = 0
        self.boost_start = self.get_clock().now()
        self.set_state(GO)
 
    def boost_active(self):
        if self.boost_start is None:
            return False
        elapsed = (self.get_clock().now() - self.boost_start).nanoseconds / 1e9
        if elapsed >= BOOST_SEC:
            self.boost_start = None
            return False
        return True
 
    def callback1(self, msg):
        # Convert the ROS Image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
 
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
 
        # STOP and WAIT: hold still. GO: use the line-following command above.
        if self.state != GO:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif self.boost_active():
            # Drive faster for BOOST_SEC after the wait, then fall back to normal speed
            twist.linear.x *= BOOST_FACTOR
 
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