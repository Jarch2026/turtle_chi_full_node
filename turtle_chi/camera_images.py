import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

# Camera
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import cv2
import cv_bridge

class Robot(Node):
    def __init__(self):
        super().__init__('turtle_chi_camera')

        # Pull turtlebot number and pad with leading zero if needed
        unformatted = os.getenv('ROS_DOMAIN_ID')
        ros_domain_id = f'{int(unformatted):02d}'
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        # Camera subscriber
        self.image_sub = self.create_subscription(
            CompressedImage,
            f'/tb{ros_domain_id}/oakd/rgb/preview/image_raw/compressed',
            self.image_callback,
            10)

        # Set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # Initialize the debugging window
        cv2.namedWindow("window", 1)

        self.saved_imgs = 0
        self.get_logger().info(f"Waiting...")
        time.sleep(5)

    def image_callback(self, msg):

        if self.saved_imgs < 20:
            # Converts the incoming ROS message to OpenCV format
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Save as Jpeg?
            cv2.imwrite(f"/home/jarch/intro_robo_ws/src/turtle_chi/turtle_chi/images/image{self.saved_imgs+50}.jpeg", image)

            self.saved_imgs += 1
            self.get_logger().info(f"Saved image number {self.saved_imgs}")

            cv2.imshow("window", image)
            cv2.waitKey(3)
            time.sleep(2)
        else:
            self.get_logger().info(f"All Images saved")


def main(args=None):
    rclpy.init(args=args)
    robot_node = Robot()
    time.sleep(2)  # Wait for publishers and subscribers to set up
    rclpy.spin(robot_node)
    robot_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
