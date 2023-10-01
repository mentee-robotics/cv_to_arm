from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2

NUMBER_OF_FRAMES_FOR_AVERAGE = 10  # Number of frames to average for tracking history

class ObjectDistanceEstimator(Node):
    def __init__(self, model_path):
  
        self.rgb_subscriber = self.create_subscription(Image, 'rgb/image_rect_color', self.rgb_callback, 10)
        self.point_cloud_subscriber = self.create_subscription(PointCloud2, 'point_cloud/cloud_registered', self.point_cloud_callback, 10)
        self.publisher = self.create_publisher(Twist, 'detected_object', 10)

        # CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Member variables to store the current data
        self.rgb_data = None
        self.point_cloud_data = None

        # Load YOLO model for object tracking
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])


    def rgb_callback(self, msg):
        self.rgb_data = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


    def point_cloud_callback(self, msg):
        self.point_cloud_data = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))


    def process_data(self):
        # Make sure we have both rgb and point cloud data
        if self.rgb_data is not None and self.point_cloud_data and self.new_data_received:
            self._process_frame(self.rgb_data, self.point_cloud_data)
            self.new_data_received = False

    def _update_track_history(self, track_id, adjusted_point):
        """
        Update the tracked positions for the detected object.
        :param track_id: The ID for the object being tracked
        :param adjusted_point: The new 3D coordinate to be added to the object's track history
        """
        self.track_history[track_id].append(adjusted_point)
        if len(self.track_history[track_id]) > NUMBER_OF_FRAMES_FOR_AVERAGE:
            self.track_history[track_id].pop(0)

    def _publish_object_data(self, track_id, adjusted_point):
        """
        Publish the processed data of the detected object.
        :param track_id: The ID for the object being tracked
        :param adjusted_point: The new 3D coordinate of the detected object
        """
        msg = Twist()  # Assuming we use Twist message to represent 3D coordinates for now
        msg.linear.x = adjusted_point[0]
        msg.linear.y = adjusted_point[1]
        msg.linear.z = adjusted_point[2]
        self.publisher.publish(msg)

    def _process_frame(self, rgb_frame, point_cloud):
        """
        Process the RGB frame to detect objects and then compute their 3D coordinates.
        :param rgb_frame: The RGB image frame
        :param point_cloud: The corresponding point cloud data
        """
        # Use the YOLO model to detect objects in the RGB frame
        results = self.model(rgb_frame)

        # Process each detected object
        for det in results:
            label, confidence, (x, y, w, h) = det  # Assuming this format for the detected result
            track_id = label + str(x) + str(y)  # Generate a unique ID based on label and position (can be improved)

            self._process_detected_object((x, y, w, h), point_cloud)



    def _process_detected_object(self, box, point_cloud):
        """Process each detected object, compute its real-world coordinates and track its movement."""
        x, y, w, h = box

        point_cloud_value = None
        try:
            point_cloud_value = point_cloud[y * self.rgb_data.shape[1] + x]
        except IndexError:
            return

        # Extract depth value
        z_value = point_cloud_value[2]

        # Adjust depth for better accuracy
        half_depth = 0.05
        z_adjusted = z_value + half_depth
        adjusted_point = [point_cloud_value[0], point_cloud_value[1], z_adjusted]

        self._publish_object_data(adjusted_point)
        return adjusted_point


    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            self.process_data()

if __name__ == "__main__":
    rclpy.init()
    estimator = ObjectDistanceEstimator('/path/to/model')
    estimator.run()
