from collections import defaultdict
import cv2
import pyzed.sl as sl
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

NUMBER_OF_FRAMES_FOR_AVERAGE = 10  # Number of frames to average for tracking history


class ObjectDistanceEstimator(Node):
    def __init__(self, model_path):
        super().__init__('object_distance_estimator')
        
        # Initialize ROS publisher
        self.publisher = self.create_publisher(Twist, 'detected_object', 10)
        
        # Load YOLO model for object tracking
        self.model = YOLO(model_path)
        
        # Store the last N 3D positions for each object
        self.track_history = defaultdict(lambda: [])  
        
        # Initialize ZED camera
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the ZED camera with desired parameters."""
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera opening failed with error:", err)
            exit(1)

        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.confidence_threshold = 100
        self.runtime_parameters.texture_confidence_threshold = 100

    def _retrieve_camera_data(self):
        """Retrieve image and point cloud data from ZED camera."""
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            frame = image.get_data()
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), point_cloud
        return None, None

    def _process_frame(self, frame, point_cloud):
        """Process each frame to detect and track objects."""
        results = self.model.track(frame, persist=True)
        if not results or not results[0].boxes:
            return

        boxes = results[0].boxes.xywh.cpu()
        annotated_frame = results[0].plot()

        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                adjusted_point = self._process_detected_object(box, track_id, point_cloud)
                
                # Display the x, y, z coordinates on the bounding box if available
                if adjusted_point:
                    x, y, w, h = box
                    center_x = int(x) 
                    center_y = int(y)
                    
                    # Drawing the bounding box
                    # cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                    
                    # Drawing the center of the bounding box
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    text = f"x:{adjusted_point[0]:.2f}, y:{adjusted_point[1]:.2f}, z:{adjusted_point[2]:.2f}"
                    cv2.putText(annotated_frame, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the annotated frame with a reduced size for better visualization
            display_scale = 0.5
            resized_frame = cv2.resize(annotated_frame, (int(annotated_frame.shape[1] * display_scale), int(annotated_frame.shape[0] * display_scale)))
            cv2.imshow("YOLOv8 Tracking", resized_frame)

        except AttributeError:
            display_scale = 0.5
            resized_frame = cv2.resize(annotated_frame, (int(annotated_frame.shape[1] * display_scale), int(annotated_frame.shape[0] * display_scale)))
            cv2.imshow("YOLOv8 Tracking", resized_frame)


    def _process_detected_object(self, box, track_id, point_cloud):
        """Process each detected object, compute its real-world coordinates and track its movement."""
        x, y, w, h = box
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        err, point_cloud_value = point_cloud.get_value(center_x, center_y)

        if err != sl.ERROR_CODE.SUCCESS:
            return

        # Compute real-world distances between points for better accuracy
        _, top_left = point_cloud.get_value(int(x), int(y))
        _, top_right = point_cloud.get_value(int(x + w), int(y))
        _, bottom_left = point_cloud.get_value(int(x), int(y + h))
        real_world_width = np.linalg.norm(np.array(top_right[:3]) - np.array(top_left[:3]))
        real_world_height = np.linalg.norm(np.array(bottom_left[:3]) - np.array(top_left[:3]))

        # Adjust depth for better accuracy
        half_depth = 0.05
        z_adjusted = point_cloud_value[2] + half_depth
        adjusted_point = [point_cloud_value[0], point_cloud_value[1], z_adjusted]

        self._update_track_history(track_id, adjusted_point)
        self._publish_object_data(track_id, adjusted_point)
        return adjusted_point

    def _update_track_history(self, track_id, adjusted_point):
        """Update the tracking history for each detected object."""
        track_positions = self.track_history[track_id]
        track_positions.append(adjusted_point)
        if len(track_positions) > NUMBER_OF_FRAMES_FOR_AVERAGE:
            track_positions.pop(0)

    def _publish_object_data(self, track_id, adjusted_point):
        """Publish the detected object's data to a ROS topic."""
        avg_position = np.mean(self.track_history[track_id], axis=0)
        twist = Twist()
        twist.linear.x = avg_position[0]
        twist.linear.y = avg_position[1]
        twist.linear.z = avg_position[2]
        twist.angular.x = float(track_id)
        self.publisher.publish(twist)

    def run(self):
        """Main loop to continuously process camera data."""
        while True:
            frame, point_cloud = self._retrieve_camera_data()
            if frame is not None:
                self._process_frame(frame, point_cloud)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    rclpy.init()
    estimator = ObjectDistanceEstimator('/home/ofek/ros2_ws/src/cv_to_arm/src/yolov8n.pt')
    estimator.run()
    rclpy.spin(estimator)
