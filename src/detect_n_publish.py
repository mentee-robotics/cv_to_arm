from collections import defaultdict
import cv2
import pyzed.sl as sl
import numpy as np
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

NUMBER_OF_FRAMES_FOR_AVERAGE = 10  # Variable to define the number of frames for averaging

class ObjectDistanceEstimator(Node):
    def __init__(self, model_path):
        super().__init__('object_distance_estimator')
        
        self.publisher = self.create_publisher(Twist, 'detected_object', 10)
        self.model = YOLO(model_path)
        # Store the last N 3D positions for each object, where N is defined by the above variable
        self.track_history = defaultdict(lambda: [])

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera opening failed with error:", err)
            exit(1)

        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.confidence_threshold = 100
        self.runtime_parameters.texture_confidence_threshold = 100

        # Initialize 3D plotting
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim([0, 3])
        self.ax.set_ylim([0, 3])
        self.ax.set_zlim([0, 3])
        

    def run(self):
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        while True:
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                
                frame = image.get_data()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Track the objects in the frame
                results = self.model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                if results and results[0].boxes:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        center_x = int(x + w / 2)
                        center_y = int(y + h / 2)
                        err, point_cloud_value = point_cloud.get_value(center_x, center_y)
                        if err == sl.ERROR_CODE.SUCCESS:
                            track_positions = self.track_history[track_id]

                            _, top_left = point_cloud.get_value(int(x), int(y))
                            _, top_right = point_cloud.get_value(int(x + w), int(y))
                            _, bottom_left = point_cloud.get_value(int(x), int(y + h))

                            # Calculate real-world width and height
                            real_world_width = np.linalg.norm(np.array(top_right[:3]) - np.array(top_left[:3]))
                            real_world_height = np.linalg.norm(np.array(bottom_left[:3]) - np.array(top_left[:3]))

                            #add depth as half of the smallest edge of the bounding box
                            # half_depth = (min(real_world_width, real_world_height) / 2)
                            half_depth = 0.05
                            # print(f"extra depth: {half_depth}")
                            z_adjusted = point_cloud_value[2] + half_depth
                            adjusted_point = [point_cloud_value[0], point_cloud_value[1], z_adjusted]
                            track_positions.append(adjusted_point)  # x, y, adjusted z position

                            if len(track_positions) > NUMBER_OF_FRAMES_FOR_AVERAGE:
                                track_positions.pop(0)

                            # Calculate average position and plot
                            avg_position = np.mean(track_positions, axis=0)
                            self.ax.scatter(avg_position[0], avg_position[1], avg_position[2], label=str(track_id))

                            # Publish the Twist message with position coordinates and object ID
                            twist = Twist()
                            twist.linear.x = avg_position[0]
                            twist.linear.y = avg_position[1]
                            twist.linear.z = avg_position[2]
                            twist.angular.x = float(track_id)
                            self.publisher.publish(twist)

                    # Display 3D plot and frame
                    plt.draw()
                    plt.legend()
                    plt.pause(0.001)
                    self.ax.clear()
                    self.ax.set_xlabel('X')
                    self.ax.set_ylabel('Y')
                    self.ax.set_zlabel('Z')
                    self.ax.set_xlim([0, 3])
                    self.ax.set_ylim([0, 3])
                    self.ax.set_zlim([0, 3])
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    continue

        cv2.destroyAllWindows()
        plt.close(self.fig)

rclpy.init()
estimator = ObjectDistanceEstimator('/home/ofek/ros2_ws/src/cv_to_arm/src/yolov8n.pt')
estimator.run()
rclpy.spin(estimator)