from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="stretch_floor_perception",
                executable="floor_object_detector.py",
                name="floor_object_detector",
                output="screen",
                parameters=[
                    {
                        "color_topic": "/camera/color/image_raw",
                        "aligned_depth_topic": "/camera/depth/image_rect_raw",
                        "camera_info_topic": "/camera/color/camera_info",
                        "camera_frame": "camera_depth_optical_frame",
                        "base_frame": "base_link",
                        "yolo_backend": "yolo-darknet",
                        "yolo_model_path": "/workspaces/stretch_dev/yolov4-tiny.weights",
                        "yolo_config_file": "/workspaces/darknet/cfg/yolov4-tiny.cfg",
                        "yolo_class_names_file": "/workspaces/darknet/cfg/coco.names",
                        "publish_mask_topic": False,
                    }
                ],
            )
        ]
    )
