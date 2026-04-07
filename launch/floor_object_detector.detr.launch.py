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
                        "camera_frame": "camera_depth_optical_frame",
                        "base_frame": "base_link",
                        "inference_backend": "detr",
                        "model_path": "facebook/detr-resnet-50",
                    }
                ],
            )
        ]
    )
