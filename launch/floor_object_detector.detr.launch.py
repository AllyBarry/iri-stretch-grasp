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
                        "vlm_enabled": False,
                        "vlm_base_url": "http://127.0.0.1:8080/v1",
                        "vlm_api_key": "sk-no-key-required",
                        "vlm_model": "",
                        "max_bbox_volume_m3": 0.0,
                        "max_distance_above_floor_m": 0.0,
                        "floor_z_in_base_frame_m": 0.0,
                    }
                ],
            )
        ]
    )
