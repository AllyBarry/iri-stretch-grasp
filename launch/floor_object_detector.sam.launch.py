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
                        "inference_backend": "fastsam",
                        "model_path": "FastSAM-s.pt",
                        "use_ram_labels": True,
                        "ram_model_path": "ram_plus_swin_large_14m.pth",
                        "ram_image_size": 384,
                        "ram_batch_size": 1,
                    }
                ],
            )
        ]
    )
