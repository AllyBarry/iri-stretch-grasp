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
                        "use_ram_labels": False,
                        "ram_model_path": "ram_plus_swin_large_14m.pth",
                        "ram_image_size": 384,
                        "ram_batch_size": 1,
                        "vlm_enabled": True,
                        "vlm_base_url": "http://127.0.0.1:8001/v1",
                        "vlm_api_key": "sk-no-key-required",
                        "vlm_model": "unsloth/gemma-4-E4B-it-GGUF",
                        "max_bbox_volume_m3": 1.0,
                        "max_distance_above_floor_m": 0.3,
                        "floor_z_in_base_frame_m": 0.0,
                    }
                ],
            )
        ]
    )
