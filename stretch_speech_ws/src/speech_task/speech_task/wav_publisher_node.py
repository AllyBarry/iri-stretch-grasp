"""Publish a 16-bit WAV file as one AudioUtterance (test the transcriber without a microphone).
    ros2 run speech_task wav_publisher --ros-args -p file:=/path/to.wav
"""
import array
import time
import uuid
import wave

import numpy as np
import rclpy
from rclpy.node import Node

from speech_task_interfaces.msg import AudioUtterance


def main():
    rclpy.init()
    node = Node('wav_publisher')
    path = node.declare_parameter('file', '').value
    pub = node.create_publisher(AudioUtterance, 'speech/utterance', 10)
    with wave.open(path) as w:
        assert w.getsampwidth() == 2, 'need 16-bit PCM'
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())[:, 0]
        rate = w.getframerate()
    if rate != 16000:  # linear resample, good enough for testing
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.int16)
    msg = AudioUtterance(utterance_id=uuid.uuid4().hex[:8], sample_rate=16000,
                         samples=array.array('h', audio.tobytes()))
    msg.header.stamp = node.get_clock().now().to_msg()
    time.sleep(1.0)  # discovery
    pub.publish(msg)
    node.get_logger().info(f'published {len(audio) / 16000:.1f}s from {path}')
    time.sleep(0.5)
    node.destroy_node()
    rclpy.try_shutdown()
