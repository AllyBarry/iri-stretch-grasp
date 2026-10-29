"""Speaks TaskCommand.response (questions, refusals, acknowledgements) and publishes speech/speaking
so the audio capture node mutes itself meanwhile. Uses espeak-ng / spd-say if installed, else just logs."""
import shutil
import subprocess

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from speech_task_interfaces.msg import TaskCommand


class TtsNode(Node):
    def __init__(self):
        super().__init__('tts')
        self.declare_parameter('engine', 'auto')   # auto | espeak-ng | spd-say | none
        self.declare_parameter('voice_args', ['-s', '160'])
        engine = self.get_parameter('engine').value
        if engine == 'auto':
            engine = next((e for e in ('espeak-ng', 'spd-say') if shutil.which(e)), 'none')
        self.engine = engine
        self.args = list(self.get_parameter('voice_args').value) if engine == 'espeak-ng' else ['-w']
        self.speaking = self.create_publisher(Bool, 'speech/speaking', 10)
        self.create_subscription(TaskCommand, 'speech/command', self._on_command, 10)
        if engine == 'none':
            self.get_logger().warn('No TTS engine found (sudo apt install espeak-ng); responses are only logged')

    def _on_command(self, msg: TaskCommand):
        if msg.status == TaskCommand.STATUS_IGNORE or not msg.response:
            return
        self.get_logger().info(f'say: "{msg.response}"')
        if self.engine == 'none':
            return
        self.speaking.publish(Bool(data=True))
        try:
            subprocess.run([self.engine, *self.args, msg.response], timeout=30, check=False)
        except subprocess.TimeoutExpired:
            pass
        finally:
            self.speaking.publish(Bool(data=False))


def main():
    rclpy.init()
    node = TtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
