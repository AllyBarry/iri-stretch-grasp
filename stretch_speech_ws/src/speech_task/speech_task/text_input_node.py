"""Type commands instead of speaking them: publishes each stdin line as a Transcript (confidence 1.0)."""
import sys
import threading
import uuid

import rclpy
from rclpy.node import Node

from speech_task_interfaces.msg import TaskCommand, Transcript

STATUS_NAMES = {TaskCommand.STATUS_EXECUTE: 'EXECUTE', TaskCommand.STATUS_CLARIFY: 'CLARIFY',
                TaskCommand.STATUS_INFEASIBLE: 'INFEASIBLE', TaskCommand.STATUS_IGNORE: 'IGNORE'}


class TextInputNode(Node):
    def __init__(self):
        super().__init__('text_input')
        self.pub = self.create_publisher(Transcript, 'speech/transcript', 10)
        self.create_subscription(TaskCommand, 'speech/command', self._on_command, 10)

    def _on_command(self, msg: TaskCommand):
        task = f'     task: {msg.task}({", ".join(f"{a.key}={a.value}" for a in msg.args)})' if msg.task else ''
        print(f'\n  -> {STATUS_NAMES.get(msg.status, msg.status)} [{msg.backend}, conf {msg.confidence:.2f}]'
              f'\n     robot: "{msg.response}"' + (f'\n{task}' if task else '') + '\n> ', end='', flush=True)

    def send(self, text: str):
        msg = Transcript()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.utterance_id = uuid.uuid4().hex[:8]
        msg.text = text
        msg.language = 'en'
        msg.confidence = 1.0
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TextInputNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print('Type a command for the robot (Ctrl-D to quit).\n> ', end='', flush=True)
    try:
        for line in sys.stdin:
            if line.strip():
                node.send(line.strip())
            else:
                print('> ', end='', flush=True)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
