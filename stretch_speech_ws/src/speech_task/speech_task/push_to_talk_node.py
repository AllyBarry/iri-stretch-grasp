"""Push-to-talk window: hold SPACE (or the mouse button on the window) to talk.

Publishes /speech/ptt (std_msgs/Bool): true while held (re-sent every 200 ms as a heartbeat), false on
release. Used with `audio_capture` in `mode:=push_to_talk`, which may run on another machine (the robot).

Keys only count while this window has keyboard focus, so typing spaces in other programs does nothing.
The window also shows what was heard and how the robot interpreted it.
"""
import os
import queue
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from speech_task_interfaces.msg import TaskCommand, Transcript

HEARTBEAT_MS = 200
RELEASE_DEBOUNCE_MS = 60   # X11 key auto-repeat sends release+press pairs while a key is held
STATUS = {TaskCommand.STATUS_EXECUTE: ('EXECUTE', '#2e7d32'), TaskCommand.STATUS_CLARIFY: ('CLARIFY', '#ef6c00'),
          TaskCommand.STATUS_INFEASIBLE: ('NOT POSSIBLE', '#c62828'), TaskCommand.STATUS_IGNORE: ('IGNORED', '#757575')}


class PushToTalkNode(Node):
    def __init__(self, ui_queue: queue.Queue):
        super().__init__('push_to_talk')
        self.declare_parameter('key', 'space')   # Tk keysym, e.g. space, F1, Control_R
        self.key = self.get_parameter('key').value
        self.pub = self.create_publisher(Bool, 'speech/ptt', 10)
        self.ui = ui_queue
        self.create_subscription(Transcript, 'speech/transcript', lambda m: self.ui.put(('heard', m)), 10)
        self.create_subscription(TaskCommand, 'speech/command', lambda m: self.ui.put(('command', m)), 10)

    def set_pressed(self, pressed: bool):
        self.pub.publish(Bool(data=pressed))


class PushToTalkWindow:
    def __init__(self, node: PushToTalkNode, ui_queue: queue.Queue):
        import tkinter as tk
        self.tk = tk
        self.node = node
        self.ui = ui_queue
        self.pressed = False
        self._release_job = None

        self.root = tk.Tk()
        self.root.title('Stretch push-to-talk')
        self.root.geometry('520x300')
        self.root.configure(bg='#202124')
        key_name = node.key.upper() if len(node.key) > 1 else node.key
        self.idle_text = f'Hold {key_name} to talk'
        self.button = tk.Label(self.root, text=self.idle_text, font=('Sans', 20, 'bold'),
                               bg='#3c4043', fg='white', height=3, relief='raised', bd=3)
        self.button.pack(fill='x', padx=16, pady=(16, 8))
        self.heard = tk.Label(self.root, text='', font=('Sans', 12), bg='#202124', fg='#bdc1c6',
                              wraplength=480, justify='left', anchor='w')
        self.heard.pack(fill='x', padx=16)
        self.status = tk.Label(self.root, text='', font=('Sans', 12, 'bold'), bg='#202124', fg='white', anchor='w')
        self.status.pack(fill='x', padx=16, pady=(8, 0))
        self.detail = tk.Label(self.root, text='', font=('Sans', 11), bg='#202124', fg='#e8eaed',
                               wraplength=480, justify='left', anchor='w')
        self.detail.pack(fill='x', padx=16)

        self.root.bind(f'<KeyPress-{node.key}>', lambda e: self._press())
        self.root.bind(f'<KeyRelease-{node.key}>', lambda e: self._schedule_release())
        self.button.bind('<ButtonPress-1>', lambda e: self._press())
        self.button.bind('<ButtonRelease-1>', lambda e: self._release())
        self.root.bind('<FocusOut>', lambda e: self._release())   # never leave the mic "stuck" open
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(HEARTBEAT_MS, self._heartbeat)
        self.root.after(50, self._poll_ui)

    # ---------------------------------------------------------------- key state
    def _press(self):
        if self._release_job is not None:   # auto-repeat: the "release" was not real
            self.root.after_cancel(self._release_job)
            self._release_job = None
        if not self.pressed:
            self.pressed = True
            self.node.set_pressed(True)
            self.button.configure(text='● Listening…', bg='#c62828', relief='sunken')

    def _schedule_release(self):
        if self._release_job is None:
            self._release_job = self.root.after(RELEASE_DEBOUNCE_MS, self._release)

    def _release(self):
        self._release_job = None
        if self.pressed:
            self.pressed = False
            self.node.set_pressed(False)
            self.button.configure(text='Processing…', bg='#3c4043', relief='raised')
            self.root.after(1500, lambda: None if self.pressed else self.button.configure(text=self.idle_text))

    def _heartbeat(self):
        if self.pressed:
            self.node.set_pressed(True)
        self.root.after(HEARTBEAT_MS, self._heartbeat)

    # ------------------------------------------------------------------ display
    def _poll_ui(self):
        while not self.ui.empty():
            kind, msg = self.ui.get_nowait()
            if kind == 'heard':
                self.heard.configure(text=f'Heard: "{msg.text}"' if msg.text else 'Heard: (nothing)')
                if not self.pressed:
                    self.button.configure(text=self.idle_text)
            else:
                name, colour = STATUS.get(msg.status, (str(msg.status), 'white'))
                self.status.configure(text=name, fg=colour)
                task = f'Task: {msg.task}({", ".join(f"{a.key}={a.value}" for a in msg.args)})' if msg.task else ''
                self.detail.configure(text=(f'Robot: "{msg.response}"' if msg.response else '') +
                                      (f'\n{task}' if task else ''))
        self.root.after(50, self._poll_ui)

    def run(self):
        self.root.mainloop()

    def close(self):
        if self.pressed:
            self.node.set_pressed(False)
        self.root.destroy()


def main():
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        sys.exit('push_to_talk needs a desktop session (no DISPLAY set). Run it from a terminal on the desktop.')
    rclpy.init()
    ui_queue: queue.Queue = queue.Queue()
    node = PushToTalkNode(ui_queue)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        PushToTalkWindow(node, ui_queue).run()
    except KeyboardInterrupt:
        pass
    finally:
        node.set_pressed(False)
        node.destroy_node()
        rclpy.try_shutdown()
