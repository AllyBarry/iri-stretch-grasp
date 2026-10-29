"""Microphone -> utterance segmentation -> AudioUtterance messages.

Two listening modes (parameter `mode`):
  vad           -- always listening; voice activity detection decides where utterances start and end
  push_to_talk  -- records only while /speech/ptt (std_msgs/Bool) is true, e.g. while SPACE is held
                   in the `push_to_talk` window. The key and the mic can be on different machines.

Lightweight (no ML models), so it can run on the robot (Stretch's ReSpeaker array) or on the PC.
Only complete utterances are sent over the network (~32 kB/s of speech), never a continuous stream.
"""
import array
import collections
import queue
import time
import os
import uuid
import wave

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from std_msgs.msg import Bool

from speech_task_interfaces.msg import AudioUtterance

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000


class EnergyVad:
    """Fallback VAD: adaptive noise floor + fixed SNR margin."""

    def __init__(self, margin_db: float = 12.0):
        self.noise_db = -60.0
        self.margin_db = margin_db

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9
        db = 20 * np.log10(rms / 32768.0)
        speech = db > self.noise_db + self.margin_db
        if not speech:  # track the noise floor only during silence
            self.noise_db = 0.95 * self.noise_db + 0.05 * db
        return speech


class WebrtcVad:
    def __init__(self, aggressiveness: int):
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)


class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture')
        self.declare_parameter('device', '')              # sounddevice name substring or index; '' = default
        self.declare_parameter('mode', 'vad')             # vad | push_to_talk
        self.declare_parameter('release_tail_ms', 250)    # push_to_talk: keep recording briefly after release
        self.declare_parameter('ptt_timeout_s', 1.0)      # push_to_talk: no /speech/ptt heartbeat -> treat as released
        self.declare_parameter('vad_aggressiveness', 2)   # webrtcvad 0..3
        self.declare_parameter('pre_roll_ms', 300)
        self.declare_parameter('end_silence_ms', 700)
        self.declare_parameter('min_utterance_ms', 400)
        self.declare_parameter('max_utterance_ms', 15000)
        self.declare_parameter('frame_id', 'respeaker_base')
        self.declare_parameter('save_dir', '')            # if set, every utterance is also saved as <id>.wav
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        self.mode = p('mode')
        if self.mode not in ('vad', 'push_to_talk'):
            raise ValueError(f'mode must be vad or push_to_talk, got {self.mode!r}')
        try:
            self.vad = WebrtcVad(int(p('vad_aggressiveness')))
        except ImportError:
            self.get_logger().warn('webrtcvad not installed; using energy VAD')
            self.vad = EnergyVad()

        self.pre_roll = collections.deque(maxlen=p('pre_roll_ms') // FRAME_MS)
        self.end_frames = p('end_silence_ms') // FRAME_MS
        self.min_frames = p('min_utterance_ms') // FRAME_MS
        self.max_frames = p('max_utterance_ms') // FRAME_MS
        self.frame_id = p('frame_id')
        self.save_dir = p('save_dir')
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.pub = self.create_publisher(AudioUtterance, 'speech/utterance', 10)
        # Mute while the robot itself is talking so it does not transcribe its own voice.
        self.muted = False
        self.create_subscription(Bool, 'speech/speaking', self._on_speaking, 10)

        self.ptt_pressed = False
        self.ptt_last_msg = 0.0
        self.ptt_release_frames = 0     # >0: key released, still recording the tail
        self.held_frames = 0            # frames recorded while the key was actually down
        self.tail_frames = max(1, p('release_tail_ms') // FRAME_MS)
        self.ptt_timeout = float(p('ptt_timeout_s'))
        if self.mode == 'push_to_talk':
            self.create_subscription(Bool, 'speech/ptt', self._on_ptt, 10)

        self.frames: queue.Queue = queue.Queue(maxsize=500)
        self.utt: list[np.ndarray] = []
        self.silence = 0
        self.in_speech = False

        device = p('device')
        device = int(device) if str(device).isdigit() else (device or None)
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=FRAME_LEN,
                                     device=device, callback=self._audio_cb)
        self.stream.start()
        self.create_timer(0.01, self._process)
        how = type(self.vad).__name__ if self.mode == 'vad' else 'push-to-talk on /speech/ptt'
        self.get_logger().info(f'Listening on "{sd.query_devices(self.stream.device)["name"]}" ({how})')

    def _on_speaking(self, msg: Bool):
        self.muted = msg.data
        if self.muted and self.mode == 'vad':  # in push_to_talk the user decides; muted frames are just skipped
            self._reset()

    def _on_ptt(self, msg: Bool):
        self.ptt_last_msg = time.monotonic()
        if msg.data and not self.ptt_pressed:
            self.ptt_pressed = True
            self.ptt_release_frames = 0
            if not self.in_speech:  # start a new utterance, including the pre-roll buffer
                self.in_speech = True
                self.utt = list(self.pre_roll)
                self.held_frames = 0
                self.get_logger().info('push-to-talk: recording')
        elif not msg.data and self.ptt_pressed:
            self._ptt_release()

    def _ptt_release(self):
        self.ptt_pressed = False
        self.ptt_release_frames = self.tail_frames

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            self.get_logger().debug(str(status))
        try:
            self.frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _process(self):
        if (self.mode == 'push_to_talk' and self.ptt_pressed
                and time.monotonic() - self.ptt_last_msg > self.ptt_timeout):
            self.get_logger().warn('push-to-talk heartbeat lost; treating key as released')
            self._ptt_release()
        while not self.frames.empty():
            frame = self.frames.get_nowait()
            if self.muted or len(frame) != FRAME_LEN:
                continue
            if self.mode == 'push_to_talk':
                self._process_ptt(frame)
            else:
                self._process_vad(frame)

    def _process_vad(self, frame: np.ndarray):
        speech = self.vad.is_speech(frame)
        if not self.in_speech:
            self.pre_roll.append(frame)
            if speech:
                self.in_speech = True
                self.utt = list(self.pre_roll)
                self.silence = 0
            return
        self.utt.append(frame)
        self.silence = 0 if speech else self.silence + 1
        if self.silence >= self.end_frames or len(self.utt) >= self.max_frames:
            self._emit()

    def _process_ptt(self, frame: np.ndarray):
        if not self.in_speech:
            self.pre_roll.append(frame)
            return
        self.utt.append(frame)
        self.held_frames += self.ptt_pressed
        if len(self.utt) >= self.max_frames:
            self.get_logger().warn('push-to-talk: max utterance length reached')
            self.ptt_pressed = False
            self._emit()
        elif not self.ptt_pressed:
            self.ptt_release_frames -= 1
            if self.ptt_release_frames <= 0:
                self._emit()

    def _emit(self):
        # push_to_talk: a tap shorter than min_utterance_ms is treated as accidental
        voiced = self.held_frames if self.mode == 'push_to_talk' else len(self.utt) - self.silence
        if voiced >= self.min_frames:
            msg = AudioUtterance()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.utterance_id = uuid.uuid4().hex[:8]
            msg.sample_rate = SAMPLE_RATE
            msg.samples = array.array('h', np.concatenate(self.utt).tobytes())
            self.pub.publish(msg)
            if self.save_dir:
                with wave.open(os.path.join(self.save_dir, f'{msg.utterance_id}.wav'), 'wb') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(msg.samples.tobytes())
            self.get_logger().info(f'utterance {msg.utterance_id}: {len(msg.samples) / SAMPLE_RATE:.1f}s')
        self._reset()

    def _reset(self):
        self.in_speech = False
        self.utt = []
        self.silence = 0
        self.held_frames = 0
        self.pre_roll.clear()

    def destroy_node(self):
        self.stream.stop()
        self.stream.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = AudioCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
