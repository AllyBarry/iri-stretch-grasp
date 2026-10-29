"""AudioUtterance -> Transcript using faster-whisper (CTranslate2). Runs on the GPU PC."""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from speech_task_interfaces.msg import AudioUtterance, Transcript

from .capabilities import Capabilities


class TranscriberNode(Node):
    def __init__(self):
        super().__init__('transcriber')
        default_caps = os.path.join(get_package_share_directory('speech_task'), 'config', 'capabilities.yaml')
        # large-v3-turbo: multilingual, ~0.4-0.7 s per utterance on an RTX 3070 Ti.
        # English-only alternatives: distil-large-v3 (~0.3 s, the old default), small.en (fastest, ~0.1 s).
        self.declare_parameter('model', 'large-v3-turbo')
        self.declare_parameter('device', 'auto')          # auto | cuda | cpu
        self.declare_parameter('compute_type', 'default')  # float16 (GPU), int8 (CPU), ...
        self.declare_parameter('language', 'auto')         # auto = detect per utterance among `languages`; or e.g. en
        self.declare_parameter('languages', ['en', 'es', 'ca'])  # candidates for detection; ['any'] = all of Whisper's
        self.declare_parameter('beam_size', 5)
        self.declare_parameter('capabilities_file', default_caps)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        from faster_whisper import WhisperModel
        device = p('device')
        if device == 'auto':
            import ctranslate2
            device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
        compute = p('compute_type')
        if compute == 'default':
            # int8 weights save ~1 GB VRAM (leaves room for the LLM on an 8 GB GPU) at near-identical accuracy
            compute = 'int8_float16' if device == 'cuda' else 'int8'
        self.get_logger().info(f'Loading whisper "{p("model")}" on {device} ({compute})...')
        self.model = WhisperModel(p('model'), device=device, compute_type=compute)
        lang = (p('language') or 'auto').strip().lower()
        self.language = None if lang == 'auto' else lang
        self.languages = [] if list(p('languages')) in ([], ['any']) else [l.strip().lower() for l in p('languages')]
        if (p('model').endswith('.en') or 'distil' in p('model')) and self.language != 'en':
            self.get_logger().warn(f'whisper "{p("model")}" is English-only: transcribing everything as English. '
                                   'Use model:=large-v3-turbo to understand other languages.')
            self.language = 'en'
        self.get_logger().info('Language: ' + (self.language or f'auto-detect among {self.languages or "all"}'))
        self.beam_size = int(p('beam_size'))
        # Priming the decoder with example commands strongly improves recognition of the command verbs
        # and of location/object names (see `asr:` in capabilities.yaml).
        caps = Capabilities.from_yaml(p('capabilities_file'))
        self.prompt = caps.asr_prompt()
        self.uncertain_prob = caps.uncertain_word_prob
        self.get_logger().info(f'ASR prompt: {self.prompt}')

        self.model.transcribe(np.zeros(16000, dtype=np.float32), language='en')  # warm-up
        self.pub = self.create_publisher(Transcript, 'speech/transcript', 10)
        self.create_subscription(AudioUtterance, 'speech/utterance', self._on_audio, 10)
        self.get_logger().info('Transcriber ready')

    def _detect_language(self, audio) -> str:
        """Most likely language among the candidates. transcribe() detects the language before it decodes
        anything, so taking `info` without reading the segments costs one encoder pass."""
        _, info = self.model.transcribe(audio, language=None, vad_filter=True, without_timestamps=True)
        probs = dict(getattr(info, 'all_language_probs', None) or [(info.language, info.language_probability)])
        candidates = self.languages or list(probs)
        best = max(candidates, key=lambda code: probs.get(code, 0.0))
        return best if probs.get(best, 0.0) > 0.0 else candidates[0]

    def _on_audio(self, msg: AudioUtterance):
        if msg.sample_rate != 16000:
            self.get_logger().error(f'expected 16 kHz audio, got {msg.sample_rate}')
            return
        audio = np.asarray(msg.samples, dtype=np.float32) / 32768.0
        language = self.language or self._detect_language(audio)
        # The priming prompt is English, so only use it for English speech: on other languages it pushes
        # Whisper towards English words. Non-English text is translated by the interpreter.
        segments, info = self.model.transcribe(
            audio, language=language, beam_size=self.beam_size, vad_filter=True,
            initial_prompt=self.prompt if language == 'en' else None, word_timestamps=True,
            condition_on_previous_text=False)
        segments = list(segments)

        out = Transcript()
        out.header = msg.header
        out.utterance_id = msg.utterance_id
        out.text = ' '.join(s.text.strip() for s in segments).strip()
        out.language = info.language
        out.audio_duration = float(len(audio) / 16000)
        if segments:
            n = sum(max(len(s.tokens), 1) for s in segments)
            avg_logprob = sum(s.avg_logprob * max(len(s.tokens), 1) for s in segments) / n
            out.confidence = float(math.exp(avg_logprob))
            out.no_speech_prob = float(max(s.no_speech_prob for s in segments))
        else:
            out.confidence, out.no_speech_prob = 0.0, 1.0
        out.uncertain_words = [w.word.strip(' .,!?') for s in segments for w in (s.words or [])
                               if w.probability < self.uncertain_prob and w.word.strip(' .,!?')]
        self.pub.publish(out)
        self.get_logger().info(f'[{out.utterance_id}] "{out.text}" (conf {out.confidence:.2f}, '
                               f'no_speech {out.no_speech_prob:.2f}, unsure: {list(out.uncertain_words)})')


def main():
    rclpy.init()
    node = TranscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
