"""Transcript -> TaskCommand. Also serves ~/parse_text for text-only use and testing.

Stop phrases are handled immediately in the subscription callback; everything else is queued to a
worker thread so a slow LLM call never delays a "stop".
"""
import json
import os
import queue
import threading

import rclpy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import KeyValue
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from speech_task_interfaces.msg import TaskCommand, Transcript
from speech_task_interfaces.srv import ParseText, ParseToJson

from .backends import BackendError, make_backend
from .capabilities import Capabilities, VERSION as CAPS_VERSION
from .interpreter import Interpreter, InterpreterConfig
from .translator import Translator
from .task_json import task_json
from std_msgs.msg import String

STATUS = {'execute': TaskCommand.STATUS_EXECUTE, 'clarify': TaskCommand.STATUS_CLARIFY,
          'infeasible': TaskCommand.STATUS_INFEASIBLE, 'ignore': TaskCommand.STATUS_IGNORE}


def to_msg(result: dict, transcript: str, utterance_id: str, stamp) -> TaskCommand:
    msg = TaskCommand()
    msg.header.stamp = stamp
    msg.utterance_id = utterance_id
    msg.transcript = transcript
    msg.status = STATUS[result['status']]
    task = result.get('task') or {}
    msg.task = task.get('name', '') if result['status'] == 'execute' else ''
    msg.args = [KeyValue(key=a['key'], value=a['value']) for a in task.get('args', [])] if msg.task else []
    msg.response = result.get('response', '')
    msg.confidence = float(result.get('confidence', 0.0))
    msg.backend = result.get('backend', '')
    msg.raw_json = json.dumps(result)
    return msg


def find_recipes_dir(given: str) -> str:
    """The recipes folder of iri-stretch-grasp: the parameter if set, else the usual checkouts on this machine."""
    import glob
    if given:
        return os.path.expanduser(given)
    home = os.path.expanduser('~')
    for pattern in ('mock_robot_ws/src/stretch_floor_perception/config/recipes',
                    '*/src/*/config/recipes', '*/src/*/*/config/recipes', '*/config/recipes', 'config/recipes'):
        for d in sorted(glob.glob(os.path.join(home, pattern))):
            if glob.glob(os.path.join(d, '*.json')):
                return d
    return ''


class InterpreterNode(Node):
    def __init__(self):
        super().__init__('interpreter')
        default_caps = os.path.join(get_package_share_directory('speech_task'), 'config', 'capabilities.yaml')
        self.declare_parameter('capabilities_file', default_caps)
        self.declare_parameter('backend', 'ollama')             # ollama | anthropic | rules
        self.declare_parameter('fallback_backend', 'rules')     # used when the primary fails; '' = none
        self.declare_parameter('model', '')                     # '' = backend default
        self.declare_parameter('ollama_host', 'http://localhost:11434')
        self.declare_parameter('effort', 'low')                 # anthropic only
        self.declare_parameter('timeout', 30.0)
        self.declare_parameter('min_asr_confidence', 0.35)
        self.declare_parameter('min_plan_confidence', 0.5)
        self.declare_parameter('clarify_timeout_s', 0.0)        # >0: follow-up questions (the robot asks back)
        self.declare_parameter('translate', True)               # non-English requests are translated to English first
        self.declare_parameter('translate_backend', '')         # '' = same LLM as `backend`
        self.declare_parameter('recipes_dir', '')               # iri-stretch-grasp/config/recipes; '' = look for it
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        caps = Capabilities.from_yaml(p('capabilities_file'))
        folder = find_recipes_dir(p('recipes_dir'))
        loaded = caps.load_recipes(folder) if folder else []
        if loaded:
            self.get_logger().info(f'Recipes from {folder}: ' + ', '.join(f'{r} ({caps.recipe_types.get(r, "?")})' for r in loaded))
        else:
            self.get_logger().warn('No recipe folder found (set recipes_dir:=.../config/recipes); '
                                   'using the recipe list in capabilities.yaml')
        backend = make_backend(p('backend'), caps, model=p('model'), host=p('ollama_host'),
                               effort=p('effort'), timeout=p('timeout'))
        fallback = make_backend(p('fallback_backend'), caps) if p('fallback_backend') else None
        if hasattr(backend, 'warmup'):
            try:
                backend.warmup()
            except BackendError as e:
                self.get_logger().warn(f'{e} -- will use fallback "{p("fallback_backend")}" until it is up')
        cfg = InterpreterConfig(min_asr_confidence=p('min_asr_confidence'),
                                min_plan_confidence=p('min_plan_confidence'),
                                clarify_timeout_s=p('clarify_timeout_s'))
        self.interp = Interpreter(caps, backend, fallback, cfg, log=self.get_logger().info)
        self.translator = None
        if p('translate'):
            name = p('translate_backend') or p('backend')
            if name == 'rules':
                self.get_logger().warn('translate: the rules backend cannot translate; non-English requests '
                                       'go to the interpreter untranslated')
            else:
                tb = backend if name == p('backend') else make_backend(
                    name, caps, model=p('model'), host=p('ollama_host'), effort=p('effort'), timeout=p('timeout'))
                self.translator = Translator(tb, caps.vocabulary, log=self.get_logger().info)
        self.lock = threading.Lock()

        self.caps = caps
        self.pub = self.create_publisher(TaskCommand, 'speech/command', 10)
        self.json_pub = self.create_publisher(String, 'speech/task_json', 10)   # what other nodes consume
        group = ReentrantCallbackGroup()
        self.create_subscription(Transcript, 'speech/transcript', self._on_transcript, 10, callback_group=group)
        self.create_service(ParseText, '~/parse_text', self._on_parse_text, callback_group=group)
        self.create_service(ParseToJson, '~/parse_json', self._on_parse_json, callback_group=group)

        self.work: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        self.get_logger().info(f'Interpreter ready (version {CAPS_VERSION}, backend={p("backend")}, fallback={p("fallback_backend")}, '
                               f'{len(caps.actions)} actions)')

    def _on_transcript(self, msg: Transcript):
        if self.interp.is_stop(msg.text):
            while not self.work.empty():  # drop anything still queued
                self.work.get_nowait()
            with self.lock:
                result = self.interp.interpret(msg.text)
            self._publish(result, msg.text, msg.utterance_id)
            return
        self.work.put(msg)

    def _worker(self):
        while rclpy.ok():
            msg = self.work.get()
            typed = msg.audio_duration <= 0.0 and msg.confidence >= 1.0
            text, origin = self._to_english(msg.text, msg.language, typed)
            with self.lock:
                result = self.interp.interpret(text, msg.confidence, msg.no_speech_prob,
                                               uncertain_words=[] if origin else list(msg.uncertain_words))
            if origin:
                result['translated_from'] = origin
            self._publish(result, text, msg.utterance_id, msg.language)

    def _on_parse_text(self, req, res):
        text, origin = self._to_english(req.text, '', typed=True)
        with self.lock:
            result = self.interp.interpret(text, use_context=req.use_dialogue_context)
        if origin:
            result['translated_from'] = origin
        res.command = to_msg(result, text, 'srv', self.get_clock().now().to_msg())
        return res

    def _on_parse_json(self, req, res):
        text, origin = self._to_english(req.text, '', typed=True)
        with self.lock:
            result = self.interp.interpret(text, use_context=False)
        if origin:
            result['translated_from'] = origin
        res.json = json.dumps(task_json(result, self.caps), ensure_ascii=False)
        return res

    def _to_english(self, text: str, language: str, typed: bool):
        """(text to interpret, {'language', 'text'} of the original if it was translated, else None)."""
        if self.translator is None or self.interp.is_stop(text) or \
                not self.translator.needs_translation(text, language, typed):
            return text, None
        try:
            english, detected = self.translator.to_english(text)
        except BackendError as e:
            self.get_logger().warn(f'translation failed ({e}); interpreting the original text')
            return text, None
        if (detected or '').startswith('en') or ' '.join(english.lower().split()) == ' '.join(text.lower().split()):
            return text, None  # it was English after all: keep the person's own words
        lang = detected or language or '?'
        self.get_logger().info(f'[translate] ({lang}) "{text}" -> "{english}"')
        return english, {'language': lang, 'text': text}

    def _publish(self, result: dict, text: str, utterance_id: str, language: str = 'en'):
        msg = to_msg(result, text, utterance_id, self.get_clock().now().to_msg())
        self.pub.publish(msg)
        out = task_json(result, self.caps)
        if out['task'] is not None:   # only actionable requests go to the other nodes
            self.json_pub.publish(String(data=json.dumps(out, ensure_ascii=False)))
            origin = result.get('translated_from')
            self.get_logger().info(f'[{utterance_id}] JSON {json.dumps(out, ensure_ascii=False)}'
                                   + (f'  (from {origin["language"]}: "{origin["text"]}")' if origin else ''))
        task = f'{msg.task}({", ".join(f"{a.key}={a.value}" for a in msg.args)})' if msg.task else ''
        self.get_logger().info(f'[{utterance_id}] {result["status"].upper()} via {msg.backend}: '
                               f'{task or "-"} | "{msg.response}"')


def main():
    rclpy.init()
    node = InterpreterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
