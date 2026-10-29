"""Local LLM via Ollama (https://ollama.com) with JSON-schema constrained decoding."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import Backend, BackendError


class OllamaBackend(Backend):
    name = 'ollama'

    def __init__(self, model: str = 'qwen2.5:7b-instruct', host: str = 'http://localhost:11434',
                 timeout: float = 30.0, keep_alive: str = '30m'):
        self.model = model
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.keep_alive = keep_alive

    def interpret(self, system_prompt: str, messages: list[dict], schema: dict) -> dict:
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': system_prompt}, *messages],
            'format': schema,
            'stream': False,
            'keep_alive': self.keep_alive,
            'options': {'temperature': 0, 'num_predict': 256},   # the JSON is short; stops runaway answers
        }
        req = urllib.request.Request(f'{self.host}/api/chat', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BackendError(f'Ollama not reachable at {self.host}: {e}') from e
        try:
            return json.loads(data['message']['content'])
        except (KeyError, json.JSONDecodeError) as e:
            raise BackendError(f'Ollama returned unparseable output: {data!r:.300}') from e

    def warmup(self) -> None:
        """Load the model into GPU memory so the first real request is fast."""
        req = urllib.request.Request(f'{self.host}/api/generate',
                                     data=json.dumps({'model': self.model, 'keep_alive': self.keep_alive}).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(req, timeout=120).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BackendError(f'Ollama warmup failed: {e}') from e
