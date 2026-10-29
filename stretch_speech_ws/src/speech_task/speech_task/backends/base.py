"""Interpreter backend interface."""
from __future__ import annotations


class BackendError(RuntimeError):
    """Backend unavailable or returned something unusable (caller may fall back)."""


class Backend:
    name = 'base'

    def interpret(self, system_prompt: str, messages: list[dict], schema: dict) -> dict:
        """Return a result dict matching `schema`.

        messages: chat history, alternating {'role': 'user'|'assistant', 'content': str},
        ending with the user's latest utterance.
        """
        raise NotImplementedError
