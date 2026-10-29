"""Claude API backend with structured outputs. Needs network + ANTHROPIC_API_KEY (or `ant auth login`)."""
from __future__ import annotations

import json

from .base import Backend, BackendError


class AnthropicBackend(Backend):
    name = 'anthropic'

    def __init__(self, model: str = 'claude-opus-5-5', effort: str = 'low',
                 max_tokens: int = 16000, timeout: float = 30.0):
        import anthropic  # imported lazily so the package works without it installed
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout, max_retries=1)
        self.model = model
        # Low effort keeps voice round-trips short; this is a constrained classification-style task.
        self.effort = effort
        self.max_tokens = max_tokens

    def interpret(self, system_prompt: str, messages: list[dict], schema: dict) -> dict:
        a = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{'type': 'text', 'text': system_prompt, 'cache_control': {'type': 'ephemeral'}}],
                messages=messages,
                output_config={'effort': self.effort, 'format': {'type': 'json_schema', 'schema': schema}},
                # Server-side fallback: if a safety classifier declines, the API re-runs on a suitable model.
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',
            )
        except a.RateLimitError as e:
            raise BackendError(f'Claude API rate limited: {e}') from e
        except a.APIStatusError as e:
            raise BackendError(f'Claude API error {e.status_code}: {e.message}') from e
        except a.APIConnectionError as e:
            raise BackendError(f'Claude API not reachable: {e}') from e

        if response.stop_reason == 'refusal':
            raise BackendError('Claude declined to interpret this request')
        if response.stop_reason == 'max_tokens':
            raise BackendError('Claude response truncated (max_tokens)')
        text = next((b.text for b in response.content if b.type == 'text'), None)
        if text is None:
            raise BackendError('Claude returned no text block')
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise BackendError(f'Claude returned invalid JSON: {text[:300]}') from e
