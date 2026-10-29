"""Keyword/regex interpreter. No dependencies; used for tests and as fallback when no LLM is reachable.

It only handles simple phrasings and asks for clarification otherwise.
"""
from __future__ import annotations

import re

from ..capabilities import NO_TASK, Capabilities
from .base import Backend

_FILLER = {'it', 'them', 'that', 'this', 'something', 'me', 'up'}


def _find(text: str, vocab: list[str]) -> str | None:
    for word in sorted(vocab, key=len, reverse=True):
        if re.search(rf'\b{re.escape(word.replace("_", " "))}s?\b', text):
            return word
    return None


class RuleBackend(Backend):
    name = 'rules'

    def __init__(self, caps: Capabilities):
        self.caps = caps

    def interpret(self, system_prompt: str, messages: list[dict], schema: dict) -> dict:
        # Merge all user turns so a clarification answer completes the original request.
        text = ' '.join(m['content'].split('\n[')[0] for m in messages if m['role'] == 'user').lower()
        loc = _find(text, self.caps.vocabulary.get('locations', []))
        obj = _find(text, self.caps.vocabulary.get('objects', []))
        if obj is None:  # objects are an open set: take the noun after "the/my/a"
            m = re.search(r"\b(?:the|my|a|an|some|your)\s+([a-z]+)", text)
            if m and m.group(1) not in _FILLER and m.group(1) not in self.caps.vocabulary.get('locations', []):
                obj = m.group(1)

        def ok(task, response, **args):
            if task not in self.caps.actions:
                return no('infeasible', "Sorry, I can't do that.")
            return {'status': 'execute', 'response': response, 'confidence': 0.6,
                    'task': {'name': task, 'args': [{'key': k, 'value': v} for k, v in args.items() if v]}}

        def no(status, response):
            return {'status': status, 'task': {'name': NO_TASK, 'args': []}, 'response': response,
                    'confidence': 0.3 if status == 'clarify' else 0.6}

        if re.search(r'\bpass\b', text):
            to = re.search(r'\bto\s+([a-z]+)\s*$', text)
            recipient = to.group(1) if to and to.group(1) not in _FILLER else None
            return ok('pass', f'Passing the {obj}.', object=obj, to=recipient) if obj \
                else no('clarify', 'What should I pass?')
        if re.search(r'\b(bring|fetch|get|grab)\b', text):
            return ok('fetch', f'Okay, getting the {obj}.', object=obj, **{'from': loc}) if obj \
                else no('clarify', 'What should I bring?')
        if re.search(r'\b(pick up|pick|take)\b', text):
            return ok('pick_up', f'Picking up the {obj}.', object=obj, location=loc) if obj \
                else no('clarify', 'What should I pick up?')
        if re.search(r'\b(put|place|drop)\b', text):
            return ok('put', f'Putting it on the {loc}.', location=loc) if loc \
                else no('clarify', 'Where should I put it?')
        if re.search(r'\b(find|look for|where is|where are|search)\b', text):
            return ok('find', f'Looking for the {obj}.', object=obj, location=loc) if obj \
                else no('clarify', 'What should I look for?')
        if re.search(r'\b(go|drive|navigate|move to)\b', text):
            return ok('go_to', f'Going to the {loc.replace("_", " ")}.', location=loc) if loc \
                else no('clarify', 'Where should I go?')
        if re.search(r'\b(give|hand)\b', text):
            return ok('give', 'Here you go.')
        if re.search(r'\b(stow|tuck|retract)\b', text):
            return ok('stow', 'Stowing my arm.')
        if re.search(r'\b(cook|clean|wash|stairs|upstairs|open the door|fly)\b', text):
            return no('infeasible', "Sorry, I can't do that.")
        return no('clarify', "Sorry, I didn't understand. What would you like me to do?")
