"""Turns transcripts into validated task results. Pure Python (no ROS) for testability.

Pipeline per utterance:
  1. empty / no-speech           -> ignore
  2. stop phrase                 -> execute [stop]            (fast path, no LLM)
  3. low ASR confidence          -> clarify "please repeat"   (no LLM)
  4. wake word missing           -> ignore                    (unless answering a clarification)
  5. LLM backend (+ fallback)    -> raw result
  6. capability validation       -> final result
  7. dialogue state update       (remember the question if status == clarify)
"""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field

from .backends import Backend, BackendError
from .capabilities import Capabilities, result_to_dict


@dataclass
class InterpreterConfig:
    min_asr_confidence: float = 0.35
    max_no_speech_prob: float = 0.6
    min_plan_confidence: float = 0.5    # execute results below this are turned into a confirmation question
    clarify_timeout_s: float = 30.0     # how long an open clarification question stays relevant
    max_clarifications: int = 3


@dataclass
class _Dialogue:
    messages: list[dict] = field(default_factory=list)   # user/assistant turns of the open request
    rounds: int = 0
    last_time: float = 0.0


class Interpreter:
    def __init__(self, caps: Capabilities, backend: Backend, fallback: Backend | None = None,
                 config: InterpreterConfig | None = None, log=print):
        self.caps = caps
        self.backend = backend
        self.fallback = fallback
        self.cfg = config or InterpreterConfig()
        self.log = log
        self._system_prompt = caps.system_prompt(follow_ups=self.cfg.clarify_timeout_s > 0)
        self._schema = caps.output_schema()
        self._dialogue = _Dialogue()
        stop = '|'.join(re.escape(p) for p in sorted(caps.stop_phrases, key=len, reverse=True))
        # Stop phrase must dominate the utterance ("stop!", "please stop now"), not just appear in it
        # ("don't stop at the kitchen").
        self._stop_re = re.compile(rf'^\W*(?:please\s+|robot\s+|{_wake(caps)})*(?:{stop})\W*(?:now|please|it)?\W*$',
                                   re.IGNORECASE) if stop else None

    # ------------------------------------------------------------------ public
    @property
    def awaiting_answer(self) -> bool:
        d = self._dialogue
        return bool(d.messages) and (time.monotonic() - d.last_time) < self.cfg.clarify_timeout_s

    def is_stop(self, text: str) -> bool:
        return bool(self._stop_re and self._stop_re.match(text.strip()))

    def reset(self) -> None:
        self._dialogue = _Dialogue()

    def interpret(self, text: str, asr_confidence: float = 1.0, no_speech_prob: float = 0.0,
                  use_context: bool = True, uncertain_words: list[str] | tuple = ()) -> dict:
        """Returns a validated result dict with an added 'backend' key."""
        text = ' '.join(text.split())
        if not text or no_speech_prob > self.cfg.max_no_speech_prob:
            return self._done(result_to_dict('ignore', confidence=1.0), 'fast_path', keep_dialogue=True)

        if self.is_stop(text):
            self.reset()
            return self._done(result_to_dict('execute', 'Stopping.', task='stop'), 'fast_path')

        if asr_confidence < self.cfg.min_asr_confidence:
            # Do not touch the dialogue state: the user should just repeat.
            return self._done(result_to_dict('clarify', "Sorry, I didn't catch that. Could you say it again?",
                                             confidence=asr_confidence), 'fast_path', keep_dialogue=True)

        continuing = use_context and self.awaiting_answer
        if not continuing:
            self.reset()
            if _ACK_RE.match(text):  # "yes"/"thanks" with no open question is not a request: small talk if there is a chat task
                if 'chat' in self.caps.actions:
                    return self._done(result_to_dict('execute', task='chat', args={'text': text}, confidence=1.0),
                                      'fast_path')
                return self._done(result_to_dict('ignore', confidence=1.0), 'fast_path')
            if self.caps.wake_words and not _has_wake_word(text, self.caps.wake_words):
                return self._done(result_to_dict('ignore', confidence=1.0), 'fast_path')

        content = text
        if uncertain_words:  # lets the LLM repair mis-hearings ("fitch" -> fetch) instead of asking
            content += (f'\n[ASR note: possibly misheard: {", ".join(map(repr, uncertain_words))}. Silently correct '
                        f'them to the closest-sounding listed word or command verb; do not ask about them unless '
                        f'two different corrections both make sense.]')
        self._dialogue.messages.append({'role': 'user', 'content': content})
        result, used = self._call_backends(self._dialogue.messages)
        result, used = self._no_questions(result, used, text)
        result = self._fill_from_input(result, text)
        result = self.caps.validate(result)
        if result['status'] == 'execute':
            result = self._check_grounding(result)
        if result['status'] == 'execute':
            result = self._check_single(result, text)
        if result['status'] == 'execute' and result['task']['name'] == 'set_recipe':
            result = self._check_recipe(result, text)
        if result['status'] == 'execute':
            result = self._check_intent(result, text)

        if result['status'] == 'execute' and result['confidence'] < self.cfg.min_plan_confidence:
            result = {**result, 'status': 'clarify', 'task': {'name': '', 'args': []},
                      'response': f'Just to check: should I {_describe(result["task"])}?',
                      'validation_note': f'execute -> clarify: confidence {result["confidence"]:.2f} '
                                         f'< min_plan_confidence {self.cfg.min_plan_confidence}'}
        if result['status'] == 'execute' and '?' in result.get('response', ''):
            # A question in an EXECUTE reply would never get its answer routed back (only CLARIFY opens a
            # dialogue), so speak a plain acknowledgement instead.
            result = {**result, 'response': f"Okay, I'll {_describe(result['task'])}."}
        if result.get('validation_note'):
            self.log(f'[validator] {result["validation_note"]}')

        if result['status'] == 'clarify':
            self._dialogue.rounds += 1
            if self._dialogue.rounds > self.cfg.max_clarifications:
                # Not "infeasible": nothing was judged impossible, the conversation just isn't converging.
                self.log(f'[dialogue] {self.cfg.max_clarifications} clarification rounds without a task; resetting')
                self.reset()
                return self._done(result_to_dict('clarify', "Sorry, I'm still not sure what you need. Let's start "
                                                 "over: what would you like me to do?", confidence=0.0), used)
            self._dialogue.messages.append({'role': 'assistant', 'content': json.dumps(_public(result))})
            self._dialogue.last_time = time.monotonic()
            return self._done(result, used, keep_dialogue=True)
        return self._done(result, used)

    # ----------------------------------------------------------------- private
    def _check_grounding(self, result: dict) -> dict:
        """Vocabulary arguments must have been said (or asked about) in this dialogue.

        Small LLMs like to "helpfully" add a location nobody mentioned. An optional argument whose value
        was never said, nor anything that sounds like it, is dropped; a required one turns into a
        confirmation question. Sound-alikes ("mustered" -> mustard) and the robot's own earlier questions
        ("Did you mean the desk?" -> "yes") count as mentioned.
        """
        task = result['task']
        action = self.caps.actions[task['name']]
        said = ' '.join(m['content'].split('\n[ASR note')[0] for m in self._dialogue.messages).lower()
        words = re.findall(r"[a-z0-9']+", said)
        kept, notes = [], []
        for kv in task['args']:
            p = action.params.get(kv['key'])
            if p is None or not (p.ref or p.enum) or not p.grounded or _mentioned(kv['value'], said, words):
                kept.append(kv)
            elif p.required:
                return {**result, 'status': 'clarify', 'task': {'name': '', 'args': []},
                        'response': f'Just to check: should I {_describe(task)}?',
                        'validation_note': f'execute -> clarify: {task["name"]}.{kv["key"]}={kv["value"]!r} '
                                           f'was never mentioned'}
            else:
                notes.append(f'dropped unmentioned {kv["key"]}={kv["value"]!r}')
        if not notes:
            return result
        prev = result.get('validation_note')
        task = {**task, 'args': kept}
        return {**result, 'task': task, 'response': f"Okay, I'll {_describe(task)}.",  # old reply named the dropped value
                'validation_note': '; '.join(filter(None, [prev, *notes]))}

    _NUDGE = ('The robot cannot ask questions back and does not refuse: it checks for itself what it can do. Answer '
              'again with status "execute": the task the person asked for (leave unclear parameters out), or the '
              'chat task for anything else they said.')

    def _no_questions(self, result: dict, used: str, text: str) -> tuple[dict, str]:
        """Without follow-up questions, "clarify" (and "infeasible") answers are dead ends: ask the LLM once more,
        telling it so. If the LLM is down and the rules fallback cannot place the text, it is chat."""
        if self.cfg.clarify_timeout_s > 0 or not isinstance(result, dict):
            return result, used
        if result.get('status') in ('clarify', 'infeasible') and used not in ('rules', 'none'):
            msgs = self._dialogue.messages + [{'role': 'assistant', 'content': json.dumps(result)},
                                              {'role': 'user', 'content': self._NUDGE}]
            again, used2 = self._call_backends(msgs)
            if used2 == used and isinstance(again, dict) and again.get('status') == 'execute':
                self.log(f'[no-questions] {result.get("status")} -> asked again -> {json.dumps(again.get("task"))}')
                return again, used
        if used in ('rules', 'none'):
            task = result.get('task') or {}
            if task.get('name') == 'find' and 'where' in self.caps.actions:
                return {**result, 'task': {**task, 'name': 'where'}}, used
            if result.get('status') in ('clarify', 'ignore', 'infeasible') and 'chat' in self.caps.actions:
                return result_to_dict('execute', task='chat', args={'text': text}, confidence=0.6), used
        return result, used

    _HABIT = re.compile(r"\b(usually|normally|often|always|typically|generally|sometimes|every (day|morning|evening|week))\b")
    _ASKS = re.compile(r"\b(let's|lets|let us|shall we|should we|we should|could we|can we|want|wanna|would like|i'd like|"
                       r"make|cook|prepare|now|today|tonight)\b")
    _NEED_IT = re.compile(r"\bi(?: could| can| would|'d)? (?:use|need|want) (?:it|them|that|one|those)\b|"
                          r"\b(?:bring|get|fetch|grab|pass|hand) (?:it|them|that|one|those)\b")

    def _check_intent(self, result: dict, text: str) -> dict:
        """Two cases small LLMs get wrong: "where's the cup? I could use it" is a fetch, and "I usually have a
        sandwich" (a habit, no request) is chat, not set_recipe."""
        task, said = result['task'], text.lower()
        new = None
        if task['name'] == 'where' and 'fetch' in self.caps.actions and self._NEED_IT.search(said):
            new = {**task, 'name': 'fetch'}
        elif (task['name'] == 'set_recipe' and 'chat' in self.caps.actions and self._HABIT.search(said)
              and not self._ASKS.search(said)):
            new = {'name': 'chat', 'args': [{'key': 'text', 'value': text}]}
        if new is None:
            return result
        return {**result, 'task': new, 'validation_note': '; '.join(filter(None, [
            result.get('validation_note'), f'{task["name"]} -> {new["name"]} ({text!r})']))}

    def _check_single(self, result: dict, text: str) -> dict:
        """Params marked `single` name exactly one thing. Two or more things ("the bowl and the spoon"), a choice
        ("either the basil or the dill") or no particular thing ("something to drink") -> the param is left out."""
        task = result['task']
        action = self.caps.actions[task['name']]
        said = ' ' + text.lower() + ' '
        several = bool(re.search(r"\beither\b", said) or re.search(
            r"\b(?:the|a|an|some|my|your)\s+[a-z][a-z' -]{0,30}?\s+(?:and|or)\s+(?:the|a|an|some|my|your)\s+[a-z]", said))
        kept, notes = [], []
        for kv in task['args']:
            p = action.params.get(kv['key'])
            v = str(kv['value']).lower().strip()
            if p is not None and p.single and (several or re.search(r"\b(?:and|or)\b|,|/", v)
                                                or re.match(r"(?:some|any|every|no)(?:thing|one|where)\b|whatever\b", v)):
                notes.append(f'{kv["key"]}={kv["value"]!r} is not one single thing: left out')
            else:
                kept.append(kv)
        if not notes:
            return result
        return {**result, 'task': {**task, 'args': kept},
                'validation_note': '; '.join(filter(None, [result.get('validation_note'), *notes]))}

    _GENERIC = set('a an the any some of our my your one nice simple quick healthy good fresh light big small little '
                   'tasty kind type sort those these this that either another other two both different best usual '
                   'favourite favorite whole proper easy great lovely'.split())

    def _check_recipe(self, result: dict, text: str) -> dict:
        """set_recipe sanity checks against the recipe files (the LLM likes to pick the nearest recipe):
        - recipe_name must be recognisable in what was said: its name or one of its ingredients;
        - a dish we have no recipe for ("a tuna sandwich", "a caprese sandwich") leaves both params out;
        - "a Greek salad or a cheese sandwich" (both types, no single recipe) leaves recipe_type out."""
        if not self.caps.recipe_keywords:
            return result
        task = result['task']
        args = {kv['key']: kv['value'] for kv in task['args']}
        said = text.lower()
        words = re.findall(r"[a-z]+", said)
        type_words = {t: re.findall(rf"(\w+)\s+{t}e?s?\b", said) for t in set(self.caps.recipe_types.values())}
        mentioned_types = {t for t in type_words if re.search(rf"\b{t}(e?s)?\b", said)}
        notes = []
        name = args.get('recipe_name')
        named = self._recipes_named(words)
        if name and len(named) > 1 and not re.search(
                r"\b(actually|instead|rather|prefer|changed? my mind|on second thought|no wait)\b", said):
            types = {self.caps.recipe_types.get(r) for r in named}   # "the caprese or the Greek salad": a choice
            notes.append(f'{" / ".join(sorted(named))} were all named: recipe_name left out')
            args = {'recipe_type': types.pop()} if len(types) == 1 else {}
            name = None
        if name in self.caps.recipe_keywords:
            kw = self.caps.recipe_keywords[name]
            rtype = self.caps.recipe_types.get(name)
            if not any(difflib.get_close_matches(w, kw, n=1, cutoff=0.85) or w.rstrip('s') in kw for w in words):
                notes.append(f'recipe_name={name!r}: neither its name nor its ingredients were said')
                args.pop('recipe_name')
                if 'recipe_type' not in args and rtype in mentioned_types and len(mentioned_types) == 1:
                    args['recipe_type'] = rtype   # "let's make a salad" -> greek_salad dropped, salad kept
            elif mentioned_types and rtype not in mentioned_types:   # "caprese sandwich", "ham and cheese salad"
                notes.append(f'recipe_name={name!r} is a {rtype}, but a {"/".join(sorted(mentioned_types))} was asked for')
                args = {}
        if 'recipe_name' not in args:   # described by its ingredients ("bread with ham, cheese and tomato")
            forms = set(words) | {w[:-1] for w in words if w.endswith('s')} | {w[:-2] for w in words if w.endswith('es')}
            score = sorted(((len(kw & forms), r) for r, kw in self.caps.recipe_keywords.items()), reverse=True)
            if score and score[0][0] >= 3 and (len(score) == 1 or score[0][0] >= score[1][0] + 2):
                r = score[0][1]
                if not mentioned_types or self.caps.recipe_types.get(r) in mentioned_types:
                    args = {'recipe_name': r}
                    notes.append(f'recipe_name={r!r}: described by its ingredients')
        if 'recipe_name' not in args and args:
            known = {t: set().union(*[self.caps.recipe_keywords[r] for r, rt in self.caps.recipe_types.items() if rt == t])
                     for t in type_words}
            quals = [(t, q) for t, qs in type_words.items() for q in qs]
            unknown = [f'{q} {t}' for t, q in quals if q not in self._GENERIC and q.rstrip('s') not in known[t] and q not in known[t]]
            if quals and len(unknown) == len(quals):   # every dish named is one we have no recipe for
                notes.append(f'no recipe for {", ".join(unknown)}')
                args = {}
            elif len(mentioned_types) > 1:
                notes.append('both a salad and a sandwich were mentioned: recipe_type left out')
                args.pop('recipe_type', None)
        if not notes:
            return result
        return {**result, 'task': {**task, 'args': [{'key': k, 'value': v} for k, v in args.items()]},
                'validation_note': '; '.join(filter(None, [result.get('validation_note'), *notes]))}

    def _recipes_named(self, words: list[str]) -> set:
        """Recipes whose distinctive name words were all said ("caprese", "ham ... tomato"); shared words like
        "cheese" and the type words do not count."""
        ids = {r: set(r.split('_')) - set(self.caps.recipe_types.values()) for r in self.caps.recipe_types}
        said = {w.rstrip('s') for w in words} | set(words)
        out = set()
        for r, ws in ids.items():
            own = ws - set().union(*[o for k, o in ids.items() if k != r])
            if own and own <= said:
                out.add(r)
        return out

    def _fill_from_input(self, result: dict, text: str) -> dict:
        """Params marked `fill: input` (chat.text) carry the person's own words, not the LLM's rewording."""
        task = result.get('task') if isinstance(result, dict) else None
        action = self.caps.actions.get((task or {}).get('name') or '')
        fills = [p.name for p in action.params.values() if p.fill == 'input'] if action else []
        if not fills:
            return result
        args = [kv for kv in task.get('args') or [] if kv.get('key') not in fills]
        return {**result, 'task': {**task, 'args': args + [{'key': n, 'value': text} for n in fills]}}

    def _call_backends(self, messages: list[dict]) -> tuple[dict, str]:
        for backend in filter(None, (self.backend, self.fallback)):
            try:
                t0 = time.monotonic()
                result = backend.interpret(self._system_prompt, messages, self._schema)
                self.log(f'[{backend.name}] {time.monotonic() - t0:.2f}s -> {json.dumps(result)}')
                return result, backend.name
            except BackendError as e:
                self.log(f'[{backend.name}] failed: {e}')
        return result_to_dict('infeasible', 'Sorry, my language system is unavailable right now.',
                              confidence=0.0), 'none'

    def _done(self, result: dict, backend: str, keep_dialogue: bool = False) -> dict:
        if not keep_dialogue:
            self.reset()
        return {**result, 'backend': backend}


def _wake(caps: Capabilities) -> str:
    """Regex alternatives for an optional leading wake word ("stretch, stop")."""
    return '|'.join(rf'{re.escape(w)}\W+' for w in caps.wake_words)


def _has_wake_word(text: str, wake_words: list[str]) -> bool:
    return any(re.search(rf'\b{re.escape(w)}\b', text, re.IGNORECASE) for w in wake_words)


_ACK_RE = re.compile(r"^\W*(yes|yeah|yep|yup|no|nope|ok|okay|sure|thanks|thank you|right|correct|great|cool)"
                     r"(\s+(please|thanks|thank you))?\W*$", re.IGNORECASE)


def _mentioned(value: str, said: str, words: list[str]) -> bool:
    """True if `value` (e.g. "living_room") or a close sound-alike of each of its words was said."""
    phrase = value.replace('_', ' ').lower()
    if phrase in said:
        return True
    return all(difflib.get_close_matches(w, words, n=1, cutoff=0.7) or
               difflib.get_close_matches(w.rstrip('s'), words, n=1, cutoff=0.7) for w in phrase.split())


def _describe(task: dict) -> str:
    args = ' '.join(kv['value'].replace('_', ' ') for kv in task.get('args', []))
    return f'{task["name"].replace("_", " ")} {args}'.strip()


def _public(result: dict) -> dict:
    return {k: result[k] for k in ('status', 'task', 'response', 'confidence') if k in result}
