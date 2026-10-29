"""Robot capability manifest: loading, LLM output schema, prompt text and task validation.

Pure Python (no ROS imports) so it can be unit-tested and reused by the CLI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

VERSION = '2026-10-29.4'   # logged at startup and printed by the test runner

STATUSES = ('execute', 'clarify', 'infeasible', 'ignore')


@dataclass
class Param:
    name: str
    type: str = 'string'
    required: bool = False
    ref: str | None = None          # name of a vocabulary list the value comes from
    strict: bool = True             # False: the vocabulary list is only a hint, other values are accepted
    enum: list[str] | None = None
    min: float | None = None
    max: float | None = None
    ask: str | None = None          # clarification question when missing
    unit: str | None = None
    grounded: bool = True           # False: the value need not have been said literally ("the Italian one" -> caprese_salad)
    unknown: str = 'clarify'        # value not in the list: 'clarify' (ask), 'drop' (leave it out), 'drop_all' (task, no params)
    fill: str | None = None         # 'input': always the person's own words (English), whatever the LLM wrote
    single: bool = False            # exactly one thing: "the fork and the knife" / "jam or honey" / "something" -> left out


@dataclass
class Action:
    name: str
    description: str
    params: dict[str, Param] = field(default_factory=dict)


@dataclass
class Capabilities:
    robot_name: str
    robot_description: str
    limitations: list[str]
    wake_words: list[str]
    vocabulary: dict[str, list[str]]
    stop_phrases: list[str]
    actions: dict[str, Action]
    asr_phrases: list[str] = field(default_factory=list)
    uncertain_word_prob: float = 0.5
    recipe_types: dict[str, str] = field(default_factory=dict)   # recipe_id -> type, from the recipe files
    recipe_keywords: dict[str, set] = field(default_factory=dict)   # recipe_id -> words of its name + ingredients

    def load_recipes(self, folder: str) -> list[str]:
        """Take the recipe list from the repo's config/recipes/*.json instead of the yaml:
        recipe ids become the `recipes` vocabulary, their `type` fields the recipe_type choices,
        and set_recipe's description lists them so the LLM can match "the ham and cheese one"."""
        import glob, json, os
        recipes = []
        for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
            try:
                with open(path, encoding='utf-8') as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue
            rid = str(r.get('recipe_id') or os.path.splitext(os.path.basename(path))[0])
            recipes.append((rid, str(r.get('name') or rid.replace('_', ' ')), str(r.get('type') or '').lower()))
            words = set(re.findall(r"[a-z]+", f"{rid.replace('_', ' ')} {r.get('name', '')} ".lower() +
                                   ' '.join(str(i) for i in r.get('ingredients') or []).lower()))
            self.recipe_keywords[rid] = words - {'salad', 'sandwich', 'and', 'with', 'the', 'of', 'a'}
        if not recipes:
            return []
        self.vocabulary['recipes'] = [rid for rid, _, _ in recipes]
        self.recipe_types = {rid: t for rid, _, t in recipes if t}
        action = self.actions.get('set_recipe')
        if action is not None:
            if 'recipe_type' in action.params and self.recipe_types:
                action.params['recipe_type'].enum = sorted(set(self.recipe_types.values()))
            listing = '; '.join(f'{rid} = {name}' + (f' ({t})' if t else '') for rid, name, t in recipes)
            action.description = action.description.split(' Recipes: ')[0] + f' Recipes: {listing}.'
        return self.vocabulary['recipes']

    @classmethod
    def from_yaml(cls, path: str) -> 'Capabilities':
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, d: dict) -> 'Capabilities':
        robot = d.get('robot', {})
        actions = {}
        for name, a in (d.get('actions') or {}).items():
            params = {pn: Param(name=pn, **(pv or {})) for pn, pv in (a.get('params') or {}).items()}
            actions[name] = Action(name=name, description=a.get('description', ''), params=params)
        return cls(
            robot_name=robot.get('name', 'Robot'),
            robot_description=' '.join(str(robot.get('description', '')).split()),
            limitations=list(robot.get('limitations') or []),
            wake_words=[w.lower() for w in (robot.get('wake_words') or [])],
            vocabulary={k: [str(v) for v in vs] for k, vs in (d.get('vocabulary') or {}).items()},
            stop_phrases=[p.lower() for p in (d.get('stop_phrases') or [])],
            actions=actions,
            asr_phrases=list((d.get('asr') or {}).get('phrases') or []),
            uncertain_word_prob=float((d.get('asr') or {}).get('uncertain_word_prob', 0.5)),
        )

    def allowed_values(self, p: Param) -> list[str] | None:
        if p.enum:
            return p.enum
        if p.ref:
            return self.vocabulary.get(p.ref, [])
        return None

    def asr_prompt(self) -> str:
        """Whisper initial prompt: short command sentences using the verbs and vocabulary.

        Whisper is biased towards words (and the style) of its prompt, so natural sentences such as
        "Fetch the apple." work much better than a bare word list, and verbs matter as much as nouns
        ("fetch" was being heard as "fitch" when only nouns were listed).
        """
        pools = {name.rstrip('s'): [w.replace('_', ' ') for w in ws] for name, ws in self.vocabulary.items() if ws}
        counters = dict.fromkeys(pools, 0)
        used: set[str] = set()

        def fill(m: re.Match) -> str:
            key = m.group(1)
            if key not in pools:
                return m.group(0)
            word = pools[key][counters[key] % len(pools[key])]
            counters[key] += 1
            used.add(word)
            return word

        sentences = [re.sub(r'\{(\w+)\}', fill, p) for p in self.asr_phrases]
        leftovers = [w for ws in pools.values() for w in ws if w not in used]
        if leftovers:
            sentences.append(', '.join(leftovers).capitalize() + '.')
        sentences.append('Stop.')
        return f'{self.robot_name}, ' + ' '.join(sentences)

    # ------------------------------------------------------------------ LLM I/O
    def output_schema(self) -> dict:
        """JSON schema the LLM output is constrained to (Ollama `format` / Claude structured outputs)."""
        return {
            'type': 'object',
            'properties': {
                'status': {'type': 'string', 'enum': list(STATUSES)},
                'task': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string', 'enum': [*self.actions, NO_TASK]},
                        'args': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {'key': {'type': 'string'}, 'value': {'type': 'string'}},
                                'required': ['key', 'value'],
                                'additionalProperties': False,
                            },
                        },
                    },
                    'required': ['name', 'args'],
                    'additionalProperties': False,
                },
                'response': {'type': 'string', 'maxLength': 120},
                'confidence': {'type': 'number'},
            },
            'required': ['status', 'task', 'response', 'confidence'],
            'additionalProperties': False,
        }

    def system_prompt(self, follow_ups: bool = True) -> str:
        """follow_ups=False: the robot never asks back, so an unclear detail is left out instead of asked about."""
        lines = [
            f'You are the command interpreter for {self.robot_name}, an assistive robot. '
            'You convert a transcribed spoken request into ONE high-level task for the robot, '
            'or decide that the robot must ask for clarification or decline. A separate planner breaks the '
            'task down into motions, so never output a sequence of steps: choose the single task that '
            'captures what the user wants.',
            '',
            f'Robot: {self.robot_description}',
            '',
            'Limitations:',
            *[f'- {lim}' for lim in self.limitations],
            '',
            'Available tasks (the ONLY task names you may use):',
        ]
        for a in self.actions.values():
            params = []
            for p in a.params.values():
                desc = f'{p.name}{"" if p.required else "?"}: {p.type}'
                allowed = self.allowed_values(p)
                if allowed:
                    desc += (f' one of [{", ".join(allowed)}]' if p.strict or p.enum
                             else f' e.g. [{", ".join(allowed)}] or any other {(p.ref or "value").rstrip("s")} name')
                if p.min is not None or p.max is not None:
                    desc += f' in [{p.min}, {p.max}]'
                if p.unit:
                    desc += f' ({p.unit})'
                params.append(desc)
            lines.append(f'- {a.name}({"; ".join(params)}): {a.description}')
        lines += [
            '',
            'Decide one status:',
            f'- "execute": the request is clear and achievable. Give the one task (e.g. {self._example()}). '
            '"response" is a short spoken acknowledgement that does not ask anything (questions are only for '
            '"clarify").',
            ('- "clarify": the request is ambiguous, incomplete, garbled, or refers to a location that is not in '
             f'the list but might be a synonym. task.name = "{NO_TASK}"; "response" is ONE short question to '
             'the user (offer options when helpful).' if follow_ups else
             '- "clarify": do not use it: the robot cannot ask questions back. When the request is one of the '
             'tasks but a detail is unclear, has several options or is not in the lists, use "execute" with that '
             'task and leave that parameter out (the robot works it out). When it is not one of the tasks, it '
             'is chat (or infeasible if it asks the robot to do something it cannot).'),
            '- "infeasible": the request is understood but none of the tasks can achieve it, or it violates a '
            f'limitation. task.name = "{NO_TASK}"; "response" briefly says why and, if possible, what you '
            'can do instead.',
            ('- "ignore": only for noise, filler or people clearly talking to each other, not to the robot. '
             'Anything else said to the robot that is not another task is the "chat" task. '
             if 'chat' in self.actions else
             '- "ignore": the utterance is not a request to the robot (chit-chat between people, filler, noise). ')
            + f'task.name = "{NO_TASK}" and response empty.',
            '',
            'Rules:',
            '- Arguments are key/value strings. Use the exact parameter names. Leave out optional arguments '
            'the user did not mention; never guess them (e.g. do not add a location for an object unless said).',
            '- Values with "one of [...]" must be exactly one of the listed values (map obvious synonyms). '
            'Values marked "or any other ... name" may be anything the user names (e.g. "keys"), written in '
            'lowercase.',
            ('- Never invent tasks or locations. If the request contains several separate tasks, use "clarify" '
             'and ask which one to do first (unless a task description below says how to handle it).' if follow_ups
             else '- Never invent tasks or locations. If one request names several things or options for one '
             'task, it is still that one task, with the parameter left out.'),
            '- When a question offers choices, only offer values from the lists above (written naturally), '
            'never made-up places.',
            '- If the conversation contains an earlier clarification question, combine the original request '
            "with the user's answer into one complete interpretation (a short \"yes\" confirms your suggestion).",
            '- confidence is your probability (0 to 1) that the interpretation is what the user meant.',
            '- "response" is at most one short sentence (under 12 words). Never answer the question or tell '
            'the joke there: the chat task passes the person\'s words on to the part of the robot that talks.',
            '- Transcripts come from speech recognition and may contain misheard words. If a word sounds like '
            'a listed object, location or a common command verb (e.g. "fitch" -> fetch, "mustered" -> mustard, '
            '"ring me" -> bring me), use the sound-alike when it gives exactly one sensible request; ask only '
            'if several readings remain plausible. Words marked as uncertain are the most likely mis-hearings.',
        ]
        return '\n'.join(lines)

    def _example(self) -> str:
        """Worked example for the prompt, built from the real tasks and vocabulary so it never shows invalid values."""
        obj = (self.vocabulary.get('objects') or ['cup'])[0]
        examples = []
        if 'fetch' in self.actions and 'object' in self.actions['fetch'].params:
            examples.append(f'"bring me the {obj}" -> fetch(object={obj})')
        if 'where' in self.actions and 'object' in self.actions['where'].params:
            examples.append('"where are my keys?" -> where(object=keys)')
        if not examples:
            return f'one task such as {next(iter(self.actions))}(...)'
        return '; '.join(examples)

    # --------------------------------------------------------------- validation
    def validate(self, result: dict) -> dict:
        """Enforce the manifest on an interpreter result. Returns a normalised (possibly downgraded) copy.

        The LLM is constrained by the schema, but values, required args and ranges are checked here so
        that a hallucinated task can never reach the executor. `validation_note` explains any downgrade.
        """
        status = result.get('status')
        if status not in STATUSES:
            return _downgrade('clarify', 'Sorry, I did not understand. Could you rephrase that?', result,
                              f'invalid status {status!r}')
        if status != 'execute':
            return {**result, 'task': _no_task()}

        task = result.get('task') or {}
        name = task.get('name')
        if name in (None, '', NO_TASK):
            return _downgrade('clarify', 'What would you like me to do?', result, 'execute without a task')
        action = self.actions.get(name)
        if action is None:
            return _downgrade('infeasible', f"I don't know how to {str(name).replace('_', ' ')}.", result,
                              f'unknown task {name!r}')

        args = {str(kv.get('key')): str(kv.get('value', '')).strip() for kv in task.get('args') or []}
        clean = {}
        for p in action.params.values():
            value = args.get(p.name, '')
            if value.lower() in UNSPECIFIED:  # "pass it over" / "give it to me" name nothing
                value = ''
            if not value:
                if p.required:
                    return _downgrade('clarify', p.ask or f'What {p.name} should I use?', result,
                                      f'missing required argument {name}.{p.name}')
                continue
            allowed = self.allowed_values(p)
            if allowed is not None:
                norm = value.lower().replace(' ', '_')
                match = next((a for a in allowed if a.lower() == norm), None)
                if match is not None:
                    value = match
                elif (p.strict or p.enum) and p.unknown == 'drop':
                    continue
                elif (p.strict or p.enum) and p.unknown == 'drop_all':   # e.g. a dish we have no recipe for
                    clean = {}
                    break
                elif p.strict or p.enum:
                    opts = ', '.join(a.replace('_', ' ') for a in allowed)
                    return _downgrade('clarify', f"I don't know '{value}'. Did you mean one of: {opts}?", result,
                                      f'{name}.{p.name}={value!r} not in {p.ref or "enum"}')
                else:
                    value = value.lower()
            if p.type == 'number':
                try:
                    num = float(value)
                except ValueError:
                    return _downgrade('clarify', f'How far should I {name.replace("_", " ")}?', result,
                                      f'{name}.{p.name}={value!r} is not a number')
                if (p.min is not None and num < p.min) or (p.max is not None and num > p.max):
                    return _downgrade(
                        'infeasible',
                        f'I can only {name.replace("_", " ")} between {p.min} and {p.max} {p.unit or ""}'.strip() + '.',
                        result, f'{name}.{p.name}={num} out of range')
                value = f'{num:g}'
            clean[p.name] = value
        dropped = sorted(k for k in args if k not in action.params)
        note = {'validation_note': f'dropped unknown args {dropped}'} if dropped else {}

        try:
            confidence = float(result.get('confidence', 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {**result, **note, 'task': {'name': name, 'args': [{'key': k, 'value': v} for k, v in clean.items()]},
                'confidence': max(0.0, min(1.0, confidence))}


NO_TASK = 'none'
# Argument values that refer to nothing concrete; treated as "not given" (required -> ask, optional -> dropped).
UNSPECIFIED = {'it', 'this', 'that', 'them', 'these', 'those', 'one', 'something', 'anything',
               'me', 'myself', 'us', 'user', 'the user', 'speaker', 'none', 'unknown'}


def _no_task() -> dict:
    return {'name': '', 'args': []}


def _downgrade(status: str, response: str, original: dict, reason: str) -> dict:
    return {**original, 'status': status, 'task': _no_task(), 'response': response,
            'validation_note': f'{original.get("status")} -> {status}: {reason}'}


def result_to_dict(status: str, response: str = '', task: str = '', args: dict | None = None,
                   confidence: float = 1.0) -> dict[str, Any]:
    return {'status': status, 'response': response, 'confidence': confidence,
            'task': {'name': task, 'args': [{'key': k, 'value': v} for k, v in (args or {}).items()]}}
