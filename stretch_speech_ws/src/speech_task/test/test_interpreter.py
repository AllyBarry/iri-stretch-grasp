import os

import pytest

from speech_task.backends import Backend, BackendError
from speech_task.backends.rule_backend import RuleBackend
from speech_task.capabilities import Capabilities
from speech_task.interpreter import Interpreter

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'capabilities.yaml')
CAPS = Capabilities.from_yaml(FIXTURE)


class Scripted(Backend):
    """Returns canned results in order and records the messages it was given."""
    name = 'scripted'

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def interpret(self, system_prompt, messages, schema):
        self.calls.append([dict(m) for m in messages])
        return self.results.pop(0)


class Down(Backend):
    name = 'down'

    def interpret(self, *a):
        raise BackendError('offline')


def task(name, conf=0.9, **args):
    return {'status': 'execute', 'task': {'name': name, 'args': [{'key': k, 'value': v} for k, v in args.items()]},
            'response': 'ok', 'confidence': conf}


def ask(q='?'):
    return {'status': 'clarify', 'task': {'name': 'none', 'args': []}, 'response': q, 'confidence': 0.5}


def args_of(r):
    return {a['key']: a['value'] for a in r['task']['args']}


def make(*results, **kw):
    b = Scripted(*results)
    return Interpreter(CAPS, b, log=lambda *_: None, **kw), b


def test_single_task_passes():
    it, _ = make(task('fetch', object='cup', **{'from': 'kitchen'}))
    r = it.interpret('get the cup from the kitchen')
    assert r['status'] == 'execute' and r['task']['name'] == 'fetch'
    assert args_of(r) == {'object': 'cup', 'from': 'kitchen'}


def test_location_normalised_to_vocabulary():
    it, _ = make(task('go_to', location='Living Room'))
    assert args_of(it.interpret('go to the living room')) == {'location': 'living_room'}


def test_unlisted_object_is_accepted():
    it, _ = make(task('find', object='Car Keys'))
    r = it.interpret('find my car keys')
    assert r['status'] == 'execute' and args_of(r) == {'object': 'car keys'}


def test_unknown_location_asks_with_options():
    it, _ = make(task('go_to', location='garage'))
    r = it.interpret('go to the garage')
    assert r['status'] == 'clarify' and 'kitchen' in r['response']
    assert "not in locations" in r['validation_note']


def test_unknown_task_is_infeasible():
    it, _ = make(task('open_door'))
    assert it.interpret('open the door')['status'] == 'infeasible'


def test_missing_required_arg_asks():
    it, _ = make(task('go_to'))
    r = it.interpret('go there')
    assert r['status'] == 'clarify' and r['response'] == 'Where should I go?'


def test_out_of_range_number_is_infeasible():
    it, _ = make(task('move', direction='forward', distance='10'))
    assert it.interpret('drive forward ten meters')['status'] == 'infeasible'


def test_unknown_args_are_dropped():
    it, _ = make(task('find', object='cup', colour='red'))
    r = it.interpret('find the red cup')
    assert args_of(r) == {'object': 'cup'} and 'colour' in r['validation_note']


def test_non_execute_has_no_task():
    it, _ = make({'status': 'infeasible', 'task': {'name': 'stop', 'args': []}, 'response': 'no', 'confidence': 1})
    r = it.interpret('go upstairs')
    assert r['status'] == 'infeasible' and r['task']['name'] == ''


def test_low_confidence_becomes_confirmation():
    it, _ = make(task('go_to', conf=0.2, location='kitchen'))
    r = it.interpret('kitchen maybe')
    assert r['status'] == 'clarify' and r['response'] == 'Just to check: should I go to kitchen?'


@pytest.mark.parametrize('text', ['stop', 'Stop!', 'please stop now', 'STOP MOVING', 'halt'])
def test_stop_fast_path_skips_llm(text):
    it, b = make()
    r = it.interpret(text)
    assert r['backend'] == 'fast_path' and r['task'] == {'name': 'stop', 'args': []}
    assert b.calls == []


def test_stop_inside_sentence_goes_to_llm():
    it, b = make(task('go_to', location='kitchen'))
    it.interpret("don't stop until you reach the kitchen")
    assert len(b.calls) == 1


def test_low_asr_confidence_asks_to_repeat():
    it, b = make()
    r = it.interpret('mumble', asr_confidence=0.1)
    assert r['status'] == 'clarify' and b.calls == []


def test_clarification_context_is_carried():
    it, b = make(ask('Which object?'), task('pick_up', object='cup'))
    assert it.interpret('pick it up')['status'] == 'clarify'
    assert it.awaiting_answer
    assert it.interpret('the cup')['status'] == 'execute'
    assert [m['role'] for m in b.calls[1]] == ['user', 'assistant', 'user']
    assert not it.awaiting_answer


def test_confirmation_yes_after_clarify_executes():
    """Regression: 'find my keys' -> 'did you mean the desk?' -> 'yes' must execute, not give up."""
    it, _ = make(ask("Did you mean the desk?"), task('find', conf=1.0, object='keys', location='desk'))
    assert it.interpret('find my keys on the day')['status'] == 'clarify'
    r = it.interpret('yes')
    assert r['status'] == 'execute' and args_of(r) == {'object': 'keys', 'location': 'desk'}


def test_too_many_clarifications_starts_over_without_saying_infeasible():
    it, _ = make(ask(), ask(), ask(), ask())
    results = [it.interpret(t) for t in ('a', 'b', 'c', 'd')]
    assert [r['status'] for r in results] == ['clarify'] * 4
    assert 'start over' in results[-1]['response']
    assert not it.awaiting_answer


def test_fallback_used_when_backend_down():
    it = Interpreter(CAPS, Down(), RuleBackend(CAPS), log=lambda *_: None)
    r = it.interpret('bring me the cup from the kitchen')
    assert r['backend'] == 'rules' and r['task']['name'] == 'fetch'


def test_no_backend_available_is_infeasible():
    it = Interpreter(CAPS, Down(), None, log=lambda *_: None)
    assert it.interpret('bring me a cup')['status'] == 'infeasible'


def test_wake_word_required():
    caps = Capabilities.from_yaml(FIXTURE)
    caps.wake_words = ['stretch']
    it = Interpreter(caps, Scripted(task('stow')), log=lambda *_: None)
    assert it.interpret('tuck your arm in')['status'] == 'ignore'
    assert it.interpret('stretch, tuck your arm in')['status'] == 'execute'


def test_rule_backend_examples():
    it = Interpreter(CAPS, RuleBackend(CAPS), log=lambda *_: None)
    r = it.interpret('bring me the remote from the living room')
    assert r['task']['name'] == 'fetch' and args_of(r) == {'object': 'remote', 'from': 'living_room'}
    r = it.interpret('where are my sunglasses')
    assert r['task']['name'] == 'find' and args_of(r) == {'object': 'sunglasses'}
    assert it.interpret('can you cook dinner')['status'] == 'infeasible'
    assert it.interpret('go somewhere')['status'] == 'clarify'
    assert it.interpret('the kitchen')['status'] == 'execute'   # completes the clarification


def test_schema_constrains_task_names():
    enum = CAPS.output_schema()['properties']['task']['properties']['name']['enum']
    assert set(enum) == set(CAPS.actions) | {'none'}


def test_uncertain_words_reach_the_llm():
    it, b = make(task('fetch', object='cup'))
    it.interpret('fitch the cup', uncertain_words=['fitch'])
    assert "possibly misheard: 'fitch'" in b.calls[0][-1]['content']


def test_asr_prompt_uses_phrases_and_all_vocabulary():
    prompt = CAPS.asr_prompt()
    assert prompt.startswith('Stretch, Fetch the cup.')
    for w in CAPS.vocabulary['locations'] + CAPS.vocabulary['objects']:
        assert w.replace('_', ' ') in prompt.lower()
    assert '{' not in prompt


def test_unmentioned_optional_location_is_dropped():
    it, _ = make(task('fetch', object='mustard', **{'from': 'desk'}))
    r = it.interpret('fitch the mustered', uncertain_words=['fitch', 'mustered'])
    assert r['status'] == 'execute' and args_of(r) == {'object': 'mustard'}
    assert "dropped unmentioned from='desk'" in r['validation_note']
    assert r['response'] == "Okay, I'll fetch mustard."


def test_unmentioned_required_location_needs_confirmation():
    it, _ = make(task('go_to', location='kitchen'))
    r = it.interpret('go over there')
    assert r['status'] == 'clarify' and r['response'] == 'Just to check: should I go to kitchen?'


def test_location_from_robot_question_counts_as_mentioned():
    it, _ = make(ask('Did you mean the desk?'), task('go_to', location='desk'))
    it.interpret('go to the dusk')
    assert it.interpret('yes')['status'] == 'execute'


@pytest.mark.parametrize('text', ['yes', 'Yes.', 'okay', 'thank you', 'no thanks'])
def test_bare_acknowledgement_without_question_is_ignored(text):
    it, b = make()
    assert it.interpret(text)['status'] == 'ignore' and b.calls == []


def test_pass_with_recipient():
    it, _ = make(task('pass', object='Salt', to='Anna'))
    r = it.interpret('could you pass the salt to Anna')
    assert r['task']['name'] == 'pass' and args_of(r) == {'object': 'salt', 'to': 'Anna'}


@pytest.mark.parametrize('obj', ['it', 'that', 'something'])
def test_pronoun_object_asks(obj):
    it, _ = make(task('pass', object=obj))
    r = it.interpret('pass it over')
    assert r['status'] == 'clarify' and r['response'] == 'What should I pass?'


def test_recipient_me_means_the_speaker():
    it, _ = make(task('pass', object='cup', to='me'))
    assert args_of(it.interpret('pass me the cup')) == {'object': 'cup'}


def test_rule_backend_pass():
    it = Interpreter(CAPS, RuleBackend(CAPS), log=lambda *_: None)
    r = it.interpret('pass the apple to anna')
    assert r['task']['name'] == 'pass' and args_of(r) == {'object': 'apple', 'to': 'anna'}


def test_set_recipe_named():
    it, _ = make(task('set_recipe', recipe='Hummus'))
    r = it.interpret("let's make hummus")
    assert r['status'] == 'execute' and args_of(r) == {'recipe': 'hummus'}


def test_set_recipe_underspecified_still_executes():
    it, _ = make(task('set_recipe'))
    r = it.interpret("let's cook something")
    assert r['status'] == 'execute' and r['task'] == {'name': 'set_recipe', 'args': []}


def test_set_recipe_hint_without_name():
    it, _ = make(task('set_recipe', hint='something with chickpeas', recipe='unknown'))
    r = it.interpret('make something with chickpeas')
    assert r['status'] == 'execute' and args_of(r) == {'hint': 'something with chickpeas'}


def test_unlisted_recipe_accepted():
    it, _ = make(task('set_recipe', recipe='shakshuka'))
    assert args_of(it.interpret("we're making shakshuka")) == {'recipe': 'shakshuka'}


def test_recipe_guessed_from_hint_is_dropped():
    it, _ = make(task('set_recipe', recipe='hummus', hint='something with chickpeas'))
    r = it.interpret('make something with chickpeas')
    assert args_of(r) == {'hint': 'something with chickpeas'}


def test_execute_reply_never_asks_a_question():
    it, _ = make({**task('set_recipe', recipe='hummus'), 'response': 'Great! Any steps in mind?'})
    r = it.interpret("let's make hummus")
    assert r['status'] == 'execute' and r['response'] == "Okay, I'll set recipe hummus."
