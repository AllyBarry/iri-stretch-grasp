"""Try the interpreter without ROS:  python -m speech_task.cli [--backend rules|ollama|anthropic] ["text" ...]

With no text arguments it runs an interactive prompt (clarification context is kept between lines).
"""
import argparse
import json
import os

from .backends import BackendError, make_backend
from .capabilities import Capabilities
from .interpreter import Interpreter

DEFAULT_CAPS = os.path.join(os.path.dirname(__file__), '..', 'config', 'capabilities.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text', nargs='*')
    ap.add_argument('--backend', default='ollama')
    ap.add_argument('--fallback', default='rules')
    ap.add_argument('--model', default='')
    ap.add_argument('--host', default='http://localhost:11434')
    ap.add_argument('--caps', default=DEFAULT_CAPS)
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()

    caps = Capabilities.from_yaml(a.caps)
    backend = make_backend(a.backend, caps, model=a.model, host=a.host)
    fallback = make_backend(a.fallback, caps) if a.fallback else None
    if hasattr(backend, 'warmup'):
        try:
            backend.warmup()  # load the model before the first request
        except BackendError as e:
            print(f'warning: {e}')
    interp = Interpreter(caps, backend, fallback, log=print if a.verbose else (lambda *_: None))

    def run(text):
        r = interp.interpret(text)
        print(json.dumps({k: r[k] for k in ('status', 'task', 'response', 'confidence', 'backend', 'validation_note')
                          if k in r}, indent=2))

    if a.text:
        for t in a.text:
            print(f'> {t}')
            run(t)
        return
    try:
        while True:
            run(input('> '))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == '__main__':
    main()
