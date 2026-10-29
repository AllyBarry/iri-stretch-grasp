from .base import Backend, BackendError


def make_backend(name: str, caps, **kw) -> Backend:
    """Factory. kw: model, host, effort, timeout (only those relevant to the backend are used)."""
    if name == 'ollama':
        from .ollama_backend import OllamaBackend
        return OllamaBackend(**{k: v for k, v in kw.items() if k in ('model', 'host', 'timeout') and v})
    if name == 'anthropic':
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(**{k: v for k, v in kw.items() if k in ('model', 'effort', 'timeout') and v})
    if name == 'rules':
        from .rule_backend import RuleBackend
        return RuleBackend(caps)
    raise ValueError(f'unknown backend {name!r} (expected ollama | anthropic | rules)')


__all__ = ['Backend', 'BackendError', 'make_backend']
