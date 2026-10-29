'''
Structured event log for the simulation. Every command the robot is given, every
result it returns, every step of an operation and every rule violation becomes a
SimEvent. Sinks receive events as they happen: the console, a JSONL file per run,
and the viewer's log panel.
'''
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Event kinds, in the order the viewer lists its filters.
COMMAND = "command"      # a request to the robot, with its arguments
RESULT = "result"        # the outcome of a request, with its outputs
STEP = "step"            # progress within an operation, e.g. a grasp phase starting
STATE = "state"          # a change of robot state, e.g. the driver mode
VIOLATION = "violation"  # the robot was asked to do something it should not
INFO = "info"            # everything else: resets, spawning, debug actions
BT = "bt"                # output of a behaviour tree process run against the sim
BELIEF = "belief"        # a change in what the behaviour tree believes, from its published state
KINDS = (COMMAND, RESULT, STEP, STATE, VIOLATION, INFO, BT, BELIEF)

_LEVELS = {VIOLATION: logging.WARNING, STEP: logging.DEBUG}

# Shared by the viewer and the PNG renderer.
EVENT_COLORS = {
    COMMAND: "#1f5fbf",
    RESULT: "#1e7d32",
    STEP: "#777777",
    STATE: "#8e44ad",
    VIOLATION: "#c62828",
    INFO: "#333333",
    BT: "#b26a00",
    BELIEF: "#00838f",
}


@dataclass
class SimEvent:
    seq: int
    time: float                         # simulation time, seconds
    kind: str
    source: str                         # the command or subsystem, e.g. "grasp", "sim"
    message: str
    op_id: Optional[int] = None         # pairs an operation's command, steps and result
    data: Dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        op = f"#{self.op_id} " if self.op_id is not None else ""
        return f"[{self.time:7.2f}s] {self.kind.upper():<9} {op}{self.source}: {self.message}"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


class EventLog:
    '''Keeps the most recent events in memory and forwards each one to the sinks.'''
    def __init__(self, keep: int = 5000):
        self.keep = keep
        self.events: List[SimEvent] = []
        self.sinks: List[Callable[[SimEvent], None]] = []
        self._seq = 0
        self._lock = threading.Lock()

    def add_sink(self, sink: Callable[[SimEvent], None]) -> None:
        self.sinks.append(sink)

    def emit(self, time: float, kind: str, source: str, message: str,
             op_id: Optional[int] = None, **data) -> SimEvent:
        with self._lock:
            self._seq += 1
            event = SimEvent(self._seq, time, kind, source, message, op_id, _jsonable(data))
            self.events.append(event)
            del self.events[:-self.keep]
        for sink in self.sinks:
            sink(event)
        return event

    def since(self, seq: int) -> List[SimEvent]:
        with self._lock:
            return [e for e in self.events if e.seq > seq]

    def clear(self) -> None:
        with self._lock:
            self.events = []


def console_sink(logger: logging.Logger = logging.getLogger("stretch_sim")) -> Callable[[SimEvent], None]:
    def sink(event: SimEvent) -> None:
        logger.log(_LEVELS.get(event.kind, logging.INFO), event.format())
    return sink


class JsonlSink:
    '''Appends one JSON object per event to a file, flushed immediately so a crash loses nothing.'''
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def __call__(self, event: SimEvent) -> None:
        self._file.write(json.dumps(asdict(event)) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_jsonl(path: Path) -> List[SimEvent]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [SimEvent(**json.loads(line)) for line in f if line.strip()]
