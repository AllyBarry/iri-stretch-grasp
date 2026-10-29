'''
Starts the cooking behaviour tree (bt_cooking/run_cooking_task.py) as a separate
process against the simulator's ROS interface, and streams its output into the event
log. The BT is the same program that runs on the robot; it is only told which tree and
recipe to run and where the locations are on the simulated map.

Nothing here imports bt_cooking: the tree names are read from stretch_bt.py's source.
'''
import ast
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

from stretch_sim.config import config_dir
from stretch_sim.events import BT, INFO
from stretch_sim.simulator import KinematicSimulator

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def isolate_ros_discovery() -> None:
    '''
    Keeps the simulator's ROS traffic on this machine, so a real Stretch on the same
    network never sees the simulated /navigate_to_pose (and vice versa). Respects a
    setting the user has made already. Child processes inherit it.
    '''
    if os.environ.get("ROS_DISTRO") == "humble":
        os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
    else:
        os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def _stretch_bt_source() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("bt_cooking")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        path = Path(list(spec.submodule_search_locations)[0]) / "stretch_bt.py"
        if path.is_file():
            return path
    source = Path(__file__).resolve().parents[1] / "bt_cooking" / "stretch_bt.py"
    return source if source.is_file() else None


def discover_trees() -> List[str]:
    '''Top-level functions in bt_cooking/stretch_bt.py that take no arguments.'''
    path = _stretch_bt_source()
    if path is None:
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
            and not node.args.args and not node.args.posonlyargs]


def discover_recipes() -> List[str]:
    recipes = []
    for path in sorted((config_dir() / "recipes").glob("*.json")):
        try:
            recipes.append(json.loads(path.read_text(encoding="utf-8"))["recipe_id"])
        except (ValueError, KeyError):
            continue
    return recipes


class BtProcess:
    '''
    location_poses replaces the sim's approach poses as the poses the BT navigates to,
    e.g. the real robot's in robot mode.
    '''
    def __init__(self, sim: KinematicSimulator, run_dir: Path, command: Optional[List[str]] = None,
                 location_poses: Optional[Dict[str, Dict[str, float]]] = None):
        self.sim = sim
        self.run_dir = Path(run_dir)
        self.command = command or [sys.executable, "-m", "bt_cooking.run_cooking_task"]
        self.location_poses = location_poses
        self.proc: Optional[subprocess.Popen] = None
        self.tree: Optional[str] = None
        self.recipe: Optional[str] = None
        self.returncode: Optional[int] = None
        self.runs = 0
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> Dict:
        return {"running": self.running, "tree": self.tree, "recipe": self.recipe,
                "returncode": None if self.running else self.returncode,
                "pid": self.proc.pid if self.running else None}

    def start(self, tree: str, recipe: Optional[str] = None, max_passes: int = 10) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError(f"a behaviour tree is already running ('{self.tree}')")
            self.runs += 1
            bt_dir = self.run_dir / f"bt-{self.runs:02d}-{tree}"
            bt_dir.mkdir(parents=True, exist_ok=True)
            poses = bt_dir / "location_poses.json"
            poses.write_text(json.dumps(self.location_poses or self.sim.world.location_poses(), indent=2),
                             encoding="utf-8")

            args = self.command + ["--tree", tree, "--run-dir", str(bt_dir), "--location-poses", str(poses),
                                   "--max-passes", str(max_passes), "--no-display-tree"]
            if recipe:
                args += ["--recipe", recipe]
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.tree, self.recipe, self.returncode = tree, recipe, None
            self.sim.bt_state = None     # the new run publishes its own beliefs
            self.sim.emit(INFO, "bt", f"starting tree '{tree}'" + (f" with recipe '{recipe}'" if recipe else ""),
                          command=args, log_dir=str(bt_dir))
            # Run from the BT's log directory: from the repository root, the source folder
            # stretch_floor_perception/ would shadow the installed package and its generated services.
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         env=env, cwd=str(bt_dir), start_new_session=True)
            threading.Thread(target=self._pump, args=(self.proc,), name="bt-output", daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = _ANSI.sub("", line).rstrip()
            if line:
                self.sim.emit(BT, "bt", line)
        proc.stdout.close()
        code = proc.wait()
        if proc is self.proc:
            self.returncode = code
        self.sim.emit(INFO, "bt", f"tree '{self.tree}' exited with code {code}", returncode=code)

    def stop(self, timeout_s: float = 5.0) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.sim.emit(INFO, "bt", f"stopping tree '{self.tree}'")
        try:
            os.killpg(proc.pid, signal.SIGINT)      # like Ctrl-C, so the runner shuts ROS down cleanly
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=timeout_s)
        except ProcessLookupError:
            pass

    def wait(self, timeout_s: Optional[float] = None) -> Optional[int]:
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None
