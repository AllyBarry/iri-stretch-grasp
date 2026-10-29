'''
Scripted command sequences for exercising the simulator from the viewer or headless,
without ROS or the behaviour tree. A script is a generator that yields operations; the
runner resumes it once the yielded operation is done.

These scripts use ground truth (they go straight to where an object really is). They
check the simulator itself, not the robot's decision making.
'''
from typing import Generator, Optional

from stretch_sim.events import INFO
from stretch_sim.operations import Operation, OpStatus
from stretch_sim.simulator import KinematicSimulator
from stretch_sim.world import HELD

Script = Generator[Operation, None, None]


class ScriptRunner:
    def __init__(self, sim: KinematicSimulator):
        self.sim = sim
        self.script: Optional[Script] = None
        self.current: Optional[Operation] = None
        self.name = ""

    @property
    def active(self) -> bool:
        return self.script is not None

    def start(self, name: str, script: Script) -> None:
        self.stop()
        self.name, self.script, self.current = name, script, None
        self.sim.emit(INFO, "script", f"started '{name}'")
        self._advance()

    def stop(self) -> None:
        if self.script is not None:
            if self.current is not None and not self.current.done:
                self.sim.cancel(self.current)
            self.script.close()
            self.sim.emit(INFO, "script", f"stopped '{self.name}'")
            self.script = None

    def update(self) -> None:
        '''Call once per simulation step.'''
        if self.script is not None and (self.current is None or self.current.done):
            self._advance()

    def _advance(self) -> None:
        try:
            self.current = next(self.script)
        except StopIteration:
            self.sim.emit(INFO, "script", f"finished '{self.name}'")
            self.script = None


def fetch(sim: KinematicSimulator, obj: str, person: str = "Human") -> Script:
    '''Drive to where obj really is, scan, grasp it, drive to the person and hand it over.'''
    location = sim.location_of(obj)
    if location == HELD or sim.world.locations[location].kind != "surface":
        sim.emit(INFO, "script", f"'{obj}' is not on a surface ({location})")
        return

    sim.set_driver_mode("navigation")
    nav = sim.navigate_to_location(location)
    yield nav
    if nav.status != OpStatus.SUCCEEDED:
        sim.emit(INFO, "script", f"could not reach the {location}")
        return
    sim.set_driver_mode("position")

    target = sim.objects[obj].label
    matches = [d for d in sim.detect() if d.label == target]
    if not matches:
        sim.emit(INFO, "script", f"'{obj}' not detected from the {location} approach pose")
        return
    best = max(matches, key=lambda d: d.score)
    grasp = sim.grasp(best.centroid, direction=best.grasp_direction, expected_label=best.label)
    yield grasp
    if not grasp.result.get("held"):
        sim.emit(INFO, "script", f"grasp of '{obj}' did not end with it in the gripper")
        return

    sim.set_driver_mode("navigation")
    yield sim.navigate_to_location(person)
    sim.set_driver_mode("position")
    yield sim.handover()


def tour(sim: KinematicSimulator) -> Script:
    '''Visit every location and scan it.'''
    for name, loc in sim.world.locations.items():
        if loc.kind != "surface":
            continue
        sim.set_driver_mode("navigation")
        yield sim.navigate_to_location(name)
        sim.detect()
