#!/usr/bin/env python3
'''
Runs the 2D Stretch apartment simulation.

    run_sim.py                                  # web viewer on http://localhost:8765/
    run_sim.py --scenario surprises --seed 4 --open
    run_sim.py --host 0.0.0.0                   # viewable from other machines (no authentication)
    run_sim.py --headless --demo Feta --snapshot /tmp/feta.png
    run_sim.py --ros                            # also offer the robot's ROS interface
    run_sim.py --bt main_demo_loop --recipe greek_salad --success-rate 0.9
    run_sim.py --headless --bt main_demo_loop --recipe greek_salad --speed 4 --snapshot /tmp/bt.png
    run_sim.py --export-location-poses /tmp/location_poses.json
    run_sim.py --robot --open                   # run BTs on the real robot, mirrored in the viewer
    run_sim.py --robot --robot-places hello-robot@192.168.68.75:~/stretch_user/maps/embrace_apartment_map.places.yaml

--ros needs a sourced ROS 2 environment; --bt also needs bt_cooking's dependencies
(py_trees, unaceba, nav2_msgs, control_msgs) and this package built with colcon.
ROS discovery is kept to this machine unless --ros-network is given.

--robot runs the behaviour tree against the real Stretch instead (see robot_link.py):
the sim offers none of the robot's interfaces, joins the robot's ROS domain, gives the
BT the robot's location poses from its places.yaml, and mirrors what the BT does on the
virtual map.

Every run writes its event log to <log dir>/stretch_sim-<timestamp>/events.jsonl;
print or filter it with log_view.py.
'''
import argparse
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

from stretch_sim.config import load_scenario, resolve
from stretch_sim.events import EventLog, JsonlSink, console_sink
from stretch_sim.simulator import KinematicSimulator
from stretch_sim.world import load_world


def default_log_root() -> Path:
    prefix = os.environ.get("COLCON_PREFIX_PATH")
    if prefix:
        return Path(prefix.split(os.pathsep)[0]).parent / "logs"
    return Path.home() / "ros2_logs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="default", help="Scenario name in config/sim/scenarios or a path. Default: default")
    parser.add_argument("--world", default=None, help="World name in config/sim/worlds or a path. Default: the scenario's world")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the scenario's seed")
    parser.add_argument("--success-rate", type=float, default=None, metavar="P",
                        help="Probability (0-1) that each robot action succeeds; overrides the scenario, "
                             "including its per-action failure probabilities")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed relative to real time (viewer). Default: 1")
    parser.add_argument("--host", default="127.0.0.1", help="Address the web viewer listens on. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Web viewer port. Default: 8765")
    parser.add_argument("--open", action="store_true", help="Open the web viewer in a browser")
    parser.add_argument("--headless", action="store_true", help="No viewer; use with --demo and/or --snapshot")
    parser.add_argument("--demo", metavar="OBJECT", default=None,
                        help="Fetch OBJECT and hand it to the Human (headless: runs immediately; viewer: the default for Fetch)")
    parser.add_argument("--max-sim-time", type=float, default=600.0, help="Headless time limit in sim seconds. Default: 600")
    parser.add_argument("--snapshot", metavar="PNG", default=None, help="Write a PNG of the final state (headless)")
    parser.add_argument("--log-dir", default=None, help="Run directory for events.jsonl. Default: <workspace>/logs/stretch_sim-<timestamp>")
    parser.add_argument("--no-log-file", action="store_true", help="Do not write events.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true", help="Also print operation steps to the console")
    parser.add_argument("--ros", action="store_true", help="Offer the robot's ROS services, actions and topics")
    parser.add_argument("--ros-network", action="store_true",
                        help="Let ROS discovery reach other machines (default: this machine only, so a real robot is unaffected)")
    parser.add_argument("--bt", metavar="TREE", default=None,
                        help="Run this tree from bt_cooking/stretch_bt.py against the sim (implies --ros)")
    parser.add_argument("--recipe", default=None, help="Recipe for --bt, e.g. greek_salad")
    parser.add_argument("--max-passes", type=int, default=10, help="--bt: stop after the tree has finished this many times")
    parser.add_argument("--bt-command", default=None,
                        help="Command that runs the BT (default: python -m bt_cooking.run_cooking_task)")
    parser.add_argument("--max-wall-time", type=float, default=900.0,
                        help="Headless --bt: give up after this many real seconds. Default: 900")
    parser.add_argument("--robot", action="store_true",
                        help="Run behaviour trees on the real robot and mirror them in the viewer (implies --ros)")
    parser.add_argument("--robot-places", metavar="SRC", default=None,
                        help="--robot: the robot's places.yaml, a local path or HOST:PATH read over ssh. "
                             "Default: stretch:~/stretch_user/maps/embrace_apartment_map.places.yaml")
    parser.add_argument("--robot-place", metavar="LOCATION=PLACE", action="append", default=[],
                        help="--robot: the place a BT location uses, where it is not the lower-case location name "
                             "(Human uses handover). Repeatable, e.g. --robot-place Fridge=kitchen")
    parser.add_argument("--robot-domain-id", type=int, default=None,
                        help="--robot: the robot's ROS_DOMAIN_ID. Default: $ROS_DOMAIN_ID, else 42")
    parser.add_argument("--export-location-poses", metavar="JSON", default=None,
                        help="Write the world's approach poses as location_poses.json for the BT, then exit")
    return parser.parse_args(argv)


def _stop_on_sigterm() -> None:
    '''Let SIGTERM (e.g. from a launch file or `kill`) shut down cleanly, like Ctrl-C.'''
    def handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)


def main(argv=None) -> int:
    args = parse_args(argv)
    _stop_on_sigterm()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    scenario = load_scenario(args.scenario, seed=args.seed)
    world = load_world(resolve(args.world or scenario.get("world", "apartment"), "worlds"))

    if args.export_location_poses:
        path = Path(args.export_location_poses)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(world.location_poses(), indent=4) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
        return 0

    log = EventLog()
    log.add_sink(console_sink())
    jsonl = None
    run_dir = Path(args.log_dir) if args.log_dir else default_log_root() / f"stretch_sim-{datetime.now():%Y%m%d_%H%M%S}"
    if not args.no_log_file:
        jsonl = JsonlSink(run_dir / "events.jsonl")
        log.add_sink(jsonl)
        logging.getLogger("stretch_sim").info(f"Event log: {jsonl.path}")

    sim = KinematicSimulator(world, scenario, log=log)
    if args.success_rate is not None:
        sim.set_success_rate(args.success_rate)
    if args.demo and args.demo not in sim.objects:
        print(f"Unknown object '{args.demo}'. Objects: {sorted(sim.objects)}", file=sys.stderr)
        return 2

    place_names = {}
    for item in args.robot_place:
        location, sep, place = item.partition("=")
        if not sep or location not in world.locations:
            print(f"--robot-place {item}: expected LOCATION=PLACE with LOCATION one of {list(world.locations)}",
                  file=sys.stderr)
            return 2
        place_names[location] = place
    from stretch_sim.ros_link import RosLink
    # The viewer can switch between the sim and the real robot later, whichever starts.
    link = RosLink(sim, run_dir, bt_command=args.bt_command.split() if args.bt_command else None,
                   ros_network=args.ros_network, places=args.robot_places, place_names=place_names,
                   domain_id=args.robot_domain_id)
    try:
        if args.robot:
            link.start_robot()
        elif args.ros or args.bt:
            link.start_sim()
    except RuntimeError as err:
        print(f"Cannot connect to the robot: {err}", file=sys.stderr)
        return 2
    except ImportError as err:
        print(f"Cannot start the ROS interface ({err}). Source a ROS 2 environment and use its Python.",
              file=sys.stderr)
        return 2
    if link.robot is not None:
        print(f"Robot location poses from {link.robot['places']}: {sorted(link.robot['poses'])} -> {link.robot['path']}",
              flush=True)
        if link.robot["missing"]:
            print(f"No place for {', '.join(link.robot['missing'])}: the BT cannot navigate there; "
                  f"map one with --robot-place.", file=sys.stderr, flush=True)

    from stretch_sim.controller import SimController
    controller = SimController(sim, speed=args.speed, scenario_name=Path(args.scenario).stem,
                               demo_object=args.demo, link=link)
    try:
        if args.headless and args.bt:
            return run_headless_bt(controller, args)
        if args.headless:
            run_headless(sim, args)
        else:
            run_web(controller, args)
    except KeyboardInterrupt:
        print("Stopping.", file=sys.stderr)
    finally:
        controller.stop()       # also stops the BT and ROS
        if jsonl:
            jsonl.close()
    return 0


def run_web(controller, args) -> None:
    from stretch_sim.web.server import WebViewer
    try:
        viewer = WebViewer(controller, host=args.host, port=args.port)
    except OSError as err:
        print(f"Cannot serve the viewer on {args.host}:{args.port} ({err.strerror}). "
              f"Is another simulator running? Choose another port with --port.", file=sys.stderr)
        return
    controller.start()
    if args.bt:
        controller.execute({"cmd": "bt_start", "tree": args.bt, "recipe": args.recipe, "max_passes": args.max_passes})
    print(f"Stretch sim viewer: {viewer.url}  (Ctrl-C to stop)", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(viewer.url)
    try:
        viewer.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.shutdown()


def run_headless_bt(controller, args) -> int:
    '''Runs the simulation in real time (times --speed) until the behaviour tree exits.'''
    sim = controller.sim
    controller.start()
    controller.execute({"cmd": "bt_start", "tree": args.bt, "recipe": args.recipe, "max_passes": args.max_passes})
    bt = controller.bt      # controller.stop() shuts ROS down, after which it has none
    code = bt.wait(timeout_s=args.max_wall_time)
    if code is None:
        print(f"The behaviour tree was still running after {args.max_wall_time:.0f} s; stopping it.", file=sys.stderr)
        bt.stop()
    controller.stop()
    delivered = sorted(n for n, o in sim.objects.items() if o.location == "Human")
    print(f"Behaviour tree '{args.bt}' exited with {bt.returncode}; delivered to the Human: {delivered or 'nothing'}")
    if args.snapshot:
        from stretch_sim.render_pil import render_png
        print(f"Wrote {render_png(sim, Path(args.snapshot))}")
    if sim.violations:
        print("Violations:\n  " + "\n  ".join(sim.violations))
    return bt.returncode or 0


def run_headless(sim: KinematicSimulator, args) -> None:
    from stretch_sim.script import ScriptRunner, fetch
    runner = ScriptRunner(sim)
    if args.demo:
        runner.start(f"fetch {args.demo}", fetch(sim, args.demo))
    while runner.active and sim.time < args.max_sim_time:
        sim.step(0.05)
        runner.update()
    if runner.active:
        runner.stop()
    if args.snapshot:
        from stretch_sim.render_pil import render_png
        print(f"Wrote {render_png(sim, Path(args.snapshot))}")
    if sim.violations:
        print("Violations:\n  " + "\n  ".join(sim.violations))


if __name__ == "__main__":
    sys.exit(main())
