#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import rclpy

from bt_cooking.stretch_task import DEFAULT_TREE, StretchTask, available_trees

def get_workspace_logs_dir():
    prefix = os.environ.get("COLCON_PREFIX_PATH")
    if prefix:
        workspace_root = Path(prefix).parent
        return workspace_root / "logs"
    return Path.home() / "ros2_logs"

def parse_args():
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    parser = argparse.ArgumentParser(description="Run the stretch cooking behaviour tree.")
    parser.add_argument(
        "--log-file",
        default="stretch_cooking",
        help="Name of the log subdirectory (without extension). Default: stretch_cooking",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Full path to the run directory. If set, overrides --log-file timestamping.",
    )
    parser.add_argument(
        "--display-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the behaviour tree while running. Default: enabled.",
    )
    parser.add_argument(
        "--tree",
        default=DEFAULT_TREE,
        help=f"Tree to run: a tree builder in bt_cooking.stretch_bt (see --list-trees). Default: {DEFAULT_TREE}",
    )
    parser.add_argument(
        "--list-trees",
        action="store_true",
        help="List the trees in bt_cooking.stretch_bt and exit.",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="Recipe id to set as recipe/name before the first tick, e.g. greek_salad.",
    )
    parser.add_argument(
        "--location-poses",
        default=None,
        help="JSON file of location poses to navigate to, instead of config/locations/location_poses.json.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=10,
        help="With --loop-until-done, stop after the tree has finished this many times. Default: 10",
    )
    parser.add_argument(
        "--loop-until-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ticking the behaviour tree until the task finishes. Default: enabled.",
    )
    args, _ = parser.parse_known_args(argv)
    return args

def main():
    args = parse_args()

    if args.list_trees:
        for name in sorted(available_trees()):
            print(name)
        return

    if args.location_poses:
        os.environ["STRETCH_LOCATION_POSES"] = str(Path(args.location_poses).resolve())

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        base_logs_dir = get_workspace_logs_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_logs_dir / f"{args.log_file}-{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    
    task = StretchTask(
        log_dir=str(run_dir),
        log_file="episodic_memory",
        tree=args.tree,
        recipe=args.recipe,
        max_passes=args.max_passes,
    )

    task.run(display_tree=args.display_tree, loop_until_done=args.loop_until_done)

if __name__ == "__main__":
    main()