#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import rclpy

from bt_cooking.stretch_task import StretchTask

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
        "--loop-until-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ticking the behaviour tree until the task finishes. Default: enabled.",
    )
    args, _ = parser.parse_known_args(argv)
    return args

def main():
    args = parse_args()

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
    )

    task.run(display_tree=args.display_tree, loop_until_done=args.loop_until_done)

if __name__ == "__main__":
    main()