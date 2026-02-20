#!/bin/env bash

SESSION="StretchPnP"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session "$SESSION" already exists. Attaching..."
    tmux attach-session -t "$SESSION"
else
    tmux new-session -d -s "$SESSION"

    # Stop gamepad teleop if running
    stretch_free_robot_process.py

    # Home the robot
    stretch_robot_home.py

    # WINDOW 1: Funmap launch (funmap, stretch-driver, realsense, rviz)
    tmux new-window -t "$SESSION:1" -n "Driver"
    tmux send-keys -t "$SESSION:Driver" "ros2 launch stretch_funmap mapping.launch.py" C-m

    # WINDOW 2: Keyboard Teleop
    tmux new-window -t "$SESSION:2" -n "Teleop"
    tmux send-keys -t "$SESSION:Teleop" "ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True" C-m

    # WINDOW 3: Grasp point service
    tmux new-window -t "$SESSION:3" -n "Grasp"
    tmux send-keys -t "$SESSION:Grasp" "ros2 run stretch_demos grasp_point" C-m

    # WINDOW 4: Scan Client
    tmux new-window -t "$SESSION:4" -n "Scan"
    tmux send-keys -t "$SESSION:Scan" "ros2 run stretch_floor_perception scan_client.py" C-m

    tmux attach-session -t "$SESSION:1"
fi

