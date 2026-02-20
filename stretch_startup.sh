#!/bin/env bash

SESSION="StretchPnP"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session "$SESSION" already exists. Attaching..."
    tmux attach-session -t "$SESSION"
else
    tmux new-session -d -s "$SESSION"

    # WINDOW 1: Funmap launch (funmap, stretch-driver, realsense, rviz)
    tmux rename-window -t "$SESSION:1" "Driver"
    tmux send-keys -t "$SESSION:Driver" "ros2 launch stretch_core stretch_driver.launch.py" C-m

    # WINDOW 2: Keyboard Teleop
    tmux new-window -t "$SESSION:2" -n "Teleop"
    tmux send-keys -t "$SESSION:Teleop" "ros2 run stretch_core keyboard_teleop" C-m

    # WINDOW 3: Grasp point service
    tmux new-window -t "$SESSION:3" -n "Grasp"
    tmux send-keys -t "$SESSION:Grasp" "ros2 run stretch_demos grasp_point" C-m

    # WINDOW 4: Scan Client
    tmux new-window -t "$SESSION:4" -n "Scan"
    tmux send-keys -t "$SESSION:Scan" "ros2 run stretch_floor_perception scan_client.py" C-m

    tmux attach-session -t "$SESSION:1"
fi

