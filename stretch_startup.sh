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

    # WINDOW 1: Discovery server 
    tmux new-window -t "$SESSION:1" -n "Discovery"
    tmux send-keys -t "$SESSION:Discovery" "fastdds discovery --server-id 0 --udp-address 192.168.1.103" C-m

    # WINDOW 2: Funmap launch (funmap, stretch-driver, realsense, rviz)
    tmux new-window -t "$SESSION:2" -n "Driver"
    tmux send-keys -t "$SESSION:Driver" "ros2 launch stretch_tools mapping.launch.py" C-m

    # WINDOW 3: Keyboard Teleop
    tmux new-window -t "$SESSION:3" -n "Teleop"
    tmux send-keys -t "$SESSION:Teleop" "ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True" C-m

    # WINDOW 4: Grasp point service
    tmux new-window -t "$SESSION:4" -n "Grasp"
    tmux send-keys -t "$SESSION:Grasp" "ros2 run stretch_floor_perception grasp_point.py" C-m

    # WINDOW 5: Scan Client
    tmux new-window -t "$SESSION:5" -n "Scan"
    tmux send-keys -t "$SESSION:Scan" "ros2 run stretch_floor_perception scan_client.py" C-m

    tmux attach-session -t "$SESSION:1"
fi

