#!/bin/env bash

SESSION="StretchPnP"

MAP_PATH="/home/hello-robot/stretch_user/maps/lab_B8_map_edited.yaml"
KEEPOUT_MAP_PATH="/home/hello-robot/stretch_user/maps/lab_B8_map_edited_keepout.yaml"


if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session "$SESSION" already exists. Attaching..."
    tmux attach-session -t "$SESSION"
else
    tmux new-session -d -s "$SESSION"

    # Stop gamepad teleop if running
    stretch_free_robot_process.py

    # Home the robot
    stretch_robot_home.py
    stretch_robot_stow.py

    # WINDOW 1: Discovery server 
    tmux new-window -t "$SESSION:1" -n "Discovery"
    tmux send-keys -t "$SESSION:Discovery" "fastdds discovery --server-id 0 --udp-address 192.168.68.54" C-m

    # WINDOW 2: Funmap launch (funmap, stretch-driver, realsense, rviz)
    tmux new-window -t "$SESSION:2" -n "Driver"
    tmux send-keys -t "$SESSION:Driver" "ros2 launch stretch_tools mapping.launch.py" C-m
    #tmux new-window -t "$SESSION:2" -n "Navigation"
    #tmux send-keys -t "$SESSION:Navigation" "ros2 launch stretch_iri_navigation  navigation.launch.py map:="$MAP_PATH" keepout_map:="$KEEPOUT_MAP_PATH" " C-m


    # WINDOW 3: Keyboard Teleop
    tmux new-window -t "$SESSION:3" -n "Teleop"
    tmux send-keys -t "$SESSION:Teleop" "ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True" C-m
    #tmux send-keys -t "$SESSION:Teleop" "ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/stretch/cmd_vel" C-m

    # WINDOW 4: Grasp point service
    tmux new-window -t "$SESSION:4" -n "Grasp"
    tmux send-keys -t "$SESSION:Grasp" "ros2 run stretch_floor_perception grasp_point.py" C-m

    # WINDOW 5: Top-down grasp point service
    tmux new-window -t "$SESSION:5" -n "TopGrasp"
    tmux send-keys -t "$SESSION:TopGrasp" "ros2 run stretch_floor_perception topdown_grasp_point.py" C-m

    # WINDOW 6: Scan Client
    tmux new-window -t "$SESSION:6" -n "Scan"
    tmux send-keys -t "$SESSION:Scan" "ros2 run stretch_floor_perception scan_client.py" C-m

    tmux attach-session -t "$SESSION:1"
fi
