# Source this in every terminal:  source ~/iri-stretch-grasp/stretch_speech_ws/env.sh
# (conda's python/libstdc++ break rclpy, so conda is deactivated first)
command -v conda >/dev/null && conda deactivate 2>/dev/null; command -v conda >/dev/null && conda deactivate 2>/dev/null
_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
source "$_WS/.venv/bin/activate"
[ -f "$_WS/install/setup.bash" ] && source "$_WS/install/setup.bash"
# ROS installs node scripts with a #!/usr/bin/python3 shebang, so expose the venv packages explicitly.
export PYTHONPATH="$_WS/.venv/lib/python3.12/site-packages:$PYTHONPATH"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}   # must match the robot when running split
