# stretch_sim

A 2D top-down simulation of the Stretch in an apartment, for testing the cooking
behaviour tree without the robot, with a web viewer. Objects spawn on the Fridge, Pantry
and Table according to `config/locations/location_priors.json` (or wherever a scenario
pins them); the person works at the Working Surface, where the robot hands objects over,
the robot drives between locations, scans, grasps and hands objects to the Human, and
every command, result and step is logged.

It does not import `bt_cooking`. The ROS interface (next step) will offer the same
services, actions and topics as the real robot, so the BT cannot tell the two apart.

## Running

From the repository root (no build needed):

```bash
python3 -m stretch_sim.run_sim                              # web viewer on http://localhost:8765/
python3 -m stretch_sim.run_sim --open --speed 4             # ...and open it in a browser
python3 -m stretch_sim.run_sim --scenario surprises --seed 4
python3 -m stretch_sim.run_sim --host 0.0.0.0               # view from another machine (no authentication)
python3 -m stretch_sim.run_sim --headless --demo Butter --snapshot /tmp/butter.png
python3 -m stretch_sim.run_sim --export-location-poses config/locations/location_poses.json
python3 -m unittest test.test_stretch_sim -v
```

After `colcon build`: `ros2 run stretch_floor_perception run_sim.py [...]`.

### Web viewer

The viewer is a single page served by the simulator (standard library only, no
internet access needed): the map, a robot panel (driver mode, held object, base and head
pose, lift, arm and gripper), the running operations with their progress, controls, the
ground-truth inventory, and the event log. Light and dark themes follow the system and
can be toggled.

| Key / mouse | Action |
|---|---|
| `1`-`4` | Drive to a location (manual commands switch the driver mode they need first) |
| `s` | Detect; detected objects get a red ring |
| `g` | Grasp the best recent detection |
| `o` | Hand over to the Human |
| `m` | Toggle the driver mode |
| `c` | Cancel everything |
| `d` / `t` | Fetch the selected (or a random) object / tour every location |
| `space`, `+`, `-` | Pause, speed up, slow down |
| `r` / `l` / `0` | Respawn with a new seed / toggle object labels / fit the map |
| scroll, drag | Zoom, pan |
| click a location | Drive there |
| drag an object (map or inventory) onto a location | Move it behind the robot's back |
| hover an object | Its category, location, detector label and position |

After a scan, the scanned surface glows yellow, detected objects get a red ring and
objects the scan missed a dashed grey one; the dashed squares are the poses the robot
parks at.

The viewer talks to the simulator over a small HTTP API, which scripts can use too:
`GET /api/world`, `GET /api/state`, `GET /api/events?since=N`, `GET /api/events.jsonl`,
`GET /api/stream` (server-sent events), and `POST /api/command` with e.g.
`{"cmd": "goto", "location": "Fridge"}` (see `SimController` for the commands).

## Action success rate

`robot.success_rate` in a scenario (or `--success-rate 0.9`, or the slider in the
viewer) is the probability that each robot action succeeds: navigation aborts part way,
a head move aborts, the object slips out of the gripper, or nobody takes the handover.
A scan covers the whole surface the robot is parked at, and each object on it is missed
with the same probability. A scenario can give one action its own probability instead
(`navigation.fail_prob`, `head.fail_prob`, `perception.miss_prob`, `grasp.fail_prob`,
`handover.no_hand_prob`); `--success-rate` and the slider replace those too. Injected
failures are marked `injected_failure` in the log.

## Running a behaviour tree against the sim

With `--ros`, the simulator offers the same ROS interface as the robot:

| Interface | Real robot | Sim |
|---|---|---|
| `/switch_to_{position,navigation,trajectory}_mode` | stretch_driver | driver mode, enforced |
| `/navigate_to_pose` (action) | Nav2 | path following on the map |
| `/stretch_controller/follow_joint_trajectory` (action) | stretch_driver | head joints only |
| `/detect_floor_objects_3d` | floor_object_detector | camera frustum, detections in `base_link`, each with the object's `grasp_direction` |
| `/scan_surface` | scan_surface | looks at the named surface (one head move stands for the sweep), detects, keeps what is on it; an unknown surface fails |
| `/grasp_carry` | grasp_carry | front grasp at the requested point |
| `/topdown_grasp` | topdown_grasp | grasp from above: rises over the object, extends, lowers onto it |
| `/handover` | handover | hands the held object to the Human |
| `/stretch/joint_states`, camera topics | driver, camera | published (camera images are placeholders) |

The sim offers only what the robot offers with `bringup_stretch.sh` running
(`ROBOT_INTERFACES` in `ros_bridge.py`; offering anything else raises), so a tree that
works against the sim does not depend on something the robot lacks.

`run_cooking_task` does not know whether it talks to the robot or the sim. It now
takes `--tree` (any tree builder in `bt_cooking/stretch_bt.py`; `--list-trees`),
`--recipe`, `--max-passes` and `--location-poses`.

```bash
# In a sourced ROS 2 workspace with this package built:
ros2 run stretch_floor_perception run_sim.py --bt main_demo_loop --recipe greek_salad --success-rate 0.9 --open
ros2 run stretch_floor_perception run_sim.py --headless --bt main_demo_loop --recipe greek_salad --speed 4 --snapshot /tmp/run.png
ros2 run stretch_floor_perception run_sim.py --ros     # then pick the tree and recipe in the viewer
```

The simulator starts the BT as a separate process with the sim's location poses
(written to its run directory), streams its output into the event log (kind `bt`), and
keeps its own logs under `<run dir>/bt-NN-<tree>/`. ROS discovery is limited to this
machine unless `--ros-network` is given, so a real robot on the network is unaffected.
To run the BT yourself instead, use the same discovery setting in its terminal
(`export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`) and pass
`--location-poses` with the file from `run_sim.py --export-location-poses`.

Requirements: ROS 2 with `nav2_msgs`, `control_msgs` and `vision_msgs`; `py_trees`;
`unaceba` (see `dependencies.repos`); this package built with colcon. Use the ROS Python,
not a conda one. Interfaces whose message packages are missing are not offered, and the
viewer lists them.

## Running a behaviour tree on the real robot

The BT can drive the real Stretch instead, while the viewer and event log still show the
run: the virtual robot mirrors what the BT does on the sim's map. Switch with **Connect
to robot** / **Disconnect** in the viewer's Behaviour tree card (with the robot's places
file and ROS domain beside it), or start connected with `--robot`. Switching stops the
running tree; the next one runs against whichever is connected.

```bash
ros2 run stretch_floor_perception run_sim.py --robot --open
ros2 run stretch_floor_perception run_sim.py --robot --bt main_demo_loop --recipe greek_salad \
    --robot-places hello-robot@192.168.68.75:~/stretch_user/maps/embrace_apartment_map.places.yaml
```

- **Interfaces:** the sim offers none of the robot's services, actions or topics, since
  the real robot offers them. The BT runs on this machine as usual.
- **Network:** joins the robot's ROS domain (`--robot-domain-id`, default
  `$ROS_DOMAIN_ID`, else 42) and lifts the localhost-only discovery limit.
- **Locations:** the BT navigates with the robot's own poses, read at startup from its
  `places.yaml` (`--robot-places`, a local path or `HOST:PATH` over ssh; default
  `stretch:~/stretch_user/maps/embrace_apartment_map.places.yaml`). Each location uses
  the place with its lower-case name, and Human uses `handover`. Map others with
  `--robot-place Fridge=kitchen`. Human's pose is the place turned 90° left
  (`--robot-place-yaw Human=90`, the default): the arm extends to the robot's right, so
  a place saved facing the person becomes one with the arm towards them. A location
  with no place is reported at startup and
  in the viewer, and the BT cannot navigate there. The poses are written to
  `<run dir>/robot_location_poses.json`.
- **The mirror** (`robot_link.py`) follows the state the BT publishes on
  `/cooking_bt/state`. It doesn't use the robot's own topics, and the sim's map is not the
  robot's map:

  | The BT | The virtual robot |
  |---|---|
  | runs a Navigate node | drives to the sim location of the same name |
  | reports a scan (`scan/objects`) | moves the objects it saw to that location, with red rings |
  | runs the Grasp node | goes to the location and waits; the object is shown in the gripper only once the BT reports the real grasp confirmed |
  | runs the Handover node | goes to the Human and waits; the object is shown handed over only once the BT reports the real handover done |
  | holds something else than the virtual gripper | the gripper is corrected to match, as soon as no animation is running |

  When the BT moves on before the animation arrives, the virtual robot jumps to the
  location. Objects the real robot has not seen stay where the sim spawned them.
- **Stopping the robot:** **Stop** cancels every Nav2 and trajectory goal on the robot
  (whoever sent it), calls the driver's `/stop_the_robot`, then stops the tree, and
  cancels goals again in case the tree sent one while shutting down. It works with no
  tree running too. A grasp, handover or scan service already running on the robot has
  no cancel and carries on: **Cancel everything** (`c`) also **runstops** the robot
  (`/runstop`), which stops those too. The robot then will not move until **Release
  runstop** (shown while runstopped) or its runstop button.
- **Viewer:** commands that would move the virtual robot (drive, detect, grasp,
  handover, mode, fetch, tour, success rate) are refused; shuffling and moving objects
  still work. Simulated failures are off.

## Logging

Every run writes `events.jsonl` to `<workspace>/logs/stretch_sim-<timestamp>/` (or
`~/ros2_logs/...` outside a colcon workspace; `--log-dir` to choose). Each event has a
kind:

| Kind | What |
|---|---|
| `command` | A request to the robot, with its arguments |
| `result` | Its outcome and outputs (status, duration, detections, held object, ...) |
| `step` | Progress within an operation: path planned, waypoint reached, grasp phase |
| `state` | Robot state changes, e.g. driver mode |
| `violation` | Something the robot should not have been asked, e.g. navigating in position mode |
| `info` | Resets, spawn placement, objects moved by hand, scripts, success rate changes |
| `bt` | Output of a behaviour tree process run against the sim |

Operations have an id (`#4`) shared by their command, steps and result.

- **Viewer**: the event log filters by kind (double-click a kind to show only it) and
  by text across messages and data; clicking an event shows its data and the timeline of
  its operation; failures and violations also pop up as notifications; Download saves
  the log.
- **Console**: commands, results and violations; `-v` adds steps.
- **Files**: `python3 -m stretch_sim.log_view --latest`, with `--kind command result`,
  `--source grasp`, `--op 4`, `--grep feta`, `--data`, `--summary`.

## Layout

| File | Role |
|---|---|
| `world.py` | Apartment map, locations and approach poses, object spawning |
| `planner.py` | Grid A* around walls and furniture |
| `perception.py` | Head camera view frustum and detection noise |
| `robot.py` | Robot state and simplified kinematics |
| `operations.py` | Long-running operations (navigate, head, grasp, handover) stepped each tick |
| `simulator.py` | `RobotSimulator` interface and the 2D `KinematicSimulator` |
| `events.py` | Event log, console and JSONL sinks |
| `controller.py` | Runs the simulation in real time and executes manual commands |
| `web/server.py`, `web/static/` | Web viewer: HTTP API, event stream, the page |
| `scene.py`, `render_pil.py` | Headless PNG snapshots |
| `script.py` | Scripted sequences (demo fetch, tour) that use ground truth |
| `ros_bridge.py` | The robot's ROS interface, backed by the simulator |
| `bt_runner.py` | Starts `run_cooking_task` against the sim and captures its output |
| `ros_link.py` | The ROS side, switched at run time between the sim playing the robot and the real robot |
| `robot_link.py` | Robot mode: the real robot's location poses, and the mirror of the BT on the virtual map |

`config/sim/worlds/*.yaml` describe apartments and `config/sim/scenarios/*.yaml` choose
a world, seed, where objects are, and failure probabilities. `apartment.yaml` is
hardcoded from the lab layout sketch (read at 1 px = 1 cm, 7.80 m x 5.37 m) until the
ROS map and zone definitions are available.

A physics-based simulator can replace `KinematicSimulator` by implementing
`RobotSimulator`; the viewer, logging and ROS layer only use that interface.

## Modelling assumptions to check against the robot

- A scan covers the whole surface the robot is parked at: the nearest surface whose
  closest edge is within 1.5 m on the robot's arm side. Where the head points is left to
  the perception stack, as on the robot. Change it with `robot: camera:` in a scenario.
- The arm extends to the base's right, reaching 0.25-0.77 m from the base origin; a grasp
  may shift the base up to 0.8 m along its heading to line up.
- Driver mode rules: navigation needs `navigation` mode; grasp and handover need
  `position` mode; head moves work in position, navigation and trajectory mode, as the
  driver's trajectory server accepts them (`driver_mode_rules` in a scenario).
- Every object is grasped from the front unless the scenario's `grasp_directions` says
  `top`; detections report it as `grasp_direction`, as the real detector does.
- The grasp services answer like the real servers after their grasp check: success only
  with something in the gripper, and `Wrong object: ...` when `expected_label` names a
  different detector label (compared ignoring case, spaces and underscores). The
  detector's 3D box in the request is logged but not used; the sim grasps at the point.
