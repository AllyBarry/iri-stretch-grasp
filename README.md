# stretch_floor_perception

ROS 2 (Humble) package for Hello Robot Stretch 2 with the dex wrist. It finds
objects on the floor with the head camera, picks them up, and hands them to a
person.

```
scan_client ──► floor_object_detector ──► (user picks an object)
                 detect_floor_objects_3d          │
                                                  ▼
                       grasp_direction == "top" ? /topdown_grasp_carry : /grasp_carry
                                                  │
                                                  ▼
                                        handover (/handover)
```

## Nodes

| Executable | Service it provides | What it does |
|---|---|---|
| `floor_object_detector.py` | `detect_floor_objects_3d` (`DetectFloorObjects3D`), `detect_objects_2d` (`DetectObjects2D`) | Runs a 2D detector on a color + aligned depth frame and projects the detections to 3D in `base_link`. Optionally asks a VLM (OpenAI-compatible API) what each object is, whether to pick it up, and whether to grasp it from the front or the top. `detect_objects_2d` runs only the 2D detector on one colour frame (no depth, VLM or floor filtering) and returns boxes in that frame's pixel coordinates; the grasp nodes use it to check what is in the gripper. |
| `scan_client.py` | — | Interactive client. Press Enter to scan, pick an object by index, and it calls the front or top-down grasp service depending on the detection's `grasp_direction`, passing the detector's label as `expected_label`. |
| `grasp_carry.py` | `/grasp_carry` (`GraspPoint`) | Front grasp at a 3D point, lift, then turn the gripper toward the head camera and roll the object onto its side. Confirms the grasp by detecting `expected_label` in the gripper (`detect_objects_2d`) and checking it is no longer at the pick location. Publishes the annotated check image on `/grasp_carry/grasp_check_image`. |
| `grasp_point.py` | `/grasp_point` (`GraspPoint`) | Older front grasp. After lifting, it waits for Enter, opens the gripper and stows. |
| `topdown_grasp_point.py` | `/topdown_grasp_point` (`GraspPoint`) | Top-down grasp off a table, desk or the floor. Measures the surface and object height around the target, raises the retracted arm, extends from above, tilts the gripper down over the object, opens and descends around it. Ends holding the object with the arm retracted, and confirms the grasp the same way as `grasp_carry` (image on `/topdown_grasp_point/grasp_check_image`). |
| `handover.py` | `/handover` (`std_srvs/Trigger`) | Hands the held object to a person. See [Handover](#handover). |
| `carry_pose.py` | `/carry_pose` (`std_srvs/Trigger`) | Puts the robot in a pose for carrying a held object: retracts the arm, then swings the gripper toward the front of the base. The wrist rolls to whichever of 0 or +-pi is nearer, so the fingers close horizontally and an object gripped from the side, such as a bottle, is held vertical. Pitch goes level, or stays pointing down after a top-down grasp. The lift and gripper are not commanded. |
| `scan_surface.py` | `/scan_surface` (`ScanSurface`) | Sweeps the head camera over a surface (desk, table, floor) once the robot has arrived, calling `detect_floor_objects_3d` at each pan/tilt view. With `surface_name`, the views are planned to cover a surface defined in the map and objects not on it are dropped. Objects are merged across views and kept only if overlapping views confirm them. Full sweep returns every object; search mode stops at the first confirmed `target_label`. See [Surface scan](#surface-scan). |
| `define_surface.py` | — | Records a surface for `scan_surface` (polygon and height in the map frame) from corners clicked in RViz. |

All grasp and handover nodes accept `dryrun:=true` (no motion) and
`debug_mode:=true` (asks `[y/N]` in the terminal before each move).

## Interfaces

- `msg/DetectedObject3D`: `class_name`, `detector_class_name`, `pick_up`, `grasp_direction`, `score`, `centroid`, `bbox3d`
- `msg/DetectedObject3DArray`
- `srv/DetectFloorObjects3D`: compressed color and depth images plus `CameraInfo` → `DetectedObject3DArray`
- `srv/GraspPoint`: `PointStamped target`, `float32 assumed_width_m` (≤ 0 uses the node's default) → `success`, `message`
- `srv/ScanSurface`: `string target_label` (empty for a full sweep), `string surface_name` (optional), optional `pan_angles_rad` / `tilt_angles_rad` → `success`, `message`, `found`, `DetectedObject3D target`, `DetectedObject3DArray detections`, `views_scanned`

## Build

```bash
cd ~/ament_ws
colcon build --packages-select stretch_floor_perception
source install/setup.bash
```

Python dependencies by feature:

| Feature | Needs |
|---|---|
| `detr` backend | `transformers`, `torch` |
| `fastsam` backend | `ultralytics`; optionally `ram` for RAM++ labels |
| `yolo-darknet` backend | darknet shared library plus `.cfg`, `.weights` and `.names` files |
| VLM labelling | `openai` and a running OpenAI-compatible server (for example llama.cpp) |
| `handover.py find_human:=true` | `ultralytics`, with weights at `~/stretch_user/models/yolov8n.pt` |

`onnxruntime` is listed as a backend but is a placeholder that returns no detections.

## Running

### Full stack

`stretch_startup.sh` homes and stows the robot, then opens a tmux session
(`StretchPnP`) with the driver, keyboard teleop, navigation, grasp services,
TTS and emergency call node. Before you use it, check `MAP_PATH`,
`KEEPOUT_MAP_PATH` and `ROBOT_IP` at the top of the script.

```bash
./stretch_startup.sh
```

### Detector

There is one launch file per backend:

```bash
ros2 launch stretch_floor_perception floor_object_detector.sam.launch.py   # FastSAM + VLM (gemma on :8001)
ros2 launch stretch_floor_perception floor_object_detector.detr.launch.py  # DETR ResNet-50
ros2 launch stretch_floor_perception floor_object_detector.yolo.launch.py  # YOLOv4-tiny (darknet)
```

The YOLO launch file points at `/workspaces/...` paths from the dev container.
Change them to match where the files are on the robot.

The detector publishes `image_with_boxes`, `detected_centroids` and
`detected_bboxes_3d` for RViz.

### Scan and grasp

```bash
ros2 run stretch_floor_perception grasp_carry.py
ros2 run stretch_floor_perception scan_client.py
```

`scan_client` waits for `detect_floor_objects_3d` and `/grasp_carry`, and
uses `/topdown_grasp_carry` if it is available. Change the service names with
`grasp_service_name` and `topdown_grasp_service_name`.

### Surface scan

Once the robot has reached a desk or table, `/scan_surface` steps the head
through a grid of pan/tilt views (starting from the end nearer the current
head pan). The views come from, in order of precedence: `pan_angles_rad` /
`tilt_angles_rad` in the request, the surface named by `surface_name`, or the
`scan_pans_rad` / `scan_tilts_rad` parameters. At each view it waits
`head_settle_s`, takes a fresh color and depth frame and calls
`detect_floor_objects_3d`. Afterwards the head returns to where it was
(`restore_head`).

The base does not move, so an object has nearly the same `base_link`
centroid in every view. The views overlap (`view_overlap`, or 0.3 rad steps in
`scan_pans_rad`), so most objects are seen more than once. These sightings are
combined and cross-checked:

- **Same object:** a detection joins an object already seen within
  `merge_radius_m` with the same name, or within `label_merge_radius_m`
  whatever its name. The second case covers the detector naming it
  differently from another angle.
- **Name:** the name given in the most views wins, with ties going to the
  higher summed score. The returned detection is the one nearest an image
  centre, where the object is least likely to be cut off.
- **Confirmation:** an object is kept only if it was detected in
  `confirm_views` of the views that had it well inside the image (more than
  `view_edge_margin_rad` from the edge), or in all of them if fewer did. A
  label the detector gives from one angle only, such as something next to the
  salt called "mustard", is dropped. An object only one view could see is
  kept. So is one seen once with a score of at least `single_view_min_score`,
  because YOLO sometimes misses a real object from a neighbouring angle.
  Set `confirm_views:=1` to keep every detection. Two different names
  detected in the same view are always two objects.

Each view's detections are logged with their positions. At the end, each
object is logged as kept or dropped, with the number of views it was seen in.

- **Full sweep** (`target_label` empty): visits every view and returns all
  confirmed objects.
- **Search** (`target_label` set): stops at the first confirmed object whose
  VLM or detector name contains `target_label` (case-insensitive). The match
  usually comes from the view after the one that first saw it. It is returned
  in `target` with `found: true`. `success` is still true if the object is not
  found; it is false only when the sweep could not run.

#### Surfaces in the map

A surface is a polygon and a height in the map frame, saved by name in
`surfaces_file` (default
`~/stretch_user/maps/embrace_apartment_map.surfaces.yaml`, next to the places
file). Surface names are separate from place names, so places such as
`desk_close` and `desk_far` can both scan the surface `desk`.

```yaml
frame_id: map
surfaces:
  desk:
    height: 0.74
    polygon: [[2.10, -0.40], [3.30, -0.40], [3.30, 0.30], [2.10, 0.30]]
```

With `surface_name` set, `/scan_surface`:

1. Moves the polygon, grown by `view_margin_m` to allow for localisation
   error, into `base_link` using the robot's current map pose.
2. Works out the pan and tilt range at which the head sees it, from the near
   edge up to `max_object_height_m` above the far edge. It then picks views
   that cover that range, using `camera_hfov_rad` / `camera_vfov_rad` and
   `view_overlap`. Bearings behind the robot use the pan range past -pi. It
   warns if part of the surface is outside `head_pan_limits_rad`.
3. Keeps only detections whose centroid is inside the polygon grown by
   `surface_margin_m`, and whose box bottom is within
   `surface_height_tolerance_m` of the surface height. The names of dropped
   objects are logged.
4. Shows the surface it is scanning in green on `/scan_surface/surfaces`
   (see below).

The file is re-read on every call, so new surfaces work without a restart.

`/scan_surface/surfaces` (MarkerArray, map frame, latched) shows the outline
and name of every surface in `surfaces_file`. The file is re-read every
`surfaces_publish_period_s`, so new or edited surfaces appear without a
restart. In RViz:

- **3D view:** add the topic to see the surfaces over the camera point cloud
  and the map. This needs the robot to be localised.
- **Over the camera image:** add a **Camera** display (not Image) on
  `/camera/color/image_raw`. It draws the 3D scene, including markers, from
  the camera's viewpoint. Under its **Visibility**, keep the surfaces display
  and untick the point cloud. The image is sideways, as the camera is mounted
  rotated.

To define a surface, localise the robot and run:

```bash
ros2 run stretch_floor_perception define_surface.py --ros-args -p name:=desk
```

In RViz, use **Publish Point** to click the corners. The grasp-test RViz
(`grasp_test.rviz`) is easiest, because it shows the camera point cloud. Any
fixed frame works, because clicks are converted to the map frame. Move the head
between clicks if the table does not fit in one view, but keep the base still.
They can be clicked in any order, because they are sorted around their
centre (so concave shapes need splitting into convex parts). Clicking on the
camera point cloud gives the height as the median of the clicks. Clicks on
the map are at z = 0, so pass `-p height:=0.74` instead. The outline so far
is shown on `/define_surface/polygon`. In the terminal, Enter saves, `u`
removes the last corner and `q` quits. Saving replaces an existing surface
with the same name.

The detector's `max_distance_above_floor_m` must allow the surface height.
`~/stretch_user/grasp_test_config/detector_yolo.yaml` sets 1.0 m; the code
default of 0.3 m drops objects on a desk.

```bash
ros2 run stretch_floor_perception floor_object_detector.py --ros-args \
  --params-file ~/stretch_user/grasp_test_config/detector_yolo.yaml
ros2 run stretch_floor_perception scan_surface.py
ros2 service call /scan_surface stretch_floor_perception/srv/ScanSurface "{surface_name: desk}"
ros2 service call /scan_surface stretch_floor_perception/srv/ScanSurface "{surface_name: desk, target_label: cup}"
# Fixed views, no surface filter
ros2 service call /scan_surface stretch_floor_perception/srv/ScanSurface "{}"
# Only look to the arm side, at two tilts
ros2 service call /scan_surface stretch_floor_perception/srv/ScanSurface \
  "{target_label: cup, pan_angles_rad: [-1.0, -1.57, -2.1], tilt_angles_rad: [-0.45, -0.95]}"
```

| Parameter | Default | Notes |
|---|---|---|
| `scan_pans_rad` | [0.6, 0.3, … -1.5, -1.8] | From slightly left of the front round to past the arm side. The upright image is ~42° wide, so 0.3 rad steps put every spot in two views. |
| `scan_tilts_rad` | [-0.7] | The upright image is ~69° tall, so one tilt covers a table 0.5–1.2 m away. |
| `head_settle_s` | 0.5 | The detector uses the latest camera TF, so the head must be still before the frame. |
| `merge_radius_m` | 0.08 | Detections with the same name this close are one object. |
| `label_merge_radius_m` | 0.04 | Detections with different names this close are one object. |
| `confirm_views` | 2 | Views an object must be detected in, out of those that had it well inside the image. 1 keeps every detection. |
| `single_view_min_score` | 0.75 | A single sighting this confident is enough. Real chickpeas missed in a neighbouring view scored 0.76; the one-off wrong labels seen so far scored 0.66–0.67. Above 1 disables it. |
| `view_edge_margin_rad` | 0.05 | |
| `detect_timeout_s` | 60.0 | Per view; the VLM is slow. |
| `restore_head` | true | |
| `dryrun` | false | Logs head moves without making them, and still detects. |
| `surfaces_file` | `~/stretch_user/maps/embrace_apartment_map.surfaces.yaml` | Also a parameter of `define_surface.py`. |
| `view_margin_m` | 0.08 | |
| `max_object_height_m` | 0.25 | |
| `camera_hfov_rad` / `camera_vfov_rad` | 0.73 / 1.20 | Of the upright image. |
| `view_overlap` | 0.55 | Fraction by which neighbouring views overlap. Over half, so every part of the surface is in at least two pans. |
| `head_pan_limits_rad` / `head_tilt_limits_rad` | [-3.9, 1.5] / [-1.5, 0.4] | |
| `surface_margin_m` | 0.05 | |
| `surface_height_tolerance_m` | 0.08 | |
| `surfaces_publish_period_s` | 2.0 | How often `/scan_surface/surfaces` is refreshed from the file. 0 disables it. |

Example of calling a grasp service directly:

```bash
ros2 service call /grasp_carry stretch_floor_perception/srv/GraspPoint \
  "{target: {header: {frame_id: base_link}, point: {x: 0.0, y: -0.6, z: 0.02}}, assumed_width_m: 0.05}"
```

## Handover

`handover.py` assumes the robot is already holding an object. It goes through
these steps:

1. **Find the person** (optional, `find_human:=true`). The head pans through
   `person_scan_pans_rad` and runs YOLO person detection on each view. It
   checks the current view first. If no one is visible, it sweeps the pans
   from the nearer end at each of `person_scan_tilts_rad` (level by default). The
   person's 3D position comes from the depth cloud. If several people are
   visible, it uses the closest. If the person is too far away for the arm to
   reach, the robot turns to face them and drives up to `max_approach_m`
   towards them. The base then rotates so the arm faces the person,
   re-checking after each turn. This step assumes there is nothing between
   the robot and the person. If nobody is found, the node does not extend the
   arm and reports failure.
2. **Present the object.** The node raises the lift to `present_lift_m`,
   levels the wrist and extends the arm by `present_extension_m`. With a
   person found, the gripper stops at least `person_standoff_m` short of
   them. The head camera then turns to the gripper.
3. **Learn the baseline.** For `baseline_s`, the node records which voxels
   around the gripper and object are occupied. The hand detector looks for
   anything new in those voxels, so without this step the gripper and object
   would themselves count as a hand. Keep hands away during this step.
4. **Wait for a hand.** A hand counts when new voxels within
   `contact_radius_m` of the gripper stay present for `release_delay_s`. If
   the hand leaves early, the wait starts over. If no hand arrives within
   `hand_timeout_s`, the arm retracts while still holding the object.
5. **Release.** The gripper opens to `release_aperture_m`. The node then waits
   up to `clear_timeout_s` for the hand to move away and retracts the arm,
   keeping the lift where it is (`retract_after_handover`, true by default).
   With `stow_after_handover:=true` it stows the robot instead.

Every motion is checked. If the driver rejects or aborts one, for example
because the robot is runstopped, the handover fails with that reason. In
particular, it never reports an object as handed over while it is still in the
gripper.

```bash
# Run once and exit
ros2 run stretch_floor_perception handover.py
# Find a person first. Use dryrun first to check the detected position and planned turn without moving.
ros2 run stretch_floor_perception handover.py --ros-args -p find_human:=true -p dryrun:=true
# Run as a service
ros2 run stretch_floor_perception handover.py --ros-args -p run_once:=false
ros2 service call /handover std_srvs/srv/Trigger
```

| Parameter | Default | Notes |
|---|---|---|
| `present_lift_m` | 0.85 | Suits a standing person. Use about 0.6–0.7 for someone seated. |
| `present_extension_m` | 0.20 | |
| `release_delay_s` | 1.0 | How long the hand must stay before the gripper opens. |
| `min_new_voxels` | 20 | Hand-detection sensitivity. Tune with `--log-level handover:=debug`. |
| `contact_radius_m` | 0.15 | |
| `hand_timeout_s` | 60.0 | |
| `find_human` | false | Must be passed after `--ros-args` (see above). The node logs the value it read at startup (`Settings: find_human=...`). |
| `stow_after_handover` | false | |
| `person_standoff_m` | 0.35 | Minimum gap between the gripper and the person. |
| `max_approach_m` | 1.0 | Furthest the robot will drive towards the person. Set to 0 to only rotate. |
| `person_scan_pans_rad` | [-3.6, -3.0, … 0.6, 1.2] | 0.6 rad (≈35°) steps, so neighbouring camera views overlap. |
| `person_scan_tilts_rad` | [-0.15] | Add a lower tilt, e.g. `[-0.15, -0.45]`, to also look for seated or close people at each stop (slower). |
| `cloud_timeout_s` | 5.0 | How long to wait for a point cloud. The node checks for one before moving. |
| `head_settle_s` | 0.3 | Pause after each head move before detecting. |
| `rotate_image` | true | The head camera is mounted sideways, so images are rotated before detection. |

With `find_human:=true`, the node switches the driver to position mode before
it moves the base. When the handover ends, whether it succeeds or not, the
node switches back to the mode the driver was in before (for example
navigation).

## Notes

- `stretch_startup.sh` starts `grasp_carry.py` (`/grasp_carry`) and
  `topdown_grasp.py` (`/topdown_grasp`) with
  `~/stretch_user/grasp_test_config/grasp_table.yaml`, which `scan_client`
  calls by default. `explore_node` and `patrol_node` (in
  `stretch_iri_navigation`) default to `/grasp_point` and
  `/topdown_grasp_point`, so the startup script passes them the new service
  names.
- The grasp nodes use `stretch_funmap` for surface scanning and pregrasp
  planning. They expect the driver to be in position mode.
