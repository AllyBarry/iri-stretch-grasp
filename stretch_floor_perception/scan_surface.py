#!/usr/bin/env python3
"""scan_surface.py: a /scan_surface service that sweeps the head camera over a surface and
detects the objects on it.

Call it once the robot has reached a location such as a desk or table. The head steps through
a grid of pan/tilt poses; at each one the node waits for the head to settle, takes a fresh color
+ depth frame and calls detect_floor_objects_3d. Detections are in base_link, which does not move
during the sweep, so sightings of the same object in overlapping views are combined (ObjectTracks):
the name given in most views wins, and an object is kept only if the views that had it well in
the image agree it is there.

  Full sweep: every confirmed object.
  Search:     target_label set; stops at the first confirmed match.

With surface_name set, the surface's polygon and height are read from surfaces_file (written by
define_surface.py). The polygon is moved into base_link with the robot's current map pose, the
head poses are planned to cover it edge to edge, and detections that are not on it are dropped.
Without it, the fixed scan_pans_rad / scan_tilts_rad poses are used and every detection is kept.

  ros2 run stretch_floor_perception scan_surface.py
  ros2 run stretch_floor_perception scan_surface.py --ros-args -p dryrun:=true   # no head motion
  ros2 service call /scan_surface stretch_floor_perception/srv/ScanSurface "{surface_name: desk}"
  ros2 service call /scan_surface stretch_floor_perception/srv/ScanSurface "{surface_name: desk, target_label: cup}"

The detector's max_distance_above_floor_m must allow the surface height (detector_yolo.yaml sets
1.0 m; the code default of 0.3 m drops objects on a desk).
Head goals are accepted in navigation mode, so the driver mode is left as it is.
"""
import math
import os
import threading
import time

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from stretch_floor_perception.srv import DetectFloorObjects3D, ScanSurface


def normalise_name(name):
    # Same as stretch_places: "Kitchen Table" -> kitchen_table
    return '_'.join(name.strip().lower().split())


def transform_to_matrix(transform):
    t, q = transform.transform.translation, transform.transform.rotation
    mat = quaternion_matrix([q.x, q.y, q.z, q.w])
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat


def grow_polygon(corners, margin_m):
    """Move each corner margin_m further from the centre. Exact enough for convex surfaces."""
    centre = corners.mean(axis=0)
    offsets = corners - centre
    dist = np.maximum(np.linalg.norm(offsets, axis=1, keepdims=True), 1e-6)
    return centre + offsets * (1.0 + margin_m / dist)


def polygon_boundary(corners, spacing_m=0.02):
    """Points every spacing_m along the closed polygon outline."""
    points = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        n = max(int(np.linalg.norm(b - a) / spacing_m), 1)
        points.append(a + (b - a) * np.linspace(0.0, 1.0, n, endpoint=False)[:, None])
    return np.vstack(points)


def inside_polygon(corners, x, y):
    inside = False
    for (x1, y1), (x2, y2) in zip(corners, np.roll(corners, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def view_ranges(outline, head, max_object_height_m):
    """Pan and tilt ranges (rad) at which the head sees the outline (Nx3, same frame as head)."""
    # Bearings from the head, unwrapped around the bearing of the outline's centre so a surface
    # behind the robot does not split at +-pi.
    offsets = outline - head
    centre_bearing = math.atan2(offsets[:, 1].mean(), offsets[:, 0].mean())
    bearings = np.arctan2(offsets[:, 1], offsets[:, 0]) - centre_bearing
    bearings = (bearings + math.pi) % (2.0 * math.pi) - math.pi + centre_bearing
    horizontal = np.hypot(offsets[:, 0], offsets[:, 1])
    # From the nearest edge of the surface to the tops of tall objects at the far edge.
    tilt_lo = np.arctan2(offsets[:, 2], horizontal).min()
    tilt_hi = np.arctan2(offsets[:, 2] + max_object_height_m, horizontal).max()
    return float(bearings.min()), float(bearings.max()), float(tilt_lo), float(tilt_hi)


def cover(lo, hi, fov, overlap):
    """View centres, each fov wide and overlapping by the fraction overlap, that cover [lo, hi]."""
    if hi - lo <= fov:
        return [(lo + hi) / 2.0]
    step = fov * (1.0 - overlap)
    n = int(math.ceil((hi - lo - fov) / step)) + 1
    return [float(c) for c in np.linspace(lo + fov / 2.0, hi - fov / 2.0, n)]


class ObjectTracks:
    """Sightings of the same object across views, and whether enough views confirm it.

    The base does not move during the sweep, so an object has (nearly) the same base_link
    centroid in every view. A detection joins the nearest track within merge_radius of it with
    the same name, or within label_merge_radius whatever its name (the detector named the same
    object differently from another angle; never for two names in one view). A track is confirmed once it has been seen in
    confirm_views of the views that had its position well inside the image, or in all of them
    if fewer could. A label the detector gives from one angle only is then dropped, unless that
    sighting scored at least single_view_min_score (the detector can miss a real object from a
    neighbouring angle, but rarely gives a wrong label a high score).
    """

    def __init__(self, views, head, hfov, vfov, edge_margin, merge_radius, label_merge_radius, confirm_views,
                 single_view_min_score=None):
        self.views = views          # [(pan, tilt)] in scan order
        self.head = head            # head position in base_link, or None if unknown
        self.hfov, self.vfov, self.edge_margin = hfov, vfov, edge_margin
        self.merge_radius, self.label_merge_radius = merge_radius, label_merge_radius
        self.confirm_views = confirm_views
        self.single_view_min_score = single_view_min_score   # None: always need confirm_views
        self.tracks = []            # each a list of (view index, detection, centrality)

    @staticmethod
    def xyz(det):
        return np.array([det.centroid.x, det.centroid.y, det.centroid.z])

    def _offsets(self, view, point):
        """Angles of point from the centre of the view's image (pan, tilt)."""
        d = point - self.head
        bearing = math.atan2(d[1], d[0])
        elevation = math.atan2(d[2], math.hypot(d[0], d[1]))
        return (bearing - view[0] + math.pi) % (2.0 * math.pi) - math.pi, elevation - view[1]

    def centrality(self, view, point):
        """0 at the centre of the view's image, 1 at its edge."""
        if self.head is None:
            return 0.0
        d_pan, d_tilt = self._offsets(view, point)
        return max(abs(d_pan) / (self.hfov / 2.0), abs(d_tilt) / (self.vfov / 2.0))

    def covers(self, view, point):
        """Whether point is well inside the view's image, so the object there should be detected."""
        if self.head is None:
            return False
        d_pan, d_tilt = self._offsets(view, point)
        return (abs(d_pan) <= self.hfov / 2.0 - self.edge_margin and
                abs(d_tilt) <= self.vfov / 2.0 - self.edge_margin)

    def add(self, view_index, detections):
        for det in detections:
            point, name = self.xyz(det), det.class_name.lower()
            best, best_dist = None, None
            for track in self.tracks:
                same_name = name in self.names(track)
                # Two names from one view are two objects (e.g. side by side, with one's depth
                # landing on the other), not one object named differently from another angle.
                if not same_name and view_index in self.seen_views(track):
                    continue
                dist = float(np.linalg.norm(self.position(track) - point))
                limit = self.merge_radius if same_name else self.label_merge_radius
                if dist < limit and (best is None or dist < best_dist):
                    best, best_dist = track, dist
            sighting = (view_index, det, self.centrality(self.views[view_index], point))
            if best is None:
                self.tracks.append([sighting])
            else:
                best.append(sighting)

    def position(self, track):
        return np.mean([self.xyz(det) for _, det, _ in track], axis=0)

    @staticmethod
    def names(track):
        return {det.class_name.lower() for _, det, _ in track}

    @staticmethod
    def seen_views(track):
        return {view for view, _, _ in track}

    def covering_views(self, track):
        position = self.position(track)
        return {i for i, view in enumerate(self.views) if self.covers(view, position)} | self.seen_views(track)

    def confirmed(self, track):
        required = max(1, min(self.confirm_views, len(self.covering_views(track))))
        if len(self.seen_views(track)) >= required:
            return True
        return self.single_view_min_score is not None and self.best_score(track) >= self.single_view_min_score

    def best_score(self, track):
        label = self.label(track)
        return max(det.score for _, det, _ in track if det.class_name.lower() == label)

    @staticmethod
    def label(track):
        """The name given in the most views; ties go to the higher summed score."""
        votes = {}
        for view, det, _ in track:
            views, score = votes.get(det.class_name.lower(), (set(), 0.0))
            votes[det.class_name.lower()] = (views | {view}, score + det.score)
        return max(votes, key=lambda name: (len(votes[name][0]), votes[name][1]))

    def best(self, track):
        """The sighting with the winning name nearest the image centre, where the object is least
        likely to be cut off by the image edge."""
        label = self.label(track)
        _, det, _ = min((s for s in track if s[1].class_name.lower() == label),
                        key=lambda s: (round(s[2], 1), -s[1].score))
        return det

    def describe(self, track):
        p = self.position(track)
        text = (f"'{self.label(track)}' at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}): seen in "
                f'{len(self.seen_views(track))} of {len(self.covering_views(track))} views, '
                f'best score {self.best_score(track):.2f}')
        others = self.names(track) - {self.label(track)}
        return text + (f", also named {', '.join(sorted(others))}" if others else '')


class ScanSurfaceNode(Node):
    def __init__(self):
        super().__init__('scan_surface')
        self.declare_parameter('service_name', '/scan_surface')
        self.declare_parameter('detect_service_name', 'detect_floor_objects_3d')
        self.declare_parameter('color_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw/compressedDepth')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        # The upright image is only ~42 deg wide, so 0.3 rad pan steps put every spot in two views to
        # cross-check detections. The default covers from slightly left of the front of the base
        # round to past the arm side (-y).
        self.declare_parameter('scan_pans_rad', [0.6, 0.3, 0.0, -0.3, -0.6, -0.9, -1.2, -1.5, -1.8])
        # The image is ~69 deg tall, so one tilt sees from just below the horizon to near the base.
        # Add a second tilt (e.g. [-0.45, -0.95]) for a deep table or a low surface.
        self.declare_parameter('scan_tilts_rad', [-0.7])
        self.declare_parameter('head_move_s', 1.0)
        # The detector looks up the latest camera TF, so the head must be still when the frame is taken.
        self.declare_parameter('head_settle_s', 0.5)
        self.declare_parameter('image_timeout_s', 3.0)
        # Generous, since the VLM is asked about each detection.
        self.declare_parameter('detect_timeout_s', 60.0)
        # Detections with the same name whose centroids are this close are one object; with different
        # names, they must be this close (the detector named it differently from another angle).
        self.declare_parameter('merge_radius_m', 0.08)
        self.declare_parameter('label_merge_radius_m', 0.04)
        # An object must be detected in this many of the views that had it well inside the image (or
        # in all of them, if fewer), so a label the detector gives from one angle only is dropped.
        # 1 keeps every detection. Needs overlapping views (view_overlap) to have any effect.
        self.declare_parameter('confirm_views', 2)
        # But one sighting scoring at least this is enough: YOLO sometimes misses a real object from a
        # neighbouring angle (chickpeas 0.76 in one view, below yolo_conf in the next), while the
        # one-off wrong labels seen so far scored 0.66-0.67. Above 1 disables it.
        self.declare_parameter('single_view_min_score', 0.75)
        # A view only counts as having had an object well inside its image if it is this far from the edge.
        self.declare_parameter('view_edge_margin_rad', 0.05)
        self.declare_parameter('restore_head', True)
        self.declare_parameter('dryrun', False)

        # Named surfaces (surface_name in the request)
        self.declare_parameter('surfaces_file', os.path.expanduser(
            '~/stretch_user/maps/embrace_apartment_map.surfaces.yaml'))
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('head_frame', 'link_head_tilt')
        # Field of view of the upright (rotated) image, and how much neighbouring views overlap.
        self.declare_parameter('camera_hfov_rad', 0.73)
        self.declare_parameter('camera_vfov_rad', 1.20)
        # Over half, so every part of the surface is in at least two pans to cross-check detections.
        self.declare_parameter('view_overlap', 0.55)
        # Head poses are kept within these. The driver's own limits are a little wider.
        self.declare_parameter('head_pan_limits_rad', [-3.9, 1.5])
        self.declare_parameter('head_tilt_limits_rad', [-1.5, 0.4])
        # Grow the surface by this when planning views, to allow for localisation error.
        self.declare_parameter('view_margin_m', 0.08)
        # Objects up to this tall at the far edge are kept in view.
        self.declare_parameter('max_object_height_m', 0.25)
        # A detection is on the surface if its centroid is inside the polygon grown by
        # surface_margin_m and the bottom of its box is within surface_height_tolerance_m of the height.
        self.declare_parameter('surface_margin_m', 0.05)
        self.declare_parameter('surface_height_tolerance_m', 0.08)
        # How often surfaces_file is re-read and every surface published on ~/surfaces (0 disables).
        self.declare_parameter('surfaces_publish_period_s', 2.0)
        self.dryrun = bool(self.get_parameter('dryrun').value)

        group = ReentrantCallbackGroup()
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.image_lock = threading.Lock()
        self.color = self.depth = self.info = None
        self.color_seq = self.depth_seq = 0
        self.create_subscription(CompressedImage, self.get_parameter('color_topic').value,
                                 self._color_cb, qos, callback_group=group)
        self.create_subscription(CompressedImage, self.get_parameter('depth_topic').value,
                                 self._depth_cb, qos, callback_group=group)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value,
                                 self._info_cb, qos, callback_group=group)

        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.create_subscription(JointState, '/stretch/joint_states', self._joint_states_cb, 1,
                                 callback_group=group)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Every surface in surfaces_file, with the one being scanned highlighted. Latched, so RViz
        # shows them as soon as it subscribes.
        self.surfaces_pub = self.create_publisher(
            MarkerArray, '~/surfaces', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.active_surface = None
        self.surfaces_error = None
        period_s = float(self.get_parameter('surfaces_publish_period_s').value)
        if period_s > 0.0:
            self.create_timer(period_s, self._publish_surfaces, callback_group=group)

        self.detect = self.create_client(DetectFloorObjects3D, self.get_parameter('detect_service_name').value,
                                         callback_group=group)
        self.trajectory = ActionClient(self, FollowJointTrajectory,
                                       '/stretch_controller/follow_joint_trajectory', callback_group=group)

        self.scan_lock = threading.Lock()
        service_name = self.get_parameter('service_name').value
        self.create_service(ScanSurface, service_name, self._scan_cb, callback_group=group)
        self.get_logger().info(f'Ready: ros2 service call {service_name} stretch_floor_perception/srv/ScanSurface '
                               f'"{{surface_name: desk, target_label: cup}}"  (dryrun={self.dryrun}, '
                               f'surfaces_file={self.get_parameter("surfaces_file").value})')

    # ------------------------------------------------------------------ inputs

    def _color_cb(self, msg):
        with self.image_lock:
            self.color = msg
            self.color_seq += 1

    def _depth_cb(self, msg):
        with self.image_lock:
            self.depth = msg
            self.depth_seq += 1

    def _info_cb(self, msg):
        with self.image_lock:
            self.info = msg

    def _joint_states_cb(self, msg):
        with self.joint_states_lock:
            self.joint_states = dict(zip(msg.name, msg.position))

    def _head_pose(self):
        with self.joint_states_lock:
            joints = self.joint_states
        if joints is None or 'joint_head_pan' not in joints or 'joint_head_tilt' not in joints:
            return None
        return joints['joint_head_pan'], joints['joint_head_tilt']

    def _wait_for_new_frame(self):
        """Color, depth and camera info, with color and depth both received after this call."""
        with self.image_lock:
            color_seq, depth_seq = self.color_seq, self.depth_seq
        deadline = time.time() + float(self.get_parameter('image_timeout_s').value)
        while time.time() < deadline:
            with self.image_lock:
                if self.color_seq > color_seq and self.depth_seq > depth_seq and self.info is not None:
                    return self.color, self.depth, self.info
            time.sleep(0.02)
        return None

    # ------------------------------------------------------------------ surfaces

    def _read_surfaces(self):
        """{name: raw entry} from surfaces_file; raises RuntimeError if it cannot be read."""
        path = self.get_parameter('surfaces_file').value
        try:
            with open(path) as f:
                return (yaml.safe_load(f) or {}).get('surfaces') or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f'cannot read surfaces_file {path}: {exc}')

    @staticmethod
    def _parse_surface(name, surface):
        try:
            corners = np.array(surface['polygon'], dtype=float)
            height = float(surface['height'])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"surface '{name}' needs a height and a polygon: {exc}")
        if corners.ndim != 2 or corners.shape[0] < 3 or corners.shape[1] != 2:
            raise RuntimeError(f"surface '{name}' needs a polygon of at least 3 [x, y] corners")
        return height, corners

    def _load_surface(self, name):
        """(height, corners Nx2 in the map frame); raises RuntimeError if it is not in the file."""
        surfaces = self._read_surfaces()
        surface = surfaces.get(normalise_name(name))
        if surface is None:
            raise RuntimeError(f"no surface '{normalise_name(name)}' in {self.get_parameter('surfaces_file').value} "
                               f"(known: {', '.join(sorted(surfaces)) or 'none'})")
        return self._parse_surface(name, surface)

    def _lookup(self, target, source):
        try:
            return transform_to_matrix(self.tf_buffer.lookup_transform(
                target, source, Time(), timeout=RclpyDuration(seconds=1.0)))
        except Exception as exc:
            raise RuntimeError(f'no transform from {source} to {target}: {exc}')

    def _plan_surface(self, name):
        """Head poses that cover the named surface, and its corners in base_link for filtering."""
        height, corners_map = self._load_surface(name)
        map_frame = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value

        map_to_base = self._lookup(base_frame, map_frame)
        head = self._lookup(base_frame, self.get_parameter('head_frame').value)[:3, 3]

        def to_base(points_xy):
            points = np.column_stack([points_xy, np.full(len(points_xy), height), np.ones(len(points_xy))])
            return (points @ map_to_base.T)[:, :3]

        filter_corners = to_base(grow_polygon(corners_map, float(self.get_parameter('surface_margin_m').value)))
        outline = to_base(polygon_boundary(grow_polygon(
            corners_map, float(self.get_parameter('view_margin_m').value))))
        if inside_polygon(outline[:, :2], head[0], head[1]):
            raise RuntimeError(f"the head is above surface '{name}', so the robot is not beside it")
        surface_z = float(outline[:, 2].mean())
        pan_lo, pan_hi, tilt_lo, tilt_hi = view_ranges(
            outline, head, float(self.get_parameter('max_object_height_m').value))

        hfov = float(self.get_parameter('camera_hfov_rad').value)
        vfov = float(self.get_parameter('camera_vfov_rad').value)
        overlap = float(self.get_parameter('view_overlap').value)
        pan_limits = [float(v) for v in self.get_parameter('head_pan_limits_rad').value]
        tilt_limits = [float(v) for v in self.get_parameter('head_tilt_limits_rad').value]

        # The pan joint goes past +-pi, so a bearing can be reached as itself or 2*pi away. Use
        # whichever puts more of the surface within reach.
        shift = max((0.0, -2.0 * math.pi, 2.0 * math.pi), key=lambda s: (
            min(pan_hi + s, pan_limits[1] + hfov / 2) - max(pan_lo + s, pan_limits[0] - hfov / 2), -abs(s)))
        pan_lo, pan_hi = pan_lo + shift, pan_hi + shift
        if pan_lo < pan_limits[0] - hfov / 2 or pan_hi > pan_limits[1] + hfov / 2:
            self.get_logger().warning(f"Part of surface '{name}' is beyond the head's pan range, so it will "
                                      'not all be seen. Park the robot facing it more directly.')
        pans = sorted({round(float(np.clip(p, *pan_limits)), 3) for p in cover(pan_lo, pan_hi, hfov, overlap)})
        tilts = sorted({round(float(np.clip(t, *tilt_limits)), 3) for t in cover(tilt_lo, tilt_hi, vfov, overlap)})

        self.get_logger().info(
            f"Surface '{name}' (height {height:.2f} m, {surface_z:.2f} m in {base_frame}) spans pan "
            f'{pan_lo:.2f}..{pan_hi:.2f} and tilt {tilt_lo:.2f}..{tilt_hi:.2f} rad: pans {pans}, tilts {tilts}.')
        return pans, tilts, (filter_corners[:, :2], surface_z)

    def _on_surface(self, det, surface):
        corners, surface_z = surface
        bottom = det.centroid.z - det.bbox3d.size.z / 2.0
        return (inside_polygon(corners, det.centroid.x, det.centroid.y) and
                abs(bottom - surface_z) <= float(self.get_parameter('surface_height_tolerance_m').value))

    def _publish_surfaces(self):
        """Outline and name of every surface in surfaces_file; the one being scanned is green."""
        try:
            surfaces = self._read_surfaces()
            error = None
        except RuntimeError as exc:
            surfaces, error = {}, str(exc)
        stamp = self.get_clock().now().to_msg()
        frame_id = self.get_parameter('map_frame').value
        # Clear first, so surfaces removed from the file disappear.
        markers = [Marker(action=Marker.DELETEALL, ns='surfaces')]
        for i, name in enumerate(sorted(surfaces)):
            try:
                height, corners = self._parse_surface(name, surfaces[name])
            except RuntimeError as exc:
                error = str(exc)
                continue
            active = name == self.active_surface
            outline = Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns='surfaces', id=2 * i)
            outline.header.frame_id, outline.header.stamp = frame_id, stamp
            outline.pose.orientation.w = 1.0
            outline.scale.x = 0.03 if active else 0.015
            outline.color.r, outline.color.g, outline.color.b, outline.color.a = (
                (0.1, 1.0, 0.1, 1.0) if active else (0.0, 0.8, 1.0, 0.8))
            outline.points = [Point(x=float(x), y=float(y), z=height)
                              for x, y in np.vstack([corners, corners[:1]])]
            label = Marker(type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, ns='surfaces', id=2 * i + 1,
                           text=f'{name} ({height:.2f} m)')
            label.header = outline.header
            label.pose.position = Point(x=float(corners[:, 0].mean()), y=float(corners[:, 1].mean()),
                                        z=height + 0.1)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.08
            label.color = outline.color
            markers.extend([outline, label])
        # Log a problem with the file once, not on every refresh.
        if error != self.surfaces_error:
            if error:
                self.get_logger().warning(error)
            self.surfaces_error = error
        self.surfaces_pub.publish(MarkerArray(markers=markers))

    # ------------------------------------------------------------------ actions

    def _wait(self, future, timeout_s):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        return future.result() if done.wait(timeout_s) else None

    def _move_head(self, pan, tilt):
        """Move the head and wait for it; return an error string, or None on success."""
        self.get_logger().info(f'Look at pan={pan:.2f}, tilt={tilt:.2f}.')
        if self.dryrun:
            return None
        if not self.trajectory.wait_for_server(timeout_sec=5.0):
            return 'follow_joint_trajectory action server is not available'
        move_s = float(self.get_parameter('head_move_s').value)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(pan), float(tilt)],
            time_from_start=Duration(sec=int(move_s), nanosec=int((move_s % 1.0) * 1e9)))]
        handle = self._wait(self.trajectory.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            return f'head goal pan={pan:.2f} tilt={tilt:.2f} was rejected'
        result = self._wait(handle.get_result_async(), move_s + 10.0)
        if result is None:
            return f'head goal pan={pan:.2f} tilt={tilt:.2f} timed out'
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return f'head goal pan={pan:.2f} tilt={tilt:.2f} failed: {result.result.error_string}'
        return None

    def _detect_view(self):
        """Detections in the current view; raises RuntimeError if there is no frame or no reply."""
        time.sleep(float(self.get_parameter('head_settle_s').value))
        frame = self._wait_for_new_frame()
        if frame is None:
            raise RuntimeError('no new color and depth images from the head camera')
        request = DetectFloorObjects3D.Request()
        request.color_msg, request.depth_msg, request.info_msg = frame
        response = self._wait(self.detect.call_async(request), float(self.get_parameter('detect_timeout_s').value))
        if response is None:
            raise RuntimeError('detect_floor_objects_3d did not reply in time')
        return list(response.detections.detections)

    # ------------------------------------------------------------------ scan

    def _view_order(self, pans, tilts):
        # Start from whichever end of the pans is nearer the head, and snake through the tilts
        # at each pan, so the head never jumps back across the sweep.
        head = self._head_pose()
        if head is not None and abs(pans[-1] - head[0]) < abs(pans[0] - head[0]):
            pans = pans[::-1]
        views = []
        for pan in pans:
            views.extend((pan, tilt) for tilt in tilts)
            tilts = tilts[::-1]
        return views

    @staticmethod
    def _matches(det, target):
        return target in det.class_name.lower() or target in det.detector_class_name.lower()

    def _make_tracks(self, views):
        try:
            head = self._lookup(self.get_parameter('base_frame').value,
                                self.get_parameter('head_frame').value)[:3, 3]
        except RuntimeError as exc:
            self.get_logger().warning(f'{exc}, so detections will not be cross-checked between views.')
            head = None
        p = lambda name: float(self.get_parameter(name).value)
        single_view_min_score = p('single_view_min_score')
        return ObjectTracks(views, head, p('camera_hfov_rad'), p('camera_vfov_rad'), p('view_edge_margin_rad'),
                            p('merge_radius_m'), p('label_merge_radius_m'), int(p('confirm_views')),
                            single_view_min_score if single_view_min_score <= 1.0 else None)

    def _scan_cb(self, request, response):
        response.success = False
        if not self.scan_lock.acquire(blocking=False):
            response.message = 'a scan is already running'
            return response
        self.active_surface = normalise_name(request.surface_name) or None
        self._publish_surfaces()
        try:
            self._scan(request, response)
        finally:
            self.active_surface = None
            self._publish_surfaces()
            self.scan_lock.release()
        log = self.get_logger().info if response.success else self.get_logger().error
        log(response.message)
        return response

    def _scan(self, request, response):
        target = request.target_label.strip().lower()
        surface_name = normalise_name(request.surface_name)
        surface = None
        pans = [float(p) for p in self.get_parameter('scan_pans_rad').value]
        tilts = [float(t) for t in self.get_parameter('scan_tilts_rad').value]
        if surface_name:
            try:
                pans, tilts, surface = self._plan_surface(surface_name)
            except RuntimeError as exc:
                response.message = str(exc)
                return
        if len(request.pan_angles_rad):
            pans = [float(p) for p in request.pan_angles_rad]
        if len(request.tilt_angles_rad):
            tilts = [float(t) for t in request.tilt_angles_rad]
        if not pans or not tilts:
            response.message = 'no head poses to scan'
            return
        if not self.detect.wait_for_service(timeout_sec=5.0):
            response.message = f'{self.detect.srv_name} is not available'
            return

        start_head = self._head_pose()
        views = self._view_order(pans, tilts)
        self.get_logger().info(
            (f"Search for '{target}'" if target else 'Full sweep') +
            (f" of surface '{surface_name}'" if surface_name else '') + f' over {len(views)} views.')

        tracks = self._make_tracks(views)
        merged = []
        try:
            for index, (pan, tilt) in enumerate(views):
                err = self._move_head(pan, tilt)
                if err:
                    response.message = err
                    return
                detections = self._detect_view()
                response.views_scanned += 1
                lines = []
                for d in detections:
                    on = surface is None or self._on_surface(d, surface)
                    lines.append(f"'{d.class_name}' {d.score:.2f} at ({d.centroid.x:.2f}, {d.centroid.y:.2f}, "
                                 f"{d.centroid.z:.2f}){'' if on else ' [not on surface]'}")
                if surface is not None:
                    detections = [d for d in detections if self._on_surface(d, surface)]
                self.get_logger().info(f'View {index + 1}/{len(views)} (pan {pan:.2f}, tilt {tilt:.2f}): ' +
                                       ('; '.join(lines) or 'nothing'))
                tracks.add(index, detections)
                if target:
                    # Stop only for a match that other views agree on, so a one-off label does not end
                    # the search. Until then, keep going: the next view usually overlaps this one.
                    matches = [tracks.best(t) for t in tracks.tracks
                               if tracks.confirmed(t) and self._matches(tracks.best(t), target)]
                    if matches:
                        response.found = True
                        response.target = max(matches, key=lambda d: d.score)
                        break
        except RuntimeError as exc:
            response.message = str(exc)
            return
        finally:
            if start_head is not None and bool(self.get_parameter('restore_head').value):
                err = self._move_head(*start_head)
                if err:
                    self.get_logger().warning(f'Could not restore the head pose: {err}')
            for track in tracks.tracks:
                if tracks.confirmed(track):
                    merged.append(tracks.best(track))
                    self.get_logger().info(f'Object {tracks.describe(track)}.')
                else:
                    self.get_logger().info(f'Dropped {tracks.describe(track)}, so not confirmed.')
            response.detections.header.frame_id = merged[0].header.frame_id if merged else 'base_link'
            response.detections.header.stamp = self.get_clock().now().to_msg()
            response.detections.detections = merged

        response.success = True
        names = ', '.join(f"'{d.class_name}'" for d in merged) or 'none'
        where = f" on '{surface_name}'" if surface_name else ''
        if target:
            if response.found:
                p = response.target.centroid
                response.message = (f"found '{response.target.class_name}'{where} after {response.views_scanned} "
                                    f'views at ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) in {response.detections.header.frame_id}')
            else:
                response.message = f"'{target}' not found{where} in {response.views_scanned} views; saw {names}"
        else:
            response.message = f'{len(merged)} objects{where} in {response.views_scanned} views: {names}'


def main():
    rclpy.init()
    node = ScanSurfaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
