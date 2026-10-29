#!/usr/bin/env python3
"""define_surface.py: record a surface (desk, table) for scan_surface by clicking its corners in RViz.

With the robot localised, set RViz's fixed frame to map, choose the "Publish Point" tool and click
each corner of the surface. Clicking on the camera point cloud also gives the surface height;
clicking on the map gives height 0, so pass height then.

  ros2 run stretch_floor_perception define_surface.py --ros-args -p name:=desk
  ros2 run stretch_floor_perception define_surface.py --ros-args -p name:=desk -p height:=0.74

In the terminal: Enter saves, 'u' removes the last corner, 'q' quits without saving. The corners
are sorted around their centre, so they can be clicked in any order (fine for convex surfaces).
The surface is written to surfaces_file under name, replacing an existing one with that name.
"""
import os
import threading

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

from stretch_floor_perception.scan_surface import normalise_name

# Clicks below this are on the map rather than the surface, so they say nothing about its height.
MIN_CLICKED_HEIGHT_M = 0.2


class DefineSurface(Node):
    def __init__(self):
        super().__init__('define_surface')
        self.name = normalise_name(self.declare_parameter('name', '').value)
        self.height = float(self.declare_parameter('height', 0.0).value)   # <= 0: median of the clicks
        self.surfaces_file = self.declare_parameter('surfaces_file', os.path.expanduser(
            '~/stretch_user/maps/embrace_apartment_map.surfaces.yaml')).value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        topic = self.declare_parameter('clicked_point_topic', '/clicked_point').value

        self.lock = threading.Lock()
        self.points = []   # [x, y, z] in the map frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(
            Marker, '~/polygon', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(PointStamped, topic, self._clicked_cb, 10)
        self.get_logger().info(f"Click the corners of '{self.name}' in RViz (Publish Point, on {topic}).")

    def _clicked_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, Time(),
                                                            timeout=Duration(seconds=1.0))
            except Exception as exc:
                self.get_logger().error(f'Ignoring click in {msg.header.frame_id}: {exc}')
                return
            msg = do_transform_point(msg, transform)
        with self.lock:
            self.points.append([msg.point.x, msg.point.y, msg.point.z])
            n = len(self.points)
        print(f'corner {n}: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})')
        self.publish_polygon()

    def corners(self):
        """Clicked corners as Nx2, sorted around their centre."""
        with self.lock:
            xy = np.array([p[:2] for p in self.points], dtype=float).reshape(-1, 2)
        if len(xy) < 3:
            return xy
        angles = np.arctan2(xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean())
        return xy[np.argsort(angles)]

    def surface_height(self):
        if self.height > 0.0:
            return self.height
        with self.lock:
            z = [p[2] for p in self.points if p[2] > MIN_CLICKED_HEIGHT_M]
        return float(np.median(z)) if z else None

    def undo(self):
        with self.lock:
            if self.points:
                self.points.pop()
            n = len(self.points)
        print(f'{n} corners left')
        self.publish_polygon()

    def publish_polygon(self):
        corners = self.corners()
        height = self.surface_height() or 0.0
        marker = Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns='define_surface', id=0)
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        marker.color.r, marker.color.g, marker.color.a = 1.0, 0.6, 1.0
        closed = np.vstack([corners, corners[:1]]) if len(corners) >= 3 else corners
        marker.points = [Point(x=float(x), y=float(y), z=height) for x, y in closed]
        self.marker_pub.publish(marker)

    def save(self):
        """Write the surface; return an error string, or None on success."""
        corners = self.corners()
        if len(corners) < 3:
            return f'need at least 3 corners, have {len(corners)}'
        height = self.surface_height()
        if height is None:
            return ('no clicks on the surface itself, so its height is unknown: click on the point cloud, '
                    'or rerun with -p height:=<metres>')
        data = {}
        if os.path.exists(self.surfaces_file):
            with open(self.surfaces_file) as f:
                data = yaml.safe_load(f) or {}
        surfaces = data.get('surfaces') or {}
        replaced = self.name in surfaces
        surfaces[self.name] = {'height': round(height, 3),
                               'polygon': [[round(float(x), 3), round(float(y), 3)] for x, y in corners]}
        tmp = self.surfaces_file + '.tmp'
        with open(tmp, 'w') as f:
            yaml.safe_dump({'frame_id': self.map_frame, 'surfaces': surfaces}, f, sort_keys=True)
        os.replace(tmp, self.surfaces_file)
        print(f"{'Replaced' if replaced else 'Saved'} '{self.name}': height {height:.3f} m, "
              f'{len(corners)} corners, in {self.surfaces_file}')
        return None


def main():
    rclpy.init()
    node = DefineSurface()
    if not node.name:
        node.get_logger().error('Pass the surface name: --ros-args -p name:=desk')
        node.destroy_node()
        rclpy.try_shutdown()
        return
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        while rclpy.ok():
            answer = input("Enter to save, 'u' to undo the last corner, 'q' to quit: ").strip().lower()
            if answer == 'q':
                break
            if answer == 'u':
                node.undo()
                continue
            err = node.save()
            if err is None:
                break
            print(err)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
