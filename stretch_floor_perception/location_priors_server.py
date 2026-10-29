#!/usr/bin/env python3
"""Answers "where is X likely to be?" from config/locations/location_priors.json.

  ros2 run stretch_floor_perception location_priors_server.py
  ros2 service call /object_locations stretch_floor_perception/srv/ObjectLocations "{object: 'olive oil'}"

Names are matched ignoring case, spaces, underscores and a plural "s"/"es", so the detector's "olive oil",
the verifier's "olive_oil" and the recipe's "Olive Oil" all work. The file is re-read when it changes on disk.
"""
import json
import os
import re

import rclpy
from rclpy.node import Node

from stretch_floor_perception.srv import ObjectLocations

CATEGORY_LABELS = {"Ingredients": "ingredient", "Tools": "tool"}


def default_priors_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("stretch_floor_perception"),
                            "config", "locations", "location_priors.json")
    except Exception:  # running from the source tree
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "config", "locations", "location_priors.json")


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


class Priors:
    """location_priors.json, indexed by normalised name, reloaded when the file changes."""

    def __init__(self, path):
        self.path, self.mtime, self.index = path, None, {}

    def _load(self):
        mtime = os.path.getmtime(self.path)
        if mtime == self.mtime:
            return
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        index = {}
        for category, objects in data.items():
            for obj, probs in objects.items():
                index[norm(obj)] = (obj, CATEGORY_LABELS.get(category, category.lower()), probs)
        self.index, self.mtime = index, mtime

    def lookup(self, name):
        self._load()
        key = norm(name)
        for k in (key, key[:-2] if key.endswith("es") else None, key[:-1] if key.endswith("s") else None, key + "s"):
            if k and k in self.index:
                return self.index[k]
        return None

    def names(self):
        self._load()
        return sorted(v[0] for v in self.index.values())


class LocationPriorsServer(Node):
    def __init__(self):
        super().__init__("location_priors_server")
        self.declare_parameter("priors_file", default_priors_file())
        self.declare_parameter("service_name", "/object_locations")
        path = os.path.abspath(self.get_parameter("priors_file").value)
        self.priors = Priors(path)
        known = self.priors.names()  # fail at startup, not on the first call, if the file is missing or broken
        name = self.get_parameter("service_name").value
        self.create_service(ObjectLocations, name, self.on_request)
        self.get_logger().info(f"{name} ready: {len(known)} objects from {path}")

    def on_request(self, req, res):
        try:
            hit = self.priors.lookup(req.object)
        except (OSError, ValueError) as e:
            res.found, res.message = False, f"cannot read priors file: {e}"
            return res
        if hit is None:
            res.found = False
            res.message = f"no location prior for '{req.object}'. Known: {', '.join(self.priors.names())}"
            self.get_logger().info(f"'{req.object}' -> unknown")
            return res
        obj, category, probs = hit
        ranked = sorted(((loc, float(p)) for loc, p in probs.items() if p > 0), key=lambda x: -x[1])
        res.found, res.object, res.category = True, obj, category
        res.locations = [loc for loc, _ in ranked]
        res.probabilities = [p for _, p in ranked]
        res.message = ", ".join(f"{loc} {p:.2f}" for loc, p in ranked)
        self.get_logger().info(f"'{req.object}' -> {obj}: {res.message}")
        return res


def main():
    rclpy.init()
    node = LocationPriorsServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
