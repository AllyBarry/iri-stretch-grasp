'''
Tests for the 2D simulator. Run from the repository root:
    python3 -m unittest test.test_stretch_sim -v
'''
import json
import math
import tempfile
import unittest
from pathlib import Path

from stretch_sim.config import load_scenario, resolve
from stretch_sim.events import COMMAND, RESULT, VIOLATION, EventLog, JsonlSink, read_jsonl
from stretch_sim.geometry import segments_intersect, world_to_local
from stretch_sim.operations import OpStatus
from stretch_sim.geometry import Pose2D
from stretch_sim.script import ScriptRunner, fetch

# Any head pose; the BT no longer moves the head, but the sim still accepts head moves as an action.
SCAN_PAN, SCAN_TILT = -1.8, -0.8
from stretch_sim.simulator import KinematicSimulator
from stretch_sim.world import HELD, load_world

WORLD = load_world(resolve("apartment", "worlds"))


def make_sim(**overrides) -> KinematicSimulator:
    scenario = load_scenario("default") | {"perception": {"miss_prob": 0.0, "position_noise_m": 0.0}}
    scenario.update(overrides)
    return KinematicSimulator(WORLD, scenario)


def run(sim, op, limit=300.0):
    end = sim.time + limit
    while not op.done and sim.time < end:
        sim.step(0.05)
    return op


def drive_to(sim, location):
    sim.set_driver_mode("navigation")
    op = run(sim, sim.navigate_to_location(location))
    sim.set_driver_mode("position")
    return op


class WorldTest(unittest.TestCase):
    def test_approach_poses_put_the_location_on_the_arm_side(self):
        for loc in WORLD.locations.values():
            lx, ly = world_to_local(loc.approach, loc.rect.cx, loc.rect.cy)
            self.assertLess(ly, 0.0, loc.name)
            self.assertLess(abs(lx), 0.05, loc.name)

    def test_spawning_is_deterministic_per_seed(self):
        a, b, c = make_sim(seed=5), make_sim(seed=5), make_sim(seed=6)
        where = lambda s: {n: o.location for n, o in s.objects.items()}
        self.assertEqual(where(a), where(b))
        self.assertNotEqual(where(a), where(c))

    def test_place_pins_objects(self):
        sim = make_sim(spawn={"objects": "all", "place": {"Feta": "Pantry", "Basil": "Table"}})
        self.assertEqual(sim.objects["Feta"].location, "Pantry")
        self.assertEqual(sim.objects["Basil"].location, "Table")

    def test_objects_spawn_on_their_surface(self):
        sim = make_sim()
        for obj in sim.objects.values():
            self.assertTrue(WORLD.locations[obj.location].rect.contains(obj.x, obj.y), obj.name)

    def test_unknown_objects_are_rejected(self):
        with self.assertRaises(ValueError):
            make_sim(spawn={"place": {"Unicorn": "Table"}})


class NavigationTest(unittest.TestCase):
    def test_paths_to_every_location_avoid_walls(self):
        sim = make_sim()
        for loc in WORLD.locations.values():
            path = sim.planner.plan((WORLD.robot_start.x, WORLD.robot_start.y), (loc.approach.x, loc.approach.y))
            self.assertIsNotNone(path, loc.name)
            for a, b in zip(path, path[1:]):
                for w0, w1 in WORLD.walls:
                    self.assertFalse(segments_intersect(a, b, w0, w1), f"{loc.name}: {a}->{b} crosses a wall")

    def test_navigation_reaches_the_approach_pose(self):
        sim = make_sim()
        op = drive_to(sim, "Pantry")
        self.assertEqual(op.status, OpStatus.SUCCEEDED)
        goal, pose = WORLD.locations["Pantry"].approach, sim.robot.pose
        self.assertAlmostEqual(pose.x, goal.x, places=2)
        self.assertAlmostEqual(pose.y, goal.y, places=2)
        self.assertAlmostEqual(pose.yaw, goal.yaw, places=2)

    def test_navigation_in_position_mode_is_a_violation(self):
        sim = make_sim()
        op = sim.navigate_to_location("Table")
        self.assertEqual(op.status, OpStatus.FAILED)
        self.assertEqual(len(sim.violations), 1)
        self.assertIn(VIOLATION, [e.kind for e in sim.log.events])

    def test_switching_mode_stops_the_base(self):
        sim = make_sim()
        sim.set_driver_mode("navigation")
        op = sim.navigate_to_location("Fridge")
        sim.step(1.0)
        sim.set_driver_mode("position")
        self.assertEqual(op.status, OpStatus.CANCELED)


class PerceptionAndManipulationTest(unittest.TestCase):
    def _scan(self, sim):
        return sim.detect()

    def test_scan_covers_the_surface_on_the_arm_side_only(self):
        sim = make_sim()
        drive_to(sim, "Fridge")
        seen = self._scan(sim)
        self.assertTrue(seen)
        self.assertTrue(all(sim.objects[d.true_object].location == "Fridge" for d in seen))
        sim.robot.pose.yaw += math.pi                   # the arm now faces away from the Fridge
        self.assertEqual(self._scan(sim), [])
        sim.robot.pose = Pose2D(5.0, 1.0, 0.0)          # open floor, nothing within reach
        self.assertEqual(self._scan(sim), [])
        self.assertIsNone(sim.last_scan["surface"])

    def test_scan_covers_the_whole_surface(self):
        sim = make_sim()
        drive_to(sim, "Table")
        seen = {d.true_object for d in self._scan(sim)}
        on_table = {n for n, o in sim.objects.items() if o.location == "Table"}
        self.assertEqual(seen, on_table)
        self.assertEqual(sim.last_scan["surface"], "Table")

    def test_success_rate_misses_objects_individually(self):
        sim = make_sim(seed=3)
        drive_to(sim, "Table")
        sim.set_success_rate(0.75)
        on_table = sum(o.location == "Table" for o in sim.objects.values())
        seen = sum(len(sim.detect()) for _ in range(100))
        self.assertAlmostEqual(seen / (100 * on_table), 0.75, delta=0.04)
        self.assertTrue(sim.last_scan["missed"] or seen == 100 * on_table)

    def test_grasp_holds_the_object_and_sets_the_aperture(self):
        sim = make_sim()
        drive_to(sim, "Fridge")
        det = max(self._scan(sim), key=lambda d: d.centroid[1])   # nearest to the robot
        op = run(sim, sim.grasp(det.centroid))
        self.assertEqual(op.status, OpStatus.SUCCEEDED)
        self.assertEqual(op.result["held"], det.true_object)
        self.assertEqual(sim.objects[det.true_object].location, HELD)
        aperture = 2 * sim.gripper_finger_rad() * sim.geom.finger_length_m
        self.assertGreater(aperture, 0.01)

    def _grasp_after_base_moved(self, shift_m, check_label=False):
        sim = make_sim()
        drive_to(sim, "Fridge")
        det = max(self._scan(sim), key=lambda d: d.centroid[1])
        yaw = sim.robot.pose.yaw
        sim.robot.pose.x += shift_m * math.cos(yaw)                 # the base moved after the scan
        sim.robot.pose.y += shift_m * math.sin(yaw)
        return sim, det, run(sim, sim.grasp(det.centroid, expected_label=det.label if check_label else ""))

    def test_stale_detection_closes_on_nothing(self):
        # Half the slot spacing: the gripper lands between two objects.
        sim, _, op = self._grasp_after_base_moved(0.075)
        self.assertEqual(op.status, OpStatus.SUCCEEDED)
        self.assertIsNone(op.result["held"])
        self.assertLess(sim.robot.aperture, 0.01)

    def test_stale_detection_can_grab_the_wrong_object(self):
        # A whole slot: the gripper lands on the neighbouring object.
        sim, det, op = self._grasp_after_base_moved(0.15)
        self.assertIsNotNone(op.result["held"])
        self.assertNotEqual(op.result["held"], det.true_object)

    def test_wrong_object_is_flagged_against_the_expected_label(self):
        _, _, op = self._grasp_after_base_moved(0.15)
        self.assertNotIn("wrong_object", op.result)                 # no expected label, no check
        _, det, op = self._grasp_after_base_moved(0.15, check_label=True)
        self.assertNotEqual(op.result["held"], det.true_object)
        self.assertTrue(op.result["wrong_object"])
        self.assertIn("not the expected", op.message)

    def test_expected_label_matches_like_the_grasp_servers(self):
        sim = make_sim(detector_labels={"Olive Oil": "olive_oil"}, spawn={"objects": "all", "place": {"Olive Oil": "Fridge"}})
        drive_to(sim, "Fridge")
        det = next(d for d in self._scan(sim) if d.true_object == "Olive Oil")
        op = run(sim, sim.grasp(det.centroid, expected_label="Olive Oil"))
        self.assertEqual(op.result["held"], "Olive Oil")
        self.assertFalse(op.result["wrong_object"])

    def test_top_grasp_objects_are_detected_and_grasped_from_above(self):
        sim = make_sim(grasp_directions={"Feta": "top"}, spawn={"objects": "all", "place": {"Feta": "Fridge"}})
        drive_to(sim, "Fridge")
        dets = self._scan(sim)
        feta = next(d for d in dets if d.true_object == "Feta")
        self.assertEqual(feta.grasp_direction, "top")
        self.assertTrue(all(d.grasp_direction == "front" for d in dets if d is not feta))
        op = run(sim, sim.grasp(feta.centroid, direction="top", expected_label=feta.label))
        self.assertEqual(op.status, OpStatus.SUCCEEDED)
        self.assertEqual(op.result["held"], "Feta")
        phases = [e.message for e in sim.log.events if e.op_id == op.id and e.kind != COMMAND]
        self.assertTrue(any("lift above" in m for m in phases), phases)
        self.assertTrue(any("lower" in m for m in phases), phases)

    def test_bad_grasp_directions_are_rejected(self):
        with self.assertRaises(ValueError):
            make_sim(grasp_directions={"Feta": "sideways"})
        op = make_sim().grasp((0.0, -0.5, 0.8), direction="sideways")
        self.assertEqual(op.status, OpStatus.FAILED)

    def test_out_of_reach_grasp_is_rejected(self):
        sim = make_sim()
        op = sim.grasp((0.0, -1.5, 0.8))
        self.assertEqual(op.status, OpStatus.FAILED)
        self.assertIn("reach", op.message)

    def test_handover_without_a_hand_keeps_the_object(self):
        sim = make_sim(handover={"no_hand_prob": 1.0, "wait_s": 0.5})
        drive_to(sim, "Fridge")
        det = max(self._scan(sim), key=lambda d: d.centroid[1])
        run(sim, sim.grasp(det.centroid))
        drive_to(sim, "Human")
        op = run(sim, sim.handover())
        self.assertFalse(op.result["handed_over"])
        self.assertEqual(sim.robot.held, det.true_object)


class ScriptAndLogTest(unittest.TestCase):
    def test_fetch_delivers_the_object_and_logs_every_command(self):
        sim = make_sim()
        drive_to(sim, "Fridge")
        target = max(self._visible(sim), key=lambda d: d.centroid[1]).true_object
        sim.reset(sim.seed)

        runner = ScriptRunner(sim)
        runner.start("fetch", fetch(sim, target))
        while runner.active and sim.time < 600:
            sim.step(0.05)
            runner.update()

        self.assertEqual(sim.objects[target].location, "Human")
        self.assertEqual(sim.violations, [])
        commands = {e.op_id for e in sim.log.events if e.kind == COMMAND and e.op_id is not None}
        results = {e.op_id for e in sim.log.events if e.kind == RESULT and e.op_id is not None}
        self.assertEqual(commands, results)

    def _visible(self, sim):
        return sim.detect()

    def test_jsonl_log_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog()
            sink = JsonlSink(Path(tmp) / "events.jsonl")
            log.add_sink(sink)
            sim = KinematicSimulator(WORLD, load_scenario("default"), log=log)
            sim.navigate_to_location("Table")
            sink.close()
            events = read_jsonl(sink.path)
            self.assertEqual([e.format() for e in events], [e.format() for e in sim.log.events])

    def test_location_poses_export_matches_the_bt_format(self):
        poses = WORLD.location_poses()
        self.assertEqual(set(poses), set(WORLD.locations))
        for pose in poses.values():
            self.assertEqual(set(pose), {"x", "y", "yaw"})
        json.dumps(poses)


class ShuffleTest(unittest.TestCase):
    def test_shuffle_moves_objects_to_other_surfaces(self):
        sim = make_sim()
        before = {n: o.location for n, o in sim.objects.items()}
        moves = sim.shuffle_objects(4, seed=1)
        self.assertEqual(len(moves), 4)
        for name, source, target in moves:
            self.assertEqual(before[name], source)
            self.assertNotEqual(source, target)
            self.assertEqual(sim.objects[name].location, target)
            self.assertTrue(WORLD.locations[target].rect.contains(sim.objects[name].x, sim.objects[name].y))
        unmoved = set(before) - {m[0] for m in moves}
        self.assertTrue(all(sim.objects[n].location == before[n] for n in unmoved))
        self.assertIn("shuffled 4 object(s)", sim.log.events[-1].message)

    def test_shuffle_leaves_delivered_and_held_objects(self):
        sim = make_sim()
        sim.move_object("Salt", "Human")
        sim.objects["Basil"].location = "Robot"
        sim.robot.held = "Basil"
        moved = {m[0] for m in sim.shuffle_objects(50, seed=2)}
        self.assertNotIn("Salt", moved)
        self.assertNotIn("Basil", moved)

    def test_shuffle_command_validates_count(self):
        from stretch_sim.controller import CommandError, SimController
        controller = SimController(make_sim())
        self.assertIn("moved", controller.execute({"cmd": "shuffle", "count": 2}))
        for bad in (0, "many"):
            with self.assertRaises(CommandError):
                controller.execute({"cmd": "shuffle", "count": bad})


class SuccessRateTest(unittest.TestCase):
    def _fetch_steps(self, sim):
        '''Runs one of each action from the Fridge and returns their operations.'''
        sim.set_driver_mode("navigation")
        nav = run(sim, sim.navigate_to_location("Fridge"))
        sim.robot.pose = WORLD.locations["Fridge"].approach.copy()     # in case navigation was made to fail
        sim.set_driver_mode("position")
        head = run(sim, sim.move_head(SCAN_PAN, SCAN_TILT, 1.0))
        sim.robot.head_pan, sim.robot.head_tilt = SCAN_PAN, SCAN_TILT
        dets = sim.detect()
        return nav, head, dets

    def test_zero_success_rate_fails_every_action(self):
        sim = make_sim()
        sim.set_success_rate(0.0)
        nav, head, dets = self._fetch_steps(sim)
        self.assertEqual(nav.status, OpStatus.FAILED)
        self.assertEqual(head.status, OpStatus.FAILED)
        self.assertEqual(dets, [])
        target = next(o for o in sim.objects.values() if o.location == "Fridge")
        from stretch_sim.geometry import world_to_local
        lx, ly = world_to_local(sim.robot.pose, target.x, target.y)
        grasp = run(sim, sim.grasp((lx, ly, target.z)))
        self.assertIsNone(grasp.result.get("held"))
        self.assertTrue(grasp.result.get("injected_failure"))

    def test_full_success_rate_never_injects_failures(self):
        sim = make_sim()
        sim.set_success_rate(1.0)
        nav, head, dets = self._fetch_steps(sim)
        self.assertEqual((nav.status, head.status), (OpStatus.SUCCEEDED, OpStatus.SUCCEEDED))
        self.assertTrue(dets)

    def test_rate_matches_observed_successes(self):
        sim = make_sim(seed=1)
        sim.set_success_rate(0.9)
        sim.set_driver_mode("position")
        ok = 0
        for i in range(500):
            op = run(sim, sim.move_head(-1.0 if i % 2 else 0.0, -0.5, 0.2))
            ok += op.status == OpStatus.SUCCEEDED
        self.assertAlmostEqual(ok / 500, 0.9, delta=0.04)

    def test_scenario_overrides_and_clearing_them(self):
        scenario = load_scenario("default")
        scenario["robot"] = {"success_rate": 0.8}
        scenario["grasp"] = {"fail_prob": 0.5}
        sim = KinematicSimulator(WORLD, scenario)
        self.assertAlmostEqual(sim.fail_prob("navigate"), 0.2)
        self.assertAlmostEqual(sim.fail_prob("grasp"), 0.5)
        sim.set_success_rate(0.9)
        self.assertAlmostEqual(sim.fail_prob("grasp"), 0.1)


class BtRunnerTest(unittest.TestCase):
    def test_trees_and_recipes_are_discovered_without_importing_bt_cooking(self):
        import sys
        from stretch_sim.bt_runner import discover_recipes, discover_trees
        self.assertIn("test_sequence", discover_trees())
        self.assertIn("greek_salad", discover_recipes())
        self.assertNotIn("bt_cooking.stretch_bt", sys.modules)

    def _fake_bt(self, tmp, body):
        script = Path(tmp) / "fake_bt.py"
        script.write_text(body)
        import sys
        return [sys.executable, str(script)]

    def test_bt_process_gets_the_tree_recipe_and_sim_location_poses(self):
        from stretch_sim.bt_runner import BtProcess
        from stretch_sim.events import BT
        with tempfile.TemporaryDirectory() as tmp:
            command = self._fake_bt(tmp, (
                "import json, sys\n"
                "args = sys.argv[1:]\n"
                "poses = json.load(open(args[args.index('--location-poses') + 1]))\n"
                "print('tree', args[args.index('--tree') + 1], 'recipe', args[args.index('--recipe') + 1])\n"
                "print('locations', ','.join(sorted(poses)))\n"))
            sim = make_sim()
            bt = BtProcess(sim, Path(tmp), command)
            bt.start("main_demo_loop", "greek_salad")
            self.assertEqual(bt.wait(10), 0)
            for _ in range(100):           # the output thread may still be emitting
                lines = [e.message for e in sim.log.events if e.kind == BT]
                if len(lines) == 2:
                    break
                import time; time.sleep(0.02)
            self.assertIn("tree main_demo_loop recipe greek_salad", lines)
            self.assertTrue(any(l.startswith("locations") and "Fridge" in l and "Human" in l for l in lines))
            self.assertEqual(bt.status()["returncode"], 0)

    def test_bt_process_can_be_stopped(self):
        from stretch_sim.bt_runner import BtProcess
        with tempfile.TemporaryDirectory() as tmp:
            command = self._fake_bt(tmp, "import time\nprint('ticking', flush=True)\ntime.sleep(60)\n")
            bt = BtProcess(make_sim(), Path(tmp), command)
            bt.start("main_demo_loop")
            self.assertTrue(bt.running)
            bt.stop(timeout_s=5)
            self.assertFalse(bt.running)

    def test_memory_reset_needs_a_running_tree(self):
        from stretch_sim.controller import CommandError, SimController
        with self.assertRaises(CommandError) as ctx:
            SimController(make_sim()).execute({"cmd": "reset_memory"})
        self.assertIn("no behaviour tree is running", str(ctx.exception))

    def test_controller_explains_why_trees_cannot_run_without_ros(self):
        from stretch_sim.controller import CommandError, SimController
        controller = SimController(make_sim())
        with self.assertRaises(CommandError) as ctx:
            controller.execute({"cmd": "bt_start", "tree": "main_demo_loop"})
        self.assertIn("ROS", str(ctx.exception))


def bt_payload(**values):
    base = {"recipe/name": "greek_salad", "target_object": None, "target_location": None, "holding_object": None,
            "confidence/Olives/Pantry": 0.5, "confidence/Olives/Fridge": 0.4, "confidence/Olives/Table": 0.1,
            "confidence/Olives/Human": 0.0, "confidence/human_reachable/Olives": 0.5,
            "required/Olives": True, "human_reachable/Olives": False}
    return {"tree": "main_demo_loop", "status": "RUNNING", "tip": "NavigateToPlaceNode", "values": base | values}


class BeliefTest(unittest.TestCase):
    def test_belief_changes_are_logged(self):
        from stretch_sim.events import BELIEF
        sim = make_sim()
        sim.set_bt_state(bt_payload())
        sim.set_bt_state(bt_payload(**{"target_object": "Olives", "target_location": "Pantry",
                                       "confidence/Olives/Pantry": 0.09, "confidence/Olives/Fridge": 0.73}))
        messages = [e.message for e in sim.log.events if e.kind == BELIEF]
        self.assertTrue(any("target_object: None -> Olives" in m for m in messages))
        self.assertTrue(any(m.startswith("Olives:") and "Pantry 50% -> 9%" in m for m in messages))
        self.assertFalse(any("human_reachable" in m for m in messages))

    def test_controller_summarises_beliefs(self):
        from stretch_sim.controller import SimController
        controller = SimController(make_sim())
        self.assertIsNone(controller.state()["beliefs"])
        controller.sim.set_bt_state(bt_payload(target_object="Olives", target_location="Pantry"))
        beliefs = controller.state()["beliefs"]
        self.assertEqual(beliefs["locations"], ["Fridge", "Table", "Pantry", "Human"])
        self.assertEqual(beliefs["probabilities"]["Olives"]["Pantry"], 0.5)
        self.assertNotIn("human_reachable", beliefs["probabilities"])
        self.assertEqual(beliefs["required"], ["Olives"])
        self.assertEqual((beliefs["target_object"], beliefs["tip"]), ("Olives", "NavigateToPlaceNode"))


class WebViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stretch_sim.controller import SimController
        from stretch_sim.web.server import WebViewer
        cls.controller = SimController(make_sim(), speed=8.0, scenario_name="default")
        cls.viewer = WebViewer(cls.controller, port=0)
        cls.viewer.start_background()
        cls.controller.start()
        cls.base = cls.viewer.url.rstrip("/")

    @classmethod
    def tearDownClass(cls):
        cls.viewer.shutdown()
        cls.controller.stop()

    def _get(self, path):
        import urllib.request
        with urllib.request.urlopen(self.base + path, timeout=5) as res:
            return res.status, res.headers.get("Content-Type"), res.read()

    def _command(self, cmd):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + "/api/command", data=json.dumps(cmd).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_world_and_page(self):
        _, _, body = self._get("/api/world")
        world = json.loads(body)
        self.assertEqual({l["name"] for l in world["locations"]}, set(WORLD.locations))
        status, ctype, page = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"app.js", page)

    def test_commands_change_the_simulation(self):
        status, body = self._command({"cmd": "move_object", "name": "Basil", "location": "Pantry"})
        self.assertEqual((status, body["ok"]), (200, True))
        self.assertEqual(self.controller.sim.location_of("Basil"), "Pantry")
        status, body = self._command({"cmd": "goto", "location": "Pantry"})
        self.assertTrue(body["ok"])
        self.assertEqual(self.controller.sim.robot.driver_mode, "navigation")

    def test_bad_commands_are_reported(self):
        for cmd in ({"cmd": "fly"}, {"cmd": "goto", "location": "Moon"}, {"cmd": "move_object", "name": "Unicorn", "location": "Pantry"}):
            status, body = self._command(cmd)
            self.assertEqual(status, 400, cmd)
            self.assertFalse(body["ok"])

    def test_stream_sends_state_and_events(self):
        import urllib.request
        with urllib.request.urlopen(self.base + "/api/stream?since=0", timeout=5) as res:
            line = res.readline().decode()
        self.assertTrue(line.startswith("data: "))
        payload = json.loads(line[len("data: "):])
        self.assertIn("robot", payload["state"])
        self.assertTrue(payload["events"])

    def test_static_files_cannot_escape_their_folder(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError):
            self._get("/../server.py")


class RenderTest(unittest.TestCase):
    def test_png_snapshot(self):
        from stretch_sim.render_pil import render_png
        with tempfile.TemporaryDirectory() as tmp:
            path = render_png(make_sim(), Path(tmp) / "snap.png")
            self.assertGreater(path.stat().st_size, 1000)


class RobotLinkTest(unittest.TestCase):
    '''Robot mode: the BT runs on the real robot and the sim mirrors the state it publishes.'''
    PLACES = """frame_id: map
places:
  pantry: {x: 0.9671, y: -1.0007, yaw: 2.8016}
  table: {x: -0.6876, y: -0.2241, yaw: -1.8897}
  handover: {x: -0.5841, y: 0.8148, yaw: 2.8282}
  home: {x: -0.0112, y: -0.1156, yaw: 1.1706}
"""

    def test_location_poses_come_from_the_robots_places(self):
        from stretch_sim.robot_link import location_poses_from_places, read_places
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "places.yaml"
            path.write_text(self.PLACES)
            places = read_places(str(path))
        poses, missing = location_poses_from_places(places, list(WORLD.locations))
        self.assertEqual(poses["Pantry"], {"x": 0.9671, "y": -1.0007, "yaw": 2.8016})
        self.assertEqual(poses["Human"], places["handover"])
        self.assertEqual(missing, ["Fridge (no place 'fridge')"])
        poses, missing = location_poses_from_places(places, list(WORLD.locations), {"Fridge": "home"})
        self.assertEqual(poses["Fridge"], places["home"])
        self.assertEqual(missing, [])

    def _state(self, tip, **values):
        base = {"target_object": None, "target_location": None, "holding_object": None, "scan/objects": None}
        return {"tree": "main_demo_loop", "status": "RUNNING", "tip": tip, "values": base | values}

    def _settle(self, sim):
        for op in list(sim.ops):
            run(sim, op)

    def test_mirror_follows_a_fetch(self):
        from stretch_sim.robot_link import RobotMirror
        sim = make_sim(spawn={"objects": "all", "place": {"Feta": "Fridge"}})
        mirror = RobotMirror(sim)
        go = dict(target_object="Feta", target_location="Pantry")

        mirror.on_bt_state(self._state("NavigateToPlaceNode", **go))
        self._settle(sim)
        self.assertEqual(sim.world.location_at_pose(sim.robot.pose), "Pantry")

        # The real robot found the feta on the Pantry, where the sim did not have it.
        seen = [{"class_name": "feta", "detector_class_name": "feta", "score": 0.9, "grasp_direction": "top"}]
        mirror.on_bt_state(self._state("SearchForObjectNode", **go, **{"scan/objects": seen}))
        self.assertEqual(sim.objects["Feta"].location, "Pantry")
        self.assertEqual([d.true_object for d in sim.last_detections], ["Feta"])

        mirror.on_bt_state(self._state("GraspObjectNode", **go, target_detection=seen[0], **{"scan/objects": seen}))
        grasp = next(op for op in sim.ops if op.name == "grasp")
        self.assertEqual(grasp.args["direction"], "top")
        self._settle(sim)
        mirror.on_bt_state(self._state("NavigateToHumanNode", **go, holding_object="Feta", **{"scan/objects": seen}))
        self.assertEqual(sim.robot.held, "Feta")
        self._settle(sim)

        mirror.on_bt_state(self._state("HandoverNode", **go, holding_object="Feta", **{"scan/objects": seen}))
        self.assertTrue(any(op.name == "handover" for op in sim.ops))
        mirror.on_bt_state(self._state("DecideTargetObjectNode", **go, **{"scan/objects": seen, "confidence/Feta/Human": 1.0}))
        self.assertIsNone(sim.robot.held)
        self.assertEqual(sim.objects["Feta"].location, "Human")

    def test_mirror_corrects_the_gripper_to_the_real_robot(self):
        from stretch_sim.robot_link import RobotMirror
        sim = make_sim(spawn={"objects": "all", "place": {"Feta": "Fridge"}})
        mirror = RobotMirror(sim)
        # The real grasp succeeded, though the mirror never animated one.
        mirror.on_bt_state(self._state("IsHoldingObjectNode", target_object="Feta", target_location="Fridge",
                                       holding_object="Feta"))
        self.assertEqual(sim.robot.held, "Feta")
        self.assertEqual(sim.objects["Feta"].location, HELD)
        # Then the grasp check found the gripper empty: the feta goes back where it was taken from.
        mirror.on_bt_state(self._state("GraspObjectNode", target_object="Feta", target_location="Fridge"))
        self.assertIsNone(sim.robot.held)
        self.assertEqual(sim.objects["Feta"].location, "Fridge")

    def test_robot_mode_refuses_manual_robot_commands(self):
        from stretch_sim.controller import CommandError, SimController
        controller = SimController(make_sim(), robot_mode=True)
        for cmd in ({"cmd": "goto", "location": "Table"}, {"cmd": "detect"}, {"cmd": "handover"}):
            with self.assertRaises(CommandError):
                controller.execute(cmd)
        self.assertIn("moved", controller.execute({"cmd": "shuffle", "count": 1}))
        self.assertTrue(controller.state()["robot_mode"])


if __name__ == "__main__":
    unittest.main()
