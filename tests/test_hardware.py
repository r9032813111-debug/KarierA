from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.robot_controller import RobotPose, WebTargetController
from backend.path_planner import plan_work_route, segment_is_safe
from backend.vision_runtime import VisionRuntime, _aruco_dictionary, _detect, _inward_polygon, _marker_top_heading, _order_corner_records, _records


class FakeConnection:
    def __init__(self) -> None:
        self.writes: list[str] = []

    @property
    def in_waiting(self) -> int:
        return 0

    def write(self, data: bytes) -> int:
        self.writes.append(data.decode("ascii").strip())
        return len(data)

    def close(self) -> None:
        pass


class VisionGeometryTests(unittest.TestCase):
    def test_detects_and_orders_four_aruco_vertices(self) -> None:
        dictionary = _aruco_dictionary("DICT_4X4_100")
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        placements = {12: (80, 70), 18: (1080, 75), 23: (1090, 560), 31: (75, 555)}
        for marker_id, (x, y) in placements.items():
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
            frame[y:y + 100, x:x + 100] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        corners, ids = _detect(frame, dictionary)
        records = _records(corners, ids)
        self.assertEqual(set(records), set(placements))
        ordered = _order_corner_records(records)
        self.assertEqual([ordered[name][0] for name in ("top_left", "top_right", "bottom_right", "bottom_left")], [12, 18, 23, 31])
        polygon = _inward_polygon(ordered)
        self.assertEqual(len(polygon), 4)
        self.assertGreater(cv2.contourArea(np.asarray(polygon, dtype=np.float32)), 300_000)

    def test_does_not_guess_vertices_before_operator_assignment(self) -> None:
        runtime = VisionRuntime({
            "hardware_enabled": False,
            "aruco_dictionary": "DICT_4X4_100",
            "controlled_agent_id": 10,
        })
        records = {
            marker_id: np.asarray(points, dtype=np.float32)
            for marker_id, points in {
                1: [(0, 0), (10, 0), (10, 10), (0, 10)],
                2: [(90, 0), (100, 0), (100, 10), (90, 10)],
                3: [(90, 90), (100, 90), (100, 100), (90, 100)],
                4: [(0, 90), (10, 90), (10, 100), (0, 100)],
            }.items()
        }
        runtime._marker_roles = {}
        self.assertEqual(runtime._selected_vertex_ids(records), [])
        runtime._marker_roles = {1: "vertex", 2: "vertex", 3: "vertex", 4: "vertex"}
        self.assertEqual(set(runtime._selected_vertex_ids(records)), {1, 2, 3, 4})

    def test_holds_lost_marker_geometry_for_one_second(self) -> None:
        runtime = VisionRuntime({"hardware_enabled": False, "marker_hold_seconds": 1.0})
        corners = np.asarray([(10, 10), (20, 10), (20, 20), (10, 20)], dtype=np.float32)
        current, held = runtime._records_with_hold({7: corners}, 10.0)
        self.assertEqual(set(current), {7})
        self.assertEqual(held, set())
        cached, held = runtime._records_with_hold({}, 10.2)
        self.assertEqual(set(cached), {7})
        self.assertEqual(held, {7})
        expired, held = runtime._records_with_hold({}, 11.01)
        self.assertEqual(expired, {})
        self.assertEqual(held, set())

    def test_marker_top_edge_is_robot_front_with_adjustable_offset(self) -> None:
        marker = np.asarray([(0, 0), (10, 0), (10, 10), (0, 10)], dtype=np.float32)
        self.assertEqual(_marker_top_heading(marker), 270.0)
        self.assertEqual(_marker_top_heading(marker, 90.0), 0.0)


class RobotControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = WebTargetController({
            "hardware_enabled": True,
            "controlled_agent_id": None,
            "robot_transport": "wifi",
            "navigation_inset_cm": 10,
            "target_tolerance_cm": 2,
            "heading_tolerance_deg": 10,
        })

    def test_selected_agent_is_bound_dynamically(self) -> None:
        accepted, message = self.controller.set_target(42, 50, 50)
        self.assertFalse(accepted)
        self.assertIn("ESP32", message)
        self.assertEqual(self.controller.status()["controlled_agent_id"], 42)

    def test_selected_agent_uses_its_saved_lan_ip(self) -> None:
        self.controller.set_agent_ips({42: "192.168.1.57"})
        self.controller.select_agent(42)
        self.assertEqual(self.controller._selected_wifi_host(), "192.168.1.57")

    def test_trajectory_settings_are_kept_per_agent(self) -> None:
        saved = self.controller.set_agent_settings(42, speed=140, target_tolerance_cm=4.5, heading_tolerance_deg=18, heading_kp=2.4, heading_kd=0.35, min_drive_pwm=75, in_place_turn_threshold_deg=105, heading_offset_deg=-30, left_motor_inverted=True, right_motor_inverted=False, stuck_timeout_s=4, stuck_min_progress_cm=2.2, stuck_action="boost", stuck_boost_pwm=45)
        self.assertEqual(saved["speed"], 140)
        self.controller.select_agent(42)
        self.assertEqual(self.controller.speed, 140)
        self.assertEqual(self.controller.target_tolerance_cm, 4.5)
        self.assertEqual(self.controller.heading_tolerance_deg, 18)
        self.assertEqual(self.controller.heading_kp, 2.4)
        self.assertEqual(self.controller.heading_kd, 0.35)
        self.assertEqual(self.controller.min_drive_pwm, 75)
        self.assertEqual(self.controller.in_place_turn_threshold_deg, 105)
        self.assertEqual(self.controller.heading_offset_for(42), -30)
        self.assertTrue(self.controller.left_motor_inverted)
        self.assertFalse(self.controller.right_motor_inverted)
        self.assertEqual(self.controller.stuck_timeout_s, 4)
        self.assertEqual(self.controller.stuck_min_progress_cm, 2.2)
        self.assertEqual(self.controller.stuck_action, "boost")
        self.assertEqual(self.controller.stuck_boost_pwm, 45)

    def test_trajectory_settings_allow_negative_p_gain(self) -> None:
        saved = self.controller.set_agent_settings(
            42, speed=140, target_tolerance_cm=4.5, heading_tolerance_deg=18,
            heading_kp=-2.4, heading_kd=0.35, min_drive_pwm=75,
            in_place_turn_threshold_deg=105, heading_offset_deg=0,
            stuck_timeout_s=4, stuck_min_progress_cm=2.2,
        )
        self.assertEqual(saved["heading_kp"], -2.4)

    def test_logical_right_turn_is_mapped_through_swapped_channels(self) -> None:
        self.assertEqual(self.controller._turn_command(True), "LEFT")
        self.assertEqual(self.controller._turn_command(False), "RIGHT")

    def test_calibration_arc_turn_powers_exactly_one_motor(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller._arc_turn(True, 125)
        self.assertEqual(connection.writes[0], "SET_MOTOR_INVERSION 0 0")
        self.assertRegex(connection.writes[-1], r"^TEST_(LEFT|RIGHT) 125$")
        if connection.writes[-1].startswith("TEST_LEFT"):
            self.controller._accept_esp_response("STATE motion=STOP left_pwm=125 right_pwm=0")
        else:
            self.controller._accept_esp_response("STATE motion=STOP left_pwm=0 right_pwm=125")
        output = self.controller.status()["motor_output"]
        self.assertEqual(sum(value != 0 for value in (output["left_percent"], output["right_percent"])), 1)
        self.assertIn(0, (output["left_percent"], output["right_percent"]))

    def test_pd_curve_drives_both_motors_forward_at_different_speeds(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller._drive_curve(36.0, 160)
        self.assertRegex(connection.writes[-1], r"^DRIVE [1-9][0-9]* [1-9][0-9]*$")
        left, right = self.controller._last_drive_pwm
        self.controller._accept_esp_response(f"STATE motion=STOP left_pwm={left} right_pwm={right}")
        output = self.controller.status()["motor_output"]
        self.assertGreater(output["left_percent"], 0)
        self.assertGreater(output["right_percent"], 0)
        self.assertNotEqual(output["left_percent"], output["right_percent"])

    def test_pd_curve_changes_outer_motor_with_turn_direction(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller._drive_curve(40.0, 160)
        right_turn = self.controller._last_drive_pwm
        self.controller._drive_curve(-40.0, 160)
        left_turn = self.controller._last_drive_pwm
        # Wire order is mirrored; physical order is swapped again for the UI.
        self.assertLess(right_turn[0], right_turn[1])
        self.assertGreater(left_turn[0], left_turn[1])
        output = self.controller.status()["motor_output"]
        self.assertLess(output["server_left_pwm"], output["server_right_pwm"])

    def test_motor_inversion_is_sent_before_drive_command(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.set_agent_settings(42, speed=120, target_tolerance_cm=2, heading_tolerance_deg=10, left_motor_inverted=True)
        self.controller.select_agent(42)
        self.controller._drive_wheels(100, 100)
        self.assertEqual(connection.writes[-2], "SET_MOTOR_INVERSION 1 0")
        self.assertEqual(connection.writes[-1], "DRIVE 100 100")

    def test_stuck_detection_stops_robot_after_no_smoothed_progress(self) -> None:
        self.controller.connection = FakeConnection()
        pose = RobotPose(20, 30, 0)
        self.assertTrue(self.controller._drive_forward(pose, 10.0))
        self.assertFalse(self.controller._drive_forward(pose, 13.0))
        self.assertEqual(self.controller.phase, "STUCK")

    def test_stuck_boost_raises_pwm_instead_of_stopping(self) -> None:
        self.controller.set_agent_settings(12, speed=120, target_tolerance_cm=2, heading_tolerance_deg=10, stuck_action="boost", stuck_boost_pwm=40)
        self.controller.select_agent(12)
        self.controller.connection = FakeConnection()
        pose = RobotPose(20, 30, 0)
        self.assertTrue(self.controller._drive_forward(pose, 10.0))
        self.assertTrue(self.controller._drive_forward(pose, 13.0))
        self.assertNotEqual(self.controller.phase, "STUCK")
        self.assertEqual(self.controller._active_stuck_boost_pwm, 40)

    def test_tracking_pwm_uses_setting_and_adds_stuck_boost(self) -> None:
        self.controller.set_agent_settings(15, speed=180, target_tolerance_cm=2, heading_tolerance_deg=10, min_drive_pwm=70, stuck_action="boost", stuck_boost_pwm=30)
        self.controller.select_agent(15)
        self.assertEqual(self.controller._tracking_pwm(), 70)
        self.controller._active_stuck_boost_pwm = 30
        self.assertEqual(self.controller._tracking_pwm(), 100)

    def test_stuck_boost_reserves_headroom_at_maximum_speed(self) -> None:
        self.controller.set_agent_settings(13, speed=250, target_tolerance_cm=2, heading_tolerance_deg=10, stuck_action="boost", stuck_boost_pwm=35)
        self.controller.select_agent(13)
        connection = FakeConnection()
        self.controller.connection = connection
        pose = RobotPose(20, 30, 0)
        self.controller._drive_forward(pose, 10.0)
        self.assertIn("DRIVE 215 215", connection.writes)
        self.controller._drive_forward(pose, 13.0)
        self.assertIn("DRIVE 250 250", connection.writes)
        self.assertNotEqual(self.controller.phase, "STUCK")

    def test_stuck_continue_keeps_current_pwm_without_stopping_or_boosting(self) -> None:
        self.controller.set_agent_settings(14, speed=120, target_tolerance_cm=2, heading_tolerance_deg=10, stuck_action="continue", stuck_boost_pwm=40)
        self.controller.select_agent(14)
        connection = FakeConnection()
        self.controller.connection = connection
        pose = RobotPose(20, 30, 0)
        self.controller._drive_forward(pose, 10.0)
        self.controller._drive_forward(pose, 13.0)
        self.assertIn("DRIVE 120 120", connection.writes)
        self.assertNotEqual(self.controller.phase, "STUCK")
        self.assertEqual(self.controller._active_stuck_boost_pwm, 0)

    def test_clear_stuck_resumes_kept_target(self) -> None:
        self.controller.connection = FakeConnection()
        self.controller.phase = "STUCK"
        self.controller._target_normalized = (50.0, 50.0)
        cleared, _ = self.controller.clear_stuck()
        self.assertTrue(cleared)
        self.assertEqual(self.controller.phase, "NAVIGATING")

    def test_web_target_drives_real_protocol_after_calibration(self) -> None:
        connection = FakeConnection()
        self.controller.select_agent(10)
        self.controller.connection = connection
        self.controller.calibrated = True
        self.controller.phase = "READY"
        accepted, _ = self.controller.set_target(10, 80, 50)
        self.assertTrue(accepted)
        self.controller.update(RobotPose(20, 50, 0), 100, 100)
        commands = connection.writes
        self.assertTrue(any(command.startswith("SET_BOUNDS 0 0 100.00 100.00") for command in commands))
        self.assertTrue(any(command.startswith("POSITION 20.00 50.00 0.0") for command in commands))
        self.assertIn("DRIVE 80 80", commands)
        self.assertEqual(self.controller.status()["phase"], "NAVIGATING")

    def test_large_heading_error_turns_in_place_with_both_motors(self) -> None:
        connection = FakeConnection()
        self.controller.select_agent(10)
        self.controller.connection = connection
        self.controller.calibrated = True
        self.controller.phase = "READY"
        accepted, _ = self.controller.set_target(10, 80, 50)
        self.assertTrue(accepted)
        self.controller.update(RobotPose(20, 50, 170), 100, 100)
        self.assertIn("SET_SPEED 250", connection.writes)
        self.assertTrue(any(command in {"LEFT", "RIGHT"} for command in connection.writes))
        self.assertFalse(any(command.startswith("DRIVE") for command in connection.writes))

    def test_emergency_stop_latches(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.calibrated = True
        self.controller.phase = "READY"
        self.controller.last_command = "FORWARD"
        self.controller.emergency_stop()
        self.assertEqual(self.controller.status()["phase"], "EMERGENCY_STOP")
        self.assertIn("STOP", connection.writes)
        accepted, _ = self.controller.set_target(10, 80, 50)
        self.assertFalse(accepted)
        self.assertEqual(self.controller.status()["phase"], "EMERGENCY_STOP")

    def test_routine_stop_uses_halt_not_latched_emergency_stop(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.last_command = "DRIVE"
        self.controller._stop("Target reached")
        self.assertIn("HALT", connection.writes)
        self.assertNotIn("STOP", connection.writes)
        self.assertFalse(self.controller._emergency_stop_requested.is_set())
        self.assertNotEqual(self.controller.status()["phase"], "EMERGENCY_STOP")

    def test_manual_wasd_drives_without_camera_pose_and_deadman_halts(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.select_agent(-1)
        self.controller.set_manual_mode(True)
        accepted, _ = self.controller.manual_drive("forward", 130)
        self.assertTrue(accepted)
        self.assertIn("MANUAL_DRIVE 130 130", connection.writes)
        self.controller.update(None, None, None)
        self.assertEqual(self.controller.status()["phase"], "MANUAL_CONTROL")
        self.controller._manual_expires_at = 0.0
        self.controller.update(None, None, None)
        self.assertIn("HALT", connection.writes)
        self.assertEqual(self.controller.status()["phase"], "MANUAL_READY")
        self.assertFalse(self.controller._emergency_stop_requested.is_set())

    def test_manual_wasd_supports_reverse_and_tank_turn_pwm(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.select_agent(-1)
        self.controller.set_manual_mode(True)
        self.assertTrue(self.controller.manual_drive("backward", 90)[0])
        self.assertIn("MANUAL_DRIVE -90 -90", connection.writes)
        self.assertTrue(self.controller.manual_drive("left", 90)[0])
        self.assertIn("MANUAL_DRIVE 90 -90", connection.writes)
        self.assertTrue(self.controller.manual_drive("right", 90)[0])
        self.assertIn("MANUAL_DRIVE -90 90", connection.writes)

    def test_manual_wasd_combines_drive_and_turn_with_one_motor(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.select_agent(-1)
        self.controller.set_manual_mode(True)
        expected = {
            "forward_left": "MANUAL_DRIVE 90 0",
            "forward_right": "MANUAL_DRIVE 0 90",
            "backward_left": "MANUAL_DRIVE 0 -90",
            "backward_right": "MANUAL_DRIVE -90 0",
        }
        for direction, wire_command in expected.items():
            with self.subTest(direction=direction):
                self.assertTrue(self.controller.manual_drive(direction, 90)[0])
                self.assertEqual(connection.writes[-1], wire_command)
                server_pwm = self.controller.status()["motor_output"]
                self.assertEqual(sum(value != 0 for value in (
                    server_pwm["server_left_pwm"], server_pwm["server_right_pwm"]
                )), 1)

    def test_servo_commands_and_esp_feedback_are_reported_per_channel(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.select_agent(42)
        self.assertTrue(self.controller.set_servo(1, 35)[0])
        self.assertEqual(connection.writes[-1], "SET_ARM 35")
        self.assertTrue(self.controller.set_servo(2, 120)[0])
        self.assertEqual(connection.writes[-1], "SET_BUCKET 120")
        self.controller._accept_esp_response(
            "STATE motion=STOP left_pwm=0 right_pwm=0 speed=90 emergency=0 "
            "arm=35 arm_active=1 bucket=120 bucket_active=1"
        )
        channels = {
            item["channel"]: item
            for item in self.controller.status()["servo_output"]["channels"]
        }
        self.assertEqual(channels[1]["esp_angle"], 35)
        self.assertEqual(channels[2]["esp_angle"], 120)
        self.assertTrue(channels[1]["confirmed"])
        self.assertTrue(channels[2]["confirmed"])

    def test_servo_motion_is_blocked_by_emergency_stop(self) -> None:
        connection = FakeConnection()
        self.controller.connection = connection
        self.controller.select_agent(42)
        self.controller._emergency_stop_requested.set()
        accepted, message = self.controller.set_servo(1, 90)
        self.assertFalse(accepted)
        self.assertIn("Аварийная", message)
        self.assertFalse(any(command.startswith("SET_ARM") for command in connection.writes))

    def test_esp_emergency_feedback_latches_server_state(self) -> None:
        self.controller._target_normalized = (80.0, 50.0)
        self.controller._accept_esp_response(
            "STATE motion=STOP left_pwm=0 right_pwm=0 emergency=1"
        )
        self.assertEqual(self.controller.status()["phase"], "EMERGENCY_STOP")
        self.assertIsNone(self.controller.status()["target"])

    def test_emergency_stop_can_be_cleared_after_link_confirmation(self) -> None:
        self.controller.connection = FakeConnection()
        self.controller.calibrated = True
        self.controller.phase = "EMERGENCY_STOP"
        resumed, _ = self.controller.resume_after_emergency_stop()
        self.assertTrue(resumed)
        self.assertEqual(self.controller.status()["phase"], "READY")
        self.assertIn("RESUME", self.controller.connection.writes)

    def test_motor_output_matches_command_direction(self) -> None:
        self.controller.connection = FakeConnection()
        self.controller._accept_esp_response("STATE motion=LEFT left_pwm=-125 right_pwm=125")
        output = self.controller.status()["motor_output"]
        self.assertEqual(output["left_percent"], 50)
        self.assertEqual(output["right_percent"], -50)

    def test_motor_output_stays_zero_until_esp_confirms_pwm(self) -> None:
        self.controller.connection = FakeConnection()
        self.controller._drive_wheels(120, 100)
        output = self.controller.status()["motor_output"]
        self.assertEqual(output["command"], "WAIT_ESP")
        self.assertFalse(output["confirmed"])
        self.assertEqual((output["left_percent"], output["right_percent"]), (0, 0))
        self.assertEqual((output["server_left_pwm"], output["server_right_pwm"]), (100, 120))


class WorkRouteTests(unittest.TestCase):
    def test_direct_route_inside_work_area(self) -> None:
        regions = [{"type": "work", "x": 10, "y": 10, "width": 70, "height": 60}]
        self.assertEqual(plan_work_route((20, 20), (70, 60), regions), [(70, 60)])

    def test_rejects_target_outside_work_area(self) -> None:
        regions = [{"type": "work", "x": 10, "y": 10, "width": 30, "height": 30}]
        with self.assertRaisesRegex(ValueError, "вне рабочей"):
            plan_work_route((20, 20), (80, 80), regions)

    def test_routes_around_l_shaped_boundary(self) -> None:
        regions = [
            {"type": "work", "x": 10, "y": 10, "width": 60, "height": 20},
            {"type": "work", "x": 50, "y": 10, "width": 20, "height": 60},
        ]
        route = plan_work_route((15, 20), (60, 60), regions)
        self.assertGreaterEqual(len(route), 2)
        points = [(15, 20)] + route
        self.assertTrue(all(segment_is_safe(a, b, regions, 1.4) for a, b in zip(points, points[1:])))


if __name__ == "__main__":
    unittest.main()
