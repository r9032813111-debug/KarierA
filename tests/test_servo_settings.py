from __future__ import annotations

import unittest

from backend.main import _servo_settings_for


class ServoSettingsTests(unittest.TestCase):
    def test_robot_types_expose_the_expected_servo_channels(self) -> None:
        loader = _servo_settings_for("loader")
        bulldozer = _servo_settings_for("bulldozer")
        dumper = _servo_settings_for("dumper")
        self.assertEqual([item["name"] for item in loader["channels"]], ["Подъём", "Ковш"])
        self.assertEqual([item["name"] for item in bulldozer["channels"]], ["Отвал"])
        self.assertEqual([item["name"] for item in dumper["channels"]], ["Кузов"])

    def test_saved_limits_and_position_are_kept_inside_safe_range(self) -> None:
        settings = _servo_settings_for("loader", {
            "channels": [
                {"channel": 1, "min_angle": 15, "max_angle": 80, "position": 45},
                {"channel": 2, "min_angle": 40, "max_angle": 120, "position": 170},
            ]
        })
        first, second = settings["channels"]
        self.assertEqual((first["min_angle"], first["max_angle"], first["position"]), (15, 80, 45))
        self.assertEqual((second["min_angle"], second["max_angle"], second["position"]), (40, 120, 120))


if __name__ == "__main__":
    unittest.main()
