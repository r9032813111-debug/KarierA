from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_PROGRAM = Path(__file__).resolve().parents[1] / "calibration_program"
sys.path.insert(0, str(CALIBRATION_PROGRAM))

import dual_camera_calibrate_v2 as mono  # noqa: E402
import stereo_calibrate_v2 as stereo  # noqa: E402


class CalibrationParallelTests(unittest.TestCase):
    def test_complete_16_logical_cpu_budget_is_enabled(self) -> None:
        self.assertEqual(mono.OPENCV_WORKER_THREADS, 16)
        self.assertEqual(stereo.OPENCV_WORKER_THREADS, 16)
        self.assertEqual(mono.MONO_PARALLEL_JOBS * mono.MONO_THREADS_PER_JOB, 16)

    def test_left_and_right_mono_solves_really_overlap(self) -> None:
        session = object.__new__(mono.UnifiedStereoSession)
        barrier = threading.Barrier(2)

        def solve(_objects, points, _size, _flags):
            barrier.wait(timeout=2.0)
            return points

        session._calibrate_points = solve
        previous_threads = cv2.getNumThreads()
        left, right = session._calibrate_left_right_parallel(
            [], ["left"], ["right"], (1920, 1080), 0
        )
        self.assertEqual((left, right), (["left"], ["right"]))
        self.assertEqual(cv2.getNumThreads(), previous_threads)

    def test_parallel_solver_calibrates_two_real_synthetic_cameras(self) -> None:
        session = object.__new__(mono.UnifiedStereoSession)
        object_template = np.zeros((42, 3), np.float32)
        object_template[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * 22.8125
        matrix = np.asarray([[1250.0, 0.0, 960.0], [0.0, 1245.0, 540.0], [0.0, 0.0, 1.0]])
        distortion = np.zeros(5)
        object_points, left_points, right_points = [], [], []
        for index in range(15):
            rotation = np.asarray([0.015 * index, -0.01 * index, 0.008 * index])
            translation = np.asarray([-90.0 + index * 8.0, -55.0 + index * 5.0, 850.0 + index * 22.0])
            left, _ = cv2.projectPoints(object_template, rotation, translation, matrix, distortion)
            right, _ = cv2.projectPoints(
                object_template, rotation, translation + np.asarray([-120.0, 0.0, 0.0]), matrix, distortion
            )
            object_points.append(object_template.copy())
            left_points.append(left.astype(np.float32))
            right_points.append(right.astype(np.float32))

        left_result, right_result = session._calibrate_left_right_parallel(
            object_points, left_points, right_points, (1920, 1080), 0
        )
        self.assertTrue(np.isfinite(left_result[0]))
        self.assertTrue(np.isfinite(right_result[0]))
        self.assertEqual(left_result[1].shape, (3, 3))
        self.assertEqual(right_result[1].shape, (3, 3))

    def test_vectorized_epipolar_error_matches_reference_formula(self) -> None:
        rng = np.random.default_rng(42)
        session = object.__new__(stereo.StereoSession)
        session.left_points = [rng.normal(size=(42, 1, 2)).astype(np.float32) for _ in range(6)]
        session.right_points = [rng.normal(size=(42, 1, 2)).astype(np.float32) for _ in range(6)]
        fundamental = rng.normal(size=(3, 3))

        expected = []
        for left, right in zip(session.left_points, session.right_points):
            x1 = np.c_[left.reshape(-1, 2), np.ones(len(left))]
            x2 = np.c_[right.reshape(-1, 2), np.ones(len(right))]
            l2 = x1 @ fundamental.T
            l1 = x2 @ fundamental
            numerator = np.sum(x2 * l2, axis=1)
            d1 = np.abs(numerator) / np.maximum(np.hypot(l1[:, 0], l1[:, 1]), 1e-12)
            d2 = np.abs(numerator) / np.maximum(np.hypot(l2[:, 0], l2[:, 1]), 1e-12)
            expected.append(float(np.sqrt(np.mean((d1 * d1 + d2 * d2) * 0.5))))

        actual = session._symmetric_epipolar_errors(fundamental)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
