#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_live_monitor.py
--------------------------
針對 LiveErgoSession 即時動態評分引擎進行端對端單元測試（無需實體攝影機）。
涵蓋：
1. RULA / REBA 即時影格處理與 Median + EMA + Deadband 防抖
2. 中立姿勢基準校正 (calibrate_baseline) 與清除 (clear_baseline)
3. 錄測錄製與最高危害瞬間捕捉 (start_recording / stop_recording)
4. 含中文目錄之快照與完整報告匯出 (JPG / CSV / HTML)
"""

import os
import shutil
import sys
import unittest

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pose_analyzer import PoseAnalyzer
from src.live_monitor_core import LiveErgoSession


class TestLiveErgoMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(PROJECT_ROOT, "HFcode", "yolo11n-pose.pt")
        cls.analyzer = PoseAnalyzer(model_path=model_path, conf_thres=0.4)
        sample_img_path = os.path.join(PROJECT_ROOT, "sample_data", "05_bending_inspection_posture.jpg")
        cls.sample_img = cv2.imread(sample_img_path)
        assert cls.sample_img is not None, f"測試影像不存在: {sample_img_path}"
        cls.temp_out_dir = os.path.join(PROJECT_ROOT, "temp_live_測試輸出")

    def tearDown(self):
        if os.path.exists(self.temp_out_dir):
            shutil.rmtree(self.temp_out_dir, ignore_errors=True)

    def test_rula_and_reba_stabilization(self):
        session = LiveErgoSession(self.analyzer)
        modifiers = {"load_weight_kg": 5.0, "activity_repeated": True, "coupling_score": 1}

        for method in ("RULA", "REBA"):
            session.reset_filters()
            res = None
            for i in range(5):
                res = session.process_frame(
                    self.sample_img,
                    method=method,
                    side="worst",
                    manual_modifiers=modifiers,
                    mirror=False,
                    timestamp=100.0 + i * 0.05,
                )
                self.assertTrue(res["detected"])
                self.assertGreaterEqual(res["score"], 1)
                self.assertGreaterEqual(res["rula_score"], 1)
                self.assertGreaterEqual(res["reba_score"], 1)
                self.assertIn("annotated_image", res)
                self.assertIn("rula_result", res)
                self.assertIn("reba_result", res)
                self.assertIn("group_a", res["assessment_result"])
                self.assertIn("group_b", res["assessment_result"])

            self.assertEqual(len(session.rula_score_history), 5 if method == "RULA" else 10)
            self.assertEqual(len(session.reba_score_history), 5 if method == "RULA" else 10)
            self.assertIsNotNone(res)

    def test_baseline_calibration_and_clear(self):
        session = LiveErgoSession(self.analyzer, deadband_deg=0.0)
        res_before = session.process_frame(
            self.sample_img, method="RULA", side="worst", mirror=False, timestamp=10.0
        )
        self.assertTrue(res_before["detected"])
        raw_trunk = res_before["raw_angles"]["trunk"]

        ok = session.calibrate_baseline()
        self.assertTrue(ok)
        res_after = session.process_frame(
            self.sample_img, method="RULA", side="worst", mirror=False, timestamp=10.05
        )
        self.assertTrue(res_after["is_calibrated"])
        if abs(raw_trunk) <= 25.0:
            self.assertAlmostEqual(res_after["angles"]["trunk"], 0.0, places=1)
        else:
            self.assertLess(abs(res_after["angles"]["trunk"]), abs(raw_trunk))

        session.clear_baseline()
        res_cleared = session.process_frame(
            self.sample_img, method="RULA", side="worst", mirror=False, timestamp=10.10
        )
        self.assertFalse(res_cleared["is_calibrated"])
        self.assertAlmostEqual(res_cleared["angles"]["trunk"], raw_trunk, delta=1.0)

    def test_recording_peak_and_unicode_export(self):
        session = LiveErgoSession(self.analyzer)
        modifiers = {"load_weight_kg": 12.0, "activity_static": True}
        session.start_recording()

        last_res = None
        for i in range(6):
            last_res = session.process_frame(
                self.sample_img,
                method="RULA",
                side="worst",
                manual_modifiers=modifiers,
                mirror=False,
                timestamp=50.0 + i * 0.5,
            )

        summary = session.stop_recording()
        self.assertEqual(summary["total_frames"], 6)
        self.assertGreater(summary["duration_sec"], 2.0)
        self.assertGreaterEqual(summary["peak_score"], 3)
        self.assertGreaterEqual(summary["peak_rula_score"], 3)
        self.assertGreaterEqual(summary["peak_reba_score"], 1)
        self.assertIsNotNone(summary["peak_rula_assessment"])
        self.assertIsNotNone(summary["peak_reba_assessment"])
        self.assertIn("group_a", summary["peak_rula_assessment"])
        self.assertIn("group_a", summary["peak_reba_assessment"])
        self.assertIsNotNone(summary["peak_rula_image"])
        self.assertIsNotNone(summary["peak_reba_image"])
        self.assertIsNotNone(summary.get("video_path"))
        self.assertTrue(os.path.exists(summary["video_path"]))
        self.assertGreater(os.path.getsize(summary["video_path"]), 0)

        snap_paths = session.export_snapshot(
            last_res, self.temp_out_dir, method="RULA", manual_modifiers=modifiers
        )
        self.assertTrue(os.path.exists(snap_paths["jpg_path"]))
        self.assertTrue(os.path.exists(snap_paths["rula_html_path"]))
        self.assertTrue(os.path.exists(snap_paths["reba_html_path"]))

        sess_paths = session.export_session_outputs(
            summary, self.temp_out_dir, method="RULA", manual_modifiers=modifiers
        )
        self.assertTrue(os.path.exists(sess_paths["rula_csv_path"]))
        self.assertTrue(os.path.exists(sess_paths["reba_csv_path"]))
        self.assertNotEqual(sess_paths["rula_csv_path"], sess_paths["reba_csv_path"])
        self.assertTrue(os.path.exists(sess_paths["peak_rula_jpg_path"]))
        self.assertTrue(os.path.exists(sess_paths["peak_rula_html_path"]))
        self.assertTrue(os.path.exists(sess_paths["peak_reba_jpg_path"]))
        self.assertTrue(os.path.exists(sess_paths["peak_reba_html_path"]))
        self.assertTrue(os.path.exists(sess_paths["video_path"]))
        self.assertGreater(os.path.getsize(sess_paths["video_path"]), 0)

        try:
            os.remove(summary["video_path"])
        except Exception:
            pass

        with open(sess_paths["rula_csv_path"], "r", encoding="utf-8-sig") as f:
            rula_csv_txt = f.read()
        self.assertIn("wrist_arm_score", rula_csv_txt)
        self.assertNotIn("coupling_score", rula_csv_txt)

        with open(sess_paths["reba_csv_path"], "r", encoding="utf-8-sig") as f:
            reba_csv_txt = f.read()
        self.assertIn("coupling_score", reba_csv_txt)
        self.assertNotIn("wrist_arm_score", reba_csv_txt)

        with open(sess_paths["peak_rula_html_path"], "r", encoding="utf-8") as f:
            self.assertIn("<!DOCTYPE html>", f.read())
        with open(sess_paths["peak_reba_html_path"], "r", encoding="utf-8") as f:
            self.assertIn("<!DOCTYPE html>", f.read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
