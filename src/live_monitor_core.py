#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_monitor_core.py
--------------------
即時動態人體工學偵測與評分共享引擎（供獨立 OpenCV 全螢幕監控與 Streamlit 網頁分頁共用）

核心機制（借鏡 20250929 random calib.py）：
1. 三層訊號穩定化：滑動中位數濾波 (Median) -> 速度自適應 EMA 平滑 -> 死區抑制 (Deadband)，
   有效消除關節角度落在 RULA/REBA 查表邊界時的臨界值跳動 (Threshold Jitter)。
2. 中立姿勢基準校正 (Baseline Calibration)：一鍵記錄自然站立時的軀幹與頸部傾角偏移並扣除，
   自動補償筆電攝影機擺放高低或視角微傾斜造成的系統性誤差。
3. 即時高危害持續計時 (High-Risk Streak)：累計連續處於 AL 3~4 的秒數與全程佔比。
4. 錄測錄製與匯出：即時追蹤全段最高危害瞬間 (Peak Risk Frame)，停止後一鍵輸出 CSV、峰值截圖與正式 HTML 檢核單。
"""

import csv
import os
import sys
from collections import deque
from datetime import datetime
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.pose_analyzer import PoseAnalyzer
    from src.report_generator import generate_html_report
    from src.rula_reba_engine import (
        calculate_reba,
        calculate_rula,
        flatten_assessment_result,
        simulate_what_if_improvements,
    )
except ImportError:
    from pose_analyzer import PoseAnalyzer
    from report_generator import generate_html_report
    from rula_reba_engine import (
        calculate_reba,
        calculate_rula,
        flatten_assessment_result,
        simulate_what_if_improvements,
    )


ANGLE_KEYS = ("trunk", "neck", "upper_arm", "lower_arm", "wrist", "knee")


class LiveErgoSession:
    """管理即時攝影機動態姿勢分析、濾波防抖、基準校正與錄製摘要狀態。"""

    def __init__(
        self,
        analyzer: PoseAnalyzer,
        median_win: int = 5,
        ema_alpha_base: float = 0.30,
        ema_alpha_max: float = 0.65,
        deadband_deg: float = 1.5,
    ):
        self.analyzer = analyzer
        self.median_win = max(1, int(median_win))
        self.ema_alpha_base = float(np.clip(ema_alpha_base, 0.05, 0.85))
        self.ema_alpha_max = max(self.ema_alpha_base, float(np.clip(ema_alpha_max, 0.05, 0.95)))
        self.deadband_deg = float(np.clip(deadband_deg, 0.0, 8.0))

        self.angle_queues: Dict[str, deque] = {
            k: deque(maxlen=self.median_win) for k in ANGLE_KEYS
        }
        self.ema_angles: Dict[str, float] = {}
        self.output_angles: Dict[str, float] = {}
        self.last_raw_angles: Optional[Dict[str, float]] = None
        self.baseline_offsets: Optional[Dict[str, float]] = None

        self.last_frame_ts: Optional[float] = None
        self.fps_ema: float = 0.0
        self.high_risk_streak_sec: float = 0.0
        self.score_history: deque = deque(maxlen=120)

        self.is_recording: bool = False
        self.record_start_ts: float = 0.0
        self.records: List[Dict[str, Any]] = []
        self.recorded_duration_sec: float = 0.0
        self.recorded_high_risk_sec: float = 0.0
        self.peak_score: int = -1
        self.peak_record: Optional[Dict[str, Any]] = None
        self.peak_assessment: Optional[Dict[str, Any]] = None
        self.peak_angles: Optional[Dict[str, float]] = None
        self.peak_diagnostics: Optional[Dict[str, Any]] = None
        self.peak_image: Optional[np.ndarray] = None

    def set_filter_params(
        self,
        ema_alpha_base: Optional[float] = None,
        deadband_deg: Optional[float] = None,
    ) -> None:
        """動態更新 EMA 平滑基準係數與死區門檻。"""
        if ema_alpha_base is not None:
            self.ema_alpha_base = float(np.clip(ema_alpha_base, 0.05, 0.85))
            self.ema_alpha_max = max(self.ema_alpha_base, self.ema_alpha_max)
        if deadband_deg is not None:
            self.deadband_deg = float(np.clip(deadband_deg, 0.0, 8.0))

    def reset_filters(self) -> None:
        """清空濾波佇列與平滑快取。"""
        for q in self.angle_queues.values():
            q.clear()
        self.ema_angles.clear()
        self.output_angles.clear()

    def calibrate_baseline(self, raw_angles: Optional[Dict[str, float]] = None) -> bool:
        """
        以自然直立姿勢作為基準（歸零軀幹與頸部基準傾角，上限 ±25°）。
        :return: 是否成功完成基準校正
        """
        src = raw_angles if raw_angles is not None else self.last_raw_angles
        if not src:
            return False

        trunk_off = float(np.clip(src.get("trunk", 0.0), -25.0, 25.0))
        neck_off = float(np.clip(src.get("neck", 0.0), -25.0, 25.0))
        self.baseline_offsets = {
            "trunk": round(trunk_off, 1),
            "neck": round(neck_off, 1),
        }
        self.reset_filters()
        return True

    def clear_baseline(self) -> None:
        """清除中立姿勢基準偏移並重置濾波器。"""
        self.baseline_offsets = None
        self.reset_filters()

    def start_recording(self) -> None:
        """開始一段即時作業錄測與峰值追蹤。"""
        self.is_recording = True
        self.record_start_ts = time.time()
        self.records = []
        self.recorded_duration_sec = 0.0
        self.recorded_high_risk_sec = 0.0
        self.high_risk_streak_sec = 0.0
        self.peak_score = -1
        self.peak_record = None
        self.peak_assessment = None
        self.peak_angles = None
        self.peak_diagnostics = None
        self.peak_image = None

    def get_recording_summary(self) -> Dict[str, Any]:
        """彙整目前已錄製片段之風險統計與峰值資訊。"""
        total_frames = len(self.records)
        if total_frames == 0:
            return {
                "total_frames": 0,
                "duration_sec": 0.0,
                "avg_score": 0.0,
                "peak_score": 0,
                "peak_record": None,
                "peak_assessment": None,
                "peak_angles": None,
                "peak_diagnostics": None,
                "peak_image": None,
                "high_risk_sec": 0.0,
                "high_risk_pct": 0.0,
                "records": [],
            }

        scores = [r["score"] for r in self.records]
        high_frames = sum(1 for r in self.records if r["action_level"] >= 3)
        duration = self.recorded_duration_sec if self.recorded_duration_sec > 0 else float(self.records[-1]["timestamp_sec"])
        return {
            "total_frames": total_frames,
            "duration_sec": round(duration, 2),
            "avg_score": round(float(np.mean(scores)), 2),
            "peak_score": int(self.peak_score if self.peak_score >= 0 else max(scores)),
            "peak_record": self.peak_record,
            "peak_assessment": self.peak_assessment,
            "peak_angles": dict(self.peak_angles) if self.peak_angles else None,
            "peak_diagnostics": dict(self.peak_diagnostics) if self.peak_diagnostics else None,
            "peak_image": self.peak_image.copy() if self.peak_image is not None else None,
            "high_risk_sec": round(self.recorded_high_risk_sec, 2),
            "high_risk_pct": round((high_frames / total_frames) * 100.0, 1),
            "records": list(self.records),
        }

    def stop_recording(self) -> Dict[str, Any]:
        """停止錄測並回傳完整摘要報告資料。"""
        self.is_recording = False
        return self.get_recording_summary()

    def _stabilize_angles(self, adjusted_angles: Dict[str, float]) -> Dict[str, float]:
        """執行 Median -> 速度自適應 EMA -> 死區抑制，回傳穩定後角度。"""
        stabilized: Dict[str, float] = {}
        for k in ANGLE_KEYS:
            val = float(adjusted_angles.get(k, 0.0))
            q = self.angle_queues[k]
            q.append(val)
            med = float(np.median(q))

            if k not in self.ema_angles:
                ema_val = med
            else:
                prev_ema = self.ema_angles[k]
                vel = abs(med - prev_ema)
                alpha = self.ema_alpha_base + (self.ema_alpha_max - self.ema_alpha_base) * min(1.0, vel / 20.0)
                ema_val = alpha * med + (1.0 - alpha) * prev_ema
            self.ema_angles[k] = ema_val

            if k not in self.output_angles:
                out_val = ema_val
            else:
                prev_out = self.output_angles[k]
                out_val = ema_val if abs(ema_val - prev_out) >= self.deadband_deg else prev_out
            self.output_angles[k] = out_val
            stabilized[k] = float(round(out_val, 1))
        return stabilized

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        method: str = "RULA",
        side: str = "worst",
        manual_modifiers: Optional[Dict[str, Any]] = None,
        mirror: bool = True,
        draw_hud: bool = False,
        show_angle_labels: bool = True,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """處理單幀即時影像並回傳骨架疊圖、穩定角度、評分與警示狀態。"""
        frame = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()

        now = float(timestamp) if timestamp is not None else time.time()
        dt = 0.0
        if self.last_frame_ts is not None:
            dt = float(np.clip(now - self.last_frame_ts, 0.0, 0.5))
        self.last_frame_ts = now
        if dt > 1e-4:
            inst_fps = 1.0 / dt
            self.fps_ema = inst_fps if self.fps_ema <= 0.0 else (0.2 * inst_fps + 0.8 * self.fps_ema)

        detection = self.analyzer.extract_keypoints(frame)
        if detection is None:
            self.high_risk_streak_sec = 0.0
            return {
                "detected": False,
                "annotated_image": frame,
                "fps": round(self.fps_ema, 1),
                "is_calibrated": self.baseline_offsets is not None,
                "is_recording": self.is_recording,
                "high_risk_streak_sec": 0.0,
                "recorded_count": len(self.records),
                "recorded_duration_sec": round(self.recorded_duration_sec, 1),
                "recorded_high_risk_sec": round(self.recorded_high_risk_sec, 1),
            }

        xy, conf, _bbox = detection
        raw_angles, side_used, diag = self.analyzer.calculate_body_angles(
            xy,
            conf,
            side=side,
            smooth=False,
            manual_modifiers=manual_modifiers,
            method=method,
        )
        self.last_raw_angles = dict(raw_angles)

        adjusted = dict(raw_angles)
        if self.baseline_offsets is not None:
            adjusted["trunk"] = float(round(adjusted["trunk"] - self.baseline_offsets["trunk"], 1))
            adjusted["neck"] = float(round(adjusted["neck"] - self.baseline_offsets["neck"], 1))

        stabilized_angles = self._stabilize_angles(adjusted)

        if method.upper() == "REBA":
            assessment_result = calculate_reba(stabilized_angles, manual_modifiers)
            score = int(assessment_result["final_reba_score"])
        else:
            assessment_result = calculate_rula(stabilized_angles, manual_modifiers)
            score = int(assessment_result["grand_score"])
        action_level = int(assessment_result["action_level"])
        flat = flatten_assessment_result(assessment_result)

        self.score_history.append(score)
        if action_level >= 3:
            self.high_risk_streak_sec += dt
        else:
            self.high_risk_streak_sec = 0.0

        annotated = self.analyzer.render_overlay(
            frame,
            xy,
            conf,
            side_used,
            assessment_result,
            stabilized_angles,
            draw_hud=draw_hud,
            show_angle_labels=show_angle_labels,
        )

        if self.is_recording:
            self.recorded_duration_sec += dt
            if action_level >= 3:
                self.recorded_high_risk_sec += dt
            rec = {
                "frame_index": len(self.records) + 1,
                "timestamp_sec": round(self.recorded_duration_sec, 2),
                "score": score,
                "action_level": action_level,
                "side_used": side_used,
                "asymmetry_alert": bool(diag.get("asymmetry_alert", False)),
                "raw_angles": dict(stabilized_angles),
                **flat,
            }
            self.records.append(rec)

            if score > self.peak_score or self.peak_image is None:
                self.peak_score = score
                self.peak_record = rec
                self.peak_assessment = assessment_result
                self.peak_angles = dict(stabilized_angles)
                self.peak_diagnostics = dict(diag)
                self.peak_image = annotated.copy()

        return {
            "detected": True,
            "annotated_image": annotated,
            "assessment_result": assessment_result,
            "flattened": flat,
            "angles": stabilized_angles,
            "raw_angles": raw_angles,
            "diagnostics": diag,
            "score": score,
            "action_level": action_level,
            "side_used": side_used,
            "fps": round(self.fps_ema, 1),
            "high_risk_streak_sec": round(self.high_risk_streak_sec, 1),
            "is_calibrated": self.baseline_offsets is not None,
            "is_recording": self.is_recording,
            "recorded_count": len(self.records),
            "recorded_duration_sec": round(self.recorded_duration_sec, 1),
            "recorded_high_risk_sec": round(self.recorded_high_risk_sec, 1),
        }

    def export_snapshot(
        self,
        frame_result: Dict[str, Any],
        output_dir: str,
        method: str = "RULA",
        manual_modifiers: Optional[Dict[str, Any]] = None,
        prefix: str = "snapshot",
    ) -> Dict[str, str]:
        """將單幀即時偵測結果存成標註 JPG 與單頁 HTML/PDF 檢核單。"""
        if not frame_result or not frame_result.get("detected"):
            raise ValueError("目前影格未偵測到有效人體骨架，無法匯出快照。")

        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{method.upper()}_{stamp}"
        jpg_path = os.path.join(output_dir, f"{base_name}.jpg")
        html_path = os.path.join(output_dir, f"{base_name}.html")

        img = frame_result["annotated_image"]
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("影像編碼失敗")
        buf.tofile(jpg_path)

        angles = frame_result["angles"]
        scenarios = simulate_what_if_improvements(angles, manual_modifiers, method=method)
        html_str = generate_html_report(
            assessment_result=frame_result["assessment_result"],
            angles=angles,
            annotated_bgr=img,
            method=method,
            manual_modifiers=manual_modifiers,
            what_if_scenarios=scenarios,
            diagnostics=frame_result.get("diagnostics"),
            item_name=f"LiveSnapshot_{stamp}",
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        return {"jpg_path": jpg_path, "html_path": html_path}

    def export_session_outputs(
        self,
        summary: Dict[str, Any],
        output_dir: str,
        method: str = "RULA",
        manual_modifiers: Optional[Dict[str, Any]] = None,
        prefix: str = "live_session",
    ) -> Dict[str, str]:
        """將錄測結果匯出為時序 CSV、最高危害瞬間 JPG 與正式 HTML 報告。"""
        records = summary.get("records") or []
        if not records:
            raise ValueError("錄測期間無有效人體影格資料，無法匯出報表。")

        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{method.upper()}_{stamp}"

        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        fieldnames = [k for k in records[0].keys() if k != "raw_angles"]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow({k: v for k, v in r.items() if k != "raw_angles"})

        outputs = {"csv_path": csv_path}
        peak_img = summary.get("peak_image")
        peak_angles = summary.get("peak_angles")
        peak_ass = summary.get("peak_assessment")

        if peak_img is not None and peak_angles is not None and peak_ass is not None:
            jpg_path = os.path.join(output_dir, f"{base_name}_peak.jpg")
            html_path = os.path.join(output_dir, f"{base_name}_peak_report.html")

            ok, buf = cv2.imencode(".jpg", peak_img)
            if ok:
                buf.tofile(jpg_path)
                outputs["peak_jpg_path"] = jpg_path

            scenarios = simulate_what_if_improvements(peak_angles, manual_modifiers, method=method)
            html_str = generate_html_report(
                assessment_result=peak_ass,
                angles=peak_angles,
                annotated_bgr=peak_img,
                method=method,
                manual_modifiers=manual_modifiers,
                what_if_scenarios=scenarios,
                diagnostics=summary.get("peak_diagnostics"),
                item_name=f"LivePeak_{stamp}",
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            outputs["peak_html_path"] = html_path

        return outputs
