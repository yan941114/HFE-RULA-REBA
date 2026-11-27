#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_ergo_monitor.py
--------------------
獨立全螢幕即時動態 RULA / REBA 人因風險監控程式 (風格對齊 20250929 random calib.py)

特色：
1. 原生 OpenCV 全螢幕/低延遲即時攝影機串流 + YOLO11-Pose 骨架偵測。
2. 三層防抖濾波 (Median + 自適應 EMA + Deadband) 與即時中立姿勢基準校正 (Baseline Calibration)。
3. 左側即時人因資訊卡、底部即時分數折線圖、身體不對稱與久蹲/彎腰超時警示。
4. 支援即時錄測與單幀快照，結束後自動產出 CSV、峰值截圖與可列印 HTML 評估檢核單。

常用熱鍵：
  M : 切換 RULA / REBA
  W : 切換側別 (worst -> auto -> left -> right)
  C : 設為中立姿勢基準 (扣除軀幹/頸部初始傾角誤差)
  X : 清除基準校正
  S : 開始 / 停止錄測並匯出完整報告
  Space : 立即拍下當前評估快照並匯出報告
  R : 切換鏡像翻轉
  A : 切換關節幾何角度標籤
  [ / ] : 調整 EMA 平滑係數 (- / +)
  , / . : 調整死區抑制門檻 (- / +)
  + / - : 調整負重重量 (+2 / -2 kg)
  B : 開關高風險持續音效警示 (預設關閉)
  F : 切換全螢幕 / 一般視窗
  Q / Esc : 離開程式
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pose_analyzer import PoseAnalyzer
from src.live_monitor_core import LiveErgoSession

try:
    import winsound
except ImportError:
    winsound = None


RISK_COLORS_BGR = {
    0: (0, 200, 0),
    1: (0, 200, 0),      # 綠 (Acceptable)
    2: (0, 215, 255),    # 黃/金 (Low / Medium)
    3: (0, 140, 255),    # 橘 (High)
    4: (0, 0, 255),      # 紅 (Very High)
}

RISK_LABELS_EN = {
    0: "Negligible Risk",
    1: "Acceptable",
    2: "Low / Investigate",
    3: "Medium-High Risk",
    4: "CRITICAL RISK",
}

SIDE_CYCLE = ["worst", "auto", "left", "right"]


def build_modifiers(
    load_kg: float = 0.0,
    act_static: bool = False,
    act_repeat: bool = False,
    coupling_score: int = 0,
) -> Dict[str, Any]:
    """依標準 RULA / REBA 規則組裝負重與環境補正字典。"""
    if load_kg < 2.0:
        rula_force = 0
    elif load_kg <= 10.0:
        rula_force = 2 if (act_static or act_repeat) else 1
    else:
        rula_force = 3

    return {
        "load_weight_kg": float(load_kg),
        "load_rapid_shock": False,
        "muscle_use_a": 1 if (act_static or act_repeat) else 0,
        "muscle_use_b": 1 if (act_static or act_repeat) else 0,
        "force_load_a": rula_force,
        "force_load_b": rula_force,
        "activity_static": act_static,
        "activity_repeated": act_repeat,
        "activity_rapid": False,
        "coupling_score": int(coupling_score),
        "legs_supported": True,
    }


def _draw_translucent_rect(img: np.ndarray, pt1, pt2, color=(18, 18, 22), alpha=0.72) -> None:
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _draw_sparkline(img: np.ndarray, scores: List[int], method: str) -> None:
    """在畫面左下角繪製即時風險分數波形圖。"""
    if len(scores) < 2:
        return
    h, w = img.shape[:2]
    box_w, box_h = 300, 100
    x0, y0 = 16, h - box_h - 18
    if x0 + box_w >= w or y0 <= 60:
        return

    _draw_translucent_rect(img, (x0, y0), (x0 + box_w, y0 + box_h), (18, 18, 22), 0.72)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (90, 95, 110), 1)
    cv2.putText(img, f"{method} Trend (last {len(scores)}f)", (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 215, 225), 1, cv2.LINE_AA)

    max_s = 15.0 if method.upper() == "REBA" else 7.0
    plot_x0, plot_y0 = x0 + 10, y0 + 26
    plot_w, plot_h = box_w - 20, box_h - 36

    # 高風險警示線
    warn_val = 8.0 if method.upper() == "REBA" else 5.0
    warn_y = int(plot_y0 + plot_h * (1.0 - min(1.0, warn_val / max_s)))
    cv2.line(img, (plot_x0, warn_y), (plot_x0 + plot_w, warn_y), (0, 100, 220), 1, cv2.LINE_AA)

    pts = []
    n = len(scores)
    for i, s in enumerate(scores):
        px = int(plot_x0 + (i / max(1, n - 1)) * plot_w)
        py = int(plot_y0 + plot_h * (1.0 - min(1.0, max(0.0, s / max_s))))
        pts.append((px, py))

    for i in range(1, len(pts)):
        s_cur = scores[i]
        col = (0, 0, 255) if s_cur >= warn_val else ((0, 215, 255) if s_cur >= 3 else (0, 220, 120))
        cv2.line(img, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)


def render_monitor_ui(
    frame_res: Dict[str, Any],
    session: LiveErgoSession,
    method: str,
    side: str,
    load_kg: float,
    beep_enabled: bool,
    toast_msg: str,
    toast_until: float,
) -> np.ndarray:
    """繪製純英文 ASCII 遙測介面（避免 OpenCV 中文問號亂碼）。"""
    out = frame_res["annotated_image"].copy()
    h, w = out.shape[:2]

    # 1. 頂部狀態與熱鍵條
    _draw_translucent_rect(out, (0, 0), (w, 58), (15, 18, 24), 0.78)
    calib_txt = "ON" if frame_res.get("is_calibrated") else "OFF"
    fps_val = frame_res.get("fps", 0.0)
    top_status = (
        f"Mode:{method}  Side:{side.upper()}  Load:{load_kg:.1f}kg  "
        f"Calib:{calib_txt}  EMA:{session.ema_alpha_base:.2f}  "
        f"Dead:{session.deadband_deg:.1f}deg  FPS:{fps_val:.1f}  "
        f"Beep:{'ON' if beep_enabled else 'OFF'}"
    )
    cv2.putText(out, top_status, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 180), 1, cv2.LINE_AA)

    hotkeys = "M:Method  W:Side  C/X:Calib/Clr  S:Rec  Space:Snap  A:Angles  R:Mirror  [/]:EMA  ,/.:Dead  +/-:Load  F:Full  Q:Quit"
    cv2.putText(out, hotkeys, (14, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 200, 215), 1, cv2.LINE_AA)

    # 錄製狀態紅點
    if frame_res.get("is_recording"):
        rec_sec = frame_res.get("recorded_duration_sec", 0.0)
        mins, secs = divmod(int(rec_sec), 60)
        rec_txt = f"REC {mins:02d}:{secs:02d} ({frame_res.get('recorded_count', 0)}f)"
        cv2.circle(out, (w - 175, 22), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, rec_txt, (w - 160, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 255), 2, cv2.LINE_AA)

    # 2. 左側遙測卡片
    if frame_res.get("detected"):
        score = frame_res["score"]
        al = frame_res["action_level"]
        angles = frame_res["angles"]
        diag = frame_res["diagnostics"]
        r_color = RISK_COLORS_BGR.get(al, (0, 215, 255))
        r_label = RISK_LABELS_EN.get(al, "Evaluate")

        card_h = 250 if not diag.get("asymmetry_alert") else 285
        _draw_translucent_rect(out, (16, 72), (336, 72 + card_h), (18, 20, 26), 0.74)
        cv2.rectangle(out, (16, 72), (336, 72 + card_h), r_color, 2)

        cv2.putText(out, f"{method}: {score}  (AL {al})", (28, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, r_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"Status: {r_label}", (28, 136),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 245), 1, cv2.LINE_AA)
        cv2.putText(out, f"Eval: {diag['side_used'].upper()}  (L:{diag['left_score']} vs R:{diag['right_score']})",
                    (28, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 210, 220), 1, cv2.LINE_AA)

        cv2.line(out, (28, 174), (322, 174), (75, 80, 95), 1)
        cv2.putText(out, f"Trunk : {angles['trunk']:+6.1f} deg", (28, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(out, f"Neck  : {angles['neck']:+6.1f} deg", (28, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(out, f"Arm   : {angles['upper_arm']:+6.1f} deg", (28, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(out, f"Elbow : {angles['lower_arm']:6.1f} deg", (28, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(out, f"Knee  : {angles['knee']:6.1f} deg", (28, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240), 1, cv2.LINE_AA)

        if diag.get("asymmetry_alert"):
            cv2.putText(out, "[!] ASYMMETRY ALERT", (28, 326),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 215, 255), 2, cv2.LINE_AA)

        # 高危害連續超時警示條
        streak = frame_res.get("high_risk_streak_sec", 0.0)
        if streak >= 2.0:
            warn_box_y = 72 + card_h + 10
            _draw_translucent_rect(out, (16, warn_box_y), (336, warn_box_y + 38), (20, 20, 140), 0.85)
            cv2.rectangle(out, (16, warn_box_y), (336, warn_box_y + 38), (0, 0, 255), 2)
            cv2.putText(out, f"HIGH RISK STREAK: {streak:.1f}s!", (28, warn_box_y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        _draw_sparkline(out, list(session.score_history), method)
    else:
        _draw_translucent_rect(out, (16, h - 62), (430, h - 16), (25, 20, 120), 0.8)
        cv2.putText(out, "NO PERSON DETECTED - STEP INTO FRAME", (28, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 160, 255), 2, cv2.LINE_AA)

    # 3. 底部操作 Toast 提示
    if toast_msg and time.time() < toast_until:
        (tw, th), _ = cv2.getTextSize(toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = max(20, (w - tw) // 2)
        ty = h - 32
        _draw_translucent_rect(out, (tx - 14, ty - th - 10), (tx + tw + 14, ty + 10), (15, 45, 25), 0.85)
        cv2.rectangle(out, (tx - 14, ty - th - 10), (tx + tw + 14, ty + 10), (0, 230, 130), 1)
        cv2.putText(out, toast_msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ErgoVision AI Live Camera RULA/REBA Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--method", type=str, default="RULA", choices=["RULA", "REBA"], help="Initial assessment method")
    parser.add_argument("--side", type=str, default="worst", choices=SIDE_CYCLE, help="Evaluation side")
    parser.add_argument("--load", type=float, default=0.0, help="Initial load weight in kg")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--output", type=str, default=os.path.join("outputs", "live_sessions"), help="Output directory")
    parser.add_argument("--no-fullscreen", action="store_true", help="Start in windowed mode instead of fullscreen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = os.path.join(os.path.dirname(__file__), "HFcode", "yolo11n-pose.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11n-pose.pt"

    analyzer = PoseAnalyzer(model_path=model_path, conf_thres=0.4, ema_alpha=0.35)
    session = LiveErgoSession(analyzer=analyzer)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"無法開啟攝影機 (camera index={args.camera})，請確認鏡頭連線或未被其他程式佔用。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win_name = "ErgoVision AI - Live RULA/REBA Monitor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    fullscreen = not args.no_fullscreen
    try:
        cv2.setWindowProperty(
            win_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )
    except Exception:
        fullscreen = False

    method = args.method.upper()
    side = args.side.lower()
    load_kg = float(args.load)
    mirror = True
    show_angle_labels = True
    beep_enabled = False
    last_beep_ts = 0.0
    toast_msg = ""
    toast_until = 0.0

    def set_toast(msg: str, sec: float = 2.8) -> None:
        nonlocal toast_msg, toast_until
        toast_msg = msg
        toast_until = time.time() + sec
        print(f"[LiveMonitor] {msg}")

    set_toast(f"Live Monitor Ready ({method}, {side.upper()})", 2.5)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[LiveMonitor] 攝影機影格讀取結束或中斷。")
                break

            modifiers = build_modifiers(load_kg=load_kg)
            frame_res = session.process_frame(
                frame,
                method=method,
                side=side,
                manual_modifiers=modifiers,
                mirror=mirror,
                draw_hud=False,
                show_angle_labels=show_angle_labels,
            )

            if beep_enabled and winsound is not None and frame_res.get("high_risk_streak_sec", 0.0) >= 2.0:
                now = time.time()
                if now - last_beep_ts >= 3.0:
                    last_beep_ts = now
                    try:
                        winsound.Beep(880, 120)
                    except Exception:
                        pass

            disp = render_monitor_ui(
                frame_res,
                session=session,
                method=method,
                side=side,
                load_kg=load_kg,
                beep_enabled=beep_enabled,
                toast_msg=toast_msg,
                toast_until=toast_until,
            )
            cv2.imshow(win_name, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                if session.is_recording:
                    summary = session.stop_recording()
                    if summary.get("records"):
                        exported = session.export_session_outputs(summary, args.output, method=method, manual_modifiers=modifiers)
                        set_toast(f"Saved session report -> {os.path.basename(exported.get('peak_html_path', exported['csv_path']))}")
                break
            elif key in (ord("m"), ord("M")):
                method = "REBA" if method == "RULA" else "RULA"
                set_toast(f"Switched Method -> {method}")
            elif key in (ord("w"), ord("W")):
                idx = SIDE_CYCLE.index(side) if side in SIDE_CYCLE else 0
                side = SIDE_CYCLE[(idx + 1) % len(SIDE_CYCLE)]
                set_toast(f"Switched Side -> {side.upper()}")
            elif key in (ord("c"), ord("C")):
                if session.calibrate_baseline():
                    set_toast(f"Baseline Calibrated (Trunk/Neck zeroed)")
                else:
                    set_toast("Calibration failed: No person detected")
            elif key in (ord("x"), ord("X")):
                session.clear_baseline()
                set_toast("Baseline Calibration Cleared")
            elif key in (ord("r"), ord("R")):
                mirror = not mirror
                set_toast(f"Mirror -> {'ON' if mirror else 'OFF'}")
            elif key in (ord("a"), ord("A")):
                show_angle_labels = not show_angle_labels
                set_toast(f"Angle Badges -> {'ON' if show_angle_labels else 'OFF'}")
            elif key in (ord("b"), ord("B")):
                beep_enabled = not beep_enabled
                set_toast(f"Audio Alert -> {'ON' if beep_enabled else 'OFF'}")
            elif key == ord("["):
                session.set_filter_params(ema_alpha_base=session.ema_alpha_base - 0.05)
                set_toast(f"EMA alpha -> {session.ema_alpha_base:.2f}")
            elif key == ord("]"):
                session.set_filter_params(ema_alpha_base=session.ema_alpha_base + 0.05)
                set_toast(f"EMA alpha -> {session.ema_alpha_base:.2f}")
            elif key == ord(","):
                session.set_filter_params(deadband_deg=session.deadband_deg - 0.5)
                set_toast(f"Deadband -> {session.deadband_deg:.1f} deg")
            elif key == ord("."):
                session.set_filter_params(deadband_deg=session.deadband_deg + 0.5)
                set_toast(f"Deadband -> {session.deadband_deg:.1f} deg")
            elif key in (ord("+"), ord("=")):
                load_kg = min(30.0, load_kg + 2.0)
                set_toast(f"Load Weight -> {load_kg:.1f} kg")
            elif key in (ord("-"), ord("_")):
                load_kg = max(0.0, load_kg - 2.0)
                set_toast(f"Load Weight -> {load_kg:.1f} kg")
            elif key == 32:  # Space: Snapshot
                if frame_res.get("detected"):
                    saved = session.export_snapshot(frame_res, args.output, method=method, manual_modifiers=modifiers)
                    set_toast(f"Snapshot Saved -> {os.path.basename(saved['html_path'])}")
                else:
                    set_toast("Snapshot skipped: No person detected")
            elif key in (ord("s"), ord("S")):
                if not session.is_recording:
                    session.start_recording()
                    set_toast("Recording Started...")
                else:
                    summary = session.stop_recording()
                    if summary.get("records"):
                        exported = session.export_session_outputs(summary, args.output, method=method, manual_modifiers=modifiers)
                        set_toast(f"Saved Peak={summary['peak_score']} -> {os.path.basename(exported.get('peak_html_path', exported['csv_path']))}", 4.0)
                    else:
                        set_toast("Recording stopped (no valid pose frames)")
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                try:
                    cv2.setWindowProperty(
                        win_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )
                except Exception:
                    pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
