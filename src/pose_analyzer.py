#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_analyzer.py
----------------
基於 YOLO11-Pose 之人體姿態估計與關節角度計算模組

特色：
1. 模型推論：整合 YOLO11-pose (17 COCO 關鍵點)，支援單人/多人目標選取 (最大框或最高信心度)。
2. 解剖幾何計算：
   - 區分矢狀面屈伸正負方向 (Flexion + vs Extension -)。
   - 支援自動判定面向方向 (Facing Direction: Left/Right)。
   - 計算上臂、前臂、手腕、頸部、軀幹與膝關節彎曲角度。
3. 左右側智能評估：
   - 支援 'auto' (自動選擇信心度高或風險高側)、'left'、'right'。
   - 自動檢測遠端遮蔽側並進行信心度警示。
4. 影片時序平滑化 (EMA)：
   - 消除骨架微幅抖動造成的臨界值跳動 (Threshold Jitter)。
5. 關鍵影格分析：
   - 影片逐幀評估、自動擷取全片最高危害瞬間 (Peak Risk Frame) 與風險等級時長佔比統計。
6. 骨架著色與可視化：
   - 依據部位風險等級 (綠/黃/紅) 即時標註關節與骨幹。
"""

import math
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional, List, Union
import cv2
import numpy as np
from ultralytics import YOLO

# 支援直接執行或作為模組載入
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.rula_reba_engine import (
        calculate_rula,
        calculate_reba,
        flatten_assessment_result,
    )
except ImportError:
    from rula_reba_engine import (
        calculate_rula,
        calculate_reba,
        flatten_assessment_result,
    )


# ==============================================================================
# COCO 17 關鍵點定義 (Ultralytics Keypoints Index)
# ==============================================================================
KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# 骨架連線關係
SKELETON_EDGES = [
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP), (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE), (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE), (KP_R_KNEE, KP_R_ANKLE)
]


# ==============================================================================
# 幾何小工具函式 (Geometry Math Helpers)
# ==============================================================================

def safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    """將 2D 向量正規化，避免除以零"""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return None
    return v / norm


def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> Optional[float]:
    """計算兩向量間的無方向夾角 (0° ~ 180°)"""
    nu = safe_unit(u)
    nv = safe_unit(v)
    if nu is None or nv is None:
        return None
    dot = float(np.clip(np.dot(nu, nv), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """計算兩點之中點"""
    return (p1 + p2) / 2.0


# ==============================================================================
# 姿態分析器核心類別 (PoseAnalyzer)
# ==============================================================================

class PoseAnalyzer:
    def __init__(self,
                 model_path: str = "HFcode/yolo11n-pose.pt",
                 conf_thres: float = 0.4,
                 ema_alpha: float = 0.35):
        """
        初始化姿態分析器
        :param model_path: YOLO11-pose 模型權重路徑
        :param conf_thres: 關鍵點信心門檻
        :param ema_alpha: 角度指數平滑係數 (0~1, 越大越偏向當前影格)
        """
        self.conf_thres = conf_thres
        self.ema_alpha = ema_alpha
        self.model = self._load_model(model_path)
        self.prev_angles: Dict[str, float] = {}

    def _load_model(self, model_path: str) -> YOLO:
        if not os.path.exists(model_path):
            fallback_paths = [
                os.path.join(os.path.dirname(__file__), "..", model_path),
                "yolo11n-pose.pt",
                "yolov8n-pose.pt"
            ]
            for p in fallback_paths:
                if os.path.exists(p):
                    model_path = p
                    break
        print(f"[PoseAnalyzer] Loading pose model from: {model_path}")
        return YOLO(model_path)

    def reset_smoother(self):
        """重設影片平滑快取"""
        self.prev_angles.clear()

    def extract_keypoints(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        從單張 BGR 影像推論人體關鍵點 (回傳最大面積目標)
        :return: (keypoints_xy [17, 2], confidences [17], bbox [4]) 或 None
        """
        results = self.model(image, conf=self.conf_thres, verbose=False)
        if not results or len(results) == 0:
            return None

        res = results[0]
        boxes = res.boxes
        kpts = res.keypoints

        if kpts is None or boxes is None or len(kpts) == 0:
            return None

        # 多人時選取最大 bounding box 者作為主要受測作業員
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_xyxy]
        best_idx = int(np.argmax(areas))

        xy = kpts.xy[best_idx].cpu().numpy()  # (17, 2)
        conf = kpts.conf[best_idx].cpu().numpy() if kpts.conf is not None else np.ones(17, dtype=np.float32)
        bbox = boxes_xyxy[best_idx]

        return xy, conf, bbox

    def estimate_facing_direction(self, xy: np.ndarray, conf: np.ndarray) -> str:
        """
        推估作業人員在畫面中的面向方向 ('right', 'left', 或 'frontal')
        利用 鼻子 vs 雙耳 / 雙肩 的水平 x 座標位移判定
        """
        nose = xy[KP_NOSE]
        l_ear, r_ear = xy[KP_L_EAR], xy[KP_R_EAR]
        l_sh, r_sh = xy[KP_L_SHOULDER], xy[KP_R_SHOULDER]

        # 參考耳部或肩部中點
        ref_x = 0.0
        valid_refs = 0
        if conf[KP_L_SHOULDER] >= self.conf_thres and conf[KP_R_SHOULDER] >= self.conf_thres:
            ref_x = (l_sh[0] + r_sh[0]) / 2.0
            valid_refs += 1
        elif conf[KP_L_EAR] >= self.conf_thres and conf[KP_R_EAR] >= self.conf_thres:
            ref_x = (l_ear[0] + r_ear[0]) / 2.0
            valid_refs += 1

        if valid_refs > 0 and conf[KP_NOSE] >= self.conf_thres:
            dx = nose[0] - ref_x
            if dx > 15.0:
                return "right"
            elif dx < -15.0:
                return "left"

        # 若雙肩寬度大於一定比例且鼻子在中間，判定為正面
        shoulder_w = abs(r_sh[0] - l_sh[0])
        if shoulder_w > 40.0:
            return "frontal"

        return "right"  # 預設面向右側

    def calculate_body_angles(self,
                              xy: np.ndarray,
                              conf: np.ndarray,
                              side: str = "auto",
                              smooth: bool = False) -> Tuple[Dict[str, float], str, Dict[str, Any]]:
        """
        從 17 關鍵點計算人體工學關節角度 (具備正負方向性)

        :param xy: (17, 2) 座標
        :param conf: (17,) 信心度
        :param side: 'auto', 'left', 'right'
        :param smooth: 是否套用時序平滑化 (EMA)
        :return: (angles_dict, side_used, diagnostics)
        """
        facing = self.estimate_facing_direction(xy, conf)

        # 判斷左右側信心度
        left_pts = [KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST, KP_L_HIP, KP_L_KNEE]
        right_pts = [KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST, KP_R_HIP, KP_R_KNEE]

        left_conf = float(np.mean([conf[i] for i in left_pts]))
        right_conf = float(np.mean([conf[i] for i in right_pts]))

        if side == "auto":
            # 優先選擇信心度較高側 (避免側面拍攝時被身體遮擋的手臂飄移)
            side_used = "left" if left_conf >= right_conf else "right"
        else:
            side_used = side.lower()

        is_left = (side_used == "left")
        sh_idx = KP_L_SHOULDER if is_left else KP_R_SHOULDER
        el_idx = KP_L_ELBOW if is_left else KP_R_ELBOW
        wr_idx = KP_L_WRIST if is_left else KP_R_WRIST
        hip_idx = KP_L_HIP if is_left else KP_R_HIP
        knee_idx = KP_L_KNEE if is_left else KP_R_KNEE
        ank_idx = KP_L_ANKLE if is_left else KP_R_ANKLE

        # 前向方向向量 (Forward sign for Flexion/Extension)
        # 影像座標中，x 向右為正；若面向 right，forward_x 為 +1，若面向 left，forward_x 為 -1
        fwd_sign = 1.0 if facing == "right" else (-1.0 if facing == "left" else 1.0)

        # 1. 軀幹向量 (Trunk Vector: Pelvis -> Shoulder_mid)
        pelvis = midpoint(xy[KP_L_HIP], xy[KP_R_HIP])
        shoulder_mid = midpoint(xy[KP_L_SHOULDER], xy[KP_R_SHOULDER])
        v_trunk = shoulder_mid - pelvis  # 從髖指向上

        # 向上垂直向量 (Vertical Up in image coords is [0, -1])
        v_up = np.array([0.0, -1.0], dtype=np.float32)
        trunk_mag = angle_between_vectors(v_trunk, v_up) or 0.0

        # 軀幹前屈 vs 後仰判定: 檢查肩中點相對骨盆在前向的偏移
        trunk_dx = (shoulder_mid[0] - pelvis[0]) * fwd_sign
        trunk_angle = trunk_mag if trunk_dx >= 0 else -trunk_mag

        # 2. 頸部向量 (Neck Vector: Shoulder_mid -> Head)
        head_pt = xy[KP_NOSE]
        if conf[KP_NOSE] < self.conf_thres:
            head_pt = midpoint(xy[KP_L_EAR], xy[KP_R_EAR])
        v_neck = head_pt - shoulder_mid
        neck_mag = angle_between_vectors(v_neck, v_up) or 0.0
        # 頸部相對軀幹之夾角
        neck_relative = angle_between_vectors(v_neck, v_trunk) or 0.0
        neck_dx = (head_pt[0] - shoulder_mid[0]) * fwd_sign
        neck_angle = neck_relative if neck_dx >= 0 else -neck_relative

        # 3. 上臂屈伸 (Upper Arm Flexion/Extension)
        # 向量: 肩膀 -> 手肘
        v_upper_arm = xy[el_idx] - xy[sh_idx]
        # 參考軀幹向下方向
        v_trunk_down = -v_trunk
        ua_mag = angle_between_vectors(v_upper_arm, v_trunk_down) or 0.0
        ua_dx = (xy[el_idx][0] - xy[sh_idx][0]) * fwd_sign
        upper_arm_angle = ua_mag if ua_dx >= 0 else -ua_mag

        # 4. 前臂屈曲 (Lower Arm Angle / Elbow Flexion)
        # 向量: 手肘 -> 手腕 與 手肘 -> 肩膀
        v_forearm = xy[wr_idx] - xy[el_idx]
        v_arm_reverse = xy[sh_idx] - xy[el_idx]
        elbow_angle = angle_between_vectors(v_forearm, v_arm_reverse)
        lower_arm_angle = elbow_angle if elbow_angle is not None else 90.0

        # 5. 手腕角度 (Wrist Angle)
        # 由於 COCO-17 僅標註到手腕點，手腕屈曲若無法精準辨識則預設為接近中立位 (0°)
        wrist_angle = 0.0

        # 6. 膝關節彎曲 (Knee Flexion)
        # 站立伸直時夾角 180° -> 屈曲 0°；深蹲彎曲 90° -> 屈曲 90°
        v_thigh = xy[hip_idx] - xy[knee_idx]
        v_shank = xy[ank_idx] - xy[knee_idx]
        leg_joint_angle = angle_between_vectors(v_thigh, v_shank)
        knee_angle = max(0.0, 180.0 - leg_joint_angle) if leg_joint_angle is not None else 0.0

        raw_angles = {
            "trunk": float(round(trunk_angle, 1)),
            "neck": float(round(neck_angle, 1)),
            "upper_arm": float(round(upper_arm_angle, 1)),
            "lower_arm": float(round(lower_arm_angle, 1)),
            "wrist": float(round(wrist_angle, 1)),
            "knee": float(round(knee_angle, 1)),
        }

        # 時序平滑 (EMA) 避免跳動
        if smooth and self.prev_angles:
            smoothed = {}
            for k, val in raw_angles.items():
                prev_val = self.prev_angles.get(k, val)
                smoothed[k] = float(round(self.ema_alpha * val + (1.0 - self.ema_alpha) * prev_val, 1))
            self.prev_angles = smoothed
            final_angles = smoothed
        else:
            self.prev_angles = raw_angles.copy()
            final_angles = raw_angles

        diagnostics = {
            "facing": facing,
            "side_used": side_used,
            "left_confidence": round(left_conf, 2),
            "right_confidence": round(right_conf, 2),
            "occluded_warning": (left_conf < self.conf_thres if is_left else right_conf < self.conf_thres)
        }

        return final_angles, side_used, diagnostics

    def analyze_image(self,
                      image_bgr: np.ndarray,
                      side: str = "auto",
                      method: str = "RULA",
                      manual_modifiers: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        完整分析單張影像：姿態偵測 -> 角度計算 -> RULA/REBA 評分 -> 繪製骨架

        :param image_bgr: 輸入 BGR 影像
        :param side: 'auto', 'left', 'right'
        :param method: 'RULA' 或 'REBA'
        :param manual_modifiers: 專家負重/環境參數
        :return: 包含標註影像、角度、評分結果之字典
        """
        detection = self.extract_keypoints(image_bgr)
        if detection is None:
            return None

        xy, conf, bbox = detection
        angles, side_used, diag = self.calculate_body_angles(xy, conf, side=side, smooth=False)

        # 執行查表評估
        if method.upper() == "REBA":
            assessment_result = calculate_reba(angles, manual_modifiers)
            score = assessment_result["final_reba_score"]
            level = assessment_result["action_level"]
        else:
            assessment_result = calculate_rula(angles, manual_modifiers)
            score = assessment_result["grand_score"]
            level = assessment_result["action_level"]

        # 標註骨架疊圖
        annotated_img = self.render_overlay(image_bgr, xy, conf, side_used, assessment_result, angles)

        return {
            "assessment_result": assessment_result,
            "flattened": flatten_assessment_result(assessment_result),
            "angles": angles,
            "diagnostics": diag,
            "annotated_image": annotated_img,
            "keypoints": xy,
            "confidences": conf,
            "bbox": bbox
        }

    def render_overlay(self,
                       img: np.ndarray,
                       xy: np.ndarray,
                       conf: np.ndarray,
                       side_used: str,
                       assessment: Dict[str, Any],
                       angles: Dict[str, float]) -> np.ndarray:
        """在影像上繪製半透明資訊看板與人體骨架"""
        out = img.copy()
        h, w = out.shape[:2]

        # 依風險水準決定顏色 (BGR)
        level = assessment["action_level"]
        if level <= 1:
            risk_color = (0, 200, 0)      # 綠色 (Low)
        elif level == 2:
            risk_color = (0, 215, 255)    # 黃/金 (Medium)
        elif level == 3:
            risk_color = (0, 140, 255)    # 橘色 (High)
        else:
            risk_color = (0, 0, 255)      # 紅色 (Very High)

        # 1. 繪製骨架連線
        for (i1, i2) in SKELETON_EDGES:
            if conf[i1] >= self.conf_thres and conf[i2] >= self.conf_thres:
                pt1 = (int(xy[i1][0]), int(xy[i1][1]))
                pt2 = (int(xy[i2][0]), int(xy[i2][1]))
                cv2.line(out, pt1, pt2, (240, 240, 240), 2, cv2.LINE_AA)

        # 2. 標註使用側的關節點 (彩色醒目顯示)
        is_left = (side_used == "left")
        active_indices = [
            KP_NOSE,
            KP_L_SHOULDER if is_left else KP_R_SHOULDER,
            KP_L_ELBOW if is_left else KP_R_ELBOW,
            KP_L_WRIST if is_left else KP_R_WRIST,
            KP_L_HIP if is_left else KP_R_HIP,
            KP_L_KNEE if is_left else KP_R_KNEE,
        ]

        for idx in active_indices:
            if conf[idx] >= self.conf_thres:
                pt = (int(xy[idx][0]), int(xy[idx][1]))
                cv2.circle(out, pt, 6, risk_color, -1, cv2.LINE_AA)
                cv2.circle(out, pt, 8, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. 繪製 HUD 數據抬頭資訊看板 (左上角半透明背景)
        overlay = out.copy()
        cv2.rectangle(overlay, (15, 15), (380, 210), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
        cv2.rectangle(out, (15, 15), (380, 210), risk_color, 2)

        method = assessment["assessment"]
        score_val = assessment.get("final_reba_score") if method == "REBA" else assessment.get("grand_score")
        risk_str = assessment["risk_level"].split()[0]

        cv2.putText(out, f"{method} Assessment: {score_val} (AL {level})", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"Status: {risk_str}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, risk_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"Eval Side: {side_used.upper()}", (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # 關節角度摘要
        ang_text_1 = f"Trunk: {angles['trunk']:+.1f} deg | Neck: {angles['neck']:+.1f} deg"
        ang_text_2 = f"Upper Arm: {angles['upper_arm']:+.1f} deg | Elbow: {angles['lower_arm']:.1f} deg"
        ang_text_3 = f"Knee: {angles['knee']:.1f} deg"
        cv2.putText(out, ang_text_1, (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(out, ang_text_2, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(out, ang_text_3, (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        return out

    def analyze_video(self,
                      video_path: str,
                      method: str = "RULA",
                      side: str = "auto",
                      manual_modifiers: Optional[Dict[str, Any]] = None,
                      sample_interval: int = 1,
                      output_annotated_path: Optional[str] = None) -> Dict[str, Any]:
        """
        處理影片時間序列：逐幀分析、EMA 平滑、統計時長佔比、偵測最高風險瞬間 (Peak Risk Frame)

        :param video_path: 輸入影片檔路徑
        :param method: 'RULA' 或 'REBA'
        :param side: 'auto', 'left', 'right'
        :param manual_modifiers: 專家負重/環境參數
        :param sample_interval: 每隔幾幀抽樣一次 (1 為逐幀)
        :param output_annotated_path: 標註輸出影片儲存路徑 (選填)
        :return: 包含時序清單、風險統計、Peak Frame 資訊之結果
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片檔案: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_annotated_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_annotated_path, fourcc, fps / sample_interval, (width, height))

        self.reset_smoother()

        frame_records: List[Dict[str, Any]] = []
        peak_score = -1
        peak_frame_record = None
        peak_frame_image = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                det = self.extract_keypoints(frame)
                if det is not None:
                    xy, conf, bbox = det
                    angles, side_used, diag = self.calculate_body_angles(xy, conf, side=side, smooth=True)

                    if method.upper() == "REBA":
                        res = calculate_reba(angles, manual_modifiers)
                        score = res["final_reba_score"]
                        level = res["action_level"]
                    else:
                        res = calculate_rula(angles, manual_modifiers)
                        score = res["grand_score"]
                        level = res["action_level"]

                    annotated = self.render_overlay(frame, xy, conf, side_used, res, angles)
                    if writer:
                        writer.write(annotated)

                    rec = {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "score": score,
                        "action_level": level,
                        "side_used": side_used,
                        **flatten_assessment_result(res)
                    }
                    frame_records.append(rec)

                    # 追蹤最高風險影格 (Peak Risk Frame)
                    if score > peak_score:
                        peak_score = score
                        peak_frame_record = rec
                        peak_frame_image = annotated.copy()

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        # 計算風險等級佔比統計 (Risk Distribution)
        total_eval_frames = len(frame_records)
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for r in frame_records:
            lvl = r["action_level"]
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        level_percentages = {
            f"AL_{lvl}": round((cnt / total_eval_frames) * 100.0, 1) if total_eval_frames > 0 else 0.0
            for lvl, cnt in level_counts.items()
        }

        # 高風險以上 (AL 3 + AL 4) 佔比
        high_risk_pct = round(((level_counts.get(3, 0) + level_counts.get(4, 0)) / max(1, total_eval_frames)) * 100.0, 1)

        return {
            "total_frames_processed": total_eval_frames,
            "fps": fps,
            "duration_sec": round(total_frames / fps, 2),
            "peak_risk_score": peak_score,
            "peak_risk_record": peak_frame_record,
            "peak_risk_image": peak_frame_image,
            "risk_distribution_pct": level_percentages,
            "high_risk_exposure_pct": high_risk_pct,
            "time_series_records": frame_records
        }


# ==============================================================================
# 自我測試入口
# ==============================================================================
if __name__ == "__main__":
    print("=== Testing PoseAnalyzer on Sample Image ===")
    test_img_path = "六人合照_姓名標示.png"
    if os.path.exists(test_img_path):
        analyzer = PoseAnalyzer()
        img = cv2.imread(test_img_path)
        res = analyzer.analyze_image(img, side="auto", method="RULA")
        if res:
            print(f"RULA Grand Score: {res['assessment_result']['grand_score']}")
            print(f"Estimated Angles: {res['angles']}")
            print(f"Diagnostics: {res['diagnostics']}")
            print("Successfully evaluated sample image!")
        else:
            print("No person detected.")
    else:
        print(f"Sample image not found: {test_img_path}")
