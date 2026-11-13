#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 14:33:56 2025

@author: erictang
"""
import cv2
import math
import time
import numpy as np

# 嘗試載入 Ultralytics YOLO Pose
from ultralytics import YOLO

# -------------- 參數 --------------
CONF_THRES = 0.5          # 偵測/關鍵點信心門檻
SMOOTH_ALPHA = 0.3        # 角度指數平滑 (0~1，越大越追新)
CAM_INDEX = 0             # webcam 編號
CAP_W, CAP_H = 1280, 720  # 解析度

# -------------- 小工具函式 --------------
def safe_unit(v):
    """將向量正規化，避免除以零"""
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n

def angle_between(u, v):
    """回傳兩向量的夾角(度)；若有缺失回傳 None"""
    if u is None or v is None:
        return None
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def midpoint(a, b):
    return (a + b) / 2.0

def draw_text(img, text, org, scale=0.9, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def try_load_model():
    """優先載入 YOLO11 pose；失敗則退回 YOLO8 pose。"""
    candidates = ["yolo11n-pose.pt", "yolov8n-pose.pt"]
    last_err = None
    for w in candidates:
        try:
            m = YOLO(w)
            # 確認是 pose task
            if getattr(m, "task", None) == "pose":
                return m, w
        except Exception as e:
            last_err = e
    raise RuntimeError(f"無法載入 YOLO pose 權重：{last_err}")

# -------------- 主程式 --------------
def main():
    model, weight_name = try_load_model()
    print(f"[INFO] Loaded pose model: {weight_name}")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    if not cap.isOpened():
        raise RuntimeError("打不開攝影機")

    # 平滑角度（左右各一）
    smooth_left = None
    smooth_right = None
    prev_t = time.time()

    # COCO 17 關鍵點索引（Ultralytics）
    # 0:nose, 5:LS, 6:RS, 11:LH, 12:RH, 13:LK, 14:RK, 15:LA, 16:RA
    LS, RS, LH, RH, LK, RK = 5, 6, 11, 12, 13, 14

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 推論（關閉詳細輸出）
        res = model(frame, conf=CONF_THRES, verbose=False)[0]

        # 畫出模型自帶的骨架疊圖（含多人）
        annotated = res.plot()  # 回傳 BGR image

        # 若有偵測到任何人，選擇最大框的人做角度計算
        kpts = res.keypoints
        boxes = res.boxes
        if kpts is not None and boxes is not None and len(kpts) > 0:
            xy = kpts.xy  # (num_person, 17, 2)
            confs = getattr(kpts, "conf", None)  # (num_person, 17)

            # 取最大面積的人
            areas = []
            for b in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b[:4]
                areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
            best_idx = int(np.argmax(areas))

            person_xy = xy[best_idx].cpu().numpy()  # (17,2)
            person_conf = confs[best_idx].cpu().numpy() if confs is not None else np.ones((17,), dtype=np.float32)

            # 取所需關鍵點（若信心過低則略過）
            def ok_pt(i): 
                return person_conf[i] >= CONF_THRES

            have_shoulders = ok_pt(LS) and ok_pt(RS)
            have_hips = ok_pt(LH) and ok_pt(RH)
            have_knees = ok_pt(LK) and ok_pt(RK)

            if have_shoulders and have_hips and have_knees:
                p_LS = person_xy[LS]; p_RS = person_xy[RS]
                p_LH = person_xy[LH]; p_RH = person_xy[RH]
                p_LK = person_xy[LK]; p_RK = person_xy[RK]

                # 以骨盆中心(左右髖的中點)近似 L5/S1
                pelvis = midpoint(p_LH, p_RH)
                # 肩線中心代表軀幹上端
                shoulder_mid = midpoint(p_LS, p_RS)

                # 軀幹向量： pelvis -> shoulder_mid
                v_trunk = safe_unit(shoulder_mid - pelvis)
                # 大腿向量（以髖->膝代表）
                v_thigh_L = safe_unit(p_LK - p_LH)
                v_thigh_R = safe_unit(p_RK - p_RH)

                ang_L = angle_between(v_trunk, v_thigh_L)
                ang_R = angle_between(v_trunk, v_thigh_R)

                # 指數平滑
                if ang_L is not None:
                    smooth_left = ang_L if smooth_left is None else (SMOOTH_ALPHA * ang_L + (1 - SMOOTH_ALPHA) * smooth_left)
                if ang_R is not None:
                    smooth_right = ang_R if smooth_right is None else (SMOOTH_ALPHA * ang_R + (1 - SMOOTH_ALPHA) * smooth_right)

                # 視覺強化：畫出軀幹主向量
                p = pelvis.astype(int)
                s = shoulder_mid.astype(int)
                cv2.line(annotated, p, s, (0, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(annotated, p, 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, s, 6, (0, 255, 255), -1, cv2.LINE_AA)

                # 顯示角度（取整數）
                if smooth_left is not None:
                    draw_text(annotated, f"Trunk–Left Thigh: {int(round(smooth_left))} deg", (20, 50), scale=0.9, color=(0, 255, 0))
                if smooth_right is not None:
                    draw_text(annotated, f"Trunk–Right Thigh: {int(round(smooth_right))} deg", (20, 90), scale=0.9, color=(0, 200, 255))

                # 額外：軀幹相對垂直的外傾角（前視左右擺動用）
                v_up = np.array([0.0, -1.0], dtype=np.float32)  # 影像座標 y 向下，所以「上」是 (0,-1)
                tilt = angle_between(v_trunk, v_up)
                if tilt is not None:
                    draw_text(annotated, f"Trunk Tilt (frontal): {int(round(tilt))} deg", (20, 130), scale=0.9, color=(255, 255, 0))

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_t) if now != prev_t else 0.0
        prev_t = now
        draw_text(annotated, f"{fps:.1f} FPS", (annotated.shape[1]-150, 40), scale=0.8, color=(200,200,255))

        cv2.imshow("Trunk vs Lower-Limb Angle (Front View) - Press q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
