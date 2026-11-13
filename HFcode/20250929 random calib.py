#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time, json, os, socket, random
from collections import deque

# ========== 可選：讀螢幕尺寸 ==========
try:
    import pyautogui
    SCREEN_SIZE = pyautogui.size()
except Exception:
    SCREEN_SIZE = None

# ========== 依賴：MediaPipe ==========
try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("請先安裝 mediapipe： pip install mediapipe")

# ========== 檔案與串流設定 ==========
CALIB_PATH   = "calib_headpose3.json"
STREAM_HOST  = "127.0.0.1"   # 目標主機（同機就 127.0.0.1）
STREAM_PORT  = 5005          # 目標埠號（自行調整）
STREAM_FMT   = "{:.1f},{:.1f}"  # 傳送格式 (x,y)

# ========== 參數 ==========
FONT_SCALE = 0.9
THICK      = 2

OPEN_THRESH_DEFAULT  = 0.24
CLOSE_THRESH_DEFAULT = 0.20
MIRROR_START         = True

# 特徵平滑（在 rx/ry 空間）
MEDIAN_WIN  = 5
EMA_ALPHA_BASE = 0.30
EMA_ALPHA_MAX  = 0.60

# 視覺化與校正
DRAW_HZ     = 20
RIDGE_LAMBDA= 1e-3
MIN_SAMPLES_PER_POINT = 1      # 你希望每個點校正一次 → 設為 1
TOTAL_POINTS = 30              # 本次隨機點數（可改）
CAM_W, CAM_H = 1280, 720
VERT_GAIN = 1.25

# 時序穩定（螢幕座標空間）
USE_KALMAN   = True
DEADBAND_PX  = 8
VEL_NORM     = 60.0  # px/s

# FaceMesh 索引
R_EYE_EAR = [33,160,158,133,153,144]
L_EYE_EAR = [263,387,385,362,380,373]
R_EYE_BOX = [33,133,160,159,158,144,145,153,154,155,246]
L_EYE_BOX = [362,263,387,386,385,373,374,380,381,382,466]
R_IRIS = [469,470,471,472]
L_IRIS = [474,475,476,477]

# ========== 小工具 ==========
def eye_aspect_ratio(pts):
    p1,p2,p3,p4,p5,p6 = pts
    A = np.linalg.norm(p2-p6); B = np.linalg.norm(p3-p5); C = np.linalg.norm(p1-p4)
    return 0. if C < 1e-6 else (A+B)/(2.0*C)

def bbox_from_landmarks(lm, idxs, w, h, pad=3):
    pts = np.array([(lm[i][0]*w, lm[i][1]*h) for i in idxs], np.float32)
    x1,y1 = np.min(pts, axis=0); x2,y2 = np.max(pts, axis=0)
    return (max(int(x1-pad),0), max(int(y1-pad),0), min(int(x2+pad),w-1), min(int(y2+pad),h-1))

def darkest_center(gray, box):
    x1,y1,x2,y2 = box
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: return None
    roi = cv2.equalizeHist(roi)
    roi = cv2.GaussianBlur(roi, (7,7), 0)
    _, _, minLoc, _ = cv2.minMaxLoc(roi)
    return (x1 + minLoc[0], y1 + minLoc[1])

def clamp01(v): return max(0.0, min(1.0, v))

def per_eye_norm(cx, cy, box):
    x1,y1,x2,y2 = box
    w = max(x2 - x1, 1); h = max(y2 - y1, 1)
    rx = (cx - x1) / w
    ry = (cy - y1) / h
    return float(np.clip(rx, 0, 1)), float(np.clip(ry, 0, 1))

def headpose_proxies(lm_pix, W, H):
    R_OUT = 33; L_OUT = 263; NOSE = 1; M_L = 61; M_R = 291
    try:
        rx, ry = lm_pix[R_OUT]; lx, ly = lm_pix[L_OUT]
        nx, ny = lm_pix[NOSE];   mxl, myl = lm_pix[M_L]; mxr, myr = lm_pix[M_R]
    except Exception:
        return 0.0, 0.0, 0.0, 0.5, 0.5, 0.0
    xs = [p[0] for p in lm_pix]; ys = [p[1] for p in lm_pix]
    fx1, fy1, fx2, fy2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    fw, fh = max(fx2 - fx1, 1), max(fy2 - fy1, 1)
    fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    face_cx_norm = fcx / W; face_cy_norm = fcy / H
    face_size = np.sqrt((fw * fh) / float(W * H))
    r_rel = (rx - fcx) / fw; l_rel = (lx - fcx) / fw
    yaw_proxy = float(np.clip((r_rel - l_rel), -1.0, 1.0))
    eye_y = (ry + ly) / 2.0; mouth_cy = (myl + myr) / 2.0
    pitch_raw = (mouth_cy - eye_y) / fh
    pitch_proxy = float(np.clip((pitch_raw - 0.22) * 3.0, -1.0, 1.0))
    dx = max(abs(lx - rx), 1e-3); dy = (ly - ry)
    roll_angle = np.degrees(np.arctan2(dy, dx))
    roll_proxy = float(np.clip(roll_angle / 25.0, -1.0, 1.0))
    return yaw_proxy, pitch_proxy, roll_proxy, float(face_cx_norm), float(face_cy_norm), float(face_size)

def feature_vec(rx, ry, yaw, pitch, roll, fc_x, fc_y, fsize):
    return np.array([
        1.0,
        rx, ry,
        yaw, pitch, roll,
        fc_x, fc_y, fsize,
        rx*ry, rx*yaw, ry*pitch, yaw*roll, pitch*roll,
        rx*fc_x, ry*fc_y, fsize*rx, fsize*ry,
        rx*rx, ry*ry, yaw*yaw, pitch*pitch, roll*roll,
    ], dtype=np.float32)

def fit_ridge_weighted(target, feats, weights, lam=RIDGE_LAMBDA):
    W = np.diag(weights.astype(np.float32))
    XtWX = feats.T @ W @ feats
    reg  = lam * np.eye(XtWX.shape[0], dtype=np.float32)
    XtWy = feats.T @ W @ target
    return np.linalg.solve(XtWX + reg, XtWy)

def predict_xy(rx, ry, yaw, pitch, roll, fcx, fcy, fsize, wx, wy):
    f = feature_vec(rx, ry, yaw, pitch, roll, fcx, fcy, fsize).reshape(1, -1)
    y_pred_x = (f @ wx).item()
    y_pred_y = (f @ wy).item()
    return float(y_pred_x), float(y_pred_y)

# ========== 產生「一次一個」的隨機校正點（偏極端） ==========
def sample_one_point(margin=0.06):
    """回傳 (nx, ny) 單一目標點；用 Beta(0.35,0.35) 讓點常在邊緣附近。"""
    nx = float(np.clip(np.random.beta(0.35, 0.35), margin, 1.0 - margin))
    ny = float(np.clip(np.random.beta(0.35, 0.35), margin, 1.0 - margin))
    return nx, ny

# ========== Kalman 常速模型 ==========
class Kalman2D:
    def __init__(self, dt=1/30.0, q=0.05, r=9.0):
        self.A = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0 ],
                           [0,0,0,1 ]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * q
        self.R = np.eye(2, dtype=np.float32) * r
        self.x = np.zeros((4,1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.ready = False
    def reset(self): self.__init__(dt=float(self.A[0,2]), q=float(self.Q[0,0]), r=float(self.R[0,0]))
    def update_dt(self, dt):
        self.A[0,2] = dt; self.A[1,3] = dt
    def step(self, z):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        z = z.reshape(2,1).astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        self.ready = True
        return float(self.x[0, 0].item()), float(self.x[1, 0].item())

# ========== 校正檔存/讀 ==========
def try_load_calib(path):
    if not os.path.exists(path): return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
        wx = np.array(d["wx"], dtype=np.float32).reshape(-1,1)
        wy = np.array(d["wy"], dtype=np.float32).reshape(-1,1)
        meta = d.get("meta", {})
        return wx, wy, meta
    except Exception as e:
        print(f"[Calib] 讀取失敗：{e}")
        return None

def save_calib(path, wx, wy, meta=None):
    try:
        data = {"wx": wx.flatten().tolist(), "wy": wy.flatten().tolist(), "meta": meta or {}}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Calib] Saved -> {path}")
    except Exception as e:
        print(f"[Calib] 儲存失敗：{e}")

# ========== 主程式 ==========
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise SystemExit("無法開啟攝影機 index=0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    win_name = "Gaze (single random target calibration) — stabilized + stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # === 預設全螢幕 ===
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        if SCREEN_SIZE:
            sw, sh = SCREEN_SIZE
            cv2.resizeWindow(win_name, sw, sh)
            cv2.moveWindow(win_name, 0, 0)

    # 狀態
    mirror = MIRROR_START
    ema_base  = EMA_ALPHA_BASE
    rx_q, ry_q = deque(maxlen=MEDIAN_WIN), deque(maxlen=MEDIAN_WIN)
    ema_rx = ema_ry = None

    calibrated = False
    wx = wy = None

    # 校正狀態：一次只出現一個點；完成後留下痕跡
    cal_mode = False
    cur_target = None        # (nx, ny) 目前要看的目標
    done_points = []         # 已完成的目標點 [(nx, ny)]
    samples_feats = []       # 與 done_points 對齊：每點的樣本列表（此版每點只需1筆）
    total_goal = TOTAL_POINTS

    last_valid = None
    last_draw = 0.0

    # 螢幕空間穩定化
    kf = Kalman2D(dt=1/30.0, q=0.05, r=9.0)
    use_kalman = USE_KALMAN
    deadband = DEADBAND_PX
    x_out = y_out = None
    last_time = time.time()

    # 串流
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    streaming = False  # T 開/關

    # 啟動時嘗試載入校正
    loaded = try_load_calib(CALIB_PATH)
    if loaded:
        wx, wy, meta = loaded
        calibrated = True
        print(f"[Calib] Loaded <- {CALIB_PATH}  (meta: {meta})")

    print("[Hotkeys] S:start calib(random single) | Space:capture | Enter:fit(save) | R:mirror | O:baseline | [/]:EMA | K:Kalman | ,/.:deadband +/- | T:stream | F:fullscreen | Q/Esc:quit")
    fullscreen = True

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = face_mesh.process(rgb)

        cv2.putText(frame,
            f"S:start  Space:capture  Enter:finish  R  O  [/]:EMA({ema_base:.2f})  K:{'ON' if use_kalman else 'OFF'}  ,/. Dead:{deadband}px  T:{'ON' if streaming else 'OFF'}  F:{'ON' if fullscreen else 'OFF'}",
            (10, int(30*FONT_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,255,0), THICK, cv2.LINE_AA)

        detected = False
        iris_ok = False
        ear = None

        if res.multi_face_landmarks:
            detected = True
            lms = res.multi_face_landmarks[0].landmark
            lm_pix = [(float(p.x)*W, float(p.y)*H) for p in lms]
            lm_norm= [(float(p.x),    float(p.y))   for p in lms]

            r_box = bbox_from_landmarks(lm_norm, R_EYE_BOX, W, H)
            l_box = bbox_from_landmarks(lm_norm, L_EYE_BOX, W, H)
            cv2.rectangle(frame, (r_box[0], r_box[1]), (r_box[2], r_box[3]), (0,255,0), 2)
            cv2.rectangle(frame, (l_box[0], l_box[1]), (l_box[2], l_box[3]), (0,255,0), 2)

            r_pts = np.array([(lm_pix[i][0], lm_pix[i][1]) for i in R_EYE_EAR], np.float32)
            l_pts = np.array([(lm_pix[i][0], lm_pix[i][1]) for i in L_EYE_EAR], np.float32)
            ear = (eye_aspect_ratio(r_pts)+eye_aspect_ratio(l_pts))/2.0

            # 瞳孔中心（優先虹膜，fallback 暗區）
            def iris_center(idxs):
                pts = np.array([(lm_pix[i][0], lm_pix[i][1]) for i in idxs], np.float32)
                c = np.mean(pts, axis=0)
                return (int(c[0]), int(c[1]))
            ri = iris_center(R_IRIS); li = iris_center(L_IRIS)
            def in_box(pt, box):
                x,y = pt; x1,y1,x2,y2 = box
                return (x1 <= x <= x2) and (y1 <= y <= y2)
            if in_box(ri, r_box) and in_box(li, l_box):
                cx = int((ri[0]+li[0])//2); cy = int((ri[1]+li[1])//2)
                iris_ok = True
            else:
                rp = darkest_center(gray, r_box); lp = darkest_center(gray, l_box)
                if rp and lp: cx = int((rp[0]+lp[0])//2); cy = int((rp[1]+lp[1])//2)
                else:         cx = int((r_box[0]+l_box[2])//2); cy = int((r_box[1]+l_box[3])//2)

            # 瞳孔白色圓環 + 中心點
            ex1 = min(r_box[0], l_box[0]); ey1 = min(r_box[1], l_box[1])
            ex2 = max(r_box[2], l_box[2]); ey2 = max(r_box[3], l_box[3])
            ew, eh = max(ex2-ex1,1), max(ey2-ey1,1)
            r_ring = int(np.clip(0.06 * min(ew, eh), 5, 18))
            cv2.circle(frame, (cx, cy), r_ring, (255,255,255), 2)
            cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)

            # 以每眼框正規化 → rx,ry（含垂直增益 + 鏡像）
            rx_r, ry_r = per_eye_norm(cx, cy, r_box)
            rx_l, ry_l = per_eye_norm(cx, cy, l_box)
            rx_raw = (rx_r + rx_l) * 0.5
            ry_raw = (ry_r + ry_l) * 0.5
            ry_raw = float(np.clip((ry_raw - 0.5) * VERT_GAIN + 0.5, 0.0, 1.0))
            rx_raw = clamp01(1.0 - rx_raw if MIRROR_START and mirror else rx_raw)

            # 雙層濾波：median -> 自適應 EMA
            rx_q.append(rx_raw); ry_q.append(ry_raw)
            rx_med = float(np.median(rx_q)); ry_med = float(np.median(ry_q))
            if ema_rx is None:
                ema_rx, ema_ry = rx_med, ry_med
            else:
                v_feat = np.hypot(rx_med - ema_rx, ry_med - ema_ry)
                a = EMA_ALPHA_BASE + (EMA_ALPHA_MAX - EMA_ALPHA_BASE) * np.clip(v_feat*5.0, 0.0, 1.0)
                ema_rx = a*rx_med + (1-a)*ema_rx
                ema_ry = a*ry_med + (1-a)*ema_ry

            # 頭部代理特徵
            yaw_p, pitch_p, roll_p, fc_x, fc_y, fsize = headpose_proxies(lm_pix, W, H)
            last_valid = (ema_rx, ema_ry, yaw_p, pitch_p, roll_p, fc_x, fc_y, fsize)

            # 狀態列
            ear_txt = "EAR=--" if ear is None else f"EAR={ear:.2f}"
            cv2.putText(frame, f"IrisOK:{'ON' if iris_ok else 'OFF'}  {ear_txt}",
                        (10, int(60*FONT_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                        ((0,255,0) if iris_ok else (0,0,255)), THICK, cv2.LINE_AA)

        # ===== 螢幕座標估計 + 串流 =====
        now = time.time()
        if calibrated and last_valid and (wx is not None) and (now - last_draw) >= (1.0/DRAW_HZ):
            last_draw = now
            sw, sh = (SCREEN_SIZE if SCREEN_SIZE else (W, H))
            rx, ry, yaw_p, pitch_p, roll_p, fc_x, fc_y, fsize = last_valid
            x_raw, y_raw = predict_xy(rx, ry, yaw_p, pitch_p, roll_p, fc_x, fc_y, fsize, wx, wy)

            # Kalman
            dt = now - last_time if last_time else 1/30.0
            last_time = now
            if USE_KALMAN:
                kf.update_dt(dt)
                x_kf, y_kf = kf.step(np.array([x_raw, y_raw], dtype=np.float32))
            else:
                x_kf, y_kf = x_raw, y_raw

            # 死區穩定
            if x_out is None:
                x_out, y_out = x_kf, y_kf
            else:
                if np.hypot(x_kf - x_out, y_kf - y_out) >= DEADBAND_PX:
                    x_out, y_out = x_kf, y_kf

            vis_x = int(np.clip(x_out / sw * W, 0, W-1))
            vis_y = int(np.clip(y_out / sh * H, 0, H-1))
            cv2.drawMarker(frame, (vis_x, vis_y), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, thickness=THICK, markerSize=28)

            # 串流輸出
            # （保留原本設計）
            # 你可用 streaming 開/關
            # if streaming:
            #     try:
            #         msg = STREAM_FMT.format(x_out, y_out).encode()
            #         sock.sendto(msg, (STREAM_HOST, STREAM_PORT))
            #     except Exception:
            #         pass

        # ===== 校正 UI：顯示已完成的點（實心）、目前目標（粗框空心） =====
        if cal_mode:
            # 畫已完成的點（持續留在螢幕）
            for nx, ny in done_points:
                px, py = int(nx * W), int(ny * H)
                cv2.circle(frame, (px, py), 9, (0,165,255), -1, cv2.LINE_AA)  # 實心橘點

            # 畫目前目標點（只有一個）
            if cur_target is not None:
                nx, ny = cur_target
                px, py = int(nx * W), int(ny * H)
                cv2.circle(frame, (px, py), 12, (0,165,255), 3, cv2.LINE_AA)  # 粗框
                cv2.circle(frame, (px, py), 8,  (0,165,255), 2, cv2.LINE_AA)  # 空心

            cv2.putText(frame,
                        f"Calib {len(done_points)}/{total_goal}  (Space=capture; Enter=finish; S=restart)",
                        (10, int(150*FONT_SCALE)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,165,255), THICK, cv2.LINE_AA)

            if cur_target is None and len(done_points) < total_goal:
                # 產生下一個隨機目標（避免離螢幕太近）
                cur_target = sample_one_point(margin=0.06)

        if not detected:
            cv2.putText(frame, "NO FACE DETECTED", (10, H-20),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE*1.2, (0,0,255), THICK+1, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # ====== 熱鍵處理 ======
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            mirror = not mirror
        elif key in (ord('s'), ord('S')):
            # 重新開始此次校正（清空已完成點），一次一個隨機目標
            cal_mode = True
            done_points = []
            samples_feats = []
            cur_target = sample_one_point(margin=0.06)
            calibrated = False
            wx = wy = None
            rx_q.clear(); ry_q.clear(); ema_rx = ema_ry = None
            last_valid = None; x_out = y_out = None; kf.reset()
            print(f"[Calib] 單點隨機校正開始。總共 {total_goal} 點；每點按 Space 一次，完成後按 Enter。")
        elif key == 32:  # Space：收樣當前點 -> 固定住，並自動換下一點
            if cal_mode and cur_target is not None and last_valid is not None:
                # 記錄樣本（此版每點只需 1 筆）
                samples_feats.append(list(last_valid))
                done_points.append(cur_target)
                print(f"[Calib] 已完成 {len(done_points)}/{total_goal} 點。")
                # 準備下一點
                cur_target = None if len(done_points) >= total_goal else sample_one_point(margin=0.06)
        elif key == 13:  # Enter -> 擬合 + 存檔
            if cal_mode:
                if len(done_points) < total_goal or len(samples_feats) < total_goal:
                    print(f"[Calib] 尚未達到設定數量：{len(done_points)}/{total_goal} 點。")
                else:
                    # 擬合
                    if SCREEN_SIZE: sw, sh = SCREEN_SIZE
                    else:           sw, sh = W, H
                    feats_all = []; target_x = []; target_y = []; weights = []
                    for (nx, ny), feat in zip(done_points, samples_feats):
                        tx = nx * sw; ty = ny * sh
                        rx, ry, yaw_p, pitch_p, roll_p, fc_x, fc_y, fsize = feat
                        feats_all.append(feature_vec(rx, ry, yaw_p, pitch_p, roll_p, fc_x, fc_y, fsize))
                        target_x.append([tx]); target_y.append([ty])
                        # 邊緣/角落點加權更大
                        w = 1.0
                        if nx < 0.15 or nx > 0.85: w *= 1.4
                        if ny < 0.15 or ny > 0.85: w *= 1.4
                        weights.append(w)
                    feats_all = np.stack(feats_all, axis=0)
                    target_x = np.array(target_x, dtype=np.float32)
                    target_y = np.array(target_y, dtype=np.float32)
                    weights  = np.array(weights,  dtype=np.float32)
                    wx = fit_ridge_weighted(target_x, feats_all, weights, RIDGE_LAMBDA)
                    wy = fit_ridge_weighted(target_y, feats_all, weights, RIDGE_LAMBDA)
                    calibrated = True
                    cal_mode = False
                    x_out = y_out = None; kf.reset()
                    print("[Calib] 完成！即時預測點啟用。")
                    # 存檔
                    meta = {
                        "screen_w": int(sw), "screen_h": int(sh),
                        "VERT_GAIN": float(VERT_GAIN),
                        "points_done": done_points,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_calib(CALIB_PATH, wx, wy, meta)
        elif key in (ord('o'), ord('O')):
            print("[Baseline]（預留）")
        elif key == ord(']'):
            ema_base = min(0.8, EMA_ALPHA_BASE + 0.05)
            print(f"[Filter] EMA base = {ema_base:.2f}")
        elif key == ord('['):
            ema_base = max(0.05, EMA_ALPHA_BASE - 0.05)
            print(f"[Filter] EMA base = {ema_base:.2f}")
        elif key in (ord('k'), ord('K')):
            use_kalman = not use_kalman
            print(f"[Filter] Kalman = {use_kalman}")
        elif key == ord(','):
            deadband = max(0, DEADBAND_PX - 1); print(f"[Filter] Deadband = {deadband}px")
        elif key == ord('.'):
            deadband = min(40, DEADBAND_PX + 1); print(f"[Filter] Deadband = {deadband}px")
        elif key in (ord('t'), ord('T')):
            streaming = not streaming
            print(f"[Stream] UDP -> {STREAM_HOST}:{STREAM_PORT}  {'ON' if streaming else 'OFF'}")
        elif key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            try:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            except Exception:
                if fullscreen and SCREEN_SIZE:
                    sw, sh = SCREEN_SIZE
                    cv2.resizeWindow(win_name, sw, sh)
                    cv2.moveWindow(win_name, 0, 0)

        cv2.imshow(win_name, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
