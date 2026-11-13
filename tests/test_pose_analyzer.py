#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pose_analyzer.py
---------------------
測試 YOLO11 姿態幾何運算、關節角度估計與 RULA/REBA 串接流程 (使用虛擬幾何資料，不依賴任何外部相片)
"""

import sys
import os
import cv2
import numpy as np

# 加入 src 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose_analyzer import (
    PoseAnalyzer,
    KP_NOSE, KP_L_EAR, KP_R_EAR,
    KP_L_SHOULDER, KP_R_SHOULDER,
    KP_L_ELBOW, KP_R_ELBOW,
    KP_L_WRIST, KP_R_WRIST,
    KP_L_HIP, KP_R_HIP,
    KP_L_KNEE, KP_R_KNEE,
    KP_L_ANKLE, KP_R_ANKLE
)


def test_synthetic_keypoints():
    print("[TEST 1] 驗證純幾何解剖關節角度計算 (Synthetic Keypoints)...")
    analyzer = PoseAnalyzer()

    # 情境 1: 面向右側、直立站姿、手肘彎曲 90 度
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)

    xy[KP_NOSE] = [320, 100]
    xy[KP_L_EAR] = [300, 100]; xy[KP_R_EAR] = [300, 100]
    xy[KP_L_SHOULDER] = [300, 150]; xy[KP_R_SHOULDER] = [300, 150]
    xy[KP_L_ELBOW] = [300, 230]; xy[KP_R_ELBOW] = [300, 230]
    xy[KP_L_WRIST] = [380, 230]; xy[KP_R_WRIST] = [380, 230]  # 手腕往前水平延伸 (前臂成 90度直角)
    xy[KP_L_HIP] = [300, 310]; xy[KP_R_HIP] = [300, 310]
    xy[KP_L_KNEE] = [300, 410]; xy[KP_R_KNEE] = [300, 410]
    xy[KP_L_ANKLE] = [300, 510]; xy[KP_R_ANKLE] = [300, 510]

    angles, side, diag = analyzer.calculate_body_angles(xy, conf, side="right")
    print(f"  直立 90度手肘角度輸出: {angles}")

    # 驗證前臂屈曲為 90°
    assert abs(angles["lower_arm"] - 90.0) < 2.0, f"Expected lower arm ~90 deg, got {angles['lower_arm']}"
    # 驗證軀幹接近 0°
    assert abs(angles["trunk"]) < 2.0, f"Expected trunk ~0 deg, got {angles['trunk']}"
    # 驗證上臂垂下接近 0°
    assert abs(angles["upper_arm"]) < 2.0, f"Expected upper arm ~0 deg, got {angles['upper_arm']}"

    # 情境 2: 大角度彎腰搬運 (軀幹前屈 60度, 抬頭看前方)
    xy_bend = xy.copy()
    # 軀幹前傾: 肩部往前移到 (400, 250)
    xy_bend[KP_L_SHOULDER] = [400, 250]; xy_bend[KP_R_SHOULDER] = [400, 250]
    xy_bend[KP_NOSE] = [450, 210]  # 抬頭看前上方 (後仰 Extension)
    angles_bend, _, _ = analyzer.calculate_body_angles(xy_bend, conf, side="right")
    print(f"  前傾 60度彎腰角度輸出: {angles_bend}")

    # 驗證軀幹為正角前屈
    assert angles_bend["trunk"] > 30.0, f"Expected trunk flexion > 30 deg, got {angles_bend['trunk']}"
    print("  -> 幾何關節運算與正負號測試順利通過！")


def test_video_recalculation():
    print("[TEST 2] 驗證階段四純重算短路快取接口 (recalculate_video_scores)...")
    # 模擬 5 幀的時序資料
    mock_records = [
        {
            "frame_index": i,
            "timestamp_sec": i * 0.1,
            "raw_angles": {"trunk": 10.0 + i * 5, "neck": 10.0, "upper_arm": 20.0, "lower_arm": 90.0, "wrist": 0.0, "knee": 0.0}
        }
        for i in range(5)
    ]

    # 初次計算 (無負重)
    res_0kg = PoseAnalyzer.recalculate_video_scores(mock_records, method="REBA", manual_modifiers={"load_weight_kg": 0.0})
    # 第二次計算 (調高負重至 15kg，毫秒級完成)
    res_15kg = PoseAnalyzer.recalculate_video_scores(mock_records, method="REBA", manual_modifiers={"load_weight_kg": 15.0})

    assert res_15kg["peak_risk_score"] > res_0kg["peak_risk_score"], "15kg load should have higher peak score than 0kg"
    print(f"  0kg Peak: {res_0kg['peak_risk_score']}, 15kg Peak: {res_15kg['peak_risk_score']}")
    print("  -> 短路純重算接口測試通過！")


if __name__ == "__main__":
    print("=" * 50)
    print("開始執行 PoseAnalyzer 改良版單元測試...")
    print("=" * 50)
    test_synthetic_keypoints()
    test_video_recalculation()
    print("=" * 50)
    print("ALL TESTS PASSED! 所有測試順利通過！")
    print("=" * 50)
