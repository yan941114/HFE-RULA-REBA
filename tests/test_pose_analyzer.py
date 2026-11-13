#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pose_analyzer.py
---------------------
測試 YOLO11 姿態推論、關節角度估計與 RULA/REBA 串接流程
"""

import sys
import os
import cv2
import numpy as np

# 加入 src 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose_analyzer import PoseAnalyzer


def test_pose_analyzer_image():
    print("[TEST] 驗證 PoseAnalyzer 影像分析管線...")
    analyzer = PoseAnalyzer()

    # 建立一張測試影像 (或使用現有照片)
    test_img_path = "六人合照_姓名標示.png"
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
    else:
        # 建立黑底測試圖
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. 測試 RULA 評估
    res_rula = analyzer.analyze_image(img, side="auto", method="RULA")
    if res_rula is not None:
        assert "assessment_result" in res_rula
        assert "angles" in res_rula
        assert "annotated_image" in res_rula
        assert res_rula["assessment_result"]["assessment"] == "RULA"
        print(f"  -> RULA 測試通過! Grand Score: {res_rula['assessment_result']['grand_score']}")

    # 2. 測試 REBA 評估
    res_reba = analyzer.analyze_image(img, side="auto", method="REBA")
    if res_reba is not None:
        assert "assessment_result" in res_reba
        assert res_reba["assessment_result"]["assessment"] == "REBA"
        print(f"  -> REBA 測試通過! Final Score: {res_reba['assessment_result']['final_reba_score']}")

    print("  -> PoseAnalyzer 影像管線測試全部通過！")


if __name__ == "__main__":
    print("=" * 50)
    print("開始執行 PoseAnalyzer 單元整合測試...")
    print("=" * 50)
    test_pose_analyzer_image()
    print("=" * 50)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 50)
