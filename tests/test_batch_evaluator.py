#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch_evaluator.py
-----------------------
測試批次評估模組 (batch_evaluator.py) 與人因統計一致性指標 (Kappa / MAE)
"""

import sys
import os
import shutil
import csv
import numpy as np
import cv2

# 加入根目錄路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from batch_evaluator import BatchEvaluator, compute_cohens_weighted_kappa, print_academic_report


def test_cohens_weighted_kappa():
    print("[TEST 1] 驗證加權 Cohen's Kappa 統計演算法...")

    # 完全一致 => Kappa = 1.0
    y_true = np.array([1, 2, 3, 4, 5, 6, 7])
    y_pred = np.array([1, 2, 3, 4, 5, 6, 7])
    k_perfect = compute_cohens_weighted_kappa(y_true, y_pred, min_val=1, max_val=7)
    assert abs(k_perfect - 1.0) < 1e-3, f"Expected 1.0, got {k_perfect}"

    # 微幅差距 (±1 分之內) => 高度一致性 (Kappa > 0.6)
    y_pred_close = np.array([1, 2, 4, 4, 6, 6, 7])
    k_close = compute_cohens_weighted_kappa(y_true, y_pred_close, min_val=1, max_val=7)
    assert k_close > 0.6, f"Expected > 0.6, got {k_close}"

    print(f"  Kappa 測試通過！完美值={k_perfect}, 近似值={k_close}")


def test_batch_pipeline():
    print("[TEST 2] 驗證批次評估管線與 CSV / Ground Truth 整合...")
    test_dir = "temp_batch_test"
    out_dir = "temp_batch_output"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 1. 建立測試照片 (空白測試圖)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_path = os.path.join(test_dir, "sample_task.jpg")
        cv2.imwrite(img_path, img)

        # 2. 執行批次評估
        evaluator = BatchEvaluator(
            method="RULA",
            output_dir=out_dir,
            manual_modifiers={"load_weight_kg": 2.0}
        )
        results = evaluator.run_directory(test_dir)

        # 驗證 CSV 輸出路徑
        csv_path = os.path.join(out_dir, "batch_evaluation_RULA.csv")
        assert os.path.exists(csv_path), "CSV report should be created"

        # 3. 測試 Ground Truth 比對報表
        gt_path = os.path.join(test_dir, "ground_truth.csv")
        with open(gt_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "final_score", "action_level", "trunk_angle", "neck_angle", "upper_arm_angle", "lower_arm_angle"])
            writer.writerow(["sample_task.jpg", "3", "2", "15.0", "10.0", "20.0", "90.0"])

        mock_results = [{
            "filename": "sample_task.jpg",
            "media_type": "image",
            "assessment": "RULA",
            "final_score": 3,
            "action_level": 2,
            "risk_level": "低度風險",
            "trunk_angle": 12.0,
            "neck_angle": 8.0,
            "upper_arm_angle": 25.0,
            "lower_arm_angle": 85.0
        }]

        # 執行統計評估
        print_academic_report(mock_results, gt_path, method="RULA")
        print("  批次評估與學術報告產生正常！")

    finally:
        # 清理測試目錄
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)


if __name__ == "__main__":
    print("=" * 50)
    print("開始執行 BatchEvaluator 單元測試...")
    print("=" * 50)
    test_cohens_weighted_kappa()
    test_batch_pipeline()
    print("=" * 50)
    print("ALL BATCH EVALUATOR TESTS PASSED!")
    print("=" * 50)
