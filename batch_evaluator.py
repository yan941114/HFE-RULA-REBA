#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_evaluator.py
------------------
專題批次數據評估與統計導出工具 (Batch Ergonomic Assessment & Ground Truth Evaluator)

特色：
1. 批次處理：指定資料夾，一次性自動分析所有照片 (JPG/PNG) 與影片 (MP4/AVI)。
2. 完整報表匯出：自動產出包含各關節角度、Table A/B/C 分數、最終 Grand Score、
   行動水準 (Action Level) 與影片峰值危害 (Peak Risk Frame) 之扁平化 CSV。
3. 人因學術驗證 (Ground Truth Comparison)：
   若提供專家標註檔 (ground_truth.csv)，自動計算：
   - 關節角度平均絕對誤差 (MAE) 與 RMSE
   - 自動評分完全一致率 (Exact Agreement %)
   - 行動水準一致率 (Action Level Agreement %)
   - 線性加權 Cohen's Kappa 係數 (Inter-rater Reliability)
   - 風險等級混淆矩陣 (Confusion Matrix)
4. 成果圖檔歸檔：自動儲存骨架標註圖與影片最高危害瞬間截圖至輸出目錄。
"""

import argparse
import csv
import glob
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# 加入路徑
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pose_analyzer import PoseAnalyzer
from src.rula_reba_engine import (
    calculate_rula,
    calculate_reba,
    flatten_assessment_result,
)


# ==============================================================================
# 學術統計評估工具 (Academic Statistics Helpers)
# ==============================================================================

def compute_cohens_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_val: int = 1, max_val: int = 7) -> float:
    """
    計算線性加權 Cohen's Kappa 係數 (Linear Weighted Kappa)
    常用於人因工程中評估 AI 自動評分與專家評分的一致性
    """
    categories = list(range(min_val, max_val + 1))
    n_cat = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # 建立混淆矩陣
    conf_mat = np.zeros((n_cat, n_cat), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in cat_to_idx and p in cat_to_idx:
            conf_mat[cat_to_idx[t], cat_to_idx[p]] += 1

    n_samples = np.sum(conf_mat)
    if n_samples == 0:
        return 0.0

    conf_mat /= n_samples

    # 線性權重矩陣 w_ij = |i - j| / (n_cat - 1)
    weights = np.zeros((n_cat, n_cat), dtype=float)
    for i in range(n_cat):
        for j in range(n_cat):
            weights[i, j] = abs(i - j) / float(max(1, n_cat - 1))

    # 觀測不一致度
    obs_disagree = np.sum(weights * conf_mat)

    # 期望不一致度 (邊際機率外積)
    hist_true = np.sum(conf_mat, axis=1)
    hist_pred = np.sum(conf_mat, axis=0)
    expected_mat = np.outer(hist_true, hist_pred)
    exp_disagree = np.sum(weights * expected_mat)

    if exp_disagree == 0:
        return 1.0

    kappa = 1.0 - (obs_disagree / exp_disagree)
    return float(round(kappa, 3))


def print_academic_report(results: List[Dict[str, Any]], gt_csv_path: str, method: str):
    """輸出符合學術論文規範的對照評估報告 (純 Python + CSV + NumPy)"""
    print("\n" + "=" * 65)
    print(f"      【人因專題實驗驗證報告】AI 自動評估 vs. 人工專家評分 ({method})")
    print("=" * 65)

    # 讀取 Ground Truth CSV
    gt_map = {}
    with open(gt_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "filename" in row:
                gt_map[row["filename"]] = row

    # 配對樣本
    matched_pairs = []
    for r in results:
        fname = r.get("filename")
        if fname in gt_map:
            matched_pairs.append((r, gt_map[fname]))

    total_matched = len(matched_pairs)
    print(f"有效比對樣本數 (Matched Samples): {total_matched}")

    if total_matched == 0:
        print("[警告] 未能在 Ground Truth 檔案中找到對應檔名的比對資料！")
        return

    # 1. 最終分數與行動等級一致性
    y_true_score = []
    y_pred_score = []
    y_true_al = []
    y_pred_al = []

    for pred, gt in matched_pairs:
        # 分數
        if "final_score" in gt and "final_score" in pred:
            try:
                y_true_score.append(int(float(gt["final_score"])))
                y_pred_score.append(int(float(pred["final_score"])))
            except (ValueError, TypeError):
                pass
        # 行動水準
        if "action_level" in gt and "action_level" in pred:
            try:
                y_true_al.append(int(float(gt["action_level"])))
                y_pred_al.append(int(float(pred["action_level"])))
            except (ValueError, TypeError):
                pass

    y_true_score = np.array(y_true_score)
    y_pred_score = np.array(y_pred_score)
    y_true_al = np.array(y_true_al)
    y_pred_al = np.array(y_pred_al)

    if len(y_true_score) > 0:
        exact_score_match = float(np.mean(y_true_score == y_pred_score) * 100.0)
        tolerance_1_match = float(np.mean(np.abs(y_true_score - y_pred_score) <= 1) * 100.0)
        min_s = 1
        max_s = 7 if method == "RULA" else 15
        kappa = compute_cohens_weighted_kappa(y_true_score, y_pred_score, min_val=min_s, max_val=max_s)
    else:
        exact_score_match = tolerance_1_match = kappa = 0.0

    exact_al_match = float(np.mean(y_true_al == y_pred_al) * 100.0) if len(y_true_al) > 0 else 0.0

    print(f"\n[1] 評分一致性指標 (Reliability Metrics):")
    print(f"  - 評分完全一致率 (Exact Score Agreement):       {exact_score_match:.1f}%")
    print(f"  - 容許 ±1 分一致率 (Tolerance ±1 Agreement):     {tolerance_1_match:.1f}%")
    print(f"  - 行動水準一致率 (Action Level Agreement):       {exact_al_match:.1f}%")
    print(f"  - 加權 Cohen's Kappa 係數 (Weighted Kappa):     {kappa:.3f}")
    if kappa >= 0.8:
        kappa_eval = "極佳一致性 (Almost Perfect Agreement)"
    elif kappa >= 0.6:
        kappa_eval = "高度一致性 (Substantial Agreement)"
    elif kappa >= 0.4:
        kappa_eval = "中度一致性 (Moderate Agreement)"
    else:
        kappa_eval = "低度一致性 (Fair/Poor Agreement)"
    print(f"    -> 統計學判定: {kappa_eval}")

    # 2. 關節角度誤差
    angle_cols = [("trunk_angle", "軀幹角度"), ("neck_angle", "頸部角度"),
                  ("upper_arm_angle", "上臂角度"), ("lower_arm_angle", "前臂手肘角度")]

    print(f"\n[2] 關鍵部位角度估計誤差 (Joint Angle Estimation Error):")
    for col, name in angle_cols:
        diffs = []
        for pred, gt in matched_pairs:
            if col in pred and col in gt:
                try:
                    diffs.append(abs(float(pred[col]) - float(gt[col])))
                except (ValueError, TypeError):
                    pass
        if diffs:
            mae = float(np.mean(diffs))
            rmse = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
            print(f"  - {name:<12}: MAE = {mae:.1f}° | RMSE = {rmse:.1f}°")

    print("=" * 65 + "\n")


# ==============================================================================
# 批次評估執行器 (BatchEvaluator)
# ==============================================================================

class BatchEvaluator:
    def __init__(self,
                 model_path: str = "HFcode/yolo11n-pose.pt",
                 method: str = "RULA",
                 side: str = "auto",
                 output_dir: str = "outputs",
                 manual_modifiers: Optional[Dict[str, Any]] = None):
        self.method = method.upper()
        self.side = side
        self.output_dir = output_dir
        self.manual_modifiers = manual_modifiers or {}
        os.makedirs(output_dir, exist_ok=True)

        self.analyzer = PoseAnalyzer(model_path=model_path)

    def process_image(self, img_path: str) -> Optional[Dict[str, Any]]:
        """處理單張影像並輸出結果與標註圖"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        res = self.analyzer.analyze_image(
            img,
            side=self.side,
            method=self.method,
            manual_modifiers=self.manual_modifiers
        )
        if res is None:
            return None

        fname = os.path.basename(img_path)
        annotated = res["annotated_image"]

        # 儲存標註圖
        save_img_path = os.path.join(self.output_dir, f"annotated_{fname}")
        cv2.imwrite(save_img_path, annotated)

        flat = res["flattened"]
        record = {
            "filename": fname,
            "media_type": "image",
            "annotated_path": save_img_path,
            "facing": res["diagnostics"]["facing"],
            "side_used": res["diagnostics"]["side_used"],
            **flat
        }
        return record

    def process_video(self, vid_path: str, sample_interval: int = 2) -> Optional[Dict[str, Any]]:
        """處理單支影片並輸出時間序列與最高風險幀截圖"""
        fname = os.path.basename(vid_path)
        base_name = os.path.splitext(fname)[0]

        annotated_vid_path = os.path.join(self.output_dir, f"annotated_{base_name}.mp4")

        try:
            res = self.analyzer.analyze_video(
                vid_path,
                method=self.method,
                side=self.side,
                manual_modifiers=self.manual_modifiers,
                sample_interval=sample_interval,
                output_annotated_path=annotated_vid_path
            )
        except Exception as e:
            print(f"[錯誤] 處理影片失敗 {fname}: {e}")
            return None

        # 儲存最高風險影格截圖
        peak_img_path = ""
        if res["peak_risk_image"] is not None:
            peak_img_path = os.path.join(self.output_dir, f"peak_{base_name}.jpg")
            cv2.imwrite(peak_img_path, res["peak_risk_image"])

        peak_rec = res["peak_risk_record"] or {}

        record = {
            "filename": fname,
            "media_type": "video",
            "duration_sec": res["duration_sec"],
            "total_frames_processed": res["total_frames_processed"],
            "peak_risk_score": res["peak_risk_score"],
            "peak_risk_timestamp": peak_rec.get("timestamp_sec", 0.0),
            "high_risk_exposure_pct": res["high_risk_exposure_pct"],
            "annotated_path": annotated_vid_path,
            "peak_frame_snapshot": peak_img_path,
            **peak_rec
        }
        return record

    def run_directory(self, input_dir: str, sample_interval: int = 2) -> List[Dict[str, Any]]:
        """掃描資料夾內所有影音檔案進行批次分析"""
        supported_img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        supported_vid_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")

        img_files: List[str] = []
        for ext in supported_img_exts:
            img_files.extend(glob.glob(os.path.join(input_dir, ext)))
            img_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        vid_files: List[str] = []
        for ext in supported_vid_exts:
            vid_files.extend(glob.glob(os.path.join(input_dir, ext)))
            vid_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        img_files = sorted(list(set(img_files)))
        vid_files = sorted(list(set(vid_files)))

        print(f"[BatchEvaluator] 掃描目錄: {input_dir}")
        print(f"  找到 {len(img_files)} 張照片, {len(vid_files)} 部影片")

        results = []

        # 處理照片
        for idx, f in enumerate(img_files, 1):
            fname = os.path.basename(f)
            print(f"  [{idx}/{len(img_files)}] 正在分析照片: {fname}...")
            r = self.process_image(f)
            if r:
                results.append(r)

        # 處理影片
        for idx, f in enumerate(vid_files, 1):
            fname = os.path.basename(f)
            print(f"  [{idx}/{len(vid_files)}] 正在分析影片: {fname}...")
            r = self.process_video(f, sample_interval=sample_interval)
            if r:
                results.append(r)

        # 寫入 CSV 報表
        csv_path = os.path.join(self.output_dir, f"batch_evaluation_{self.method}.csv")
        default_fields = ["filename", "media_type", "assessment", "final_score", "action_level", "risk_level"]
        fieldnames = list(default_fields)
        if results:
            for r in results:
                for k in r.keys():
                    if k not in fieldnames and k != "raw_angles":
                        fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow({k: v for k, v in r.items() if k != "raw_angles"})

        print(f"\n[完成] 批次評估完成！共處理 {len(results)} 筆檔案。")
        print(f"  報告已儲存至: {csv_path}")

        return results


# ==============================================================================
# CLI 命令列入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ErgoVision AI 批次人體工學評估與專題驗證工具")
    parser.add_argument("--input-dir", type=str, required=True, help="待評估的照片或影片資料夾路徑")
    parser.add_argument("--method", type=str, default="RULA", choices=["RULA", "REBA"], help="評估量表 (RULA 或 REBA)")
    parser.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"], help="評估肢體側別")
    parser.add_argument("--output-dir", type=str, default="outputs", help="結果 CSV 與標註圖檔儲存目錄")
    parser.add_argument("--sample-interval", type=int, default=2, help="影片影格抽樣間隔 (預設每隔 2 幀)")
    parser.add_argument("--ground-truth", type=str, default=None, help="專家人工標註 CSV 檔案路徑 (用於驗證 Kappa 與 MAE)")

    # 專家環境參數 (Manual Modifiers)
    parser.add_argument("--load-kg", type=float, default=0.0, help="負重重量 (kg)")
    parser.add_argument("--muscle-use", type=int, default=0, choices=[0, 1], help="RULA 肌肉重複或靜態使用 (0 或 1)")
    parser.add_argument("--coupling", type=int, default=0, choices=[0, 1, 2, 3], help="REBA 抓握品質評分 (0:良, 1:可, 2:差, 3:危險)")

    args = parser.parse_args()

    modifiers = {
        "load_weight_kg": args.load_kg,
        "muscle_use_a": args.muscle_use,
        "muscle_use_b": args.muscle_use,
        "coupling_score": args.coupling,
    }

    evaluator = BatchEvaluator(
        method=args.method,
        side=args.side,
        output_dir=args.output_dir,
        manual_modifiers=modifiers
    )

    results = evaluator.run_directory(args.input_dir, sample_interval=args.sample_interval)

    # 若有提供 Ground Truth，輸出學術驗證指標
    if args.ground_truth and os.path.exists(args.ground_truth):
        print_academic_report(results, args.ground_truth, method=args.method)


if __name__ == "__main__":
    main()
