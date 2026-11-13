#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
------
ErgoVision AI - RULA / REBA 人體工學自動風險評估系統 (Web Dashboard)

特色：
1. 支援照片即時評估、工作影片時間序列分析、專題批次資料夾驗證。
2. 骨架幾何疊圖：以綠/黃/橘/紅動態變色標註人體危害部位。
3. 影片峰值捕捉：自動截圖並標記全片最高危害瞬間 (Peak Risk Frame)。
4. 狀態快取短路機制 (State Cache Bypass)：
   調參 (如更改負重或支撐) 毫秒級重算，完全不重複執行 YOLO 影片推論！
5. 人因專題學術驗證：內建加權 Cohen's Kappa 與 MAE 對照分析。
"""

import os
import sys
import tempfile
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import streamlit as st

# 加入專案路徑
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pose_analyzer import PoseAnalyzer
from src.rula_reba_engine import (
    calculate_rula,
    calculate_reba,
    flatten_assessment_result,
)
from batch_evaluator import compute_cohens_weighted_kappa


# ==============================================================================
# Streamlit 頁面全域設定
# ==============================================================================
st.set_page_config(
    page_title="ErgoVision AI - RULA / REBA 人因評估系統",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .score-card-al1 {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 16px;
        border-radius: 8px;
    }
    .score-card-al2 {
        background-color: #FEFCE8;
        border-left: 6px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
    }
    .score-card-al3 {
        background-color: #FFF7ED;
        border-left: 6px solid #F97316;
        padding: 16px;
        border-radius: 8px;
    }
    .score-card-al4 {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
    }
    .metric-val {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 模型快取載入 (Singleton Cache)
# ==============================================================================
@st.cache_resource
def load_pose_analyzer():
    model_path = os.path.join(os.path.dirname(__file__), "HFcode", "yolo11n-pose.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11n-pose.pt"
    return PoseAnalyzer(model_path=model_path, conf_thres=0.4, ema_alpha=0.35)


analyzer = load_pose_analyzer()


# ==============================================================================
# 側邊欄控制項 (Sidebar Parameters)
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/worker-male.png", width=64)
st.sidebar.title("人因評估參數設定")

method = st.sidebar.radio(
    "評估量表 (Assessment Method)",
    options=["RULA", "REBA"],
    help="RULA 偏向上肢與坐姿負荷；REBA 偏向全身、搬運與站姿負荷。"
)

side_choice = st.sidebar.selectbox(
    "肢體側別 (Evaluation Side)",
    options=["auto", "left", "right"],
    format_func=lambda x: "自動選擇最佳可見側 (Auto)" if x == "auto" else ("左側 (Left)" if x == "left" else "右側 (Right)")
)

st.sidebar.markdown("---")
st.sidebar.subheader("作業環境與負重參數")

load_kg = st.sidebar.slider(
    "負重重量 (Load Weight, kg)",
    min_value=0.0,
    max_value=30.0,
    value=0.0,
    step=0.5,
    help="作業員手持或搬運之物品重量"
)

load_shock = st.sidebar.checkbox("瞬間衝擊或急劇施力 (Shock Force)")

st.sidebar.markdown("---")
st.sidebar.subheader("肌肉活動性 (Muscle / Activity)")

col_m1, col_m2 = st.sidebar.columns(2)
with col_m1:
    act_static = st.checkbox("靜態姿勢維持 > 1分")
with col_m2:
    act_repeat = st.checkbox("動作高頻重複 > 4次/分")

act_rapid = False
coupling_score = 0
if method == "REBA":
    act_rapid = st.sidebar.checkbox("姿勢劇烈變動或基底不穩")
    coupling_score = st.sidebar.selectbox(
        "握把耦合程度 (Coupling Quality)",
        options=[0, 1, 2, 3],
        format_func=lambda x: ["0: 良好 (舒適握把與中等握力)", "1: 尚可 (可握但不理想)", "2: 差 (勉強能抓但易滑脫)", "3: 危險 (無握把或不安全)"][x]
    )

st.sidebar.markdown("---")
st.sidebar.subheader("姿勢與環境人工補正")
with st.sidebar.expander("進階姿態補正項 (進一步微調)", expanded=False):
    shoulder_raised = st.checkbox("肩膀明顯聳起 (+1)")
    arm_abducted = st.checkbox("手臂外展或側向張開 (+1)")
    arm_supported = st.checkbox("手臂有支撐或有扶手 (-1)")
    wrist_deviated = st.checkbox("手腕側偏 (橈偏/尺偏) (+1)")
    wrist_twist = st.selectbox("手腕扭轉程度", options=[1, 2], format_func=lambda x: "中立範圍 (1分)" if x == 1 else "極限旋轉 (2分)")
    trunk_twisted = st.checkbox("軀幹扭轉 (+1)")
    trunk_side_bending = st.checkbox("軀幹側彎 (+1)")
    neck_twisted = st.checkbox("頸部扭轉 (+1)")
    neck_side_bending = st.checkbox("頸部側彎 (+1)")
    legs_supported = st.checkbox("雙腳支撐良好且平衡", value=True)

# RULA Force/Load Score (0~3) 標準轉換邏輯
if load_kg < 2.0:
    rula_force = 0
elif load_kg <= 10.0:
    rula_force = 2 if (act_static or act_repeat or load_shock) else 1
else:
    rula_force = 3 if (act_static or act_repeat or load_shock) else 2

# 整合 modifiers 字典
manual_modifiers = {
    "load_weight_kg": load_kg,
    "load_rapid_shock": load_shock,
    "muscle_use_a": 1 if (act_static or act_repeat) else 0,
    "muscle_use_b": 1 if (act_static or act_repeat) else 0,
    "force_load_a": rula_force,
    "force_load_b": rula_force,
    "activity_static": act_static,
    "activity_repeated": act_repeat,
    "activity_rapid": act_rapid,
    "coupling_score": coupling_score,
    "shoulder_raised": shoulder_raised,
    "arm_abducted": arm_abducted,
    "arm_supported": arm_supported,
    "wrist_deviated": wrist_deviated,
    "wrist_twist": wrist_twist,
    "wrist_twisted": (wrist_twist == 2),  # 支援 REBA 布林格式連動
    "trunk_twisted": trunk_twisted,
    "trunk_side_bending": trunk_side_bending,
    "neck_twisted": neck_twisted,
    "neck_side_bending": neck_side_bending,
    "legs_supported": legs_supported,
}


# ==============================================================================
# 主頁面標題與分頁
# ==============================================================================
st.markdown('<div class="main-header">🏋️‍♂️ ErgoVision AI 人體工學自動風險評估系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">結合 YOLO11-Pose 骨架偵測與國際標準 RULA / REBA 查表量表 ｜ 人因工程與實驗設計專題</div>', unsafe_allow_html=True)

tab_photo, tab_video, tab_batch = st.tabs([
    "📷 單張照片評估 (Photo Mode)",
    "🎥 影片時序分析與峰值捕捉 (Video Mode)",
    "📊 專題批次數據與學術驗證 (Batch & Ground Truth)"
])


# ==============================================================================
# Tab 1: 單張照片評估 (Photo Mode)
# ==============================================================================
with tab_photo:
    st.subheader("上傳工作照片進行即時姿勢評估")
    uploaded_image = st.file_uploader(
        "選擇影像檔 (JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png", "webp"],
        key="photo_uploader"
    )

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 執行姿態推論與評分
        with st.spinner("AI 正在提取人體骨架並計算關節角度..."):
            res = analyzer.analyze_image(
                img_bgr,
                side=side_choice,
                method=method,
                manual_modifiers=manual_modifiers
            )

        if res is not None:
            col_img, col_score = st.columns([1.1, 1.0])

            with col_img:
                st.markdown("#### 骨架與關節疊圖標註")
                img_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_column_width=True)
                diag = res["diagnostics"]
                st.caption(f"面向判定: **{diag['facing'].upper()}** ｜ 使用肢體側: **{diag['side_used'].upper()}** ｜ 左側信心: {diag['left_confidence']} ｜ 右側信心: {diag['right_confidence']}")

            with col_score:
                st.markdown("#### 評估結果看板")
                ass = res["assessment_result"]
                al = ass["action_level"]
                score_val = ass.get("final_reba_score") if method == "REBA" else ass.get("grand_score")

                # 樣式卡片
                card_class = f"score-card-al{min(4, max(1, al))}"
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #374151;">{method} 總分 (Grand Score)</div>
                    <div class="metric-val">{score_val} <span style="font-size: 1.2rem; font-weight: 500;">/ 行動水準 AL {al}</span></div>
                    <div style="font-size: 1.05rem; font-weight: 600; margin-top: 6px;">風險狀態：{ass['risk_level']}</div>
                    <div style="font-size: 0.95rem; color: #4B5563; margin-top: 4px;">{ass['action_description']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### 關節角度摘要")
                angles = res["angles"]
                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("軀幹角度", f"{angles['trunk']:+.1f}°")
                col_a2.metric("頸部角度", f"{angles['neck']:+.1f}°")
                col_a3.metric("膝關節角度", f"{angles['knee']:.1f}°")

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("上臂屈伸", f"{angles['upper_arm']:+.1f}°")
                col_b2.metric("前臂屈曲", f"{angles['lower_arm']:.1f}°")
                col_b3.metric("手腕角度", f"{angles['wrist']:.1f}°")

                # 詳細子分數表格
                with st.expander("查看查表拆解明細 (Table A / B / C 矩陣)", expanded=False):
                    st.json(res["flattened"])
        else:
            st.warning("未能在此影像中偵測到清晰的人體關鍵點，請嘗試其他角度或光線充足之照片。")
    else:
        st.info("請於上方上傳一張作業姿勢照片開始評估。")


# ==============================================================================
# Tab 2: 影片時序分析與峰值捕捉 (Video Mode)
# ==============================================================================
with tab_video:
    st.subheader("上傳作業影片進行時間序列姿勢與峰值風險分析")
    uploaded_video = st.file_uploader(
        "選擇影片檔 (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    sample_step = st.slider("影格抽樣間隔 (Frame Interval)", min_value=1, max_value=5, value=2, help="數值越大處理越快，1 為逐影格精確分析")

    if uploaded_video is not None:
        # 將上傳檔案存為臨時檔案
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name
        tfile.close()  # 立即關閉檔案控制代碼，防止 Windows 檔案鎖定衝突

        col_vbtn, col_vstatus = st.columns([1, 3])
        with col_vbtn:
            start_btn = st.button("🚀 開始全片姿勢分析", type="primary")

        if start_btn or ("cached_video_records" in st.session_state and st.session_state.get("cached_video_path") == uploaded_video.name):
            if start_btn:
                prog_bar = st.progress(0, text="正在逐幀進行 YOLO11 人體骨架追蹤...")
                try:
                    res_video = analyzer.analyze_video(
                        tfile_path,
                        method=method,
                        side=side_choice,
                        manual_modifiers=manual_modifiers,
                        sample_interval=sample_step
                    )
                    st.session_state["cached_video_records"] = res_video["time_series_records"]
                    st.session_state["cached_video_peak_img"] = res_video["peak_risk_image"]
                    st.session_state["cached_video_peak_rec"] = res_video["peak_risk_record"]
                    st.session_state["cached_video_path"] = uploaded_video.name
                    prog_bar.progress(100, text="分析完成！")
                except Exception as e:
                    st.error(f"影片分析發生錯誤: {e}")
                    res_video = None
            else:
                # 觸發極速短路純重算 (State Cache Bypass，不重跑 YOLO！)
                records = st.session_state["cached_video_records"]
                res_recalc = PoseAnalyzer.recalculate_video_scores(records, method=method, manual_modifiers=manual_modifiers)
                st.session_state["cached_video_records"] = res_recalc["time_series_records"]
                st.session_state["cached_video_peak_rec"] = res_recalc["peak_risk_record"]

            # 顯示結果
            records = st.session_state.get("cached_video_records", [])
            peak_img = st.session_state.get("cached_video_peak_img")
            peak_rec = st.session_state.get("cached_video_peak_rec", {})

            if records:
                st.success(f"影片分析完成！共提取 {len(records)} 幀有效姿態資料。")

                # 指標看板
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                peak_score = peak_rec.get("score", 0)
                col_m1.metric("全片最高危害分數 (Peak Score)", f"{peak_score} 分")
                col_m2.metric("最高風險瞬間時間戳", f"{peak_rec.get('timestamp_sec', 0.0)} 秒")
                scores = [r["score"] for r in records]
                col_m3.metric("作業全程平均分數", f"{np.mean(scores):.2f} 分")
                high_risk_cnt = sum(1 for r in records if r["action_level"] >= 3)
                high_pct = round((high_risk_cnt / len(records)) * 100.0, 1)
                col_m4.metric("高危害時長佔比 (AL 3~4)", f"{high_pct}%")

                st.markdown("---")
                col_chart, col_peak = st.columns([1.3, 1.0])

                with col_chart:
                    st.markdown("#### 人體工學風險時序變化曲線")
                    timestamps = [r["timestamp_sec"] for r in records]
                    chart_data = {
                        "時間 (秒)": timestamps,
                        f"{method} 風險分數": scores
                    }
                    st.line_chart(chart_data, x="時間 (秒)", y=f"{method} 風險分數")

                with col_peak:
                    st.markdown("#### 🚨 全片最高危害瞬間截圖 (Peak Hazard Frame)")
                    if peak_img is not None:
                        peak_rgb = cv2.cvtColor(peak_img, cv2.COLOR_BGR2RGB)
                        st.image(peak_rgb, caption=f"時間: {peak_rec.get('timestamp_sec', 0.0)}s ｜ 分數: {peak_score} ｜ AL: {peak_rec.get('action_level', 0)}", use_column_width=True)

                # 下載完整 CSV
                import io
                import csv
                csv_buffer = io.StringIO()
                if records:
                    fieldnames = [k for k in records[0].keys() if k != "raw_angles"]
                    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in records:
                        writer.writerow({k: v for k, v in r.items() if k != "raw_angles"})

                st.download_button(
                    label="📥 下載逐幀評估報表 (.CSV)",
                    data=csv_buffer.getvalue().encode("utf-8-sig"),
                    file_name=f"{uploaded_video.name}_ergo_analysis.csv",
                    mime="text/csv"
                )


# ==============================================================================
# Tab 3: 專題批次數據與學術驗證 (Batch & Ground Truth)
# ==============================================================================
with tab_batch:
    st.subheader("期末專題研究：自動評估 vs. 人工專家評分 (Ground Truth 驗證)")
    st.markdown("""
    本專案專為 **2026 人因工程專題 (題目 01)** 設計，支援一鍵計算 AI 與專家評分的信度指標：
    - **評分完全一致率 (Exact Agreement %)** 與 **容許 ±1 分一致率**
    - **線性加權 Cohen's Kappa 係數** (國際人體工學標準評估指標)
    - **關節角度平均絕對誤差 (MAE) 與 RMSE**
    """)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        folder_path = st.text_input("本機測試資料夾路徑", value="", placeholder="例如: C:/Users/yan/Desktop/test_dataset")
    with col_b2:
        gt_file = st.file_uploader("專家人工評分標準檔 (ground_truth.csv)", type=["csv"])

    if st.button("📊 執行批次分析與學術驗證", type="primary"):
        if folder_path and os.path.exists(folder_path):
            from batch_evaluator import BatchEvaluator
            with st.spinner("正在批次分析資料夾內之照片與影片..."):
                evaluator = BatchEvaluator(
                    method=method,
                    side=side_choice,
                    output_dir="outputs",
                    manual_modifiers=manual_modifiers
                )
                results = evaluator.run_directory(folder_path)

            st.success(f"批次評估完成！共處理 {len(results)} 筆檔案，報表已儲存至 outputs 目錄。")

            # 若有上傳 Ground Truth，即刻計算 Kappa 與 MAE
            if gt_file is not None:
                gt_text = gt_file.read().decode("utf-8-sig").splitlines()
                reader = csv.DictReader(gt_text)
                gt_map = {row["filename"]: row for row in reader if "filename" in row}

                matched = [(r, gt_map[r["filename"]]) for r in results if r["filename"] in gt_map]
                st.subheader(f"📈 實驗驗證統計結果 (有效比對樣本: {len(matched)})")

                if matched:
                    y_true_s = [int(float(gt["final_score"])) for _, gt in matched if "final_score" in gt]
                    y_pred_s = [int(float(pred["final_score"])) for pred, _ in matched if "final_score" in pred]

                    if y_true_s and y_pred_s:
                        exact_match = np.mean(np.array(y_true_s) == np.array(y_pred_s)) * 100.0
                        tol_match = np.mean(np.abs(np.array(y_true_s) - np.array(y_pred_s)) <= 1) * 100.0
                        kappa = compute_cohens_weighted_kappa(np.array(y_true_s), np.array(y_pred_s), min_val=1, max_val=7 if method=="RULA" else 15)

                        col_k1, col_k2, col_k3 = st.columns(3)
                        col_k1.metric("評分完全一致率", f"{exact_match:.1f}%")
                        col_k2.metric("容許 ±1分一致率", f"{tol_match:.1f}%")
                        col_k3.metric("加權 Cohen's Kappa", f"{kappa:.3f}")
        else:
            st.warning("請先輸入有效的本機測試資料夾路徑。")
