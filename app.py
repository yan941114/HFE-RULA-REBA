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

import csv
import io
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
from src.live_monitor_core import LiveErgoSession
from src.rula_reba_engine import (
    calculate_rula,
    calculate_reba,
    compute_rula_force_score,
    flatten_assessment_result,
    simulate_what_if_improvements,
)
from src.report_generator import generate_html_report
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

# 自訂 CSS 樣式 (深淺色模式通用高對比設計)
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
    .score-card {
        padding: 18px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .score-card-al1 {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        color: #065F46 !important;
    }
    .score-card-al2 {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        color: #78350F !important;
    }
    .score-card-al3 {
        background-color: #FFF7ED;
        border-left: 6px solid #F97316;
        color: #7C2D12 !important;
    }
    .score-card-al4 {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #7F1D1D !important;
    }
    .metric-val {
        font-size: 2.6rem;
        font-weight: 900;
        line-height: 1.15;
        margin: 6px 0;
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
    "骨架著色與批次驗證主量表",
    options=["RULA", "REBA"],
    help="照片、影片與即時監測模式皆會【同步計算 RULA 與 REBA】；此選項用於決定骨架疊圖主色與批次驗證基準。"
)

side_choice = st.sidebar.selectbox(
    "肢體側別 (Evaluation Side)",
    options=["auto", "worst", "left", "right"],
    format_func=lambda x: {
        "auto": "自動選擇最佳可見側 (Auto)",
        "worst": "自動取較高危害側 (Worst-Case 保守原則)",
        "left": "左側 (Left: 僅評估左半身)",
        "right": "右側 (Right: 僅評估右半身)"
    }[x]
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

act_rapid = st.sidebar.checkbox("姿勢劇烈變動或基底不穩 (REBA 專用)")
coupling_score = st.sidebar.selectbox(
    "握把耦合程度 (REBA 專用)",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["0: 良好 (舒適握把與中等握力)", "1: 尚可 (可握但不理想)", "2: 差 (勉強能抓但易滑脫)", "3: 危險 (無握把或不安全)"][x]
)

st.sidebar.markdown("---")
st.sidebar.subheader("姿勢與環境人工補正")
with st.sidebar.expander("進階姿態補正項 (進一步微調)", expanded=False):
    shoulder_raised = st.checkbox("肩膀明顯聳起 (+1)")
    arm_abducted = st.checkbox("手臂外展或側向張開 (+1)")
    arm_supported = st.checkbox("手臂有支撐或有扶手 (-1)")
    across_midline = st.checkbox("前臂越過身體中線 (+1，RULA 專用)")
    out_to_side = st.checkbox("前臂向身體外側張開 (+1，RULA 專用)")
    wrist_deviated = st.checkbox("手腕側偏 (橈偏/尺偏) (+1)")
    wrist_twist = st.selectbox("手腕扭轉程度", options=[1, 2], format_func=lambda x: "中立範圍 (1分)" if x == 1 else "極限旋轉 (2分)")
    trunk_twisted = st.checkbox("軀幹扭轉 (+1)")
    trunk_side_bending = st.checkbox("軀幹側彎 (+1)")
    neck_twisted = st.checkbox("頸部扭轉 (+1)")
    neck_side_bending = st.checkbox("頸部側彎 (+1)")
    is_sitting = st.checkbox("作業員為坐姿 (REBA 不計膝彎曲)")
    sitting_supported = st.checkbox("坐姿且具良好椅背支撐 (RULA 軀幹=1分)")
    legs_supported = st.checkbox("雙腳支撐良好且平衡", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("視覺呈現設定")
draw_hud = st.sidebar.checkbox(
    "在影像疊加 HUD 抬頭資訊框",
    value=False,
    help="預設關閉，保持畫面整潔不遮擋人體；所有評估資訊均已完整顯示於右側看板"
)
show_angle_labels = st.sidebar.checkbox(
    "在骨架旁標註關節幾何角度 (科技標籤)",
    value=False,
    help="在肩膀、手肘、軀幹旁繪製半透明角度數值膠囊，適合簡報與報告截圖"
)

# RULA Force/Load Score (0~3) 標準轉換邏輯
rula_force = compute_rula_force_score(
    load_kg=load_kg,
    act_static=act_static,
    act_repeat=act_repeat,
    load_shock=load_shock,
)

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
    "across_midline": across_midline,
    "out_to_side": out_to_side,
    "wrist_deviated": wrist_deviated,
    "wrist_twist": wrist_twist,
    "wrist_twisted": (wrist_twist == 2),  # 支援 REBA 布林格式連動
    "trunk_twisted": trunk_twisted,
    "trunk_side_bending": trunk_side_bending,
    "neck_twisted": neck_twisted,
    "neck_side_bending": neck_side_bending,
    "is_sitting": is_sitting,
    "sitting_supported": sitting_supported,
    "legs_supported": legs_supported,
}


# ==============================================================================
# 共用雙量表 UI 渲染輔助函式 (避免重複程式碼)
# ==============================================================================
AL_THEME_MAP = {
    0: {"border": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "score": "#047857"},
    1: {"border": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "score": "#047857"},
    2: {"border": "#F59E0B", "bg": "#FFFBEB", "text": "#78350F", "score": "#B45309"},
    3: {"border": "#F97316", "bg": "#FFF7ED", "text": "#7C2D12", "score": "#C2410C"},
    4: {"border": "#EF4444", "bg": "#FEF2F2", "text": "#7F1D1D", "score": "#B91C1C"},
}


def build_scorecard_html(title: str, score_val: int, al: int, risk_level: str, action_desc: str = "", extra_meta: str = "") -> str:
    """產生高對比 RULA / REBA 風險總分卡片 HTML。"""
    theme = AL_THEME_MAP.get(min(4, max(0, int(al))), AL_THEME_MAP[1])
    extra_html = f'<div style="font-size: 0.85rem; color: #4B5563; margin-top: 4px;">{extra_meta}</div>' if extra_meta else ""
    desc_html = f'<div style="font-size: 0.88rem; color: #374151; margin-top: 6px; line-height: 1.4;">📋 {action_desc}</div>' if action_desc else ""
    return f"""
    <div style="background-color: {theme['bg']}; border-left: 6px solid {theme['border']}; padding: 14px 16px; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 3px 5px -1px rgba(0,0,0,0.06);">
        <div style="font-size: 1.0rem; font-weight: 700; color: {theme['text']};">{title}</div>
        <div style="font-size: 2.2rem; font-weight: 900; color: {theme['score']}; line-height: 1.15; margin: 4px 0;">
            {score_val} <span style="font-size: 1.0rem; font-weight: 600; color: {theme['text']};">分 ｜ AL {al}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 700; color: {theme['text']};">{risk_level}</div>
        {extra_html}
        {desc_html}
    </div>
    """


def render_method_breakdown_and_export(
    method_name: str,
    ass_obj: Dict[str, Any],
    angles_obj: Dict[str, float],
    annotated_bgr_obj: np.ndarray,
    diag_obj: Optional[Dict[str, Any]],
    modifiers_obj: Dict[str, Any],
    item_name: str,
    dl_key: str,
    expanded_whatif: bool = False,
) -> None:
    """渲染單一量表 (RULA 或 REBA) 之查表拆解明細、What-If 改善模擬與 HTML 檢核單下載按鈕。"""
    flat = flatten_assessment_result(ass_obj)
    is_reba = (method_name.upper() == "REBA")

    with st.expander(f"📊 查看 {method_name} 查表拆解明細 (表 A / 表 B / 表 C 得分矩陣)", expanded=False):
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            if not is_reba:
                st.markdown("**📌 表 A 群 (上臂、前臂、手腕)**")
                st.write(f"- 上臂姿勢得分: **{flat.get('upper_arm_score', '-')} 分**")
                st.write(f"- 前臂姿勢得分: **{flat.get('lower_arm_score', '-')} 分**")
                st.write(f"- 手腕與扭轉得分: **{flat.get('wrist_score', '-')} 分**")
                st.write(f"- 表 A 查表得分: **{flat.get('table_a_score', '-')} 分**")
                st.write(f"- 修正後得分 C: **{flat.get('wrist_arm_score', '-')} 分**")
            else:
                st.markdown("**📌 表 A 群 (軀幹、頸部、腿部)**")
                st.write(f"- 軀幹姿勢得分: **{flat.get('trunk_score', '-')} 分**")
                st.write(f"- 頸部姿勢得分: **{flat.get('neck_score', '-')} 分**")
                st.write(f"- 腿部得分 (含支撐+膝彎曲): **{flat.get('legs_score', '-')} 分**")
                st.write(f"- 表 A 查表得分: **{flat.get('table_a_score', '-')} 分**")
                st.write(f"- 修正後得分 A: **{flat.get('score_a', '-')} 分**")
        with col_sub2:
            if not is_reba:
                st.markdown("**📌 表 B 群 (頸部、軀幹、腿部)**")
                st.write(f"- 頸部姿勢得分: **{flat.get('neck_score', '-')} 分**")
                st.write(f"- 軀幹姿勢得分: **{flat.get('trunk_score', '-')} 分**")
                st.write(f"- 腿部支撐平衡得分 (不計膝角): **{flat.get('legs_score', '-')} 分**")
                st.write(f"- 表 B 查表得分: **{flat.get('table_b_score', '-')} 分**")
                st.write(f"- 修正後得分 D: **{flat.get('neck_trunk_score', '-')} 分**")
            else:
                st.markdown("**📌 表 B 群 (上臂、前臂、手腕)**")
                st.write(f"- 上臂姿勢得分: **{flat.get('upper_arm_score', '-')} 分**")
                st.write(f"- 前臂姿勢得分: **{flat.get('lower_arm_score', '-')} 分**")
                st.write(f"- 手腕姿勢得分: **{flat.get('wrist_score', '-')} 分**")
                st.write(f"- 表 B 查表得分: **{flat.get('table_b_score', '-')} 分**")
                st.write(f"- 修正後得分 B: **{flat.get('score_b', '-')} 分**")

    what_if_scenarios = simulate_what_if_improvements(angles_obj, modifiers_obj, method=method_name)
    with st.expander(f"💡 {method_name} AI 自動根因診斷與改善模擬 (What-If Simulation)", expanded=expanded_whatif):
        if what_if_scenarios:
            st.markdown("**🎯 改善效益排序與敏感度分析（AI 自動試算中立位效益）：**")
            for idx, sc in enumerate(what_if_scenarios, 1):
                delta = sc["score_drop"]
                badge = f"🔥 總分 -{delta} 分 (效益顯著)" if delta >= 2 else (f"✨ 總分 -{delta} 分" if delta > 0 else "持平 (0分)")
                st.markdown(f"**{idx}. {sc['category']}** ｜ `{badge}`")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;➜ **模擬操作**：{sc['action']}")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;➜ **預期結果**：總分降至 **{sc['new_score']} 分** (行動水準 AL {sc['new_action_level']}) ｜ {sc.get('priority', '')}")
        else:
            st.info("當前姿勢各項關鍵角度均接近建議中立位，暫無顯著降分改善項目。")

    html_report = generate_html_report(
        assessment_result=ass_obj,
        angles=angles_obj,
        annotated_bgr=annotated_bgr_obj,
        method=method_name,
        manual_modifiers=modifiers_obj,
        what_if_scenarios=what_if_scenarios,
        diagnostics=diag_obj,
        item_name=item_name,
    )
    st.download_button(
        label=f"📄 下載 {method_name} 正式人因工程檢核單 (.html / PDF)",
        data=html_report.encode("utf-8"),
        file_name=f"ErgoReport_{method_name}_{item_name}.html",
        mime="text/html",
        key=dl_key,
    )


# ==============================================================================
# 主頁面標題與分頁
# ==============================================================================
st.markdown('<div class="main-header">🏋️‍♂️ ErgoVision AI 人體工學自動風險評估系統</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">結合 YOLO11-Pose 骨架偵測與國際標準 RULA / REBA 查表量表 ｜ 人因工程與實驗設計專題</div>', unsafe_allow_html=True)

tab_photo, tab_video, tab_live, tab_batch = st.tabs([
    "📷 單張照片評估 (Photo Mode)",
    "🎥 影片時序分析與峰值捕捉 (Video Mode)",
    "📹 即時攝影機偵測 (Live Camera)",
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
                manual_modifiers=manual_modifiers,
                draw_hud=draw_hud,
                show_angle_labels=show_angle_labels
            )

        if res is not None:
            col_img, col_score = st.columns([1.1, 1.0])

            with col_img:
                st.markdown("#### 骨架與關節疊圖標註")
                img_rgb = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(img_rgb, width="stretch")
                diag = res["diagnostics"]
                facing_zh = "朝左 (Left)" if diag['facing'] == 'left' else ("朝右 (Right)" if diag['facing'] == 'right' else "正面/未定")
                side_zh = "左側 (Left)" if diag['side_used'] == 'left' else "右側 (Right)"
                st.caption(f"🧭 面向判定: **{facing_zh}** ｜ 🎯 主要評估肢體: **{side_zh}** ｜ 左側信心: {diag['left_confidence']:.2f} ｜ 右側信心: {diag['right_confidence']:.2f}")

                # 左右側身體負荷不對稱警示 (Asymmetry Alert)
                if diag.get("asymmetry_alert", False):
                    st.warning(f"⚠️ **身體負荷不對稱警示**：{diag.get('asymmetry_reason')}。建議檢查作業員是否有單側代償或單手提重不良姿勢！")

            with col_score:
                st.markdown("#### 雙量表同步評估結果看板 (RULA & REBA)")
                rula_ass = res["rula_result"]
                reba_ass = res["reba_result"]
                rula_score_v = rula_ass.get("grand_score", 1)
                reba_score_v = reba_ass.get("final_reba_score", 1)

                col_card_rula, col_card_reba = st.columns(2)
                with col_card_rula:
                    st.markdown(
                        build_scorecard_html(
                            "RULA 綜合總分 (上肢/坐姿)",
                            rula_score_v,
                            rula_ass["action_level"],
                            rula_ass["risk_level"],
                            rula_ass["action_description"],
                        ),
                        unsafe_allow_html=True,
                    )
                with col_card_reba:
                    st.markdown(
                        build_scorecard_html(
                            "REBA 綜合總分 (全身/搬運)",
                            reba_score_v,
                            reba_ass["action_level"],
                            reba_ass["risk_level"],
                            reba_ass["action_description"],
                        ),
                        unsafe_allow_html=True,
                    )

                st.markdown("##### 關節角度摘要")
                angles = res["angles"]
                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("軀幹角度", f"{angles['trunk']:+.1f}°")
                col_a2.metric("頸部角度", f"{angles['neck']:+.1f}°")
                col_a3.metric("膝關節角度 (僅 REBA)", f"{angles['knee']:.1f}°")

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("上臂屈伸", f"{angles['upper_arm']:+.1f}°")
                col_b2.metric("前臂屈曲", f"{angles['lower_arm']:.1f}°")
                col_b3.metric("手腕角度", f"{angles['wrist']:.1f}°")

                sub_tab_rula, sub_tab_reba = st.tabs(["📋 RULA 詳細拆解與改善建議", "📋 REBA 詳細拆解與改善建議"])
                with sub_tab_rula:
                    render_method_breakdown_and_export(
                        method_name="RULA",
                        ass_obj=rula_ass,
                        angles_obj=angles,
                        annotated_bgr_obj=res["annotated_rula"],
                        diag_obj=diag,
                        modifiers_obj=manual_modifiers,
                        item_name=uploaded_image.name,
                        dl_key="dl_photo_report_rula",
                        expanded_whatif=True,
                    )
                with sub_tab_reba:
                    render_method_breakdown_and_export(
                        method_name="REBA",
                        ass_obj=reba_ass,
                        angles_obj=angles,
                        annotated_bgr_obj=res["annotated_reba"],
                        diag_obj=diag,
                        modifiers_obj=manual_modifiers,
                        item_name=uploaded_image.name,
                        dl_key="dl_photo_report_reba",
                        expanded_whatif=True,
                    )
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
                        sample_interval=sample_step,
                        draw_hud=draw_hud,
                        show_angle_labels=show_angle_labels
                    )
                    st.session_state["cached_video_records"] = res_video["time_series_records"]
                    st.session_state["cached_video_peak_img"] = res_video["peak_risk_image"]
                    st.session_state["cached_video_peak_rec"] = res_video["peak_risk_record"]
                    st.session_state["cached_video_peak_rula_img"] = res_video["peak_rula_image"]
                    st.session_state["cached_video_peak_rula_rec"] = res_video["peak_rula_record"]
                    st.session_state["cached_video_peak_reba_img"] = res_video["peak_reba_image"]
                    st.session_state["cached_video_peak_reba_rec"] = res_video["peak_reba_record"]
                    st.session_state["cached_video_rula_high_pct"] = res_video["rula_high_risk_exposure_pct"]
                    st.session_state["cached_video_reba_high_pct"] = res_video["reba_high_risk_exposure_pct"]
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
                st.session_state["cached_video_peak_rula_rec"] = res_recalc["peak_rula_record"]
                st.session_state["cached_video_peak_reba_rec"] = res_recalc["peak_reba_record"]
                st.session_state["cached_video_rula_high_pct"] = res_recalc["rula_high_risk_exposure_pct"]
                st.session_state["cached_video_reba_high_pct"] = res_recalc["reba_high_risk_exposure_pct"]

            # 顯示結果
            records = st.session_state.get("cached_video_records", [])
            peak_rula_img = st.session_state.get("cached_video_peak_rula_img")
            peak_rula_rec = st.session_state.get("cached_video_peak_rula_rec") or {}
            peak_reba_img = st.session_state.get("cached_video_peak_reba_img")
            peak_reba_rec = st.session_state.get("cached_video_peak_reba_rec") or {}

            if records:
                st.success(f"影片雙量表分析完成！共提取 {len(records)} 幀有效姿態資料。")

                rula_scores_v = [r["rula_score"] for r in records]
                reba_scores_v = [r["reba_score"] for r in records]
                rula_high_pct_v = st.session_state.get("cached_video_rula_high_pct", 0.0)
                reba_high_pct_v = st.session_state.get("cached_video_reba_high_pct", 0.0)

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("RULA 最高危害分", f"{peak_rula_rec.get('rula_score', 0)} 分")
                col_m2.metric("RULA 峰值時間點", f"{peak_rula_rec.get('timestamp_sec', 0.0)} 秒")
                col_m3.metric("RULA 全程平均分", f"{np.mean(rula_scores_v):.2f} 分")
                col_m4.metric("RULA 高危害佔比 (AL 3~4)", f"{rula_high_pct_v}%")

                col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                col_n1.metric("REBA 最高危害分", f"{peak_reba_rec.get('reba_score', 0)} 分")
                col_n2.metric("REBA 峰值時間點", f"{peak_reba_rec.get('timestamp_sec', 0.0)} 秒")
                col_n3.metric("REBA 全程平均分", f"{np.mean(reba_scores_v):.2f} 分")
                col_n4.metric("REBA 高危害佔比 (AL 3~4)", f"{reba_high_pct_v}%")

                st.markdown("---")
                col_chart, col_peak = st.columns([1.2, 1.0])

                with col_chart:
                    st.markdown("#### 雙量表風險時序變化曲線 (RULA vs. REBA)")
                    timestamps = [r["timestamp_sec"] for r in records]
                    chart_data = {
                        "時間 (秒)": timestamps,
                        "RULA 風險分數": rula_scores_v,
                        "REBA 風險分數": reba_scores_v,
                    }
                    st.line_chart(chart_data, x="時間 (秒)", y=["RULA 風險分數", "REBA 風險分數"])

                    # 下載完整雙量表 CSV
                    csv_buffer = io.StringIO()
                    fieldnames = [k for k in records[0].keys() if k != "raw_angles"]
                    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for r in records:
                        writer.writerow({k: v for k, v in r.items() if k != "raw_angles"})

                    st.download_button(
                        label="📥 下載雙量表逐幀評估報表 (.CSV)",
                        data=csv_buffer.getvalue().encode("utf-8-sig"),
                        file_name=f"{uploaded_video.name}_dual_ergo_analysis.csv",
                        mime="text/csv",
                        key="dl_video_csv",
                    )

                with col_peak:
                    st.markdown("#### 🚨 各自最高危害瞬間截圖 (Independent Peak Frames)")
                    vtab_rula, vtab_reba = st.tabs(["🚨 RULA 最高危害瞬間", "🚨 REBA 最高危害瞬間"])

                    for sub_tab, m_name, p_img, p_rec in (
                        (vtab_rula, "RULA", peak_rula_img, peak_rula_rec),
                        (vtab_reba, "REBA", peak_reba_img, peak_reba_rec),
                    ):
                        with sub_tab:
                            if p_img is not None and p_rec:
                                st.image(cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB), width="stretch")
                                p_score = p_rec.get(f"{m_name.lower()}_score", p_rec.get("score", 0))
                                p_al = p_rec.get(f"{m_name.lower()}_action_level", p_rec.get("action_level", 1))
                                p_risk = p_rec.get(f"{m_name.lower()}_risk_level", p_rec.get("risk_level", ""))
                                p_time = p_rec.get("timestamp_sec", 0.0)
                                st.markdown(
                                    build_scorecard_html(
                                        f"{m_name} 危害峰值 (時間點: {p_time} 秒)",
                                        p_score,
                                        p_al,
                                        p_risk,
                                    ),
                                    unsafe_allow_html=True,
                                )

                                p_angles = p_rec.get("raw_angles", {
                                    "trunk": p_rec.get("trunk_angle", 0.0),
                                    "neck": p_rec.get("neck_angle", 0.0),
                                    "upper_arm": p_rec.get("upper_arm_angle", 0.0),
                                    "lower_arm": p_rec.get("lower_arm_angle", 90.0),
                                    "wrist": p_rec.get("wrist_angle", 0.0),
                                    "knee": p_rec.get("knee_angle", 0.0),
                                })
                                col_pk1, col_pk2 = st.columns(2)
                                with col_pk1:
                                    st.write(f"📐 軀幹角度: **{p_angles.get('trunk', 0.0):+.1f}°**")
                                    st.write(f"📐 頸部角度: **{p_angles.get('neck', 0.0):+.1f}°**")
                                    st.write(f"📐 膝關節 (僅 REBA): **{p_angles.get('knee', 0.0):.1f}°**")
                                with col_pk2:
                                    st.write(f"📐 上臂角度: **{p_angles.get('upper_arm', 0.0):+.1f}°**")
                                    st.write(f"📐 前臂手肘: **{p_angles.get('lower_arm', 0.0):.1f}°**")

                                peak_ass = calculate_reba(p_angles, manual_modifiers) if m_name == "REBA" else calculate_rula(p_angles, manual_modifiers)
                                render_method_breakdown_and_export(
                                    method_name=m_name,
                                    ass_obj=peak_ass,
                                    angles_obj=p_angles,
                                    annotated_bgr_obj=p_img,
                                    diag_obj={"facing": "N/A", "side_used": p_rec.get("side_used", "auto"), "left_confidence": 1.0, "right_confidence": 1.0},
                                    modifiers_obj=manual_modifiers,
                                    item_name=f"Peak_{m_name}_{uploaded_video.name}",
                                    dl_key=f"dl_peak_report_{m_name.lower()}",
                                    expanded_whatif=False,
                                )


# ==============================================================================
# Tab 3: 即時攝影機動態偵測與自動評分 (Live Camera Mode)
# ==============================================================================
with tab_live:
    st.subheader("即時攝影機動態骨架偵測、基準校正與 RULA / REBA 自動評分")
    st.info(
        "💡 支援三層防抖濾波（Median + 自適應 EMA + 死區抑制）與「中立姿勢基準校正」，並即時連動左側邊欄負重／環境參數。"
        "若於期末專題上台 Live Demo 需要最高流暢度與全螢幕熱鍵控制，也可在終端機執行：`python live_ergo_monitor.py`。"
    )

    if "live_session" not in st.session_state:
        st.session_state["live_session"] = LiveErgoSession(analyzer=analyzer)
    live_session: LiveErgoSession = st.session_state["live_session"]

    col_lc1, col_lc2, col_lc3, col_lc4 = st.columns([1.0, 1.0, 1.2, 1.2])
    with col_lc1:
        cam_idx = st.number_input("攝影機編號 (Index)", min_value=0, max_value=5, value=0, step=1, key="live_cam_idx")
        live_mirror = st.checkbox("鏡像翻轉 (Mirror)", value=True, key="live_mirror_chk")
    with col_lc2:
        live_ema = st.slider("EMA 平滑係數", min_value=0.10, max_value=0.80, value=0.30, step=0.05, key="live_ema_slider")
        live_deadband = st.slider("死區抑制門檻 (°)", min_value=0.0, max_value=5.0, value=1.5, step=0.5, key="live_dead_slider")
    with col_lc3:
        run_live = st.toggle("🟢 開啟即時攝影機動態偵測", value=False, key="live_run_toggle")
        record_live = st.checkbox("🔴 同步錄製並自動捕捉最高危害峰值", value=True, key="live_rec_chk")
    with col_lc4:
        if st.button("🎯 設為中立姿勢基準 (校正鏡頭傾角)", key="live_calib_btn"):
            if live_session.calibrate_baseline():
                st.success("已成功記錄目前直立姿勢為 0° 基準！")
            else:
                st.session_state["live_calib_pending"] = True
                st.info("已預約校正：將於開啟攝影機偵測到人體後自動歸零基準。")
        if st.button("🔄 清除基準校正", key="live_clear_calib_btn"):
            st.session_state["live_calib_pending"] = False
            live_session.clear_baseline()
            st.info("已清除基準偏移量。")

    if run_live:
        live_session.set_filter_params(ema_alpha_base=live_ema, deadband_deg=live_deadband)
        if record_live and not live_session.is_recording:
            live_session.start_recording()
        elif not record_live and live_session.is_recording:
            if live_session.records:
                st.session_state["live_last_summary"] = live_session.stop_recording()
            else:
                live_session.stop_recording()

        col_live_img, col_live_info = st.columns([1.15, 1.0])
        with col_live_img:
            ph_img = st.empty()
        with col_live_info:
            ph_card = st.empty()
            ph_angles = st.empty()
            ph_warn = st.empty()
            ph_chart = st.empty()

        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(cam_idx))

        if not cap.isOpened():
            st.error(
                f"無法開啟攝影機 (Index={int(cam_idx)})。請確認電腦已連接 Webcam、瀏覽器/系統權限已允許，且無其他程式佔用鏡頭。"
            )
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            live_al_theme = {
                0: {"border": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "score": "#047857"},
                1: {"border": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "score": "#047857"},
                2: {"border": "#F59E0B", "bg": "#FFFBEB", "text": "#78350F", "score": "#B45309"},
                3: {"border": "#F97316", "bg": "#FFF7ED", "text": "#7C2D12", "score": "#C2410C"},
                4: {"border": "#EF4444", "bg": "#FEF2F2", "text": "#7F1D1D", "score": "#B91C1C"},
            }
            fail_reads = 0
            loop_idx = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        fail_reads += 1
                        if fail_reads >= 15:
                            ph_warn.error("攝影機讀取中斷，已自動停止即時偵測。")
                            break
                        time.sleep(0.03)
                        continue
                    fail_reads = 0
                    loop_idx += 1

                    res_live = live_session.process_frame(
                        frame,
                        method=method,
                        side=side_choice,
                        manual_modifiers=manual_modifiers,
                        mirror=live_mirror,
                        draw_hud=draw_hud,
                        show_angle_labels=show_angle_labels,
                    )

                    if st.session_state.get("live_calib_pending") and res_live.get("detected"):
                        live_session.calibrate_baseline()
                        st.session_state["live_calib_pending"] = False

                    img_rgb = cv2.cvtColor(res_live["annotated_image"], cv2.COLOR_BGR2RGB)
                    ph_img.image(img_rgb, width="stretch")

                    if res_live.get("detected"):
                        rula_ass_l = res_live["rula_result"]
                        reba_ass_l = res_live["reba_result"]
                        rula_s_l = res_live["rula_score"]
                        rula_al_l = res_live["rula_action_level"]
                        reba_s_l = res_live["reba_score"]
                        reba_al_l = res_live["reba_action_level"]
                        ang_l = res_live["angles"]
                        diag_l = res_live["diagnostics"]
                        calib_tag = "已校正" if res_live.get("is_calibrated") else "未校正"
                        rec_tag = f" ｜ 🔴 錄製中 {res_live.get('recorded_duration_sec', 0.0):.1f}s" if res_live.get("is_recording") else ""

                        card_rula_html = build_scorecard_html(
                            f"RULA 即時評估 ｜ FPS: {res_live['fps']:.1f}",
                            rula_s_l,
                            rula_al_l,
                            rula_ass_l["risk_level"],
                            extra_meta=f"評估側: {diag_l['side_used'].upper()} (L:{diag_l.get('left_rula_score', '-')} R:{diag_l.get('right_rula_score', '-')}) ｜ 基準: {calib_tag}",
                        )
                        card_reba_html = build_scorecard_html(
                            f"REBA 即時評估{rec_tag}",
                            reba_s_l,
                            reba_al_l,
                            reba_ass_l["risk_level"],
                            extra_meta=f"評估側: {diag_l['side_used'].upper()} (L:{diag_l.get('left_reba_score', '-')} R:{diag_l.get('right_reba_score', '-')})",
                        )
                        ph_card.markdown(
                            f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">{card_rula_html}{card_reba_html}</div>',
                            unsafe_allow_html=True,
                        )

                        ph_angles.markdown(
                            f"**軀幹**: `{ang_l['trunk']:+.1f}°` ｜ **頸部**: `{ang_l['neck']:+.1f}°` ｜ **上臂**: `{ang_l['upper_arm']:+.1f}°`  \n"
                            f"**前臂**: `{ang_l['lower_arm']:.1f}°` ｜ **手腕**: `{ang_l['wrist']:.1f}°` ｜ **膝關節 (僅 REBA)**: `{ang_l['knee']:.1f}°`"
                        )

                        warnings_txt = []
                        if diag_l.get("asymmetry_alert"):
                            warnings_txt.append(f"⚠️ **身體負荷不對稱警示**：{diag_l.get('asymmetry_reason')}")
                        if res_live.get("high_risk_streak_sec", 0.0) >= 2.0:
                            warnings_txt.append(
                                f"🚨 **高危害姿勢持續過久**：RULA AL≥3 連續 {res_live.get('rula_high_risk_streak_sec', 0.0):.1f}s / "
                                f"REBA AL≥3 連續 {res_live.get('reba_high_risk_streak_sec', 0.0):.1f}s，請立即調整姿勢！"
                            )
                        if warnings_txt:
                            ph_warn.warning("  \n".join(warnings_txt))
                        else:
                            ph_warn.empty()

                        if loop_idx % 5 == 0 and len(live_session.rula_score_history) >= 2:
                            ph_chart.line_chart({
                                "RULA 即時分數": list(live_session.rula_score_history),
                                "REBA 即時分數": list(live_session.reba_score_history),
                            })

                        if loop_idx % 10 == 0 and live_session.is_recording and live_session.records:
                            st.session_state["live_last_summary"] = live_session.get_recording_summary()
                    else:
                        ph_card.info(f"⏳ 等待人員進入攝影機畫面... (FPS: {res_live['fps']:.1f})")
                        ph_angles.empty()
                        ph_warn.empty()
            finally:
                cap.release()
                if live_session.is_recording:
                    if live_session.records:
                        st.session_state["live_last_summary"] = live_session.stop_recording()
                    else:
                        live_session.stop_recording()
    else:
        if live_session.is_recording:
            if live_session.records:
                st.session_state["live_last_summary"] = live_session.stop_recording()
            else:
                live_session.stop_recording()

        summary = st.session_state.get("live_last_summary")
        if summary and summary.get("records"):
            st.markdown("---")
            st.subheader("📊 最近一次即時動態雙量表評估摘要與峰值報告")
            col_lm1, col_lm2, col_lm3, col_lm4 = st.columns(4)
            col_lm1.metric("RULA 最高危害分", f"{summary.get('peak_rula_score', summary['peak_score'])} 分")
            col_lm2.metric("RULA 平均 / 高危害佔比", f"{summary.get('avg_rula_score', summary['avg_score']):.1f} / {summary.get('rula_high_risk_pct', summary['high_risk_pct']):.1f}%")
            col_lm3.metric("REBA 最高危害分", f"{summary.get('peak_reba_score', 0)} 分")
            col_lm4.metric("REBA 平均 / 高危害佔比", f"{summary.get('avg_reba_score', 0.0):.1f} / {summary.get('reba_high_risk_pct', 0.0):.1f}%")

            col_lc_left, col_lc_right = st.columns([1.2, 1.0])
            with col_lc_left:
                st.markdown(f"#### 雙量表時序風險曲線 (總錄測時長：{summary['duration_sec']:.1f} 秒)")
                live_recs = summary["records"]
                st.line_chart(
                    {
                        "時間 (秒)": [r["timestamp_sec"] for r in live_recs],
                        "RULA 風險分數": [r["rula_score"] for r in live_recs],
                        "REBA 風險分數": [r["reba_score"] for r in live_recs],
                    },
                    x="時間 (秒)",
                    y=["RULA 風險分數", "REBA 風險分數"],
                )
                csv_buf_live = io.StringIO()
                fnames_live = [k for k in live_recs[0].keys() if k != "raw_angles"]
                writer_live = csv.DictWriter(csv_buf_live, fieldnames=fnames_live, extrasaction="ignore")
                writer_live.writeheader()
                for r in live_recs:
                    writer_live.writerow({k: v for k, v in r.items() if k != "raw_angles"})
                st.download_button(
                    label="📥 下載即時動態雙量表逐幀報表 (.CSV)",
                    data=csv_buf_live.getvalue().encode("utf-8-sig"),
                    file_name="live_ergo_dual_RULA_REBA.csv",
                    mime="text/csv",
                    key="dl_live_csv",
                )

            with col_lc_right:
                st.markdown("#### 🚨 各自最高危害瞬間截圖 (Independent Peak Frames)")
                ltab_rula, ltab_reba = st.tabs(["🚨 RULA 最高危害瞬間", "🚨 REBA 最高危害瞬間"])
                for l_sub, m_name, key_pfx in (
                    (ltab_rula, "RULA", "rula"),
                    (ltab_reba, "REBA", "reba"),
                ):
                    with l_sub:
                        p_img_l = summary.get(f"peak_{key_pfx}_image")
                        p_ass_l = summary.get(f"peak_{key_pfx}_assessment")
                        p_ang_l = summary.get(f"peak_{key_pfx}_angles")
                        p_rec_l = summary.get(f"peak_{key_pfx}_record") or {}
                        p_diag_l = summary.get(f"peak_{key_pfx}_diagnostics")
                        p_score_l = summary.get(f"peak_{key_pfx}_score", 0)
                        if p_img_l is not None and p_ass_l is not None and p_ang_l is not None:
                            st.image(cv2.cvtColor(p_img_l, cv2.COLOR_BGR2RGB), width="stretch")
                            st.markdown(
                                build_scorecard_html(
                                    f"{m_name} 峰值 (時間點: {p_rec_l.get('timestamp_sec', 0.0)} 秒)",
                                    p_score_l,
                                    p_ass_l.get("action_level", 1),
                                    p_ass_l.get("risk_level", ""),
                                ),
                                unsafe_allow_html=True,
                            )
                            render_method_breakdown_and_export(
                                method_name=m_name,
                                ass_obj=p_ass_l,
                                angles_obj=p_ang_l,
                                annotated_bgr_obj=p_img_l,
                                diag_obj=p_diag_l,
                                modifiers_obj=manual_modifiers,
                                item_name=f"LiveSession_{m_name}",
                                dl_key=f"dl_live_html_{key_pfx}",
                                expanded_whatif=False,
                            )
        else:
            st.info("請在上方開啟「🟢 開啟即時攝影機動態偵測」開始動態評估與自動評分；關閉開關後將自動彙整本次峰值得分與改善報告。")


# ==============================================================================
# Tab 4: 專題批次數據與學術驗證 (Batch & Ground Truth)
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
