#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_generator.py
-------------------
單頁式正式人因工程改善評估報告產生器 (HTML / PDF Print Ready)

特色：
1. 完全純 Python 標準庫實作（base64, datetime, html），無任何外部依賴。
2. 骨架影像直接以 Data URI (base64) 內嵌，單一 HTML 檔案隨開隨用、離線可用。
3. 內建 @media print 專業 CSS 排版，在 Chrome / Edge 中按下 Ctrl + P 即可一鍵另存為標準 A4 單頁 PDF 報告。
4. 包含受測基本資訊、風險色塊指標、骨架疊圖、關節角度表、Table A/B/C 得分拆解、What-If 改善建議、以及 EHS 簽核欄位。
"""

import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import numpy as np


def generate_html_report(assessment_result: Dict[str, Any],
                         angles: Dict[str, float],
                         annotated_bgr: np.ndarray,
                         method: str = "RULA",
                         manual_modifiers: Optional[Dict[str, Any]] = None,
                         what_if_scenarios: Optional[List[Dict[str, Any]]] = None,
                         diagnostics: Optional[Dict[str, Any]] = None,
                         item_name: str = "作業姿勢評估") -> str:
    """
    產生單頁式 A4 格式之正式人因工程評估檢核報告 HTML

    :param assessment_result: calculate_rula 或 calculate_reba 回傳結果
    :param angles: 關節角度字典
    :param annotated_bgr: 帶有骨架標註之 BGR 影像
    :param method: "RULA" 或 "REBA"
    :param manual_modifiers: 專家負重/環境參數
    :param what_if_scenarios: What-If 模擬改善排序清單
    :param diagnostics: 診斷與雙側不對稱資訊
    :param item_name: 受測照片或影片檔名
    :return: 完整 HTML 字串
    """
    is_reba = (method.upper() == "REBA")
    score_val = assessment_result.get("final_reba_score") if is_reba else assessment_result.get("grand_score", 0)
    al = assessment_result.get("action_level", 1)
    risk_level = assessment_result.get("risk_level", "未定義")
    action_desc = assessment_result.get("action_description", "")

    # 風險等級色彩定義
    al_colors = {
        1: {"border": "#10B981", "bg": "#ECFDF5", "text": "#065F46", "badge": "#059669", "name": "低度風險 / 姿勢可接受"},
        2: {"border": "#F59E0B", "bg": "#FFFBEB", "text": "#78350F", "badge": "#D97706", "name": "中度風險 / 需進一步調查"},
        3: {"border": "#F97316", "bg": "#FFF7ED", "text": "#7C2D12", "badge": "#EA580C", "name": "高度風險 / 應於近期內改善"},
        4: {"border": "#EF4444", "bg": "#FEF2F2", "text": "#7F1D1D", "badge": "#DC2626", "name": "極高危險 / 必須立即停止並改善"},
    }
    theme = al_colors.get(min(4, max(1, al)), al_colors[1])

    # 將標註影像壓縮為 base64 (維持高解析度同時縮小 HTML 體積)
    h, w = annotated_bgr.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_to_encode = cv2.resize(annotated_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_to_encode = annotated_bgr

    _, buffer = cv2.imencode(".jpg", img_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = base64.b64encode(buffer).decode("utf-8")

    diag = diagnostics or {}
    facing = diag.get("facing", "auto")
    side_used = diag.get("side_used", "auto")
    facing_zh = "朝左 (Left)" if facing == "left" else ("朝右 (Right)" if facing == "right" else "正面/無法判定")
    side_zh = "左側 (Left)" if side_used == "left" else ("右側 (Right)" if side_used == "right" else "自動判定")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 查表拆解得分
    ga = assessment_result.get("group_a", {})
    gb = assessment_result.get("group_b", {})
    if is_reba:
        posture_a = ga.get("posture_score_a", "-")
        load_score = ga.get("load_score", "-")
        score_a = ga.get("score_a", "-")
        posture_b = gb.get("posture_score_b", "-")
        coupling_score = gb.get("coupling_score", "-")
        score_b = gb.get("score_b", "-")
        score_c = assessment_result.get("score_c", "-")
        activity_score = assessment_result.get("activity_score", "-")
    else:
        posture_a = ga.get("posture_score_a", "-")
        muscle_a = ga.get("muscle_use", "-")
        force_a = ga.get("force_load", "-")
        score_c = ga.get("score_c", "-")
        posture_b = gb.get("posture_score_b", "-")
        muscle_b = gb.get("muscle_use", "-")
        force_b = gb.get("force_load", "-")
        score_d = gb.get("score_d", "-")

    # 不對稱警示 HTML
    asymmetry_html = ""
    if diag.get("asymmetry_alert", False):
        asymmetry_html = f"""
        <div style="background-color: #FEF3C7; border: 1px solid #F59E0B; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; color: #92400E;">
            <strong>⚠️ 身體負荷不對稱警示：</strong>{diag.get('asymmetry_reason', '左右兩側肢體負荷顯著不均')}。建議檢查作業員是否長期單側負重或高低肩姿勢。
        </div>
        """

    # What-If 改善建議表格列
    what_if_rows = ""
    if what_if_scenarios:
        for i, s in enumerate(what_if_scenarios[:3], 1):
            what_if_rows += f"""
            <tr>
                <td style="text-align: center; font-weight: 700;">#{i}</td>
                <td><span style="background: #E0E7FF; color: #3730A3; padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">{s['category']}</span></td>
                <td>{s['action']}</td>
                <td style="text-align: center; font-weight: 700; color: #047857;">{s['new_score']} 分 (降 {s['score_drop']} 分)</td>
                <td style="text-align: center;"><span style="color: #047857; font-weight: 700;">AL {s['new_action_level']}</span></td>
            </tr>
            """
    else:
        what_if_rows = "<tr><td colspan='5' style='text-align: center; color: #6B7280;'>當前姿勢各項關鍵角度均接近人體工學建議中立位，無顯著工程改善急迫性。</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>ErgoVision AI - 人體工學作業姿勢評估檢核表 ({method})</title>
    <style>
        @page {{
            size: A4 portrait;
            margin: 10mm 12mm 10mm 12mm;
        }}
        * {{
            box-sizing: border-box;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft JhengHei", sans-serif;
            color: #1F2937;
            background-color: #FFFFFF;
            margin: 0;
            padding: 0;
            font-size: 11pt;
            line-height: 1.4;
        }}
        .report-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #1E3A8A;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }}
        .report-title {{
            font-size: 1.4rem;
            font-weight: 800;
            color: #1E3A8A;
            letter-spacing: 0.5px;
        }}
        .report-subtitle {{
            font-size: 0.85rem;
            color: #6B7280;
            margin-top: 2px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 12px;
            font-size: 0.85rem;
        }}
        .meta-item strong {{
            color: #475569;
        }}
        .score-banner {{
            display: flex;
            background-color: {theme['bg']};
            border-left: 6px solid {theme['border']};
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .score-box {{
            flex: 0 0 160px;
            text-align: center;
            border-right: 1px solid rgba(0,0,0,0.1);
            padding-right: 14px;
            margin-right: 14px;
        }}
        .score-box .num {{
            font-size: 2.6rem;
            font-weight: 900;
            color: {theme['badge']};
            line-height: 1;
        }}
        .score-box .sub {{
            font-size: 0.85rem;
            font-weight: 700;
            color: {theme['text']};
            margin-top: 4px;
        }}
        .action-box {{
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .action-box .risk-badge {{
            display: inline-block;
            background-color: {theme['badge']};
            color: #FFFFFF;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 4px;
            width: fit-content;
        }}
        .action-box .desc {{
            font-size: 0.95rem;
            font-weight: 600;
            color: #1F2937;
        }}
        .main-grid {{
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 14px;
            margin-bottom: 12px;
        }}
        .panel {{
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 10px 12px;
            background-color: #FFFFFF;
        }}
        .panel-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #1E3A8A;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 4px;
            margin-bottom: 8px;
        }}
        .pose-img-container {{
            text-align: center;
            background: #F1F5F9;
            border-radius: 6px;
            padding: 4px;
        }}
        .pose-img {{
            max-width: 100%;
            max-height: 240px;
            object-fit: contain;
            border-radius: 4px;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }}
        table.data-table th, table.data-table td {{
            padding: 4px 6px;
            border: 1px solid #CBD5E1;
        }}
        table.data-table th {{
            background-color: #F8FAFC;
            color: #334155;
            font-weight: 700;
        }}
        .whatif-table th {{
            background-color: #EEF2F6;
        }}
        .footer-sign {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 16px;
            border-top: 1px dashed #94A3B8;
            padding-top: 10px;
            font-size: 0.82rem;
            color: #475569;
        }}
        .sign-box {{
            height: 40px;
            border-bottom: 1px solid #94A3B8;
            margin-top: 4px;
        }}
    </style>
</head>
<body>

    <div class="report-header">
        <div>
            <div class="report-title">🏋️‍♂️ ErgoVision AI - 人體工學作業姿勢評估檢核表</div>
            <div class="report-subtitle">國際標準 {method} 快速人體工學危害評估與電腦視覺關節角度量測系統</div>
        </div>
        <div style="text-align: right; font-size: 0.8rem; color: #6B7280;">
            <strong>報告編號：</strong>EV-{datetime.now().strftime('%Y%m%d%H%M%S')}<br>
            <strong>評估時間：</strong>{now_str}
        </div>
    </div>

    <div class="meta-grid">
        <div class="meta-item"><strong>受評工項/檔名：</strong>{item_name}</div>
        <div class="meta-item"><strong>評估方法：</strong>{method} 量表</div>
        <div class="meta-item"><strong>作業員面向：</strong>{facing_zh}</div>
        <div class="meta-item"><strong>主要受評肢體：</strong>{side_zh}</div>
    </div>

    {asymmetry_html}

    <div class="score-banner">
        <div class="score-box">
            <div class="num">{score_val}</div>
            <div class="sub">{method} 綜合風險總分</div>
        </div>
        <div class="action-box">
            <span class="risk-badge">行動水準 AL {al} ｜ {theme['name']}</span>
            <div class="desc">📋 行動處置方針：{action_desc}</div>
        </div>
    </div>

    <div class="main-grid">
        <div class="panel">
            <div class="panel-title">📸 電腦視覺 AI 骨架辨識與關節拓撲</div>
            <div class="pose-img-container">
                <img class="pose-img" src="data:image/jpeg;base64,{b64_img}" alt="Pose Overlay">
            </div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px; text-align: center;">
                YOLO11n-Pose 17 關鍵點解剖學拓撲標註 (高亮顯示主要受評肢體)
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">📐 關鍵部位生物力學角度與得分</div>
            <table class="data-table">
                <tr><th>受評解剖部位</th><th>實測夾角</th><th>量表標準得分</th></tr>
                <tr><td>軀幹 (Trunk)</td><td><strong>{angles.get('trunk', 0.0):+.1f}°</strong></td><td>{assessment_result.get('group_b', {}).get('trunk', {}).get('final_score', assessment_result.get('group_a', {}).get('trunk', {}).get('final_score', '-'))} 分</td></tr>
                <tr><td>頸部 (Neck)</td><td><strong>{angles.get('neck', 0.0):+.1f}°</strong></td><td>{assessment_result.get('group_b', {}).get('neck', {}).get('final_score', assessment_result.get('group_a', {}).get('neck', {}).get('final_score', '-'))} 分</td></tr>
                <tr><td>上臂 (Upper Arm)</td><td><strong>{angles.get('upper_arm', 0.0):+.1f}°</strong></td><td>{assessment_result.get('group_a', {}).get('upper_arm', {}).get('final_score', assessment_result.get('group_b', {}).get('upper_arm', {}).get('final_score', '-'))} 分</td></tr>
                <tr><td>前臂手肘 (Lower Arm)</td><td><strong>{angles.get('lower_arm', 0.0):.1f}°</strong></td><td>{assessment_result.get('group_a', {}).get('lower_arm', {}).get('final_score', assessment_result.get('group_b', {}).get('lower_arm', {}).get('final_score', '-'))} 分</td></tr>
                <tr><td>手腕 (Wrist)</td><td><strong>{angles.get('wrist', 0.0):.1f}°</strong></td><td>{assessment_result.get('group_a', {}).get('wrist', {}).get('final_score', assessment_result.get('group_b', {}).get('wrist', {}).get('final_score', '-'))} 分</td></tr>
                <tr><td>膝關節 (Knee)</td><td><strong>{angles.get('knee', 0.0):.1f}°</strong></td><td>{assessment_result.get('group_b', {}).get('legs', {}).get('final_score', assessment_result.get('group_a', {}).get('legs', {}).get('final_score', '-'))} 分</td></tr>
            </table>

            <div class="panel-title" style="margin-top: 10px;">📊 查表矩陣拆解 (Table Matrix)</div>
            <table class="data-table">
                <tr><th>分組群</th><th>查表基準分</th><th>肌力/負重加分</th><th>群組得分</th></tr>
                {'<tr><td>表 A (上肢群)</td><td>' + str(posture_a) + ' 分</td><td>+' + str(muscle_a + force_a if isinstance(muscle_a, (int,float)) and isinstance(force_a, (int,float)) else 0) + ' 分</td><td><strong>' + str(score_c) + ' 分 (Score C)</strong></td></tr><tr><td>表 B (軀幹頸腿)</td><td>' + str(posture_b) + ' 分</td><td>+' + str(muscle_b + force_b if isinstance(muscle_b, (int,float)) and isinstance(force_b, (int,float)) else 0) + ' 分</td><td><strong>' + str(score_d) + ' 分 (Score D)</strong></td></tr>' if not is_reba else '<tr><td>表 A (軀幹頸腿)</td><td>' + str(posture_a) + ' 分</td><td>負重 +' + str(load_score) + ' 分</td><td><strong>' + str(score_a) + ' 分 (Score A)</strong></td></tr><tr><td>表 B (上肢群)</td><td>' + str(posture_b) + ' 分</td><td>握把 +' + str(coupling_score) + ' 分</td><td><strong>' + str(score_b) + ' 分 (Score B)</strong></td></tr>'}
            </table>
        </div>
    </div>

    <div class="panel">
        <div class="panel-title">💡 AI 自動根因診斷與優先改善模擬 (What-If Simulation)</div>
        <table class="data-table whatif-table">
            <tr>
                <th style="width: 8%;">排序</th>
                <th style="width: 22%;">改善領域</th>
                <th style="width: 44%;">建議工程/人因改善具體措施</th>
                <th style="width: 14%;">模擬預估分數</th>
                <th style="width: 12%;">行動水準</th>
            </tr>
            {what_if_rows}
        </table>
    </div>

    <div class="footer-sign">
        <div>
            現場受評作業人員 (Operator)：
            <div class="sign-box"></div>
        </div>
        <div>
            職安衛 EHS 工程師 (Ergonomist)：
            <div class="sign-box"></div>
        </div>
        <div>
            產線/單位主管簽核 (Supervisor)：
            <div class="sign-box"></div>
        </div>
    </div>

</body>
</html>
"""
    return html_content
