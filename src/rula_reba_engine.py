#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rula_reba_engine.py
-------------------
人因工程評估核心邏輯庫 (RULA & REBA Rule Engine)

特色：
1. 嚴格解耦：純邏輯運算，不依賴任何電腦視覺套件。
2. 雙參數輸入介面 (Function Signature)：
   - auto_angles: 由視覺模型估計之連續角度 (頸部、軀幹、上臂、前臂、手腕、膝等)
   - manual_modifiers: 人工/作業環境補正參數 (負重、支撐、活動性質、抓握品質等)
3. 包含完整的 Table A, Table B, Table C 查表矩陣與行動水準 (Action Level) 判定。
4. 支援矢狀面正負方向性 (屈曲 Flexion vs 伸展 Extension)。

標準文獻依據：
- RULA: McAtamney & Corlett (1993)
- REBA: Hignett & McAtamney (2000)
"""

from typing import Dict, Any, Tuple
import numpy as np


# ==============================================================================
# RULA 查表矩陣 (Standard RULA Matrices)
# ==============================================================================

# Table A: Upper Arm (1-6) x Lower Arm (1-3) x Wrist (1-4) x Wrist Twist (1-2) -> Posture Score A (1-8)
# 維度: [upper_arm - 1][lower_arm - 1][wrist - 1][wrist_twist - 1]
RULA_TABLE_A = np.array([
    # Upper Arm 1
    [
        [[1, 2], [2, 2], [2, 3], [3, 3]],  # Lower Arm 1
        [[2, 2], [2, 2], [3, 3], [3, 3]],  # Lower Arm 2
        [[2, 3], [3, 3], [3, 3], [4, 4]],  # Lower Arm 3
    ],
    # Upper Arm 2
    [
        [[2, 3], [3, 3], [3, 4], [4, 4]],  # Lower Arm 1
        [[3, 3], [3, 3], [3, 4], [4, 4]],  # Lower Arm 2
        [[3, 4], [4, 4], [4, 4], [5, 5]],  # Lower Arm 3
    ],
    # Upper Arm 3
    [
        [[3, 3], [4, 4], [4, 4], [5, 5]],  # Lower Arm 1
        [[3, 4], [4, 4], [4, 4], [5, 5]],  # Lower Arm 2
        [[4, 4], [4, 4], [4, 5], [5, 5]],  # Lower Arm 3
    ],
    # Upper Arm 4
    [
        [[4, 4], [4, 5], [5, 5], [6, 6]],  # Lower Arm 1
        [[4, 4], [4, 5], [5, 5], [6, 6]],  # Lower Arm 2
        [[4, 4], [4, 5], [5, 5], [6, 6]],  # Lower Arm 3
    ],
    # Upper Arm 5
    [
        [[5, 5], [5, 6], [6, 7], [7, 7]],  # Lower Arm 1
        [[5, 6], [6, 7], [7, 7], [7, 8]],  # Lower Arm 2
        [[6, 6], [7, 7], [7, 8], [8, 8]],  # Lower Arm 3
    ],
    # Upper Arm 6
    [
        [[7, 7], [7, 8], [8, 8], [8, 9]],  # Lower Arm 1
        [[8, 8], [8, 8], [8, 9], [9, 9]],  # Lower Arm 2
        [[9, 9], [9, 9], [9, 9], [9, 9]],  # Lower Arm 3
    ],
], dtype=int)

# Table B: Neck (1-6) x Trunk (1-6) x Legs (1-2) -> Posture Score B (1-7)
# 維度: [neck - 1][trunk - 1][legs - 1]
RULA_TABLE_B = np.array([
    # Neck 1
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    # Neck 2
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    # Neck 3
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    # Neck 4
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    # Neck 5
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    # Neck 6
    [[8, 8], [8, 8], [8, 8], [8, 8], [8, 8], [8, 8]],
], dtype=int)

# Table C: Wrist & Arm Score (Score C, 1-8+) x Neck, Trunk & Leg Score (Score D, 1-7+) -> Grand Score (1-7)
# 維度: [score_c - 1][score_d - 1]
RULA_TABLE_C = np.array([
    [1, 2, 3, 3, 4, 5, 5],  # Score C = 1
    [2, 2, 3, 4, 4, 5, 5],  # Score C = 2
    [3, 3, 3, 4, 4, 5, 6],  # Score C = 3
    [3, 3, 3, 4, 5, 6, 6],  # Score C = 4
    [4, 4, 4, 5, 6, 7, 7],  # Score C = 5
    [4, 4, 5, 6, 6, 7, 7],  # Score C = 6
    [5, 5, 6, 6, 7, 7, 8],  # Score C = 7
    [5, 5, 6, 7, 7, 8, 8],  # Score C = 8+
], dtype=int)


# ==============================================================================
# REBA 查表矩陣 (Standard REBA Matrices)
# ==============================================================================

# Table A: Trunk (1-5) x Neck (1-3) x Legs (1-4) -> Posture Score A (1-9)
# 維度: [trunk - 1][neck - 1][legs - 1]
REBA_TABLE_A = np.array([
    # Trunk 1
    [
        [1, 2, 3, 4],  # Neck 1
        [1, 2, 3, 4],  # Neck 2
        [3, 3, 5, 6],  # Neck 3
    ],
    # Trunk 2
    [
        [2, 3, 4, 5],  # Neck 1
        [3, 4, 5, 6],  # Neck 2
        [4, 5, 6, 7],  # Neck 3
    ],
    # Trunk 3
    [
        [2, 4, 5, 6],  # Neck 1
        [4, 5, 6, 7],  # Neck 2
        [5, 6, 7, 8],  # Neck 3
    ],
    # Trunk 4
    [
        [3, 5, 6, 7],  # Neck 1
        [5, 6, 7, 8],  # Neck 2
        [6, 7, 8, 9],  # Neck 3
    ],
    # Trunk 5
    [
        [4, 6, 7, 8],  # Neck 1
        [6, 7, 8, 9],  # Neck 2
        [7, 8, 9, 9],  # Neck 3
    ],
], dtype=int)

# Table B: Upper Arm (1-6) x Lower Arm (1-2) x Wrist (1-3) -> Posture Score B (1-9)
# 維度: [upper_arm - 1][lower_arm - 1][wrist - 1]
REBA_TABLE_B = np.array([
    # Upper Arm 1
    [[1, 2, 2], [1, 2, 3]],
    # Upper Arm 2
    [[1, 2, 3], [2, 3, 4]],
    # Upper Arm 3
    [[3, 4, 5], [4, 5, 5]],
    # Upper Arm 4
    [[4, 5, 5], [5, 6, 7]],
    # Upper Arm 5
    [[6, 7, 8], [7, 8, 8]],
    # Upper Arm 6
    [[7, 8, 8], [8, 9, 9]],
], dtype=int)

# Table C: Score A (1-12) x Score B (1-12) -> Score C (1-12)
# 維度: [score_a - 1][score_b - 1]
REBA_TABLE_C = np.array([
    # Score A = 1
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    # Score A = 2
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    # Score A = 3
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    # Score A = 4
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    # Score A = 5
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    # Score A = 6
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    # Score A = 7
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    # Score A = 8
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    # Score A = 9
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    # Score A = 10
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    # Score A = 11
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    # Score A = 12
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
], dtype=int)


# ==============================================================================
# 角度離散化與部位評分函式 (Angle Discretization Helpers)
# ==============================================================================

def score_upper_arm(angle_deg: float,
                    shoulder_raised: bool = False,
                    arm_abducted: bool = False,
                    arm_supported: bool = False,
                    is_reba: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    評估上臂分數 (Upper Arm Score)
    正角為屈曲 (Flexion)，負角為伸展 (Extension)
    """
    # 基準分
    if -20.0 <= angle_deg <= 20.0:
        base = 1
    elif angle_deg < -20.0 or (20.0 < angle_deg <= 45.0):
        base = 2
    elif 45.0 < angle_deg <= 90.0:
        base = 3
    else:  # > 90.0
        base = 4

    adj = 0
    if shoulder_raised:
        adj += 1
    if arm_abducted:
        adj += 1
    if arm_supported:
        adj -= 1

    final = max(1, min(6, base + adj))
    details = {
        "angle": angle_deg,
        "base_score": base,
        "shoulder_raised": shoulder_raised,
        "arm_abducted": arm_abducted,
        "arm_supported": arm_supported,
        "final_score": final
    }
    return final, details


def score_lower_arm(angle_deg: float,
                    across_midline: bool = False,
                    out_to_side: bool = False,
                    is_reba: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    評估前臂分數 (Lower Arm Score)
    正常範圍為 60° ~ 100°
    """
    if 60.0 <= angle_deg <= 100.0:
        base = 1
    else:
        base = 2

    adj = 0
    if not is_reba:
        if across_midline:
            adj += 1
        if out_to_side:
            adj += 1

    max_score = 2 if is_reba else 3
    final = max(1, min(max_score, base + adj))
    details = {
        "angle": angle_deg,
        "base_score": base,
        "across_midline": across_midline,
        "out_to_side": out_to_side,
        "final_score": final
    }
    return final, details


def score_wrist(angle_deg: float,
                deviated: bool = False,
                twisted: bool = False,
                is_reba: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    評估手腕分數 (Wrist Score)
    angle_deg: 屈曲或伸展角度 (取絕對值)
    """
    abs_angle = abs(angle_deg)
    if is_reba:
        base = 1 if abs_angle <= 15.0 else 2
        adj = 1 if (deviated or twisted) else 0
        final = max(1, min(3, base + adj))
    else:
        if abs_angle < 1e-3:
            base = 1
        elif abs_angle <= 15.0:
            base = 2
        else:
            base = 3
        adj = 1 if deviated else 0
        final = max(1, min(4, base + adj))

    details = {
        "angle": abs_angle,
        "base_score": base,
        "deviated": deviated,
        "twisted": twisted,
        "final_score": final
    }
    return final, details


def score_neck(angle_deg: float,
               twisted: bool = False,
               side_bending: bool = False,
               is_reba: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    評估頸部分數 (Neck Score)
    正角為前屈 (Flexion)，負角為後仰 (Extension)
    """
    if is_reba:
        # REBA: 0-20°屈曲為 1；>20°屈曲或後仰為 2
        base = 1 if (0.0 <= angle_deg <= 20.0) else 2
    else:
        # RULA: 0-10°為 1; 10-20°為 2; >20°為 3; 後仰為 4
        if angle_deg < 0.0:
            base = 4
        elif angle_deg <= 10.0:
            base = 1
        elif angle_deg <= 20.0:
            base = 2
        else:
            base = 3

    adj = 0
    if twisted:
        adj += 1
    if side_bending:
        adj += 1

    max_score = 3 if is_reba else 6
    final = max(1, min(max_score, base + adj))
    details = {
        "angle": angle_deg,
        "base_score": base,
        "twisted": twisted,
        "side_bending": side_bending,
        "final_score": final
    }
    return final, details


def score_trunk(angle_deg: float,
                twisted: bool = False,
                side_bending: bool = False,
                sitting_supported: bool = False,
                is_reba: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    評估軀幹分數 (Trunk Score)
    正角為前傾 (Flexion)，負角為後仰 (Extension)
    """
    abs_angle = abs(angle_deg)
    if is_reba:
        if abs_angle < 1e-3:
            base = 1
        elif 0.0 < abs_angle <= 20.0:
            base = 2
        elif 20.0 < abs_angle <= 60.0 or (angle_deg < -20.0):
            base = 3
        else:  # > 60°
            base = 4
    else:
        if sitting_supported and abs_angle <= 5.0:
            base = 1
        elif 0.0 <= abs_angle <= 20.0:
            base = 2
        elif 20.0 < abs_angle <= 60.0:
            base = 3
        else:  # > 60°
            base = 4

    adj = 0
    if twisted:
        adj += 1
    if side_bending:
        adj += 1

    max_score = 5 if is_reba else 6
    final = max(1, min(max_score, base + adj))
    details = {
        "angle": angle_deg,
        "base_score": base,
        "twisted": twisted,
        "side_bending": side_bending,
        "final_score": final
    }
    return final, details


def score_legs(supported: bool = True,
               knee_angle_deg: float = 0.0,
               is_sitting: bool = False,
               is_reba: bool = False) -> Tuple[int, Dict[str, Any]]:
    """
    評估腿部分數 (Legs Score)
    """
    if is_reba:
        # REBA: 雙腿承重/坐姿/行走 = 1；單腿承重/站姿不穩 = 2
        base = 1 if supported else 2
        adj = 0
        if not is_sitting:
            if 30.0 <= knee_angle_deg <= 60.0:
                adj += 1
            elif knee_angle_deg > 60.0:
                adj += 2
        final = max(1, min(4, base + adj))
    else:
        # RULA: 雙腳支撐均勻 = 1; 未支撐/不均勻 = 2
        base = 1 if supported else 2
        adj = 0
        final = max(1, min(2, base))

    details = {
        "supported": supported,
        "knee_angle": knee_angle_deg,
        "is_sitting": is_sitting,
        "final_score": final
    }
    return final, details


# ==============================================================================
# RULA 計算核心 (RULA Calculator)
# ==============================================================================

def calculate_rula(auto_angles: Dict[str, float],
                   manual_modifiers: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    計算 RULA 分數 (Rapid Upper Limb Assessment)

    :param auto_angles: 視覺估計角度字典
        - upper_arm (deg): 屈伸角度 (正前屈，負後伸)
        - lower_arm (deg): 屈曲角度
        - wrist (deg): 屈伸角度
        - neck (deg): 頸部角度 (正前屈，負後仰)
        - trunk (deg): 軀幹角度 (正前傾，負後仰)
    :param manual_modifiers: 人工環境/作業補正參數
        - shoulder_raised (bool): 肩膀是否聳起
        - arm_abducted (bool): 手臂是否外展
        - arm_supported (bool): 手臂是否有支撐
        - across_midline (bool): 前臂是否越過身體中線
        - out_to_side (bool): 前臂是否向外張開
        - wrist_deviated (bool): 手腕是否橈偏/尺偏
        - wrist_twist (int): 手腕旋轉 (1: 中線範圍, 2: 極限範圍)
        - neck_twisted (bool): 頸部扭轉
        - neck_side_bending (bool): 頸部側彎
        - trunk_twisted (bool): 軀幹扭轉
        - trunk_side_bending (bool): 軀幹側彎
        - legs_supported (bool): 雙腳是否有支撐且平衡 (預設 True)
        - muscle_use_a (int): 上肢肌肉重複/靜態負荷 (0: 否, 1: 是)
        - force_load_a (int): 上肢負荷/力量 (0: <2kg, 1: 2-10kg間歇, 2: 2-10kg靜態, 3: >10kg)
        - muscle_use_b (int): 頸軀肌肉重複/靜態負荷 (0: 否, 1: 是)
        - force_load_b (int): 頸軀負荷/力量 (0: <2kg, 1: 2-10kg間歇, 2: 2-10kg靜態, 3: >10kg)
    :return: 包含完整 breakdown 與 Grand Score 的字典
    """
    m = manual_modifiers or {}

    # 1. Group A 各關節評分
    ua_score, ua_det = score_upper_arm(
        angle_deg=auto_angles.get("upper_arm", 0.0),
        shoulder_raised=m.get("shoulder_raised", False),
        arm_abducted=m.get("arm_abducted", False),
        arm_supported=m.get("arm_supported", False),
        is_reba=False
    )
    la_score, la_det = score_lower_arm(
        angle_deg=auto_angles.get("lower_arm", 90.0),
        across_midline=m.get("across_midline", False),
        out_to_side=m.get("out_to_side", False),
        is_reba=False
    )
    wr_score, wr_det = score_wrist(
        angle_deg=auto_angles.get("wrist", 0.0),
        deviated=m.get("wrist_deviated", False),
        is_reba=False
    )
    wrist_twist = int(m.get("wrist_twist", 1))
    wrist_twist = max(1, min(2, wrist_twist))

    # 查 Table A
    posture_score_a = int(RULA_TABLE_A[ua_score - 1, la_score - 1, wr_score - 1, wrist_twist - 1])

    # 疊加 Group A 負荷
    muscle_a = int(m.get("muscle_use_a", 0))
    force_a = int(m.get("force_load_a", 0))
    score_c = posture_score_a + muscle_a + force_a

    # 2. Group B 各關節評分
    neck_score, neck_det = score_neck(
        angle_deg=auto_angles.get("neck", 0.0),
        twisted=m.get("neck_twisted", False),
        side_bending=m.get("neck_side_bending", False),
        is_reba=False
    )
    trunk_score, trunk_det = score_trunk(
        angle_deg=auto_angles.get("trunk", 0.0),
        twisted=m.get("trunk_twisted", False),
        side_bending=m.get("trunk_side_bending", False),
        sitting_supported=m.get("sitting_supported", False),
        is_reba=False
    )
    leg_score, leg_det = score_legs(
        supported=m.get("legs_supported", True),
        is_reba=False
    )

    # 查 Table B
    posture_score_b = int(RULA_TABLE_B[neck_score - 1, trunk_score - 1, leg_score - 1])

    # 疊加 Group B 負荷
    muscle_b = int(m.get("muscle_use_b", 0))
    force_b = int(m.get("force_load_b", 0))
    score_d = posture_score_b + muscle_b + force_b

    # 3. 查 Table C (Grand Score)
    c_idx = min(max(score_c, 1), 8) - 1
    d_idx = min(max(score_d, 1), 7) - 1
    raw_grand = int(RULA_TABLE_C[c_idx, d_idx])
    # 依標準，RULA 最終評分為 1~7 (>=7 均為 7分)
    grand_score = min(7, raw_grand)

    # 行動水準判定
    if grand_score in (1, 2):
        action_level = 1
        risk_level = "可接受 (Acceptable)"
        action_desc = "姿勢良好，暫無危害；若長時間維持或高頻重複需適當輪替。"
    elif grand_score in (3, 4):
        action_level = 2
        risk_level = "低度風險 (Further investigation needed)"
        action_desc = "需要進一步調查，可能需要改善作業姿勢或設備。"
    elif grand_score in (5, 6):
        action_level = 3
        risk_level = "中度風險 (Investigation and changes required soon)"
        action_desc = "應於近期內展開詳細調查並實施工程或人因改善。"
    else:  # >= 7
        action_level = 4
        risk_level = "高度風險 (Investigation and changes required immediately)"
        action_desc = "危害顯著，必須立即停止不良姿勢作業並立刻進行改善！"

    return {
        "assessment": "RULA",
        "grand_score": grand_score,
        "action_level": action_level,
        "risk_level": risk_level,
        "action_description": action_desc,
        "group_a": {
            "upper_arm": ua_det,
            "lower_arm": la_det,
            "wrist": wr_det,
            "wrist_twist": wrist_twist,
            "posture_score_a": posture_score_a,
            "muscle_use": muscle_a,
            "force_load": force_a,
            "score_c": score_c,
        },
        "group_b": {
            "neck": neck_det,
            "trunk": trunk_det,
            "legs": leg_det,
            "posture_score_b": posture_score_b,
            "muscle_use": muscle_b,
            "force_load": force_b,
            "score_d": score_d,
        }
    }


# ==============================================================================
# REBA 計算核心 (REBA Calculator)
# ==============================================================================

def calculate_reba(auto_angles: Dict[str, float],
                   manual_modifiers: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    計算 REBA 分數 (Rapid Entire Body Assessment)

    :param auto_angles: 視覺估計角度字典
        - trunk (deg): 軀幹前屈/後仰 (正屈曲，負後仰)
        - neck (deg): 頸部角度 (正屈曲，負後仰)
        - knee (deg): 膝關節彎曲角度 (0~180°)
        - upper_arm (deg): 上臂屈伸 (正前屈，負後伸)
        - lower_arm (deg): 前臂屈曲
        - wrist (deg): 手腕角度
    :param manual_modifiers: 人工環境/作業補正參數
        - trunk_twisted (bool): 軀幹扭轉
        - trunk_side_bending (bool): 軀幹側彎
        - neck_twisted (bool): 頸部扭轉
        - neck_side_bending (bool): 頸部側彎
        - legs_supported (bool): 是否雙腿承重/平衡
        - is_sitting (bool): 是否為坐姿作業
        - shoulder_raised (bool): 肩膀是否聳起
        - arm_abducted (bool): 上臂是否外展或旋轉
        - arm_supported (bool): 上臂是否有支撐
        - wrist_deviated (bool): 手腕是否側向偏移或扭轉
        - load_weight_kg (float): 負重重量 (kg)
        - load_rapid_shock (bool): 是否有瞬間衝擊或急劇增強之施力
        - coupling_score (int): 握把耦合 (0:良好, 1:尚可, 2:差但可抓, 3:無握把/危險)
        - activity_static (bool): 姿勢靜止維持 > 1 分鐘 (+1)
        - activity_repeated (bool): 每分鐘重複動作 > 4 次 (+1)
        - activity_rapid (bool): 大範圍姿勢快速變化或基底不穩 (+1)
    :return: 包含完整 breakdown 與 REBA Score 的字典
    """
    m = manual_modifiers or {}

    # 1. Group A (Trunk, Neck, Legs)
    trunk_score, trunk_det = score_trunk(
        angle_deg=auto_angles.get("trunk", 0.0),
        twisted=m.get("trunk_twisted", False),
        side_bending=m.get("trunk_side_bending", False),
        is_reba=True
    )
    neck_score, neck_det = score_neck(
        angle_deg=auto_angles.get("neck", 0.0),
        twisted=m.get("neck_twisted", False),
        side_bending=m.get("neck_side_bending", False),
        is_reba=True
    )
    leg_score, leg_det = score_legs(
        supported=m.get("legs_supported", True),
        knee_angle_deg=auto_angles.get("knee", 0.0),
        is_sitting=m.get("is_sitting", False),
        is_reba=True
    )

    # 查 Table A
    posture_score_a = int(REBA_TABLE_A[trunk_score - 1, neck_score - 1, leg_score - 1])

    # 負荷分數 (Load/Force Score)
    weight = float(m.get("load_weight_kg", 0.0))
    if weight < 5.0:
        load_score = 0
    elif 5.0 <= weight <= 10.0:
        load_score = 1
    else:  # > 10.0 kg
        load_score = 2

    if m.get("load_rapid_shock", False):
        load_score += 1

    score_a = posture_score_a + load_score

    # 2. Group B (Upper Arm, Lower Arm, Wrist)
    ua_score, ua_det = score_upper_arm(
        angle_deg=auto_angles.get("upper_arm", 0.0),
        shoulder_raised=m.get("shoulder_raised", False),
        arm_abducted=m.get("arm_abducted", False),
        arm_supported=m.get("arm_supported", False),
        is_reba=True
    )
    la_score, la_det = score_lower_arm(
        angle_deg=auto_angles.get("lower_arm", 90.0),
        is_reba=True
    )
    wr_score, wr_det = score_wrist(
        angle_deg=auto_angles.get("wrist", 0.0),
        deviated=m.get("wrist_deviated", False),
        twisted=m.get("wrist_twisted", False),
        is_reba=True
    )

    # 查 Table B
    posture_score_b = int(REBA_TABLE_B[ua_score - 1, la_score - 1, wr_score - 1])

    # 抓握分數 (Coupling Score)
    coupling = int(m.get("coupling_score", 0))
    coupling = max(0, min(3, coupling))
    score_b = posture_score_b + coupling

    # 3. 查 Table C (Score C)
    a_idx = min(max(score_a, 1), 12) - 1
    b_idx = min(max(score_b, 1), 12) - 1
    score_c = int(REBA_TABLE_C[a_idx, b_idx])

    # 4. 活動分數 (Activity Score)
    activity = 0
    if m.get("activity_static", False):
        activity += 1
    if m.get("activity_repeated", False):
        activity += 1
    if m.get("activity_rapid", False):
        activity += 1

    final_reba_score = min(15, score_c + activity)

    # 行動水準判定
    if final_reba_score == 1:
        action_level = 0
        risk_level = "可忽略風險 (Negligible risk)"
        action_desc = "無須採取行動。"
    elif 2 <= final_reba_score <= 3:
        action_level = 1
        risk_level = "低度風險 (Low risk)"
        action_desc = "可能需要改善作業姿勢。"
    elif 4 <= final_reba_score <= 7:
        action_level = 2
        risk_level = "中度風險 (Medium risk)"
        action_desc = "需要進一步調查並儘速改善。"
    elif 8 <= final_reba_score <= 10:
        action_level = 3
        risk_level = "高度風險 (High risk)"
        action_desc = "需要進行調查並立刻採取改善措施。"
    else:  # 11 ~ 15
        action_level = 4
        risk_level = "極高風險 (Very high risk)"
        action_desc = "危害極大，必須立即停止作業並強制改善！"

    return {
        "assessment": "REBA",
        "final_reba_score": final_reba_score,
        "action_level": action_level,
        "risk_level": risk_level,
        "action_description": action_desc,
        "group_a": {
            "trunk": trunk_det,
            "neck": neck_det,
            "legs": leg_det,
            "posture_score_a": posture_score_a,
            "load_score": load_score,
            "score_a": score_a,
        },
        "group_b": {
            "upper_arm": ua_det,
            "lower_arm": la_det,
            "wrist": wr_det,
            "posture_score_b": posture_score_b,
            "coupling_score": coupling,
            "score_b": score_b,
        },
        "score_c": score_c,
        "activity_score": activity
    }


# ==============================================================================
# 簡單 CLI 自我測試範例
# ==============================================================================
if __name__ == "__main__":
    print("=== Testing RULA Calculator ===")
    test_rula = calculate_rula(
        auto_angles={"upper_arm": 30.0, "lower_arm": 90.0, "wrist": 10.0, "neck": 15.0, "trunk": 10.0},
        manual_modifiers={"force_load_a": 0, "muscle_use_a": 0}
    )
    print(f"RULA Grand Score: {test_rula['grand_score']}, Level: {test_rula['action_level']}")

    print("\n=== Testing REBA Calculator ===")
    test_reba = calculate_reba(
        auto_angles={"trunk": 35.0, "neck": 25.0, "knee": 40.0, "upper_arm": 60.0, "lower_arm": 110.0, "wrist": 20.0},
        manual_modifiers={"load_weight_kg": 6.0, "coupling_score": 1, "activity_repeated": True}
    )
    print(f"REBA Final Score: {test_reba['final_reba_score']}, Level: {test_reba['action_level']}")
