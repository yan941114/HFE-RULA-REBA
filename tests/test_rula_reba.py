#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_rula_reba.py
-----------------
單元測試集：驗證 RULA 與 REBA 查表與評分邏輯之精確度
"""

import sys
import os

# 加入 src 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rula_reba_engine import (
    calculate_rula,
    calculate_reba,
    score_upper_arm,
    score_lower_arm,
    score_wrist,
    score_neck,
    score_trunk,
    score_legs,
)


def test_angle_discretization():
    print("[TEST] 驗證各部位角度離散化與方向性...")

    # 1. 上臂屈伸 (正前屈，負後伸)
    score, _ = score_upper_arm(15.0)  # 20°以內
    assert score == 1, f"Expected 1, got {score}"
    score, _ = score_upper_arm(30.0)  # 20°~45°
    assert score == 2, f"Expected 2, got {score}"
    score, _ = score_upper_arm(-25.0) # 後伸 >20°
    assert score == 2, f"Expected 2, got {score}"
    score, _ = score_upper_arm(60.0)  # 45°~90°
    assert score == 3, f"Expected 3, got {score}"
    score, _ = score_upper_arm(100.0) # >90°
    assert score == 4, f"Expected 4, got {score}"

    # 補正項：聳肩與外展
    score, _ = score_upper_arm(30.0, shoulder_raised=True, arm_abducted=True)
    assert score == 4, f"Expected 2+1+1=4, got {score}"

    # 2. 前臂
    score, _ = score_lower_arm(80.0)
    assert score == 1, f"Expected 1, got {score}"
    score, _ = score_lower_arm(110.0)
    assert score == 2, f"Expected 2, got {score}"

    # 3. 頸部 (RULA)
    score, _ = score_neck(5.0, is_reba=False)
    assert score == 1, f"Expected 1, got {score}"
    score, _ = score_neck(15.0, is_reba=False)
    assert score == 2, f"Expected 2, got {score}"
    score, _ = score_neck(-10.0, is_reba=False)  # 後仰
    assert score == 4, f"Expected 4, got {score}"

    # 4. 軀幹 (REBA)
    score, _ = score_trunk(0.0, is_reba=True)
    assert score == 1, f"Expected 1, got {score}"
    score, _ = score_trunk(30.0, twisted=True, is_reba=True)  # 20-60° (3分) + 扭轉 (1分)
    assert score == 4, f"Expected 4, got {score}"

    print("  -> 關節離散化與補正測試通過！")


def test_rula_benchmarks():
    print("[TEST] 驗證 RULA 標準基準情境...")

    # 情境 1: 完美人體工學坐姿 (低風險 AL 1)
    res_ideal = calculate_rula(
        auto_angles={
            "upper_arm": 10.0,
            "lower_arm": 90.0,
            "wrist": 0.0,
            "neck": 5.0,
            "trunk": 0.0
        },
        manual_modifiers={
            "sitting_supported": True,
            "legs_supported": True,
            "muscle_use_a": 0,
            "force_load_a": 0,
            "muscle_use_b": 0,
            "force_load_b": 0
        }
    )
    assert res_ideal["grand_score"] == 1, f"Expected RULA Grand Score 1, got {res_ideal['grand_score']}"
    assert res_ideal["action_level"] == 1

    # 情境 2: 課堂講義 P.47 高危害不良姿勢 (高風險 AL 4, Score = 7)
    res_hazard = calculate_rula(
        auto_angles={
            "upper_arm": 80.0,  # 45-90° (3分)
            "lower_arm": 115.0, # >100° (2分)
            "wrist": 25.0,      # >15° (3分)
            "neck": 35.0,       # >20° (3分)
            "trunk": 45.0       # 20-60° (3分)
        },
        manual_modifiers={
            "shoulder_raised": True, # 上臂 +1 => 4
            "arm_abducted": True,   # 上臂 +1 => 5
            "wrist_deviated": True, # 手腕 +1 => 4
            "wrist_twist": 2,       # 手腕扭轉 => 2
            "neck_twisted": True,   # 頸部 +1 => 4
            "trunk_twisted": True,  # 軀幹 +1 => 4
            "legs_supported": False, # 雙腳不均 => 2
            "muscle_use_a": 1,      # 靜態姿勢 > 1 min
            "force_load_a": 2,      # 5.2kg 重複
            "muscle_use_b": 1,
            "force_load_b": 2
        }
    )
    assert res_hazard["grand_score"] == 7, f"Expected RULA Grand Score 7, got {res_hazard['grand_score']}"
    assert res_hazard["action_level"] == 4

    print("  -> RULA 基準測試全部通過！")


def test_reba_benchmarks():
    print("[TEST] 驗證 REBA 標準基準情境...")

    # 情境 1: 完美直立站姿、無負重 (可忽略風險 AL 0)
    res_ideal = calculate_reba(
        auto_angles={
            "trunk": 0.0,
            "neck": 5.0,
            "knee": 0.0,
            "upper_arm": 0.0,
            "lower_arm": 90.0,
            "wrist": 0.0
        },
        manual_modifiers={
            "load_weight_kg": 0.0,
            "coupling_score": 0,
            "legs_supported": True
        }
    )
    assert res_ideal["final_reba_score"] == 1, f"Expected REBA Score 1, got {res_ideal['final_reba_score']}"
    assert res_ideal["action_level"] == 0

    # 情境 2: 課堂講義 P.64 負重搬運重度風險案例 (Score = 9, AL 3)
    res_lifting = calculate_reba(
        auto_angles={
            "trunk": 35.0,      # 20-60° (base 3)
            "neck": 25.0,       # >20° (base 2)
            "knee": 40.0,       # 30-60° (leg adj +1)
            "upper_arm": 50.0,  # 45-90° (base 3)
            "lower_arm": 110.0, # >100° (base 2)
            "wrist": 20.0       # >15° (base 2)
        },
        manual_modifiers={
            "trunk_side_bending": True, # trunk +1 => 4
            "neck_twisted": True,       # neck +1 => 3
            "legs_supported": True,     # base 1 + knee 1 => 2
            # Group A Table A[4-1, 3-1, 2-1] = 7
            "load_weight_kg": 12.0,     # >10kg => +2 -> Score A = 9
            "arm_abducted": True,       # upper arm +1 => 4
            "wrist_twisted": True,      # wrist +1 => 3
            # Group B Table B[4-1, 2-1, 3-1] = 7
            "coupling_score": 1,        # fair (+1) -> Score B = 8
            # Table C[9-1, 8-1] = 10 (or 9-10)
            "activity_static": False,
            "activity_repeated": False,
            "activity_rapid": False
        }
    )
    # Check that score reaches High Risk Level 3 (8-10)
    assert res_lifting["action_level"] == 3 or res_lifting["action_level"] == 4, \
        f"Expected High Risk Level, got {res_lifting['action_level']} (Score: {res_lifting['final_reba_score']})"

    print("  -> REBA 基準測試全部通過！")


if __name__ == "__main__":
    print("=" * 50)
    print("開始執行 RULA / REBA 核心引擎單元測試...")
    print("=" * 50)
    test_angle_discretization()
    test_rula_benchmarks()
    test_reba_benchmarks()
    print("=" * 50)
    print("ALL TESTS PASSED! 所有單元測試順利通過！")
    print("=" * 50)
