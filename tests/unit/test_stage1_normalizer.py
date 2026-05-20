"""Tests for Stage 1 JSON normalization."""
from __future__ import annotations

from pa_agent.ai.json_validator import JsonValidator
from pa_agent.ai.stage1_normalizer import normalize_stage1
from tests.integration.conftest import VALID_STAGE1


def test_maps_recommended_strategy_files() -> None:
    raw = {**VALID_STAGE1}
    del raw["strategy_files_needed"]
    raw["recommended_strategy_files"] = ["下跌通道分析识别.txt"]
    out = normalize_stage1(raw)
    assert out["strategy_files_needed"] == ["下跌通道分析识别.txt"]


def test_normalizes_gate_2_3_directional_answer() -> None:
    raw = {**VALID_STAGE1}
    raw["gate_trace"] = [
        {
            "node_id": "2.3",
            "question": "当前方向是多头还是空头？",
            "answer": "空头",
            "reason": "EMA下倾",
            "bar_range": "K20-K1",
        }
    ]
    out = normalize_stage1(raw)
    assert out["gate_trace"][0]["answer"] == "是"
    assert out["gate_trace"][0]["branch"] == "bearish"


def test_validator_accepts_normalized_user_payload() -> None:
    """Regression: payload like user's failing response after normalize."""
    import json

    payload = {
        "cycle_position": "micro_channel",
        "direction": "bearish",
        "diagnosis_confidence": 82,
        "spike_stage": None,
        "market_phase": "transitioning",
        "transition_risk": "medium",
        "detected_patterns": [],
        "key_signals": ["sig"],
        "htf_context": "htf",
        "entry_setup": "等待",
        "recommended_strategy_files": ["下跌通道策略"],
        "risk_warning": "warn",
        "gate_trace": [
            {
                "node_id": "2.3",
                "question": "当前方向是多头还是空头？",
                "answer": "空头",
                "reason": "bear",
                "bar_range": "K20-K1",
            }
        ],
        "gate_result": "proceed",
    }
    normalized = normalize_stage1(payload)
    normalized["gate_trace"] = VALID_STAGE1["gate_trace"]
    normalized["strategy_files_needed"] = ["下跌通道分析识别.txt"]
    result = JsonValidator().validate("stage1", json.dumps(normalized, ensure_ascii=False))
    from pa_agent.ai.json_validator import Ok

    assert isinstance(result, Ok)
