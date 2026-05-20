"""Unit tests for PromptAssembler (task 7.3)."""
from __future__ import annotations

import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pa_agent.ai.prompt_assembler import PromptAssembler
from pa_agent.data.base import KlineBar, KlineFrame, IndicatorBundle


def _make_frame(n: int = 5) -> KlineFrame:
    bars = tuple(
        KlineBar(
            seq=i + 1,
            ts_open=float(1_700_000_000 - i * 3600),
            open=2600.0 + i,
            high=2610.0 + i,
            low=2590.0 + i,
            close=2605.0 + i,
            volume=1000.0,
            closed=(i != 0),
        )
        for i in range(n)
    )
    indicators = IndicatorBundle(
        ema20=tuple(2600.0 + i for i in range(n)),
        atr14=tuple(5.0 for _ in range(n)),
    )
    return KlineFrame(
        symbol="XAUUSD",
        timeframe="1h",
        bars=bars,
        indicators=indicators,
        snapshot_ts_local_ms=1_700_000_000_000,
    )


@pytest.fixture()
def assembler(tmp_path: Path) -> PromptAssembler:
    """PromptAssembler with fake prompt files."""
    for fname in [
        "提示词大纲_人设与思维方式.txt",
        "市场诊断框架.txt",
        "二元决策.txt",
        "文件16-K线信号识别.txt",
        "文件17-止损和止盈与仓位管理.txt",
        "上涨通道分析识别.txt",
        "上涨通道交易策略.txt",
        "文件13-窄通道与宽通道策略.txt",
    ]:
        (tmp_path / fname).write_text(f"[CONTENT OF {fname}]", encoding="utf-8")
    return PromptAssembler(prompt_dir=tmp_path)


def test_stage1_system_prompt_order(assembler: PromptAssembler):
    """Stage 1 system prompt must contain the 4 always-on files in order."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    system = messages[0]["content"]
    pos_persona = system.find("提示词大纲_人设与思维方式")
    pos_diag = system.find("市场诊断框架")
    pos_binary = system.find("二元决策")
    pos_signal = system.find("文件16-K线信号识别")
    assert pos_persona < pos_diag < pos_binary < pos_signal, (
        "Stage 1 system prompt files are out of order"
    )


def test_stage1_user_prompt_contains_required_fields(assembler: PromptAssembler):
    """Stage 1 user prompt must contain symbol, timeframe, bar count."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    user = messages[1]["content"]
    assert "XAUUSD" in user
    assert "1h" in user
    assert "序号" in user
    assert "更高时间框架" not in user


def test_stage2_user_prompt_includes_gate_trace(assembler: PromptAssembler):
    frame = _make_frame()
    stage1_json = {
        "cycle_position": "normal_channel",
        "direction": "bullish",
        "gate_result": "proceed",
        "gate_trace": [{"node_id": "0.1", "question": "q", "answer": "是", "reason": "r"}],
    }
    messages = assembler.build_stage2(frame, stage1_json, [], [])
    user = messages[1]["content"]
    assert "gate_result=proceed" in user
    assert "gate_trace" in user or "0.1" in user


def test_stage2_system_prompt_order(assembler: PromptAssembler):
    """Stage 2 system prompt: 人设 → 二元决策 → 策略 → 风控 → 契约."""
    frame = _make_frame()
    stage1_json = {"cycle_position": "normal_channel", "direction": "bullish"}
    strategy_files = ["上涨通道分析识别.txt", "上涨通道交易策略.txt", "文件13-窄通道与宽通道策略.txt"]
    messages = assembler.build_stage2(frame, stage1_json, strategy_files, [])
    system = messages[0]["content"]

    pos_persona = system.find("提示词大纲_人设与思维方式")
    pos_binary = system.find("二元决策")
    pos_strategy = system.find("上涨通道分析识别")
    pos_risk = system.find("文件17-止损和止盈与仓位管理")
    assert pos_persona < pos_binary < pos_strategy < pos_risk, (
        "Stage 2 system prompt files are out of order"
    )


def test_stage2_user_prompt_contains_stage1_json(assembler: PromptAssembler):
    """Stage 2 user prompt must embed the Stage 1 JSON."""
    frame = _make_frame()
    stage1_json = {"cycle_position": "spike", "direction": "bearish"}
    messages = assembler.build_stage2(frame, stage1_json, [], [])
    user = messages[1]["content"]
    assert "spike" in user
    assert "bearish" in user


def test_stage1_output_reminder_present(assembler: PromptAssembler):
    """Stage 1 system prompt must contain the output format reminder."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    system = messages[0]["content"]
    assert "cycle_position" in system
    assert "diagnosis_confidence" in system
    assert "gate_trace" in system
    assert "gate_result" in system


def test_stage2_output_contract_present(assembler: PromptAssembler):
    """Stage 2 system prompt must contain the output contract with null rule."""
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [])
    system = messages[0]["content"]
    assert "不下单" in system
    assert "order_direction" in system
    assert "decision_trace" in system
    assert "terminal" in system


def test_stage2_experience_entries_included(assembler: PromptAssembler):
    """Stage 2 system prompt must include experience entries when provided."""
    frame = _make_frame()
    entries = [{"cycle_position": "spike", "outcome": "success"}]
    messages = assembler.build_stage2(frame, {}, [], entries)
    system = messages[0]["content"]
    assert "经验库" in system
    assert "案例 1" in system


def test_stage2_system_prompt_only_matches_build_stage2(assembler: PromptAssembler):
    """stage2_system_prompt_only must return the same system content as build_stage2."""
    frame = _make_frame()
    strategy_files = ["上涨通道分析识别.txt"]
    entries = [{"note": "test"}]
    messages = assembler.build_stage2(frame, {}, strategy_files, entries)
    system_from_build = messages[0]["content"]
    system_only = assembler.stage2_system_prompt_only(strategy_files, entries)
    assert system_from_build == system_only


def test_kline_table_contains_nan_as_na(assembler: PromptAssembler):
    """K-line table renders NaN indicator values as 'N/A'."""
    bars = (
        KlineBar(seq=1, ts_open=1_700_000_000.0, open=2600.0, high=2610.0,
                 low=2590.0, close=2605.0, volume=1000.0, closed=False),
    )
    indicators = IndicatorBundle(
        ema20=(float("nan"),),
        atr14=(float("nan"),),
    )
    frame = KlineFrame(
        symbol="XAUUSD", timeframe="1h", bars=bars,
        indicators=indicators, snapshot_ts_local_ms=1_700_000_000_000,
    )
    messages = assembler.build_stage1(frame)
    user = messages[1]["content"]
    assert "N/A" in user


def test_stage1_message_roles(assembler: PromptAssembler):
    """build_stage1 must return exactly [system, user] messages."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_stage2_message_roles(assembler: PromptAssembler):
    """build_stage2 must return exactly [system, user] messages."""
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
