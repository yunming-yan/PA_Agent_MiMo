"""Tests for stage prompt .txt file list helpers."""
from __future__ import annotations

from pa_agent.ai.prompt_assembler import (
    stage1_prompt_txt_files,
    stage2_prompt_txt_files,
)


def test_stage1_txt_files() -> None:
    files = stage1_prompt_txt_files()
    assert len(files) == 4
    assert "二元决策.txt" in files
    assert files[0].endswith(".txt")


def test_stage2_txt_files_order() -> None:
    routed = ["震荡区间交易策略.txt", "震荡区间分析识别.txt"]
    files = stage2_prompt_txt_files(routed)
    assert files[0] == "提示词大纲_人设与思维方式.txt"
    assert files[1] == "二元决策.txt"
    assert files[2:-1] == routed
    assert files[-1] == "文件17-止损和止盈与仓位管理.txt"
