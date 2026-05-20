"""Tests for forming-bar close detection."""
from __future__ import annotations

from pa_agent.data.bar_close_wait import (
    current_forming_ts,
    forming_bar_has_closed,
    seconds_until_bar_closes,
    timeframe_to_seconds,
)
from pa_agent.data.base import KlineBar


def _bar(ts: int) -> KlineBar:
    return KlineBar(
        seq=1,
        ts_open=float(ts),
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
        closed=False,
    )


def test_timeframe_to_seconds() -> None:
    assert timeframe_to_seconds("5m") == 300
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("2h") == 7200


def test_seconds_until_bar_closes() -> None:
    ts_open = 1_000_000
    now = ts_open + 240_000  # 4 min into 5m bar
    assert seconds_until_bar_closes(ts_open, "5m", now_ms=now) == 60
    assert seconds_until_bar_closes(ts_open, "5m", now_ms=ts_open + 300_000) == 0


def test_forming_bar_has_closed_when_ts_changes() -> None:
    waited = 1000
    before = [_bar(1000), _bar(900)]
    after = [_bar(2000), _bar(1000)]
    assert current_forming_ts(before) == 1000
    assert not forming_bar_has_closed(waited, before)
    assert forming_bar_has_closed(waited, after)
