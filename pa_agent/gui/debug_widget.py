"""DebugWidget — Tab 3 debug panel.

Displays all AI turns in the current session with full prompt/response detail,
copy buttons, JSON export, API-key masking, and a manual streak-reset button.

Design reference: design.md §B.11 (Tab3)
Tasks: 16.1, 16.2, 16.3, 16.4
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal

from pa_agent.security.secret_store import mask_secret
from pa_agent.config.paths import RECORDS_PENDING_DIR

if TYPE_CHECKING:
    from pa_agent.orchestrator.exception_counter import ExceptionCounter

logger = logging.getLogger(__name__)


class DebugWidget(QWidget):
    """Tab 3 debug panel.

    streak_reset:
        Emitted after the user clears the consecutive validation-error counter.

    Left side: QListWidget listing all turns (Stage1, Stage2, Followup-N …).
    Right side: 4 read-only QTextEdit blocks:
        1. System prompt
        2. User prompt
        3. Raw response (HTTP status, headers, body, reasoning_content, content,
           usage, request_id)
        4. Validation / retry / exception classification

    Turn data model (dict):
        label           : str   — e.g. "Stage1", "Stage2", "Followup-1"
        system_prompt   : str
        user_prompt     : str
        raw_response    : dict  — AIReply.raw dict
        validation_info : str   — validation result or exception info

    Parameters
    ----------
    api_key:
        Plaintext API key to mask in all displayed text.  Defaults to "".
    exc_counter:
        Optional ExceptionCounter instance.  When provided the
        "清除连续异常计数" button calls ``exc_counter.reset_streak()``.
    parent:
        Optional parent widget.
    """

    streak_reset = pyqtSignal()

    def __init__(
        self,
        api_key: str = "",
        exc_counter: Optional["ExceptionCounter"] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._exc_counter = exc_counter
        self._turns: list[dict] = []
        self._setup_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ── Main splitter: list (left) | detail (right) ───────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: turn list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(QLabel("会话轮次"))
        self._list_widget = QListWidget()
        self._list_widget.currentRowChanged.connect(self._on_turn_selected)
        left_layout.addWidget(self._list_widget)
        splitter.addWidget(left_widget)

        # Right: 4 text areas + button row
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._system_edit = self._make_text_edit("System Prompt")
        self._user_edit = self._make_text_edit("User Prompt")
        self._response_edit = self._make_text_edit("Raw Response")
        self._validation_edit = self._make_text_edit("Validation / Exception")

        for label, edit in (
            ("System Prompt", self._system_edit),
            ("User Prompt", self._user_edit),
            ("Raw Response", self._response_edit),
            ("Validation / Exception", self._validation_edit),
        ):
            right_layout.addWidget(QLabel(f"<b>{label}</b>"))
            right_layout.addWidget(edit)

        # Button row
        right_layout.addLayout(self._build_button_row())

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        outer.addWidget(splitter)

    def _make_text_edit(self, placeholder: str = "") -> QTextEdit:
        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText(placeholder)
        return edit

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        btn_copy_system = QPushButton("复制 system")
        btn_copy_system.clicked.connect(self._copy_system)
        row.addWidget(btn_copy_system)

        btn_copy_user = QPushButton("复制 user")
        btn_copy_user.clicked.connect(self._copy_user)
        row.addWidget(btn_copy_user)

        btn_copy_response = QPushButton("复制 response")
        btn_copy_response.clicked.connect(self._copy_response)
        row.addWidget(btn_copy_response)

        btn_export = QPushButton("导出本轮 JSON")
        btn_export.clicked.connect(self._export_turn_json)
        row.addWidget(btn_export)

        btn_reset_streak = QPushButton("清除连续异常计数")
        btn_reset_streak.clicked.connect(self._reset_streak)
        row.addWidget(btn_reset_streak)

        row.addStretch()
        return row

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(self, turn_data: dict) -> None:
        """Append a turn to the list.

        Parameters
        ----------
        turn_data:
            Dict with keys: label, system_prompt, user_prompt,
            raw_response, validation_info.
        """
        self._turns.append(turn_data)
        label = turn_data.get("label", f"Turn-{len(self._turns)}")
        item = QListWidgetItem(label)
        self._list_widget.addItem(item)
        # Auto-select the newly added turn
        self._list_widget.setCurrentRow(len(self._turns) - 1)

    def clear(self) -> None:
        """Clear all turns from the widget."""
        self._turns.clear()
        self._list_widget.clear()
        self._system_edit.clear()
        self._user_edit.clear()
        self._response_edit.clear()
        self._validation_edit.clear()

    # ── Turn selection ────────────────────────────────────────────────────────

    def _on_turn_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._turns):
            self._system_edit.clear()
            self._user_edit.clear()
            self._response_edit.clear()
            self._validation_edit.clear()
            return

        turn = self._turns[row]
        self._system_edit.setPlainText(self._mask(turn.get("system_prompt", "")))
        self._user_edit.setPlainText(self._mask(turn.get("user_prompt", "")))
        self._response_edit.setPlainText(
            self._mask(self._format_raw_response(turn.get("raw_response", {})))
        )
        self._validation_edit.setPlainText(
            self._mask(turn.get("validation_info", ""))
        )

    def _current_row(self) -> int:
        return self._list_widget.currentRow()

    # ── Masking ───────────────────────────────────────────────────────────────

    def _mask(self, text: str) -> str:
        """Replace all occurrences of the plaintext API key with its masked form."""
        if not self._api_key or not text:
            return text
        masked = mask_secret(self._api_key)
        return text.replace(self._api_key, masked)

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_raw_response(raw: dict) -> str:
        """Render the raw response dict as a human-readable string."""
        if not raw:
            return ""
        try:
            return json.dumps(raw, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(raw)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _copy_system(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._system_edit.toPlainText())

    def _copy_user(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._user_edit.toPlainText())

    def _copy_response(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._response_edit.toPlainText())

    def _export_turn_json(self) -> None:
        """Write the current turn's data to records/pending/<label>.debug-<row>.json."""
        row = self._current_row()
        if row < 0 or row >= len(self._turns):
            QMessageBox.information(self, "导出", "没有选中的轮次。")
            return

        turn = self._turns[row]
        label = turn.get("label", f"turn-{row}")
        # Sanitise label for use in filename
        safe_label = label.replace(" ", "_").replace("/", "-")
        filename = f"{safe_label}.debug-{row}.json"

        try:
            RECORDS_PENDING_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RECORDS_PENDING_DIR / filename
            out_path.write_text(
                json.dumps(turn, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Debug turn exported to %s", out_path)
            QMessageBox.information(self, "导出成功", f"已写入：\n{out_path}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to export debug turn: %s", exc)
            QMessageBox.critical(self, "导出失败", str(exc))

    def _reset_streak(self) -> None:
        """Show confirmation dialog and reset the exception streak on confirm."""
        reply = QMessageBox.question(
            self,
            "清除连续异常计数",
            "确定要将连续异常计数重置为 0 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self._exc_counter is not None:
            self._exc_counter.reset_streak()
            self.streak_reset.emit()
            logger.info("DebugWidget: exception streak manually cleared via UI")
            QMessageBox.information(self, "已清除", "连续异常计数已重置为 0。")
        else:
            logger.warning("DebugWidget: reset_streak called but no exc_counter set")
            QMessageBox.warning(self, "未配置", "未绑定异常计数器，无法重置。")
