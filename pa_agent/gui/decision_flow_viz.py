"""Decision flow visualization — branched sci-fi flowchart (gate + strategy path)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsObject,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pa_agent.ai.decision_tree import (
    _BRANCH_DISPLAY_ZH,
    format_trace_answer,
    merge_traces,
    plain_trace_question,
)
from pa_agent.gui.theme import tokens as T

_OUTCOME_ZH = {
    "wait": "等待",
    "reject": "放弃",
    "trade": "交易",
    "proceed": "继续评估",
}
_OUTCOME_COLOR = {
    "wait": T.ACCENT_WARNING,
    "reject": T.ACCENT_DANGER,
    "trade": T.ACCENT_SUCCESS,
    "proceed": T.ACCENT_PRIMARY,
}
_PHASE_ZH = {"gate": "闸门", "decision": "策略"}
_ANSWER_COLOR = {
    "是": T.ACCENT_SUCCESS,
    "否": T.ACCENT_DANGER,
    "中性": T.ACCENT_WARNING,
    "等待": T.ACCENT_WARNING,
    "不适用": T.TEXT_MUTED,
}

# Larger nodes for readability in the sidebar
_NODE_W = 420
_NODE_H = 128
_STUB_W = 168
_STUB_H = 52
_TERMINAL_W = 460
_TERMINAL_H = 100
_LEVEL_DY = 200
_BRANCH_DX = 260
_MIN_ZOOM = 0.82


def _answer_color(answer: str) -> str:
    base = str(answer).split("（", 1)[0]
    return _ANSWER_COLOR.get(base, T.ACCENT_PRIMARY)


def _taken_branch_side(item: dict[str, Any]) -> Literal["left", "right", "down"]:
    """Which branch the AI took: left=否, right=是, down=跳过/直线."""
    if item.get("skipped"):
        return "down"
    ans = format_trace_answer(item) or str(item.get("answer", ""))
    base = ans.split("（", 1)[0]
    if base == "否":
        return "left"
    if base in ("是", "等待", "中性"):
        return "right"
    return "down"


@dataclass
class _Placed:
    x: float
    y: float
    kind: str
    item: dict[str, Any] | None = None
    step: int = 0
    active: bool = True


class _FlowScene(QGraphicsScene):
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, QColor("#05070b"))
        step = 32
        left = int(rect.left()) - (int(rect.left()) % step)
        top = int(rect.top()) - (int(rect.top()) % step)
        painter.setPen(QPen(QColor(28, 38, 52, 70)))
        x = left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += step
        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += step


class _BranchEdge(QGraphicsObject):
    """Bezier branch between parent and child (scene coordinates)."""

    def __init__(
        self,
        p0: QPointF,
        p1: QPointF,
        label: str,
        *,
        active: bool,
    ) -> None:
        super().__init__()
        self._p0 = p0
        self._p1 = p1
        self._label = label
        self._active = active
        self.setZValue(1)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(self._p0, self._p1).normalized().adjusted(-40, -24, 40, 24)

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = QColor(T.ACCENT_PRIMARY if self._active else T.TEXT_MUTED)
        if not self._active:
            col.setAlpha(90)
        glow = QPen(QColor(col.red(), col.green(), col.blue(), 50 if self._active else 25))
        glow.setWidth(7 if self._active else 4)
        path = self._curve()
        painter.setPen(glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        pen = QPen(col)
        pen.setWidth(3 if self._active else 1.5)
        if not self._active:
            pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawPath(path)
        # Arrow
        painter.setBrush(QBrush(col))
        dx = self._p1.x() - self._p0.x()
        dy = self._p1.y() - self._p0.y()
        angle = math.atan2(dy, dx)
        ah = 9
        tip = self._p1
        p2 = QPointF(
            tip.x() - ah * math.cos(angle - 0.45),
            tip.y() - ah * math.sin(angle - 0.45),
        )
        p3 = QPointF(
            tip.x() - ah * math.cos(angle + 0.45),
            tip.y() - ah * math.sin(angle + 0.45),
        )
        tri = QPainterPath(tip)
        tri.lineTo(p2)
        tri.lineTo(p3)
        tri.closeSubpath()
        painter.drawPath(tri)
        if self._label:
            mid = QPointF(
                (self._p0.x() + self._p1.x()) / 2,
                (self._p0.y() + self._p1.y()) / 2 - 6,
            )
            badge_w, badge_h = 56, 26
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(8, 12, 20, 240)))
            painter.drawRoundedRect(
                QRectF(
                    mid.x() - badge_w / 2,
                    mid.y() - badge_h / 2,
                    badge_w,
                    badge_h,
                ),
                8,
                8,
            )
            painter.setPen(QPen(col))
            font = QFont(T.FONT_UI.split(",")[0].strip('"'), 11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QRectF(mid.x() - badge_w / 2, mid.y() - badge_h / 2, badge_w, badge_h),
                int(Qt.AlignmentFlag.AlignCenter),
                self._label,
            )

    def _curve(self) -> QPainterPath:
        path = QPainterPath(self._p0)
        mid_y = (self._p0.y() + self._p1.y()) / 2
        c1 = QPointF(self._p0.x(), mid_y)
        c2 = QPointF(self._p1.x(), mid_y)
        path.cubicTo(c1, c2, self._p1)
        return path


class _PhaseBandItem(QGraphicsObject):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self.setZValue(2)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-280, -16, 560, 32)

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(T.ACCENT_REASONING))
        painter.setPen(pen)
        painter.drawLine(QPointF(-260, 0), QPointF(-70, 0))
        painter.drawLine(QPointF(70, 0), QPointF(260, 0))
        font = QFont(T.FONT_UI.split(",")[0].strip('"'), 11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(-70, -12, 140, 24), int(Qt.AlignmentFlag.AlignCenter), self._title)


class _DecisionNode(QGraphicsObject):
    """Large diamond — active decision on the walk path."""

    def __init__(self, item: dict[str, Any], step: int) -> None:
        super().__init__()
        self._item = item
        self._step = step
        self.setZValue(5)
        self.setAcceptHoverEvents(True)
        self._hover = False
        phase = str(item.get("phase", ""))
        self._phase_zh = _PHASE_ZH.get(phase, phase)
        self._nid = str(item.get("node_id", "?"))
        self._question = plain_trace_question(item)
        self._answer = format_trace_answer(item) or str(item.get("answer", "—"))
        self._branch = item.get("branch")
        self._skipped = bool(item.get("skipped"))
        tip = [self._question]
        if item.get("bar_range"):
            tip.append(f"K线：{item.get('bar_range')}")
        if item.get("reason"):
            tip.append(str(item.get("reason")))
        self.setToolTip("\n".join(tip))

    def boundingRect(self) -> QRectF:  # noqa: N802
        pad = 14
        return QRectF(-_NODE_W / 2 - pad, -pad, _NODE_W + pad * 2, _NODE_H + pad * 2)

    def hoverEnterEvent(self, _event: Any) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, _event: Any) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = _NODE_W, _NODE_H
        accent = QColor(_answer_color(self._answer))
        if self._skipped:
            accent = QColor(T.TEXT_MUTED)

        if self._hover:
            glow = QPen(QColor(accent.red(), accent.green(), accent.blue(), 120))
            glow.setWidth(12)
            painter.setPen(glow)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            _diamond(painter, 0, 0, w + 16, h + 16)

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor("#243552"))
        grad.setColorAt(0.5, QColor("#18243a"))
        grad.setColorAt(1, QColor("#0a0e16"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(accent, 2.5 if self._hover else 2))
        _diamond(painter, 0, 0, w, h)

        font_id = QFont("Consolas", 11)
        font_id.setBold(True)
        painter.setFont(font_id)
        painter.setPen(QPen(QColor(T.ACCENT_PRIMARY)))
        painter.drawText(
            QRectF(-w / 2 + 14, 12, w - 28, 22),
            int(Qt.AlignmentFlag.AlignLeft),
            f"#{self._step:02d}   [{self._phase_zh}]   §{self._nid}",
        )

        font_q = QFont(T.FONT_UI.split(",")[0].strip('"'), 11)
        painter.setFont(font_q)
        painter.setPen(QPen(QColor(T.TEXT_PRIMARY)))
        painter.drawText(
            QRectF(-w / 2 + 14, 36, w - 28, 52),
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            self._question,
        )

        ans = self._answer
        if self._branch:
            bzh = _BRANCH_DISPLAY_ZH.get(str(self._branch), str(self._branch))
            if bzh and bzh not in ans:
                ans = f"{ans}  ·  {bzh}"
        font_a = QFont(T.FONT_UI.split(",")[0].strip('"'), 13)
        font_a.setBold(True)
        painter.setFont(font_a)
        painter.setPen(QPen(accent))
        painter.drawText(
            QRectF(-w / 2 + 14, h - 36, w - 28, 28),
            int(Qt.AlignmentFlag.AlignLeft),
            f"▸ {ans}",
        )

    def port_bottom(self) -> QPointF:
        return self.scenePos() + QPointF(0, _NODE_H)

    def port_top(self) -> QPointF:
        return self.scenePos()

    def port_left(self) -> QPointF:
        return self.scenePos() + QPointF(-_NODE_W / 4, _NODE_H * 0.92)

    def port_right(self) -> QPointF:
        return self.scenePos() + QPointF(_NODE_W / 4, _NODE_H * 0.92)


class _StubNode(QGraphicsObject):
    """Dimmed leaf — branch not taken."""

    def __init__(self, label: str = "未走") -> None:
        super().__init__()
        self._label = label
        self.setZValue(3)
        self.setToolTip("该分支未进入，分析在另一侧继续")

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-_STUB_W / 2 - 4, -4, _STUB_W + 8, _STUB_H + 8)

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(-_STUB_W / 2, 0, _STUB_W, _STUB_H)
        painter.setBrush(QBrush(QColor(18, 22, 30, 200)))
        painter.setPen(QPen(QColor(T.TEXT_MUTED), 1.5, Qt.PenStyle.DotLine))
        painter.drawRoundedRect(rect, 10, 10)
        font = QFont(T.FONT_UI.split(",")[0].strip('"'), 11)
        painter.setFont(font)
        painter.setPen(QPen(QColor(T.TEXT_MUTED)))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), self._label)

    def port_top(self) -> QPointF:
        return self.scenePos()


class _TerminalNode(QGraphicsObject):
    def __init__(self, terminal: dict[str, Any]) -> None:
        super().__init__()
        self.setZValue(6)
        self._nid = str(terminal.get("node_id", "?"))
        self._outcome = str(terminal.get("outcome", ""))
        self._label = str(terminal.get("label", ""))
        self._outcome_zh = _OUTCOME_ZH.get(self._outcome, self._outcome)
        self._color = QColor(_OUTCOME_COLOR.get(self._outcome, T.ACCENT_PRIMARY))
        self.setToolTip(f"§{self._nid} · {self._outcome_zh}\n{self._label}")

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-_TERMINAL_W / 2 - 8, -8, _TERMINAL_W + 16, _TERMINAL_H + 16)

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = _TERMINAL_W, _TERMINAL_H
        rect = QRectF(-w / 2, 0, w, h)
        glow = QPen(QColor(self._color.red(), self._color.green(), self._color.blue(), 100))
        glow.setWidth(10)
        painter.setPen(glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(-5, -5, 5, 5), 16, 16)
        grad = QLinearGradient(-w / 2, 0, w / 2, 0)
        grad.setColorAt(0, QColor("#1a2844"))
        grad.setColorAt(0.5, QColor("#223358"))
        grad.setColorAt(1, QColor("#1a2844"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(self._color, 2.5))
        painter.drawRoundedRect(rect, 14, 14)
        font = QFont(T.FONT_UI.split(",")[0].strip('"'), 13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(self._color))
        painter.drawText(
            QRectF(-w / 2, 14, w, 28),
            int(Qt.AlignmentFlag.AlignCenter),
            f"◆  终点  §{self._nid}  ·  {self._outcome_zh}",
        )
        font.setBold(False)
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QPen(QColor(T.TEXT_PRIMARY)))
        painter.drawText(
            QRectF(-w / 2 + 16, 44, w - 32, 48),
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
            self._label,
        )

    def port_top(self) -> QPointF:
        return self.scenePos()


class _EmptyHint(QGraphicsObject):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-240, -40, 480, 80)

    def paint(self, painter: QPainter, _option: Any, _widget: Any = None) -> None:
        painter.setPen(QPen(QColor(T.TEXT_MUTED)))
        painter.setFont(QFont(T.FONT_UI.split(",")[0].strip('"'), 13))
        painter.drawText(self.boundingRect(), int(Qt.AlignmentFlag.AlignCenter), self._text)


def _diamond(painter: QPainter, cx: float, top: float, w: float, h: float) -> None:
    path = QPainterPath()
    path.moveTo(cx, top)
    path.lineTo(cx + w / 2, top + h / 2)
    path.lineTo(cx, top + h)
    path.lineTo(cx - w / 2, top + h / 2)
    path.closeSubpath()
    painter.drawPath(path)


def _layout_branched_path(
    merged: list[dict[str, Any]],
    terminal: dict[str, Any] | None,
) -> tuple[list[_Placed], list[tuple[QPointF, QPointF, str, bool]], list[_Placed]]:
    """Return nodes, edges (p0,p1,label,active), phase bands."""
    nodes: list[_Placed] = []
    edges: list[tuple[QPointF, QPointF, str, bool]] = []
    bands: list[_Placed] = []

    if not merged:
        return nodes, edges, bands

    x, y = 0.0, 40.0
    last_phase: str | None = None

    for i, item in enumerate(merged):
        phase = str(item.get("phase", ""))
        if last_phase == "gate" and phase == "decision":
            bands.append(_Placed(x=x, y=y - 28, kind="band"))
            y += 20
        last_phase = phase

        nodes.append(_Placed(x=x, y=y, kind="decision", item=item, step=i + 1, active=True))
        ny = y + _LEVEL_DY
        side = _taken_branch_side(item)
        port_l = QPointF(x - _NODE_W / 4, y + _NODE_H * 0.9)
        port_r = QPointF(x + _NODE_W / 4, y + _NODE_H * 0.9)
        port_m = QPointF(x, y + _NODE_H)

        if side == "down":
            child_x = x
            edges.append(
                (
                    port_m,
                    QPointF(child_x, ny),
                    "跳过" if item.get("skipped") else "→",
                    True,
                )
            )
            x, y = child_x, ny
            continue

        lx = x - _BRANCH_DX
        rx = x + _BRANCH_DX
        if side == "left":
            edges.append((port_l, QPointF(lx, ny), "否", True))
            edges.append((port_r, QPointF(rx, ny), "是", False))
            nodes.append(_Placed(x=rx, y=ny, kind="stub", active=False))
            x, y = lx, ny
        else:
            edges.append((port_l, QPointF(lx, ny), "否", False))
            edges.append((port_r, QPointF(rx, ny), "是", True))
            nodes.append(_Placed(x=lx, y=ny, kind="stub", active=False))
            x, y = rx, ny

    if terminal:
        ty = y + _LEVEL_DY - 20
        edges.append(
            (
                QPointF(x, y + _NODE_H),
                QPointF(x, ty),
                "→",
                True,
            )
        )
        nodes.append(_Placed(x=x, y=ty, kind="terminal", item=terminal, active=True))

    return nodes, edges, bands


class DecisionFlowVizPanel(QWidget):
    """Branched flowchart of the AI walk (yes=右 / no=左)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = _FlowScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self._view.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setBackgroundBrush(QBrush(QColor("#05070b")))

        title = QLabel("决策路径可视化")
        title.setObjectName("toolbarTitle")
        sub = QLabel(
            "菱形=判断节点 · 左=否 / 右=是 · 高亮=AI 实际路径 · 虚线框=未走分支"
            " · 拖拽平移 · Ctrl+滚轮缩放"
        )
        sub.setObjectName("mutedLabel")
        sub.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addWidget(self._view, stretch=1)
        self.clear()

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = math.pow(1.1, event.angleDelta().y() / 120.0)
            self._view.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _fit_scene(self, rect: QRectF) -> None:
        self._view.resetTransform()
        self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        if self._view.transform().m11() < _MIN_ZOOM:
            self._view.resetTransform()
            self._view.scale(_MIN_ZOOM, _MIN_ZOOM)
        self._view.centerOn(rect.center())

    def clear(self) -> None:
        self._scene.clear()
        hint = _EmptyHint("等待分析…\n提交后将显示左右分支决策流程图")
        hint.setPos(-200, 60)
        self._scene.addItem(hint)
        rect = QRectF(-400, 0, 800, 200)
        self._scene.setSceneRect(rect)
        self._fit_scene(rect)

    def set_trace(
        self,
        *,
        gate_trace: list[dict[str, Any]] | None = None,
        decision_trace: list[dict[str, Any]] | None = None,
        terminal: dict[str, Any] | None = None,
        gate_result: str | None = None,
        gate_shortcircuited: bool = False,
    ) -> None:
        merged = merge_traces(gate_trace, decision_trace)
        self._scene.clear()
        if not merged and not terminal:
            self.clear()
            return

        placed, edge_specs, bands = _layout_branched_path(merged, terminal)

        if gate_shortcircuited and merged:
            last = placed[-1] if placed else None
            if last and last.kind == "terminal":
                bands.append(_Placed(x=last.x, y=last.y - 50, kind="band_short"))
            elif placed:
                p = placed[-1]
                bands.append(_Placed(x=p.x, y=p.y + _NODE_H + 30, kind="band_short"))

        for b in bands:
            title = "阶段二 · 策略评估"
            if b.kind == "band_short":
                title = "阶段二已短路"
            band = _PhaseBandItem(title)
            band.setPos(b.x, b.y)
            self._scene.addItem(band)

        for p0, p1, label, active in edge_specs:
            self._scene.addItem(_BranchEdge(p0, p1, label, active=active))

        for p in placed:
            if p.kind == "decision" and p.item:
                node = _DecisionNode(p.item, p.step)
                node.setPos(p.x - _NODE_W / 2, p.y)
                self._scene.addItem(node)
            elif p.kind == "stub":
                stub = _StubNode("未走")
                stub.setPos(p.x - _STUB_W / 2, p.y)
                self._scene.addItem(stub)
            elif p.kind == "terminal" and p.item:
                term = _TerminalNode(p.item)
                term.setPos(p.x - _TERMINAL_W / 2, p.y)
                self._scene.addItem(term)

        xs = [p.x for p in placed]
        ys = [p.y for p in placed]
        max_y = max(ys) + _TERMINAL_H + 80 if ys else 400
        min_x = min(xs) - _NODE_W if xs else -400
        max_x = max(xs) + _NODE_W if xs else 400
        rect = QRectF(min_x - 80, 0, max_x - min_x + 160, max_y)
        self._scene.setSceneRect(rect)
        self._fit_scene(rect)
