"""
RoboSpeed Durability Intelligence Platform  v2.4
PyQt6 + PyQtGraph  |  Dark Industrial Theme

Changes in v2.4 (PDF update 3):
  - HOME / RESET / RECORD TRAJECTORY / DOWNLOAD REPORT / EXIT → filled grey #7A7D82 (hover #95989E)
  - EXIT → slightly darker grey #5A5D62 for visual separation
  - All button icons use identical font-size (10pt) for equal visual weight
  - Pause icon changed to ‖ (U+2016, double vertical line) — same cap-height as ■ stop square
  - Logo background: PIL pixel-replacement maps near-black logo pixels → exact panel colour
    so no dark halo / rectangular border visible
  - Save button: emoji icon removed, text-only
  - Apply Settings button: filled solid blue (user request)
"""

import sys, os, time, math, random, threading
from collections import deque
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame, QSizePolicy, QToolButton,
    QScrollArea, QCheckBox, QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QFont, QPixmap, QPainter, QBrush, QLinearGradient, QIcon
import pyqtgraph as pg

# ═══════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════
C = dict(
    BG          = "#1C1C1E",
    PANEL       = "#2A2A2D",
    PANEL_DARK  = "#232326",
    PANEL_CARD  = "#1E1E21",
    BORDER      = "#3A3A3D",
    HEADER_BG   = "#1A1A1D",
    TEXT        = "#E5E5EA",
    TEXT_SUB    = "#8E8E93",
    TEXT_MED    = "#C0C0C5",
    ACCENT      = "#0A84FF",
    ACCENT_LT   = "#3D9FFF",
    GREEN       = "#30D158",
    GREEN_DK    = "#22C55E",
    AMBER       = "#FF9F0A",
    RED         = "#FF453A",
    BLUE_DK     = "#2563EB",
    SLATE_DK    = "#6366F1",
    TEAL_DK     = "#06B6D4",
    PURPLE_DK   = "#7C3AED",
    GRAPH_BG    = "#141416",
    FORCE_LINE  = "#0A84FF",
    FORCE_BAND  = "#0A84FF",
    DOT_RUN     = "#30D158",
    DOT_PAUSE   = "#FF9F0A",
    DOT_STOP    = "#48484A",
    DOT_ERR     = "#FF453A",
    TAB_BG      = "#2A2A2D",
    TAB_HOVER   = "#3A3A3D",
    # PDF-specified button colours
    BTN_START   = "#1F8F4E",
    BTN_START_H = "#27A55B",
    BTN_PAUSE   = "#C27A1A",
    BTN_PAUSE_H = "#D48B23",
    BTN_STOP    = "#B4232C",
    BTN_STOP_H  = "#D92D36",
    # PDF update 3 — grey fill for HOME/RESET/RECORD/DOWNLOAD/EXIT
    BTN_GREY    = "#7A7D82",
    BTN_GREY_H  = "#95989E",
)

PANEL_W = 260

DEFAULTS = dict(
    vel=300, acc=300, jerk=1000,
    target_cycles=100, baseline_cycles=30,
    force_min=0.5, force_max=1.8,
    surface_capture_every=25,
    led_capture_every=25,
    point_cloud_capture_every=50,
)
LIMITS = dict(
    vel=(0,1000), acc=(0,2000), jerk=(0,10000),
    target_cycles=(1,99999), baseline_cycles=(1,500),
    force_min=(0.0,100.0), force_max=(0.0,100.0),
    surface_capture_every=(0,500),
    led_capture_every=(0,500),
    point_cloud_capture_every=(0,500),
)

# ═══════════════════════════════════════════════════════════════════
# LOGO + ICON ASSET SEARCH
# ═══════════════════════════════════════════════════════════════════
def _find_logo():
    """
    Returns (logo_path, icon_path).
    logo_path → full horizontal logo (robot + ROBOSPEED text) for the left panel.
    icon_path → square icon-only crop for the window titlebar / taskbar.
    Priority: pre-cropped outputs dir first, then uploads, then generate on the fly.
    """
    try:    here = os.path.dirname(os.path.abspath(__file__))
    except: here = os.getcwd()

    outputs  = "/mnt/user-data/outputs"
    uploads  = "/mnt/user-data/uploads"
    search   = [here, os.path.join(here,"assets"), "/home/claude", outputs, uploads]

    # ── locate the horizontal logo ───────────────────────────────
    logo_candidates = [
        "logo_cropped.png",            # pre-cropped tight version (best)
        "robospeed_logo.png",          # original upload – black bg, white text ✓
        "robospeed_logo_white.png",
        "RoboSpeed_logo_white.png",
        "robospeed_logo_black.png",
        "robospeed_Logo_C2_black_resized.png",
    ]
    logo_path = None
    for name in logo_candidates:
        for folder in search:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                logo_path = p
                break
        if logo_path:
            break

    # ── locate the square icon (favicon) ────────────────────────
    icon_candidates = [
        "favicon_64.png",
        "favicon_32.png",
        "favicon.ico",
    ]
    icon_path = None
    for name in icon_candidates:
        for folder in [outputs, here, os.path.join(here,"assets")]:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                icon_path = p
                break
        if icon_path:
            break

    # If we still have no icon, extract it from the logo on the fly
    if not icon_path and logo_path and logo_path.endswith(".png"):
        try:
            from PIL import Image
            import numpy as np
            im   = Image.open(logo_path)
            arr  = np.array(im)
            # the robot icon sits in the left ~420 px of the 2000-wide image
            icon_px = arr[:, :420, :]
            mask    = icon_px.max(axis=2) > 10
            rows    = np.where(mask.any(axis=1))[0]
            cols    = np.where(mask.any(axis=0))[0]
            if len(rows) and len(cols):
                pad  = 8
                crop = icon_px[
                    max(0, rows[0]-pad):rows[-1]+pad,
                    max(0, cols[0]-pad):cols[-1]+pad,
                ]
                h, w = crop.shape[:2]
                sz   = max(h, w)
                sq   = np.zeros((sz, sz, 3), dtype=np.uint8)
                sq[(sz-h)//2:(sz-h)//2+h, (sz-w)//2:(sz-w)//2+w] = crop
                tmp  = os.path.join(here, "_rs_icon_tmp.png")
                Image.fromarray(sq).resize((64,64), Image.LANCZOS).save(tmp)
                icon_path = tmp
        except Exception:
            pass

    return logo_path, icon_path

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def mkfont(size=10, bold=False):
    f = QFont("Segoe UI", size); f.setBold(bold); return f

def clamp(v, lo, hi): return max(lo, min(hi, v))

def _btn_filled(bg, hover, border=None):
    bdr = border or hover
    return f"""
        QPushButton{{
            background:{bg};color:#FFFFFF;border:1px solid {bdr};
            border-radius:5px;padding:5px 8px;font-weight:bold;font-size:10pt;}}
        QPushButton:hover{{background:{hover};border-color:{hover};}}
        QPushButton:pressed{{background:{bg};border-color:{bdr};}}
        QPushButton:disabled{{background:#2E2E31;color:#555558;border-color:#3A3A3D;}}
    """

def _btn_ghost(border, hover_bg, text_color=None):
    tc = text_color or border
    return f"""
        QPushButton{{
            background:transparent;color:{tc};border:1px solid {border};
            border-radius:5px;padding:5px 8px;font-weight:bold;font-size:10pt;}}
        QPushButton:hover{{background:{hover_bg};color:#FFFFFF;border-color:{hover_bg};}}
        QPushButton:pressed{{background:{border};color:#FFFFFF;border-color:{border};}}
        QPushButton:disabled{{background:transparent;color:#555558;border-color:#3A3A3D;}}
    """

def _btn_accent_ghost():
    return _btn_ghost(C["ACCENT"], C["ACCENT"], C["TEXT_MED"])

def _chk_css():
    return f"""
        QCheckBox{{color:{C['TEXT']};spacing:5px;background:transparent;font-size:9pt;}}
        QCheckBox::indicator{{width:14px;height:14px;border-radius:3px;
            border:1.5px solid {C['BORDER']};background:{C['PANEL_DARK']};}}
        QCheckBox::indicator:checked{{background:{C['ACCENT']};border:1.5px solid {C['ACCENT']};}}
    """

def _radio_css():
    return f"""
        QRadioButton{{color:{C['TEXT']};spacing:5px;background:transparent;font-size:9pt;}}
        QRadioButton::indicator{{width:13px;height:13px;border-radius:7px;
            border:1.5px solid {C['BORDER']};background:{C['PANEL_DARK']};}}
        QRadioButton::indicator:checked{{background:{C['ACCENT']};border:1.5px solid {C['ACCENT']};}}
    """

# ═══════════════════════════════════════════════════════════════════
# BASE WIDGETS
# ═══════════════════════════════════════════════════════════════════
class IEdit(QLineEdit):
    def __init__(self, text="", placeholder="", align_right=True, parent=None):
        super().__init__(text, parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(26); self.setFont(mkfont(10))
        align = Qt.AlignmentFlag.AlignRight if align_right else Qt.AlignmentFlag.AlignLeft
        self.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        self.setStyleSheet(f"""
            QLineEdit{{background:{C['PANEL_DARK']};color:{C['TEXT']};
                border:1px solid {C['BORDER']};border-radius:4px;padding:2px 6px;
                selection-background-color:{C['ACCENT']};}}
            QLineEdit:focus{{border:1.5px solid {C['ACCENT']};}}
        """)

class ILabel(QLabel):
    def __init__(self, text="", size=10, bold=False, color=None, parent=None):
        super().__init__(text, parent)
        self.setFont(mkfont(size, bold))
        self.setStyleSheet(f"color:{color or C['TEXT']};background:transparent;")
        self.setWordWrap(False)

class HSep(QFrame):
    def __init__(self, color=None):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background:{color or C['BORDER']};border:none;")

class VSep(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFixedWidth(1)
        self.setStyleSheet(f"background:{C['BORDER']};border:none;")

class IChk(QCheckBox):
    def __init__(self, text, checked=True, bold=False, parent=None):
        super().__init__(text, parent)
        self.setChecked(checked); self.setFont(mkfont(9, bold=bold))
        self.setStyleSheet(_chk_css())

class IRadio(QRadioButton):
    def __init__(self, text, checked=False, parent=None):
        super().__init__(text, parent)
        self.setChecked(checked); self.setFont(mkfont(9))
        self.setStyleSheet(_radio_css())

# ═══════════════════════════════════════════════════════════════════
# COLLAPSIBLE SECTION
# ═══════════════════════════════════════════════════════════════════
class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self.setStyleSheet("background:transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        v = QVBoxLayout(self); v.setContentsMargins(0,0,0,4); v.setSpacing(2)

        self._hdr = QToolButton()
        self._hdr.setText(f"  ▼  {title}")
        self._hdr.setCheckable(True); self._hdr.setChecked(True)
        self._hdr.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._hdr.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hdr.setMinimumHeight(26); self._hdr.setFont(mkfont(8, bold=True))
        self._hdr.setStyleSheet(f"""
            QToolButton{{background:{C['HEADER_BG']};color:{C['TEXT_SUB']};
                border:none;border-bottom:1px solid {C['BORDER']};border-radius:0;
                text-align:left;padding-left:6px;letter-spacing:0.8px;}}
            QToolButton:hover{{color:{C['TEXT']};background:#1F1F22;}}
        """)
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hdr.clicked.connect(self._toggle)
        v.addWidget(self._hdr)

        self._body = QWidget()
        self._body.setAutoFillBackground(True)
        p = self._body.palette()
        p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL"]))
        self._body.setPalette(p)
        self._bl = QVBoxLayout(self._body)
        self._bl.setContentsMargins(6,6,6,6); self._bl.setSpacing(5)
        v.addWidget(self._body)

    def _toggle(self, checked):
        self._body.setVisible(checked)
        self._hdr.setText(f"  {'▼' if checked else '▶'}  {self._title}")

    def add(self, w): self._bl.addWidget(w)
    def add_layout(self, l): self._bl.addLayout(l)

# ═══════════════════════════════════════════════════════════════════
# FIELD ROW
# ═══════════════════════════════════════════════════════════════════
class FieldRow(QWidget):
    def __init__(self, label, default, obj_name, lbl_w=130, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        h = QHBoxLayout(self); h.setContentsMargins(0,0,0,0); h.setSpacing(4)
        lbl = ILabel(label, size=9, color=C["TEXT_SUB"])
        lbl.setMinimumWidth(lbl_w); lbl.setWordWrap(True)
        self.edit = IEdit(str(default))
        self.edit.setObjectName(obj_name); self.edit.setFixedWidth(60)
        h.addWidget(lbl, stretch=1); h.addWidget(self.edit)

# ═══════════════════════════════════════════════════════════════════
# STATUS DOT
# ═══════════════════════════════════════════════════════════════════
class StatusDot(QWidget):
    def __init__(self, d=16, parent=None):
        super().__init__(parent)
        self._c = QColor(C["DOT_STOP"]); self._d = d
        self.setFixedSize(d+6, d+6); self.setStyleSheet("background:transparent;")

    def set_color(self, hx): self._c = QColor(hx); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width()//2, self.height()//2; r = self._d//2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._c.darker(200)))
        p.drawEllipse(cx-r-2, cy-r-2, 2*r+4, 2*r+4)
        p.setBrush(QBrush(self._c))
        p.drawEllipse(cx-r+1, cy-r+1, 2*r-2, 2*r-2)
        hi = QColor(self._c); hi.setAlpha(130); hi = hi.lighter(170)
        p.setBrush(QBrush(hi))
        p.drawEllipse(cx-r//2+1, cy-r+3, r//2+2, r//3+2)
        p.end()

# ═══════════════════════════════════════════════════════════════════
# LOGO WIDGET  –  HiDPI-crisp, bg-matched to panel (no dark halo)
# ═══════════════════════════════════════════════════════════════════
class LogoWidget(QWidget):
    """
    Loads the RoboSpeed logo and replaces its near-black background pixels
    with the exact panel colour before rendering, so there is zero visible
    background rectangle — the robot icon and text float on the panel.
    """
    def __init__(self, logo_path, target_w=224, parent=None):
        super().__init__(parent)
        self._pix = None
        panel_hex = C["PANEL"]   # "#2A2A2D"
        pr = int(panel_hex[1:3], 16)
        pg_ = int(panel_hex[3:5], 16)
        pb  = int(panel_hex[5:7], 16)
        self._bg = QColor(pr, pg_, pb)

        if logo_path and os.path.exists(logo_path):
            try:
                # PIL path: replace all near-black pixels (brightness < 30) with
                # the exact panel colour so Qt compositing sees no black rectangle
                from PIL import Image as _PILImage
                import numpy as _np
                im   = _PILImage.open(logo_path).convert("RGBA")
                arr  = _np.array(im, dtype=_np.uint8)
                # Mask: pixels where ALL channels < 40 are "background"
                bg_mask = (arr[:,:,0] < 40) & (arr[:,:,1] < 40) & (arr[:,:,2] < 40)
                arr[bg_mask, 0] = pr
                arr[bg_mask, 1] = pg_
                arr[bg_mask, 2] = pb
                arr[bg_mask, 3] = 255   # fully opaque
                patched = _PILImage.fromarray(arr, "RGBA")
                # Convert to QPixmap via bytes
                import io as _io
                buf = _io.BytesIO(); patched.save(buf, format="PNG"); buf.seek(0)
                raw = QPixmap(); raw.loadFromData(buf.read())
            except Exception:
                raw = QPixmap(logo_path)

            if raw and not raw.isNull():
                big       = raw.scaledToWidth(target_w * 2, Qt.TransformationMode.SmoothTransformation)
                self._pix = big.scaledToWidth(target_w,     Qt.TransformationMode.SmoothTransformation)

        h = (self._pix.height() if self._pix else 36)
        self.setFixedSize(target_w, h + 8)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), self._bg)
        if self._pix:
            x = (self.width()  - self._pix.width())  // 2
            y = (self.height() - self._pix.height()) // 2
            p.drawPixmap(x, y, self._pix)
        else:
            p.setPen(QColor(C["ACCENT"])); p.setFont(mkfont(16, bold=True))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "ROBOSPEED")
        p.end()

# ═══════════════════════════════════════════════════════════════════
# GRADIENT PROGRESS BAR
# ═══════════════════════════════════════════════════════════════════
class GradBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pct = 0.0; self.setFixedHeight(10)
        self.setStyleSheet("background:transparent;")

    def set_pct(self, pct): self._pct = clamp(pct, 0, 1); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height(); r = h//2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(C["PANEL_DARK"]))); p.drawRoundedRect(0,0,w,h,r,r)
        fw = int(w*self._pct)
        if fw > 1:
            g = QLinearGradient(0,0,fw,0)
            g.setColorAt(0.0, QColor(C["ACCENT"])); g.setColorAt(1.0, QColor(C["GREEN"]))
            p.setBrush(QBrush(g)); p.drawRoundedRect(0,0,fw,h,r,r)
        p.end()

# ═══════════════════════════════════════════════════════════════════
# FORCE GRAPH PANEL
# ═══════════════════════════════════════════════════════════════════
class ForceGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_CARD"]))
        self.setPalette(pal)
        pg.setConfigOptions(antialias=True)
        self._mode = "time"
        self._time_window_s = 10.0
        self._tbuf = deque(maxlen=3000)
        self._fbuf = deque(maxlen=3000)
        self._cbuf = deque(maxlen=3000)

        v = QVBoxLayout(self); v.setContentsMargins(6,6,6,6); v.setSpacing(4)
        tr = QHBoxLayout(); tr.setSpacing(6)
        tr.addWidget(ILabel("Live Force Monitor", size=11, bold=True))
        tr.addStretch()
        trow = QHBoxLayout(); trow.setSpacing(0)
        self._btn_time  = QPushButton("Force vs Time")
        self._btn_cycle = QPushButton("Force vs Cycle")
        for b in (self._btn_time, self._btn_cycle):
            b.setFont(mkfont(8, bold=True)); b.setFixedHeight(22)
            b.setCheckable(True); b.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_time.setChecked(True)
        self._btn_time.clicked.connect(lambda: self._set_mode("time"))
        self._btn_cycle.clicked.connect(lambda: self._set_mode("cycle"))
        trow.addWidget(self._btn_time); trow.addWidget(self._btn_cycle)
        self._style_toggle()
        tr.addLayout(trow); tr.addSpacing(8)
        self._peak_lbl = ILabel("Peak: —", size=9, color=C["TEXT_SUB"])
        self._peak_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tr.addWidget(self._peak_lbl)
        v.addLayout(tr)

        self._pw = pg.PlotWidget(background=C["GRAPH_BG"])
        self._pw.showGrid(x=True, y=True, alpha=0.18)
        self._pw.setYRange(-0.2, 2.2, padding=0)
        self._pw.setLabel("left",   "Force (lbs)", color=C["TEXT_SUB"], size="9pt")
        self._pw.setLabel("bottom", "Time (s)",    color=C["TEXT_SUB"], size="9pt")
        for ax in ("left","bottom"):
            self._pw.getAxis(ax).setTextPen(C["TEXT_SUB"])
            self._pw.getAxis(ax).setPen(C["BORDER"])
        v.addWidget(self._pw, stretch=1)

        self._band = pg.LinearRegionItem(
            [0.5,1.8], orientation="horizontal",
            brush=pg.mkBrush(10,132,255,25),
            pen=pg.mkPen(C["FORCE_BAND"], width=1, style=Qt.PenStyle.DashLine),
            movable=False)
        self._pw.addItem(self._band)
        self._curve   = self._pw.plot(pen=pg.mkPen(C["FORCE_LINE"], width=2))
        self._scatter = pg.ScatterPlotItem(size=9, pen=pg.mkPen(None))
        self._pw.addItem(self._scatter)

        pr = QHBoxLayout(); pr.setContentsMargins(0,0,0,0)
        self._bl_lbl  = ILabel("Learning Baseline 0/30", size=8, color=C["TEXT_SUB"])
        self._prog_pct = ILabel("0 %", size=8, color=C["TEXT_SUB"])
        self._prog_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pr.addWidget(self._bl_lbl); pr.addStretch(); pr.addWidget(self._prog_pct)
        v.addLayout(pr)
        self._bar = GradBar(); v.addWidget(self._bar)

    def _style_toggle(self):
        for b, active in [(self._btn_time, self._mode=="time"), (self._btn_cycle, self._mode=="cycle")]:
            if active:
                b.setStyleSheet(f"QPushButton{{background:{C['ACCENT']};color:#fff;"
                                f"border:1px solid {C['ACCENT']};border-radius:3px;padding:1px 7px;}}")
            else:
                b.setStyleSheet(f"QPushButton{{background:{C['PANEL_DARK']};color:{C['TEXT_SUB']};"
                                f"border:1px solid {C['BORDER']};border-radius:3px;padding:1px 7px;}}"
                                f"QPushButton:hover{{background:{C['BORDER']};}}")

    def _set_mode(self, mode):
        self._mode = mode
        self._btn_time.setChecked(mode=="time"); self._btn_cycle.setChecked(mode=="cycle")
        self._style_toggle()
        self._pw.setLabel("bottom", "Time (s)" if mode=="time" else "Cycle #",
                          color=C["TEXT_SUB"], size="9pt")
        self._redraw()

    def _redraw(self):
        if len(self._tbuf) < 2: return
        xs = np.asarray(self._tbuf if self._mode=="time" else self._cbuf)
        self._curve.setData(xs, np.asarray(self._fbuf))
        if len(xs) > 1: self._pw.setXRange(xs[0], xs[-1], padding=0)

    def push(self, t, force, cycle=0):
        self._tbuf.append(t); self._fbuf.append(force); self._cbuf.append(cycle)

        # Match Stage D behavior: show a rolling 10-second window.
        while self._tbuf and (t - self._tbuf[0] > self._time_window_s):
            self._tbuf.popleft()
            self._fbuf.popleft()
            self._cbuf.popleft()

        self._redraw()
        if self._fbuf: self._peak_lbl.setText(f"Peak: {max(self._fbuf):.3f} lbs")

    def set_band(self, fmin, fmax): self._band.setRegion([fmin, fmax])

    def set_peaks(self, peaks):
        if peaks:
            self._scatter.setData(
                [p[0] for p in peaks], [p[1] for p in peaks],
                brush=[pg.mkBrush(255,69,58) if p[3] else pg.mkBrush(48,209,88) for p in peaks])
        else: self._scatter.clear()

    def set_progress(self, pct):
        self._bar.set_pct(pct); self._prog_pct.setText(f"{int(pct*100)} %")

    def set_baseline_mode(self, running, bc=0, bmax=30):
        if running:
            self._bl_lbl.setText("Detecting Anomalies")
            self._bl_lbl.setStyleSheet(f"color:{C['GREEN']};background:transparent;font-weight:bold;")
        else:
            self._bl_lbl.setText(f"Learning Baseline {bc}/{bmax}")
            self._bl_lbl.setStyleSheet(f"color:{C['TEXT_SUB']};background:transparent;font-weight:normal;")

# ═══════════════════════════════════════════════════════════════════
# VISION PANEL
# ═══════════════════════════════════════════════════════════════════
class VisionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_CARD"]))
        self.setPalette(pal)

        v = QVBoxLayout(self); v.setContentsMargins(6,6,6,6); v.setSpacing(4)
        tr = QHBoxLayout(); tr.setSpacing(6)
        tr.addWidget(ILabel("Visual Inspection System", size=11, bold=True))
        tr.addStretch()
        self._conn = ILabel("● Disconnected", size=9, color=C["RED"])
        self._conn.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tr.addWidget(self._conn)
        _sp = QWidget(); _sp.setFixedSize(1, 22); _sp.setStyleSheet("background:transparent;")
        tr.addWidget(_sp)
        v.addLayout(tr)

        self._view = pg.ImageView()
        self._view.ui.roiBtn.hide(); self._view.ui.menuBtn.hide(); self._view.ui.histogram.hide()
        self._view.setStyleSheet(f"background:{C['GRAPH_BG']};border:none;")
        noise = np.random.randint(25, 55, (240, 320, 3), dtype=np.uint8)
        self._view.setImage(noise, autoLevels=False, levels=(0,255))
        v.addWidget(self._view, stretch=1)

        bot = QHBoxLayout(); bot.setContentsMargins(0,2,0,0)
        self._feed_lbl = ILabel("Active feed: —", size=8, color=C["TEXT_SUB"])
        bot.addWidget(self._feed_lbl); bot.addStretch()
        self._conn2 = ILabel("No feed", size=8, color=C["TEXT_SUB"])
        self._conn2.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bot.addWidget(self._conn2)
        v.addLayout(bot)
        _bar_sp = QWidget(); _bar_sp.setFixedHeight(10); _bar_sp.setStyleSheet("background:transparent;")
        v.addWidget(_bar_sp)

    def set_connected(self, ok):
        if ok:
            self._conn.setText("● Connected")
            self._conn.setStyleSheet(f"color:{C['GREEN']};background:transparent;")
            self._conn2.setText("Feed active")
        else:
            self._conn.setText("● Disconnected")
            self._conn.setStyleSheet(f"color:{C['RED']};background:transparent;")
            self._conn2.setText("No feed")

    def set_active_feed(self, label):
        self._feed_lbl.setText(f"Active feed: {label}")

# ═══════════════════════════════════════════════════════════════════
# FLY-OUT TAB
# ═══════════════════════════════════════════════════════════════════
class FlyTab(QWidget):
    clicked = pyqtSignal()

    def __init__(self, label, side="left", parent=None):
        super().__init__(parent)
        self._label = label; self._side = side; self._hover = False
        self.setFixedWidth(22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def enterEvent(self, _): self._hover = True;  self.update()
    def leaveEvent(self, _): self._hover = False; self.update()
    def mousePressEvent(self, _): self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QBrush(QColor(C["TAB_HOVER"] if self._hover else C["TAB_BG"])))
        p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(0,0,w,h,4,4)
        p.setPen(QColor(C["TEXT_SUB"])); p.setFont(mkfont(8, bold=True))
        p.save()
        if self._side == "left":
            p.translate(w//2, h-6); p.rotate(-90)
        else:
            p.translate(w//2, 6); p.rotate(90)
        p.drawText(-h//2+4, 4, self._label)
        p.restore(); p.end()

# ═══════════════════════════════════════════════════════════════════
# LEFT PANEL  –  integrated, cohesive design
# ═══════════════════════════════════════════════════════════════════
class LeftPanel(QWidget):
    sig_start  = pyqtSignal()
    sig_pause  = pyqtSignal()
    sig_stop   = pyqtSignal()
    sig_home   = pyqtSignal()
    sig_reset  = pyqtSignal()
    sig_report = pyqtSignal()
    sig_exit   = pyqtSignal()
    sig_fields = pyqtSignal(dict)
    sig_record = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(PANEL_W)
        self.setAutoFillBackground(True)
        p = self.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL"])); self.setPalette(p)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea{{background:{C['PANEL']};border:none;}}
            QScrollBar:vertical{{background:{C['PANEL_DARK']};width:6px;border-radius:3px;}}
            QScrollBar::handle:vertical{{background:{C['BORDER']};border-radius:3px;min-height:20px;}}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}
        """)
        inner = QWidget(); inner.setStyleSheet(f"background:{C['PANEL']};")
        root = QVBoxLayout(inner); root.setContentsMargins(10,14,10,10); root.setSpacing(6)

        logo_path, _icon_path = _find_logo()
        lw = LogoWidget(logo_path, target_w=220)
        root.addWidget(lw, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.addSpacing(4)
        root.addWidget(HSep())
        root.addSpacing(4)

        # Primary control buttons — NOT in a collapsible (always visible & prominent)
        self._build_run(root)

        root.addSpacing(6)
        root.addWidget(HSep())

        self._set_sec = CollapsibleSection("MOTION CONTROL")
        self._build_settings()
        root.addWidget(self._set_sec)

        root.addStretch()
        ver = ILabel("v2.4  |  Stage D", size=8, color=C["TEXT_SUB"])
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(ver)

        scroll.setWidget(inner)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(scroll)

    def _build_run(self, root):
        """
        Button hierarchy per PDF update 3:
          START             – filled Deep Emerald  #1F8F4E
          PAUSE | STOP      – filled Amber / Red   (PDF spec)
          HOME  | RESET     – filled Grey #7A7D82  hover #95989E
          RECORD TRAJECTORY – filled Grey #7A7D82
          ─────────────────────────────────────────
          DOWNLOAD REPORT   – filled Grey #7A7D82
          EXIT              – filled Grey #7A7D82  (darker on hover)

        Icon rule: all icons rendered at font-size 10pt so they share
        identical visual weight. Unicode symbols chosen for equal cap-height:
          ▶  play  (filled triangle)
          ‖  pause (two equal vertical bars — same height as ■)
          ■  stop  (filled square)
          ⌂  home  ↺  reset  ●  record  ↓  download  ✕  exit
        """
        ICON_FONT = 10   # pt — all icons at this size
        BTN_H_LG  = 46   # START
        BTN_H_MD  = 38   # PAUSE / STOP
        BTN_H_SM  = 34   # HOME / RESET / RECORD / DOWNLOAD / EXIT

        grey_ss  = _btn_filled(C["BTN_GREY"],  C["BTN_GREY_H"])
        grey_sm  = _btn_filled(C["BTN_GREY"],  C["BTN_GREY_H"])   # same, reuse

        # ─ START ────────────────────────────────────────────────────
        self.btnStart = QPushButton("▶   START")
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setFont(mkfont(ICON_FONT + 1, bold=True))
        self.btnStart.setFixedHeight(BTN_H_LG)
        self.btnStart.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnStart.setStyleSheet(_btn_filled(C["BTN_START"], C["BTN_START_H"]))
        self.btnStart.clicked.connect(self._on_start)
        root.addWidget(self.btnStart)

        # ─ PAUSE | STOP ─────────────────────────────────────────────
        r1 = QHBoxLayout(); r1.setSpacing(6)
        # Pause: use ‖ (two identical bars, U+2016) — same glyph weight as ■
        self.btnPause = QPushButton("‖   PAUSE")
        self.btnPause.setObjectName("btnPause")
        self.btnPause.setFont(mkfont(ICON_FONT, bold=True))
        self.btnPause.setFixedHeight(BTN_H_MD)
        self.btnPause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnPause.setStyleSheet(_btn_filled(C["BTN_PAUSE"], C["BTN_PAUSE_H"]))
        self.btnPause.clicked.connect(self._on_pause)

        self.btnStop = QPushButton("■   STOP")
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setFont(mkfont(ICON_FONT, bold=True))
        self.btnStop.setFixedHeight(BTN_H_MD)
        self.btnStop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnStop.setStyleSheet(_btn_filled(C["BTN_STOP"], C["BTN_STOP_H"]))
        self.btnStop.clicked.connect(self._on_stop)
        r1.addWidget(self.btnPause, stretch=1); r1.addWidget(self.btnStop, stretch=1)
        root.addLayout(r1)

        # ─ HOME | RESET — grey filled ───────────────────────────────
        r2 = QHBoxLayout(); r2.setSpacing(6)
        self.btnHome = QPushButton("⌂   HOME")
        self.btnHome.setObjectName("btnHome")
        self.btnHome.setFont(mkfont(ICON_FONT, bold=True))
        self.btnHome.setFixedHeight(BTN_H_SM)
        self.btnHome.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnHome.setStyleSheet(grey_ss)
        self.btnHome.clicked.connect(self._on_home)

        self.btnReset = QPushButton("↺   RESET")
        self.btnReset.setObjectName("btnReset")
        self.btnReset.setFont(mkfont(ICON_FONT, bold=True))
        self.btnReset.setFixedHeight(BTN_H_SM)
        self.btnReset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnReset.setStyleSheet(grey_ss)
        self.btnReset.clicked.connect(self._on_reset)
        r2.addWidget(self.btnHome, stretch=1); r2.addWidget(self.btnReset, stretch=1)
        root.addLayout(r2)

        # ─ RECORD TRAJECTORY — grey filled ──────────────────────────
        self.btnRecord = QPushButton("●   RECORD TRAJECTORY")
        self.btnRecord.setObjectName("btnRecordTrajectory")
        self.btnRecord.setFont(mkfont(ICON_FONT, bold=True))
        self.btnRecord.setFixedHeight(BTN_H_SM)
        self.btnRecord.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnRecord.setStyleSheet(grey_sm)
        self.btnRecord.clicked.connect(self.sig_record.emit)
        root.addWidget(self.btnRecord)

        root.addWidget(HSep(color="#333336"))

        # ─ DOWNLOAD REPORT — grey filled ────────────────────────────
        self.btnDownloadReport = QPushButton("↓   DOWNLOAD REPORT")
        self.btnDownloadReport.setObjectName("btnDownloadReport")
        self.btnDownloadReport.setFont(mkfont(ICON_FONT, bold=True))
        self.btnDownloadReport.setFixedHeight(BTN_H_SM)
        self.btnDownloadReport.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnDownloadReport.setStyleSheet(grey_sm)
        self.btnDownloadReport.clicked.connect(self._on_report)
        root.addWidget(self.btnDownloadReport)

        # ─ EXIT — grey filled (slightly darker tone) ─────────────────
        exit_ss = _btn_filled("#5A5D62", "#6E7175")
        self.btnExit = QPushButton("✕   EXIT")
        self.btnExit.setObjectName("btnExit")
        self.btnExit.setFont(mkfont(ICON_FONT, bold=True))
        self.btnExit.setFixedHeight(BTN_H_SM)
        self.btnExit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnExit.setStyleSheet(exit_ss)
        self.btnExit.clicked.connect(self._on_exit)
        root.addWidget(self.btnExit)

    def _build_settings(self):
        s = self._set_sec
        for lbl, dflt, key, oname in [
            ("Velocity (0–1000)",  DEFAULTS["vel"],             "vel",            "txtVel"),
            ("Accel (0–2000)",     DEFAULTS["acc"],             "acc",            "txtAcc"),
            ("Jerk (0–10000)",     DEFAULTS["jerk"],            "jerk",           "txtJerk"),
            ("Cycles",             DEFAULTS["target_cycles"],   "target_cycles",  "txtCycles"),
            ("Baseline cycles",    DEFAULTS["baseline_cycles"], "baseline_cycles","txtBaseline"),
        ]:
            fr = FieldRow(lbl, dflt, oname)
            fr.edit.editingFinished.connect(self._emit_fields)
            s.add(fr)
            if not hasattr(self, "_fields"): self._fields = {}
            self._fields[key] = fr.edit

        s.add(HSep())
        s.add(ILabel("Force range (lbs)", size=9, bold=True, color=C["TEXT_SUB"]))
        frow = QHBoxLayout(); frow.setSpacing(6)
        lbl_min = ILabel("Min", size=9, color=C["TEXT_SUB"]); lbl_min.setFixedWidth(22)
        self.txtForceMin = IEdit(str(DEFAULTS["force_min"]), "min")
        self.txtForceMin.setObjectName("txtForceMin"); self.txtForceMin.setFixedWidth(64)
        lbl_max = ILabel("Max", size=9, color=C["TEXT_SUB"]); lbl_max.setFixedWidth(26)
        self.txtForceMax = IEdit(str(DEFAULTS["force_max"]), "max")
        self.txtForceMax.setObjectName("txtForceMax"); self.txtForceMax.setFixedWidth(64)
        frow.addWidget(lbl_min); frow.addWidget(self.txtForceMin)
        frow.addWidget(lbl_max); frow.addWidget(self.txtForceMax)
        frow.addStretch(); s.add_layout(frow)
        self.txtForceMin.editingFinished.connect(self._emit_fields)
        self.txtForceMax.editingFinished.connect(self._emit_fields)
        self._fields["force_min"] = self.txtForceMin
        self._fields["force_max"] = self.txtForceMax

        self.btnApply = QPushButton("✔  Apply Settings")
        self.btnApply.setObjectName("btnApplySettings")
        self.btnApply.setFont(mkfont(9, bold=True)); self.btnApply.setFixedHeight(32)
        self.btnApply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnApply.setStyleSheet(_btn_filled(C["ACCENT"], C["ACCENT_LT"]))
        self.btnApply.clicked.connect(self._emit_fields)
        s.add(self.btnApply)

    def _pi(self, t, fb):
        try: return int(float(t))
        except: return fb
    def _pf(self, t, fb):
        try: return float(t)
        except: return fb

    def get_fields(self):
        d = {}
        for key, edit in self._fields.items():
            lo, hi = LIMITS[key]
            d[key] = (clamp(self._pf(edit.text(), DEFAULTS[key]), lo, hi)
                      if key in ("force_min","force_max")
                      else clamp(self._pi(edit.text(), DEFAULTS[key]), lo, hi))
        return d

    def _emit_fields(self): self.sig_fields.emit(self.get_fields())
    def _on_start(self):  self._emit_fields(); self.sig_start.emit()
    def _on_pause(self):  self.sig_pause.emit()
    def _on_stop(self):   self.sig_stop.emit()
    def _on_home(self):   self.sig_home.emit()
    def _on_reset(self):  self.sig_reset.emit()
    def _on_report(self): self._emit_fields(); self.sig_report.emit()
    def _on_exit(self):   self.sig_exit.emit()

# ═══════════════════════════════════════════════════════════════════
# RIGHT PANEL  –  Insights + Camera/Overlay + Freq + AI Analyst
# ═══════════════════════════════════════════════════════════════════
class RightPanel(QWidget):
    sig_freq_updated   = pyqtSignal(str)
    sig_camera_changed = pyqtSignal(str)
    sig_ask_ai         = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(PANEL_W)
        self.setAutoFillBackground(True)
        p = self.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL"])); self.setPalette(p)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea{{background:{C['PANEL']};border:none;}}
            QScrollBar:vertical{{background:{C['PANEL_DARK']};width:6px;border-radius:3px;}}
            QScrollBar::handle:vertical{{background:{C['BORDER']};border-radius:3px;min-height:20px;}}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}
        """)
        inner = QWidget(); inner.setStyleSheet(f"background:{C['PANEL']};")
        root = QVBoxLayout(inner); root.setContentsMargins(8,10,8,8); root.setSpacing(6)

        self._di = CollapsibleSection("DURABILITY INSIGHTS")
        self._build_insights(); root.addWidget(self._di)

        self._vc = CollapsibleSection("VISUAL CONTROLS")
        self._build_visual_controls(); root.addWidget(self._vc)

        self._if_sec = CollapsibleSection("INSPECTION FREQUENCY")
        self._build_inspection_freq(); root.addWidget(self._if_sec)

        self._ai_sec = CollapsibleSection("AI ANALYST")
        self._build_ai(); root.addWidget(self._ai_sec)

        root.addStretch()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(scroll)

    def _build_insights(self):
        s = self._di
        s.add(ILabel("— INSPECTION PIPELINES —", size=8, bold=True, color=C["TEXT_SUB"]))
        for cb in (
            IChk("Surface Cosmetics  •  Active (C1)", checked=True),
            IChk("LED Sequence  •  Active (C1)",       checked=True),
            IChk("3D Geometry  •  Idle (C2)",           checked=False),
        ): s.add(cb)
        s.add(HSep())
        s.add(ILabel("— ANALYSIS CONFIGURATION —", size=8, bold=True, color=C["TEXT_SUB"]))
        for cb in (
            IChk("Cosmetic Defects",   checked=True, bold=True),
            IChk("LED Performance",    checked=True, bold=True),
            IChk("Geometry Deviation", checked=False),
        ): s.add(cb)

    def _build_visual_controls(self):
        s = self._vc
        s.add(ILabel("Camera Selection", size=9, bold=True, color=C["TEXT_SUB"]))
        cam_row = QHBoxLayout(); cam_row.setSpacing(8)
        self._rbC1    = IRadio("C1",         checked=True)
        self._rbC2    = IRadio("C2",         checked=False)
        self._rbSplit = IRadio("Split View", checked=False)
        grp = QButtonGroup(self)
        for rb in (self._rbC1, self._rbC2, self._rbSplit):
            grp.addButton(rb); cam_row.addWidget(rb)
        cam_row.addStretch(); s.add_layout(cam_row)
        self._rbC1.toggled.connect(   lambda on: self._emit_cam("C1")         if on else None)
        self._rbC2.toggled.connect(   lambda on: self._emit_cam("C2")         if on else None)
        self._rbSplit.toggled.connect(lambda on: self._emit_cam("Split View") if on else None)
        s.add(HSep())
        s.add(ILabel("Overlays", size=9, bold=True, color=C["TEXT_SUB"]))
        grid = QHBoxLayout(); grid.setSpacing(6)
        col1 = QVBoxLayout(); col1.setSpacing(3)
        col2 = QVBoxLayout(); col2.setSpacing(3)
        for name, dflt, col in [
            ("ROI",              True,  col1),
            ("Feature Tracking", True,  col1),
            ("Defect Highlight",  False, col1),
            ("Bounding Boxes",   True,  col2),
            ("LED Tracking",     True,  col2),
            ("Mesh Overlay",     True,  col2),
        ]:
            cb = IChk(name, checked=dflt); col.addWidget(cb)
        grid.addLayout(col1); grid.addLayout(col2); grid.addStretch()
        s.add_layout(grid)

    def _emit_cam(self, label): self.sig_camera_changed.emit(label)

    def _build_inspection_freq(self):
        s = self._if_sec
        self._vs_fields = {}
        for lbl, dflt, key, oname in [
            ("Surface capture every (cyc)", DEFAULTS["surface_capture_every"],     "surface_capture_every",     "txtCaptureEverySurface"),
            ("LED sequence every (cyc)",     DEFAULTS["led_capture_every"],         "led_capture_every",         "txtCaptureEveryLed"),
            ("Point cloud every (cyc)",      DEFAULTS["point_cloud_capture_every"], "point_cloud_capture_every", "txtCaptureEveryPointCloud"),
        ]:
            fr = FieldRow(lbl, dflt, oname)
            fr.edit.editingFinished.connect(self._on_freq_changed)
            s.add(fr); self._vs_fields[key] = fr.edit
        self.txtCaptureEverySurface    = self._vs_fields["surface_capture_every"]
        self.txtCaptureEveryLed        = self._vs_fields["led_capture_every"]
        self.txtCaptureEveryPointCloud = self._vs_fields["point_cloud_capture_every"]
        self._freq_msg = ILabel("", size=8, color=C["GREEN"])
        self._freq_msg.setWordWrap(True); s.add(self._freq_msg)
        self._freq_timer = QTimer(); self._freq_timer.setSingleShot(True)
        self._freq_timer.timeout.connect(lambda: self._freq_msg.setText(""))

    def _on_freq_changed(self):
        self._freq_msg.setText("✔  Inspection frequency updated.")
        self._freq_timer.start(3000)
        self.sig_freq_updated.emit("Inspection frequency updated")

    def _build_ai(self):
        s = self._ai_sec
        ask_row = QHBoxLayout(); ask_row.setSpacing(6)
        self.txtAiQuestion = IEdit("", "Ask a question about this test run…", align_right=False)
        self.txtAiQuestion.setObjectName("txtAiQuestion")
        self.txtAiQuestion.returnPressed.connect(self._on_ask)
        self.btnAskAi = QPushButton("Ask AI")
        self.btnAskAi.setObjectName("btnAskAi")
        self.btnAskAi.setFont(mkfont(9, bold=True))
        self.btnAskAi.setFixedWidth(70); self.btnAskAi.setFixedHeight(28)
        self.btnAskAi.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnAskAi.setStyleSheet(_btn_filled(C["PURPLE_DK"], "#9333EA"))
        self.btnAskAi.clicked.connect(self._on_ask)
        ask_row.addWidget(self.txtAiQuestion); ask_row.addWidget(self.btnAskAi)
        s.add_layout(ask_row)
        s.add(HSep())
        s.add(ILabel("AI Response", size=8, bold=True, color=C["TEXT_SUB"]))
        self._ai_scroll = QScrollArea()
        self._ai_scroll.setWidgetResizable(True)
        self._ai_scroll.setFixedHeight(200)
        self._ai_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ai_scroll.setStyleSheet(f"""
            QScrollArea{{background:{C['PANEL_DARK']};border:1px solid {C['BORDER']};border-radius:4px;}}
            QScrollBar:vertical{{background:{C['PANEL_DARK']};width:5px;border-radius:2px;}}
            QScrollBar::handle:vertical{{background:{C['BORDER']};border-radius:2px;min-height:14px;}}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}
        """)
        self.lblAiStatus = ILabel(
            "Ask a question to get AI analysis of the test run.",
            size=9, color=C["TEXT_SUB"])
        self.lblAiStatus.setObjectName("lblAiStatus")
        self.lblAiStatus.setWordWrap(True)
        self.lblAiStatus.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lblAiStatus.setContentsMargins(6, 5, 6, 5)
        self._ai_scroll.setWidget(self.lblAiStatus)
        s.add(self._ai_scroll)

    def _on_ask(self): self.sig_ask_ai.emit(self.txtAiQuestion.text().strip())

    def set_ai_response(self, text):
        self.lblAiStatus.setText(text)
        sb = self._ai_scroll.verticalScrollBar()
        QTimer.singleShot(50, lambda: sb.setValue(sb.maximum()))

    def get_vision_settings(self):
        d = {}
        for key, edit in self._vs_fields.items():
            lo, hi = LIMITS[key]
            try: d[key] = clamp(int(float(edit.text())), lo, hi)
            except: d[key] = DEFAULTS[key]
        return d

# ═══════════════════════════════════════════════════════════════════
# BOTTOM STATUS BAR
# ═══════════════════════════════════════════════════════════════════
class BottomBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setAutoFillBackground(True)
        p = self.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_DARK"])); self.setPalette(p)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(HSep())
        inner = QWidget(); inner.setStyleSheet("background:transparent;"); outer.addWidget(inner)
        root = QHBoxLayout(inner); root.setContentsMargins(14,5,14,5); root.setSpacing(10)
        self.dotStatus = StatusDot(14); root.addWidget(self.dotStatus)
        self.lblStatus = ILabel("State: STOPPED", size=10, bold=True)
        self.lblStatus.setObjectName("lblStatus"); root.addWidget(self.lblStatus)
        root.addWidget(VSep())
        self.lblParams = ILabel("—", size=9, color=C["TEXT_SUB"])
        root.addWidget(self.lblParams, stretch=1)
        root.addWidget(VSep())
        self.lblFailures = ILabel("", size=8, color=C["AMBER"])
        root.addWidget(self.lblFailures, stretch=1)

    def set_status(self, mode, cycle, target, amsg="", aclr=None):
        clrs = dict(RUNNING=C["DOT_RUN"], PAUSED=C["DOT_PAUSE"], STOPPED=C["DOT_STOP"], ERROR=C["DOT_ERR"])
        self.dotStatus.set_color(aclr or clrs.get(mode, C["DOT_STOP"]))
        txt = f"State: {mode}   Cycle: {cycle} / {target}"
        if amsg: txt += f"   ⚠  {amsg}"
        self.lblStatus.setText(txt)

    def set_params(self, text): self.lblParams.setText(text)

    def set_failures(self, oor, ret):
        parts = [f"{b}: oor={oor.get(b,0)} ret={ret.get(b,0)}" for b in ("A","B","C","D")]
        self.lblFailures.setText("⚠ Failures — " + "  |  ".join(parts))

# ═══════════════════════════════════════════════════════════════════
# MOCK DATA THREAD
# ═══════════════════════════════════════════════════════════════════
class MockDataThread(QThread):
    sig_data = pyqtSignal(float, float, int)

    def __init__(self, state_ref, lock_ref, parent=None):
        super().__init__(parent)
        self._state = state_ref; self._lock = lock_ref; self._go = True

    def stop(self): self._go = False; self.wait(3000)

    def run(self):
        t0 = time.time()
        while self._go:
            t = time.time() - t0
            noise = random.gauss(0, 0.004)
            wave  = 0.012 * math.sin(2 * math.pi * 0.45 * t) + noise
            with self._lock:
                running = self._state.get("running", False)
                cycle   = self._state.get("cycle_count", 0)
            if running:
                wave += 1.1 * (math.sin(2 * math.pi * 0.2 * t) > 0.93)
            self.sig_data.emit(t, max(-0.3, wave), cycle)
            time.sleep(0.02)

# ═══════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboSpeed  –  Durability Intelligence Platform")
        self.setMinimumSize(1280, 820); self.resize(1540, 940)
        self._left_visible = True; self._right_visible = True

        # ── Window icon (titlebar + taskbar) ─────────────────────
        logo_path, icon_path = _find_logo()
        if icon_path and os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        elif logo_path and os.path.exists(logo_path):
            # fall back to full logo scaled square
            self.setWindowIcon(QIcon(logo_path))

        # ── Terminal startup banner ───────────────────────────────
        _W = 60
        print()
        print("╔" + "═" * (_W-2) + "╗")
        print("║" + "  RoboSpeed Durability Intelligence Platform  v2.3".center(_W-2) + "║")
        print("╠" + "═" * (_W-2) + "╣")
        print("║" + f"  Logo  : {logo_path or 'built-in fallback'}".ljust(_W-2)[:_W-2] + "║")
        print("║" + f"  Icon  : {icon_path or 'not found'}".ljust(_W-2)[:_W-2] + "║")
        print("║" + f"  Theme : Dark Industrial  |  PyQt6 + PyQtGraph".ljust(_W-2) + "║")
        print("╚" + "═" * (_W-2) + "╝")
        print()

        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,     QColor(C["BG"]))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(C["TEXT"]))
        pal.setColor(QPalette.ColorRole.Base,       QColor(C["PANEL_DARK"]))
        pal.setColor(QPalette.ColorRole.Text,       QColor(C["TEXT"]))
        QApplication.setPalette(pal)
        self.setStyleSheet(f"QMainWindow{{background:{C['BG']};}}")

        self._state = dict(
            running=False, paused=False, stopped=True,
            cycle_count=0, target_cycles=DEFAULTS["target_cycles"],
            force_min=DEFAULTS["force_min"], force_max=DEFAULTS["force_max"],
            vel=DEFAULTS["vel"], acc=DEFAULTS["acc"], jerk=DEFAULTS["jerk"],
            baseline_cycles=DEFAULTS["baseline_cycles"],
            baseline_ready=False, baseline_count=0,
            force_out_of_range=dict(A=0,B=0,C=0,D=0),
            button_did_not_retract=dict(A=0,B=0,C=0,D=0),
            alert_msg="", alert_color=None, alert_until=0.0,
        )
        self._lock = threading.RLock(); self._t0 = time.time()
        self._peak_events = deque(maxlen=200)

        central = QWidget(); self.setCentralWidget(central)
        main_v = QVBoxLayout(central); main_v.setContentsMargins(0,0,0,0); main_v.setSpacing(0)
        main_v.addWidget(self._build_identity_bar())
        main_v.addWidget(HSep())

        body = QWidget(); body.setStyleSheet(f"background:{C['BG']};")
        body_h = QHBoxLayout(body); body_h.setContentsMargins(0,0,0,0); body_h.setSpacing(0)

        self.left = LeftPanel()
        self._left_tab = FlyTab("CONTROLS", side="left"); self._left_tab.setVisible(False)
        self._left_tab.clicked.connect(self._show_left)
        body_h.addWidget(self.left); body_h.addWidget(self._left_tab); body_h.addWidget(VSep())

        centre = QWidget(); centre.setStyleSheet(f"background:{C['BG']};")
        cv = QVBoxLayout(centre); cv.setContentsMargins(0,0,0,0); cv.setSpacing(0)
        graphs = QHBoxLayout(); graphs.setContentsMargins(8,8,8,4); graphs.setSpacing(8)
        self.force_graph  = ForceGraph()
        self.vision_panel = VisionPanel()
        graphs.addWidget(self.force_graph,  stretch=1)
        graphs.addWidget(self.vision_panel, stretch=1)
        cv.addLayout(graphs, stretch=1)
        cv.addWidget(HSep())
        self.bottom = BottomBar(); cv.addWidget(self.bottom)
        body_h.addWidget(centre, stretch=1)

        body_h.addWidget(VSep())
        self._right_tab = FlyTab("INSIGHTS", side="right"); self._right_tab.setVisible(False)
        self._right_tab.clicked.connect(self._show_right)
        body_h.addWidget(self._right_tab)
        self.right = RightPanel(); body_h.addWidget(self.right)

        main_v.addWidget(body, stretch=1)

        sb = self.statusBar()
        sb.setStyleSheet(f"""
            QStatusBar{{background:{C['HEADER_BG']};color:{C['TEXT_SUB']};
                font-size:9pt;border-top:1px solid {C['BORDER']};padding:1px 6px;}}
        """)
        sb.showMessage("RoboSpeed v2.4  |  Durability Intelligence Platform  |  Ready")

        self.left.sig_start.connect(self.on_start)
        self.left.sig_pause.connect(self.on_pause)
        self.left.sig_stop.connect(self.on_stop)
        self.left.sig_home.connect(self.on_home)
        self.left.sig_reset.connect(self.on_reset)
        self.left.sig_report.connect(self.on_report)
        self.left.sig_exit.connect(self.on_exit)
        self.left.sig_fields.connect(self.on_fields)
        self.left.sig_record.connect(self.on_record)
        self.right.sig_freq_updated.connect(lambda m: self.statusBar().showMessage(m, 3000))
        self.right.sig_camera_changed.connect(self.on_camera_changed)
        self.right.sig_ask_ai.connect(self.on_ai_ask)

        self._ui_timer = QTimer(self); self._ui_timer.timeout.connect(self._refresh_ui); self._ui_timer.start(100)
        self._thread = MockDataThread(self._state, self._lock, self)
        self._thread.sig_data.connect(lambda t, f, c: self.force_graph.push(t, f, c))
        self._thread.start()

    def _build_identity_bar(self):
        bar = QWidget(); bar.setFixedHeight(48)
        bar.setAutoFillBackground(True)
        p = bar.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_DARK"])); bar.setPalette(p)
        h = QHBoxLayout(bar); h.setContentsMargins(12,6,12,6); h.setSpacing(10)

        h.addWidget(ILabel("RoboSpeed  Durability Intelligence Platform", size=12, bold=True, color=C["ACCENT"]))
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"background:{C['BORDER']};border:none;"); sep.setFixedWidth(1)
        h.addWidget(sep)

        h.addWidget(ILabel("Project:", size=9, bold=True, color=C["TEXT_MED"]))
        self.txtProject = IEdit("Button Toy", "", align_right=False)
        self.txtProject.setObjectName("txtProject"); self.txtProject.setFixedWidth(120)
        h.addWidget(self.txtProject)

        h.addWidget(ILabel("|", size=11, color=C["BORDER"]))

        h.addWidget(ILabel("Test Profile:", size=9, bold=True, color=C["TEXT_MED"]))
        self.txtTestProfile = IEdit("1.5lb Cycle Test", "", align_right=False)
        self.txtTestProfile.setObjectName("txtTestProfile"); self.txtTestProfile.setFixedWidth(140)
        h.addWidget(self.txtTestProfile)

        # ── Save button — right of Test Profile ───────────────
        self.btnSave = QPushButton("Save")
        self.btnSave.setFont(mkfont(9, bold=True))
        self.btnSave.setFixedHeight(28); self.btnSave.setFixedWidth(76)
        self.btnSave.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnSave.setStyleSheet(f"""
            QPushButton{{background:transparent;color:{C['ACCENT']};
                border:1px solid {C['ACCENT']};border-radius:4px;
                padding:2px 8px;font-weight:bold;font-size:9pt;}}
            QPushButton:hover{{background:{C['ACCENT']};color:#fff;}}
            QPushButton:pressed{{background:{C['ACCENT_LT']};color:#fff;}}
        """)
        self.btnSave.clicked.connect(self._on_save)
        h.addWidget(self.btnSave)

        h.addStretch()

        self._btn_hide_left = QPushButton("◀ Controls")
        self._btn_hide_left.setFont(mkfont(9, bold=True)); self._btn_hide_left.setFixedHeight(28)
        self._btn_hide_left.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_hide_left.clicked.connect(self._toggle_left)
        self._btn_hide_left.setStyleSheet(f"""
            QPushButton{{background:{C['ACCENT']};color:#fff;border:1px solid {C['ACCENT']};
                border-radius:4px;padding:2px 10px;font-weight:bold;}}
            QPushButton:hover{{background:{C['ACCENT_LT']};}}
        """)
        h.addWidget(self._btn_hide_left)

        self._btn_hide_right = QPushButton("Insights ▶")
        self._btn_hide_right.setFont(mkfont(9, bold=True)); self._btn_hide_right.setFixedHeight(28)
        self._btn_hide_right.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_hide_right.clicked.connect(self._toggle_right)
        self._btn_hide_right.setStyleSheet(f"""
            QPushButton{{background:{C['ACCENT']};color:#fff;border:1px solid {C['ACCENT']};
                border-radius:4px;padding:2px 10px;font-weight:bold;}}
            QPushButton:hover{{background:{C['ACCENT_LT']};}}
        """)
        h.addWidget(self._btn_hide_right)
        return bar

    def _toggle_left(self):
        if self._left_visible: self._hide_left()
        else: self._show_left()

    def _toggle_right(self):
        if self._right_visible: self._hide_right()
        else: self._show_right()

    def _hide_left(self):
        self.left.setVisible(False); self._left_tab.setVisible(True)
        self._left_visible = False; self._btn_hide_left.setText("▶ Controls")

    def _show_left(self):
        self.left.setVisible(True); self._left_tab.setVisible(False)
        self._left_visible = True; self._btn_hide_left.setText("◀ Controls")

    def _hide_right(self):
        self.right.setVisible(False); self._right_tab.setVisible(True)
        self._right_visible = False; self._btn_hide_right.setText("Insights ◀")

    def _show_right(self):
        self.right.setVisible(True); self._right_tab.setVisible(False)
        self._right_visible = True; self._btn_hide_right.setText("Insights ▶")

    def _refresh_ui(self):
        with self._lock: st = dict(self._state)
        mode = "RUNNING" if st["running"] else "PAUSED" if st["paused"] else "STOPPED"
        now  = time.time()
        amsg = st["alert_msg"]   if now <= st["alert_until"] else ""
        aclr = st["alert_color"] if now <= st["alert_until"] else None
        self.bottom.set_status(mode, st["cycle_count"], st["target_cycles"], amsg, aclr)
        self.force_graph.set_baseline_mode(st["running"], st["baseline_count"], st["baseline_cycles"])
        bl = ("Detecting Anomalies" if st["running"]
              else ("Baseline READY" if st["baseline_ready"]
                    else f"Baseline {st['baseline_count']}/{st['baseline_cycles']}"))
        self.bottom.set_params(f"Vel:{st['vel']}  Acc:{st['acc']}  Jerk:{st['jerk']}  |  {bl}")
        self.bottom.set_failures(st["force_out_of_range"], st["button_did_not_retract"])
        self.force_graph.set_band(st["force_min"], st["force_max"])
        tc = max(1, st["target_cycles"])
        self.force_graph.set_progress(min(1.0, st["cycle_count"] / tc))
        t_now = time.time() - self._t0
        while self._peak_events and t_now - self._peak_events[0][0] > 10.0:
            self._peak_events.popleft()
        self.force_graph.set_peaks(list(self._peak_events))

    def on_start(self):
        with self._lock: self._state.update(running=True, paused=False, stopped=False)
        self._alert(C["GREEN"], "Test started", 2.0); self.statusBar().showMessage("● Running…")

    def on_pause(self):
        with self._lock: self._state.update(running=False, paused=True)
        self._alert(C["AMBER"], "Paused", 2.0); self.statusBar().showMessage("⏸ Paused")

    def on_stop(self):
        with self._lock: self._state.update(running=False, paused=False, stopped=True)
        self._alert(C["DOT_STOP"], "Stopped", 2.0); self.statusBar().showMessage("■ Stopped  –  Ready")

    def on_home(self):
        self._alert(C["ACCENT"], "Homing robot…", 2.0); self.statusBar().showMessage("Homing robot arm…")

    def on_reset(self):
        with self._lock:
            self._state.update(
                force_out_of_range=dict(A=0,B=0,C=0,D=0),
                button_did_not_retract=dict(A=0,B=0,C=0,D=0),
                cycle_count=0, baseline_ready=False, baseline_count=0,
            )
        self._peak_events.clear()
        self._alert(C["GREEN"], "Counters reset", 2.0); self.statusBar().showMessage("Reset complete")

    def on_report(self):
        self._alert(C["PURPLE_DK"], "Generating report…", 2.0)
        self.statusBar().showMessage("Download report – connect hardware first")

    def on_exit(self): self.close()

    def on_record(self):
        self._alert(C["TEAL_DK"], "Recording trajectory…", 3.0)
        self.statusBar().showMessage("Recording robot trajectory…")

    def on_fields(self, d):
        with self._lock:
            for k, v in d.items():
                if k in self._state: self._state[k] = v
        self.statusBar().showMessage("Motion control settings applied")

    def on_camera_changed(self, label):
        self.vision_panel.set_active_feed(label)
        self.statusBar().showMessage(f"Camera feed: {label}", 2000)

    def on_ai_ask(self, question):
        if not question:
            self.right.set_ai_response("⚠  Enter a question before pressing Ask AI.")
            return
        prev = question[:60] + ("…" if len(question) > 60 else "")
        self.right.set_ai_response(
            f"AI Analyst (mock):\n\n"
            f"For \"{prev}\"\n\n"
            f"• Force drift increased 4.2% after cycle 78.\n"
            f"• LED intensity decay is within tolerance.\n"
            f"• Projected lifecycle remaining: 2,900 cycles.\n\n"
            f"Recommend reviewing force band bounds and comparing\n"
            f"peak scatter against the baseline detection window.")
        self._alert(C["PURPLE_DK"], "AI response ready", 2.0)

    def _on_save(self):
        self._alert(C["ACCENT"], "Profile saved", 1.5)
        self.statusBar().showMessage(f"Profile saved: {self.txtTestProfile.text()}", 3000)

    def _alert(self, color, msg, dur=1.5):
        with self._lock:
            self._state["alert_color"] = color
            self._state["alert_msg"]   = msg
            self._state["alert_until"] = time.time() + dur

    def closeEvent(self, e): self._thread.stop(); e.accept()

# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv)
    app.setApplicationName("RoboSpeed DIP"); app.setOrganizationName("RoboSpeed")
    app.setFont(mkfont(10))

    # Set application-level icon (dock / taskbar)
    _logo, _icon = _find_logo()
    if _icon and os.path.exists(_icon):
        app.setWindowIcon(QIcon(_icon))
    elif _logo and os.path.exists(_logo):
        app.setWindowIcon(QIcon(_logo))

    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()