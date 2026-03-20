"""
RoboSpeed Durability Intelligence Platform  v2.7
PyQt6 + PyQtGraph  |  Dark Industrial Theme

v2.7 — Bug fixes per PDF spec (all 7 issues):

  Bug 1  RIGHT PANEL CLIPPING
         setFixedWidth → setMinimumWidth(PANEL_W+60)
         Panel expands freely; scrollbar no longer eats usable width.

  Bug 2  ENABLE/DISABLE BROKEN (root cause)
         Old: self._body.setEnabled(False) cascades to ALL children of _body,
              including the toggle button itself → impossible to re-enable.
         Fix: toggle row added to _body as before (always live).
              All content below (ROI mgr, checks, freq) lives in a new
              _content_w widget.  Only _content_w is disabled on toggle.

  Bug 3  GEOMETRY ENABLED BUTTON LOOKS RED
         Old: background:{col}22 — hex alpha suffix on orange = brownish-red on dark bg.
         Fix: background:{C['PANEL_DARK']} (dark), only border+text use module colour.

  Bug 4  ROI ADDABLE WHEN MODULE DISABLED
         ROIManager.set_module_enabled(False) explicitly disables Add ROI +
         shape picker buttons.  Called by DefectModule._apply_content_state().

  Bug 5  EYE / HIDE-SHOW NOT WORKING
         Old: full _rebuild_list() on every toggle — worked visually but only
              when _body wasn't disabled (bug 2 masked this).
         Fix: _update_eye() finds the eye button by objectName and updates only
              its text in-place; no full rebuild needed.  hide_all/show_all use
              _update_eye too.  Visibility is UI-only (PDF p.4).

  Bug 6  SIGNAL ARCHITECTURE (PDF p.3-4)
         ROIManager exposes 5 request signals:
           sig_add_roi_requested(module_key, shape)
           sig_delete_roi_requested(module_key, roi_id)
           sig_edit_roi_requested(module_key, roi_id)
           sig_lock_roi_requested(module_key, roi_id, new_locked_bool)
           sig_visibility_roi_changed(module_key, roi_id, is_visible)
         Buttons emit signals; GUI does NOT directly mutate ROI state.
         Default _confirm_* handlers self-wire for standalone operation.
         Backend disconnects and intercepts signals when integrated.

  Items 4/5/6/7 from prior release (v2.6) preserved:
    • ❚❚ pause icon (thicker, matches ■ stop)
    • Bottom bar two rows
    • 💾 Save Profile → .rsprofile file
    • 📂 Open Profile → load .rsprofile
"""

import sys, os, time, math, random, threading, json
from collections import deque
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFrame, QSizePolicy, QToolButton,
    QScrollArea, QCheckBox, QRadioButton, QButtonGroup, QFileDialog, QMessageBox,
    QGraphicsOpacityEffect, QComboBox,
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
        QCheckBox:hover{{color:{C['TEXT']};}}
        QCheckBox:disabled{{color:#555558;}}
        QCheckBox::indicator{{width:14px;height:14px;border-radius:3px;
            border:1.5px solid {C['BORDER']};background:{C['PANEL_DARK']};}}
        QCheckBox::indicator:hover{{border:1.5px solid {C['TEXT_MED']};background:{C['PANEL_DARK']};}}
        QCheckBox::indicator:checked{{background:{C['ACCENT']};border:1.5px solid {C['ACCENT']};}}
        QCheckBox::indicator:checked:hover{{background:{C['ACCENT_LT']};border:1.5px solid {C['ACCENT_LT']};}}
        QCheckBox::indicator:disabled{{background:#2A2A2D;border:1.5px solid #333336;}}
        QCheckBox::indicator:checked:disabled{{background:#404044;border:1.5px solid #404044;}}
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
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Durability Intelligence Platform")
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
# FORCE GRAPH PANEL  –  rolling 30-second / 60-cycle window
# ═══════════════════════════════════════════════════════════════════
class ForceGraph(QWidget):
    ROLL_SECS   = 30     # rolling window in time-mode
    ROLL_CYCLES = 60     # rolling window in cycle-mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_CARD"]))
        self.setPalette(pal)
        pg.setConfigOptions(antialias=True)
        self._mode = "time"
        # Use deques with large maxlen; rolling window is applied at draw time
        self._tbuf = deque(maxlen=6000)
        self._fbuf = deque(maxlen=6000)
        self._cbuf = deque(maxlen=6000)

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
        self._peak_labels = []

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
        self.set_peaks(getattr(self, "_last_peaks", []))

    def _redraw(self):
        if len(self._tbuf) < 2: return
        t_arr = np.asarray(self._tbuf)
        f_arr = np.asarray(self._fbuf)
        c_arr = np.asarray(self._cbuf)

        if self._mode == "time":
            # Rolling window: only show last ROLL_SECS seconds
            t_now = t_arr[-1]
            mask  = t_arr >= (t_now - self.ROLL_SECS)
            xs, ys = t_arr[mask], f_arr[mask]
            if len(xs) >= 2:
                self._curve.setData(xs, ys)
                self._pw.setXRange(xs[0], xs[-1], padding=0.02)
        else:
            # Rolling window: only show last ROLL_CYCLES cycles
            c_now = c_arr[-1]
            mask  = c_arr >= max(0, c_now - self.ROLL_CYCLES)
            xs, ys = c_arr[mask], f_arr[mask]
            if len(xs) >= 2:
                self._curve.setData(xs, ys)
                x0, x1 = xs[0], xs[-1]
                if x0 == x1: x1 = x0 + 1
                self._pw.setXRange(x0, x1, padding=0.02)

    def push(self, t, force, cycle=0):
        self._tbuf.append(t); self._fbuf.append(force); self._cbuf.append(cycle)
        self._redraw()
        # Peak label: max of visible window only
        if self._fbuf:
            self._peak_lbl.setText(f"Peak: {max(self._fbuf):.3f} lbs")

    def set_band(self, fmin, fmax): self._band.setRegion([fmin, fmax])

    def set_peaks(self, peaks):
        self._last_peaks = list(peaks)
        for item in self._peak_labels:
            self._pw.removeItem(item)
        self._peak_labels = []

        if not peaks:
            self._scatter.clear()
            return

        spots = []
        for peak in peaks:
            if isinstance(peak, dict):
                x = peak.get("t", 0.0) if self._mode == "time" else peak.get("cycle", 0)
                y = peak.get("y", 0.0)
                button = peak.get("button", "?")
                missed = bool(peak.get("missed", False))
                anomaly = peak.get("anomaly_type", "normal")
            else:
                x, y = peak[0], peak[1]
                button = peak[2] if len(peak) > 2 else "?"
                missed = bool(peak[3]) if len(peak) > 3 else False
                anomaly = peak[4] if len(peak) > 4 else ("missed_peak" if missed else "normal")

            if missed:
                brush = pg.mkBrush(255, 69, 58)
                label = f"{button} MISS"
            elif anomaly == "baseline_deviation":
                brush = pg.mkBrush(255, 159, 10)
                label = f"{button} DEV"
            elif anomaly == "force_out_of_range":
                brush = pg.mkBrush(255, 214, 10)
                label = f"{button} OOR"
            else:
                brush = pg.mkBrush(48, 209, 88)
                label = button

            spots.append({"pos": (x, y), "brush": brush, "data": {"label": label}})
            text = pg.TextItem(label, color=brush.color())
            text.setPos(x, y + (0.08 if missed else 0.05))
            self._pw.addItem(text)
            self._peak_labels.append(text)

        self._scatter.setData(spots)

    def set_progress(self, pct):
        self._bar.set_pct(pct); self._prog_pct.setText(f"{int(pct*100)} %")

    def set_baseline_label(self, baseline_ready, baseline_count, baseline_max):
        """
        Correct logic:
          - While baseline_count < baseline_max: show 'Learning Baseline X/Y'
          - After baseline is complete: show 'Detecting Anomalies' in green
        """
        if baseline_ready:
            self._bl_lbl.setText("Detecting Anomalies")
            self._bl_lbl.setStyleSheet(f"color:{C['GREEN']};background:transparent;font-weight:bold;font-size:8pt;")
        else:
            self._bl_lbl.setText(f"Learning Baseline {baseline_count}/{baseline_max}")
            self._bl_lbl.setStyleSheet(f"color:{C['TEXT_SUB']};background:transparent;font-weight:normal;font-size:8pt;")

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
        ver = ILabel("v2.7  |  Stage D", size=8, color=C["TEXT_SUB"])
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
        self.btnPause = QPushButton("❚❚  PAUSE")
        self.btnPause.setObjectName("btnPause")
        # Increase font 2pt so ❚❚ visually matches the weight/size of ■ in STOP
        self.btnPause.setFont(mkfont(ICON_FONT + 2, bold=True))
        self.btnPause.setFixedHeight(BTN_H_MD)
        self.btnPause.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnPause.setStyleSheet(_btn_filled(C["BTN_PAUSE"], C["BTN_PAUSE_H"]))
        self.btnPause.clicked.connect(self._on_pause)

        self.btnStop = QPushButton("■  STOP")
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setFont(mkfont(ICON_FONT + 2, bold=True))
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
# ROI ITEM  –  represents one region of interest
# ═══════════════════════════════════════════════════════════════════
class ROIItem:
    SHAPES = ["Rectangle", "Polygon", "Circle"]

    def __init__(self, index, shape="Rectangle"):
        self.index  = index                        # 1-based
        self.name   = f"ROI {index}"
        self.shape  = shape
        self.locked = False
        self.hidden = False

    def __repr__(self):
        return f"<ROIItem {self.name} {self.shape} locked={self.locked}>"


# ═══════════════════════════════════════════════════════════════════
# ROI MANAGER  –  signal-based, backend-ready  (PDF p.3-4)
# ═══════════════════════════════════════════════════════════════════
class ROIManager(QWidget):
    """
    SIGNAL ARCHITECTURE (PDF p.3-4):
    • Buttons emit *request* signals — they do NOT directly mutate ROI state.
    • Default _confirm_* handlers make it work standalone.
    • A real backend disconnects those handlers and intercepts signals instead.

    Signals:
      sig_add_roi_requested(module_key, shape)
      sig_delete_roi_requested(module_key, roi_id)
      sig_edit_roi_requested(module_key, roi_id)
      sig_lock_roi_requested(module_key, roi_id, new_locked_bool)
      sig_visibility_roi_changed(module_key, roi_id, is_visible)  # UI-only, no processing effect
    """
    MAX_ROI = 5

    MODULE_COLORS = dict(
        cosmetic = "#0A84FF",
        led      = "#30D158",
        geometry = "#FF9F0A",
    )

    sig_add_roi_requested      = pyqtSignal(str, str)       # (module_key, shape)
    sig_delete_roi_requested   = pyqtSignal(str, int)        # (module_key, roi_id)
    sig_edit_roi_requested     = pyqtSignal(str, int)        # (module_key, roi_id)
    sig_lock_roi_requested     = pyqtSignal(str, int, bool)  # (module_key, roi_id, new_state)
    sig_visibility_roi_changed = pyqtSignal(str, int, bool)  # (module_key, roi_id, is_visible)

    def __init__(self, module_key: str, camera: str, parent=None):
        super().__init__(parent)
        self._key            = module_key
        self._camera         = camera
        self._color          = self.MODULE_COLORS.get(module_key, C["ACCENT"])
        self._rois           = []       # list[ROIItem]
        self._shape_choice   = "Rectangle"
        self._mod_enabled    = True     # gate: add/shape disabled when module is off
        self.setStyleSheet("background:transparent;")

        v = QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(4)

        # ── Header: count + shape picker + Add ROI button ─────────
        hr = QHBoxLayout(); hr.setSpacing(4)
        self._count_lbl = ILabel("ROIs (0 / 5)", size=8, bold=True, color=C["TEXT_MED"])
        hr.addWidget(self._count_lbl)
        hr.addStretch()

        self._shape_combo = QPushButton("▾ Rectangle")
        self._shape_combo.setFont(mkfont(8)); self._shape_combo.setFixedHeight(22)
        self._shape_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._shape_combo.setStyleSheet(f"""
            QPushButton{{background:{C['PANEL_DARK']};color:{C['TEXT_MED']};
                border:1px solid {C['BORDER']};border-radius:3px;padding:1px 6px;}}
            QPushButton:hover{{background:{C['BORDER']};}}
            QPushButton:disabled{{color:#3A3A3D;border-color:#333;}}""")
        self._shape_combo.clicked.connect(self._cycle_shape)
        hr.addWidget(self._shape_combo)

        self._add_btn = QPushButton("+ Add ROI")
        self._add_btn.setFont(mkfont(8, bold=True)); self._add_btn.setFixedHeight(22)
        self._add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_btn.setStyleSheet(f"""
            QPushButton{{background:{self._color};color:#fff;
                border:1px solid {self._color};border-radius:3px;padding:1px 8px;font-weight:bold;}}
            QPushButton:hover{{background:{self._color};color:#fff;
                border:1px solid {self._color};opacity:0.85;}}
            QPushButton:pressed{{background:{self._color};color:#fff;border:1px solid {self._color};}}
            QPushButton:disabled{{background:#3A3A3D;color:#555;border-color:#3A3A3D;}}""")
        self._add_btn.setToolTip("Add a new ROI (draw on camera view)")
        self._add_btn.clicked.connect(self._req_add)
        hr.addWidget(self._add_btn)
        v.addLayout(hr)

        # ── ROI list container ────────────────────────────────────
        self._list_w = QWidget(); self._list_w.setStyleSheet("background:transparent;")
        self._list_v = QVBoxLayout(self._list_w)
        self._list_v.setContentsMargins(0,0,0,0); self._list_v.setSpacing(2)
        v.addWidget(self._list_w)

        # ── Wire default standalone confirm handlers ───────────────
        self.sig_add_roi_requested.connect(self._confirm_add)
        self.sig_delete_roi_requested.connect(self._confirm_delete)
        self.sig_lock_roi_requested.connect(self._confirm_lock)
        self.sig_visibility_roi_changed.connect(self._confirm_visibility)

    # ── Shape picker ──────────────────────────────────────────────
    def _cycle_shape(self):
        idx = ROIItem.SHAPES.index(self._shape_choice)
        self._shape_choice = ROIItem.SHAPES[(idx + 1) % len(ROIItem.SHAPES)]
        self._shape_combo.setText(f"▾ {self._shape_choice}")

    # ── Module enabled gate (called by DefectModule) ───────────────
    def set_module_enabled(self, enabled: bool):
        self._mod_enabled = enabled
        n = len(self._rois)
        self._add_btn.setEnabled(enabled and n < self.MAX_ROI)
        self._shape_combo.setEnabled(enabled)

    # ── Request signals — GUI never mutates state directly ─────────
    def _req_add(self):
        if not self._mod_enabled or len(self._rois) >= self.MAX_ROI:
            return
        self.sig_add_roi_requested.emit(self._key, self._shape_choice)

    def _req_delete(self, roi_id: int):
        self.sig_delete_roi_requested.emit(self._key, roi_id)

    def _req_edit(self, roi_id: int):
        self.sig_edit_roi_requested.emit(self._key, roi_id)

    def _req_lock(self, roi_id: int, currently_locked: bool):
        self.sig_lock_roi_requested.emit(self._key, roi_id, not currently_locked)

    def _req_visibility(self, roi_id: int):
        roi = next((r for r in self._rois if r.index == roi_id), None)
        if roi:
            # Signal: toggle visibility — new state is the opposite of current hidden
            self.sig_visibility_roi_changed.emit(self._key, roi_id, roi.hidden)

    # ── Default confirm handlers (standalone mode) ─────────────────
    def _confirm_add(self, _key: str, shape: str):
        if len(self._rois) >= self.MAX_ROI:
            return
        roi = ROIItem(len(self._rois) + 1, shape)
        self._rois.append(roi)
        self._append_row(roi)

    def _confirm_delete(self, _key: str, roi_id: int):
        self._rois = [r for r in self._rois if r.index != roi_id]
        for i, r in enumerate(self._rois):
            r.index = i + 1
            r.name  = f"ROI {r.index}"   # FIX: must sync name after re-index
        self._rebuild_list()

    def _confirm_lock(self, _key: str, roi_id: int, new_locked: bool):
        for roi in self._rois:
            if roi.index == roi_id:
                roi.locked = new_locked
                self._rebuild_row(roi_id)
                break

    def _confirm_visibility(self, _key: str, roi_id: int, make_visible: bool):
        # make_visible = True means show (hidden → False)
        for roi in self._rois:
            if roi.index == roi_id:
                roi.hidden = not make_visible
                self._update_eye(roi_id)
                break

    # ── Row building ──────────────────────────────────────────────
    def _append_row(self, roi):
        self._list_v.addWidget(self._make_row(roi))
        self._sync_header()

    def _rebuild_list(self):
        while self._list_v.count():
            it = self._list_v.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for roi in self._rois:
            self._list_v.addWidget(self._make_row(roi))
        self._sync_header()

    def _rebuild_row(self, roi_id: int):
        roi = next((r for r in self._rois if r.index == roi_id), None)
        if not roi:
            return
        for i in range(self._list_v.count()):
            it = self._list_v.itemAt(i)
            if it and it.widget() and it.widget().objectName() == f"rr_{self._key}_{roi_id}":
                old = self._list_v.takeAt(i)
                if old.widget():
                    old.widget().deleteLater()
                self._list_v.insertWidget(i, self._make_row(roi))
                return

    def _make_row(self, roi) -> QWidget:
        rid = roi.index
        oid = f"rr_{self._key}_{rid}"
        w = QWidget(); w.setObjectName(oid)
        w.setStyleSheet(f"""
            QWidget#{oid}{{background:{C['PANEL_DARK']};
                border:1px solid {C['BORDER']};
                border-left:2px solid {self._color};
                border-radius:3px;}}""")
        h = QHBoxLayout(w); h.setContentsMargins(5,2,4,2); h.setSpacing(3)

        # Eye button – visibility toggle, UI only, no processing effect (PDF p.4)
        eye = QPushButton("👁" if not roi.hidden else "🔴")
        eye.setObjectName(f"eye_{self._key}_{rid}")
        eye.setFont(mkfont(9)); eye.setFixedSize(22, 20); eye.setFlat(True)
        eye.setCursor(Qt.CursorShape.PointingHandCursor)
        eye.setStyleSheet("QPushButton{background:transparent;border:none;padding:0;}")
        eye.setToolTip("Toggle visibility (does not affect processing)")
        eye.clicked.connect(lambda _, r=rid: self._req_visibility(r))
        h.addWidget(eye)

        # Name
        nm = ILabel(f"{rid}. {roi.name}  ({roi.shape})", size=8)
        nm.setObjectName(f"nm_{self._key}_{rid}")
        if roi.locked:
            nm.setStyleSheet(f"color:{C['TEXT_SUB']};font-style:italic;background:transparent;font-size:8pt;")
        h.addWidget(nm, stretch=1)

        # Edit | Lock/Unlock | Delete
        for lbl, cb in [
            ("Edit",
                lambda _, r=rid: self._req_edit(r)),
            ("Unlock" if roi.locked else "Lock",
                lambda _, r=rid, lk=roi.locked: self._req_lock(r, lk)),
            ("Delete",
                lambda _, r=rid: self._req_delete(r)),
        ]:
            b = QPushButton(lbl); b.setFont(mkfont(7)); b.setFixedHeight(18)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            clr = C["BTN_STOP"] if lbl == "Delete" else C["TEXT_SUB"]
            b.setStyleSheet(f"""
                QPushButton{{background:transparent;color:{clr};
                    border:1px solid {C['BORDER']};border-radius:2px;padding:0 3px;}}
                QPushButton:hover{{background:{C['BORDER']};color:{C['TEXT']};}}
                QPushButton:disabled{{background:transparent;color:#555558;border-color:#3A3A3D;}}""")
            b.clicked.connect(cb)
            # Enforce: Delete disabled when ROI is locked
            if lbl == "Delete" and roi.locked:
                b.setEnabled(False)
            h.addWidget(b)
        return w

    def _sync_header(self):
        n = len(self._rois)
        self._count_lbl.setText(f"ROIs ({n} / {self.MAX_ROI})")
        self._add_btn.setEnabled(self._mod_enabled and n < self.MAX_ROI)
        self._add_btn.setToolTip(
            "Maximum of 5 ROIs" if n >= self.MAX_ROI else "Add a new ROI")

    def _update_eye(self, roi_id: int):
        """Update only the eye icon in place — no full rebuild needed."""
        roi = next((r for r in self._rois if r.index == roi_id), None)
        if not roi:
            return
        btn = self._list_w.findChild(QPushButton, f"eye_{self._key}_{roi_id}")
        if btn:
            btn.setText("👁" if not roi.hidden else "🔴")

    # ── Global show/hide (Visual Controls) ───────────────────────
    def hide_all(self):
        for r in self._rois:
            r.hidden = True
            self._update_eye(r.index)

    def show_all(self):
        for r in self._rois:
            r.hidden = False
            self._update_eye(r.index)

    @property
    def camera(self): return self._camera

    @property
    def rois(self): return list(self._rois)


# ═══════════════════════════════════════════════════════════════════
# DEFECT MODULE WIDGET  –  supports Cycle Durability & Standalone Visual
# ═══════════════════════════════════════════════════════════════════
class DefectModule(CollapsibleSection):
    """
    Inspection mode switching via apply_inspection_mode(mode):
      'cycle_durability'  → freq visible, no golden, no run button
      'standalone_visual' → golden section shown, freq greyed, run button visible

    Enable/Disable: toggle row always interactive (not in _content_w).
    Opacity effect dims content to 35% when disabled.
    No setCheckable — avoids Windows native :checked platform overlay.
    """
    sig_camera_request           = pyqtSignal(str, str)
    sig_capture_golden_requested = pyqtSignal(str)        # module_key
    sig_load_golden_requested    = pyqtSignal(str)        # module_key
    sig_replace_golden_requested = pyqtSignal(str)        # module_key
    sig_run_inspection_requested = pyqtSignal(str)        # module_key

    STATE_COLORS = {
        "COMPLETE":    "#30D158",
        "PROCESSING…": "#FF9F0A",
        "IDLE":        "#8E8E93",
        "READY":       "#30D158",
        "ERROR":       "#FF453A",
    }

    def __init__(self, title: str, module_key: str, camera: str,
                 checks: list, freq_label: str, freq_default: int,
                 run_label: str = "",
                 enabled: bool = True, state: str = "IDLE", parent=None):
        super().__init__(title, parent)
        self._module_key   = module_key
        self._camera_id    = camera
        self._enabled      = enabled
        self._state_str    = state
        self._module_color = ROIManager.MODULE_COLORS.get(module_key, C["ACCENT"])
        self._run_label    = run_label or f"▶  Run {title.split('(')[0].strip().title()} Inspection"
        self._golden_path  = None
        self._golden_ts    = None

        # ── Toggle row — in _body, NEVER disabled ─────────────────
        top_row = QHBoxLayout(); top_row.setSpacing(6)
        self._enable_btn = QPushButton("⬤  Enabled" if enabled else "◯  Disabled")
        self._enable_btn.setFont(mkfont(8, bold=True))
        self._enable_btn.setFixedHeight(26)
        self._enable_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._enable_btn.clicked.connect(self._on_toggle_enable)
        self._update_enable_style()
        top_row.addWidget(self._enable_btn); top_row.addStretch()
        self._state_lbl = ILabel("", size=8)
        self._update_state_badge()
        top_row.addWidget(self._state_lbl)
        self.add_layout(top_row)
        self.add(HSep(color="#2C2C2F"))

        # ── _content_w — only this gets disabled / dimmed ─────────
        self._content_w = QWidget()
        self._content_w.setStyleSheet("background:transparent;")
        self._opacity_fx = QGraphicsOpacityEffect(self._content_w)
        self._opacity_fx.setOpacity(1.0)
        self._content_w.setGraphicsEffect(self._opacity_fx)
        cv = QVBoxLayout(self._content_w)
        cv.setContentsMargins(0,0,0,0); cv.setSpacing(5)

        # ── Golden Reference (Standalone Visual only) ──────────────
        self._golden_w = self._build_golden_section()
        self._golden_w.setVisible(False)
        cv.addWidget(self._golden_w)

        # ── ROI Manager ───────────────────────────────────────────
        self.roi_mgr = ROIManager(module_key, camera)
        self.roi_mgr.sig_add_roi_requested.connect(
            lambda mk, sh: self.sig_camera_request.emit(
                self._camera_id,
                f"Switched to {self._camera_id} for {self._module_key.title()} ROI"))
        cv.addWidget(self.roi_mgr)
        cv.addWidget(HSep(color="#2C2C2F"))

        # ── Checks grid ───────────────────────────────────────────
        cv.addWidget(ILabel("Checks:", size=8, bold=True, color=C["TEXT_SUB"]))
        check_grid = QWidget(); check_grid.setStyleSheet("background:transparent;")
        cg = QHBoxLayout(check_grid); cg.setContentsMargins(0,0,0,0); cg.setSpacing(4)
        col1 = QVBoxLayout(); col1.setSpacing(2)
        col2 = QVBoxLayout(); col2.setSpacing(2)
        self._checks = {}
        for i, (text, chk) in enumerate(checks):
            cb = IChk(text, checked=chk)
            self._checks[text] = cb
            (col1 if i % 2 == 0 else col2).addWidget(cb)
        cg.addLayout(col1); cg.addLayout(col2); cg.addStretch()
        cv.addWidget(check_grid)

        # ── Freq separator + row (visible always; greyed in standalone) ─
        self._freq_sep = HSep(color="#2C2C2F")
        cv.addWidget(self._freq_sep)
        self._freq_row_w = QWidget(); self._freq_row_w.setStyleSheet("background:transparent;")
        fr = QHBoxLayout(self._freq_row_w)
        fr.setContentsMargins(0,0,0,0); fr.setSpacing(6)
        self._freq_lbl = ILabel(freq_label, size=8, color=C["TEXT_SUB"])
        fr.addWidget(self._freq_lbl); fr.addStretch()
        self.freq_edit = IEdit(str(freq_default))
        self.freq_edit.setFixedWidth(52); self.freq_edit.setFixedHeight(24)
        fr.addWidget(self.freq_edit)
        cv.addWidget(self._freq_row_w)

        # ── Run Inspection button (Standalone Visual only) ─────────
        self._run_btn = QPushButton(self._run_label)
        self._run_btn.setFont(mkfont(9, bold=True))
        self._run_btn.setFixedHeight(34)
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setStyleSheet(f"""
            QPushButton{{background:{C['ACCENT']};color:#fff;
                border:1px solid {C['ACCENT']};border-radius:5px;
                padding:4px 14px;font-weight:bold;font-size:9pt;}}
            QPushButton:hover{{background:{C['ACCENT_LT']};}}
            QPushButton:pressed{{background:{C['ACCENT']};}}""")
        self._run_btn.clicked.connect(
            lambda: self.sig_run_inspection_requested.emit(self._module_key))
        self._run_btn.setVisible(False)
        cv.addWidget(self._run_btn)

        self.add(self._content_w)
        self._apply_content_state(enabled)

    # ── Golden Reference section ───────────────────────────────────
    def _build_golden_section(self) -> QWidget:
        col = self._module_color
        w = QWidget(); w.setStyleSheet("background:transparent;")
        v = QVBoxLayout(w); v.setContentsMargins(0,0,0,4); v.setSpacing(4)

        hdr_r = QHBoxLayout()
        hdr_r.addWidget(ILabel("Golden Reference", size=8, bold=True, color=col))
        hdr_r.addStretch()
        v.addLayout(hdr_r)
        v.addWidget(HSep(color="#333336"))

        self._golden_status_lbl = ILabel("◎  No golden loaded", size=8, color=C["TEXT_SUB"])
        self._golden_ts_lbl     = ILabel("", size=7, color=C["TEXT_SUB"])
        v.addWidget(self._golden_status_lbl)
        v.addWidget(self._golden_ts_lbl)

        btn_r = QHBoxLayout(); btn_r.setSpacing(5)
        for lbl, slot in [
            ("📷 Capture Golden", self._on_capture_golden),
            ("📂 Load Golden",    self._on_load_golden),
            ("↺ Replace",         self._on_replace_golden),
        ]:
            b = QPushButton(lbl); b.setFont(mkfont(7, bold=True)); b.setFixedHeight(24)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(f"""
                QPushButton{{background:{C['PANEL_DARK']};color:{col};
                    border:1px solid {col};border-radius:3px;padding:1px 6px;}}
                QPushButton:hover{{background:{col};color:#fff;}}
                QPushButton:pressed{{background:{C['PANEL_DARK']};}}""")
            b.clicked.connect(slot)
            btn_r.addWidget(b)
        btn_r.addStretch()
        v.addLayout(btn_r)
        v.addWidget(HSep(color="#333336"))
        return w

    def _on_capture_golden(self):
        self._golden_ts   = time.strftime("%H:%M:%S")
        self._golden_path = "<captured>"
        col = self._module_color
        self._golden_status_lbl.setText("✅  Golden Loaded")
        self._golden_status_lbl.setStyleSheet(
            f"color:{col};background:transparent;font-size:8pt;font-weight:bold;")
        self._golden_ts_lbl.setText(f"Captured: {self._golden_ts}")
        self.sig_capture_golden_requested.emit(self._module_key)

    def _on_load_golden(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Golden Reference", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if not path:
            return
        self._golden_path = path
        self._golden_ts   = time.strftime("%H:%M:%S")
        col = self._module_color
        self._golden_status_lbl.setText("✅  Golden Loaded")
        self._golden_status_lbl.setStyleSheet(
            f"color:{col};background:transparent;font-size:8pt;font-weight:bold;")
        self._golden_ts_lbl.setText(
            f"Loaded: {self._golden_ts}  |  {os.path.basename(path)}")
        self.sig_load_golden_requested.emit(self._module_key)

    def _on_replace_golden(self):
        self._golden_path = None; self._golden_ts = None
        self._golden_status_lbl.setText("◎  No golden loaded")
        self._golden_status_lbl.setStyleSheet(
            f"color:{C['TEXT_SUB']};background:transparent;font-size:8pt;")
        self._golden_ts_lbl.setText("")
        self.sig_replace_golden_requested.emit(self._module_key)

    # ── Inspection mode switching ──────────────────────────────────
    def apply_inspection_mode(self, mode: str):
        """
        'cycle_durability'  → freq enabled, no golden, no run btn
        'standalone_visual' → golden visible, freq greyed out, run btn shown
        """
        standalone = (mode == "standalone_visual")
        self._golden_w.setVisible(standalone)
        self._run_btn.setVisible(standalone)

        # Freq row: always present but disabled/greyed in standalone
        freq_on = not standalone
        self._freq_row_w.setEnabled(freq_on)
        self.freq_edit.setEnabled(freq_on)
        grey = C["TEXT_SUB"] if freq_on else "#3A3A3D"
        self._freq_lbl.setStyleSheet(
            f"color:{grey};background:transparent;font-size:8pt;")

        if standalone and self._state_str not in ("READY", "COMPLETE", "ERROR"):
            self.set_state("READY")
        elif not standalone and self._state_str == "READY":
            self.set_state("IDLE")

    # ── Enable / disable ──────────────────────────────────────────
    def _on_toggle_enable(self):
        self._enabled = not self._enabled
        self._enable_btn.setText("⬤  Enabled" if self._enabled else "◯  Disabled")
        self._update_enable_style()
        self._apply_content_state(self._enabled)

    def _apply_content_state(self, enabled: bool):
        self._content_w.setEnabled(enabled)
        self._opacity_fx.setOpacity(1.0 if enabled else 0.35)
        self.roi_mgr.set_module_enabled(enabled)

    def _update_enable_style(self):
        col = self._module_color
        if self._enabled:
            self._enable_btn.setStyleSheet(f"""
                QPushButton{{background:{C['PANEL_DARK']};color:{col};
                    border:1.5px solid {col};border-radius:4px;
                    padding:3px 10px;font-weight:bold;font-size:8pt;}}
                QPushButton:hover{{background:{C['PANEL']};border:1.5px solid {col};}}
                QPushButton:pressed{{background:{C['PANEL_DARK']};}}""")
        else:
            self._enable_btn.setStyleSheet(f"""
                QPushButton{{background:transparent;color:{C['TEXT_SUB']};
                    border:1px solid {C['BORDER']};border-radius:4px;
                    padding:3px 10px;font-weight:bold;font-size:8pt;}}
                QPushButton:hover{{background:{C['BORDER']};color:{C['TEXT']};}}
                QPushButton:pressed{{background:{C['PANEL_DARK']};}}""")

    # ── State badge ───────────────────────────────────────────────
    def set_state(self, state: str):
        self._state_str = state
        self._update_state_badge()

    def _update_state_badge(self):
        col = self.STATE_COLORS.get(self._state_str, C["TEXT_SUB"])
        self._state_lbl.setText(f"● {self._state_str}")
        self._state_lbl.setStyleSheet(
            f"color:{col};background:transparent;font-size:8pt;font-weight:bold;")

    @property
    def camera_id(self): return self._camera_id


# ═══════════════════════════════════════════════════════════════════
# RIGHT PANEL  –  Inspection Mode aware
# ═══════════════════════════════════════════════════════════════════
class RightPanel(QWidget):
    sig_camera_changed    = pyqtSignal(str)
    sig_ask_ai            = pyqtSignal(str)
    sig_freq_updated      = pyqtSignal(str)
    sig_mode_changed      = pyqtSignal(str)   # 'cycle_durability' | 'standalone_visual'

    MODES = ["Cycle Durability", "Standalone Visual"]
    MODE_KEYS = {"Cycle Durability": "cycle_durability",
                 "Standalone Visual": "standalone_visual"}

    def __init__(self, parent=None):
        super().__init__(parent)
        # Single source of truth for inspection mode
        self._inspection_mode = "cycle_durability"
        self._current_mode = "cycle_durability"
        self.setMinimumWidth(PANEL_W + 60)
        self.setAutoFillBackground(True)
        p = self.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL"])); self.setPalette(p)

        # ── Fixed header ──────────────────────────────────────────
        hdr = QWidget(); hdr.setFixedHeight(34)
        hdr.setAutoFillBackground(True)
        hp = hdr.palette(); hp.setColor(QPalette.ColorRole.Window, QColor(C["HEADER_BG"])); hdr.setPalette(hp)
        hh = QHBoxLayout(hdr); hh.setContentsMargins(10,0,10,0)
        hh.addWidget(ILabel("DURABILITY INSIGHTS", size=9, bold=True, color=C["TEXT_MED"]))
        hh.addStretch()

        # ── Scrollable content ────────────────────────────────────
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
        root  = QVBoxLayout(inner); root.setContentsMargins(8,8,14,8); root.setSpacing(6)

        # ── Inspection Mode status (read-only — selector is in identity bar only) ─
        mode_w = QWidget(); mode_w.setStyleSheet(
            f"background:{C['PANEL_DARK']};border-radius:4px;")
        mv = QHBoxLayout(mode_w); mv.setContentsMargins(8,8,8,8); mv.setSpacing(8)
        mv.addWidget(ILabel("INSPECTION MODE", size=8, bold=True, color=C["TEXT_MED"]))
        mv.addWidget(ILabel("—", size=8, color=C["BORDER"]))
        self._mode_status_lbl = ILabel("Cycle Durability", size=9, bold=True, color=C["ACCENT"])
        mv.addWidget(self._mode_status_lbl)
        mv.addStretch()
        root.addWidget(mode_w)

        # ── Run Baseline section (Cycle Durability only) ──────────
        self._baseline_w = QWidget()
        self._baseline_w.setStyleSheet(
            f"background:{C['PANEL_DARK']};border-radius:4px;")
        bv = QVBoxLayout(self._baseline_w); bv.setContentsMargins(8,6,8,8); bv.setSpacing(4)
        bv.addWidget(ILabel("RUN BASELINE  (Auto)", size=8, bold=True, color=C["TEXT_MED"]))
        self._baseline_status = ILabel(
            "⏳  Pending — will auto-capture at START", size=8, color=C["TEXT_SUB"])
        bv.addWidget(self._baseline_status)
        self._baseline_ts_lbl = ILabel("", size=7, color=C["TEXT_SUB"])
        bv.addWidget(self._baseline_ts_lbl)
        root.addWidget(self._baseline_w)

        # ── Cosmetic Defects (C1) ─────────────────────────────────
        self.mod_cosmetic = DefectModule(
            "COSMETIC DEFECTS  (C1)", "cosmetic", "C1",
            checks=[
                ("Scratch Detection", True), ("Color Variation", True),
                ("Coating Wear",      True), ("Stress Marks",    True),
                ("Peeling",           True), ("Leakage",         True),
            ],
            freq_label="Surface Capture (cyc)", freq_default=25,
            run_label="▶  Run Cosmetic Inspection",
            enabled=True, state="COMPLETE",
        )
        self.mod_cosmetic.sig_camera_request.connect(self._on_camera_request)
        root.addWidget(self.mod_cosmetic)

        # ── LED Performance (C1) ──────────────────────────────────
        self.mod_led = DefectModule(
            "LED PERFORMANCE  (C1)", "led", "C1",
            checks=[
                ("Intensity Drop",  True), ("Color Shift",     True),
                ("Sequence Timing", True),
            ],
            freq_label="LED Light Capture (cyc)", freq_default=25,
            run_label="▶  Run LED Performance Inspection",
            enabled=True, state="PROCESSING…",
        )
        self.mod_led.sig_camera_request.connect(self._on_camera_request)
        root.addWidget(self.mod_led)

        # ── Geometry Deviation (C2) ───────────────────────────────
        self.mod_geometry = DefectModule(
            "GEOMETRY DEVIATION  (C2)", "geometry", "C2",
            checks=[
                ("Height Deviation", False), ("Breakage",    False),
                ("Flatness Error",   False), ("Small Part",  False),
                ("Crack Detection",  False), ("Warping",     False),
            ],
            freq_label="Point Cloud (cyc)", freq_default=50,
            run_label="▶  Run Geometry Deviation Inspection",
            enabled=False, state="IDLE",
        )
        self.mod_geometry.sig_camera_request.connect(self._on_camera_request)
        root.addWidget(self.mod_geometry)

        # ── Visual Controls ───────────────────────────────────────
        self._vc = CollapsibleSection("VISUAL CONTROLS")
        self._build_visual_controls()
        root.addWidget(self._vc)

        # ── AI Analyst ────────────────────────────────────────────
        self._ai_sec = CollapsibleSection("AI ANALYST")
        self._build_ai()
        root.addWidget(self._ai_sec)

        root.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(hdr)
        outer.addWidget(HSep())
        outer.addWidget(scroll)

    # ── Mode switching ─────────────────────────────────────────────
    def _on_mode_changed(self, text: str):
        mode = self.MODE_KEYS.get(text, "cycle_durability")
        self._inspection_mode = mode  # Single source of truth
        self._current_mode = mode
        standalone = (mode == "standalone_visual")
        # Update read-only status label in right panel
        self._mode_status_lbl.setText(text)
        self._baseline_w.setVisible(not standalone)
        for mod in (self.mod_cosmetic, self.mod_led, self.mod_geometry):
            mod.apply_inspection_mode(mode)
        self.sig_mode_changed.emit(mode)

    def current_mode(self) -> str:
        return self._inspection_mode

    def set_baseline_captured(self, ts: str = ""):
        """Called by MainWindow when START is pressed in Cycle Durability mode."""
        t = ts or time.strftime("%H:%M:%S")
        self._baseline_status.setText("✅  Captured")
        self._baseline_status.setStyleSheet(
            f"color:{C['GREEN']};background:transparent;font-size:8pt;font-weight:bold;")
        self._baseline_ts_lbl.setText(f"Captured at: {t}")

    def reset_baseline(self):
        self._baseline_status.setText("⏳  Pending — will auto-capture at START")
        self._baseline_status.setStyleSheet(
            f"color:{C['TEXT_SUB']};background:transparent;font-size:8pt;")
        self._baseline_ts_lbl.setText("")

    def _build_visual_controls(self):
        s = self._vc

        # Camera selection
        s.add(ILabel("CAMERA VIEW", size=8, bold=True, color=C["TEXT_SUB"]))
        cam_row = QHBoxLayout(); cam_row.setSpacing(10)
        self._rbC1    = IRadio("C1",    checked=True)
        self._rbC2    = IRadio("C2",    checked=False)
        self._rbSplit = IRadio("Split", checked=False)
        grp = QButtonGroup(self)
        for rb in (self._rbC1, self._rbC2, self._rbSplit):
            grp.addButton(rb); cam_row.addWidget(rb)
        cam_row.addStretch()
        s.add_layout(cam_row)
        self._rbC1.toggled.connect(   lambda on: self.sig_camera_changed.emit("C1")         if on else None)
        self._rbC2.toggled.connect(   lambda on: self.sig_camera_changed.emit("C2")         if on else None)
        self._rbSplit.toggled.connect(lambda on: self.sig_camera_changed.emit("Split View") if on else None)

        s.add(HSep(color="#333336"))

        # ROI visibility controls
        s.add(ILabel("ROIs", size=8, bold=True, color=C["TEXT_SUB"]))
        roi_row = QHBoxLayout(); roi_row.setSpacing(6)
        for label, slot in [
            ("Hide All", self._hide_all_rois),
            ("Show All", self._show_all_rois),
        ]:
            btn = QPushButton(label)
            btn.setFont(mkfont(8, bold=True)); btn.setFixedHeight(24)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton{{background:{C['PANEL_DARK']};color:{C['TEXT_MED']};
                    border:1px solid {C['BORDER']};border-radius:3px;padding:1px 10px;}}
                QPushButton:hover{{background:{C['BORDER']};color:{C['TEXT']};}}""")
            btn.clicked.connect(slot)
            roi_row.addWidget(btn)
        roi_row.addStretch()
        s.add_layout(roi_row)

        # ROI colour legend
        s.add(HSep(color="#333336"))
        s.add(ILabel("ROI Colours", size=8, bold=True, color=C["TEXT_SUB"]))
        for label, col in [("Cosmetic", "#0A84FF"), ("LED", "#30D158"), ("Geometry", "#FF9F0A")]:
            legend_row = QHBoxLayout(); legend_row.setSpacing(6)
            dot = QLabel("●"); dot.setFont(mkfont(10))
            dot.setStyleSheet(f"color:{col};background:transparent;")
            dot.setFixedWidth(14)
            legend_row.addWidget(dot)
            legend_row.addWidget(ILabel(label, size=8, color=C["TEXT_SUB"]))
            legend_row.addStretch()
            s.add_layout(legend_row)

    def _hide_all_rois(self):
        for mod in (self.mod_cosmetic, self.mod_led, self.mod_geometry):
            mod.roi_mgr.hide_all()

    def _show_all_rois(self):
        for mod in (self.mod_cosmetic, self.mod_led, self.mod_geometry):
            mod.roi_mgr.show_all()

    def _on_camera_request(self, camera: str, msg: str):
        """Auto-switch camera view when user adds ROI in a module."""
        if camera == "C1":
            self._rbC1.setChecked(True)
        elif camera == "C2":
            self._rbC2.setChecked(True)
        self.sig_freq_updated.emit(msg)  # reuse this signal to show status message

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
        """Returns dict of capture frequencies from all three modules."""
        return dict(
            surface_capture_every     = self._safe_int(self.mod_cosmetic.freq_edit.text(),  25),
            led_capture_every         = self._safe_int(self.mod_led.freq_edit.text(),        25),
            point_cloud_capture_every = self._safe_int(self.mod_geometry.freq_edit.text(),  50),
        )

    def _safe_int(self, s, default):
        try: return max(0, int(float(s)))
        except: return default

# ═══════════════════════════════════════════════════════════════════
# BOTTOM STATUS BAR  –  two rows for breathing room
# ═══════════════════════════════════════════════════════════════════
class BottomBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(66)     # taller to fit two rows
        self.setAutoFillBackground(True)
        p = self.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_DARK"])); self.setPalette(p)

        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        outer.addWidget(HSep())

        inner = QWidget(); inner.setStyleSheet("background:transparent;"); outer.addWidget(inner)
        vlay  = QVBoxLayout(inner); vlay.setContentsMargins(14,4,14,4); vlay.setSpacing(2)

        # ── Row 1: state dot + status + cycle ────────────────────
        row1 = QHBoxLayout(); row1.setContentsMargins(0,0,0,0); row1.setSpacing(8)
        self.dotStatus = StatusDot(13)
        self.lblStatus = ILabel("State: STOPPED", size=9, bold=True)
        self.lblStatus.setObjectName("lblStatus")
        row1.addWidget(self.dotStatus)
        row1.addWidget(self.lblStatus)
        row1.addStretch()
        # Alert message lives on row1 right side
        self.lblAlert = ILabel("", size=9, color=C["AMBER"])
        self.lblAlert.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row1.addWidget(self.lblAlert)
        vlay.addLayout(row1)

        # ── Row 2: motion params | baseline | failures ────────────
        row2 = QHBoxLayout(); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
        self.lblParams = ILabel("—", size=8, color=C["TEXT_SUB"])
        row2.addWidget(self.lblParams, stretch=1)
        row2.addWidget(VSep())
        self.lblFailures = ILabel("", size=8, color=C["AMBER"])
        row2.addWidget(self.lblFailures, stretch=1)
        vlay.addLayout(row2)

    def set_status(self, mode, cycle, target, amsg="", aclr=None):
        clrs = dict(RUNNING=C["DOT_RUN"], PAUSED=C["DOT_PAUSE"], STOPPED=C["DOT_STOP"], ERROR=C["DOT_ERR"])
        self.dotStatus.set_color(aclr or clrs.get(mode, C["DOT_STOP"]))
        self.lblStatus.setText(f"State: {mode}   Cycle: {cycle} / {target}")
        if amsg:
            col = aclr or C["AMBER"]
            self.lblAlert.setText(f"⚠  {amsg}")
            self.lblAlert.setStyleSheet(f"color:{col};background:transparent;font-size:9pt;")
        else:
            self.lblAlert.setText("")

    def set_params(self, text): self.lblParams.setText(text)

    def set_failures(self, oor, ret):
        parts = [f"{b}: oor={oor.get(b,0)} ret={ret.get(b,0)}" for b in ("A","B","C","D")]
        self.lblFailures.setText("⚠ " + "  |  ".join(parts))

# ═══════════════════════════════════════════════════════════════════
# MOCK DATA THREAD  –  proper cycle state machine
# ═══════════════════════════════════════════════════════════════════
class MockDataThread(QThread):
    """
    Models a realistic button-press cycle:

    IDLE        ~0.8 s   — small noise around 0, robot moving to position
    PRESS_RAMP  ~0.4 s   — force rises from 0 to peak (1.0–1.6 lbs)
    HOLD        ~0.3 s   — hold near peak with small oscillation
    RELEASE     ~0.4 s   — force drops back to 0
    RETRACT     ~0.5 s   — robot retracts, near-zero force
                ──────
    Total cycle ~ 2.4 s → ~25 cycles/min

    cycle_count only increments ONCE at the END of RETRACT.
    baseline_count increments each cycle while < baseline_cycles.
    baseline_ready becomes True once baseline_count == baseline_cycles.
    """
    sig_data = pyqtSignal(float, float, int)

    # Phase durations (seconds)
    _IDLE    = 0.8
    _RAMP    = 0.4
    _HOLD    = 0.3
    _RELEASE = 0.4
    _RETRACT = 0.5

    def __init__(self, state_ref, lock_ref, parent=None):
        super().__init__(parent)
        self._state = state_ref
        self._lock  = lock_ref
        self._go    = True

    def stop(self): self._go = False; self.wait(3000)

    def run(self):
        t0       = time.time()
        phase    = "IDLE"       # current phase of the cycle
        phase_t  = time.time()  # when this phase started
        peak     = 1.2          # target peak force for current cycle
        dt       = 0.020        # 50 Hz sample rate

        while self._go:
            now = time.time()
            t   = now - t0
            elapsed_in_phase = now - phase_t

            with self._lock:
                running        = self._state.get("running",        False)
                cycle_count    = self._state.get("cycle_count",    0)
                target_cycles  = self._state.get("target_cycles",  100)
                bl_cycles      = self._state.get("baseline_cycles", 30)
                bl_count       = self._state.get("baseline_count", 0)
                bl_ready       = self._state.get("baseline_ready", False)

            if not running:
                # Not running — emit tiny noise and reset phase
                force    = random.gauss(0, 0.003)
                phase    = "IDLE"
                phase_t  = now
                self.sig_data.emit(t, max(-0.05, force), cycle_count)
                time.sleep(dt)
                continue

            # ── Cycle finished? ──────────────────────────────────────
            if cycle_count >= target_cycles:
                with self._lock:
                    self._state["running"] = False
                    self._state["stopped"] = True
                continue

            # ── State machine ────────────────────────────────────────
            noise = random.gauss(0, 0.006)

            if phase == "IDLE":
                force = noise * 0.5
                if elapsed_in_phase >= self._IDLE:
                    phase   = "RAMP"
                    phase_t = now
                    peak    = random.uniform(1.05, 1.55)  # vary each cycle

            elif phase == "RAMP":
                pct   = min(1.0, elapsed_in_phase / self._RAMP)
                # ease-in: pct² for smooth acceleration
                force = peak * pct * pct + noise
                if elapsed_in_phase >= self._RAMP:
                    phase   = "HOLD"
                    phase_t = now

            elif phase == "HOLD":
                force = peak + noise * 2
                if elapsed_in_phase >= self._HOLD:
                    phase   = "RELEASE"
                    phase_t = now

            elif phase == "RELEASE":
                pct   = min(1.0, elapsed_in_phase / self._RELEASE)
                # ease-out: 1 - pct² for smooth deceleration
                force = peak * (1.0 - pct * pct) + noise
                if elapsed_in_phase >= self._RELEASE:
                    phase   = "RETRACT"
                    phase_t = now

            elif phase == "RETRACT":
                force = noise * 0.4
                if elapsed_in_phase >= self._RETRACT:
                    # ── Complete one cycle ───────────────────────────
                    phase   = "IDLE"
                    phase_t = now
                    new_cycle = cycle_count + 1
                    with self._lock:
                        self._state["cycle_count"] = new_cycle
                        # baseline tracking
                        if not bl_ready:
                            new_bl = bl_count + 1
                            self._state["baseline_count"] = new_bl
                            if new_bl >= bl_cycles:
                                self._state["baseline_ready"] = True
                        # simulate occasional force-out-of-range
                        if random.random() < 0.04:
                            btn = random.choice(["A","B","C","D"])
                            self._state["force_out_of_range"][btn] += 1

            else:
                phase = "IDLE"; phase_t = now; force = 0.0

            self.sig_data.emit(t, max(-0.05, force), cycle_count)
            time.sleep(dt)

# ═══════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Durability Intelligence Platform")
        self.setMinimumSize(1280, 820); self.resize(1540, 940)
        self._left_visible = True; self._right_visible = True
        # Track widgets to enable/disable during RUN
        self._widgets_disable_on_run = []

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
        print("║" + "  RoboSpeed Durability Intelligence Platform  v2.7".center(_W-2) + "║")
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
            baseline_mean={},
            baseline_peaks={},
            force_out_of_range=dict(A=0,B=0,C=0,D=0),
            button_did_not_retract=dict(A=0,B=0,C=0,D=0),
            current_button="—", current_phase="idle",
            next_button="A", next_phase="above",
            status_detail="Ready",
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

        # Collect widgets to disable during RUN
        self._widgets_disable_on_run = [
            self.txtProject,
            self.txtTestProfile,
            self._mode_bar_combo,
            self.btnOpenProfile,
            self.btnSave,
            self.right.mod_cosmetic,
            self.right.mod_led,
            self.right.mod_geometry,
        ]

        main_v.addWidget(body, stretch=1)

        sb = self.statusBar()
        sb.setStyleSheet(f"""
            QStatusBar{{background:{C['HEADER_BG']};color:{C['TEXT_SUB']};
                font-size:9pt;border-top:1px solid {C['BORDER']};padding:1px 6px;}}
        """)
        sb.showMessage("RoboSpeed v2.7  |  Durability Intelligence Platform  |  Ready")

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
        # Sync right-panel mode combo → identity bar combo (one-way; bar→panel handled separately)
        self.right.sig_mode_changed.connect(self._on_right_mode_changed)

        # ── Create UI refresh timer ONCE ──────────────────────────
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start(100)

        # ── Create mock data thread ONCE ──────────────────────────
        self._thread = MockDataThread(self._state, self._lock, self)
        self._thread.sig_data.connect(self._on_thread_data)
        self._thread.start()

    def _set_run_mode_enabled(self, enabled: bool):
        # If enabled is False, we are entering RUN mode (disable widgets)
        # If enabled is True, we are leaving RUN mode (enable widgets)
        for w in self._widgets_disable_on_run:
            if isinstance(w, DefectModule):
                # DefectModule: when disabling for RUN, always dim content.
                # When re-enabling after RUN, restore to module's OWN state.
                if enabled:
                    w._apply_content_state(w._enabled)
                else:
                    w._content_w.setEnabled(False)
                    w._opacity_fx.setOpacity(0.35)
            elif hasattr(w, '_content_w'):
                w._content_w.setEnabled(enabled)
                if hasattr(w, '_opacity_fx'):
                    w._opacity_fx.setOpacity(1.0 if enabled else 0.35)
            else:
                w.setEnabled(enabled)

    def _build_identity_bar(self):
        bar = QWidget(); bar.setFixedHeight(48)
        bar.setAutoFillBackground(True)
        p = bar.palette(); p.setColor(QPalette.ColorRole.Window, QColor(C["PANEL_DARK"])); bar.setPalette(p)
        h = QHBoxLayout(bar); h.setContentsMargins(12,6,12,6); h.setSpacing(12)

        title_lbl = ILabel("Durability Intelligence Platform", size=12, bold=True, color=C["ACCENT"])
        title_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        h.addWidget(title_lbl)
        
        # Vertical separator after title
        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"background:{C['BORDER']};border:none;"); sep1.setFixedWidth(1); sep1.setFixedHeight(28)
        h.addWidget(sep1)
        h.addSpacing(4)

        h.addWidget(ILabel("Project:", size=9, bold=True, color=C["TEXT_MED"]))
        self.txtProject = IEdit("Button Toy", "", align_right=False)
        self.txtProject.setObjectName("txtProject"); self.txtProject.setFixedWidth(120)
        h.addWidget(self.txtProject)

        h.addWidget(ILabel("|", size=11, color=C["BORDER"]))

        h.addWidget(ILabel("Test Profile:", size=9, bold=True, color=C["TEXT_MED"]))
        self.txtTestProfile = IEdit("1.5lb Cycle Test", "", align_right=False)
        self.txtTestProfile.setObjectName("txtTestProfile"); self.txtTestProfile.setFixedWidth(140)
        h.addWidget(self.txtTestProfile)

        h.addWidget(ILabel("|", size=11, color=C["BORDER"]))

        h.addWidget(ILabel("Mode:", size=9, bold=True, color=C["TEXT_MED"]))
        self._mode_bar_combo = QComboBox()
        self._mode_bar_combo.addItems(["Cycle Durability", "Standalone Visual"])
        self._mode_bar_combo.setFont(mkfont(9, bold=True))
        self._mode_bar_combo.setFixedHeight(28); self._mode_bar_combo.setFixedWidth(180)
        self._mode_bar_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_bar_combo.setStyleSheet(f"""
            QComboBox{{background:{C['PANEL']};color:{C['ACCENT']};
                border:1.5px solid {C['ACCENT']};border-radius:4px;
                padding:2px 8px;font-weight:bold;font-size:9pt;}}
            QComboBox:hover{{border:1.5px solid {C['ACCENT_LT']};}}
            QComboBox::drop-down{{border:none;width:20px;}}
            QComboBox QAbstractItemView{{
                background:{C['PANEL_DARK']};color:{C['TEXT']};
                selection-background-color:{C['ACCENT']};
                border:1px solid {C['BORDER']};}}""")
        # Keep identity bar combo synced with right panel combo
        self._mode_bar_combo.currentTextChanged.connect(self._on_mode_bar_changed)
        h.addWidget(self._mode_bar_combo)

        h.addSpacing(8)
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"background:{C['BORDER']};border:none;"); sep2.setFixedWidth(1); sep2.setFixedHeight(28)
        h.addWidget(sep2)
        h.addSpacing(8)

        _btn_bar_style = f"""
            QPushButton{{background:transparent;color:{C['ACCENT']};
                border:1px solid {C['ACCENT']};border-radius:4px;
                padding:2px 8px;font-weight:bold;font-size:9pt;}}
            QPushButton:hover{{background:{C['ACCENT']};color:#fff;}}
            QPushButton:pressed{{background:{C['ACCENT_LT']};color:#fff;}}
        """

        # ── Open Profile button ───────────────────────────────────
        self.btnOpenProfile = QPushButton("Open Profile")
        self.btnOpenProfile.setFont(mkfont(9, bold=True))
        self.btnOpenProfile.setFixedHeight(28)
        self.btnOpenProfile.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnOpenProfile.setStyleSheet(_btn_bar_style)
        self.btnOpenProfile.clicked.connect(self._on_open_profile)
        h.addWidget(self.btnOpenProfile)

        # ── Save Profile button ───────────────────────────────────
        self.btnSave = QPushButton("Save Profile")
        self.btnSave.setFont(mkfont(9, bold=True))
        self.btnSave.setFixedHeight(28)
        self.btnSave.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btnSave.setStyleSheet(_btn_bar_style)
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

    def _on_thread_data(self, t, f, c):
        """Slot for MockDataThread.sig_data — push sample to force graph."""
        self.force_graph.push(t, f, c)

    def _refresh_ui(self):
        with self._lock: st = dict(self._state)
        mode = "RUNNING" if st["running"] else "PAUSED" if st["paused"] else "STOPPED"
        now  = time.time()
        amsg = st["alert_msg"]   if now <= st["alert_until"] else ""
        aclr = st["alert_color"] if now <= st["alert_until"] else None
        phase_txt = f"{st.get('current_button', '—')}-{st.get('current_phase', 'idle')}"
        status_msg = amsg or st.get("status_detail", "")
        if status_msg:
            status_msg = f"{phase_txt} · {status_msg}"
        else:
            status_msg = phase_txt
        self.bottom.set_status(mode, st["cycle_count"], st["target_cycles"], status_msg, aclr)

        # Correct baseline/anomaly label logic
        bl_ready = st["baseline_ready"]
        bl_count = st["baseline_count"]
        bl_max   = st["baseline_cycles"]
        self.force_graph.set_baseline_label(bl_ready, bl_count, bl_max)

        # Bottom bar params: show baseline progress until ready, then anomaly detection
        if bl_ready:
            bl_txt = "Detecting Anomalies"
        else:
            bl_txt = f"Baseline {bl_count}/{bl_max}"
        next_txt = f"Next:{st.get('next_button', 'A')}-{st.get('next_phase', 'above')}"
        base_mean = st.get("baseline_mean", {})
        base_parts = [f"{b}:{base_mean[b]:.2f}" for b in ("A","B","C","D") if b in base_mean]
        base_detail = f"  |  Mean {' / '.join(base_parts)}" if base_parts else ""
        self.bottom.set_params(
            f"Vel:{st['vel']}  Acc:{st['acc']}  Jerk:{st['jerk']}  |  {bl_txt}  |  {next_txt}{base_detail}"
        )
        self.bottom.set_failures(st["force_out_of_range"], st["button_did_not_retract"])
        self.force_graph.set_band(st["force_min"], st["force_max"])
        tc = max(1, st["target_cycles"])
        self.force_graph.set_progress(min(1.0, st["cycle_count"] / tc))
        t_now = time.time() - self._t0
        while self._peak_events and t_now - self._peak_events[0].get("t", 0.0) > 10.0:
            self._peak_events.popleft()
        self.force_graph.set_peaks(list(self._peak_events))

    def on_start(self):
        with self._lock: self._state.update(running=True, paused=False, stopped=False)
        self._alert(C["GREEN"], "Test started", 2.0)
        self.statusBar().showMessage("● Running…")
        # In Cycle Durability mode, auto-capture baseline on START
        if self.right.current_mode() == "cycle_durability":
            self.right.set_baseline_captured()
        # Disable top bar and defect modules, keep Visual/AI/Motion enabled
        self._set_run_mode_enabled(False)

    def _on_mode_bar_changed(self, text: str):
        """Identity-bar mode combo changed → update right panel."""
        self.right._on_mode_changed(text)
        mode_key = self.right.MODE_KEYS.get(text, "cycle_durability")
        self.statusBar().showMessage(
            f"Mode: {text}  |  "
            f"{'Golden Reference active' if mode_key=='standalone_visual' else 'Cycle Durability active'}",
            3000)

    def _on_right_mode_changed(self, mode_key: str):
        """Right-panel mode changed → sync identity-bar combo."""
        label = {v: k for k, v in self.right.MODE_KEYS.items()}.get(mode_key, "Cycle Durability")
        self._mode_bar_combo.blockSignals(True)
        self._mode_bar_combo.setCurrentText(label)
        self._mode_bar_combo.blockSignals(False)

    def on_pause(self):
        with self._lock: self._state.update(running=False, paused=True)
        self._alert(C["AMBER"], "Paused", 2.0); self.statusBar().showMessage("⏸ Paused")

    def on_stop(self):
        with self._lock: self._state.update(running=False, paused=False, stopped=True)
        self._alert(C["DOT_STOP"], "Stopped", 2.0)
        self.statusBar().showMessage("■ Stopped  –  Ready")
        self.right.reset_baseline()
        self._set_run_mode_enabled(True)

    def on_home(self):
        self._alert(C["ACCENT"], "Homing robot…", 2.0); self.statusBar().showMessage("Homing robot arm…")
        self._set_run_mode_enabled(True)

    def on_reset(self):
        with self._lock:
            self._state.update(
                force_out_of_range=dict(A=0,B=0,C=0,D=0),
                button_did_not_retract=dict(A=0,B=0,C=0,D=0),
                cycle_count=0, baseline_ready=False, baseline_count=0,
                baseline_mean={}, baseline_peaks={},
                current_button="—", current_phase="idle",
                next_button="A", next_phase="above",
                status_detail="Ready",
            )
        self._peak_events.clear()
        self.right.reset_baseline()
        self._alert(C["GREEN"], "Counters reset", 2.0); self.statusBar().showMessage("Reset complete")
        self._set_run_mode_enabled(True)

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

    # ─────────────────────────────────────────────────────────────
    # PROFILE  –  collect / apply all settings
    # ─────────────────────────────────────────────────────────────
    def _collect_profile(self) -> dict:
        """Gather every user-editable setting into a dict for JSON serialisation."""
        lf = self.left.get_fields()

        def _roi_list(mod):
            """Snapshot each ROI: index, name, shape, locked, hidden, + size placeholder."""
            return [
                dict(
                    index  = r.index,
                    name   = r.name,
                    shape  = r.shape,
                    locked = r.locked,
                    hidden = r.hidden,
                    # Size / coords placeholder — populated by backend after real draw
                    coords = {"x": 0, "y": 0, "w": 0, "h": 0},
                )
                for r in mod.roi_mgr.rois
            ]

        def _mod_state(mod):
            return dict(
                enabled   = mod._enabled,
                checks    = {k: cb.isChecked() for k, cb in mod._checks.items()},
                frequency = mod.freq_edit.text(),
                rois      = _roi_list(mod),
            )

        cam = "C1"
        if self.right._rbC2.isChecked():      cam = "C2"
        elif self.right._rbSplit.isChecked(): cam = "Split View"

        return dict(
            version         = "2.8",
            project         = self.txtProject.text(),
            test_profile    = self.txtTestProfile.text(),
            inspection_mode = self.right.current_mode(),
            motion          = lf,
            cosmetic        = _mod_state(self.right.mod_cosmetic),
            led             = _mod_state(self.right.mod_led),
            geometry        = _mod_state(self.right.mod_geometry),
            camera          = cam,
        )

    def _apply_profile(self, data: dict):
        """Apply a loaded profile dict to every widget."""
        if "project"      in data: self.txtProject.setText(str(data["project"]))
        if "test_profile" in data: self.txtTestProfile.setText(str(data["test_profile"]))

        # Restore inspection mode
        mode = data.get("inspection_mode", "cycle_durability")
        label = {v: k for k, v in self.right.MODE_KEYS.items()}.get(mode, "Cycle Durability")
        self._mode_bar_combo.blockSignals(True)
        self._mode_bar_combo.setCurrentText(label)
        self._mode_bar_combo.blockSignals(False)
        self.right._on_mode_changed(label)

        # Motion fields
        motion = data.get("motion", {})
        for key, edit in self.left._fields.items():
            if key in motion:
                edit.setText(str(motion[key]))
        self.left._emit_fields()

        def _load_mod(mod, mdata):
            if not mdata: return
            en = bool(mdata.get("enabled", True))
            if mod._enabled != en:
                mod._enabled = en
                mod._enable_btn.setText("⬤  Enabled" if en else "◯  Disabled")
                mod._update_enable_style()
                mod._apply_content_state(en)
            for label, checked in mdata.get("checks", {}).items():
                if label in mod._checks:
                    mod._checks[label].setChecked(bool(checked))
            if "frequency" in mdata:
                mod.freq_edit.setText(str(mdata["frequency"]))
            # Restore ROIs
            roi_mgr = mod.roi_mgr
            roi_mgr._rois.clear()
            roi_mgr._rebuild_list()
            for r in mdata.get("rois", []):
                item = type('ROIItem', (), {
                    'index':  r.get("index", 1),
                    'name':   r.get("name",  f"ROI {r.get('index',1)}"),
                    'shape':  r.get("shape", "Rectangle"),
                    'locked': r.get("locked", False),
                    'hidden': r.get("hidden", False),
                })()
                roi_mgr._rois.append(item)
            if roi_mgr._rois:
                roi_mgr._rebuild_list()

        _load_mod(self.right.mod_cosmetic, data.get("cosmetic"))
        _load_mod(self.right.mod_led,      data.get("led"))
        _load_mod(self.right.mod_geometry, data.get("geometry"))

        cam = data.get("camera", "C1")
        if cam == "C2":           self.right._rbC2.setChecked(True)
        elif cam == "Split View": self.right._rbSplit.setChecked(True)
        else:                     self.right._rbC1.setChecked(True)

    def _on_save(self):
        """Save all settings to a .rsprofile text file (JSON under the hood)."""
        default_name = (
            f"{self.txtProject.text().replace(' ','_')}_"
            f"{self.txtTestProfile.text().replace(' ','_')}.rsprofile"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", default_name,
            "RoboSpeed Profile (*.rsprofile);;Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return   # user cancelled

        profile = self._collect_profile()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# RoboSpeed Profile  v2.8\n")
                f.write(f"# Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(json.dumps(profile, indent=2))
            self._alert(C["GREEN"], "Profile saved", 2.0)
            self.statusBar().showMessage(f"Profile saved → {os.path.basename(path)}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save profile:\n{e}")

    def _on_open_profile(self):
        """Open a .rsprofile file and load all settings back into the GUI."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile", "",
            "RoboSpeed Profile (*.rsprofile);;Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return   # user cancelled

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Strip comment lines that start with #
            json_text = "\n".join(
                line for line in raw.splitlines()
                if not line.strip().startswith("#")
            )
            data = json.loads(json_text)
            self._apply_profile(data)
            self._alert(C["ACCENT"], "Profile loaded", 2.0)
            self.statusBar().showMessage(f"Profile loaded ← {os.path.basename(path)}", 4000)
        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Load Error", f"File is not a valid profile:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load profile:\n{e}")

    def _alert(self, color, msg, dur=1.5):
        with self._lock:
            self._state["alert_color"] = color
            self._state["alert_msg"]   = msg
            self._state["alert_until"] = time.time() + dur

    def closeEvent(self, e):
        self._ui_timer.stop()
        self._thread.stop()
        e.accept()

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
