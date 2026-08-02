#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI to visualize MIDAS integrator lineout data (AM dataset format) from HDF5 files.

by AC (cchuang@anl.gov)
"""

import sys
import os
import functools
import h5py
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLineEdit,
                             QLabel, QFileDialog, QMessageBox, QCheckBox,
                             QComboBox, QDialog, QTabWidget)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if args and hasattr(args[0], 'show_error_message'):
                self = args[0]
                self.show_error_message('Error', str(e))
                if hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(f'Error: {e}')
            else:
                print(f"Error in {func.__name__}: {e}")
        return None
    return wrapper


def parse_frame_selection(text, n_frames):
    """Parse numpy-style index string into an array of frame indices.

    Supports: '0:10', '0:10:2', '0,5,10', '0:5,10,20:30'
    """
    text = text.strip()
    if not text:
        return np.array([], dtype=int)
    if text.lower() == 'all':
        return np.arange(n_frames)

    parts = text.split(',')
    slices = []
    for part in parts:
        part = part.strip()
        if ':' in part:
            tokens = part.split(':')
            args = [int(t) if t else None for t in tokens]
            slices.append(slice(*args))
        else:
            slices.append(int(part))

    indices = np.r_[tuple(slices)]
    indices = indices[(indices >= 0) & (indices < n_frames)]
    return indices


# ============================================================================
# 1D Lineout Window (canvas only — controls live in main panel)
# ============================================================================

class Lineout1DWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.x_axes = {}
        self.indices = np.array([0])
        self.frame_names = None
        self.dataset_name = 'lineouts'
        self.multi_mode = False
        self.avg_mode = False
        self.plot_opts = {}
        self.line_style = {}
        self._prev_xaxis_key = None
        self._programmatic_update = False
        self.setWindowTitle('1D Lineout')
        self.setGeometry(120, 120, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

    def set_data(self, data, x_axes, indices, frame_names, dataset_name,
                 multi_mode, avg_mode, plot_opts):
        self.data = data
        self.x_axes = x_axes
        self.indices = indices
        self.frame_names = frame_names
        self.dataset_name = dataset_name
        self.multi_mode = multi_mode
        self.avg_mode = avg_mode
        self.plot_opts = plot_opts
        self.update_plot()

    def _get_frame_label(self, idx):
        if self.frame_names is not None and idx < len(self.frame_names):
            return f'{idx}: {self.frame_names[idx]}'
        return f'Frame {idx}'

    def show_error_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    @handle_errors
    def update_plot(self, *_args):
        if self.data is None or len(self.indices) == 0:
            return

        opts = self.plot_opts

        # Save current axis limits and line style before clearing
        prev_xlim = prev_ylim = None
        if self.figure.axes:
            ax_prev = self.figure.axes[0]
            prev_xlim = ax_prev.get_xlim()
            prev_ylim = ax_prev.get_ylim()
            lines = [l for l in ax_prev.get_lines() if not l.get_label().startswith('_')]
            if lines:
                l = lines[0]
                self.line_style = {
                    'linewidth': l.get_linewidth(),
                    'linestyle': l.get_linestyle(),
                    'marker': l.get_marker(),
                    'markersize': l.get_markersize(),
                    'markeredgecolor': l.get_markeredgecolor(),
                    'markerfacecolor': l.get_markerfacecolor(),
                }

        self._programmatic_update = True
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)

        style = self.line_style.copy()
        if not style:
            style = {'linewidth': 0.8, 'linestyle': '-', 'markersize': 5}
            if not self.multi_mode:
                style['markeredgecolor'] = '#ff0000ff'
                style['markerfacecolor'] = '#ffffff00'
        elif self.multi_mode:
            style.pop('markeredgecolor', None)
            style.pop('markerfacecolor', None)
        style['marker'] = opts.get('marker', 'o')

        xaxis_key = opts.get('xaxis_key', 'R (pixel)')
        x_values = self.x_axes.get(xaxis_key, self.x_axes.get('R (pixel)'))

        if self.avg_mode and len(self.indices) > 1:
            avg = np.mean(self.data[self.indices], axis=0)
            ax.plot(x_values, avg, label=f'Avg ({len(self.indices)} frames)', **style)
        else:
            for idx in self.indices:
                ax.plot(x_values, self.data[idx], label=self._get_frame_label(idx), **style)

        ax.set_xlabel(xaxis_key)
        ax.set_ylabel('Intensity')
        if len(self.indices) == 1:
            ax.set_title(f'Frame: {self.indices[0]}')
        else:
            ax.set_title(f'Frames: {self.indices[0]}–{self.indices[-1]} ({len(self.indices)} frames)')

        if opts.get('log_y', False):
            ax.set_yscale('log')

        ax.grid(True, linestyle='--', alpha=0.5)

        # X-axis: manual range takes priority, then preserve previous
        xaxis_changed = xaxis_key != self._prev_xaxis_key
        self._prev_xaxis_key = xaxis_key
        if not opts.get('auto_x', True):
            ax.set_xlim(opts.get('xmin'), opts.get('xmax'))
        elif prev_xlim and not xaxis_changed:
            ax.set_xlim(prev_xlim)

        # Y-axis: manual range takes priority, then auto
        if not opts.get('auto_range', True):
            ax.set_ylim(opts.get('ymin'), opts.get('ymax'))

        if opts.get('legend', True) and len(self.indices) <= 30:
            ax.legend(fontsize='small')

        self.figure.tight_layout()
        self.canvas.draw()
        self._programmatic_update = False

    def _on_xlim_changed(self, ax):
        if self._programmatic_update:
            return
        xmin, xmax = ax.get_xlim()
        parent = self.parent()
        if parent is None:
            return
        parent._1d_auto_x.blockSignals(True)
        parent._1d_auto_x.setChecked(False)
        parent._1d_xmin.setEnabled(True)
        parent._1d_xmax.setEnabled(True)
        parent._1d_auto_x.blockSignals(False)
        parent._1d_xmin.setText(f'{xmin:.6g}')
        parent._1d_xmax.setText(f'{xmax:.6g}')

    def _on_ylim_changed(self, ax):
        if self._programmatic_update:
            return
        ymin, ymax = ax.get_ylim()
        parent = self.parent()
        if parent is None:
            return
        parent._1d_auto_range.blockSignals(True)
        parent._1d_auto_range.setChecked(False)
        parent._1d_ymin.setEnabled(True)
        parent._1d_ymax.setEnabled(True)
        parent._1d_auto_range.blockSignals(False)
        parent._1d_ymin.setText(f'{ymin:.6g}')
        parent._1d_ymax.setText(f'{ymax:.6g}')


# ============================================================================
# 2D Lineout Window (canvas only — controls live in main panel)
# ============================================================================

class Lineout2DWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.x_axes = {}
        self.dataset_name = 'lineouts'
        self.plot_opts = {}
        self._prev_xaxis_key = None
        self._programmatic_update = False
        self.setWindowTitle('2D Lineout')
        self.setGeometry(150, 150, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

    def set_data(self, data, x_axes, dataset_name, plot_opts):
        self.data = data
        self.x_axes = x_axes
        self.dataset_name = dataset_name
        self.plot_opts = plot_opts
        self.setWindowTitle(f'2D Lineout — {dataset_name}')
        self.update_plot()

    def update_plot(self, *_args):
        if self.data is None:
            return

        opts = self.plot_opts
        display = self.data.copy().astype(np.float64)

        if opts.get('log_scale', False):
            min_pos = np.min(display[display > 0]) if np.any(display > 0) else 1
            display = np.where(display > 0, display, min_pos)
            display = np.log10(display)
            label = 'log₁₀(Intensity)'
        else:
            label = 'Intensity'

        xaxis_key = opts.get('xaxis_key', 'R (pixel)')
        x_values = self.x_axes.get(xaxis_key, self.x_axes.get('R (pixel)'))
        y_values = np.arange(self.data.shape[0])

            # Save limits before clearing
        prev_xlim = prev_ylim = None
        if self.figure.axes:
            prev_xlim = self.figure.axes[0].get_xlim()
            prev_ylim = self.figure.axes[0].get_ylim()

        self._programmatic_update = True
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)
        cmap = opts.get('cmap', 'viridis')
        equally_spaced = len(x_values) > 1 and np.allclose(np.diff(x_values), np.diff(x_values)[0])

        if equally_spaced:
            extent = [x_values[0], x_values[-1], self.data.shape[0] - 0.5, -0.5]
            im = ax.imshow(display, aspect='auto', cmap=cmap,
                           extent=extent, interpolation='nearest')
        else:
            im = ax.pcolormesh(x_values, y_values, display,
                               cmap=cmap, shading='nearest')
            ax.set_ylim(self.data.shape[0] - 0.5, -0.5)

        if not opts.get('auto_int', True):
            vmin = opts.get('imin')
            vmax = opts.get('imax')
            if opts.get('log_scale', False):
                if vmin is not None and vmin > 0:
                    vmin = np.log10(vmin)
                if vmax is not None and vmax > 0:
                    vmax = np.log10(vmax)
            im.set_clim(vmin, vmax)

        ax.set_xlabel(xaxis_key)
        ax.set_ylabel('Frame Seq.')
        ax.set_title(f'{self.dataset_name} — {self.data.shape[0]} frames')
        self.figure.colorbar(im, ax=ax, label=label)

        # X-axis: manual range takes priority, then preserve previous
        xaxis_changed = xaxis_key != self._prev_xaxis_key
        self._prev_xaxis_key = xaxis_key
        if not opts.get('auto_x', True):
            ax.set_xlim(opts.get('xmin'), opts.get('xmax'))
        elif prev_xlim and not xaxis_changed:
            ax.set_xlim(prev_xlim)

        # Y-axis: always inverted (frame 0 at top)
        if not opts.get('auto_y', True):
            y1 = opts.get('ymin')
            y2 = opts.get('ymax')
            if y1 is not None and y2 is not None:
                ax.set_ylim(max(y1, y2), min(y1, y2))
            elif prev_ylim:
                ax.set_ylim(prev_ylim)
            else:
                ax.set_ylim(self.data.shape[0] - 0.5, -0.5)
        else:
            ax.set_ylim(self.data.shape[0] - 0.5, -0.5)

        self.figure.tight_layout()
        self.canvas.draw()
        self._programmatic_update = False

    def _on_xlim_changed(self, ax):
        if self._programmatic_update:
            return
        xmin, xmax = ax.get_xlim()
        parent = self.parent()
        if parent is None:
            return
        parent._2d_auto_x.blockSignals(True)
        parent._2d_auto_x.setChecked(False)
        parent._2d_xmin.setEnabled(True)
        parent._2d_xmax.setEnabled(True)
        parent._2d_auto_x.blockSignals(False)
        parent._2d_xmin.setText(f'{xmin:.6g}')
        parent._2d_xmax.setText(f'{xmax:.6g}')

    def _on_ylim_changed(self, ax):
        if self._programmatic_update:
            return
        y1, y2 = ax.get_ylim()
        parent = self.parent()
        if parent is None:
            return
        parent._2d_auto_y.blockSignals(True)
        parent._2d_auto_y.setChecked(False)
        parent._2d_ymin.setEnabled(True)
        parent._2d_ymax.setEnabled(True)
        parent._2d_auto_y.blockSignals(False)
        parent._2d_ymin.setText(str(int(round(min(y1, y2)))))
        parent._2d_ymax.setText(str(int(round(max(y1, y2)))))


# ============================================================================
# OmegaSumFrame Window (canvas only — controls live in main panel)
# ============================================================================

class OmegaSumWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_2d = None
        self.x_axes = {}
        self.eta_axis = None
        self.frame_idx = 0
        self.plot_opts = {}
        self._prev_xaxis_key = None
        self._programmatic_update = False
        self.setWindowTitle('OmegaSumFrame')
        self.setGeometry(180, 180, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

    def set_data(self, data_2d, x_axes, eta_axis, frame_idx, plot_opts):
        self.data_2d = data_2d
        self.x_axes = x_axes
        self.eta_axis = eta_axis
        self.frame_idx = frame_idx
        self.plot_opts = plot_opts
        self.setWindowTitle(f'OmegaSumFrame — Frame {frame_idx}')
        self.update_plot()

    def update_plot(self, *_args):
        if self.data_2d is None:
            return

        opts = self.plot_opts
        display = self.data_2d.copy().astype(np.float64)

        if opts.get('log_scale', False):
            min_pos = np.min(display[display > 0]) if np.any(display > 0) else 1
            display = np.where(display > 0, display, min_pos)
            display = np.log10(display)
            label = 'log₁₀(Intensity)'
        else:
            label = 'Intensity'

        xaxis_key = opts.get('xaxis_key', 'R (pixel)')
        x_values = self.x_axes.get(xaxis_key, self.x_axes.get('R (pixel)'))

        prev_xlim = prev_ylim = None
        if self.figure.axes:
            prev_xlim = self.figure.axes[0].get_xlim()
            prev_ylim = self.figure.axes[0].get_ylim()

        self._programmatic_update = True
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)

        cmap = opts.get('cmap', 'viridis')
        x_eq = len(x_values) > 1 and np.allclose(np.diff(x_values), np.diff(x_values)[0])
        y_eq = len(self.eta_axis) > 1 and np.allclose(np.diff(self.eta_axis), np.diff(self.eta_axis)[0])

        if x_eq and y_eq:
            extent = [x_values[0], x_values[-1], self.eta_axis[0], self.eta_axis[-1]]
            im = ax.imshow(display.T, aspect='auto', cmap=cmap,
                           extent=extent, origin='lower', interpolation='nearest')
        else:
            im = ax.pcolormesh(x_values, self.eta_axis, display.T,
                               cmap=cmap, shading='nearest')

        if not opts.get('auto_int', True):
            vmin = opts.get('imin')
            vmax = opts.get('imax')
            if opts.get('log_scale', False):
                if vmin is not None and vmin > 0:
                    vmin = np.log10(vmin)
                if vmax is not None and vmax > 0:
                    vmax = np.log10(vmax)
            im.set_clim(vmin, vmax)

        ax.set_xlabel(xaxis_key)
        ax.set_ylabel('Eta (deg)')
        ax.set_title(f'OmegaSumFrame — Frame {self.frame_idx}')
        self.figure.colorbar(im, ax=ax, label=label)

        # X-axis
        xaxis_changed = xaxis_key != self._prev_xaxis_key
        self._prev_xaxis_key = xaxis_key
        if not opts.get('auto_x', True):
            ax.set_xlim(opts.get('xmin'), opts.get('xmax'))
        elif prev_xlim and not xaxis_changed:
            ax.set_xlim(prev_xlim)

        # Y-axis (eta)
        if not opts.get('auto_y', True):
            ax.set_ylim(opts.get('ymin'), opts.get('ymax'))
        elif prev_ylim:
            ax.set_ylim(prev_ylim)

        self.figure.tight_layout()
        self.canvas.draw()
        self._programmatic_update = False

    def _on_xlim_changed(self, ax):
        if self._programmatic_update:
            return
        xmin, xmax = ax.get_xlim()
        parent = self.parent()
        if parent is None:
            return
        parent._om_auto_x.blockSignals(True)
        parent._om_auto_x.setChecked(False)
        parent._om_xmin.setEnabled(True)
        parent._om_xmax.setEnabled(True)
        parent._om_auto_x.blockSignals(False)
        parent._om_xmin.setText(f'{xmin:.6g}')
        parent._om_xmax.setText(f'{xmax:.6g}')

    def _on_ylim_changed(self, ax):
        if self._programmatic_update:
            return
        ymin, ymax = ax.get_ylim()
        parent = self.parent()
        if parent is None:
            return
        parent._om_auto_y.blockSignals(True)
        parent._om_auto_y.setChecked(False)
        parent._om_ymin.setEnabled(True)
        parent._om_ymax.setEnabled(True)
        parent._om_auto_y.blockSignals(False)
        parent._om_ymin.setText(f'{ymin:.6g}')
        parent._om_ymax.setText(f'{ymax:.6g}')


PARAM_LABELS = ['Amplitude', 'Center', 'Width', 'Eta']
FILTER_PARAMS = ['Amplitude', 'Center', 'Width', 'Eta', 'Chi²',
                 'Amplitude Err', 'Center Err', 'Width Err', 'Eta Err']


class MaskFilterDialog(QDialog):

    def __init__(self, on_apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Mask Filter')
        self._on_apply = on_apply
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel('Enable'), 0, 0)
        grid.addWidget(QLabel('Parameter'), 0, 1)
        grid.addWidget(QLabel('Min'), 0, 2)
        grid.addWidget(QLabel('Max'), 0, 3)

        self._rows = []
        for i, name in enumerate(FILTER_PARAMS):
            cb = QCheckBox()
            lbl = QLabel(name)
            fmin = QLineEdit()
            fmax = QLineEdit()
            fmin.setMaximumWidth(80)
            fmax.setMaximumWidth(80)
            fmin.setPlaceholderText('—')
            fmax.setPlaceholderText('—')
            grid.addWidget(cb, i + 1, 0)
            grid.addWidget(lbl, i + 1, 1)
            grid.addWidget(fmin, i + 1, 2)
            grid.addWidget(fmax, i + 1, 3)
            self._rows.append((cb, fmin, fmax))

        layout.addLayout(grid)

        self._show_masked = QCheckBox('Show masked points (gray)')
        self._show_masked.setChecked(True)
        layout.addWidget(self._show_masked)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton('Apply')
        reset_btn = QPushButton('Reset')
        apply_btn.clicked.connect(self._apply)
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

    def _apply(self):
        self._on_apply()

    def _reset(self):
        for cb, fmin, fmax in self._rows:
            cb.setChecked(False)
            fmin.clear()
            fmax.clear()
        self._apply()

    def get_filters(self):
        filters = []
        for i, (cb, fmin, fmax) in enumerate(self._rows):
            if not cb.isChecked():
                continue
            vmin = vmax = None
            try:
                if fmin.text().strip():
                    vmin = float(fmin.text())
            except ValueError:
                pass
            try:
                if fmax.text().strip():
                    vmax = float(fmax.text())
            except ValueError:
                pass
            if vmin is not None or vmax is not None:
                filters.append({'param_idx': i, 'min': vmin, 'max': vmax})
        return {'filters': filters, 'show_masked': self._show_masked.isChecked()}


class LineoutFitWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit_data = {}
        self.plot_opts = {}
        self._programmatic_update = False
        self._prev_phase = None
        self._prev_peak = None
        self._prev_param = None
        self._prev_norm = False
        self.setWindowTitle('Lineout Fit Results')
        self.setGeometry(220, 220, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

    def set_data(self, fit_data, plot_opts):
        self.fit_data = fit_data
        self.plot_opts = plot_opts
        self.update_plot()

    def update_plot(self, *_args):
        if not self.fit_data:
            return

        opts = self.plot_opts
        phase = opts.get('phase', '')
        param_idx = opts.get('param_idx', 1)
        peak_sel = opts.get('peak', 'All')

        phase_data = self.fit_data.get(phase)
        if phase_data is None:
            return

        params = phase_data['params']
        n_frames = params.shape[0]
        n_peaks = params.shape[1]
        frame_numbers = np.arange(n_frames)

        if peak_sel.lower() == 'all' or peak_sel == '':
            peak_indices = list(range(n_peaks))
        else:
            try:
                peak_indices = []
                for tok in peak_sel.replace(',', ' ').split():
                    if '-' in tok:
                        a, b = tok.split('-', 1)
                        peak_indices.extend(range(int(a), int(b) + 1))
                    else:
                        peak_indices.append(int(tok))
                peak_indices = [p for p in peak_indices if 0 <= p < n_peaks]
            except ValueError:
                peak_indices = list(range(n_peaks))

        prev_xlim = prev_ylim = None
        if self.figure.axes:
            prev_xlim = self.figure.axes[0].get_xlim()
            prev_ylim = self.figure.axes[0].get_ylim()

        self._programmatic_update = True
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
        ax.callbacks.connect('ylim_changed', self._on_ylim_changed)

        if param_idx < 4:
            param_name = PARAM_LABELS[param_idx]
        elif param_idx == 4:
            param_name = 'Chi²'
        else:
            param_name = PARAM_LABELS[param_idx - 5] + ' Err'
        norm_enabled = opts.get('norm_enabled', False) and param_idx == 1
        norm_mode = opts.get('norm_mode', 'Mean')
        norm_values = opts.get('norm_values', [])
        norm_scaling = opts.get('norm_scaling', 1e6)
        mask_filters = opts.get('filters', [])
        show_masked = opts.get('show_masked', True)

        unit = f'x{norm_scaling:.3g}' if norm_enabled else opts.get('radial_unit', '')
        print(f"--- {phase} | {param_name} (lineout) ---")
        print(f"  peakID: Mean / Median / Std ({unit})")
        for i, pi in enumerate(peak_indices):
            if param_idx < 4:
                values = params[:, pi, param_idx].copy()
            elif param_idx == 4:
                values = phase_data['chi2'][:, pi].copy()
            else:
                values = phase_data['errors'][:, pi, param_idx - 5].copy()
            valid = ~np.isnan(values)

            for flt in mask_filters:
                fi = flt['param_idx']
                if fi < 4:
                    fvals = params[:, pi, fi]
                elif fi == 4:
                    fvals = phase_data['chi2'][:, pi]
                else:
                    fvals = phase_data['errors'][:, pi, fi - 5]
                if flt['min'] is not None:
                    valid &= fvals >= flt['min']
                if flt['max'] is not None:
                    valid &= fvals <= flt['max']
            user_masked = ~valid

            if norm_enabled:
                if norm_mode == 'Mean':
                    ref = np.mean(values[valid])
                elif norm_mode == 'Median':
                    ref = np.median(values[valid])
                else:
                    ref = norm_values[i] if i < len(norm_values) else 0
                if ref != 0:
                    values[valid] = (values[valid] - ref) / ref * norm_scaling
                    values[user_masked] = (values[user_masked] - ref) / ref * norm_scaling
            v = values[valid]
            if len(v) > 0:
                print(f"  {pi}: {np.mean(v):>12.6g} / {np.median(v):>12.6g} / {np.std(v):>12.6g}")
            label = f'{phase} peak {pi}'
            ax.plot(frame_numbers[valid], values[valid], '-o', ms=4, label=label)
            if show_masked and np.any(user_masked):
                ax.scatter(frame_numbers[user_masked], values[user_masked],
                           s=10, marker='x', color='gray', alpha=0.4, zorder=1)

        ax.set_xlabel('Frame No.')
        ylabel = f'({param_name} - ref) / ref  [x{norm_scaling:.3g}]' if norm_enabled else param_name
        ax.set_ylabel(ylabel)
        ax.set_title(f'{phase} — {param_name} (lineout)')
        ax.grid(True, alpha=0.3)
        if len(peak_indices) > 1:
            ax.legend(fontsize=8, markerscale=2)

        selection_changed = (phase != self._prev_phase or
                             peak_sel != self._prev_peak or
                             param_idx != self._prev_param or
                             norm_enabled != self._prev_norm)
        self._prev_phase = phase
        self._prev_peak = peak_sel
        self._prev_param = param_idx
        self._prev_norm = norm_enabled

        if not opts.get('auto_x', True):
            ax.set_xlim(opts.get('xmin'), opts.get('xmax'))
        elif prev_xlim and not selection_changed:
            ax.set_xlim(prev_xlim)

        if not opts.get('auto_y', True):
            ax.set_ylim(opts.get('ymin'), opts.get('ymax'))
        elif prev_ylim and not selection_changed:
            ax.set_ylim(prev_ylim)

        self.figure.tight_layout()
        self.canvas.draw()
        self._programmatic_update = False

    def _on_xlim_changed(self, ax):
        if self._programmatic_update:
            return
        xmin, xmax = ax.get_xlim()
        parent = self.parent()
        if parent is None:
            return
        parent._lf_auto_x.blockSignals(True)
        parent._lf_auto_x.setChecked(False)
        parent._lf_xmin.setEnabled(True)
        parent._lf_xmax.setEnabled(True)
        parent._lf_auto_x.blockSignals(False)
        parent._lf_xmin.setText(f'{xmin:.6g}')
        parent._lf_xmax.setText(f'{xmax:.6g}')

    def _on_ylim_changed(self, ax):
        if self._programmatic_update:
            return
        ymin, ymax = ax.get_ylim()
        parent = self.parent()
        if parent is None:
            return
        parent._lf_auto_y.blockSignals(True)
        parent._lf_auto_y.setChecked(False)
        parent._lf_ymin.setEnabled(True)
        parent._lf_ymax.setEnabled(True)
        parent._lf_auto_y.blockSignals(False)
        parent._lf_ymin.setText(f'{ymin:.6g}')
        parent._lf_ymax.setText(f'{ymax:.6g}')


class FitResultWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fit_data = {}
        self.eta_axis = None
        self.plot_opts = {}
        self._programmatic_update = False
        self._prev_phase = None
        self._prev_peak = None
        self._prev_param = None
        self._prev_norm = False
        self.setWindowTitle('Fit Results')
        self.setGeometry(200, 200, 900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

    def set_data(self, fit_data, eta_axis, plot_opts):
        self.fit_data = fit_data
        self.eta_axis = eta_axis
        self.plot_opts = plot_opts
        self.update_plot()

    def update_plot(self, *_args):
        if not self.fit_data or self.eta_axis is None:
            return

        opts = self.plot_opts
        phase = opts.get('phase', '')
        param_idx = opts.get('param_idx', 1)
        frame_idx = opts.get('frame_idx', 0)
        peak_sel = opts.get('peak', 'All')

        phase_data = self.fit_data.get(phase)
        if phase_data is None:
            return

        params = phase_data['params']
        mask = phase_data['mask']
        n_frames_fit = params.shape[0]
        n_peaks = params.shape[1]

        if frame_idx >= n_frames_fit:
            return

        if peak_sel.lower() == 'all' or peak_sel == '':
            peak_indices = list(range(n_peaks))
        else:
            try:
                peak_indices = []
                for tok in peak_sel.replace(',', ' ').split():
                    if '-' in tok:
                        a, b = tok.split('-', 1)
                        peak_indices.extend(range(int(a), int(b) + 1))
                    else:
                        peak_indices.append(int(tok))
                peak_indices = [p for p in peak_indices if 0 <= p < n_peaks]
            except ValueError:
                peak_indices = list(range(n_peaks))

        plot_type = opts.get('plot_type', 'Cartesian')
        is_polar = plot_type == 'Polar'

        prev_xlim = prev_ylim = None
        if self.figure.axes:
            prev_xlim = self.figure.axes[0].get_xlim()
            prev_ylim = self.figure.axes[0].get_ylim()

        self._programmatic_update = True
        self.figure.clear()
        if is_polar:
            ax = self.figure.add_subplot(111, projection='polar')
        else:
            ax = self.figure.add_subplot(111)
            ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            ax.callbacks.connect('ylim_changed', self._on_ylim_changed)

        if param_idx < 4:
            param_name = PARAM_LABELS[param_idx]
        elif param_idx == 4:
            param_name = 'Chi²'
        else:
            param_name = PARAM_LABELS[param_idx - 5] + ' Err'
        norm_enabled = opts.get('norm_enabled', False) and param_idx == 1
        norm_mode = opts.get('norm_mode', 'Mean')
        norm_values = opts.get('norm_values', [])
        norm_scaling = opts.get('norm_scaling', 1e6)
        mask_filters = opts.get('filters', [])
        show_masked = opts.get('show_masked', True)

        unit = f'x{norm_scaling:.3g}' if norm_enabled else opts.get('radial_unit', '')
        print(f"--- {phase} | {param_name} | Frame {frame_idx} ---")
        print(f"  peakID: Mean / Median / Std ({unit})")
        for i, pi in enumerate(peak_indices):
            if param_idx < 4:
                values = params[frame_idx, pi, :, param_idx].copy()
            elif param_idx == 4:
                values = phase_data['chi2'][frame_idx, pi, :].copy()
            else:
                values = phase_data['errors'][frame_idx, pi, :, param_idx - 5].copy()
            valid = mask[frame_idx, pi, :].copy()

            # Apply user mask filters (cross-parameter)
            for flt in mask_filters:
                fi = flt['param_idx']
                if fi < 4:
                    fvals = params[frame_idx, pi, :, fi]
                elif fi == 4:
                    fvals = phase_data['chi2'][frame_idx, pi, :]
                else:
                    fvals = phase_data['errors'][frame_idx, pi, :, fi - 5]
                if flt['min'] is not None:
                    valid &= fvals >= flt['min']
                if flt['max'] is not None:
                    valid &= fvals <= flt['max']
            user_masked = mask[frame_idx, pi, :] & ~valid

            if norm_enabled:
                if norm_mode == 'Mean':
                    ref = np.mean(values[valid])
                elif norm_mode == 'Median':
                    ref = np.median(values[valid])
                else:
                    ref = norm_values[i] if i < len(norm_values) else 0
                if ref != 0:
                    values[valid] = (values[valid] - ref) / ref * norm_scaling
                    values[user_masked] = (values[user_masked] - ref) / ref * norm_scaling
            v = values[valid]
            if len(v) > 0:
                print(f"  {pi}: {np.mean(v):>12.6g} / {np.median(v):>12.6g} / {np.std(v):>12.6g}")
            label = f'{phase} peak {pi}'
            if is_polar:
                theta = np.deg2rad(self.eta_axis)
                ax.scatter(theta[valid], values[valid], s=10, label=label)
                if show_masked and np.any(user_masked):
                    ax.scatter(theta[user_masked], values[user_masked],
                               s=10, marker='x', color='gray', alpha=0.4, zorder=1)
            else:
                ax.scatter(self.eta_axis[valid], values[valid], s=10, label=label)
                if show_masked and np.any(user_masked):
                    ax.scatter(self.eta_axis[user_masked], values[user_masked],
                               s=10, marker='x', color='gray', alpha=0.4, zorder=1)

        ylabel = f'({param_name} - ref) / ref  [x{norm_scaling:.3g}]' if norm_enabled else param_name
        ax.set_title(f'{phase} — {param_name} — Frame {frame_idx}')
        ax.grid(True, alpha=0.3)
        if len(peak_indices) > 1:
            ax.legend(fontsize=8, markerscale=2)

        if is_polar:
            if norm_enabled:
                theta_ref = np.linspace(0, 2 * np.pi, 361)
                ax.plot(theta_ref, [0] * 361, 'k--', lw=1.5, label='Ref.')
        else:
            ax.set_xlabel('Eta (deg)')
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(-180, 181, 30))

        selection_changed = (phase != self._prev_phase or
                             peak_sel != self._prev_peak or
                             param_idx != self._prev_param or
                             norm_enabled != self._prev_norm)
        self._prev_phase = phase
        self._prev_peak = peak_sel
        self._prev_param = param_idx
        self._prev_norm = norm_enabled

        if not is_polar:
            if not opts.get('auto_x', True):
                ax.set_xlim(opts.get('xmin'), opts.get('xmax'))
            elif prev_xlim and not selection_changed:
                ax.set_xlim(prev_xlim)

        if not opts.get('auto_y', True):
            ax.set_ylim(opts.get('ymin'), opts.get('ymax'))
        elif prev_ylim and not selection_changed:
            ax.set_ylim(prev_ylim)

        self.figure.tight_layout()
        self.canvas.draw()
        self._programmatic_update = False

    def _on_xlim_changed(self, ax):
        if self._programmatic_update:
            return
        xmin, xmax = ax.get_xlim()
        parent = self.parent()
        if parent is None:
            return
        parent._fr_auto_x.blockSignals(True)
        parent._fr_auto_x.setChecked(False)
        parent._fr_xmin.setEnabled(True)
        parent._fr_xmax.setEnabled(True)
        parent._fr_auto_x.blockSignals(False)
        parent._fr_xmin.setText(f'{xmin:.6g}')
        parent._fr_xmax.setText(f'{xmax:.6g}')

    def _on_ylim_changed(self, ax):
        if self._programmatic_update:
            return
        ymin, ymax = ax.get_ylim()
        parent = self.parent()
        if parent is None:
            return
        parent._fr_auto_y.blockSignals(True)
        parent._fr_auto_y.setChecked(False)
        parent._fr_ymin.setEnabled(True)
        parent._fr_ymax.setEnabled(True)
        parent._fr_auto_y.blockSignals(False)
        parent._fr_ymin.setText(f'{ymin:.6g}')
        parent._fr_ymax.setText(f'{ymax:.6g}')


# ============================================================================
# Main Control Panel
# ============================================================================

class WAXSViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.filepath = None
        self.x_axes = {}
        self.lineouts = None
        self.lineouts_mean = None
        self.frame_names = None
        self.n_frames = 0
        self._h5file = None
        self._omega_dataset = None
        self.eta_axis = None
        self._1d_window = None
        self._2d_windows = []
        self._omega_window = None
        self._fit_window = None
        self._fit_window_polar = None
        self._fit_data = {}
        self._fit_eta = None
        self._lineout_fit_window = None
        self._lineout_fit_data = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('WAXS Viewer')
        self.setGeometry(100, 100, 900, 350)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Row 1: File ---
        file_layout = QHBoxLayout()
        file_label = QLabel('File:')
        file_label.setFixedWidth(65)
        file_layout.addWidget(file_label)
        self.file_combo = QComboBox()
        self.file_combo.activated.connect(self.on_file_selected)
        file_layout.addWidget(self.file_combo, stretch=1)
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        main_layout.addLayout(file_layout)

        # --- Row 2: Dataset + frame controls ---
        control_layout = QHBoxLayout()

        dataset_label = QLabel('Dataset:')
        dataset_label.setFixedWidth(65)
        control_layout.addWidget(dataset_label)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['lineouts', 'lineouts_simple_mean'])
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        control_layout.addWidget(self.dataset_combo)

        control_layout.addWidget(QLabel('Frame:'))
        self.frame_input = QLineEdit('0')
        self.frame_input.setPlaceholderText('e.g. 0, 0:10, 0,5,10')
        self.frame_input.setMinimumWidth(200)
        self.frame_input.editingFinished.connect(self.on_settings_changed)
        control_layout.addWidget(self.frame_input)

        self.prev_btn = QPushButton('<')
        self.prev_btn.setMaximumWidth(30)
        self.prev_btn.clicked.connect(self.step_frame_backward)
        control_layout.addWidget(self.prev_btn)

        self.step_input = QLineEdit('1')
        self.step_input.setMaximumWidth(40)
        self.step_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.step_input)

        self.next_btn = QPushButton('>')
        self.next_btn.setMaximumWidth(30)
        self.next_btn.clicked.connect(self.step_frame_forward)
        control_layout.addWidget(self.next_btn)

        self.frame_info_label = QLabel('Frames: 0')
        control_layout.addWidget(self.frame_info_label)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # --- Plot type tabs ---
        self._tab_widget = QTabWidget()

        raw_tab = QWidget()
        raw_layout = QVBoxLayout(raw_tab)
        raw_layout.addWidget(self._build_1d_panel())
        raw_layout.addWidget(self._build_2d_panel())
        raw_layout.addWidget(self._build_omega_panel())
        raw_layout.addStretch()
        self._tab_widget.addTab(raw_tab, 'Integrated Data')

        fit_tab = QWidget()
        fit_layout = QVBoxLayout(fit_tab)
        fit_layout.addWidget(self._build_fit_panel())
        fit_layout.addWidget(self._build_lineout_fit_panel())
        fit_layout.addStretch()
        self._tab_widget.addTab(fit_tab, 'Fit Results')

        main_layout.addWidget(self._tab_widget)

        # --- Status bar ---
        self.statusBar().showMessage('Ready')
        version_label = QLabel('ver.1.0.0 (2026/05/20 by AC)')
        version_label.setStyleSheet('color: gray; font-size: 10pt;')
        self.statusBar().addPermanentWidget(version_label)

    # ----- Sub-panel builders -----

    def _build_1d_panel(self):
        widget = QWidget()
        self._1d_panel = widget
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self._1d_toggle = QCheckBox('1D Lineout')
        self._1d_toggle.toggled.connect(self._on_toggle_1d)
        top.addWidget(self._1d_toggle)
        top.addStretch()
        layout.addLayout(top)

        # Row 1: X-axis, Auto X, XMin, XMax, Marker
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('X-axis:'))
        self._1d_xaxis = QComboBox()
        self._1d_xaxis.setFixedWidth(130)
        self._1d_xaxis.addItems(['R (pixel)', 'TTH (deg)', 'Q (1/A)'])
        self._1d_xaxis.currentTextChanged.connect(self._on_1d_changed)
        row1.addWidget(self._1d_xaxis)

        self._1d_auto_x = QCheckBox('Auto')
        self._1d_auto_x.setChecked(True)
        self._1d_auto_x.stateChanged.connect(self._on_1d_auto_x_changed)
        row1.addWidget(self._1d_auto_x)

        row1.addWidget(QLabel('Min:'))
        self._1d_xmin = QLineEdit()
        self._1d_xmin.setPlaceholderText('XMin')
        self._1d_xmin.setMaximumWidth(80)
        self._1d_xmin.setEnabled(False)
        self._1d_xmin.editingFinished.connect(self._on_1d_changed)
        row1.addWidget(self._1d_xmin)

        row1.addWidget(QLabel('Max:'))
        self._1d_xmax = QLineEdit()
        self._1d_xmax.setPlaceholderText('XMax')
        self._1d_xmax.setMaximumWidth(80)
        self._1d_xmax.setEnabled(False)
        self._1d_xmax.editingFinished.connect(self._on_1d_changed)
        row1.addWidget(self._1d_xmax)

        row1.addWidget(QLabel('Marker:'))
        self._1d_marker = QComboBox()
        markers = [('Circle', 'o'), ('Square', 's'), ('Triangle Up', '^'),
                   ('Triangle Down', 'v'), ('Diamond', 'D'), ('Plus', '+'),
                   ('Cross', 'x'), ('Star', '*'), ('Point', '.'), ('None', 'None')]
        for label, code in markers:
            self._1d_marker.addItem(label, userData=code)
        self._1d_marker.setCurrentIndex(0)
        self._1d_marker.currentIndexChanged.connect(self._on_1d_changed)
        row1.addWidget(self._1d_marker)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Y-axis, Auto Range, YMin, YMax, Avg, Log10, Legend
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Y-axis:'))
        self._1d_yaxis = QComboBox()
        self._1d_yaxis.setFixedWidth(130)
        self._1d_yaxis.addItems(['Intensity'])
        row2.addWidget(self._1d_yaxis)

        self._1d_auto_range = QCheckBox('Auto')
        self._1d_auto_range.setChecked(True)
        self._1d_auto_range.stateChanged.connect(self._on_1d_auto_y_changed)
        row2.addWidget(self._1d_auto_range)

        row2.addWidget(QLabel('Min:'))
        self._1d_ymin = QLineEdit()
        self._1d_ymin.setPlaceholderText('YMin')
        self._1d_ymin.setMaximumWidth(80)
        self._1d_ymin.setEnabled(False)
        self._1d_ymin.editingFinished.connect(self._on_1d_changed)
        row2.addWidget(self._1d_ymin)

        row2.addWidget(QLabel('Max:'))
        self._1d_ymax = QLineEdit()
        self._1d_ymax.setPlaceholderText('YMax')
        self._1d_ymax.setMaximumWidth(80)
        self._1d_ymax.setEnabled(False)
        self._1d_ymax.editingFinished.connect(self._on_1d_changed)
        row2.addWidget(self._1d_ymax)

        self._1d_avg = QCheckBox('Avg.')
        self._1d_avg.stateChanged.connect(self._on_1d_changed)
        row2.addWidget(self._1d_avg)

        self._1d_log_y = QCheckBox('Log10')
        self._1d_log_y.stateChanged.connect(self._on_1d_changed)
        row2.addWidget(self._1d_log_y)

        self._1d_legend = QCheckBox('Legend')
        self._1d_legend.setChecked(True)
        self._1d_legend.stateChanged.connect(self._on_1d_changed)
        row2.addWidget(self._1d_legend)

        row2.addStretch()
        layout.addLayout(row2)

        return widget

    def _build_2d_panel(self):
        widget = QWidget()
        self._2d_panel = widget
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self._2d_toggle = QCheckBox('2D Lineout')
        self._2d_toggle.toggled.connect(self._on_toggle_2d)
        top.addWidget(self._2d_toggle)
        top.addStretch()
        layout.addLayout(top)

        # Row 1: X-axis, Auto, Min, Max
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('X-axis:'))
        self._2d_xaxis = QComboBox()
        self._2d_xaxis.setFixedWidth(130)
        self._2d_xaxis.addItems(['R (pixel)', 'TTH (deg)', 'Q (1/A)'])
        self._2d_xaxis.currentTextChanged.connect(self._on_2d_changed)
        row1.addWidget(self._2d_xaxis)

        self._2d_auto_x = QCheckBox('Auto')
        self._2d_auto_x.setChecked(True)
        self._2d_auto_x.stateChanged.connect(self._on_2d_auto_x_changed)
        row1.addWidget(self._2d_auto_x)

        row1.addWidget(QLabel('Min:'))
        self._2d_xmin = QLineEdit()
        self._2d_xmin.setPlaceholderText('XMin')
        self._2d_xmin.setMaximumWidth(80)
        self._2d_xmin.setEnabled(False)
        self._2d_xmin.editingFinished.connect(self._on_2d_changed)
        row1.addWidget(self._2d_xmin)

        row1.addWidget(QLabel('Max:'))
        self._2d_xmax = QLineEdit()
        self._2d_xmax.setPlaceholderText('XMax')
        self._2d_xmax.setMaximumWidth(80)
        self._2d_xmax.setEnabled(False)
        self._2d_xmax.editingFinished.connect(self._on_2d_changed)
        row1.addWidget(self._2d_xmax)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Y-axis (Seq.), Auto, Min, Max
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Y-axis:'))
        self._2d_yaxis = QComboBox()
        self._2d_yaxis.setFixedWidth(130)
        self._2d_yaxis.addItems(['Frame Seq.'])
        row2.addWidget(self._2d_yaxis)

        self._2d_auto_y = QCheckBox('Auto')
        self._2d_auto_y.setChecked(True)
        self._2d_auto_y.stateChanged.connect(self._on_2d_auto_y_changed)
        row2.addWidget(self._2d_auto_y)

        row2.addWidget(QLabel('Min:'))
        self._2d_ymin = QLineEdit()
        self._2d_ymin.setPlaceholderText('YMin')
        self._2d_ymin.setMaximumWidth(80)
        self._2d_ymin.setEnabled(False)
        self._2d_ymin.editingFinished.connect(self._on_2d_changed)
        row2.addWidget(self._2d_ymin)

        row2.addWidget(QLabel('Max:'))
        self._2d_ymax = QLineEdit()
        self._2d_ymax.setPlaceholderText('YMax')
        self._2d_ymax.setMaximumWidth(80)
        self._2d_ymax.setEnabled(False)
        self._2d_ymax.editingFinished.connect(self._on_2d_changed)
        row2.addWidget(self._2d_ymax)

        row2.addStretch()
        layout.addLayout(row2)

        # Row 3: Z-axis (Intensity), Auto, Min, Max, Colormap, Log10
        row3 = QHBoxLayout()
        row3.addWidget(QLabel('Z-axis:'))
        self._2d_zaxis = QComboBox()
        self._2d_zaxis.setFixedWidth(130)
        self._2d_zaxis.addItems(['Intensity'])
        row3.addWidget(self._2d_zaxis)

        self._2d_auto_int = QCheckBox('Auto')
        self._2d_auto_int.setChecked(True)
        self._2d_auto_int.stateChanged.connect(self._on_2d_auto_int_changed)
        row3.addWidget(self._2d_auto_int)

        row3.addWidget(QLabel('Min:'))
        self._2d_imin = QLineEdit()
        self._2d_imin.setPlaceholderText('ZMin')
        self._2d_imin.setMaximumWidth(80)
        self._2d_imin.setEnabled(False)
        self._2d_imin.editingFinished.connect(self._on_2d_changed)
        row3.addWidget(self._2d_imin)

        row3.addWidget(QLabel('Max:'))
        self._2d_imax = QLineEdit()
        self._2d_imax.setPlaceholderText('ZMax')
        self._2d_imax.setMaximumWidth(80)
        self._2d_imax.setEnabled(False)
        self._2d_imax.editingFinished.connect(self._on_2d_changed)
        row3.addWidget(self._2d_imax)

        row3.addWidget(QLabel('Colormap:'))
        self._2d_cmap = QComboBox()
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'gray', 'bone', 'hot', 'cool', 'jet', 'turbo']
        self._2d_cmap.addItems(cmaps)
        self._2d_cmap.currentTextChanged.connect(self._on_2d_changed)
        row3.addWidget(self._2d_cmap)

        self._2d_log = QCheckBox('Log10')
        self._2d_log.stateChanged.connect(self._on_2d_changed)
        row3.addWidget(self._2d_log)

        row3.addStretch()
        layout.addLayout(row3)

        return widget

    def _build_omega_panel(self):
        widget = QWidget()
        self._om_panel = widget
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self._om_toggle = QCheckBox('OmegaSumFrame')
        self._om_toggle.toggled.connect(self._on_toggle_omega)
        top.addWidget(self._om_toggle)
        top.addStretch()
        layout.addLayout(top)

        # Row 1: X-axis, Auto, Min, Max
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('X-axis:'))
        self._om_xaxis = QComboBox()
        self._om_xaxis.setFixedWidth(130)
        self._om_xaxis.addItems(['R (pixel)', 'TTH (deg)', 'Q (1/A)'])
        self._om_xaxis.currentTextChanged.connect(self._on_omega_changed)
        row1.addWidget(self._om_xaxis)

        self._om_auto_x = QCheckBox('Auto')
        self._om_auto_x.setChecked(True)
        self._om_auto_x.stateChanged.connect(self._on_omega_auto_x_changed)
        row1.addWidget(self._om_auto_x)

        row1.addWidget(QLabel('Min:'))
        self._om_xmin = QLineEdit()
        self._om_xmin.setPlaceholderText('XMin')
        self._om_xmin.setMaximumWidth(80)
        self._om_xmin.setEnabled(False)
        self._om_xmin.editingFinished.connect(self._on_omega_changed)
        row1.addWidget(self._om_xmin)

        row1.addWidget(QLabel('Max:'))
        self._om_xmax = QLineEdit()
        self._om_xmax.setPlaceholderText('XMax')
        self._om_xmax.setMaximumWidth(80)
        self._om_xmax.setEnabled(False)
        self._om_xmax.editingFinished.connect(self._on_omega_changed)
        row1.addWidget(self._om_xmax)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Y-axis (Eta), Auto, Min, Max
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Y-axis:'))
        self._om_yaxis = QComboBox()
        self._om_yaxis.setFixedWidth(130)
        self._om_yaxis.addItems(['Eta (deg)'])
        row2.addWidget(self._om_yaxis)

        self._om_auto_y = QCheckBox('Auto')
        self._om_auto_y.setChecked(True)
        self._om_auto_y.stateChanged.connect(self._on_omega_auto_y_changed)
        row2.addWidget(self._om_auto_y)

        row2.addWidget(QLabel('Min:'))
        self._om_ymin = QLineEdit()
        self._om_ymin.setPlaceholderText('YMin')
        self._om_ymin.setMaximumWidth(80)
        self._om_ymin.setEnabled(False)
        self._om_ymin.editingFinished.connect(self._on_omega_changed)
        row2.addWidget(self._om_ymin)

        row2.addWidget(QLabel('Max:'))
        self._om_ymax = QLineEdit()
        self._om_ymax.setPlaceholderText('YMax')
        self._om_ymax.setMaximumWidth(80)
        self._om_ymax.setEnabled(False)
        self._om_ymax.editingFinished.connect(self._on_omega_changed)
        row2.addWidget(self._om_ymax)

        row2.addStretch()
        layout.addLayout(row2)

        # Row 3: Z-axis (Intensity), Auto, Min, Max, Colormap, Log10
        row3 = QHBoxLayout()
        row3.addWidget(QLabel('Z-axis:'))
        self._om_zaxis = QComboBox()
        self._om_zaxis.setFixedWidth(130)
        self._om_zaxis.addItems(['Intensity'])
        row3.addWidget(self._om_zaxis)

        self._om_auto_int = QCheckBox('Auto')
        self._om_auto_int.setChecked(True)
        self._om_auto_int.stateChanged.connect(self._on_omega_auto_int_changed)
        row3.addWidget(self._om_auto_int)

        row3.addWidget(QLabel('Min:'))
        self._om_imin = QLineEdit()
        self._om_imin.setPlaceholderText('ZMin')
        self._om_imin.setMaximumWidth(80)
        self._om_imin.setEnabled(False)
        self._om_imin.editingFinished.connect(self._on_omega_changed)
        row3.addWidget(self._om_imin)

        row3.addWidget(QLabel('Max:'))
        self._om_imax = QLineEdit()
        self._om_imax.setPlaceholderText('ZMax')
        self._om_imax.setMaximumWidth(80)
        self._om_imax.setEnabled(False)
        self._om_imax.editingFinished.connect(self._on_omega_changed)
        row3.addWidget(self._om_imax)

        row3.addWidget(QLabel('Colormap:'))
        self._om_cmap = QComboBox()
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                 'gray', 'bone', 'hot', 'cool', 'jet', 'turbo']
        self._om_cmap.addItems(cmaps)
        self._om_cmap.currentTextChanged.connect(self._on_omega_changed)
        row3.addWidget(self._om_cmap)

        self._om_log = QCheckBox('Log10')
        self._om_log.stateChanged.connect(self._on_omega_changed)
        row3.addWidget(self._om_log)

        row3.addStretch()
        layout.addLayout(row3)

        return widget

    def _build_fit_panel(self):
        widget = QWidget()
        self._fr_panel = widget
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self._fr_toggle = QCheckBox('Fit Results (OmegaSumFrame)')
        self._fr_toggle.toggled.connect(self._on_toggle_fit)
        top.addWidget(self._fr_toggle)
        top.addWidget(QLabel('Plot:'))
        self._fr_plot_type = QComboBox()
        self._fr_plot_type.setFixedWidth(90)
        self._fr_plot_type.addItems(['Cartesian', 'Polar', 'Both'])
        self._fr_plot_type.currentTextChanged.connect(self._on_fit_changed)
        top.addWidget(self._fr_plot_type)
        top.addStretch()
        layout.addLayout(top)

        # Rows 1-2: Phase/Peak/Param/Mask + Norm/Mode/Ref/Scaling
        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setColumnStretch(5, 1)

        grid.addWidget(QLabel('Phase:'), 0, 0)
        self._fr_phase = QComboBox()
        self._fr_phase.setFixedWidth(130)
        self._fr_phase.currentTextChanged.connect(self._on_fit_phase_changed)
        grid.addWidget(self._fr_phase, 0, 1)

        grid.addWidget(QLabel('Peak:'), 0, 2)
        pk_row = QHBoxLayout()
        self._fr_peak = QLineEdit('All')
        self._fr_peak.setFixedWidth(60)
        self._fr_peak.setPlaceholderText('All')
        self._fr_peak.editingFinished.connect(self._on_fit_changed)
        pk_row.addWidget(self._fr_peak)
        fr_pk_prev = QPushButton('<')
        fr_pk_prev.setFixedWidth(24)
        fr_pk_prev.clicked.connect(lambda: self._step_peak(self._fr_peak, -1, self._fit_data, self._fr_phase))
        pk_row.addWidget(fr_pk_prev)
        fr_pk_next = QPushButton('>')
        fr_pk_next.setFixedWidth(24)
        fr_pk_next.clicked.connect(lambda: self._step_peak(self._fr_peak, 1, self._fit_data, self._fr_phase))
        pk_row.addWidget(fr_pk_next)
        pk_row.addStretch()
        grid.addLayout(pk_row, 0, 3)

        grid.addWidget(QLabel('Param:'), 0, 4)
        self._fr_param = QComboBox()
        self._fr_param.addItems(['Amplitude', 'Center', 'Width', 'Eta', 'Chi²',
                                 'Amplitude Err', 'Center Err', 'Width Err', 'Eta Err'])
        self._fr_param.setCurrentIndex(1)
        self._fr_param.currentTextChanged.connect(self._on_fit_changed)
        grid.addWidget(self._fr_param, 0, 5)

        self._mask_dialog = MaskFilterDialog(on_apply=self._on_fit_changed, parent=self)
        mask_btn = QPushButton('Mask')
        mask_btn.clicked.connect(lambda: self._mask_dialog.show())
        grid.addWidget(mask_btn, 0, 6)

        self._fr_norm = QCheckBox('Norm.')
        self._fr_norm.stateChanged.connect(self._on_fit_changed)
        grid.addWidget(self._fr_norm, 1, 0)
        self._fr_norm_mode = QComboBox()
        self._fr_norm_mode.addItems(['Mean', 'Median', 'Custom'])
        self._fr_norm_mode.setFixedWidth(80)
        self._fr_norm_mode.currentTextChanged.connect(self._on_fit_changed)
        grid.addWidget(self._fr_norm_mode, 1, 1)
        grid.addWidget(QLabel('Ref:'), 1, 2)
        self._fr_norm_vals = QLineEdit()
        self._fr_norm_vals.setPlaceholderText('ref. centers')
        self._fr_norm_vals.editingFinished.connect(self._on_norm_vals_changed)
        grid.addWidget(self._fr_norm_vals, 1, 3)
        grid.addWidget(QLabel('Scaling:'), 1, 4)
        self._fr_scaling = QLineEdit('1e6')
        self._fr_scaling.setFixedWidth(80)
        self._fr_scaling.editingFinished.connect(self._on_fit_changed)
        grid.addWidget(self._fr_scaling, 1, 5)

        layout.addLayout(grid)

        # Row 3: X-axis (Eta), Auto, Min, Max
        row3 = QHBoxLayout()
        row3.addWidget(QLabel('X-axis:'))
        fr_xlabel = QComboBox()
        fr_xlabel.setFixedWidth(130)
        fr_xlabel.addItems(['Eta (deg)'])
        row3.addWidget(fr_xlabel)

        self._fr_auto_x = QCheckBox('Auto')
        self._fr_auto_x.setChecked(True)
        self._fr_auto_x.stateChanged.connect(self._on_fit_auto_x_changed)
        row3.addWidget(self._fr_auto_x)

        row3.addWidget(QLabel('Min:'))
        self._fr_xmin = QLineEdit()
        self._fr_xmin.setPlaceholderText('XMin')
        self._fr_xmin.setMaximumWidth(80)
        self._fr_xmin.setEnabled(False)
        self._fr_xmin.editingFinished.connect(self._on_fit_changed)
        row3.addWidget(self._fr_xmin)

        row3.addWidget(QLabel('Max:'))
        self._fr_xmax = QLineEdit()
        self._fr_xmax.setPlaceholderText('XMax')
        self._fr_xmax.setMaximumWidth(80)
        self._fr_xmax.setEnabled(False)
        self._fr_xmax.editingFinished.connect(self._on_fit_changed)
        row3.addWidget(self._fr_xmax)

        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Y-axis, Auto, Min, Max
        row4 = QHBoxLayout()
        row4.addWidget(QLabel('Y-axis:'))
        self._fr_ylabel = QComboBox()
        self._fr_ylabel.setFixedWidth(130)
        self._fr_ylabel.addItems(['Auto'])
        row4.addWidget(self._fr_ylabel)

        self._fr_auto_y = QCheckBox('Auto')
        self._fr_auto_y.setChecked(True)
        self._fr_auto_y.stateChanged.connect(self._on_fit_auto_y_changed)
        row4.addWidget(self._fr_auto_y)

        row4.addWidget(QLabel('Min:'))
        self._fr_ymin = QLineEdit()
        self._fr_ymin.setPlaceholderText('YMin')
        self._fr_ymin.setMaximumWidth(80)
        self._fr_ymin.setEnabled(False)
        self._fr_ymin.editingFinished.connect(self._on_fit_changed)
        row4.addWidget(self._fr_ymin)

        row4.addWidget(QLabel('Max:'))
        self._fr_ymax = QLineEdit()
        self._fr_ymax.setPlaceholderText('YMax')
        self._fr_ymax.setMaximumWidth(80)
        self._fr_ymax.setEnabled(False)
        self._fr_ymax.editingFinished.connect(self._on_fit_changed)
        row4.addWidget(self._fr_ymax)

        row4.addStretch()
        layout.addLayout(row4)

        return widget

    def _build_lineout_fit_panel(self):
        widget = QWidget()
        self._lf_panel = widget
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self._lf_toggle = QCheckBox('Fit Results (Lineout)')
        self._lf_toggle.toggled.connect(self._on_toggle_lineout_fit)
        top.addWidget(self._lf_toggle)
        top.addStretch()
        layout.addLayout(top)

        # Rows 1-2: Phase/Peak/Param/Mask + Norm/Mode/Ref/Scaling
        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setColumnStretch(5, 1)

        grid.addWidget(QLabel('Phase:'), 0, 0)
        self._lf_phase = QComboBox()
        self._lf_phase.setFixedWidth(130)
        self._lf_phase.currentTextChanged.connect(self._on_lf_phase_changed)
        grid.addWidget(self._lf_phase, 0, 1)

        grid.addWidget(QLabel('Peak:'), 0, 2)
        pk_row = QHBoxLayout()
        self._lf_peak = QLineEdit('All')
        self._lf_peak.setFixedWidth(60)
        self._lf_peak.setPlaceholderText('All')
        self._lf_peak.editingFinished.connect(self._on_lineout_fit_changed)
        pk_row.addWidget(self._lf_peak)
        lf_pk_prev = QPushButton('<')
        lf_pk_prev.setFixedWidth(24)
        lf_pk_prev.clicked.connect(lambda: self._step_peak(self._lf_peak, -1, self._lineout_fit_data, self._lf_phase))
        pk_row.addWidget(lf_pk_prev)
        lf_pk_next = QPushButton('>')
        lf_pk_next.setFixedWidth(24)
        lf_pk_next.clicked.connect(lambda: self._step_peak(self._lf_peak, 1, self._lineout_fit_data, self._lf_phase))
        pk_row.addWidget(lf_pk_next)
        pk_row.addStretch()
        grid.addLayout(pk_row, 0, 3)

        grid.addWidget(QLabel('Param:'), 0, 4)
        self._lf_param = QComboBox()
        self._lf_param.addItems(['Amplitude', 'Center', 'Width', 'Eta', 'Chi²',
                                 'Amplitude Err', 'Center Err', 'Width Err', 'Eta Err'])
        self._lf_param.setCurrentIndex(1)
        self._lf_param.currentTextChanged.connect(self._on_lineout_fit_changed)
        grid.addWidget(self._lf_param, 0, 5)

        self._lf_mask_dialog = MaskFilterDialog(on_apply=self._on_lineout_fit_changed, parent=self)
        lf_mask_btn = QPushButton('Mask')
        lf_mask_btn.clicked.connect(lambda: self._lf_mask_dialog.show())
        grid.addWidget(lf_mask_btn, 0, 6)

        self._lf_norm = QCheckBox('Norm.')
        self._lf_norm.stateChanged.connect(self._on_lineout_fit_changed)
        grid.addWidget(self._lf_norm, 1, 0)
        self._lf_norm_mode = QComboBox()
        self._lf_norm_mode.addItems(['Mean', 'Median', 'Custom'])
        self._lf_norm_mode.setFixedWidth(80)
        self._lf_norm_mode.currentTextChanged.connect(self._on_lineout_fit_changed)
        grid.addWidget(self._lf_norm_mode, 1, 1)
        grid.addWidget(QLabel('Ref:'), 1, 2)
        self._lf_norm_vals = QLineEdit()
        self._lf_norm_vals.setPlaceholderText('ref. centers')
        self._lf_norm_vals.editingFinished.connect(self._on_lf_norm_vals_changed)
        grid.addWidget(self._lf_norm_vals, 1, 3)
        grid.addWidget(QLabel('Scaling:'), 1, 4)
        self._lf_scaling = QLineEdit('1e6')
        self._lf_scaling.setFixedWidth(80)
        self._lf_scaling.editingFinished.connect(self._on_lineout_fit_changed)
        grid.addWidget(self._lf_scaling, 1, 5)

        layout.addLayout(grid)

        # Row 3: X-axis (Frame No.), Auto, Min, Max
        row3 = QHBoxLayout()
        row3.addWidget(QLabel('X-axis:'))
        lf_xlabel = QComboBox()
        lf_xlabel.setFixedWidth(130)
        lf_xlabel.addItems(['Frame No.'])
        row3.addWidget(lf_xlabel)

        self._lf_auto_x = QCheckBox('Auto')
        self._lf_auto_x.setChecked(True)
        self._lf_auto_x.stateChanged.connect(self._on_lf_auto_x_changed)
        row3.addWidget(self._lf_auto_x)

        row3.addWidget(QLabel('Min:'))
        self._lf_xmin = QLineEdit()
        self._lf_xmin.setPlaceholderText('XMin')
        self._lf_xmin.setMaximumWidth(80)
        self._lf_xmin.setEnabled(False)
        self._lf_xmin.editingFinished.connect(self._on_lineout_fit_changed)
        row3.addWidget(self._lf_xmin)

        row3.addWidget(QLabel('Max:'))
        self._lf_xmax = QLineEdit()
        self._lf_xmax.setPlaceholderText('XMax')
        self._lf_xmax.setMaximumWidth(80)
        self._lf_xmax.setEnabled(False)
        self._lf_xmax.editingFinished.connect(self._on_lineout_fit_changed)
        row3.addWidget(self._lf_xmax)

        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Y-axis, Auto, Min, Max
        row4 = QHBoxLayout()
        row4.addWidget(QLabel('Y-axis:'))
        self._lf_ylabel = QComboBox()
        self._lf_ylabel.setFixedWidth(130)
        self._lf_ylabel.addItems(['Auto'])
        row4.addWidget(self._lf_ylabel)

        self._lf_auto_y = QCheckBox('Auto')
        self._lf_auto_y.setChecked(True)
        self._lf_auto_y.stateChanged.connect(self._on_lf_auto_y_changed)
        row4.addWidget(self._lf_auto_y)

        row4.addWidget(QLabel('Min:'))
        self._lf_ymin = QLineEdit()
        self._lf_ymin.setPlaceholderText('YMin')
        self._lf_ymin.setMaximumWidth(80)
        self._lf_ymin.setEnabled(False)
        self._lf_ymin.editingFinished.connect(self._on_lineout_fit_changed)
        row4.addWidget(self._lf_ymin)

        row4.addWidget(QLabel('Max:'))
        self._lf_ymax = QLineEdit()
        self._lf_ymax.setPlaceholderText('YMax')
        self._lf_ymax.setMaximumWidth(80)
        self._lf_ymax.setEnabled(False)
        self._lf_ymax.editingFinished.connect(self._on_lineout_fit_changed)
        row4.addWidget(self._lf_ymax)

        row4.addStretch()
        layout.addLayout(row4)

        return widget

    # ----- 1D panel callbacks -----

    def _on_1d_auto_x_changed(self):
        auto = self._1d_auto_x.isChecked()
        self._1d_xmin.setEnabled(not auto)
        self._1d_xmax.setEnabled(not auto)
        if auto:
            self._1d_xmin.clear()
            self._1d_xmax.clear()
        self._on_1d_changed()

    def _on_1d_auto_y_changed(self):
        auto = self._1d_auto_range.isChecked()
        self._1d_ymin.setEnabled(not auto)
        self._1d_ymax.setEnabled(not auto)
        if auto:
            self._1d_ymin.clear()
            self._1d_ymax.clear()
        self._on_1d_changed()

    def _on_1d_changed(self, *_args):
        self._update_1d_window()

    def _on_toggle_1d(self, checked):
        if checked:
            if self._1d_window is None:
                self._1d_window = Lineout1DWindow(parent=self)
            self._1d_window.show()
            self._update_1d_window()
        else:
            if self._1d_window is not None:
                self._1d_window.hide()

    def _get_1d_opts(self):
        xmin = xmax = ymin = ymax = None
        try:
            if self._1d_xmin.text():
                xmin = float(self._1d_xmin.text())
            if self._1d_xmax.text():
                xmax = float(self._1d_xmax.text())
            if self._1d_ymin.text():
                ymin = float(self._1d_ymin.text())
            if self._1d_ymax.text():
                ymax = float(self._1d_ymax.text())
        except ValueError:
            pass
        return {
            'xaxis_key': self._1d_xaxis.currentText(),
            'auto_x': self._1d_auto_x.isChecked(),
            'xmin': xmin,
            'xmax': xmax,
            'log_y': self._1d_log_y.isChecked(),
            'legend': self._1d_legend.isChecked(),
            'auto_range': self._1d_auto_range.isChecked(),
            'ymin': ymin,
            'ymax': ymax,
            'marker': self._1d_marker.currentData(),
        }

    # ----- 2D panel callbacks -----

    def _on_2d_auto_x_changed(self):
        auto = self._2d_auto_x.isChecked()
        self._2d_xmin.setEnabled(not auto)
        self._2d_xmax.setEnabled(not auto)
        if auto:
            self._2d_xmin.clear()
            self._2d_xmax.clear()
        self._on_2d_changed()

    def _on_2d_auto_y_changed(self):
        auto = self._2d_auto_y.isChecked()
        self._2d_ymin.setEnabled(not auto)
        self._2d_ymax.setEnabled(not auto)
        if auto:
            self._2d_ymin.clear()
            self._2d_ymax.clear()
        self._on_2d_changed()

    def _on_2d_auto_int_changed(self):
        auto = self._2d_auto_int.isChecked()
        self._2d_imin.setEnabled(not auto)
        self._2d_imax.setEnabled(not auto)
        if auto:
            self._2d_imin.clear()
            self._2d_imax.clear()
        self._on_2d_changed()

    def _on_2d_changed(self, *_args):
        self._update_2d_windows()

    def _on_toggle_2d(self, checked):
        if checked:
            if self.lineouts is None:
                self.statusBar().showMessage('Load a file first')
                self._2d_toggle.setChecked(False)
                return
            data, dataset_name = self._get_active_data()
            win = Lineout2DWindow(parent=self)
            win.show()
            win.set_data(data, self.x_axes, dataset_name, self._get_2d_opts())
            self._2d_windows.append(win)
        else:
            for win in self._2d_windows:
                if win.isVisible():
                    win.hide()

    def _get_2d_opts(self):
        xmin = xmax = ymin = ymax = imin = imax = None
        try:
            if self._2d_xmin.text():
                xmin = float(self._2d_xmin.text())
            if self._2d_xmax.text():
                xmax = float(self._2d_xmax.text())
            if self._2d_ymin.text():
                ymin = int(float(self._2d_ymin.text()))
            if self._2d_ymax.text():
                ymax = int(float(self._2d_ymax.text()))
            if self._2d_imin.text():
                imin = float(self._2d_imin.text())
            if self._2d_imax.text():
                imax = float(self._2d_imax.text())
        except ValueError:
            pass
        return {
            'xaxis_key': self._2d_xaxis.currentText(),
            'cmap': self._2d_cmap.currentText(),
            'log_scale': self._2d_log.isChecked(),
            'auto_x': self._2d_auto_x.isChecked(),
            'xmin': xmin,
            'xmax': xmax,
            'auto_y': self._2d_auto_y.isChecked(),
            'ymin': ymin,
            'ymax': ymax,
            'auto_int': self._2d_auto_int.isChecked(),
            'imin': imin,
            'imax': imax,
        }

    # ----- OmegaSumFrame panel callbacks -----

    def _on_omega_auto_x_changed(self):
        auto = self._om_auto_x.isChecked()
        self._om_xmin.setEnabled(not auto)
        self._om_xmax.setEnabled(not auto)
        if auto:
            self._om_xmin.clear()
            self._om_xmax.clear()
        self._on_omega_changed()

    def _on_omega_auto_y_changed(self):
        auto = self._om_auto_y.isChecked()
        self._om_ymin.setEnabled(not auto)
        self._om_ymax.setEnabled(not auto)
        if auto:
            self._om_ymin.clear()
            self._om_ymax.clear()
        self._on_omega_changed()

    def _on_omega_auto_int_changed(self):
        auto = self._om_auto_int.isChecked()
        self._om_imin.setEnabled(not auto)
        self._om_imax.setEnabled(not auto)
        if auto:
            self._om_imin.clear()
            self._om_imax.clear()
        self._on_omega_changed()

    def _on_omega_changed(self, *_args):
        self._update_omega_window()

    def _on_toggle_omega(self, checked):
        if checked:
            if self._omega_dataset is None:
                self.statusBar().showMessage('No OmegaSumFrame in this file')
                self._om_toggle.setChecked(False)
                return
            if self._omega_window is None:
                self._omega_window = OmegaSumWindow(parent=self)
            self._omega_window.show()
            self._update_omega_window()
        else:
            if self._omega_window is not None:
                self._omega_window.hide()

    def _get_omega_opts(self):
        xmin = xmax = ymin = ymax = imin = imax = None
        try:
            if self._om_xmin.text():
                xmin = float(self._om_xmin.text())
            if self._om_xmax.text():
                xmax = float(self._om_xmax.text())
            if self._om_ymin.text():
                ymin = float(self._om_ymin.text())
            if self._om_ymax.text():
                ymax = float(self._om_ymax.text())
            if self._om_imin.text():
                imin = float(self._om_imin.text())
            if self._om_imax.text():
                imax = float(self._om_imax.text())
        except ValueError:
            pass
        return {
            'xaxis_key': self._om_xaxis.currentText(),
            'cmap': self._om_cmap.currentText(),
            'log_scale': self._om_log.isChecked(),
            'auto_x': self._om_auto_x.isChecked(),
            'xmin': xmin,
            'xmax': xmax,
            'auto_y': self._om_auto_y.isChecked(),
            'ymin': ymin,
            'ymax': ymax,
            'auto_int': self._om_auto_int.isChecked(),
            'imin': imin,
            'imax': imax,
        }

    # ----- Fit Results panel callbacks -----

    def _on_fit_auto_x_changed(self):
        auto = self._fr_auto_x.isChecked()
        self._fr_xmin.setEnabled(not auto)
        self._fr_xmax.setEnabled(not auto)
        if auto:
            self._fr_xmin.clear()
            self._fr_xmax.clear()
        self._on_fit_changed()

    def _on_fit_auto_y_changed(self):
        auto = self._fr_auto_y.isChecked()
        self._fr_ymin.setEnabled(not auto)
        self._fr_ymax.setEnabled(not auto)
        if auto:
            self._fr_ymin.clear()
            self._fr_ymax.clear()
        self._on_fit_changed()

    def _on_fit_phase_changed(self, phase):
        self._fr_peak.setText('All')
        self._on_fit_changed()

    def _step_peak(self, peak_edit, direction, fit_data, phase_combo):
        phase = phase_combo.currentText()
        phase_d = fit_data.get(phase)
        n_peaks = phase_d['params'].shape[1] if phase_d is not None else 0
        cur = peak_edit.text().strip()
        if cur.lower() == 'all' or cur == '':
            idx = 0 if direction > 0 else n_peaks - 1
        else:
            try:
                idx = int(cur) + direction
            except ValueError:
                idx = 0
        if idx < 0 or idx >= n_peaks:
            peak_edit.setText('All')
        else:
            peak_edit.setText(str(idx))
        peak_edit.editingFinished.emit()

    def _on_norm_vals_changed(self):
        if self._fr_norm.isChecked():
            self._update_fit_window()

    def _on_fit_changed(self, *_args):
        self._update_fit_window()

    def _on_toggle_fit(self, checked):
        if checked:
            if not self._fit_data:
                self.statusBar().showMessage('No fit results in this file')
                self._fr_toggle.setChecked(False)
                return
            if self._fit_window is None:
                self._fit_window = FitResultWindow(parent=self)
            self._fit_window.show()
            self._update_fit_window()
        else:
            if self._fit_window is not None:
                self._fit_window.hide()
            if self._fit_window_polar is not None:
                self._fit_window_polar.hide()

    def _get_fit_opts(self):
        xmin = xmax = ymin = ymax = None
        try:
            if self._fr_xmin.text():
                xmin = float(self._fr_xmin.text())
            if self._fr_xmax.text():
                xmax = float(self._fr_xmax.text())
            if self._fr_ymin.text():
                ymin = float(self._fr_ymin.text())
            if self._fr_ymax.text():
                ymax = float(self._fr_ymax.text())
        except ValueError:
            pass
        param_map = {'Amplitude': 0, 'Center': 1, 'Width': 2, 'Eta': 3, 'Chi²': 4,
                     'Amplitude Err': 5, 'Center Err': 6, 'Width Err': 7, 'Eta Err': 8}
        indices = self._get_indices()
        frame_idx = indices[0] if indices else 0
        norm_vals = []
        if self._fr_norm_vals.text().strip():
            try:
                norm_vals = [float(x) for x in
                             self._fr_norm_vals.text().replace(',', ' ').split()]
            except ValueError:
                pass
        return {
            'phase': self._fr_phase.currentText(),
            'peak': self._fr_peak.text().strip() or 'All',
            'param_idx': param_map.get(self._fr_param.currentText(), 1),
            'frame_idx': frame_idx,
            'plot_type': self._fr_plot_type.currentText(),
            'auto_x': self._fr_auto_x.isChecked(),
            'xmin': xmin,
            'xmax': xmax,
            'auto_y': self._fr_auto_y.isChecked(),
            'ymin': ymin,
            'ymax': ymax,
            'norm_enabled': self._fr_norm.isChecked(),
            'norm_mode': self._fr_norm_mode.currentText(),
            'norm_values': norm_vals,
            'norm_scaling': float(self._fr_scaling.text() or '1e6'),
            'radial_unit': self._1d_xaxis.currentText(),
            **self._mask_dialog.get_filters(),
        }

    # ----- Lineout Fit Results panel callbacks -----

    def _on_lf_auto_x_changed(self):
        auto = self._lf_auto_x.isChecked()
        self._lf_xmin.setEnabled(not auto)
        self._lf_xmax.setEnabled(not auto)
        if auto:
            self._lf_xmin.clear()
            self._lf_xmax.clear()
        self._on_lineout_fit_changed()

    def _on_lf_auto_y_changed(self):
        auto = self._lf_auto_y.isChecked()
        self._lf_ymin.setEnabled(not auto)
        self._lf_ymax.setEnabled(not auto)
        if auto:
            self._lf_ymin.clear()
            self._lf_ymax.clear()
        self._on_lineout_fit_changed()

    def _on_lf_phase_changed(self, phase):
        self._lf_peak.setText('All')
        self._on_lineout_fit_changed()

    def _on_lf_norm_vals_changed(self):
        if self._lf_norm.isChecked():
            self._update_lineout_fit_window()

    def _on_lineout_fit_changed(self, *_args):
        self._update_lineout_fit_window()

    def _on_toggle_lineout_fit(self, checked):
        if checked:
            if not self._lineout_fit_data:
                self.statusBar().showMessage('No lineout fit results in this file')
                self._lf_toggle.setChecked(False)
                return
            if self._lineout_fit_window is None:
                self._lineout_fit_window = LineoutFitWindow(parent=self)
            self._lineout_fit_window.show()
            self._update_lineout_fit_window()
        else:
            if self._lineout_fit_window is not None:
                self._lineout_fit_window.hide()

    def _get_lineout_fit_opts(self):
        xmin = xmax = ymin = ymax = None
        try:
            if self._lf_xmin.text():
                xmin = float(self._lf_xmin.text())
            if self._lf_xmax.text():
                xmax = float(self._lf_xmax.text())
            if self._lf_ymin.text():
                ymin = float(self._lf_ymin.text())
            if self._lf_ymax.text():
                ymax = float(self._lf_ymax.text())
        except ValueError:
            pass
        param_map = {'Amplitude': 0, 'Center': 1, 'Width': 2, 'Eta': 3, 'Chi²': 4,
                     'Amplitude Err': 5, 'Center Err': 6, 'Width Err': 7, 'Eta Err': 8}
        norm_vals = []
        if self._lf_norm_vals.text().strip():
            try:
                norm_vals = [float(x) for x in
                             self._lf_norm_vals.text().replace(',', ' ').split()]
            except ValueError:
                pass
        return {
            'phase': self._lf_phase.currentText(),
            'peak': self._lf_peak.text().strip() or 'All',
            'param_idx': param_map.get(self._lf_param.currentText(), 1),
            'auto_x': self._lf_auto_x.isChecked(),
            'xmin': xmin,
            'xmax': xmax,
            'auto_y': self._lf_auto_y.isChecked(),
            'ymin': ymin,
            'ymax': ymax,
            'norm_enabled': self._lf_norm.isChecked(),
            'norm_mode': self._lf_norm_mode.currentText(),
            'norm_values': norm_vals,
            'norm_scaling': float(self._lf_scaling.text() or '1e6'),
            'radial_unit': self._1d_xaxis.currentText(),
            **self._lf_mask_dialog.get_filters(),
        }

    # --- File handling ---

    def _populate_file_list(self, directory, select_file=None):
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        h5_files = sorted(
            f for f in os.listdir(directory)
            if f.endswith(('.h5', '.hdf5', '.hdf'))
        )
        for f in h5_files:
            self.file_combo.addItem(f, userData=os.path.join(directory, f))
        if select_file:
            basename = os.path.basename(select_file)
            idx = self.file_combo.findText(basename)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
        self.file_combo.blockSignals(False)

    @handle_errors
    def browse_file(self, checked=False):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select HDF5 File', '',
            'HDF5 Files (*.h5 *.hdf5 *.hdf);;All Files (*)')
        if filename:
            self._populate_file_list(os.path.dirname(filename), select_file=filename)
            self.load_file()

    def on_file_selected(self, index):
        self.load_file()

    @handle_errors
    def load_file(self, checked=False):
        idx = self.file_combo.currentIndex()
        filepath = self.file_combo.itemData(idx) if idx >= 0 else self.file_combo.currentText().strip()
        if not filepath:
            raise ValueError("Please specify a file path.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        self._close_h5()

        f = h5py.File(filepath, 'r')

        has_fit = 'fit_results' in f or 'fit_results_lineout' in f
        fit_only = 'lineouts' not in f and has_fit
        if 'lineouts' not in f and not has_fit:
            f.close()
            raise KeyError('"lineouts" dataset not found in file.')

        if fit_only:
            self.lineouts = None
            self.lineouts_mean = None
            self.x_axes = {}
            self.eta_axis = None
            self.frame_names = None
            self._omega_dataset = None
            f.close()

            self.filepath = filepath
            n_frames = None
            with h5py.File(filepath, 'r') as fh:
                for grp_name in ('fit_results', 'fit_results_lineout'):
                    if grp_name not in fh:
                        continue
                    fr = fh[grp_name]
                    for key in fr:
                        if key in ('eta', 'metadata'):
                            continue
                        if 'params' in fr[key]:
                            n_frames = fr[key]['params'].shape[0]
                            break
                    if n_frames is not None:
                        break
            if n_frames is None:
                raise KeyError('No valid fit results found in file.')
            self.n_frames = n_frames

            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.blockSignals(False)
            self.frame_info_label.setText(f'Frames: {self.n_frames}')
            self.statusBar().showMessage(
                f'Loaded {os.path.basename(filepath)} — '
                f'fit results only (no lineout data)')

            self._load_fit_results(filepath)
            self._update_panel_availability()
            QTimer.singleShot(0, self._update_fit_window)
            QTimer.singleShot(0, self._update_lineout_fit_window)
            return

        self.lineouts = f['lineouts'][:]
        self.lineouts_mean = f['lineouts_simple_mean'][:] if 'lineouts_simple_mean' in f else None

        n_bins = self.lineouts.shape[1]
        fallback = np.arange(n_bins, dtype=float)
        self.x_axes = {
            'R (pixel)': f['geometry_maps/R_map'][:, 0] if 'geometry_maps/R_map' in f else fallback,
            'TTH (deg)': f['geometry_maps/TTh_map'][:, 0] if 'geometry_maps/TTh_map' in f else fallback,
            'Q (1/A)':   f['geometry_maps/Q_map'][:, 0] if 'geometry_maps/Q_map' in f else fallback,
        }

        self.eta_axis = f['geometry_maps/Eta_map'][0, :] if 'geometry_maps/Eta_map' in f else None

        if 'frame_names' in f:
            raw = f['frame_names'][:]
            self.frame_names = np.array([
                n.decode() if isinstance(n, bytes) else str(n) for n in raw
            ])
        else:
            self.frame_names = None

        if 'OmegaSumFrame' in f:
            self._h5file = f
            self._omega_dataset = f['OmegaSumFrame']
        else:
            self._omega_dataset = None
            f.close()

        self.filepath = filepath
        self.n_frames = self.lineouts.shape[0]

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem('lineouts')
        if self.lineouts_mean is not None:
            self.dataset_combo.addItem('lineouts_simple_mean')
        self.dataset_combo.blockSignals(False)

        self.frame_info_label.setText(f'Frames: {self.n_frames}')
        self.statusBar().showMessage(
            f'Loaded {os.path.basename(filepath)} — '
            f'{self.n_frames} frames, {self.lineouts.shape[1]} bins')

        self._load_fit_results(filepath)
        self._update_panel_availability()

        self._update_1d_window()
        self._update_2d_windows()
        QTimer.singleShot(0, self._update_omega_window)
        QTimer.singleShot(0, self._update_fit_window)
        QTimer.singleShot(0, self._update_lineout_fit_window)

    # --- Frame stepping ---

    def _get_step(self):
        try:
            return max(1, int(self.step_input.text()))
        except ValueError:
            return 1

    def _shift_frame_text(self, delta):
        text = self.frame_input.text().strip()
        if not text:
            return
        parts = []
        for part in text.split(','):
            part = part.strip()
            if ':' in part:
                tokens = part.split(':')
                shifted = []
                for t in tokens:
                    t = t.strip()
                    if t:
                        v = int(t) + delta
                        shifted.append(str(max(0, min(v, self.n_frames))))
                    else:
                        shifted.append('')
                parts.append(':'.join(shifted))
            else:
                v = int(part) + delta
                parts.append(str(max(0, min(v, self.n_frames - 1))))
        self.frame_input.setText(','.join(parts))
        self.on_settings_changed()

    def step_frame_backward(self):
        self._shift_frame_text(-self._get_step())

    def step_frame_forward(self):
        self._shift_frame_text(self._get_step())

    # --- Data helpers ---

    def _get_active_data(self):
        name = self.dataset_combo.currentText()
        if name == 'lineouts_simple_mean' and self.lineouts_mean is not None:
            return self.lineouts_mean, name
        return self.lineouts, 'lineouts'

    def _get_indices(self):
        return parse_frame_selection(self.frame_input.text(), self.n_frames)

    # --- Panel availability ---

    def _update_panel_availability(self):
        has_lineouts = self.lineouts is not None
        has_omega = self._omega_dataset is not None
        has_fit = bool(self._fit_data)
        has_lf = bool(self._lineout_fit_data)

        for toggle, panel, available in [
            (self._1d_toggle, self._1d_panel, has_lineouts),
            (self._2d_toggle, self._2d_panel, has_lineouts),
            (self._om_toggle, self._om_panel, has_omega),
            (self._fr_toggle, self._fr_panel, has_fit),
            (self._lf_toggle, self._lf_panel, has_lf),
        ]:
            if not available and toggle.isChecked():
                toggle.setChecked(False)
            panel.setEnabled(available)

        self._tab_widget.setTabEnabled(0, has_lineouts or has_omega)
        self._tab_widget.setTabEnabled(1, has_fit or has_lf)

    # --- Settings changed → push to windows ---

    def on_settings_changed(self, *_args):
        if self.lineouts is not None:
            self._update_1d_window()
            QTimer.singleShot(0, self._update_omega_window)
        QTimer.singleShot(0, self._update_fit_window)
        QTimer.singleShot(0, self._update_lineout_fit_window)

    def on_dataset_changed(self, *_args):
        if self.lineouts is not None:
            self._update_1d_window()
            self._update_2d_windows()
            QTimer.singleShot(0, self._update_omega_window)

    # --- 1D window update ---

    def _update_1d_window(self):
        if self._1d_window is None or not self._1d_window.isVisible():
            return
        if self.lineouts is None:
            return
        data, dataset_name = self._get_active_data()
        indices = self._get_indices()
        if len(indices) == 0:
            self.statusBar().showMessage('No valid frames selected')
            return
        self._1d_window.set_data(
            data, self.x_axes, indices, self.frame_names, dataset_name,
            len(indices) > 1, self._1d_avg.isChecked(),
            self._get_1d_opts())
        self.statusBar().showMessage(f'{dataset_name} — {len(indices)} frame(s)')

    # --- 2D window update ---

    def _update_2d_windows(self):
        if self.lineouts is None:
            return
        data, dataset_name = self._get_active_data()
        self._2d_windows = [w for w in self._2d_windows if w.isVisible()]
        opts = self._get_2d_opts()
        for win in self._2d_windows:
            win.set_data(data, self.x_axes, dataset_name, opts)

    # --- Fit Results window update ---

    def _update_fit_window(self):
        if self._fit_window is None or not self._fit_window.isVisible():
            return
        if not self._fit_data:
            return
        opts = self._get_fit_opts()
        plot_type = opts.get('plot_type', 'Cartesian')

        if plot_type == 'Both':
            cart_opts = dict(opts, plot_type='Cartesian')
            self._fit_window.set_data(self._fit_data, self._fit_eta, cart_opts)
            if self._fit_window_polar is None:
                self._fit_window_polar = FitResultWindow(parent=self)
                self._fit_window_polar.setWindowTitle('Fit Results (Polar)')
            self._fit_window_polar.show()
            polar_opts = dict(opts, plot_type='Polar')
            self._fit_window_polar.set_data(self._fit_data, self._fit_eta, polar_opts)
        else:
            self._fit_window.set_data(self._fit_data, self._fit_eta, opts)
            if self._fit_window_polar is not None:
                self._fit_window_polar.hide()

    # --- Lineout Fit Results window update ---

    def _update_lineout_fit_window(self):
        if self._lineout_fit_window is None or not self._lineout_fit_window.isVisible():
            return
        if not self._lineout_fit_data:
            return
        self._lineout_fit_window.set_data(self._lineout_fit_data,
                                          self._get_lineout_fit_opts())

    # --- OmegaSumFrame window update ---

    def _update_omega_window(self):
        if self._omega_window is None or not self._omega_window.isVisible():
            return
        if self._omega_dataset is None:
            return
        indices = self._get_indices()
        if len(indices) == 0:
            return
        frame_idx = indices[0]
        self._omega_window.set_data(
            self._omega_dataset[frame_idx], self.x_axes, self.eta_axis,
            frame_idx, self._get_omega_opts())

    # --- Fit result loading ---

    def _load_fit_results(self, filepath):
        """Load fit results from inplace file or separate _fitresults.h5."""
        self._fit_data = {}
        self._fit_eta = None
        self._lineout_fit_data = {}

        sources = [filepath]
        base, ext = os.path.splitext(filepath)
        sources.append(base + '_fitresults' + ext)

        for src in sources:
            try:
                with h5py.File(src, 'r') as fh:
                    if 'fit_results' in fh:
                        fr = fh['fit_results']
                        if 'eta' in fr:
                            self._fit_eta = fr['eta'][:]
                        for key in fr:
                            if key in ('eta', 'metadata'):
                                continue
                            grp = fr[key]
                            if 'params' in grp:
                                self._fit_data[key] = {
                                    'params': grp['params'][:],
                                    'errors': grp['errors'][:],
                                    'chi2': grp['chi2'][:],
                                    'mask': grp['mask'][:].astype(bool),
                                }
                    if 'fit_results_lineout' in fh:
                        fr = fh['fit_results_lineout']
                        for key in fr:
                            if key == 'metadata':
                                continue
                            grp = fr[key]
                            if 'params' in grp:
                                self._lineout_fit_data[key] = {
                                    'params': grp['params'][:],
                                    'errors': grp['errors'][:],
                                    'chi2': grp['chi2'][:],
                                }
                    if self._fit_data or self._lineout_fit_data:
                        self.statusBar().showMessage(
                            f'Fit results loaded from {os.path.basename(src)}')
                        break
            except Exception:
                continue

        # Populate phase combos
        self._fr_phase.blockSignals(True)
        self._fr_phase.clear()
        for phase in self._fit_data:
            self._fr_phase.addItem(phase)
        self._fr_phase.blockSignals(False)
        if self._fit_data:
            self._on_fit_phase_changed(self._fr_phase.currentText())

        self._lf_phase.blockSignals(True)
        self._lf_phase.clear()
        for phase in self._lineout_fit_data:
            self._lf_phase.addItem(phase)
        self._lf_phase.blockSignals(False)
        if self._lineout_fit_data:
            self._on_lf_phase_changed(self._lf_phase.currentText())

    # --- HDF5 file handle ---

    def _close_h5(self):
        self._omega_dataset = None
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def closeEvent(self, event):
        self._close_h5()
        super().closeEvent(event)

    # --- Error dialog ---

    def show_error_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox {
        font-size: 12pt;}
    """)
    viewer = WAXSViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
