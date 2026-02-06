#!/usr/bin/env python3
"""Visual proof tests for robotics modules: synthetic scenes on Edge TPU.

Produces annotated output images proving correct behavior for:
  1. SpotTracker (bright) — 3x3 grid of Gaussian dots, flat + textured bg
  2. SpotTracker (color_red) — 3x3 grid of red dots on RGB, flat + textured bg
  3. LoomingDetector — 3 scenes with zone heatmaps
  4. PatternTracker — checkerboard template at 3 positions
  5. MatMulEngine — reversal transform, scatter, error histogram

Usage:
    python -m tests.test_visual_robotics                    # all 5
    python -m tests.test_visual_robotics --models spot_tracker spot_tracker_color
"""

import argparse
import math
import os
import sys
import urllib.request

import numpy as np
import pytest

pytest.importorskip("PIL", reason="Pillow is required: pip install Pillow")
from PIL import Image, ImageDraw, ImageFont

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(img, name):
    """Save image to results directory."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    path = os.path.join(_RESULTS_DIR, name)
    img.save(path)
    print(f"  Saved: {path}")
    return path


def _get_font(size=14):
    """Try to get a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def _draw_text_bg(draw, xy, text, font, fg=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with a solid background rectangle."""
    bbox = draw.textbbox(xy, text, font=font)
    pad = 2
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg)
    draw.text(xy, text, fill=fg, font=font)


def _draw_crosshair(draw, cx, cy, size, color, width=2, style="plus"):
    """Draw a + or x crosshair marker."""
    s = size // 2
    if style == "plus":
        draw.line([(cx - s, cy), (cx + s, cy)], fill=color, width=width)
        draw.line([(cx, cy - s), (cx, cy + s)], fill=color, width=width)
    else:  # "x"
        draw.line([(cx - s, cy - s), (cx + s, cy + s)], fill=color, width=width)
        draw.line([(cx - s, cy + s), (cx + s, cy - s)], fill=color, width=width)


def _draw_line_plot(draw, bbox, values, color, y_range=None, width=2):
    """Draw a simple line plot inside a bounding rectangle.

    Args:
        bbox: (x0, y0, x1, y1) plot area.
        values: list/array of y values.
        color: line color tuple.
        y_range: (ymin, ymax) or None for auto.
    """
    x0, y0, x1, y1 = bbox
    n = len(values)
    if n < 2:
        return
    vals = np.asarray(values, dtype=np.float64)
    if y_range is None:
        ymin, ymax = float(vals.min()), float(vals.max())
    else:
        ymin, ymax = y_range
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0

    plot_w = x1 - x0
    plot_h = y1 - y0
    points = []
    for i, v in enumerate(vals):
        px = x0 + i * plot_w / (n - 1)
        py = y1 - (v - ymin) / (ymax - ymin) * plot_h
        points.append((px, py))
    draw.line(points, fill=color, width=width)


def _draw_scatter(draw, bbox, x_vals, y_vals, color, radius=2):
    """Draw a scatter plot inside a bounding rectangle."""
    x0, y0, x1, y1 = bbox
    xv = np.asarray(x_vals, dtype=np.float64)
    yv = np.asarray(y_vals, dtype=np.float64)

    xmin, xmax = float(xv.min()), float(xv.max())
    ymin, ymax = float(yv.min()), float(yv.max())
    if xmax - xmin < 1e-12:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0

    plot_w = x1 - x0
    plot_h = y1 - y0

    # Reference y=x line
    ref_min = max(xmin, ymin)
    ref_max = min(xmax, ymax)
    if ref_max > ref_min:
        rx0 = x0 + (ref_min - xmin) / (xmax - xmin) * plot_w
        ry0 = y1 - (ref_min - ymin) / (ymax - ymin) * plot_h
        rx1 = x0 + (ref_max - xmin) / (xmax - xmin) * plot_w
        ry1 = y1 - (ref_max - ymin) / (ymax - ymin) * plot_h
        draw.line([(rx0, ry0), (rx1, ry1)], fill=(128, 128, 128), width=1)

    for xi, yi in zip(xv, yv):
        px = x0 + (xi - xmin) / (xmax - xmin) * plot_w
        py = y1 - (yi - ymin) / (ymax - ymin) * plot_h
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     fill=color)


def _draw_histogram(draw, bbox, values, n_bins=30, color=(100, 180, 255)):
    """Draw a bar histogram inside a bounding rectangle."""
    x0, y0, x1, y1 = bbox
    vals = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(vals, bins=n_bins)
    max_count = float(counts.max()) if counts.max() > 0 else 1.0

    plot_w = x1 - x0
    plot_h = y1 - y0
    bar_w = plot_w / n_bins

    for i, c in enumerate(counts):
        bx0 = x0 + i * bar_w
        bx1 = bx0 + bar_w - 1
        bar_h = (c / max_count) * plot_h
        by0 = y1 - bar_h
        if bar_h > 0:
            draw.rectangle([bx0, by0, bx1, y1], fill=color)


def _val_to_heat(val, vmin, vmax):
    """Map scalar to blue -> red color tuple."""
    if vmax - vmin < 1e-12:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    # blue(0) -> cyan(0.25) -> green(0.5) -> yellow(0.75) -> red(1.0)
    r = int(255 * min(1.0, max(0.0, 2.0 * t - 0.5)))
    g = int(255 * min(1.0, max(0.0, 1.0 - abs(2.0 * t - 1.0))))
    b = int(255 * min(1.0, max(0.0, 1.0 - 2.0 * t)))
    return (r, g, b)


def _make_gaussian_dot(size, cx, cy, radius=6, peak=255, bg=16):
    """Create a grayscale image with a Gaussian dot at (cx, cy)."""
    img = np.full((size, size), bg, dtype=np.float64)
    yy, xx = np.mgrid[:size, :size]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    img += (peak - bg) * np.exp(-dist_sq / (2.0 * radius ** 2))
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_gaussian_dot_rgb(size, cx, cy, radius=6, dot_color=(255, 32, 32),
                           bg_color=(32, 32, 32)):
    """Create an RGB image with a colored Gaussian dot at (cx, cy)."""
    img = np.full((size, size, 3), bg_color, dtype=np.float64)
    yy, xx = np.mgrid[:size, :size]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    gauss = np.exp(-dist_sq / (2.0 * radius ** 2))
    for c in range(3):
        img[:, :, c] += (dot_color[c] - bg_color[c]) * gauss
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_noisy_gaussian_dot(size, cx, cy, radius=6, peak=255,
                             noise_mean=80, noise_std=30, seed=0):
    """Create a grayscale image with a Gaussian dot on a noisy background."""
    rng = np.random.RandomState(seed)
    img = rng.normal(noise_mean, noise_std, (size, size))
    yy, xx = np.mgrid[:size, :size]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    gauss = np.exp(-dist_sq / (2.0 * radius ** 2))
    img += (peak - img) * gauss
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_noisy_color_dot_rgb(size, cx, cy, radius=6,
                              dot_color=(255, 32, 32),
                              noise_mean=80, noise_std=30, seed=0):
    """Create an RGB image with a colored Gaussian dot on a noisy background."""
    rng = np.random.RandomState(seed)
    img = rng.normal(noise_mean, noise_std, (size, size, 3))
    yy, xx = np.mgrid[:size, :size]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    gauss = np.exp(-dist_sq / (2.0 * radius ** 2))
    for c in range(3):
        img[:, :, c] += (dot_color[c] - img[:, :, c]) * gauss
    return np.clip(img, 0, 255).astype(np.uint8)


def _run_spot_grid(tracker, make_scene, size, positions, panel_size=200,
                   gap=4, tolerance=0.15):
    """Run tracker on dot positions and return an annotated 3x3 grid image.

    Args:
        tracker: opened SpotTracker instance.
        make_scene: callable(cx, cy) -> ndarray, scene with dot at (cx, cy).
        size: image dimension (e.g. 64).
        positions: list of 9 (cx, cy) tuples.
        panel_size: display size per cell (pixels).
        gap: gap between cells (pixels).
        tolerance: max error for PASS.

    Returns:
        (grid_image, pass_count, max_err, mean_err)
    """
    grid_n = 3
    grid_w = grid_n * panel_size + (grid_n - 1) * gap
    grid_h = grid_w
    canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font_sm = _get_font(10)

    errors = []
    pass_count = 0
    mid = size // 2

    for idx, (dot_x, dot_y) in enumerate(positions):
        row, col = divmod(idx, grid_n)
        img = make_scene(dot_x, dot_y)

        x_off, y_off = tracker.track(img, resize=False)

        exp_x = (dot_x - mid) / mid
        exp_y = (dot_y - mid) / mid

        err_x = abs(x_off - exp_x)
        err_y = abs(y_off - exp_y)
        err = max(err_x, err_y)
        errors.append(err)
        passed = err < tolerance
        if passed:
            pass_count += 1

        # Draw panel
        px = col * (panel_size + gap)
        py = row * (panel_size + gap)

        if img.ndim == 2:
            panel_img = Image.fromarray(img).convert("RGB")
        else:
            panel_img = Image.fromarray(img)
        panel_img = panel_img.resize((panel_size, panel_size), Image.NEAREST)
        canvas.paste(panel_img, (px, py))

        scale = panel_size / size

        # Green + at expected position
        ecx = int(px + dot_x * scale)
        ecy = int(py + dot_y * scale)
        _draw_crosshair(draw, ecx, ecy, 16, (0, 255, 0), 2, "plus")

        # Red x at detected position
        det_px = x_off * mid + mid
        det_py = y_off * mid + mid
        dcx = int(px + det_px * scale)
        dcy = int(py + det_py * scale)
        _draw_crosshair(draw, dcx, dcy, 14, (255, 60, 60), 2, "x")

        # Text overlay
        status_color = (0, 255, 0) if passed else (255, 60, 60)
        label = f"exp({exp_x:+.2f},{exp_y:+.2f}) det({x_off:+.2f},{y_off:+.2f})"
        err_label = f"err={err:.3f} {'OK' if passed else 'FAIL'}"

        text_y = py + panel_size - 32
        _draw_text_bg(draw, (px + 4, text_y), label, font_sm,
                      fg=(200, 200, 200), bg=(0, 0, 0))
        _draw_text_bg(draw, (px + 4, text_y + 14), err_label, font_sm,
                      fg=status_color, bg=(0, 0, 0))

        print(f"    [{idx}] pos=({dot_x},{dot_y}) exp=({exp_x:+.2f},{exp_y:+.2f}) "
              f"det=({x_off:+.2f},{y_off:+.2f}) err={err:.3f} "
              f"{'PASS' if passed else 'FAIL'}")

    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    return canvas, pass_count, max_err, mean_err


def _compose_spot_test(tracker, title, flat_fn, noisy_fn, size, positions):
    """Run flat + textured grids and compose into a single image.

    Returns:
        (canvas, flat_pass, noisy_pass)
    """
    panel = 200
    gap = 4
    grid_n = 3
    header_h = 25
    summary_h = 22
    section_gap = 8

    grid_w = grid_n * panel + (grid_n - 1) * gap
    grid_h = grid_w
    canvas_h = 2 * (header_h + grid_h + summary_h) + section_gap + 30
    canvas = Image.new("RGB", (grid_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(12)

    y = 0

    # ── Flat background ──
    _draw_text_bg(draw, (10, y + 4), "Flat Background", font,
                  fg=(255, 255, 255), bg=(50, 50, 50))
    y += header_h

    print("  --- Flat Background ---")
    flat_grid, flat_pass, flat_max, flat_mean = _run_spot_grid(
        tracker, flat_fn, size, positions, panel, gap)
    canvas.paste(flat_grid, (0, y))
    y += grid_h

    flat_summary = (f"Flat: {flat_pass}/9 pass "
                    f"(max_err={flat_max:.3f}, mean_err={flat_mean:.3f})")
    flat_color = (0, 255, 0) if flat_pass == 9 else (255, 200, 0)
    _draw_text_bg(draw, (10, y + 2), flat_summary, font,
                  fg=flat_color, bg=(0, 0, 0))
    y += summary_h + section_gap

    # ── Textured background ──
    _draw_text_bg(draw, (10, y + 4), "Textured Background", font,
                  fg=(255, 255, 255), bg=(50, 50, 50))
    y += header_h

    print("  --- Textured Background ---")
    noisy_grid, noisy_pass, noisy_max, noisy_mean = _run_spot_grid(
        tracker, noisy_fn, size, positions, panel, gap)
    canvas.paste(noisy_grid, (0, y))
    y += grid_h

    noisy_summary = (f"Textured: {noisy_pass}/9 pass "
                     f"(max_err={noisy_max:.3f}, mean_err={noisy_mean:.3f})")
    noisy_color = (0, 255, 0) if noisy_pass == 9 else (255, 200, 0)
    _draw_text_bg(draw, (10, y + 2), noisy_summary, font,
                  fg=noisy_color, bg=(0, 0, 0))
    y += summary_h

    # ── Overall ──
    overall = f"{title}: Flat {flat_pass}/9, Textured {noisy_pass}/9"
    total = flat_pass + noisy_pass
    overall_color = (0, 255, 0) if total == 18 else (255, 200, 0)
    _draw_text_bg(draw, (10, y + 4), overall, font,
                  fg=overall_color, bg=(0, 0, 0))
    print(f"  Overall: {overall}")

    return canvas, flat_pass, noisy_pass


# ── 1. SpotTracker (bright) ────────────────────────────────────────────────

@pytest.mark.hardware
def test_spot_tracker():
    """Run SpotTracker (bright) on flat + textured backgrounds."""
    print("\n[SpotTracker] 64x64 bright tracker — flat + textured backgrounds")
    from libredgetpu.spot_tracker import SpotTracker

    SIZE = 64
    tracker = SpotTracker.from_template(SIZE, variant="bright")

    margin = 10
    mid = SIZE // 2
    far = SIZE - margin - 1
    positions = [
        (margin, margin), (mid, margin), (far, margin),
        (margin, mid),    (mid, mid),    (far, mid),
        (margin, far),    (mid, far),    (far, far),
    ]

    def flat_scene(cx, cy):
        return _make_gaussian_dot(SIZE, cx, cy, radius=6, peak=255, bg=16)

    def noisy_scene(cx, cy):
        return _make_noisy_gaussian_dot(
            SIZE, cx, cy, radius=6, peak=255, seed=cx * 100 + cy)

    with tracker:
        canvas, _, _ = _compose_spot_test(
            tracker, "SpotTracker 64x64 bright",
            flat_scene, noisy_scene, SIZE, positions)
    _save(canvas, "spot_tracker_64x64.png")


# ── 1b. SpotTracker (color_red) ────────────────────────────────────────────

@pytest.mark.hardware
def test_spot_tracker_color():
    """Run SpotTracker (color_red) on flat + textured backgrounds."""
    print("\n[SpotTracker Color] 64x64 color_red tracker — flat + textured")
    from libredgetpu.spot_tracker import SpotTracker

    SIZE = 64
    DOT_COLOR = (255, 32, 32)
    tracker = SpotTracker.from_template(SIZE, variant="color_red")

    margin = 10
    mid = SIZE // 2
    far = SIZE - margin - 1
    positions = [
        (margin, margin), (mid, margin), (far, margin),
        (margin, mid),    (mid, mid),    (far, mid),
        (margin, far),    (mid, far),    (far, far),
    ]

    def flat_scene(cx, cy):
        return _make_gaussian_dot_rgb(
            SIZE, cx, cy, radius=6, dot_color=DOT_COLOR,
            bg_color=(32, 32, 32))

    def noisy_scene(cx, cy):
        return _make_noisy_color_dot_rgb(
            SIZE, cx, cy, radius=6, dot_color=DOT_COLOR,
            seed=cx * 100 + cy + 10000)

    with tracker:
        canvas, _, _ = _compose_spot_test(
            tracker, "SpotTracker 64x64 color_red",
            flat_scene, noisy_scene, SIZE, positions)
    _save(canvas, "spot_tracker_color_64x64.png")


# ── Looming helpers ────────────────────────────────────────────────────────

def _load_real_looming_images(size):
    """Download and prepare real images for looming detection test."""
    base_url = ("https://raw.githubusercontent.com/"
                "google-coral/test_data/master")
    sources = [
        (f"{base_url}/grace_hopper.bmp", "grace_hopper.png", "Portrait"),
        (f"{base_url}/cat.bmp", "cat.png", "Cat"),
        (f"{base_url}/bird.bmp", "bird.png", "Bird"),
    ]
    scenes = []
    for url, filename, label in sources:
        local = os.path.join(_MODELS_DIR, filename)
        if not os.path.isfile(local):
            os.makedirs(_MODELS_DIR, exist_ok=True)
            tmp = local + ".tmp"
            try:
                print(f"  Downloading: {url}")
                urllib.request.urlretrieve(url, tmp)
                Image.open(tmp).save(local, "PNG")
            except Exception as e:
                print(f"  Warning: failed to download {filename}: {e}")
                continue
            finally:
                if os.path.isfile(tmp):
                    os.remove(tmp)
        img = Image.open(local).convert("L").resize(
            (size, size), Image.BILINEAR)
        scenes.append((np.array(img, dtype=np.uint8), label))
    return scenes


def _render_looming_row(draw, canvas, detector, scenes, start_x, start_y,
                        panel_w, gap, font, font_sm, font_tau):
    """Render a horizontal row of looming detection scenes.

    Each scene gets: title, input image, 3x3 zone heatmap, tau value,
    interpretation label, and center/peripheral detail.

    Returns (tau_list, max_y) where tau_list is [(label, tau), ...].
    """
    from libredgetpu.looming_detector import LoomingDetector

    tau_list = []
    max_y = start_y
    cell_size = panel_w // 3

    for i, (scene, label) in enumerate(scenes):
        px = start_x + i * (panel_w + gap)
        py = start_y

        # Title
        _draw_text_bg(draw, (px, py), label, font,
                      fg=(255, 255, 255), bg=(50, 50, 50))
        py += 22

        # Input image (scaled up)
        img_disp = Image.fromarray(scene).convert("RGB")
        img_disp = img_disp.resize((panel_w, panel_w), Image.NEAREST)
        canvas.paste(img_disp, (px, py))
        py += panel_w + 4

        # Run detector
        zones = detector.detect(scene)
        tau = LoomingDetector.compute_tau(zones)
        tau_list.append((label, tau))

        # 3x3 zone heatmap
        zone_grid = zones.reshape(3, 3)
        zmin, zmax = float(zones.min()), float(zones.max())

        for zr in range(3):
            for zc in range(3):
                val = zone_grid[zr, zc]
                color = _val_to_heat(val, zmin, zmax)
                cx0 = px + zc * cell_size
                cy0 = py + zr * cell_size
                draw.rectangle([cx0, cy0, cx0 + cell_size - 1,
                                cy0 + cell_size - 1], fill=color)
                val_text = f"{val:.1f}"
                _draw_text_bg(draw, (cx0 + 4, cy0 + cell_size // 2 - 6),
                              val_text, font_sm,
                              fg=(255, 255, 255), bg=(0, 0, 0))

        py += 3 * cell_size + 6

        # Tau value (prominent, colored by interpretation)
        if tau > 1.1:
            interp, interp_color = "Approaching", (255, 80, 80)
        elif tau < 0.9:
            interp, interp_color = "Receding", (80, 180, 255)
        else:
            interp, interp_color = "Stable", (180, 255, 180)

        _draw_text_bg(draw, (px, py), f"tau = {tau:.3f}", font_tau,
                      fg=interp_color, bg=(0, 0, 0))
        py += 20
        _draw_text_bg(draw, (px, py), interp, font_tau,
                      fg=interp_color, bg=(0, 0, 0))
        py += 20

        center_val = float(zones[4])
        periph_mean = float(np.mean(zones[[0, 1, 2, 3, 5, 6, 7, 8]]))
        detail = f"center={center_val:.1f}  periph={periph_mean:.1f}"
        _draw_text_bg(draw, (px, py), detail, font_sm,
                      fg=(200, 200, 200), bg=(0, 0, 0))
        py += 16

        max_y = max(max_y, py)

        print(f"  [{label}] tau={tau:.3f} center={center_val:.2f} "
              f"periph={periph_mean:.2f} -> {interp}")

    return tau_list, max_y


# ── 2. LoomingDetector ──────────────────────────────────────────────────────

@pytest.mark.hardware
def test_looming_detector():
    """Run LoomingDetector on synthetic scenes and real images."""
    print("\n[LoomingDetector] Synthetic scenes + real images (64x64)")
    from libredgetpu.looming_detector import LoomingDetector

    SIZE = 64
    detector = LoomingDetector.from_template(SIZE)

    # ── Synthetic scenes ──
    scene_a = np.full((SIZE, SIZE), 128, dtype=np.uint8)
    scene_b = np.full((SIZE, SIZE), 16, dtype=np.uint8)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dist = np.sqrt((xx - SIZE // 2) ** 2 + (yy - SIZE // 2) ** 2)
    scene_b[dist < 16] = 240
    scene_c = np.full((SIZE, SIZE), 16, dtype=np.uint8)
    scene_c[:4, :] = 240
    scene_c[-4:, :] = 240
    scene_c[:, :4] = 240
    scene_c[:, -4:] = 240

    synth = [(scene_a, "Uniform"), (scene_b, "Center Circle"),
             (scene_c, "Border Frame")]

    # ── Real images (downloaded, cached in tests/models/) ──
    real = _load_real_looming_images(SIZE)

    # ── Layout ──
    PANEL_W = 180
    GAP = 8
    N_COLS = 3
    HEADER_H = 26
    cell_size = PANEL_W // 3
    row_h = 22 + PANEL_W + 4 + 3 * cell_size + 6 + 20 + 20 + 16

    canvas_w = N_COLS * PANEL_W + (N_COLS - 1) * GAP + 20
    has_real = len(real) > 0
    section_gap = 12
    canvas_h = HEADER_H + row_h
    if has_real:
        canvas_h += section_gap + HEADER_H + row_h
    canvas_h += 50  # summary (two lines when real images present)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(12)
    font_sm = _get_font(10)
    font_tau = _get_font(14)

    all_taus = []

    with detector:
        y = 0

        # Section 1: Synthetic scenes
        _draw_text_bg(draw, (10, y + 4), "Synthetic Scenes", font,
                      fg=(255, 255, 255), bg=(60, 60, 120))
        y += HEADER_H

        print("  --- Synthetic Scenes ---")
        synth_taus, y = _render_looming_row(
            draw, canvas, detector, synth, 10, y,
            PANEL_W, GAP, font, font_sm, font_tau)
        all_taus.extend(synth_taus)

        # Section 2: Real images
        if has_real:
            y += section_gap
            _draw_text_bg(draw, (10, y + 4), "Real Images", font,
                          fg=(255, 255, 255), bg=(60, 120, 60))
            y += HEADER_H

            print("  --- Real Images ---")
            real_taus, y = _render_looming_row(
                draw, canvas, detector, real, 10, y,
                PANEL_W, GAP, font, font_sm, font_tau)
            all_taus.extend(real_taus)

    # Summary (split across lines to fit canvas width)
    synth_summary = "  |  ".join(
        f"{lbl}: {t:.3f}" for lbl, t in all_taus[:3])
    _draw_text_bg(draw, (10, y + 6), f"Synthetic: {synth_summary}", font,
                  fg=(255, 255, 255), bg=(0, 0, 0))
    if has_real:
        real_summary = "  |  ".join(
            f"{lbl}: {t:.3f}" for lbl, t in all_taus[3:])
        _draw_text_bg(draw, (10, y + 22), f"Real: {real_summary}", font,
                      fg=(255, 255, 255), bg=(0, 0, 0))

    full_summary = "Tau: " + "  |  ".join(
        f"{lbl}: {t:.3f}" for lbl, t in all_taus)
    print(f"  Summary: {full_summary}")

    _save(canvas, "looming_detector_64x64.png")


# ── Pattern tracker helpers ────────────────────────────────────────────────

def _render_pattern_section(tracker, make_search, template, search_size,
                            kernel_size, placements, canvas, draw, start_y,
                            font_sm, tolerance=0.15):
    """Render pattern tracker placements as rows on canvas.

    Args:
        make_search: callable(place_x, place_y) -> ndarray search image.

    Returns:
        (pass_count, max_err, mean_err, next_y)
    """
    row_h = 190
    row_gap = 6
    tmpl_disp_size = 80
    search_disp_size = 180
    search_x_offset = 110

    errors = []
    pass_count = 0
    mid = search_size / 2

    for idx, (place_x, place_y, label) in enumerate(placements):
        ry = start_y + idx * (row_h + row_gap)

        search_img = make_search(place_x, place_y)

        center_x = place_x + kernel_size / 2
        center_y = place_y + kernel_size / 2
        exp_x = (center_x - mid) / mid
        exp_y = (center_y - mid) / mid

        x_off, y_off = tracker.track(search_img, resize=False)

        err_x = abs(x_off - exp_x)
        err_y = abs(y_off - exp_y)
        err = max(err_x, err_y)
        errors.append(err)
        passed = err < tolerance
        if passed:
            pass_count += 1

        # Left: template scaled up
        tmpl_img = Image.fromarray(template).convert("RGB")
        tmpl_img = tmpl_img.resize((tmpl_disp_size, tmpl_disp_size),
                                   Image.NEAREST)
        canvas.paste(tmpl_img, (10, ry + 20))
        _draw_text_bg(draw, (10, ry + 2),
                      f"Template {kernel_size}x{kernel_size}", font_sm,
                      fg=(200, 200, 200), bg=(0, 0, 0))

        # Center: search image with overlay
        search_scale = search_disp_size / search_size
        search_disp = Image.fromarray(search_img).convert("RGB")
        search_disp = search_disp.resize(
            (search_disp_size, search_disp_size), Image.NEAREST)
        canvas.paste(search_disp, (search_x_offset, ry + 20))

        # Green box at placed location
        gx0 = int(search_x_offset + place_x * search_scale)
        gy0 = int(ry + 20 + place_y * search_scale)
        gx1 = int(search_x_offset + (place_x + kernel_size) * search_scale)
        gy1 = int(ry + 20 + (place_y + kernel_size) * search_scale)
        draw.rectangle([gx0, gy0, gx1, gy1], outline=(0, 255, 0), width=2)

        # Red crosshair at detected location
        det_px = x_off * mid + mid
        det_py = y_off * mid + mid
        dcx = int(search_x_offset + det_px * search_scale)
        dcy = int(ry + 20 + det_py * search_scale)
        _draw_crosshair(draw, dcx, dcy, 14, (255, 60, 60), 2, "x")

        _draw_text_bg(draw, (search_x_offset, ry + 2), f"Search ({label})",
                      font_sm, fg=(200, 200, 200), bg=(0, 0, 0))

        # Right: text details
        tx = search_x_offset + search_disp_size + 16
        status_color = (0, 255, 0) if passed else (255, 60, 60)

        lines = [
            (f"Placed: ({place_x},{place_y})", (200, 200, 200)),
            (f"Expected: ({exp_x:+.2f}, {exp_y:+.2f})", (200, 200, 200)),
            (f"Detected: ({x_off:+.2f}, {y_off:+.2f})", (200, 200, 200)),
            (f"Error: {err:.3f}  {'PASS' if passed else 'FAIL'}", status_color),
        ]
        for li, (text, color) in enumerate(lines):
            _draw_text_bg(draw, (tx, ry + 24 + li * 18), text, font_sm,
                          fg=color, bg=(0, 0, 0))

        print(f"    [{label}] exp=({exp_x:+.2f},{exp_y:+.2f}) "
              f"det=({x_off:+.2f},{y_off:+.2f}) err={err:.3f} "
              f"{'PASS' if passed else 'FAIL'}")

    max_err = max(errors) if errors else 0
    mean_err = sum(errors) / len(errors) if errors else 0
    next_y = start_y + len(placements) * (row_h + row_gap)
    return pass_count, max_err, mean_err, next_y


# ── 3. PatternTracker ───────────────────────────────────────────────────────

@pytest.mark.hardware
def test_pattern_tracker():
    """Run PatternTracker on flat + textured backgrounds."""
    print("\n[PatternTracker] Checkerboard at 3 positions — flat + textured "
          "(64x64, 8x8)")
    from libredgetpu.pattern_tracker import PatternTracker

    SEARCH = 64
    KERNEL = 8
    tracker = PatternTracker.from_template(SEARCH, KERNEL, channels=1)

    # 8x8 checkerboard template
    template = np.zeros((KERNEL, KERNEL), dtype=np.uint8)
    for r in range(KERNEL):
        for c in range(KERNEL):
            if (r + c) % 2 == 0:
                template[r, c] = 240
            else:
                template[r, c] = 16

    placements = [
        (12, 12, "Upper-left"),
        (28, 28, "Center"),
        (44, 44, "Lower-right"),
    ]

    def flat_search(px, py):
        img = np.full((SEARCH, SEARCH), 32, dtype=np.uint8)
        img[py:py + KERNEL, px:px + KERNEL] = template
        return img

    def noisy_search(px, py):
        rng = np.random.RandomState(px * 100 + py)
        img = rng.normal(80, 30, (SEARCH, SEARCH))
        img[py:py + KERNEL, px:px + KERNEL] = template.astype(np.float64)
        return np.clip(img, 0, 255).astype(np.uint8)

    # Layout
    ROW_H = 190
    ROW_GAP = 6
    CANVAS_W = 640
    HEADER_H = 25
    SUMMARY_H = 22
    SECTION_GAP = 10

    n_place = len(placements)
    section_rows_h = n_place * (ROW_H + ROW_GAP)
    section_h = HEADER_H + section_rows_h + SUMMARY_H
    canvas_h = 2 * section_h + SECTION_GAP + 30
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(12)
    font_sm = _get_font(10)

    with tracker:
        y = 0

        # ── Flat Background ──
        _draw_text_bg(draw, (10, y + 4), "Flat Background", font,
                      fg=(255, 255, 255), bg=(50, 50, 50))
        y += HEADER_H

        print("  --- Flat Background ---")
        flat_pass, flat_max, flat_mean, y = _render_pattern_section(
            tracker, flat_search, template, SEARCH, KERNEL,
            placements, canvas, draw, y, font_sm)

        flat_summary = (f"Flat: {flat_pass}/3 pass "
                        f"(max_err={flat_max:.3f}, mean_err={flat_mean:.3f})")
        flat_color = (0, 255, 0) if flat_pass == 3 else (255, 200, 0)
        _draw_text_bg(draw, (10, y + 2), flat_summary, font,
                      fg=flat_color, bg=(0, 0, 0))
        y += SUMMARY_H + SECTION_GAP

        # ── Textured Background ──
        _draw_text_bg(draw, (10, y + 4), "Textured Background", font,
                      fg=(255, 255, 255), bg=(50, 50, 50))
        y += HEADER_H

        print("  --- Textured Background ---")
        noisy_pass, noisy_max, noisy_mean, y = _render_pattern_section(
            tracker, noisy_search, template, SEARCH, KERNEL,
            placements, canvas, draw, y, font_sm)

        noisy_summary = (f"Textured: {noisy_pass}/3 pass "
                         f"(max_err={noisy_max:.3f}, mean_err={noisy_mean:.3f})")
        noisy_color = (0, 255, 0) if noisy_pass == 3 else (255, 200, 0)
        _draw_text_bg(draw, (10, y + 2), noisy_summary, font,
                      fg=noisy_color, bg=(0, 0, 0))
        y += SUMMARY_H

    # Overall summary
    overall = (f"PatternTracker 64x64/8x8: "
               f"Flat {flat_pass}/3, Textured {noisy_pass}/3")
    total = flat_pass + noisy_pass
    overall_color = (0, 255, 0) if total == 6 else (255, 200, 0)
    _draw_text_bg(draw, (10, y + 4), overall, font,
                  fg=overall_color, bg=(0, 0, 0))
    print(f"  Overall: {overall}")

    _save(canvas, "pattern_tracker_64x64_8x8.png")


# ── 4. MatMulEngine ─────────────────────────────────────────────────────────

@pytest.mark.hardware
def test_matmul_engine():
    """Run MatMulEngine with reversal, random, and error analysis."""
    print("\n[MatMulEngine] 256x256 reversal + random + error analysis")
    from libredgetpu.matmul_engine import MatMulEngine

    N = 256
    engine = MatMulEngine.from_template(N)

    CANVAS_W = 720
    PANEL_H = 160
    GAP = 10
    canvas_h = 3 * PANEL_H + 2 * GAP + 80
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(12)
    font_sm = _get_font(10)

    w_range = engine.weight_range
    if w_range is None:
        raise RuntimeError("Template missing weight_range metadata")
    wmin, wmax = w_range
    # Use a value safely within the weight range for the anti-diagonal
    diag_val = min(abs(wmin), abs(wmax)) * 0.8

    # ── Panel A: Reversal Transform ──
    print("  Panel A: Reversal transform")
    W_rev = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        W_rev[i, N - 1 - i] = diag_val

    x_input = np.array(
        [0.4 * math.sin(2 * math.pi * i / N) for i in range(N)],
        dtype=np.float32)

    numpy_rev = W_rev @ x_input
    with engine:
        engine.set_weights(W_rev)
        tpu_rev = engine.matmul(x_input)

    # Compute shared y range for overlay
    all_vals = np.concatenate([x_input, numpy_rev, tpu_rev])
    y_range = (float(all_vals.min()) * 1.1, float(all_vals.max()) * 1.1)

    py_a = 10
    # Title
    _draw_text_bg(draw, (10, py_a), "A. Reversal Transform (anti-diagonal W)",
                  font, fg=(255, 255, 255), bg=(50, 50, 50))
    py_a += 20

    plot_bbox = (60, py_a, CANVAS_W - 20, py_a + PANEL_H - 30)

    # Draw axes background
    draw.rectangle(plot_bbox, fill=(20, 20, 20), outline=(80, 80, 80))

    # Input (blue), NumPy (green dashed approx), TPU (red)
    _draw_line_plot(draw, plot_bbox, x_input, (80, 130, 255), y_range, 2)
    _draw_line_plot(draw, plot_bbox, numpy_rev, (80, 255, 80), y_range, 1)
    _draw_line_plot(draw, plot_bbox, tpu_rev, (255, 80, 80), y_range, 2)

    # Legend
    lx = 70
    ly = py_a + 4
    for label, color in [("Input", (80, 130, 255)),
                          ("NumPy", (80, 255, 80)),
                          ("TPU", (255, 80, 80))]:
        draw.rectangle([lx, ly + 2, lx + 14, ly + 10], fill=color)
        draw.text((lx + 18, ly), label, fill=color, font=font_sm)
        lx += 70

    rev_err = float(np.max(np.abs(tpu_rev - numpy_rev)))
    rev_corr = float(np.corrcoef(tpu_rev, numpy_rev)[0, 1])
    print(f"    max_error={rev_err:.4f} correlation={rev_corr:.6f}")

    # ── Panel B: Random Matrix Scatter ──
    print("  Panel B: Random matrix correlation scatter")
    py_b = PANEL_H + GAP + 10

    rng = np.random.RandomState(42)
    W_rand = rng.uniform(wmin * 0.5, wmax * 0.5, (N, N)).astype(np.float32)
    x_rand = rng.uniform(-0.3, 0.3, N).astype(np.float32)

    numpy_rand = W_rand @ x_rand
    with engine:
        engine.set_weights(W_rand)
        tpu_rand = engine.matmul(x_rand)

    # R-squared
    ss_res = float(np.sum((tpu_rand - numpy_rand) ** 2))
    ss_tot = float(np.sum((numpy_rand - np.mean(numpy_rand)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    _draw_text_bg(draw, (10, py_b), "B. Random Matrix Correlation", font,
                  fg=(255, 255, 255), bg=(50, 50, 50))
    py_b += 20

    scatter_bbox = (60, py_b, CANVAS_W // 2 - 20, py_b + PANEL_H - 30)
    draw.rectangle(scatter_bbox, fill=(20, 20, 20), outline=(80, 80, 80))
    _draw_scatter(draw, scatter_bbox, numpy_rand, tpu_rand,
                  (100, 200, 255), radius=2)

    # Annotations
    ax = CANVAS_W // 2
    _draw_text_bg(draw, (ax, py_b + 10),
                  f"R^2 = {r_squared:.6f}", font,
                  fg=(100, 255, 100), bg=(0, 0, 0))
    _draw_text_bg(draw, (ax, py_b + 30),
                  f"N = {N} points", font_sm,
                  fg=(200, 200, 200), bg=(0, 0, 0))
    _draw_text_bg(draw, (ax, py_b + 48),
                  f"x-axis: NumPy  y-axis: TPU", font_sm,
                  fg=(160, 160, 160), bg=(0, 0, 0))
    _draw_text_bg(draw, (ax, py_b + 66),
                  f"Gray line: y=x reference", font_sm,
                  fg=(128, 128, 128), bg=(0, 0, 0))

    print(f"    R^2={r_squared:.6f}")

    # ── Panel C: Error Distribution ──
    print("  Panel C: Error distribution histogram")
    py_c = 2 * (PANEL_H + GAP) + 10

    errors = tpu_rand - numpy_rand
    err_mean = float(np.mean(errors))
    err_std = float(np.std(errors))
    err_max = float(np.max(np.abs(errors)))

    _draw_text_bg(draw, (10, py_c), "C. Error Distribution (TPU - NumPy)", font,
                  fg=(255, 255, 255), bg=(50, 50, 50))
    py_c += 20

    hist_bbox = (60, py_c, CANVAS_W // 2 - 20, py_c + PANEL_H - 30)
    draw.rectangle(hist_bbox, fill=(20, 20, 20), outline=(80, 80, 80))
    _draw_histogram(draw, hist_bbox, errors, n_bins=30, color=(100, 180, 255))

    # Annotations
    _draw_text_bg(draw, (CANVAS_W // 2, py_c + 10),
                  f"mean = {err_mean:.6f}", font,
                  fg=(200, 200, 200), bg=(0, 0, 0))
    _draw_text_bg(draw, (CANVAS_W // 2, py_c + 28),
                  f"std  = {err_std:.6f}", font,
                  fg=(200, 200, 200), bg=(0, 0, 0))
    _draw_text_bg(draw, (CANVAS_W // 2, py_c + 46),
                  f"max  = {err_max:.6f}", font,
                  fg=(200, 200, 200), bg=(0, 0, 0))

    print(f"    mean={err_mean:.6f} std={err_std:.6f} max={err_max:.6f}")

    # Summary banner
    sy = 3 * PANEL_H + 2 * GAP + 30
    summary = (f"MatMulEngine 256: rev_corr={rev_corr:.4f} "
               f"R^2={r_squared:.6f} err_max={err_max:.6f}")
    _draw_text_bg(draw, (10, sy), summary, font,
                  fg=(0, 255, 0), bg=(0, 0, 0))

    _save(canvas, "matmul_engine_256.png")


# ── 5. OpticalFlow ───────────────────────────────────────────────────────────

@pytest.mark.hardware
def test_optical_flow():
    """Run OpticalFlow on synthetic shifted frames."""
    print("\n[OpticalFlow] Synthetic frame shifts (64x64)")
    from libredgetpu.optical_flow_module import OpticalFlow
    from libredgetpu.optical_flow.templates import list_templates

    SIZE = 64
    templates = list_templates()
    if not templates:
        print("  SKIPPED: No optical flow templates available. "
              "Generate with: python -m libredgetpu.optical_flow_gen")
        return

    flow = OpticalFlow.from_template(SIZE)

    CANVAS_W = 640
    ROW_H = 200
    GAP = 10
    shifts = [
        (4, 0, "Right shift (4px)"),
        (0, 4, "Down shift (4px)"),
        (-4, 0, "Left shift (4px)"),
        (0, -4, "Up shift (4px)"),
        (0, 0, "No motion"),
    ]
    n_rows = len(shifts)
    canvas_h = n_rows * (ROW_H + GAP) + 50
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(12)
    font_sm = _get_font(10)

    rng = np.random.RandomState(42)
    frame_t = rng.randint(30, 200, (SIZE, SIZE), dtype=np.uint8)

    results = []
    with flow:
        for idx, (sx, sy, label) in enumerate(shifts):
            # Create shifted frame
            frame_t1 = np.zeros_like(frame_t)
            # Shift content: positive sx = content moves right
            src_y0 = max(0, -sy)
            src_y1 = SIZE - max(0, sy)
            src_x0 = max(0, -sx)
            src_x1 = SIZE - max(0, sx)
            dst_y0 = max(0, sy)
            dst_x0 = max(0, sx)
            frame_t1[dst_y0:dst_y0 + (src_y1 - src_y0),
                      dst_x0:dst_x0 + (src_x1 - src_x0)] = \
                frame_t[src_y0:src_y1, src_x0:src_x1]

            vx, vy = flow.compute(frame_t, frame_t1)
            direction = OpticalFlow.flow_to_direction(vx, vy)
            results.append((label, sx, sy, vx, vy, direction))

            ry = idx * (ROW_H + GAP)

            # Draw frame_t
            panel_size = 150
            img_t = Image.fromarray(frame_t).convert("RGB")
            img_t = img_t.resize((panel_size, panel_size), Image.NEAREST)
            canvas.paste(img_t, (10, ry + 25))
            _draw_text_bg(draw, (10, ry + 4), f"Frame t", font_sm,
                          fg=(200, 200, 200), bg=(0, 0, 0))

            # Draw frame_t1
            img_t1 = Image.fromarray(frame_t1).convert("RGB")
            img_t1 = img_t1.resize((panel_size, panel_size), Image.NEAREST)
            canvas.paste(img_t1, (170, ry + 25))
            _draw_text_bg(draw, (170, ry + 4), f"Frame t+1 ({label})", font_sm,
                          fg=(200, 200, 200), bg=(0, 0, 0))

            # Flow vector arrow
            cx = 340 + 80
            cy = ry + 25 + panel_size // 2
            arrow_scale = 15
            ax = cx + int(vx * arrow_scale)
            ay = cy + int(vy * arrow_scale)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(128, 128, 128))
            if abs(vx) > 0.2 or abs(vy) > 0.2:
                draw.line([(cx, cy), (ax, ay)], fill=(255, 100, 100), width=3)

            # Text details
            tx = 480
            check = ""
            if sx == 0 and sy == 0:
                check = "OK" if abs(vx) < 1.0 and abs(vy) < 1.0 else "WARN"
            elif sx > 0:
                check = "OK" if vx > 0.3 else "FAIL"
            elif sx < 0:
                check = "OK" if vx < -0.3 else "FAIL"
            elif sy > 0:
                check = "OK" if vy > 0.3 else "FAIL"
            elif sy < 0:
                check = "OK" if vy < -0.3 else "FAIL"

            status_color = (0, 255, 0) if check == "OK" else (255, 100, 100)
            lines = [
                (f"Expected: dx={sx}, dy={sy}", (200, 200, 200)),
                (f"Got: vx={vx:.2f}, vy={vy:.2f}", (200, 200, 200)),
                (f"Direction: {direction}", (200, 200, 200)),
                (f"Status: {check}", status_color),
            ]
            for li, (text, color) in enumerate(lines):
                _draw_text_bg(draw, (tx, ry + 30 + li * 18), text, font_sm,
                              fg=color, bg=(0, 0, 0))

            print(f"    [{label}] expected=({sx},{sy}) "
                  f"got=({vx:.2f},{vy:.2f}) dir={direction} {check}")

    # Summary
    sy = n_rows * (ROW_H + GAP) + 10
    summary = "OpticalFlow 64x64: " + " | ".join(
        f"{r[0]}: ({r[3]:.1f},{r[4]:.1f})" for r in results)
    _draw_text_bg(draw, (10, sy), summary, font,
                  fg=(0, 255, 0), bg=(0, 0, 0))
    print(f"  Summary: {summary}")

    _save(canvas, "optical_flow_64x64.png")


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_TASKS = {
    "spot_tracker": test_spot_tracker,
    "spot_tracker_color": test_spot_tracker_color,
    "looming": test_looming_detector,
    "pattern_tracker": test_pattern_tracker,
    "matmul": test_matmul_engine,
    "optical_flow": test_optical_flow,
}


def main():
    parser = argparse.ArgumentParser(
        description="Visual proof tests for libredgetpu robotics modules")
    parser.add_argument(
        "--models", nargs="+", choices=list(ALL_TASKS.keys()),
        help="Which modules to test (default: all)")
    args = parser.parse_args()

    tasks = args.models or list(ALL_TASKS.keys())
    passed = 0
    failed = 0

    for task_name in tasks:
        func = ALL_TASKS[task_name]
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            err_str = str(e)
            if "No such device" in err_str or "No Google Coral" in err_str:
                print("\n  Device lost — stopping (physical replug required)")
                break

    print(f"\nRobotics visual tests: {passed} passed, {failed} failed")
    print(f"Results in: {_RESULTS_DIR}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
