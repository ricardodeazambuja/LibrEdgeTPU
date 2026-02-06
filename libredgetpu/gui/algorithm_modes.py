"""Algorithm-specific processing modes for interactive GUI.

Each algorithm gets an AlgorithmMode class that handles:
- Hardware and synthetic (CPU fallback) processing
- Mouse interaction state management
- Overlay rendering with algorithm-specific visualizations

Hardware modules use the from_template() → open() → use → close() lifecycle.
The AlgorithmMode base class manages open/close via cleanup().
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

# Conditional imports — set HARDWARE_AVAILABLE based on whether the
# libredgetpu package can be imported (requires pyusb, numpy, flatbuffers).
# When False and --cpu is not passed, the app will raise at startup.
try:
    from .. import (
        SpotTracker, PatternTracker, LoomingDetector, OpticalFlow,
        VisualCompass, MatMulEngine, ReservoirComputer, EmbeddingSimilarity,
        SimpleInvoker
    )
    HARDWARE_AVAILABLE = True
except Exception:
    HARDWARE_AVAILABLE = False

from . import overlay


class AlgorithmMode(ABC):
    """Base class for algorithm modes."""

    def __init__(self, synthetic: bool = False, cpu_replica: bool = False):
        """Initialize algorithm mode.

        Args:
            synthetic: Use CPU fallback instead of Edge TPU
            cpu_replica: Use CPU integer-replica pipeline (OpticalFlow/VisualCompass only)
        """
        self.synthetic = synthetic
        self.cpu_replica = cpu_replica
        self.last_latency_ms = 0.0
        self._hw_resources = []  # Track opened hardware resources for cleanup
        self.hw_error = None  # Set to error string if hardware init fails
        self._active_params: Dict[str, Any] = {}

    @classmethod
    def get_param_schema(cls) -> list:
        """Return list of tunable parameter descriptors for this algorithm.

        Each entry: {"name", "label", "type", "default", "description",
                     and optionally "options", "min", "max", "step"}
        Control types: select, number, range, checkbox
        """
        return []

    def get_active_params(self) -> Dict[str, Any]:
        """Return the active parameter values for this algorithm instance."""
        return dict(self._active_params)

    @abstractmethod
    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        """Process a frame and return annotated result.

        Args:
            frame: Input frame (BGR uint8)
            mouse_state: Dictionary with click coordinates, drag state, etc.

        Returns:
            Annotated frame (BGR uint8)
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get algorithm-specific metrics for display."""
        return {"latency_ms": round(self.last_latency_ms, 2)}

    def cleanup(self):
        """Release hardware resources."""
        for resource in self._hw_resources:
            try:
                resource.close()
            except Exception:
                pass
        self._hw_resources.clear()

    def _check_hw_error(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """If hardware init failed, draw error on frame and return it.

        Returns None if no error (caller should continue normal processing).
        """
        if self.hw_error is not None:
            annotated = frame.copy()
            overlay.draw_bottom_message(
                annotated,
                f"Edge TPU error: {self.hw_error} — Restart with --cpu for CPU mode",
                text_color=overlay.COLOR_RED
            )
            return annotated
        return None

    def _open_hw(self, resource):
        """Open a hardware resource and track it for cleanup."""
        resource.open()
        self._hw_resources.append(resource)
        return resource


class SpotTrackerMode(AlgorithmMode):
    """Visual servoing via soft argmax spot tracking."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "image_size", "label": "Image Size", "type": "select",
             "default": 64, "options": [64, 128],
             "description": "Input image resolution (pixels)"},
            {"name": "variant", "label": "Variant", "type": "select",
             "default": "bright", "options": [
                 "bright", "color_red", "color_green", "color_blue",
                 "color_yellow", "color_cyan", "color_magenta", "color_white"],
             "description": "Tracking variant (bright spot or color target)"},
        ]

    def __init__(self, synthetic: bool = False, image_size: int = 64,
                 variant: str = "bright"):
        super().__init__(synthetic)
        self.image_size = image_size
        self.variant = variant
        self._active_params = {"image_size": image_size, "variant": variant}
        self.tracker = None

        if not synthetic and HARDWARE_AVAILABLE:
            try:
                self.tracker = SpotTracker.from_template(image_size, variant=variant)
                self._open_hw(self.tracker)
            except Exception as e:
                self.hw_error = str(e)
                self.tracker = None

        self.click_position = None
        self.last_offset = (0.0, 0.0)

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        if mouse_state.get("clicked"):
            self.click_position = (mouse_state["x"], mouse_state["y"])

        t0 = time.perf_counter()

        if self.synthetic:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.click_position:
                x_frac = self.click_position[0] / w
                y_frac = self.click_position[1] / h
            else:
                max_idx = gray.argmax()
                y_max, x_max = divmod(max_idx, gray.shape[1])
                x_frac = x_max / gray.shape[1]
                y_frac = y_max / gray.shape[0]
            offset_x = (x_frac - 0.5) * 2  # Map to [-1, +1]
            offset_y = (y_frac - 0.5) * 2
        else:
            # Prepare input matching tracker's expected channels
            if self.tracker._channels == 1:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                img = frame
            # Resize to expected input size
            img_resized = cv2.resize(img, (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_AREA)
            offset_x, offset_y = self.tracker.track(img_resized)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_offset = (offset_x, offset_y)

        # Convert offset [-1, +1] to pixel coords
        x_pixel = (offset_x / 2 + 0.5) * w
        y_pixel = (offset_y / 2 + 0.5) * h
        overlay.draw_crosshair(annotated, x_pixel, y_pixel, overlay.COLOR_GREEN, size=30)

        text = f"Offset: ({offset_x:+.3f}, {offset_y:+.3f})"
        overlay.draw_text_with_background(annotated, text, (10, h - 10))

        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["x_offset"] = round(self.last_offset[0], 3)
        metrics["y_offset"] = round(self.last_offset[1], 3)
        return metrics


class PatternTrackerMode(AlgorithmMode):
    """Template matching via Conv2D correlation."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "search_size", "label": "Search Size", "type": "select",
             "default": 64, "options": [64, 128],
             "description": "Search window resolution (pixels)"},
            {"name": "kernel_size", "label": "Kernel Size", "type": "select",
             "default": 16, "options": [8, 16, 32],
             "description": "Template kernel size (pixels)"},
            {"name": "channels", "label": "Channels", "type": "select",
             "default": 1, "options": [1, 3],
             "description": "Number of input channels (1=grayscale, 3=color)"},
        ]

    def __init__(self, synthetic: bool = False, search_size: int = 64,
                 kernel_size: int = 16, channels: int = 1):
        super().__init__(synthetic)
        self._default_search_size = search_size
        self._default_kernel_size = kernel_size
        self._default_channels = channels
        self._active_params = {
            "search_size": search_size,
            "kernel_size": kernel_size,
            "channels": channels,
        }
        self.template = None
        self.template_gray = None
        self.template_roi = None
        self.drag_start = None
        self.match_position = None
        self.tracker = None
        self.search_size = None  # Store search size for resizing

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Handle drag to select template ROI
        if mouse_state.get("drag_start"):
            self.drag_start = (mouse_state["drag_start"]["x"], mouse_state["drag_start"]["y"])

        if mouse_state.get("dragging") and self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = mouse_state.get("x", x1), mouse_state.get("y", y1)
            overlay.draw_text_with_background(
                annotated, "Drag to select template", (10, h - 40),
                text_color=overlay.COLOR_YELLOW
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), overlay.COLOR_YELLOW, 2)

        if mouse_state.get("drag_end") and self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = mouse_state["drag_end"]["x"], mouse_state["drag_end"]["y"]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w > 4 and roi_h > 4:
                self.template = frame[y1:y2, x1:x2].copy()
                self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                self.template_roi = (x1, y1, roi_w, roi_h)

                # Hardware: PatternTracker.from_template(search_size, kernel_size)
                if not self.synthetic and HARDWARE_AVAILABLE:
                    try:
                        # Close previous tracker if any
                        self.cleanup()
                        self.hw_error = None  # Reset on new template selection
                        # Use configured defaults for search/kernel size
                        search_size = self._default_search_size
                        kernel_size = self._default_kernel_size
                        channels = self._default_channels
                        self.search_size = search_size  # Store for later resizing
                        self.tracker = PatternTracker.from_template(
                            search_size, kernel_size, channels=channels
                        )
                        self._open_hw(self.tracker)
                        # set_template takes grayscale patch
                        patch = cv2.resize(self.template_gray, (kernel_size, kernel_size),
                                           interpolation=cv2.INTER_AREA)
                        self.tracker.set_template(patch)
                    except Exception as e:
                        self.hw_error = str(e)
                        self.tracker = None

            self.drag_start = None

        # Perform tracking if template is set
        if self.template is not None:
            t0 = time.perf_counter()

            if self.synthetic:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(gray_frame, self.template_gray, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                match_x, match_y = max_loc
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to expected search size
                if self.search_size:
                    gray_frame = cv2.resize(gray_frame, (self.search_size, self.search_size),
                                            interpolation=cv2.INTER_AREA)
                offsets = self.tracker.track(gray_frame)
                # offsets are [-1, +1] centered, convert to pixel (in original frame coords)
                match_x = int((offsets[0] / 2 + 0.5) * w)
                match_y = int((offsets[1] / 2 + 0.5) * h)

            self.last_latency_ms = (time.perf_counter() - t0) * 1000
            self.match_position = (match_x, match_y)

            th, tw = self.template.shape[:2]
            cv2.rectangle(annotated, (match_x, match_y),
                          (match_x + tw, match_y + th), overlay.COLOR_RED, 2)
            text = f"Match: ({match_x}, {match_y})"
            overlay.draw_text_with_background(annotated, text, (10, h - 10))
        else:
            overlay.draw_text_with_background(
                annotated, "Drag to select a template region", (10, h - 40),
                text_color=overlay.COLOR_CYAN
            )

        return annotated


class LoomingDetectorMode(AlgorithmMode):
    """Collision avoidance via edge density analysis."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "image_size", "label": "Image Size", "type": "select",
             "default": 64, "options": [64, 128],
             "description": "Input image resolution (pixels)"},
        ]

    def __init__(self, synthetic: bool = False, image_size: int = 64):
        super().__init__(synthetic)
        self.image_size = image_size
        self._active_params = {"image_size": image_size}
        self.detector = None
        self.last_densities = np.zeros((3, 3))

        if not synthetic and HARDWARE_AVAILABLE:
            try:
                self.detector = LoomingDetector.from_template(image_size)
                self._open_hw(self.detector)
            except Exception as e:
                self.hw_error = str(e)
                self.detector = None

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        t0 = time.perf_counter()

        if self.synthetic:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            zone_h, zone_w = h // 3, w // 3
            densities = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    zone = edges[i * zone_h:(i + 1) * zone_h, j * zone_w:(j + 1) * zone_w]
                    densities[i, j] = zone.sum() / max(zone.size, 1)
        else:
            # detect() takes grayscale image, returns flat [9] float32
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize to expected input size
            gray_resized = cv2.resize(gray, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_AREA)
            result = self.detector.detect(gray_resized)
            densities = result.reshape(3, 3)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_densities = densities

        overlay.draw_heatmap_grid(annotated, densities, 3, 3, alpha=0.3)

        # Show tau value
        center = float(densities[1, 1])
        periph = float(np.mean(densities) - center) / 8 * 9  # mean of 8 peripheral zones
        periph_vals = np.concatenate([densities[0, :], densities[1, [0, 2]], densities[2, :]])
        periph_mean = float(periph_vals.mean())
        tau = center / periph_mean if periph_mean > 1e-6 else 1.0

        text = f"Tau: {tau:.3f}  Max: {densities.max():.3f}"
        overlay.draw_text_with_background(annotated, text, (10, h - 10))

        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        center = float(self.last_densities[1, 1])
        periph_vals = np.concatenate([
            self.last_densities[0, :], self.last_densities[1, [0, 2]], self.last_densities[2, :]
        ])
        periph_mean = float(periph_vals.mean())
        tau = center / periph_mean if periph_mean > 1e-6 else 1.0
        metrics["tau"] = round(tau, 3)
        return metrics


class OpticalFlowMode(AlgorithmMode):
    """Global ego-motion estimation via Gabor features."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "image_size", "label": "Image Size", "type": "select",
             "default": 64, "options": [64, 128],
             "description": "Input image resolution (pixels)"},
            {"name": "pooled", "label": "Pooled Mode", "type": "checkbox",
             "default": False,
             "description": "Fuse Gabor+AVG_POOL in one model (16x less USB)"},
            {"name": "pool_factor", "label": "Pool Factor", "type": "select",
             "default": 2, "options": [1, 2, 4, 8],
             "description": "Spatial pooling factor (higher = coarser)"},
            {"name": "search_range", "label": "Search Range", "type": "number",
             "default": 4, "min": 1, "max": 8, "step": 1,
             "description": "Max displacement to search (pooled pixels)"},
            {"name": "temperature", "label": "Temperature", "type": "range",
             "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
             "description": "Soft argmax temperature (lower = sharper peak)"},
        ]

    def __init__(self, synthetic: bool = False, image_size: int = 64,
                 cpu_replica: bool = False, pooled: bool = False,
                 pool_factor: int = 2, search_range: int = 4,
                 temperature: float = 0.1):
        super().__init__(synthetic, cpu_replica=cpu_replica)
        self.image_size = image_size
        self._active_params = {
            "image_size": image_size, "pooled": pooled,
            "pool_factor": pool_factor, "search_range": search_range,
            "temperature": temperature,
        }
        self.prev_gray = None
        self.flow_engine = None
        self.cpu_replica_engine = None
        self.last_flow = (0.0, 0.0)

        if cpu_replica:
            from .cpu_replica import CPUReplicaOpticalFlow
            self.cpu_replica_engine = CPUReplicaOpticalFlow(
                height=image_size, width=image_size
            )
        elif not synthetic and HARDWARE_AVAILABLE:
            try:
                self.flow_engine = OpticalFlow.from_template(
                    image_size, pooled=pooled, pool_factor=pool_factor,
                    search_range=search_range, temperature=temperature,
                )
                self._open_hw(self.flow_engine)
            except Exception as e:
                self.hw_error = str(e)
                self.flow_engine = None

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            overlay.draw_text_with_background(annotated, "Initializing...", (10, h - 40))
            return annotated

        t0 = time.perf_counter()

        sz = self.image_size
        if self.cpu_replica:
            prev_resized = cv2.resize(self.prev_gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            curr_resized = cv2.resize(gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            dx, dy = self.cpu_replica_engine.compute(prev_resized, curr_resized)
        elif self.synthetic:
            shift, _ = cv2.phaseCorrelate(
                self.prev_gray.astype(np.float32), gray.astype(np.float32)
            )
            dx, dy = shift
        else:
            # compute() takes two grayscale frames, returns (vx, vy)
            # Resize both frames to expected input size
            prev_resized = cv2.resize(self.prev_gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            curr_resized = cv2.resize(gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            dx, dy = self.flow_engine.compute(prev_resized, curr_resized)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_flow = (float(dx), float(dy))

        overlay.draw_flow_arrow(annotated, dx, dy, scale=5.0)

        direction = "still"
        if abs(dx) > 0.3 or abs(dy) > 0.3:
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        text = f"Flow: ({dx:.2f}, {dy:.2f}) mag={magnitude:.2f} [{direction}]"
        overlay.draw_text_with_background(annotated, text, (10, h - 10))

        self.prev_gray = gray.copy()
        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["vx"] = round(self.last_flow[0], 2)
        metrics["vy"] = round(self.last_flow[1], 2)
        return metrics


class VisualCompassMode(AlgorithmMode):
    """Yaw estimation wrapper around OpticalFlow."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "image_size", "label": "Image Size", "type": "select",
             "default": 64, "options": [64, 128],
             "description": "Input image resolution (pixels)"},
            {"name": "fov_deg", "label": "FOV (degrees)", "type": "number",
             "default": 90.0, "min": 1.0, "max": 360.0, "step": 1.0,
             "description": "Camera horizontal field of view in degrees"},
            {"name": "pooled", "label": "Pooled Mode", "type": "checkbox",
             "default": False,
             "description": "Fuse Gabor+AVG_POOL in one model (16x less USB)"},
            {"name": "pool_factor", "label": "Pool Factor", "type": "select",
             "default": 2, "options": [1, 2, 4, 8],
             "description": "Spatial pooling factor (higher = coarser)"},
            {"name": "search_range", "label": "Search Range", "type": "number",
             "default": 4, "min": 1, "max": 8, "step": 1,
             "description": "Max displacement to search (pooled pixels)"},
            {"name": "temperature", "label": "Temperature", "type": "range",
             "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
             "description": "Soft argmax temperature (lower = sharper peak)"},
        ]

    def __init__(self, synthetic: bool = False, image_size: int = 64,
                 fov_deg: float = 90.0, cpu_replica: bool = False,
                 pooled: bool = False, pool_factor: int = 2,
                 search_range: int = 4, temperature: float = 0.1):
        super().__init__(synthetic, cpu_replica=cpu_replica)
        self.image_size = image_size
        self._active_params = {
            "image_size": image_size, "fov_deg": fov_deg,
            "pooled": pooled, "pool_factor": pool_factor,
            "search_range": search_range, "temperature": temperature,
        }
        self.prev_gray = None
        self.compass = None
        self.cpu_replica_engine = None
        self.cumulative_yaw = 0.0
        self._fov_deg = fov_deg

        if cpu_replica:
            from .cpu_replica import CPUReplicaOpticalFlow
            self.cpu_replica_engine = CPUReplicaOpticalFlow(
                height=image_size, width=image_size
            )
            pool_factor_val = self.cpu_replica_engine.params.pool_factor
            self._deg_per_pooled_px = fov_deg * pool_factor_val / image_size
        elif not synthetic and HARDWARE_AVAILABLE:
            try:
                self.compass = VisualCompass.from_template(
                    image_size, fov_deg, pooled=pooled, pool_factor=pool_factor,
                    search_range=search_range, temperature=temperature,
                )
                self._open_hw(self.compass)
            except Exception as e:
                self.hw_error = str(e)
                self.compass = None

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reset yaw on click
        if mouse_state.get("clicked"):
            self.cumulative_yaw = 0.0

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            overlay.draw_text_with_background(annotated, "Initializing... (click to reset yaw)", (10, h - 40))
            return annotated

        t0 = time.perf_counter()

        sz = self.image_size
        if self.cpu_replica:
            prev_resized = cv2.resize(self.prev_gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            curr_resized = cv2.resize(gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            vx, _ = self.cpu_replica_engine.compute(prev_resized, curr_resized)
            delta_yaw = vx * self._deg_per_pooled_px
        elif self.synthetic:
            shift, _ = cv2.phaseCorrelate(
                self.prev_gray.astype(np.float32), gray.astype(np.float32)
            )
            dx, _ = shift
            delta_yaw = dx * 0.5  # Rough degrees-per-pixel estimate
        else:
            # compute_yaw() takes two grayscale frames, returns yaw delta in degrees
            # Resize both frames to expected input size
            prev_resized = cv2.resize(self.prev_gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            curr_resized = cv2.resize(gray, (sz, sz),
                                      interpolation=cv2.INTER_AREA)
            delta_yaw = self.compass.compute_yaw(prev_resized, curr_resized)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.cumulative_yaw += delta_yaw

        overlay.draw_compass_needle(annotated, self.cumulative_yaw)

        text = f"Yaw: {self.cumulative_yaw:.1f} deg (click to reset)"
        overlay.draw_text_with_background(annotated, text, (10, h - 10))

        self.prev_gray = gray.copy()
        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["yaw_deg"] = round(self.cumulative_yaw, 1)
        return metrics


class MatMulEngineMode(AlgorithmMode):
    """Runtime weight-swapping matrix multiply."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "dim", "label": "Dimension", "type": "select",
             "default": 256, "options": [256, 512, 1024],
             "description": "Matrix dimension (NxN)"},
        ]

    def __init__(self, synthetic: bool = False, dim: int = 256):
        super().__init__(synthetic)
        self.dim = dim
        self._active_params = {"dim": dim}
        self.engine = None
        self.x_data = []
        self.y_data = []

        # Generate weights within reasonable range
        self.weights = np.random.randn(dim, dim).astype(np.float32) * 0.01

        if not synthetic and HARDWARE_AVAILABLE:
            try:
                self.engine = MatMulEngine.from_template(dim)
                self._open_hw(self.engine)
                self.engine.set_weights(self.weights)
            except Exception as e:
                self.hw_error = str(e)
                self.engine = None

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()

        # Randomize weights on click
        if mouse_state.get("clicked"):
            self.weights = np.random.randn(self.dim, self.dim).astype(np.float32) * 0.01
            if self.engine is not None:
                self.engine.set_weights(self.weights)
            self.x_data = []
            self.y_data = []

        x = np.random.randn(self.dim).astype(np.float32) * 0.1

        t0 = time.perf_counter()

        if self.synthetic:
            y = self.weights @ x
        else:
            y = self.engine.matmul(x)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        # Track input-output correlation
        self.x_data.append(float(x[0]))
        self.y_data.append(float(y[0]))
        if len(self.x_data) > 100:
            self.x_data.pop(0)
            self.y_data.pop(0)

        # Draw scatter plot
        if len(self.x_data) > 1:
            overlay.draw_scatter_plot(
                annotated,
                np.array(self.x_data),
                np.array(self.y_data),
                (10, 10),
                title="x[0] vs y[0]"
            )

        # Draw histogram of output
        overlay.draw_histogram(annotated, y, (10, 180), title="Output distribution")

        text = f"Click to randomize weights ({self.dim}x{self.dim})"
        overlay.draw_text_with_background(annotated, text, (10, frame.shape[0] - 10))

        return annotated


class ReservoirComputerMode(AlgorithmMode):
    """Echo state network via MatMulEngine."""

    INPUT_DIM = 4  # Fixed input dimension for GUI demo

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "dim", "label": "Dimension", "type": "select",
             "default": 256, "options": [256, 512, 1024],
             "description": "Reservoir state dimension"},
            {"name": "spectral_radius", "label": "Spectral Radius", "type": "range",
             "default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05,
             "description": "Reservoir weight spectral radius"},
            {"name": "leak_rate", "label": "Leak Rate", "type": "range",
             "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01,
             "description": "Leaky integrator rate (1.0 = no leak)"},
            {"name": "activation", "label": "Activation", "type": "select",
             "default": "tanh", "options": ["tanh", "relu", "identity"],
             "description": "Reservoir activation function"},
        ]

    def __init__(self, synthetic: bool = False, dim: int = 256,
                 spectral_radius: float = 0.95, leak_rate: float = 1.0,
                 activation: str = "tanh"):
        super().__init__(synthetic)
        self.dim = dim
        self._active_params = {
            "dim": dim, "spectral_radius": spectral_radius,
            "leak_rate": leak_rate, "activation": activation,
        }
        self.reservoir = None

        # Synthetic fallback state
        self.state = np.zeros(dim, dtype=np.float32)
        self.syn_w_res = np.random.randn(dim, dim).astype(np.float32) * 0.01
        self.syn_w_in = np.random.randn(dim, self.INPUT_DIM).astype(np.float32) * 0.1

        if not synthetic and HARDWARE_AVAILABLE:
            try:
                self.reservoir = ReservoirComputer.from_template(
                    dim, self.INPUT_DIM, spectral_radius=spectral_radius,
                    leak_rate=leak_rate, activation=activation,
                )
                self._open_hw(self.reservoir)
            except Exception as e:
                self.hw_error = str(e)
                self.reservoir = None

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h = frame.shape[0]

        # Build input: inject spike on click, otherwise small noise
        u = np.random.randn(self.INPUT_DIM).astype(np.float32) * 0.01
        if mouse_state.get("clicked"):
            u[0] = 1.0

        t0 = time.perf_counter()

        if self.synthetic:
            # Simple ESN step: x(t) = tanh(W_res @ x(t-1) + W_in @ u)
            self.state = np.tanh(self.syn_w_res @ self.state + self.syn_w_in @ u)
            state = self.state
        else:
            # step() takes input [M], returns state [N]
            state = self.reservoir.step(u)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        # Draw state as 16x16 heatmap
        state_grid = state[:256].reshape(16, 16)
        overlay.draw_heatmap_grid(annotated, state_grid, 16, 16, alpha=0.5)

        norm = float(np.linalg.norm(state))
        text = f"State norm: {norm:.3f} (click to inject spike)"
        overlay.draw_text_with_background(annotated, text, (10, h - 10))

        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["state_norm"] = round(float(np.linalg.norm(self.state)), 3)
        return metrics


class EmbeddingSimilarityMode(AlgorithmMode):
    """Cosine similarity search via MatMulEngine."""

    @classmethod
    def get_param_schema(cls) -> list:
        return [
            {"name": "dim", "label": "Dimension", "type": "select",
             "default": 256, "options": [256, 512, 1024],
             "description": "Embedding dimension"},
        ]

    def __init__(self, synthetic: bool = False, dim: int = 256):
        super().__init__(synthetic)
        self.dim = dim
        self._active_params = {"dim": dim}
        self.similarity = None
        self.gallery_thumbnails = []  # List of (label, thumbnail) for display
        self.gallery_embeddings = []  # Synthetic: store embeddings directly
        self.gallery_count = 0

        if not synthetic and HARDWARE_AVAILABLE:
            try:
                self.similarity = EmbeddingSimilarity.from_template(dim)
                self._open_hw(self.similarity)
            except Exception as e:
                self.hw_error = str(e)
                self.similarity = None

    def _frame_to_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Generate a pseudo-embedding from frame (mean color + spatial stats)."""
        # Use spatial statistics as a simple embedding
        h, w = frame.shape[:2]
        # Divide into 8x8 grid, compute mean per cell per channel
        grid_h, grid_w = 8, 8
        cell_h, cell_w = h // grid_h, w // grid_w
        features = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = frame[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                features.append(cell.mean(axis=(0, 1)))  # [3] RGB means
        features = np.concatenate(features).astype(np.float32)  # [192]
        # Pad or truncate to dim
        emb = np.zeros(self.dim, dtype=np.float32)
        n = min(len(features), self.dim)
        emb[:n] = features[:n]
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb /= norm
        return emb

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Add to gallery on click (max 10)
        if mouse_state.get("clicked") and self.gallery_count < 10:
            embedding = self._frame_to_embedding(frame)
            thumbnail = cv2.resize(frame, (64, 48))
            label = f"img_{self.gallery_count}"

            if self.synthetic:
                self.gallery_embeddings.append(embedding)
            else:
                # EmbeddingSimilarity.add(label, embedding)
                self.similarity.add(label, embedding)

            self.gallery_thumbnails.append((label, thumbnail))
            self.gallery_count += 1

        if self.gallery_count > 0:
            query = self._frame_to_embedding(frame)

            t0 = time.perf_counter()

            if self.synthetic:
                mat = np.array(self.gallery_embeddings)
                scores = mat @ query
                # Build sorted results
                indices = np.argsort(scores)[::-1]
                results = [(self.gallery_thumbnails[i][0], float(scores[i]))
                           for i in indices[:3]]
            else:
                # query(embedding, top_k) returns [(label, score), ...]
                results = self.similarity.query(query, top_k=3)

            self.last_latency_ms = (time.perf_counter() - t0) * 1000

            # Draw top-3 thumbnails
            for i, (lbl, score) in enumerate(results):
                # Find thumbnail by label
                thumb = None
                for gl, gt in self.gallery_thumbnails:
                    if gl == lbl:
                        thumb = gt
                        break
                if thumb is not None:
                    y_off = 100 + i * 60
                    if y_off + 48 < h and 74 < w:
                        annotated[y_off:y_off + 48, 10:74] = thumb
                        overlay.draw_text_with_background(
                            annotated, f"#{i+1} {score:.3f}", (80, y_off + 25)
                        )

        text = f"Gallery: {self.gallery_count}/10 (click to add)"
        overlay.draw_text_with_background(annotated, text, (10, h - 10))

        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["gallery_size"] = self.gallery_count
        return metrics


class SimpleInvokerMode(AlgorithmMode):
    """Standard ML inference with model selection and post-processing."""

    @classmethod
    def get_param_schema(cls) -> list:
        from .model_zoo import get_model_names
        return [
            {"name": "model", "label": "Model", "type": "select",
             "default": "Classification (MobileNet V1)",
             "options": get_model_names(),
             "description": "Pre-trained model with post-processing"},
        ]

    def __init__(self, synthetic: bool = False,
                 model: str = "Classification (MobileNet V1)"):
        super().__init__(synthetic)
        from .model_zoo import MODEL_REGISTRY, download_model, download_labels

        self._active_params = {"model": model}
        self._model_meta = MODEL_REGISTRY.get(model)
        self._invoker = None
        self._tflite_bytes = None
        self._labels = []

        if self._model_meta is None:
            self.hw_error = f"Unknown model: {model}"
            return

        if synthetic:
            # No hardware — show informative message in process()
            return

        if not HARDWARE_AVAILABLE:
            self.hw_error = "Edge TPU not available"
            return

        try:
            model_path = download_model(model)
            with open(model_path, "rb") as f:
                self._tflite_bytes = f.read()
            self._invoker = SimpleInvoker(model_path)
            self._open_hw(self._invoker)
            self._labels = download_labels(self._model_meta.get("labels"))
        except ImportError as e:
            self.hw_error = str(e)
        except Exception as e:
            self.hw_error = str(e)

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        err_frame = self._check_hw_error(frame)
        if err_frame is not None:
            return err_frame

        annotated = frame.copy()

        if self.synthetic:
            overlay.draw_bottom_message(
                annotated,
                "SimpleInvoker requires Edge TPU hardware. "
                "Run without --synthetic to use real inference.",
                text_color=overlay.COLOR_CYAN
            )
            return annotated

        meta = self._model_meta
        input_w, input_h = meta["input_size"]

        # Resize frame to model input size
        resized = cv2.resize(frame, (input_w, input_h),
                             interpolation=cv2.INTER_AREA)

        # Prepare input bytes
        if meta["input_type"] == "int8":
            input_int8 = (resized.astype(np.float32) - 127).astype(np.int8)
            input_bytes = (input_int8.view(np.uint8) ^ 0x80).tobytes()
        else:
            input_bytes = resized.tobytes()

        t0 = time.perf_counter()
        raw_outputs = self._invoker.invoke_raw_outputs(input_bytes)
        output_layers = self._invoker.output_layers
        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        # Dispatch to post-processor
        pp = meta["postprocessor"]
        if pp == "classification":
            self._process_classification(annotated, raw_outputs)
        elif pp == "ssd_detection":
            self._process_detection(annotated, raw_outputs, output_layers)
        elif pp == "deeplabv3":
            self._process_segmentation(annotated, raw_outputs, output_layers)
        elif pp == "posenet":
            self._process_posenet(annotated, raw_outputs, output_layers, input_w, input_h)
        elif pp == "multipose":
            self._process_multipose(annotated, raw_outputs, output_layers, input_w, input_h)

        return annotated

    def _process_classification(self, frame, raw_outputs):
        scores = np.frombuffer(raw_outputs[0], dtype=np.uint8).astype(np.float32)
        top5_idx = np.argsort(scores)[-5:][::-1]
        top_labels = []
        for idx in top5_idx:
            label = self._labels[idx] if idx < len(self._labels) else f"class {idx}"
            top_labels.append((label, scores[idx]))
        overlay.draw_classification_labels(frame, top_labels)

    def _process_detection(self, frame, raw_outputs, output_layers):
        from ..postprocess.ssd_decoder import postprocess_ssd
        detections = postprocess_ssd(raw_outputs, output_layers,
                                     self._tflite_bytes, score_threshold=0.5)
        overlay.draw_bounding_boxes(frame, detections, self._labels)

    def _process_segmentation(self, frame, raw_outputs, output_layers):
        from ..postprocess.deeplabv3 import postprocess_deeplabv3
        seg_map = postprocess_deeplabv3(raw_outputs, output_layers,
                                        self._tflite_bytes)
        overlay.draw_segmentation_overlay(frame, seg_map)

    def _process_posenet(self, frame, raw_outputs, output_layers, input_w, input_h):
        from ..postprocess.posenet_decoder import postprocess_posenet
        poses = postprocess_posenet(raw_outputs, output_layers,
                                    self._tflite_bytes)
        overlay.draw_skeleton(frame, poses, (input_w, input_h))

    def _process_multipose(self, frame, raw_outputs, output_layers, input_w, input_h):
        from ..postprocess.multipose_decoder import postprocess_multipose
        poses = postprocess_multipose(raw_outputs, output_layers,
                                      self._tflite_bytes)
        # Filter: keep only poses with at least 3 confident keypoints
        good_poses = [p for p in poses
                      if int(np.sum(p.keypoint_scores > 0.3)) >= 3]
        overlay.draw_skeleton(frame, good_poses, (input_w, input_h))


# Algorithm registry
ALGORITHM_MODES = {
    "SpotTracker": SpotTrackerMode,
    "PatternTracker": PatternTrackerMode,
    "LoomingDetector": LoomingDetectorMode,
    "OpticalFlow": OpticalFlowMode,
    "VisualCompass": VisualCompassMode,
    "MatMulEngine": MatMulEngineMode,
    "ReservoirComputer": ReservoirComputerMode,
    "EmbeddingSimilarity": EmbeddingSimilarityMode,
    "SimpleInvoker": SimpleInvokerMode,
}
