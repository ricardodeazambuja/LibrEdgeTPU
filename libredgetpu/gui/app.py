"""Flask server for interactive LibrEdgeTPU GUI.

Provides web-based interface with:
- MJPEG video streaming
- Server-Sent Events for metrics
- Mouse interaction via POST endpoints
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, Any
from flask import Flask, Response, render_template, request, jsonify
import json

from .camera import create_camera, Camera
from .algorithm_modes import ALGORITHM_MODES, AlgorithmMode, HARDWARE_AVAILABLE
from . import overlay


class VideoStream:
    """Manages video capture and algorithm processing."""

    def __init__(self, camera: Camera, algorithm_name: str = "SpotTracker",
                 synthetic: bool = False, cpu_replica: bool = False):
        self.camera = camera
        self.algorithm_name = algorithm_name
        self.synthetic = synthetic
        self.cpu_replica = cpu_replica
        self.algorithm: Optional[AlgorithmMode] = None
        self.lock = threading.Lock()

        # Metrics
        self.fps = 0.0
        self.frame_times = []
        self.last_frame_time = time.time()

        # Mouse interaction state
        self.mouse_state: Dict[str, Any] = {}

        # Pause state
        self.paused = False
        self.frozen_frame = None  # JPEG bytes for efficient streaming
        self.frozen_frame_decoded = None  # Numpy array for algorithm processing

        # Initialize algorithm
        self._switch_algorithm(algorithm_name)

    def _switch_algorithm(self, algorithm_name: str, params: Optional[Dict[str, Any]] = None):
        """Switch to a different algorithm mode.

        Args:
            algorithm_name: Name of the algorithm to switch to
            params: Optional dict of tunable parameters to pass to the constructor
        """
        if algorithm_name not in ALGORITHM_MODES:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        with self.lock:
            # Cleanup previous algorithm's hardware resources
            if self.algorithm is not None:
                self.algorithm.cleanup()
            self.algorithm_name = algorithm_name
            self.mouse_state = {}  # Reset mouse state on switch
            mode_class = ALGORITHM_MODES[algorithm_name]
            extra = params or {}

            # cpu_replica mode: only OpticalFlow and VisualCompass support it
            if self.cpu_replica and algorithm_name in ("OpticalFlow", "VisualCompass"):
                self.algorithm = mode_class(cpu_replica=True, **extra)
            elif self.cpu_replica:
                # Unsupported mode: fall back to synthetic
                self.algorithm = mode_class(synthetic=True, **extra)
            else:
                self.algorithm = mode_class(synthetic=self.synthetic, **extra)

    def switch_algorithm(self, algorithm_name: str, params: Optional[Dict[str, Any]] = None):
        """Thread-safe algorithm switching."""
        self._switch_algorithm(algorithm_name, params)

    def apply_params(self, algorithm_name: str, params: Dict[str, Any]):
        """Re-initialize the current algorithm with new parameters."""
        self._switch_algorithm(algorithm_name, params)

    def update_mouse_state(self, state: Dict[str, Any]):
        """Update mouse interaction state."""
        with self.lock:
            self.mouse_state.update(state)

    def clear_mouse_state(self):
        """Clear mouse state after processing."""
        with self.lock:
            # Keep persistent state, clear one-shot events
            if "clicked" in self.mouse_state:
                del self.mouse_state["clicked"]
            if "drag_end" in self.mouse_state:
                del self.mouse_state["drag_end"]

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new pause state."""
        with self.lock:
            self.paused = not self.paused
            if not self.paused:
                # Clear frozen frames when resuming
                self.frozen_frame = None
                self.frozen_frame_decoded = None
            return self.paused

    def get_frame(self) -> Optional[bytes]:
        """Get latest annotated frame as JPEG bytes."""
        # Determine which frame to process
        if self.paused and self.frozen_frame_decoded is not None:
            # Use frozen frame for processing (shows rectangle, correct template)
            frame = self.frozen_frame_decoded
            is_frozen = True
        else:
            # Read new frame from camera
            success, frame = self.camera.read()
            if not success or frame is None:
                return None
            is_frozen = False

        # Process with current algorithm (now uses frozen frame if paused!)
        latency_ms = 0.0
        with self.lock:
            if self.algorithm is not None:
                try:
                    annotated = self.algorithm.process(frame, self.mouse_state.copy())
                    latency_ms = self.algorithm.last_latency_ms
                except Exception as e:
                    # Fallback: draw error on frame (bottom, word-wrapped)
                    annotated = frame.copy()
                    overlay.draw_bottom_message(
                        annotated, f"Error: {e}", text_color=overlay.COLOR_RED
                    )
            else:
                annotated = frame.copy()

        # Draw performance HUD
        if self.cpu_replica and self.algorithm and self.algorithm.cpu_replica:
            mode_str = "CPU_REPLICA"
        elif self.synthetic:
            mode_str = "SYNTHETIC"
        else:
            mode_str = "HARDWARE"
        if self.paused:
            mode_str += " [PAUSED]"
        overlay.draw_performance_hud(annotated, self.fps, latency_ms, mode_str)

        # Update FPS (only when not frozen)
        if not is_frozen:
            self._update_fps()

        # Clear one-shot mouse events
        self.clear_mouse_state()

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        # Cache frozen frame when pause is triggered
        if self.paused and not is_frozen:
            # First frame after pause: cache both JPEG and numpy array
            self.frozen_frame = frame_bytes
            self.frozen_frame_decoded = frame.copy()
        elif self.paused and is_frozen:
            # Update frozen JPEG with new annotations (rectangle, match box)
            self.frozen_frame = frame_bytes

        return frame_bytes

    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time

        # Keep last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        # Calculate FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            if self.cpu_replica and self.algorithm and self.algorithm.cpu_replica:
                mode = "CPU_REPLICA"
            elif self.synthetic:
                mode = "SYNTHETIC"
            else:
                mode = "HARDWARE"
            metrics = {
                "fps": round(self.fps, 1),
                "algorithm": self.algorithm_name,
                "mode": mode,
            }
            if self.algorithm:
                metrics.update(self.algorithm.get_metrics())
            return metrics


def _validate_params(params: dict, schema: list) -> tuple:
    """Validate user-provided params against the schema.

    Returns:
        (validated_dict, errors_list) — errors_list is empty on success.
    """
    validated = {}
    errors = []
    schema_by_name = {s["name"]: s for s in schema}

    for name, spec in schema_by_name.items():
        if name not in params:
            # Use default
            validated[name] = spec["default"]
            continue

        raw = params[name]
        ptype = spec["type"]

        if ptype == "select":
            # Coerce to match option type (int, float, or str)
            options = spec["options"]
            sample = options[0] if options else ""
            try:
                if isinstance(sample, int):
                    val = int(raw)
                elif isinstance(sample, float):
                    val = float(raw)
                else:
                    val = str(raw)
            except (ValueError, TypeError):
                errors.append(f"{spec['label']}: invalid value '{raw}'")
                continue
            if val not in options:
                errors.append(f"{spec['label']}: must be one of {options}")
                continue
            validated[name] = val

        elif ptype == "number":
            try:
                # Detect if the default is int or float
                if isinstance(spec["default"], int) and "." not in str(raw):
                    val = int(raw)
                else:
                    val = float(raw)
            except (ValueError, TypeError):
                errors.append(f"{spec['label']}: must be a number")
                continue
            lo = spec.get("min")
            hi = spec.get("max")
            if lo is not None and val < lo:
                errors.append(f"{spec['label']}: must be >= {lo}")
                continue
            if hi is not None and val > hi:
                errors.append(f"{spec['label']}: must be <= {hi}")
                continue
            validated[name] = val

        elif ptype == "range":
            try:
                val = float(raw)
            except (ValueError, TypeError):
                errors.append(f"{spec['label']}: must be a number")
                continue
            lo = spec.get("min", 0)
            hi = spec.get("max", 1)
            if val < lo or val > hi:
                errors.append(f"{spec['label']}: must be between {lo} and {hi}")
                continue
            validated[name] = val

        elif ptype == "checkbox":
            if isinstance(raw, bool):
                validated[name] = raw
            elif isinstance(raw, str):
                validated[name] = raw.lower() in ("true", "1", "yes")
            else:
                validated[name] = bool(raw)

        else:
            validated[name] = raw

    # Reject unknown params
    for name in params:
        if name not in schema_by_name:
            errors.append(f"Unknown parameter: '{name}'")

    return validated, errors


def create_app(
    camera_id: Optional[int] = None,
    synthetic_video: bool = False,
    synthetic_algorithms: bool = False,
    cpu_replica: bool = False,
    pattern: str = "wandering_dot",
    width: int = 640,
    height: int = 480,
    target_fps: int = 30
) -> Flask:
    """Create Flask app instance.

    Args:
        camera_id: Webcam device ID (None = auto-detect)
        synthetic_video: Use synthetic video instead of webcam
        synthetic_algorithms: Use CPU algorithms instead of Edge TPU
        cpu_replica: Use CPU integer-replica pipeline for OpticalFlow/VisualCompass
        pattern: Synthetic pattern type (for synthetic video)
        width: Frame width
        height: Frame height
        target_fps: Target frames per second for video stream (default: 30)

    Returns:
        Flask app instance
    """
    app = Flask(__name__)

    # Note: HARDWARE_AVAILABLE refers to Edge TPU, not webcam
    # Video source and algorithm backend are independent
    if not HARDWARE_AVAILABLE and not synthetic_algorithms and not cpu_replica:
        raise RuntimeError(
            "Edge TPU not available (import failed). "
            "Pass --cpu or --cpu-replica to run without hardware."
        )

    # Initialize camera (independent of Edge TPU availability)
    camera = create_camera(camera_id, synthetic_video, pattern, width, height)

    # Initialize video stream with separate algorithm backend control
    stream = VideoStream(camera, algorithm_name="SpotTracker",
                         synthetic=synthetic_algorithms, cpu_replica=cpu_replica)

    def generate_mjpeg():
        """Generator for MJPEG stream with FPS limiting."""
        frame_time = 1.0 / target_fps

        while True:
            start = time.perf_counter()

            frame_bytes = stream.get_frame()
            if frame_bytes is None:
                time.sleep(0.01)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Sleep to maintain target FPS
            elapsed = time.perf_counter() - start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    @app.route('/')
    def index():
        """Render main page."""
        return render_template('index.html', algorithms=list(ALGORITHM_MODES.keys()))

    @app.route('/video_feed')
    def video_feed():
        """MJPEG video stream endpoint."""
        return Response(
            generate_mjpeg(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/metrics')
    def metrics():
        """Server-Sent Events endpoint for real-time metrics."""
        def event_stream():
            while True:
                metrics_data = stream.get_metrics()
                yield f"data: {json.dumps(metrics_data)}\n\n"
                time.sleep(0.5)  # Update every 500ms

        return Response(event_stream(), mimetype='text/event-stream')

    @app.route('/switch_algorithm', methods=['POST'])
    def switch_algorithm():
        """Switch to a different algorithm."""
        data = request.get_json()
        algorithm_name = data.get('algorithm')

        if algorithm_name not in ALGORITHM_MODES:
            return jsonify({"error": f"Unknown algorithm: {algorithm_name}"}), 400

        try:
            stream.switch_algorithm(algorithm_name)
            mode_class = ALGORITHM_MODES[algorithm_name]
            schema = mode_class.get_param_schema()
            active_params = {}
            with stream.lock:
                if stream.algorithm:
                    active_params = stream.algorithm.get_active_params()
            return jsonify({
                "success": True, "algorithm": algorithm_name,
                "schema": schema, "params": active_params,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/param_schema', methods=['GET'])
    def param_schema():
        """Get parameter schema and active values for an algorithm."""
        algorithm_name = request.args.get('algorithm', stream.algorithm_name)
        if algorithm_name not in ALGORITHM_MODES:
            return jsonify({"error": f"Unknown algorithm: {algorithm_name}"}), 400

        mode_class = ALGORITHM_MODES[algorithm_name]
        schema = mode_class.get_param_schema()
        active_params = {}
        with stream.lock:
            if stream.algorithm and stream.algorithm_name == algorithm_name:
                active_params = stream.algorithm.get_active_params()
            else:
                # Return defaults from schema
                active_params = {s["name"]: s["default"] for s in schema}
        return jsonify({
            "algorithm": algorithm_name,
            "schema": schema, "params": active_params,
        })

    @app.route('/apply_params', methods=['POST'])
    def apply_params():
        """Validate and apply new parameters to the current algorithm."""
        data = request.get_json()
        algorithm_name = data.get('algorithm', stream.algorithm_name)

        if algorithm_name not in ALGORITHM_MODES:
            return jsonify({"error": f"Unknown algorithm: {algorithm_name}"}), 400

        mode_class = ALGORITHM_MODES[algorithm_name]
        schema = mode_class.get_param_schema()
        raw_params = data.get('params', {})

        validated, errors = _validate_params(raw_params, schema)
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        try:
            stream.apply_params(algorithm_name, validated)
            active_params = {}
            with stream.lock:
                if stream.algorithm:
                    active_params = stream.algorithm.get_active_params()
            return jsonify({
                "success": True, "algorithm": algorithm_name,
                "params": active_params,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/pause', methods=['POST'])
    def toggle_pause():
        """Toggle pause/play state."""
        paused = stream.toggle_pause()
        return jsonify({"success": True, "paused": paused})

    @app.route('/click', methods=['POST'])
    def handle_click():
        """Handle canvas click."""
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)

        stream.update_mouse_state({"clicked": True, "x": x, "y": y})
        return jsonify({"success": True})

    @app.route('/drag_start', methods=['POST'])
    def handle_drag_start():
        """Handle drag start."""
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)

        stream.update_mouse_state({
            "drag_start": {"x": x, "y": y},
            "dragging": True
        })
        return jsonify({"success": True})

    @app.route('/drag_move', methods=['POST'])
    def handle_drag_move():
        """Handle drag move."""
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)

        stream.update_mouse_state({"x": x, "y": y, "dragging": True})
        return jsonify({"success": True})

    @app.route('/drag_end', methods=['POST'])
    def handle_drag_end():
        """Handle drag end."""
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)

        stream.update_mouse_state({
            "drag_end": {"x": x, "y": y},
            "dragging": False
        })
        return jsonify({"success": True})

    @app.route('/screenshot', methods=['GET'])
    def screenshot():
        """Capture screenshot."""
        frame_bytes = stream.get_frame()
        if frame_bytes is None:
            return jsonify({"error": "No frame available"}), 500

        # Return as downloadable file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"libredgetpu_screenshot_{timestamp}.jpg"

        return Response(
            frame_bytes,
            mimetype='image/jpeg',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    # Note: Don't use @app.teardown_appcontext for camera cleanup!
    # It gets called after EVERY request, not just on shutdown.
    # Instead, rely on Python's garbage collection when the app exits,
    # or use signal handlers for proper cleanup on SIGTERM/SIGINT.

    return app


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LibrEdgeTPU Interactive GUI")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--camera", type=int, default=None, help="Camera device ID")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic video (default: webcam)")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU algorithms instead of Edge TPU (default: use hardware)")
    parser.add_argument("--cpu-replica", action="store_true", dest="cpu_replica",
                        help="Use CPU integer-replica of Edge TPU pipeline (OpticalFlow/VisualCompass)")
    parser.add_argument("--pattern", type=str, default="wandering_dot",
                        choices=["noise", "checkerboard", "rotating", "panning", "wandering_dot"],
                        help="Synthetic pattern type")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for video stream (default: 30)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")

    args = parser.parse_args()

    if args.cpu and args.cpu_replica:
        parser.error("--cpu and --cpu-replica are mutually exclusive")

    app = create_app(
        camera_id=args.camera,
        synthetic_video=args.synthetic,
        synthetic_algorithms=args.cpu,
        cpu_replica=args.cpu_replica,
        pattern=args.pattern,
        width=args.width,
        height=args.height,
        target_fps=args.fps
    )

    if args.cpu_replica:
        algo_mode = "CPU REPLICA (integer pipeline)"
    elif args.cpu:
        algo_mode = "CPU"
    else:
        algo_mode = "EDGE TPU"

    print(f"\n{'='*60}")
    print(f"LibrEdgeTPU Interactive GUI")
    print(f"{'='*60}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Video: {'SYNTHETIC ({})'.format(args.pattern) if args.synthetic else 'WEBCAM'}")
    print(f"Algorithms: {algo_mode}")
    print(f"Camera ID: {args.camera if args.camera is not None else 'auto-detect'}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Target FPS: {args.fps}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
