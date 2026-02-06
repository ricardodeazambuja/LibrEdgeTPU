# LibrEdgeTPU Interactive GUI

## Developer Guide

This document is for developers who want to understand the GUI architecture, add new algorithm modes, or modify the interface.

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  index.html  │  │  style.css   │  │   app.js     │       │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘       │
│         │                                    │               │
│         │ SSE (metrics)                      │ POST (clicks) │
│         │ MJPEG (video)                      │               │
└─────────┼────────────────────────────────────┼───────────────┘
          ↓                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                      Flask Server (app.py)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    VideoStream                        │   │
│  │  ┌──────────┐  ┌───────────────┐  ┌──────────────┐  │   │
│  │  │  Camera  │  │ AlgorithmMode │  │ Mouse State  │  │   │
│  │  └────┬─────┘  └───────┬───────┘  └──────────────┘  │   │
│  │       │                │                             │   │
│  │       └────────────────┴─────────────────────────────┘   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          ↓                                    ↓
┌──────────────────┐              ┌────────────────────────┐
│  Real Camera     │              │  Algorithm (Hardware)  │
│  (cv2.VideoCapture)             │  - SpotTracker         │
│                  │              │  - PatternTracker      │
│  Synthetic Camera│              │  - etc.                │
│  (pattern gen)   │              │                        │
└──────────────────┘              │  Algorithm (Synthetic) │
                                  │  - NumPy CPU fallback  │
                                  └────────────────────────┘
```

### Component Responsibilities

**Frontend (HTML/CSS/JS):**
- Render UI layout and controls
- Draw MJPEG stream to canvas
- Capture mouse events (click, drag)
- Send interaction to backend via POST
- Receive metrics via Server-Sent Events
- Handle screenshot download

**Backend (Flask):**
- Serve static files (HTML/CSS/JS)
- Stream MJPEG video via `/video_feed`
- Stream metrics via `/metrics` (SSE)
- Handle mouse interaction via POST endpoints
- Manage algorithm switching
- Thread-safe state management

**VideoStream:**
- Owns camera and algorithm instances
- Processes frames through current algorithm
- Tracks performance metrics (FPS, latency)
- Manages mouse interaction state
- Pause/play state with frozen frame preservation
- Thread-safe locking

**AlgorithmMode:**
- Base class for algorithm-specific processing
- `process(frame, mouse_state)` → annotated frame
- Handles both hardware (Edge TPU) and synthetic (CPU) modes
- Uses overlay helpers for consistent rendering

**Camera:**
- `RealCamera`: OpenCV VideoCapture wrapper
- `SyntheticCamera`: Pattern generators for offline testing
- Unified `read()` interface

**Overlay:**
- Reusable OpenCV drawing functions
- Extracted from `test_visual_robotics.py`
- Consistent colors, fonts, styles

## File Organization

```
libredgetpu/gui/
├── __init__.py              # Package exports
├── __main__.py              # CLI entry point
├── app.py                   # Flask server + routes (400 lines)
├── algorithm_modes.py       # 9 AlgorithmMode classes (800 lines)
├── camera.py                # Camera abstractions (300 lines)
├── overlay.py               # OpenCV rendering (200 lines)
├── templates/
│   └── index.html           # Main page (200 lines)
└── static/
    ├── style.css            # UI styles (150 lines)
    └── app.js               # Frontend logic (250 lines)
```

## Adding a New Algorithm Mode

### 1. Define AlgorithmMode Class

In `algorithm_modes.py`:

```python
class MyAlgorithmMode(AlgorithmMode):
    """Brief description."""

    def __init__(self, synthetic: bool = False):
        super().__init__(synthetic)

        if not synthetic and HARDWARE_AVAILABLE:
            try:
                # Initialize hardware algorithm
                self.algorithm = MyAlgorithm.from_template(...)
            except Exception as e:
                print(f"Hardware init failed, using synthetic: {e}")
                self.synthetic = True

        # Initialize synthetic mode state if needed

    def process(self, frame: np.ndarray, mouse_state: Dict[str, Any]) -> np.ndarray:
        """Process frame and return annotated result."""
        annotated = frame.copy()

        # Handle mouse interactions
        if mouse_state.get("clicked"):
            x, y = mouse_state["x"], mouse_state["y"]
            # ... do something with click ...

        # Measure latency
        t0 = time.perf_counter()

        if self.synthetic:
            # CPU fallback implementation
            result = self._synthetic_process(frame)
        else:
            # Hardware Edge TPU implementation
            result = self.algorithm.process(frame)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        # Draw overlays using overlay helpers
        overlay.draw_crosshair(annotated, x, y)
        overlay.draw_text_with_background(annotated, f"Result: {result}", (10, 30))

        return annotated

    def get_metrics(self) -> Dict[str, Any]:
        """Return algorithm-specific metrics."""
        metrics = super().get_metrics()
        metrics["custom_metric"] = self.some_value
        return metrics
```

### 2. Register Algorithm

In `algorithm_modes.py`:

```python
ALGORITHM_MODES = {
    # ... existing algorithms ...
    "MyAlgorithm": MyAlgorithmMode,
}
```

### 3. Update Frontend (Optional)

Add instructions in `templates/index.html`:

```html
<p><strong>MyAlgorithm:</strong> Click to do something</p>
```

### 4. Test

```bash
python -m libredgetpu.gui
# Select "MyAlgorithm" from dropdown
# Test interaction
```

## Mouse Interaction Protocol

**State dictionary** passed to `AlgorithmMode.process()`:

```python
{
    # One-shot events (cleared after processing)
    "clicked": True,           # Simple click occurred
    "drag_end": {"x": 123, "y": 456},  # Drag finished

    # Persistent state
    "x": 123,                  # Current mouse X
    "y": 456,                  # Current mouse Y
    "dragging": True,          # Drag in progress
    "drag_start": {"x": 100, "y": 200},  # Drag origin
}
```

**Event flow:**

1. User mousedown → POST `/drag_start` → `drag_start`, `dragging=True`
2. User mousemove (while down) → POST `/drag_move` → `x`, `y` updated
3. User mouseup → POST `/drag_end` → `drag_end`, `dragging=False`
4. User click (down+up quickly) → POST `/click` → `clicked=True`

**Thread safety:**
- `VideoStream.mouse_state` protected by lock
- `update_mouse_state()` merges new state
- `clear_mouse_state()` clears one-shot events after processing

## Overlay Helpers

Available in `overlay.py`:

### Drawing Functions

```python
# Shapes
overlay.draw_crosshair(img, x, y, color, size, thickness)
overlay.draw_flow_arrow(img, dx, dy, scale, color, thickness)
overlay.draw_compass_needle(img, yaw_deg, radius, center, color, thickness)

# Text
overlay.draw_text_with_background(img, text, position, font_scale, thickness, text_color, bg_color, padding)

# Heatmaps
overlay.draw_heatmap_grid(img, values, grid_rows, grid_cols, alpha)
overlay.value_to_heatmap_color(value, vmin, vmax) -> (b, g, r)

# Plots
overlay.draw_scatter_plot(img, x_data, y_data, position, size, title, color)
overlay.draw_histogram(img, data, position, size, bins, title, color)

# HUD
overlay.draw_performance_hud(img, fps, latency_ms, mode)
```

### Color Constants

```python
overlay.COLOR_GREEN   # (0, 255, 0)
overlay.COLOR_RED     # (0, 0, 255)
overlay.COLOR_CYAN    # (255, 255, 0)
overlay.COLOR_YELLOW  # (0, 255, 255)
overlay.COLOR_WHITE   # (255, 255, 255)
overlay.COLOR_BLACK   # (0, 0, 0)
```

## Camera Sources

### RealCamera

```python
from libredgetpu.gui.camera import RealCamera

camera = RealCamera(camera_id=0, width=640, height=480)
success, frame = camera.read()  # Returns (bool, np.ndarray or None)
camera.release()
```

### SyntheticCamera

```python
from libredgetpu.gui.camera import SyntheticCamera

camera = SyntheticCamera(width=640, height=480, pattern="wandering_dot", fps=30.0)
success, frame = camera.read()  # Always True, generates pattern
```

**Patterns:**
- `noise` - Random RGB noise
- `checkerboard` - Static checkerboard
- `rotating` - Rotating line pattern (30°/s)
- `panning` - Horizontal scrolling stripes (50 px/s)
- `wandering_dot` - Lissajous curve Gaussian dot

### Factory

```python
from libredgetpu.gui.camera import create_camera

camera = create_camera(
    camera_id=0,        # Webcam device (None = auto)
    synthetic=False,    # Force synthetic mode
    pattern="noise",    # Synthetic pattern
    width=640,
    height=480
)
```

## Flask Routes

| Route | Method | Purpose | Returns |
|-------|--------|---------|---------|
| `/` | GET | Main page | HTML |
| `/video_feed` | GET | MJPEG stream | `multipart/x-mixed-replace` |
| `/metrics` | GET | SSE metrics | `text/event-stream` |
| `/switch_algorithm` | POST | Change algorithm | JSON `{success, algorithm}` |
| `/click` | POST | Mouse click | JSON `{success}` |
| `/drag_start` | POST | Drag start | JSON `{success}` |
| `/drag_move` | POST | Drag move | JSON `{success}` |
| `/drag_end` | POST | Drag end | JSON `{success}` |
| `/screenshot` | GET | Download frame | JPEG (attachment) |

## Performance Considerations

### MJPEG Overhead

- Encoding: ~50-100 ms per frame (depends on resolution)
- Bandwidth: ~10-20× larger than H.264
- Latency: One-way ~50-150 ms (depends on network)

**Optimization:**
- Reduce resolution (`--width 320 --height 240`)
- Lower JPEG quality (currently 85)
- Use LAN instead of WiFi

### FPS Calculation

```python
# Rolling average over last 30 frames
frame_times = [t1-t0, t2-t1, ..., t30-t29]
avg_frame_time = sum(frame_times) / len(frame_times)
fps = 1.0 / avg_frame_time
```

### Thread Safety

All access to `VideoStream.algorithm` and `VideoStream.mouse_state` is protected by `VideoStream.lock`.

**Lock hierarchy:**
1. Flask request handlers acquire lock
2. `get_frame()` acquires lock
3. No nested locks

## Testing

### Manual QA Checklist

✓ Launch without hardware (`--synthetic`)
✓ Launch with hardware
✓ Switch algorithms via dropdown
✓ Click interaction per algorithm
✓ Drag interaction (PatternTracker)
✓ Screenshot download
✓ FPS/latency metrics update
✓ Graceful camera disconnect handling
✓ Network access from remote device

### Unit Testing

The GUI is designed for manual testing, but you can unit test individual components:

```python
# Test algorithm mode processing
from libredgetpu.gui.algorithm_modes import SpotTrackerMode
import numpy as np

mode = SpotTrackerMode(synthetic=True)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
mouse_state = {"clicked": True, "x": 320, "y": 240}

annotated = mode.process(frame, mouse_state)
assert annotated.shape == frame.shape

# Test camera
from libredgetpu.gui.camera import SyntheticCamera

camera = SyntheticCamera(pattern="wandering_dot")
success, frame = camera.read()
assert success
assert frame.shape == (480, 640, 3)
```

## Debugging

### Enable Flask Debug Mode

**WARNING:** Only use on trusted networks (security risk)

```python
# In app.py main()
app.run(host=args.host, port=args.port, debug=True, threaded=True)
```

### Browser Console

Open browser DevTools (F12):
- **Console tab:** JavaScript errors, SSE messages
- **Network tab:** MJPEG stream status, POST responses
- **Performance tab:** FPS profiling

### Backend Logs

Flask prints to stdout:
```
127.0.0.1 - - [07/Feb/2026 14:30:52] "POST /click HTTP/1.1" 200 -
127.0.0.1 - - [07/Feb/2026 14:30:52] "GET /video_feed HTTP/1.1" 200 -
```

Add logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Notes

- **No authentication** - Run on trusted network only
- **No HTTPS** - Plaintext video stream
- **No input validation** - Assumes trusted clients
- **File upload disabled** - Screenshot download only

**Production deployment:**
- Add authentication (Flask-Login, OAuth)
- Enable HTTPS (nginx reverse proxy + Let's Encrypt)
- Validate all POST inputs
- Rate limiting

## Future Enhancements

**Out of scope for initial implementation:**

1. **Multi-camera support** - Switch via hotkey
2. **Parameter tuning UI** - Sliders for algorithm params
3. **Recording with overlays** - MP4 export
4. **Batch mode** - Load image sequence instead of webcam
5. **Plugin system** - User-defined algorithm modes
6. **WebRTC streaming** - Lower latency than MJPEG
7. **Mobile optimization** - Touch events, responsive layout
8. **3D visualization** - Three.js overlays (depth, point clouds)
9. **Multi-user collaboration** - Shared session with cursors

## See Also

- [GUI_GUIDE.md](../../docs/GUI_GUIDE.md) - User guide
- [ROBOTICS_STATUS.md](../../docs/ROBOTICS_STATUS.md) - Algorithm specs
- [test_visual_robotics.py](../../tests/test_visual_robotics.py) - Overlay rendering reference
