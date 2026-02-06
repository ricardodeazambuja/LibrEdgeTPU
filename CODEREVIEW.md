# LibrEdgeTPU Code Review

**Reviewed**: 2026-02-05
**Commit**: `86a83fc` (main)
**Scope**: Full repository — architecture, source code, tests, CI, documentation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Assessment](#architecture-assessment)
3. [Source Code Findings](#source-code-findings)
   - [Critical / High Severity](#critical--high-severity)
   - [Medium Severity](#medium-severity)
   - [Low Severity / Observations](#low-severity--observations)
4. [Test Suite Assessment](#test-suite-assessment)
5. [CI/CD Assessment](#cicd-assessment)
6. [Documentation Assessment](#documentation-assessment)
7. [Packaging & Build](#packaging--build)
8. [Recommendations](#recommendations)

---

## Executive Summary

LibrEdgeTPU is a pure-Python Edge TPU inference engine that independently analyzes the DarwiNN executable format and USB protocol to drive Coral accelerators directly, bypassing Google's `libedgetpu` and `tflite_runtime`. The codebase is well-structured with clean separation of concerns, excellent documentation, and a comprehensive test suite. The main areas for improvement are **defensive coding at hardware trust boundaries** (USB communication, DarwiNN parsing, firmware handling) and a few **CI/test gaps**.

### Strengths
- Clean layered architecture (transport -> driver -> delegate -> user API)
- Minimal public API: 5 classes via `__init__.py`
- Only 3 runtime dependencies (`pyusb`, `numpy`, `flatbuffers`)
- Excellent user and developer documentation (2,600+ lines)
- 151 tests with clear hardware/offline separation
- Graceful fallback: optional C extension for 2x USB speed, pure Python otherwise
- Pre-compiled templates ship with the package (no edgetpu_compiler needed at runtime)

### Key Concerns
- USB resource leaks on error paths in `transport.py`
- Missing bounds checks for DMA offsets from untrusted DarwiNN data
- Firmware download with no timeout
- Race condition in parameter cache state
- Pattern tracker tests missing from CI pipeline

---

## Architecture Assessment

### Module Dependency Graph

```
User API Layer
  SimpleInvoker ──┐
  MatMulEngine ───┤
  LoomingDetector ─┼── EdgeTPUModelBase (_base.py)
  SpotTracker ────┤         │
  PatternTracker ─┘         │
                     ┌──────┴──────┐
                     │             │
               delegate.py   tflite_parser.py
               (DarwiNN)      (TFLite FlatBuf)
                     │
               driver.py
               (HW init + execution)
                     │
               transport.py
               (USB + firmware)
                     │
               registers.py
               (HW register map)
```

**Assessment**: The layering is clean and enforced. `driver.py` never touches USB directly (delegates to `transport.py`). User-facing classes never touch the driver directly (delegates through `_base.py`). This makes the code maintainable and testable.

### Data Flow

1. **Model loading**: TFLite bytes -> `tflite_parser.py` -> `delegate.py` (DarwiNN extraction) -> `_base.py` (classification + caching)
2. **Inference**: User input -> quantize -> `_base.py._execute_raw()` -> `driver.py` -> `transport.py` (USB) -> Edge TPU -> output bytes -> dequantize -> user output
3. **Weight swapping** (MatMulEngine): float32 weights -> quantize to int8 -> generate param blob (`_generate_param_blob`) -> USB upload

### Design Decisions (Good)

- **Context manager protocol**: All user-facing classes implement `__enter__`/`__exit__` for clean USB lifecycle.
- **Cached vs standalone detection**: Automatic classification of DarwiNN executables (PARAMETER_CACHING + EXECUTION_ONLY vs STAND_ALONE) with appropriate execution paths.
- **DMA hint execution**: General-purpose `execute_dma_hints()` handles multi-chunk models (PoseNet, DeepLabV3) without special-casing.
- **Template system**: Pre-compiled models with JSON sidecar metadata avoids runtime dependency on `edgetpu_compiler`.

---

## Source Code Findings

### Critical / High Severity

#### 1. USB Resource Leak on Initialization Failure

**File**: `libredgetpu/transport.py:82-141`
**Severity**: Critical
**Confidence**: 95%

In `USBTransport.open()`, if an exception occurs after the pyusb device is found but before `self._dev` is assigned, the USB device handle leaks. Affected paths:

- Line 90: `_download_firmware(dev)` throws -> `dev` handle leaked
- Line 104: `_CUsbDevice()` constructor throws -> pyusb `dev` was already disposed but C device is leaked
- Line 107: `cdev.reset()` throws OSError -> `cdev` is not cleaned up (partially mitigated by the `except OSError: pass`)
- Line 119: Second `usb.core.find()` returns None -> first `dev` was set to None on line 92, so no leak, but device state is unknown

```python
# Current code — no try/finally for cleanup:
def open(self) -> None:
    dev = usb.core.find(idVendor=_CORAL_VID, idProduct=_CORAL_PID)
    if dev is None:
        dev = usb.core.find(idVendor=_CORAL_BOOT_VID, idProduct=_CORAL_BOOT_PID)
        if dev is None:
            raise RuntimeError("No Google Coral Edge TPU found")
        self._download_firmware(dev)  # exception here leaks dev
        ...
```

**Impact**: Repeated failed initialization attempts exhaust USB file descriptors. On embedded systems (the primary target), this can make the device unusable until reboot.

**Recommendation**: Wrap the entire `open()` body in try/except, calling `usb.util.dispose_resources(dev)` and/or `cdev.close()` in the cleanup path.

---

#### 2. Firmware Download Has No Timeout

**File**: `libredgetpu/transport.py:155-177`
**Severity**: High
**Confidence**: 90%

The firmware auto-download uses `urllib.request.urlretrieve()` with no timeout:

```python
urllib.request.urlretrieve(_FIRMWARE_URL, fw_path)
```

**Issues**:
- No timeout: can hang indefinitely on network issues or slow/malicious servers
- Downloads directly to package directory (`os.path.dirname(__file__)`) which may be read-only in installed packages
- The SHA256 check (line 163-170) is good and mitigates integrity concerns, but the hang risk remains

**Impact**: On a robot or embedded system, a network hiccup during first-time setup could block the process indefinitely. The package directory write may fail in pip-installed environments.

**Recommendation**: Use `urllib.request.urlopen(url, timeout=30)` with manual file write, or document that firmware must be pre-installed for production use.

---

#### 3. Missing Bounds Check on DMA Parameter Slicing

**File**: `libredgetpu/driver.py:256-265`
**Severity**: High
**Confidence**: 90%

In `execute_dma_hints()`, the parameter and input buffer slicing trusts DMA step offsets from the DarwiNN executable without validation:

```python
elif step.kind == "parameter":
    if params is not None:
        chunk = params[step.offset:step.offset + step.size]  # no bounds check
        t.send(chunk, TAG_PARAMETERS)
```

Python's slice operator silently returns a shorter-than-expected result when indices exceed buffer length. This means a corrupted or malicious DarwiNN executable could cause truncated data to be sent to the hardware without any error.

**Impact**: Silent data corruption sent to Edge TPU hardware. Could cause incorrect inference results, hardware state corruption, or undefined behavior in the TPU.

**Note**: The input path (line 260) has the same issue, though it is partially mitigated by the padding logic on lines 246-254 which extends input data to cover the maximum DMA range.

**Recommendation**: Add explicit bounds check:
```python
if step.offset + step.size > len(params):
    raise ValueError(f"DMA parameter step exceeds buffer: "
                     f"offset={step.offset}, size={step.size}, buf={len(params)}")
```

---

#### 4. Race Condition in Parameter Cache Token

**File**: `libredgetpu/_base.py:169-184`
**Severity**: High
**Confidence**: 85%

The `_ensure_params_cached()` method reads and writes `self._driver._cached_token` without synchronization:

```python
def _ensure_params_cached(self) -> None:
    token = self._pc_token
    if token == 0 or token != self._driver._cached_token:
        self._driver.reset_cached_parameters()
        # ... upload parameters ...
        self._driver._cached_token = token
```

If two threads call `invoke()` concurrently, both could see `token != cached_token`, both upload parameters, and corrupt the cache state. Additionally, `MatMulEngine.set_weights_raw()` (line 427) writes `self._driver._cached_token = -1` and `SpotTracker.set_color()` (line 316) writes `self._driver._cached_token = 0` — both without locks.

**Impact**: Concurrent inference calls could corrupt parameter state, leading to wrong results or hardware errors. This is especially relevant for robotics applications where multiple sensor processing threads might share a model instance.

**Recommendation**: Either add a threading lock around the cache check/upload sequence, or document clearly that instances are not thread-safe and must not be shared across threads.

---

#### 5. No Subprocess Timeout for edgetpu_compiler

**File**: `libredgetpu/matmul_engine.py:160-163` and `libredgetpu/pattern_tracker.py:408-411`
**Severity**: High
**Confidence**: 85%

Both `_recompile_and_extract_params()` functions call `subprocess.run()` without a timeout:

```python
result = subprocess.run(
    ["edgetpu_compiler", "-s", "-o", tmpdir, model_path],
    capture_output=True, text=True,
)
```

**Impact**: If `edgetpu_compiler` hangs (e.g., on a malformed model), the calling process blocks indefinitely. On a robot, this could freeze the entire control loop.

**Recommendation**: Add `timeout=60` to both calls.

---

### Medium Severity

#### 6. Negative Quantization Scale Not Validated

**Files**:
- `libredgetpu/looming_detector.py:123-126`
- `libredgetpu/spot_tracker.py:220-223`
- `libredgetpu/pattern_tracker.py:199-202`
- `libredgetpu/simple_invoker.py:71-74`

**Severity**: Medium
**Confidence**: 80%

The quantization guard `max(in_info.scale, QUANT_EPSILON)` protects against zero scale but not negative scale:

```python
quantized = np.clip(
    np.round(image.astype(np.float32) / max(in_info.scale, QUANT_EPSILON) + in_info.zero_point),
    0, 255
).astype(np.uint8)
```

If `in_info.scale` is negative (malformed TFLite model), `max(negative, 1e-9)` returns `1e-9` — which is likely wrong but at least not a crash. However, `SimpleInvoker.invoke()` (line 72) does not use the epsilon guard at all:

```python
quantized = np.clip(
    np.round(input_array / in_info.scale + in_info.zero_point),  # no guard
    0, 255
).astype(np.uint8)
```

**Impact**: A TFLite model with `scale=0.0` would cause division by zero in `SimpleInvoker.invoke()`. Other modules would produce garbage quantization with negative scales.

**Recommendation**: Validate `scale > 0` during TFLite parsing in `tflite_parser.py`, or add `abs()` + warning at each use site. Also add the epsilon guard to `SimpleInvoker.invoke()`.

---

#### 7. DarwiNN Magic Byte Search Could Match False Positives

**File**: `libredgetpu/delegate.py:280-284`
**Severity**: Medium
**Confidence**: 85%

The DarwiNN parser searches for the `"DWN1"` magic in the custom op data:

```python
magic_pos = buf.find(b"DWN1")
if magic_pos < 4:
    raise ValueError("DWN1 magic not found in custom op data")
package_offset = magic_pos - 4
```

If the custom op data happens to contain `"DWN1"` as part of a parameter tensor or instruction bitstream (before the actual Package header), the parser will misinterpret data as a flatbuffer header.

**Impact**: Corrupted parsing of a valid model, leading to incorrect parameters being sent to hardware.

**Recommendation**: After finding the magic, validate that the Package flatbuffer at that offset has a reasonable structure (e.g., valid vtable, expected version field).

---

#### 8. Firmware Upload Has No Error Recovery

**File**: `libredgetpu/transport.py:179-195`
**Severity**: Medium
**Confidence**: 82%

The firmware download loop performs 257+ USB control transfers with no error handling or progress tracking:

```python
for i in range(0, len(fw), 0x100):
    dev.ctrl_transfer(0x21, 1, cnt, 0, fw[i : i + 0x100])
    dev.ctrl_transfer(0xA1, 3, 0, 0, 6)
    cnt += 1
```

If any control transfer fails (USB disconnect, power glitch), the device is left in an unknown firmware state. The only recovery is a power cycle.

**Impact**: Partial firmware upload could leave the device unresponsive. Since this targets embedded/robotics applications, a power cycle may not be trivial.

**Recommendation**: Wrap the loop in try/except, log the failure point, and document that partial firmware upload requires a device power cycle.

---

#### 9. `buf.find(buffer_data)` May Match Wrong Buffer Location

**Files**:
- `libredgetpu/matmul_engine.py:137` (`_patch_tflite_weights`)
- `libredgetpu/pattern_tracker.py:384` (`_patch_conv2d_weights`)

**Severity**: Medium
**Confidence**: 80%

Both weight-patching functions use `buf.find(buffer_data)` to locate a weight buffer in the raw TFLite bytes:

```python
offset = buf.find(buffer_data)
if offset >= 0:
    buf[offset:offset + expected_size] = weight_bytes
```

If the same byte sequence appears elsewhere in the file (e.g., in another buffer or in flatbuffer metadata), `find()` returns the first occurrence, which may not be the weight buffer. This would corrupt the TFLite file.

**Impact**: Weight patching could corrupt non-weight data in the TFLite file, producing a malformed model that crashes or gives wrong results after compilation.

**Recommendation**: Use the TFLite flatbuffer structure to compute exact buffer offsets instead of searching by content. The `parse_full()` function already extracts buffer indices; the buffer data offset within the file could be computed from the flatbuffer vector pointer.

---

#### 10. Pre-compiled `.so` Binary Checked Into Repository

**File**: `libredgetpu/_usb_accel.cpython-311-x86_64-linux-gnu.so`
**Severity**: Medium
**Confidence**: 90%

A pre-compiled C extension binary is checked into the repository. This is problematic because:
- It only works for CPython 3.11 on x86_64 Linux
- Users on other Python versions or architectures silently fall back to the slow path with no indication
- Binary files make code review impossible and increase repository size
- The `.gitignore` excludes `*.so` but this file was committed before that rule

**Recommendation**: Remove the pre-compiled `.so` from the repository. The `setup.py` already handles building it from source. Users who want the C extension can build it with `pip install -e .` (which invokes `setup.py`).

---

### Low Severity / Observations

#### 11. Duplicated Code Across Tracker Modules

**Files**: `spot_tracker.py:148-215` and `pattern_tracker.py:129-194`

**Severity**: Low
**Confidence**: 95%

`_normalize_input()`, `_resize_image()`, `_quantize_input()`, and `_decode_output()` are nearly identical between `SpotTracker` and `PatternTracker`. This creates a maintenance burden — bugs fixed in one must be manually propagated to the other.

**Recommendation**: Extract shared methods into `EdgeTPUModelBase` or a mixin class. The only difference is the dimension property names (`_height`/`_width` vs `_search_height`/`_search_width`).

---

#### 12. Inconsistent Quantization Guard Usage

**Files**: Multiple modules

**Severity**: Low
**Confidence**: 90%

- `SimpleInvoker.invoke()` (line 72): **No** epsilon guard on division by `in_info.scale`
- `LoomingDetector.detect()` (line 124): Uses `max(in_info.scale, QUANT_EPSILON)`
- `SpotTracker._quantize_input()` (line 221): Uses `max(in_info.scale, QUANT_EPSILON)`
- `PatternTracker._quantize_input()` (line 200): Uses `max(in_info.scale, QUANT_EPSILON)`
- `MatMulEngine.matmul()` (line 488): **No** epsilon guard on division by `in_info.scale`

**Recommendation**: Standardize quantization into a single utility function in `_base.py` or `_constants.py` that all modules call.

---

#### 13. `_default_output_size()` Called Before Subclass `__init__`

**File**: `libredgetpu/_base.py:77-80`

**Severity**: Low
**Confidence**: 75%

The base class constructor calls `self._default_output_size()` (line 79) as a fallback when DarwiNN output layers are unavailable. However, this virtual method is called during `super().__init__()`, before the subclass `__init__` has run. `LoomingDetector._default_output_size()` (line 46-52) works around this with `hasattr(self, '_metadata')`, which is fragile.

**Recommendation**: Defer the output size computation to a lazy property or call it after `__init__` completes.

---

#### 14. Magic Numbers in Hardware Init

**File**: `libredgetpu/driver.py:158-173`

**Severity**: Low (informational)
**Confidence**: N/A

Phase 8 of `init_hardware()` writes magic byte values derived from USB traces:

```python
t.write_register("omc0_d4", b"\x01\x00\x00\x80")
t.write_register("rambist_ctrl_1", b"\x7f\x00\x00\x00")
t.write_register("scu_ctr_7", b"\x3f\x00\x00\x00")
```

These are documented as "from USB trace, not in libedgetpu open-source" which is appropriate given the protocol analysis context. The earlier phases (1-7) have good inline comments explaining each register write.

---

## Test Suite Assessment

### Coverage Summary

| Test Module | Tests | In CI? | Scope |
|---|---|---|---|
| `test_parse_models.py` | 12 | Yes | TFLite parsing, DarwiNN extraction (10 models) |
| `test_transport_guards.py` | 5 | Yes | USB safety barriers |
| `test_matmul_engine.py` | 25 | Yes | Weight quantization, blob generation, compilation |
| `test_looming.py` | 17 | Yes | Tau/TTC math, template loading |
| `test_spot_tracker.py` | 51 | Yes | Color templates, quantization, direction helpers |
| `test_pattern_tracker.py` | 26 | **NO** | Sidecar validation, template management |
| `test_hardware.py` | 11 | No (needs HW) | 9 real models, post-processing |
| `test_visual.py` | 4 | No (diagnostic) | Visual proof images |

**Total**: ~151 tests, ~120 offline (79%), ~31 hardware (21%)

### Strengths

- Good pytest infrastructure: fixtures, parametrization, custom `@pytest.mark.hardware` marker
- Numerical correctness validated against CPU reference implementations
- Real Edge TPU models in test suite (MobileNet, Inception, EfficientNet, SSD, PoseNet, DeepLabV3)
- Template discovery and metadata validation is thorough
- Transport guard tests ensure operations fail gracefully when device is not open

### Gaps

| Gap | Impact | Priority |
|---|---|---|
| **`test_pattern_tracker.py` missing from CI** | 26 offline tests (17% of suite) never run in CI | **P0** |
| No tests for `driver.py` | Hardware init sequence untested offline | P1 |
| No tests for `template_gen.py`, `looming_gen.py`, `spot_tracker_gen.py`, `pattern_tracker_gen.py` | Template generation untested | P2 |
| No error/recovery tests | Malformed models, USB disconnects untested | P1 |
| No thread-safety tests | Race condition (finding #4) has no test exposure | P1 |
| No code coverage tracking | Cannot measure or trend coverage | P2 |
| No `pytest-timeout` | Hanging tests could block CI indefinitely | P2 |

---

## CI/CD Assessment

**File**: `.github/workflows/test-offline.yml`

### Current Configuration

```yaml
on: push/PR to main/master
matrix: ubuntu-latest x Python [3.9, 3.11, 3.12]
steps:
  1. Install: pip install -e ".[dev]"
  2. test_parse_models.py
  3. test_matmul_engine.py -k "not hardware"
  4. test_transport_guards.py
  5. test_looming.py -k "not hardware"
  6. test_spot_tracker.py -k "not hardware"
```

### Issues

1. **Missing `test_pattern_tracker.py`**: 26 offline tests not running. This is a clear oversight — should be added as step 7.

2. **No Python 3.10 or 3.13 in matrix**: `pyproject.toml` claims support for 3.9-3.13, but CI only tests 3.9, 3.11, 3.12. Consider adding 3.10 and 3.13.

3. **No coverage reporting**: No `pytest-cov` or coverage upload to a service. Cannot track test coverage trends.

4. **No caching**: `pip install` runs from scratch each time. Adding pip caching would speed up CI.

5. **Test steps run sequentially**: Each test module runs as a separate step. If the first step fails, subsequent steps are skipped. Consider running all in one pytest invocation, or using `continue-on-error` to get full results.

---

## Documentation Assessment

### User-Facing: `README.md` (14 KB)

**Quality**: Excellent

- Clear installation instructions including libusb requirements and udev rules
- Quick start example in 5 lines
- Comprehensive API documentation for all 5 public classes
- Performance benchmark table comparing libredgetpu vs libedgetpu
- Testing instructions for offline, hardware, benchmarks, and visual tests
- Key constraints section (8-bit precision, ~8 MB weight cache, 0.28 ms latency floor)

### Developer-Facing: `docs/` (2,663 lines total)

| Document | Lines | Content |
|---|---|---|
| `ADVANCED_README.md` | 701 | Module architecture, data flow, blob format, param generation |
| `HARDWARE_ANALYSIS.md` | 1,478 | ISA, registers, DMA, patents — comprehensive HW reference |
| `MATMUL_ENGINE_NOTES.md` | 185 | MatMulEngine technical specification |
| `ROBOTICS_STATUS.md` | 299 | Module status, benchmarks, known issues |

**Quality**: Exceptional for an independent hardware analysis project. The documentation captures both the "what" and the "why" of design decisions, and includes cross-references to patents and libedgetpu source.

### Code Documentation

- All modules have module-level docstrings with usage examples
- All public methods have docstrings with args/returns/raises
- Internal functions have sufficient inline comments
- The one area that could improve is `driver.py` Phase 8 magic values — while documented as "from USB trace", the specific bit meanings are not captured

---

## Packaging & Build

### `pyproject.toml`

- Properly configured with `setuptools >= 64` build backend
- Good metadata: keywords, classifiers, URLs
- Correct optional dependency groups: `[dev]` for pytest, `[postprocess]` for Pillow
- Package data includes firmware, templates, and metadata files
- `py.typed` marker present for type checkers

### `setup.py`

- `OptionalBuildExt` class gracefully handles missing `libusb-1.0-dev`
- Uses `pkg-config` for portable dependency detection
- Falls back to pure Python without breaking the install
- `-O2` optimization enabled

### Concerns

1. The `_usb_accel.cpython-311-x86_64-linux-gnu.so` binary should not be in the repo (see finding #10)
2. The 36 MB firmware binary (`apex_latest_single_ep.bin`) ships inside the package. This is functional but makes the package large. Consider lazy downloading on first use (already implemented as fallback) and removing from the package data.
3. `requirements.txt` duplicates the `dependencies` in `pyproject.toml`. Consider removing `requirements.txt` or generating it from `pyproject.toml`.

---

## Recommendations

### Priority 1 — Immediate (bug fixes / CI)

| # | Action | Files |
|---|---|---|
| 1 | **Add `test_pattern_tracker.py` to CI** | `.github/workflows/test-offline.yml` |
| 2 | **Add try/finally cleanup to `USBTransport.open()`** | `libredgetpu/transport.py` |
| 3 | **Add bounds checks before DMA buffer slicing** | `libredgetpu/driver.py` |
| 4 | **Add `timeout=60` to all `subprocess.run()` calls** | `libredgetpu/matmul_engine.py`, `libredgetpu/pattern_tracker.py` |
| 5 | **Add epsilon guard to `SimpleInvoker.invoke()` quantization** | `libredgetpu/simple_invoker.py` |

### Priority 2 — Short-term (hardening)

| # | Action | Files |
|---|---|---|
| 6 | **Add timeout to firmware download** | `libredgetpu/transport.py` |
| 7 | **Document thread-safety constraints** (or add a lock to `_ensure_params_cached`) | `libredgetpu/_base.py`, `README.md` |
| 8 | **Remove pre-compiled `.so` from repository** | `.gitignore`, git history |
| 9 | **Validate quantization scale > 0 during TFLite parsing** | `libredgetpu/tflite_parser.py` |
| 10 | **Add DarwiNN Package header validation beyond magic bytes** | `libredgetpu/delegate.py` |

### Priority 3 — Medium-term (test infrastructure)

| # | Action | Status |
|---|---|---|
| 11 | Add `pytest-cov` and establish a coverage baseline | **DONE** — added to `[dev]` deps, CI generates coverage report |
| 12 | Add `pytest-timeout` to prevent hanging tests | **DONE** — added to `[dev]` deps, `timeout = 300` in pyproject.toml |
| 13 | Add error handling tests (malformed models, USB disconnects) | **DONE** — 27 tests in `tests/test_error_handling.py` |
| 14 | Add Python 3.10 and 3.13 to CI matrix | **DONE** — matrix now `["3.9", "3.10", "3.11", "3.12", "3.13"]` |
| 15 | Extract shared tracker code (`_normalize_input`, `_resize_image`, etc.) to base class | **DONE** — shared methods in `_base.py`, ~170 lines removed from trackers |

### Priority 4 — Nice-to-have

| # | Action | Status |
|---|---|---|
| 16 | Replace `buf.find(buffer_data)` with flatbuffer-derived offsets in weight patching | **DONE** — `buffer_offsets` added to `TFLiteModelFull` |
| 17 | Add pip caching to CI workflow | **DONE** — `cache: 'pip'` in `setup-python@v5` |
| 18 | Consider lazy firmware download instead of shipping 36 MB binary in package | **DONE** — removed from `package-data`, auto-download already in transport.py |
| 19 | Standardize quantization into a shared utility function | **DONE** — `_quantize.py` with `quantize_uint8/int8/dequantize` |
| 20 | Add firmware upload error recovery / progress logging | |

---

*End of review.*
