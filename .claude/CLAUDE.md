# libredgetpu Project Context

## Project Goal
Use Google Coral Edge TPU accelerators for **custom computations beyond ML inference** (e.g., FFT, arbitrary linear transforms, signal processing). The Edge TPU is an int8 MAC engine — we want to exploit it as a general-purpose accelerator.

## Pre-Commit Checklist
**IMPORTANT**: After ANY code modification, always review and update documentation to reflect what changed:
- **`README.md`** — user-facing: install, API usage, examples, performance, constraints
- **`docs/ADVANCED_README.md`** — developer-facing: architecture, internals, blob format, protocol, gotchas, experiments
- **`docs/VISUAL_TESTS.md`** — visual proof tests guide with reference images (if tests changed)
- **`docs/ROBOTICS_STATUS.md`** — robotics roadmap status (if modules changed)

If you added/changed a module, API, template, test, or constraint, review and update ALL relevant docs above. Do not skip this step.

## Repository Structure

```
LibrEdgeTPU/
├── README.md                    # User-facing documentation
├── LICENSE                      # Apache-2.0
├── pyproject.toml               # Package config
├── setup.py                     # C extension build
├── requirements.txt             # Dependencies
├── .gitignore
├── .github/workflows/           # CI (GitHub Actions)
├── .claude/                     # Claude context
│   ├── CLAUDE.md                # This file
│   └── context/                 # Detailed context files
├── libredgetpu/                   # The installable Python package
│   ├── __init__.py              # exports SimpleInvoker, MatMulEngine, etc.
│   ├── _base.py, _constants.py  # shared base + constants
│   ├── _quantize.py             # shared quantization utilities
│   ├── simple_invoker.py        # user-facing inference API
│   ├── matmul_engine.py         # runtime weight-swapping matrix multiply
│   ├── looming_detector.py      # collision avoidance via edge density
│   ├── spot_tracker.py          # visual servoing via soft argmax
│   ├── pattern_tracker.py       # template matching via Conv2D correlation
│   ├── optical_flow_module.py   # global optical flow via Gabor features
│   ├── visual_compass.py       # yaw estimation wrapper around OpticalFlow
│   ├── reservoir.py             # echo state network via MatMulEngine
│   ├── embedding_similarity.py  # cosine similarity search via MatMulEngine
│   ├── template_gen.py          # dev-time Dense(N) template generation
│   ├── looming_gen.py           # dev-time looming template generation
│   ├── spot_tracker_gen.py      # dev-time spot tracker template generation
│   ├── pattern_tracker_gen.py   # dev-time pattern tracker template generation
│   ├── optical_flow_gen.py      # dev-time optical flow template generation
│   ├── tflite_parser.py         # manual TFLite flatbuffer reader
│   ├── tflite_builder.py        # TFLite FlatBuffer builder (no TF needed)
│   ├── delegate.py              # DarwiNN executable + DMA hint extraction
│   ├── driver.py                # hardware init + execution orchestration
│   ├── transport.py             # USB layer (firmware, registers, bulk transfer)
│   ├── registers.py             # hardware register constants
│   ├── _usb_accel.c             # optional C extension for direct libusb-1.0
│   ├── darwinn/                 # generated FlatBuffer bindings
│   ├── templates/               # pre-compiled Dense(N) Edge TPU models
│   ├── looming/                 # looming detection package + templates
│   ├── tracker/                 # spot tracker package + templates
│   ├── pattern/                 # pattern tracker package + templates
│   ├── optical_flow/            # optical flow package + templates
│   ├── postprocess/             # CPU post-processing (deeplabv3, posenet, multipose)
│   └── gui/                     # Flask web GUI with algorithm modes + CPU replica
├── examples/                    # Standalone robotics scripts (one per module)
├── tests/                       # Test suite
├── experiments/                 # Research/validation scripts (not shipped)
└── docs/                        # Developer documentation
    ├── ADVANCED_README.md
    ├── MATMUL_ENGINE_NOTES.md
    ├── HARDWARE_ANALYSIS.md
    └── ROBOTICS_STATUS.md
```

### libredgetpu/ — Pure-Python Edge TPU Inference (working, hardware-tested)
No libedgetpu, no tflite_runtime. 3 deps: pyusb, numpy, flatbuffers.

## Running Tests

**Test Suite:** 517 total tests (449 offline + 68 hardware) — all pass ✅

```bash
pip install -e ".[dev]"

# All offline tests (no hardware needed)
pytest tests/ -v --ignore=tests/test_hardware.py -k "not hardware"

# Individual test suites (offline)
pytest tests/test_transport_guards.py -v      # USB safety checks (5)
pytest tests/test_parse_models.py -v          # TFLite/DarwiNN parsing (12)
pytest tests/test_matmul_engine.py -v -k "not hardware"   # MatMulEngine (18)
pytest tests/test_looming.py -v -k "not hardware"         # LoomingDetector (12)
pytest tests/test_spot_tracker.py -v -k "not hardware"    # SpotTracker (38)
pytest tests/test_pattern_tracker.py -v -k "not hardware" # PatternTracker (26)
pytest tests/test_error_handling.py -v        # Error handling edge cases (27)
pytest tests/test_tflite_builder.py -v       # TFLite builder (121)
pytest tests/test_optical_flow.py -v -k "not hardware"  # OpticalFlow (54)
pytest tests/test_visual_compass.py -v                  # VisualCompass (25)
pytest tests/test_reservoir.py -v                       # ReservoirComputer (32)
pytest tests/test_embedding_similarity.py -v            # EmbeddingSimilarity (29)
pytest tests/test_cpu_replica.py -v                     # CPU replica pipeline (26)
pytest tests/test_ground_truth.py -v                    # Ground-truth validation (20)
pytest tests/test_visual_robotics.py -v --run-hardware  # Visual proof tests (6, requires hardware)

# Hardware tests (requires USB Edge TPU, pass --run-hardware)
pytest tests/ -v --run-hardware
```

## Open Questions
1. Decode `unk_3` sub-fields (which bits = stride vs. limit vs. mask) via systematic experimentation
2. Map `v_cmd` (20 values) to DMA channel/TTU register selectors
3. Find where requantization multiplier is embedded (possibly v_op_2=7 constant 0xfc056600)
4. Analyze matmul_256 and dense_1_8_mul programs to understand vector MAC pipeline

## Python Environment
- TensorFlow installed (`pip install tensorflow`) in current env
- `pyusb`, `numpy`, `flatbuffers` are the runtime dependencies

## Robotics Roadmap Status

**Implemented** (9 modules):
- SimpleInvoker — standard ML inference
- MatMulEngine — runtime weight-swapping y=Wx (0.28 ms)
- LoomingDetector — collision avoidance via edge density (1-1.5 ms)
- SpotTracker — visual servoing via soft argmax (~1 ms)
- PatternTracker — template matching via Conv2D correlation (~5 ms)
- OpticalFlow — global ego-motion via Gabor features + CPU correlation (~2 ms)
- VisualCompass — yaw estimation wrapper around OpticalFlow (~2 ms)
- ReservoirComputer — echo state network via MatMulEngine (~0.6 ms)
- EmbeddingSimilarity — cosine similarity search via MatMulEngine (~0.28 ms)

**Designed but not yet implemented** (see `docs/ROBOTICS_STATUS.md` for full specs):
- Learned Controllers (MLP policies at 1+ kHz)
- Batch MPC (neural dynamics + trajectory selection)

## Continuous Learning

When you encounter new insights, gotchas, debugging discoveries, or mistakes to avoid, **proactively record them** in the project auto-memory (`MEMORY.md`). You don't need to be asked. Examples:
- Subtle bug patterns found and fixed
- API quirks, undocumented behavior, Edge TPU hardware surprises
- Strategies that worked or failed for specific problems
- Performance findings, test flakiness, dependency issues
- When test counts change, update MEMORY.md, CLAUDE.md, README.md, and ADVANCED_README.md

Also update `docs/ROBOTICS_STATUS.md` when a roadmap item's status changes (implemented, discarded, design revised).

## Context Files (read on demand)
Detailed context is split into topic files under `.claude/context/`:
- **`hardware.md`** — ISA bit layout, scalar/branch ops, memory architecture, systolic array tiling, patents
- **`libredgetpu-internals.md`** — Weight swap protocol, param blob format, key learnings (execution protocol, DMA hints, USB gotchas, TFLite parsing)
- **`research.md`** — Optical flow analysis, image compression assessment, edgetpu_compiler binary analysis
- **`development-history.md`** — Full changelog (items 1-41) archived for reference
