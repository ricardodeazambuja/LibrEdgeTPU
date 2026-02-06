# Development History (Archived)
> Note: Early items reference the old multi-project repository (coral/, edgetpuxray/) which is no longer part of this repo.

## Early Exploration (items 1-11)
1. Full exploration of coral/ (FFT pipeline) and edgetpuxray/ (hardware analysis)
2. Decoded all 23 non-inception .coral programs
3. Systematic diffing of all program pairs (key finding: 1 instruction differs)
4. Generated TFLite models for mul2/div2/div4/add1/sub1/relu, extracted quantization params
5. Proved operations are encoded in weights + host-side dequantization, not instructions
6. Cross-referenced 4 Google patents to decode branch field, identify TTU in unk_3
7. All findings written to `edgetpu_hardware_analysis_findings.md`
8. Compiled Dense(NxN) models from 256 to 12288 — all compile and fully map to Edge TPU
9. Discovered two parameter modes: **cached** (<=8 MB, fast) vs **streamed** (>8 MB, USB-bottlenecked)
10. Verified compiled TFLite structure: `edgetpu-custom-op` = runs on Edge TPU, all other ops = CPU
11. Proved compiler creates at most ONE Edge TPU subgraph — unsupported op breaks the chain permanently

## libredgetpu Development (items 12-41)
12. Rewrote `init_hardware()` to follow libedgetpu's 8-phase sequence with proper bitfield manipulation and register polling
13. Added `poll_register()` to transport with timeout and bitmask support
14. Implemented smart parameter caching via `parameter_caching_token` tracking — skips param upload on repeated inferences
15. Discovered and implemented DMA hint-driven execution — parses `DmaHints` from DarwiNN executables to follow exact transfer order
16. Root cause of PoseNet/DeepLabV3 failures: split-input DMAs (overlapping byte ranges) and multi-chunk instruction sequences
17. Hardware-verified: PoseNet (17/17 keypoints, 17ms), DeepLabV3 (segmentation, 27ms), Dense 256-2048 (0.27ms cached), SSD MobileDet (12ms cached)
18. Added test suite (tests/model_zoo.py, test_parse_models.py, test_hardware.py) with auto-download from EdgeTPUModelZoo
19. Fixed EO-phase parameter bug — Inception V1 EXECUTION_ONLY has 393KB params that must be sent during inference
20. Fixed input DMA padding — split-input models need zero-padding past raw tensor (confirmed by libedgetpu `dma_info_extractor.cc`)
21. All 9 EdgeTPUModelZoo models pass hardware tests (classification, detection, segmentation, pose estimation)
22. Added `parse_full()` to tflite_parser — extracts all tensors, operators, buffers, opcodes from TFLite models (no TF dependency). Corrected Model.buffers field index: field 4 (not 3; field 3 = description string).
23. Implemented `postprocess/deeplabv3.py` — CPU post-processing (tile-resize, concat, two 1x1 convs, argmax) using weights extracted via `parse_full()`. Verified identical to TF-schema approach (0.0 max diff).
24. Implemented `postprocess/posenet_decoder.py` — full PersonLab algorithm ported from C++ (`posenet_decoder.cc`). FlexBuffer param parsing. Bilinear sampling, local maxima, mid+short offset following, graph backtracking, keypoint NMS, soft NMS.
25. Added `invoke_raw_outputs()` to SimpleInvoker — returns per-output-layer byte arrays for multi-output models.
26. Added validated hardware tests: `posenet_validated` (Grace Hopper >= 1 pose, >= 5 keypoints) and `deeplabv3_validated` (Grace Hopper -> person class >5% coverage).
27. Implemented `MatMulEngine` for runtime weight-swapping matrix multiply on Edge TPU. Pre-compiled Dense(N) templates (256, 512, 1024) with sidecar JSON metadata.
28. Implemented `template_gen.py` — dev-time Dense(N) template generator (TF + edgetpu_compiler). Saves compiled + uncompiled TFLite + JSON sidecar.
29. **Experiment 1 (recompilation consistency)**: Compiled Dense(256) with different weight distributions. Found PARAMETER_CACHING instructions are architecture-determined (always `f801dadfbd170b86` for Dense 256). EXECUTION_ONLY instructions encode requant multiplier — differ when weight_scale changes. Corrected DarwiNN exec type mapping (was wrong in initial analysis).
30. **Experiment 1c (fixed-scale patching)**: Patched int8 weights into uncompiled TFLite while preserving quant metadata. Confirmed: EO instructions, PC instructions, and output_scale are all IDENTICAL across 5 weight variants. Only the PC parameter blob changes. This validates the `set_weights()` algorithm: quantize with template scale -> patch -> recompile -> extract params only.
31. Added `TestRecompilationConsistency` (2 tests) to test suite: verifies EO instruction identity after weight patching, verifies PC param blob changes.
32. Fixed critical input quantization bug: template representative dataset used [0,255] giving input_scale=1.0 — float [-1,1] mapped to uint8 [0,1] (all zeros). Fixed by using uniform(-1,1) representative data. New input_scale~=0.00784, output_scale~=0.019 (260x finer).
33. Rewrote hardware tests with proper weight ranges and tightened tolerances. All 40 tests pass (10 parse + 19 matmul + 11 inference).
34. Code review fixes (7 issues): added missing `return result` in `apply_field_offsets()`, bounds checks in `relayout_output()`, `shutil.which("edgetpu_compiler")` checks in matmul_engine + template_gen, SHA256 firmware verification in transport, `_ensure_open()` guards on 5 USB methods, `max(topk, 1)` division guard in posenet_decoder. Added test_transport_guards.py (5 tests) and 2 field-offset tests. All 30 offline + 11 hardware tests pass.
35. USB transfer optimizations: header coalescing (send header+data in single write), pre-allocated read buffers, cached SimpleInvoker invariants, zero-copy DMA padding. MobileNet V1: 7.2ms->4.2ms with pyusb alone.
36. Optional C extension (`_usb_accel.c`): direct libusb-1.0 calls bypassing pyusb array.array/ctypes overhead. `setup.py` with `OptionalBuildExt` for graceful fallback. MobileNet V1: 4.0ms (matches libedgetpu). Discovered USB data boundary misalignment — `read_output()` must not cap at `max_size`. All 47 tests pass.
37. Code review (2 issues): added 4GB bounds check in `_usb_accel.c:send()` before uint32 cast, added thread-safety documentation (Edge TPU requires sequential command/response). Commit `7f4fc75`.
38. **LoomingDetector** (collision avoidance): Fixed-weight Sobel edge detection -> squared magnitude -> 3x3 zone pooling -> tau computation. Architecture: Conv2D(Sobel_X, /8) + Conv2D(Sobel_Y, /8) -> Mul(self) -> Add -> AvgPool(H/3xW/3) -> Reshape[9]. All ops fully mapped to Edge TPU. Tau = center/mean(periphery), TTC from tau history. 64x64: 1.0ms, 128x128: ~1.5ms. 12 offline + 5 hardware tests (all pass). Commit `0f0a5c0`.
39. **Critical finding: Edge TPU always outputs uint8** (Experiment 2). For int8 output models, raw bytes have sign bit flipped: `hw_byte = cpu_byte XOR 0x80`. Fixed `simple_invoker.py:invoke()` with dtype check + XOR. Fixed SpotTracker: removed wrong `negate_outputs` hack, replaced with correct uint8->int8 conversion. All 54 hardware tests pass (26 spot tracker + 28 existing).
40. **Experiment 3: DarwiNN param blob format empirically determined.** Phase A: values are XOR 0x80 (int8->uint8 sign flip). Phase B: overhead is per-channel float32 requant multipliers, weight-independent. Phase C: overhead = N*8 bytes across all sizes. Phase D: 64-row groups with 4-column tiling. 25/25 byte-perfect matches vs compiler (5 sizes x 5 weight patterns). Integrated `_generate_param_blob()` into `matmul_engine.py` as fast path — eliminates x86-only `edgetpu_compiler` dependency for `set_weights()`. Template sidecar JSON now includes `param_overhead` (base64). All 96 tests pass (41 offline + 55 hardware).
41. **Color tracking + runtime color swap + documentation overhaul.** Generated all 7 color SpotTracker templates at 64x64 and 128x128. Added `set_color([R,G,B])` for runtime color swapping — patches Conv2D filter weights (bytes [0,1,2] of DarwiNN param blob, same XOR 0x80 encoding). Conv2D color filter weights always at blob offset [0,1,2] (verified by diffing all 7 compiled color templates — only 3 bytes differ out of 132,416). Added `from_color_weights()` closest-match API, `find_closest_color()`, `--color-weights` CLI flag. Stored `color_weight_scale` in sidecar JSON. Rewrote `README.md` (user-facing). Created `ADVANCED_README.md` (dev-facing: architecture, blob format, protocol, gotchas). Added pre-commit doc checklist to both CLAUDE.md files. 24 new tests (18 offline + 7 hardware, 55 total spot tracker tests). All pass. Commit `ee4f393`.
