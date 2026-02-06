# libredgetpu Internals

## Partial Model Loading & On-the-Fly Weight Updates
- Architecture separates instructions (tag=0), parameters (tag=2), and activations (tag=1) as independent USB data streams
- Cached models split into two DarwiNN executables: `PARAMETER_CACHING` (type=1, load weights once) + `EXECUTION_ONLY` (type=2, run inference). Type 0 = `STAND_ALONE` (streamed mode).
- `parameter_caching_token` (uint64) allows models to share cached weights
- `FieldOffset` mechanism patches base addresses (parameter, input, output, scratch) into instructions at load time
- **Weight swapping is viable and experimentally verified** (Experiment 1c):
  - Patch int8 weight values into uncompiled TFLite → recompile → extract new PC params
  - As long as quantization metadata (weight_scale, output_scale) is preserved in the TFLite, the compiler produces **identical instructions** — only the parameter blob changes
  - This means `set_weights()` only needs to swap the PC param blob; no instruction update needed
  - Verified across 5 different weight distributions (random, identity, near-zero) for Dense(256)
- **Caveat**: requantization multiplier = (input_scale * weight_scale / output_scale) is baked into EO instructions — changing quant metadata changes instructions
- **Caveat**: no partial weight update — entire parameter blob must be resent
- **Caveat**: weight dimensions must match compiled model exactly (only values can change)
- **Caveat (RESOLVED)**: ~~requires `edgetpu_compiler` at runtime~~ → Experiment 3 empirically determined the param blob format; `set_weights()` now generates blobs directly in NumPy
- Robotics applications: adaptive controllers, online learning, dynamic transforms, multi-task switching

## DarwiNN Parameter Blob Format (Experiment 3, SOLVED)
- **Compiler-free blob generation**: `edgetpu_compiler` NO LONGER needed for `set_weights()` at runtime
- Blob structure: **64-row groups**, each = `[overhead: 64*8 bytes][weights: 64*N bytes]`
- Value transform: `blob_byte = int8_weight XOR 0x80` (sign-bit flip, same as output convention)
- Weight layout within group: **4-column tiles** — `offset = col_block*(64*4) + local_row*4 + local_col`
- Full formula: `blob_off = (row//64)*(64*8+64*N) + 64*8 + (col//4)*(64*4) + (row%64)*4 + (col%4)`
- Overhead = per-channel float32 requant multipliers, **weight-independent** (copy from template)
- **25/25 byte-perfect matches** across 5 sizes (64-1024) x 5 weight patterns; all 96 tests pass (including hardware)
- Experiment scripts: `experiments/exp3_param_blob_format.py`, `experiments/exp3b_validate_blob_gen.py`

## MatMulEngine Summary
- Runtime weight-swapping matrix multiplication engine wrapping pre-compiled Dense(N) templates
- `MatMulEngine.from_template(256)` loads a pre-compiled template
- `set_weights(W)`: float32 → quantize → generate param blob (pure NumPy) → upload via USB
- Fast path: when `param_overhead` is in sidecar JSON, generates blobs directly (~microseconds, works on ARM)
- Fallback: when overhead unavailable, patches TFLite → recompiles with `edgetpu_compiler` (x86-only, ~50ms)
- `matmul(x)`: quantize input → send to EO executable → dequantize output → return float32
- Templates: `dense_{256,512,1024}_edgetpu.tflite` + `.json` sidecar + uncompiled `.tflite`
- Generate more: `python -m libredgetpu.template_gen --sizes 2048`
- Weight constraint: must fit `[-128 * weight_scale, 127 * weight_scale]` (clipped otherwise)

## Key libredgetpu Learnings
- **Execution protocol**: param_cache(instr+params→status) then execute(instr+input→output→status). For large outputs, EP 0x81 output data arrives in chunks BEFORE the EP 0x82 status packet.
- **Hardware init**: libedgetpu 8-phase sequence: Open (clear PHY bits), EnableReset (force sleep, poll, pulse gcbb_credit0), QuitReset (max clocks, poll running, poll scalarCoreRunControl=0, poll tileconfig0), EnableHardwareClockGate, InitializeChip, DoRunControl, Interrupts, Misc. Key register: `scu_ctrl_3` bits [23:22]=force_sleep, [9:8]=cur_pwr_state, [21:20]=clkdiv.
- **Parameter caching**: `parameter_caching_token` (uint64) in DarwiNN executable identifies cached weight set. Skip param upload when token matches. Speedup: 5x (Dense 256) to 50x (Dense 2048).
- **DMA hints**: DarwiNN `DmaHints` table specifies exact USB transfer order. Complex models need split-input DMAs (overlapping byte ranges) and multi-chunk instruction sequences. Without following hints, PoseNet/DeepLabV3 hang silently.
- **Split-input streaming**: PoseNet sends input as 2 overlapping DMAs (~490KB + ~471KB with ~36KB overlap). DeepLabV3 sends 2 input DMAs + 2 instruction chunks interleaved. The overlap ensures the pipeline doesn't stall at narrow memory boundaries.
- **Multi-output models** (e.g., SSD, PoseNet): outputs read per DMA hint step (not concatenated). Each output layer has its own read_output() call with the exact size from the hint.
- **TFLite flatbuffer gotchas**: opcode_index defaults to 0 when absent in vtable (don't skip operators with missing field). Vtable sharing between tables is common.
- **Models with CPU ops**: PoseNet has `PosenetDecoderOp` (CPU custom op) for keypoint decoding. DeepLabV3 has 8 CPU ops (RESIZE_BILINEAR, HARD_SWISH, CONCATENATION, 2xCONV_2D, 2xRESIZE_BILINEAR, ARG_MAX). The Edge TPU `edgetpu-custom-op` portion works independently — CPU post-processing reimplemented in `postprocess/` package.
- **DeepLabV3 CPU op graph**: Edge TPU outputs ASPP [33,33,256] + image_pooling [1,1,256] → RESIZE_BILINEAR (tile to 33x33) → HARD_SWISH (requantize to match ASPP scale) → CONCATENATION [33,33,512] → CONV_2D concat_projection [33,33,256] → CONV_2D logits [33,33,21] → RESIZE_BILINEAR x2 (upscale to 513x513) → ARG_MAX. The HARD_SWISH op is actually a requantization (scale 0.01242→0.03520, both zp=0). Weight extraction from TFLite buffers verified identical to TF-schema approach.
- **PoseNet CPU op graph**: Edge TPU outputs 3 tensors: heatmaps [31,41,17], short_offsets [31,41,34], mid_offsets [31,41,64] → PosenetDecoderOp (PersonLab algorithm). FlexBuffer custom_options contain: max_detections=20, score_threshold=0.2, stride=16, nms_radius=10.0. Offsets are dequantized then divided by stride to get block-space coordinates (matches C++ `DequantizeTensor` with `extra_scale=1.0/stride`).
- **TFLite Model field ordering**: version=0, operator_codes=1, subgraphs=2, description=3, buffers=4 (NOT field 3 for buffers — field 3 is a string description). OperatorCode: deprecated_builtin_code=0(int8), custom_code=1(old)/4(new), builtin_code=3(int32). Builtin opcode enum: CONCATENATION=2, CONV_2D=3, RESIZE_BILINEAR=23, ARG_MAX=56, QUANTIZE=80, HARD_SWISH=114.
- **EO-phase parameters**: Some cached models (e.g., Inception V1) have parameters in the EXECUTION_ONLY executable that must be sent during inference — not just during PARAMETER_CACHING. Always pass `eo.parameters` to `execute_dma_hints()`.
- **Input DMA padding**: Split-input models (PoseNet, DeepLabV3) have DMAs that read ~5 bytes past the raw tensor. The `execute_dma_hints()` method zero-pads input to cover `max(offset+size)` across all input DMA steps. Confirmed by libedgetpu `dma_info_extractor.cc`: "DMA may request a small amount of data past the end of the input buffer" using `allow_overflow=true`.
- **USB output boundaries**: `read_output(max_size)` can return MORE or FEWER bytes than `max_size`. USB data doesn't align with DMA hint output step sizes. Must NOT cap returned data. C extension allocates `max_size + 32768` headroom. Chunked reads must always request 32768 bytes (smaller triggers `LIBUSB_ERROR_OVERFLOW`).
- **Hardware-tested models (EdgeTPUModelZoo, 11/11 pass)**: MobileNet V1 (4.0ms), MobileNet V2 (4.3ms), Inception V1 (5.4ms), EfficientNet-S (8.6ms), SSD MobileDet (11.4ms), SSD MNv1 (10.1ms), DeepLabV3 (26.3ms), PoseNet (17.0ms). Also tested: Dense 256/1024/2048 (cached), Dense 12288 (streamed), Looming 64x64 (1.0ms), Looming 128x128 (~1.5ms).
- **Device recovery**: Failed USB operations can make device disappear from bus — requires physical replug to reset to bootloader mode (1a6e:089a).
- **Edge TPU always outputs uint8**: Hardware outputs bytes in uint8 format regardless of TFLite model's output type. For int8 output models (TFLite dtype=9), raw bytes have sign bit flipped vs CPU TFLite: `hw_byte = cpu_byte XOR 0x80`. Fix: `(uint8_bytes ^ 0x80).view(int8)` before dequantization. Most models (MobileNet, SSD, etc.) use uint8 output and are unaffected. Only int8 output models (e.g., SpotTracker) need the XOR correction.
