# Edge TPU Hardware Details

## 128-bit ISA (fully decoded bit layout)
```
[127:110] unk_3(18)      — TTU config (loop mask/stride/limit)
[109:105] vs_reg_w(5)    — Vector storage register write
[104:102] v_op_2(3)      — Float constant load to vector regs
[101:70]  imm_scalar(32) — Scalar immediate
[69:65]   s_y(5)         — Scalar source Y
[64:60]   s_x(5)         — Scalar dest/source X
[59:54]   s_op(6)        — Scalar opcode (0x00-0x0F reg, 0x20-0x2F imm)
[53:49]   vs_reg(5)      — Vector storage register
[48:44]   v_cmd(5)       — Vector/DMA subcommand
[43:36]   v_offset(8)    — Vector offset / stride selector
[35:31]   v_op(5)        — Primary vector op (0xa=load vreg, 0xc=USB output)
[30:19]   imm_size(12)   — Transfer size / program length
[18:14]   vs_reg_v1(5)   — Vector storage register V1
[13:12]   enable_vector(2) — Vector unit enable
[11]      enable_scalar(1) — Scalar unit enable
[10:6]    branch(5)      — Opcode/branch (maps to patent Table 300)
[5]       unk_0(1)       — Unknown (DMA direction?)
[4]       yes_pred(1)    — Predicate sense
[3:1]     pred_reg(3)    — Predicate register (0-7)
[0]       gate(1)        — Enable predication
```

## Scalar ops
NOP/ADD/SUB/AND/ORR/XOR/SHL/SHR/ASR/EQ/NEQ/GT/ULT/GEQ/GES/MOV (+ immediate variants with 0x20 bit)

## Branch field → patent opcodes
- 0x04=DMAOp.In, 0x05=DMAOp.Out, 0x06=DMAOp.W-N, 0x07=DMAOp.N-W
- 0x08=DMAOp.R-bus, 0x11=TileFenceOp, 0x1A=InFeed/TTU config
- 0x1E=BRANCH_START, 0x01=HALT, 0x1F=END_MARKER

## Critical Discovery: Operations encoded in weights + quantization, NOT instructions
- mul2, div2, div4 produce nearly/fully identical .coral programs
- Only 1 instruction differs (offset 0x6B0) across add1/mul2/div2/relu
- The actual math is determined by: weight_scale, output_zero_point (host-side TFLite metadata)
- Edge TPU is a pure int8 MAC engine; it doesn't know what operation it performs

## Weight encoding
- All simple ops: int8 weight = 127 (max), weight_zp = -128
- real_weight = 255 * weight_scale (mul2: scale=0.00784->2.0, div2: scale=0.00196->0.5)
- Offsets via output_zero_point (add1: zp=-128 adds 1.0, sub1: zp=+127 subtracts 1.0)

## Output pattern
v_op=0xa (load 4 vector regs) then v_op=0xc (trigger USB output)

## Hardware specs
- "Beagle" chip, estimated 64x64 MAC array at 480MHz = 4 TOPS
- USB endpoints: EP1(write), EP0x82(status), EP0x81(output)
- USB framing: [uint32 length][uint32 tag] + data. Tags: 0=Instructions, 1=InputAct, 2=Params, 3=OutputAct, 4-7=Interrupts

## Memory Architecture (Two Separate SRAM Systems)
- **Wide memory** (~8 MB): stores weights/parameters. Capacity registers: `tileMemoryCapacity` (0x44010)
- **Narrow memory** (per tile): stores activations (inputs, intermediates, outputs). Capacity: `scMemoryCapacity` (0x44008)
- DMA moves data between them: `DMAOp.W-N` (branch=0x06), `DMAOp.N-W` (branch=0x07)
- Compiler uses two executable modes:
  - **Cached** (weights ≤ ~8 MB): PARAMETER_CACHING + EXECUTION_ONLY executables (weights loaded once)
  - **Streamed** (weights > ~8 MB): STAND_ALONE executable (weights sent over USB every inference)
- Narrow memory per tile measured: 832 B (Dense 256), 9,216 B (Dense 8192), 171 KB (SSDLite MobileDet)
- Generated flatbuffer Python bindings via `flatc --python executable.fbs` → `platforms/darwinn/`

## unk_3 (18 bits)
TTU configuration (patent: loop mask + stride/limit values for tensor traversal unit)

## Compiled Model & Matrix Size Findings
- `edgetpu-custom-op` in compiled TFLite always runs on Edge TPU hardware
- Compiler can only create ONE contiguous Edge TPU subgraph per model
- **Cached mode** (weights ≤ ~8 MB): weights in on-chip SRAM, compute-bound, full 4 TOPS
- **Streamed mode** (weights > ~8 MB): weights sent over USB every inference, I/O-bound
- Practical cached limit: ~2,896x2,896 square matrix (8 MB weights)
- No compiler-side matrix size limit found (tested up to 12,288x12,288 = 144 MB)
- edgetpu_compiler version: 16.0.384591198

## Systolic Array Tiling (from qengineering.eu, cross-referenced with our findings)
- 64x64 MAC array has **fixed dimensions** — input vectors wider than 64 are split into 64-element chunks
- Each chunk is a separate tiling pass; partial sums accumulate in on-chip buffers; **weights reload each pass**
- Dense(256): 4 tiling passes. Large-kernel Conv2D (e.g., 64x64 kernel = 4096 weights/filter): ~64 passes
- Compiler appears to have a **tiling pass limit or memory budget** — SpotTracker's full-image Conv2D compiles at 16x16 (256 weights) but fails at >=64x64 (4096 weights)
- **Weight loading is a bottleneck**: each pass reloads weights → ops with many weights are bandwidth-limited, not compute-limited
- Post-MAC activation is **hardwired ROM** (likely ReLU) — compiler controls bypass, but cannot change the function
- EfficientNet design principle (avoid 1x1+3x3 splits, minimize weight reloads) directly reflects this hardware cost

## Key Patents
- US20190050717A1 — Tile architecture, MAC array, memory hierarchy
- US20180197068A1 — TTU, TensorOp/DMAOp, sync flags, opcodes
- GB2558980A — Opcode Table 300 (opcodes 0-12)
- US20210373895A1 — TTU counter/stride/limit, address generation
