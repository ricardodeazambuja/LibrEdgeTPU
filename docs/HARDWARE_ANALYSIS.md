# Edge TPU Hardware Analysis & Custom Computation Findings

## Table of Contents
1. [Overview: Two Approaches to Custom Edge TPU Computation](#1-overview)
2. [coral/ — FFT via TensorFlow-to-TFLite Pipeline](#2-coral-fft)
3. [edgetpuxray/ — Direct Hardware Analysis](#3-edgetpuxray)
4. [Edge TPU Hardware Architecture (Beagle)](#4-hardware-architecture)
5. [128-bit Instruction Set Architecture](#5-isa)
6. [Decoded Instruction Fields & Semantics](#6-decoded-fields)
7. [Program Structure & Boilerplate](#7-program-structure)
8. [Vector Operation Catalog](#8-vector-ops)
9. [Weight & Quantization Encoding](#9-weight-encoding)
10. [Systematic Program Comparison Results](#10-program-comparison)
11. [v_op_2 Float Constant Loading](#11-vop2)
12. [Output Descriptor Pattern](#12-output-pattern)
13. [Data Flow Protocol](#13-data-flow)
14. [Patent-Derived Architecture Details](#14-patent-details)
15. [Refined Unknowns (Post-Patent Analysis)](#15-unknowns)
16. [Compiled Model Structure & Edge TPU / CPU Split](#16-compiled-model)
17. [Matrix Size Limits & Weight Caching vs. Streaming](#17-matrix-limits)
18. [Partial Model Loading & On-the-Fly Weight Updates](#18-partial-loading)
19. [libredgetpu: libedgetpu-Informed Driver & Parameter Caching](#19-libredgetpu-driver)
20. [Higher-Precision Arithmetic via Byte Decomposition](#20-byte-decomposition)
21. [Implications for Custom Computation](#21-implications)

---

## 1. Overview: Two Approaches to Custom Edge TPU Computation <a name="1-overview"></a>

The repository contains two complementary approaches to running custom (non-ML) computations on Google's Coral Edge TPU:

| Aspect | coral/ | edgetpuxray/ |
|--------|--------|--------------|
| **Level** | High-level TFLite API | Raw USB + binary instructions |
| **Approach** | Express algorithm as TF ops → quantize → compile | Analyze ISA, write/modify instructions directly |
| **Dependencies** | TensorFlow, libedgetpu, edgetpu_compiler | pyusb only (after compilation) |
| **Version coupling** | Must match TF/libedgetpu/compiler versions | None for execution (raw USB) |
| **Flexibility** | Limited to TFLite-compatible ops | Full hardware control |

---

## 2. coral/ — FFT via TensorFlow-to-TFLite Pipeline <a name="2-coral-fft"></a>

### How It Works
1. **`fft_model.py`** implements Cooley-Tukey FFT as TensorFlow operations:
   - Bit-reversal reordering of input indices
   - Twiddle factors (complex roots of unity) precomputed per stage
   - Butterfly operations via `tf.split`, `tf.concat`, `tf.reshape`, arithmetic
   - Log2(N) stages, input/output shape: `(batch_size, fft_size, 2)` where `2 = [real, imag]`

2. TF graph → **TFLite with INT8 quantization** (representative dataset of 500 random samples) → `edgetpu_compiler`

3. **C++ runtime** (`main.cpp`, `interpreter.cpp`) loads compiled model, reads IQ samples from RTL-SDR, quantizes to INT8, runs inference, dequantizes, writes complex64 results.

### Key Design Patterns
- Fixed FFT size and batch size per compiled model
- Complex numbers as 2-element tensors (last dimension)
- All math via standard TF ops (no custom ops needed)
- Twiddle factors are constants embedded in graph
- Producer-consumer threading with circular buffer (capacity 1000)

### Build System
- CMakeLists.txt: C++17, TensorFlow v2.16.1 fetched from GitHub
- Dependencies: TFLite, libedgetpu, librtlsdr, readerwriterqueue

### Constraints
- FFT size must be power of 2
- INT8 quantization introduces dB-level accuracy loss
- Each config (size + batch) needs separate compiled model
- USB bandwidth constraints (multiples of 512 bytes, ideally 16384)

---

## 3. edgetpuxray/ — Direct Hardware Analysis <a name="3-edgetpuxray"></a>

### File Inventory

```
edgetpuxray/
├── beagle_csr_offsets.h      # 1804 lines - Hardware CSR register map
├── executable.fbs            # FlatBuffers schema for DarwiNN executables
├── main.cc                   # C++ wrapper with USB interposition (sniffing tool)
├── build.sh                  # Build script for main.cc
├── connect.py                # Full inference via raw USB (reimplements libedgetpu)
├── simple.py                 # Register read/write + custom program execution
├── decompiler.py             # 128-bit instruction decoder/encoder/assembler
├── read.py                   # Image preprocessing (resize to 299x299)
├── compile/
│   ├── docker.sh             # Docker wrapper for edgetpu_compiler
│   ├── generate_model.py     # Creates quantized TFLite models
│   └── parse.py              # Extracts DarwiNN bitstreams from compiled TFLite
└── programs/                 # 25 pre-compiled .coral binary programs
    ├── add1.coral            # x + 1
    ├── sub1.coral            # x - 1
    ├── mul2.coral            # x * 2
    ├── div2.coral            # x / 2
    ├── div4.coral            # x / 4
    ├── mul2_add10.coral      # x * 2 + 10
    ├── relu.coral            # ReLU(x)
    ├── relu_2.coral          # ReLU variant
    ├── relu_4.coral          # ReLU variant
    ├── relu_sub_1.coral      # ReLU(x) - 1
    ├── sigmoid.coral         # sigmoid(x)
    ├── dense_1_8_mul.coral   # Dense layer 1x8
    ├── matmul_256.coral      # Matrix multiply 256x256
    ├── inception_0.coral     # Inception module (261KB)
    ├── div2_8.coral through div2_80.coral  # Size variations
    ├── div2_*_aarch64.coral  # ARM64 variants
    ├── weight_copy_in_0x80.coral / 0x100.coral  # Memory ops
    └── compare.py            # Side-by-side program diff utility
```

### How Programs Were Created
Two paths exist:
1. **Compiler path**: Keras model → quantize → `edgetpu_compiler` (Docker) → `parse.py` extracts bitstream
2. **Direct assembly**: `decompiler.py`'s `mins()` helper constructs raw 128-bit instructions

### How Programs Are Executed (simple.py)
```python
dev = open_device()                                    # USB connect
# Configure ~40 hardware registers (tile, DMA, interrupts, etc.)
write_register(dev, 'scalarCoreRunControl', ...)
llsend(dev, prog, 0)                                   # Send instructions (tag 0)
write_register(dev, 'scalarCoreRunControl', RUN)        # Start execution
llsend(dev, weight_data, 2)                             # Send parameters (tag 2)
llsend(dev, input_data, 1)                              # Send input (tag 1)
dat = dev.read(0x82, ...)                               # Read status
dat = dev.read(0x81, ...)                               # Read output tensor
# Dump scalar registers, predicate registers, PC for debugging
```

---

## 4. Edge TPU Hardware Architecture (Beagle) <a name="4-hardware-architecture"></a>

### Scalar Core
- 32 scalar registers (32-bit each), accessible at CSR `scalarRegisterFile` (0x44400)
- 8 predicate registers for conditional execution, at `predicateRegisterFile` (0x44500)
- Program counter at `currentPc`
- Run control at `scalarCoreRunControl` (0x44018): 0x01=run, 0x02=halt, 0x03=single-step
- Status at `scalarCoreRunStatus` (0x44258)
- Breakpoint support at `scalarCoreBreakPoint`

### Vector Processing
- Multi-tile distributed architecture
- Tile configuration via `tileconfig0` (0x48788), typical value 0x7f
- Vector operations execute alongside scalar operations (VLIW-style)

### Interconnect
- Mesh bus (4 channels): `meshBus0-3RunControl`
- Ring bus: `ringBusConsumer0-1RunControl`, `ringBusProducerRunControl`

### Data Paths
- `infeedRunControl` — input activation data path
- `outfeedRunControl` — output data path
- `parameterPopRunControl` — weight/parameter data path
- `avDataPopRunControl` — activation data path
- `opRunControl` — operation execution
- `narrowToWideRunControl` / `wideToNarrowRunControl` — data width conversion

### Memory Architecture (Two Separate SRAM Systems)

The Edge TPU has **two physically separate on-chip memory systems** serving different roles:

#### Wide Memory (Parameter SRAM) — ~8 MB total
- Stores **weights/parameters** (16-64 bits per unit)
- Shared across all tiles
- Fed by `parameterPopRunControl` data path from USB (tag=2 packets)
- Capacity reported by compiler as ~7.73-7.88 MiB available for caching
- Capacity register at `tileMemoryCapacity` (0x44010) — exact value requires hardware read
- DMA operations `DMAOp.W-N` (branch=0x06) and `DMAOp.N-W` (branch=0x07) move data between wide and narrow memory

#### Narrow Memory (Activation SRAM) — per tile, size varies by model
- Stores **activations**: input tensor, intermediate layer results, output tensor
- Each tile has its **own narrow memory** — not shared
- Capacity register at `scMemoryCapacity` (0x44008) — exact value requires hardware read
- Fed by `infeedRunControl` / `avDataPopRunControl` from USB (tag=1 packets)
- Results flow out via `outfeedRunControl` to USB (tag=3 packets)
- The DarwiNN executable records `used_narrow_memory_bytes_per_tile` for each compiled model

**Measured narrow memory usage per tile (from compiled DarwiNN executables):**

| Model | Narrow mem/tile | Notes |
|---|---|---|
| Dense 256×256 | 832 bytes | Minimal: 256 in + 256 out + overhead |
| Dense 4096×4096 | 4,864 bytes | Activation tiles for 4096-wide vectors |
| Dense 8192×8192 | 9,216 bytes | Activation tiles for 8192-wide vectors |
| SSDLite MobileDet (134 ops) | 174,812 bytes (~171 KB) | Peak activation across all layers |

The SSDLite model's 171 KB/tile reflects the peak intermediate feature map size across all 134 operations (the model processes 320×320×3 images through many convolutional layers).

#### How Parameters Are Managed: Cached vs. Streamed

The compiler uses **two different executable strategies** depending on whether weights fit in wide memory:

**Cached (weights ≤ ~8 MB):** Two executables in the DarwiNN package:
- `PARAMETER_CACHING` executable: loads weights into wide memory SRAM once (runs at model load time)
- `EXECUTION_ONLY` executable: runs inference using already-cached weights (no weight transfer per inference)
- Example: SSDLite MobileDet — 4.80 MB weights cached, 14,629-instruction execution program + 843-instruction caching program

**Streamed (weights > ~8 MB):** One executable:
- `STAND_ALONE` executable: weights bundled inline, sent over USB every inference call
- Example: Dense 8192×8192 — 64 MB weights streamed each inference, 723-instruction program

**Verified from DarwiNN MultiExecutable metadata:**

| Model | Executable Type | Parameters | Instructions | Narrow/tile |
|---|---|---|---|---|
| Dense 256×256 | PARAM_CACHING + EXEC_ONLY | 66 KB (cached) | 67 + 264 | 832 B |
| Dense 4096×4096 | STAND_ALONE | 16 MB (streamed) | 704 | 4,864 B |
| Dense 8192×8192 | STAND_ALONE | 64 MB (streamed) | 723 | 9,216 B |
| SSDLite MobileDet | PARAM_CACHING + EXEC_ONLY | 4.8 MB (cached) | 843 + 14,629 | 171 KB |

#### Memory-Related Registers
- `scMemoryCapacity` (0x44008) — scalar core / narrow memory capacity
- `tileMemoryCapacity` (0x44010) — tile / wide memory capacity
- `scMemoryAccess` (0x44040) / `scMemoryData` (0x44048) — scalar memory read/write port
- `memoryAccess` (0x42010) / `memoryData` (0x42018) — tile memory read/write port
- `outfeed_chunk_length` (0x4c058) — output chunk size configuration
- `descr_ep` (0x4c148) — descriptor endpoint
- `omc0_d4`, `omc0_d8` — output memory controller registers
- `narrowMemoryContext_0..3` — narrow memory bank context selectors (UNUSED on Beagle)
- `narrowMemoryIsolation`, `narrowMemoryRetention` — power management (UNUSED on Beagle)

### USB Endpoints
- **EP1** (write): Instructions, weights, input activations
- **EP 0x82** (read): Status/interrupt packets (8 bytes typical)
- **EP 0x81** (read): Output tensor data

### USB Framing
```
Header: [uint32 data_length][uint32 descriptor_tag]
Tags:   0=kInstructions, 1=kInputActivations, 2=kParameters,
        3=kOutputActivations, 4-7=kInterrupt0-3
Data:   Raw bytes, sent in 1MB chunks if large
```

---

## 5. 128-bit Instruction Set Architecture <a name="5-isa"></a>

Each instruction is 128 bits (16 bytes), little-endian, parsed in reverse byte order.

### Bit Layout (from MSB to LSB after reversal)

| Bits | Field | Size | Description |
|------|-------|------|-------------|
| 127-110 | `unk_3` | 18 | **Unknown** — possibly TTU/DMA config |
| 109-105 | `vs_reg_w` | 5 | Vector storage register write target |
| 104-102 | `v_op_2` | 3 | Secondary vector operation (float constant load) |
| 101-70 | `imm_scalar` | 32 | Scalar immediate value |
| 69-65 | `s_y` | 5 | Scalar source register Y |
| 64-60 | `s_x` | 5 | Scalar destination/source register X |
| 59-54 | `s_op` | 6 | Scalar operation code |
| 53-49 | `vs_reg` | 5 | Vector storage register |
| 48-44 | `v_cmd` | 5 | Vector command |
| 43-36 | `v_offset` | 8 | Vector offset |
| 35-31 | `v_op` | 5 | Primary vector operation |
| 30-19 | `imm_size` | 12 | Immediate size field |
| 18-14 | `vs_reg_v1` | 5 | Vector storage register V1 |
| 13-12 | `enable_vector` | 2 | Vector unit enable (0=off, 1-3=on with mode) |
| 11 | `enable_scalar` | 1 | Scalar unit enable |
| 10-6 | `branch` | 5 | Branch/control operation |
| 5 | `unk_0` | 1 | **Unknown** |
| 4 | `yes_pred` | 1 | Predicate sense (1=execute if true) |
| 3-1 | `pred_reg` | 3 | Predicate register index (0-7) |
| 0 | `gate` | 1 | Enable predication (1=conditional on pred_reg) |

---

## 6. Decoded Instruction Fields & Semantics <a name="6-decoded-fields"></a>

### Scalar Operations (s_op, 6 bits)

When `enable_scalar=1` and `branch=0`:

**Register-register operations (bit 5 = 0):**

| s_op | Name | Operation |
|------|------|-----------|
| 0x00 | NOP | No operation |
| 0x01 | ADD | s_x ← s_y + s[imm_scalar & 0x1F] |
| 0x02 | SUB | s_x ← s_y - s[imm_scalar & 0x1F] |
| 0x03 | AND | s_x ← s_y & s[imm_scalar & 0x1F] |
| 0x04 | ORR | s_x ← s_y \| s[imm_scalar & 0x1F] |
| 0x05 | XOR | s_x ← s_y ^ s[imm_scalar & 0x1F] |
| 0x06 | SHL | s_x ← s_y << s[imm_scalar & 0x1F] |
| 0x07 | SHR | s_x ← s_y >> s[imm_scalar & 0x1F] (logical) |
| 0x08 | ASR | s_x ← s_y >> s[imm_scalar & 0x1F] (arithmetic) |
| 0x09 | EQ | pred(s_x) ← (s_y == s[imm_scalar & 0x1F]) |
| 0x0A | NEQ | pred(s_x) ← (s_y != s[imm_scalar & 0x1F]) |
| 0x0B | GT | pred(s_x) ← (s_y > s[imm_scalar & 0x1F]) (signed) |
| 0x0C | ULT | pred(s_x) ← (s_y < s[imm_scalar & 0x1F]) (unsigned) |
| 0x0D | GEQ | pred(s_x) ← (s_y >= s[imm_scalar & 0x1F]) |
| 0x0E | GES | pred(s_x) ← (s_y >= s[imm_scalar & 0x1F]) (unsigned) |
| 0x0F | MOV | s_x ← s[imm_scalar & 0x1F] |

**Immediate operations (bit 5 = 1, add 0x20):**
Same as above but use the full 32-bit `imm_scalar` as the operand instead of register indirect. E.g., `ADDI (0x21)`, `MOVI (0x2F)`.

### Branch Operations (branch field, 5 bits)

| branch | Name | Description |
|--------|------|-------------|
| 0x00 | (none) | Normal instruction execution |
| 0x01 | (halt variant) | Halt with vsv1 set |
| 0x1E (0x3c>>1) | BRANCH_START | Program entry point, `imm_size` = program length in bits |
| 0x01 (2>>1) | HALT | Stop scalar core execution |
| 0x1F (0x3e>>1) | END_MARKER | End of program marker, `imm_size` = 0x80 |
| 0x04 | (common) | 148 occurrences — likely "sync" or "barrier" |
| 0x1A | (common) | 165 occurrences — likely DMA/data-path configuration |
| 0x05 | (common) | 21 occurrences — often with `enable_vector=1` |
| 0x06 | (common) | 16 occurrences — often with `v_op=0x8` |
| 0x07 | (common) | 19 occurrences — often predicated on p6 or p7 |
| 0x11 | (common) | 17 occurrences — often `v_cmd=0x4` or `v_cmd=0x10` |
| 0x13 | (common) | 28 occurrences — DMA transfer related |
| 0x17 | (common) | 44 occurrences — appears at program section boundaries |

### Predication
- `gate=1`: Enable conditional execution
- `pred_reg`: Which of the 8 predicate registers to test
- `yes_pred=1`: Execute if predicate is TRUE
- `yes_pred=0`: Execute if predicate is FALSE (negated)

---

## 7. Program Structure & Boilerplate <a name="7-program-structure"></a>

### Minimal Valid Program Structure (from decompiler.py)

```python
# 1. Start instruction: branch to beginning + program length
prog = mins(branch=0x3c>>1, enable_scalar=1, imm_size=(num_instr+5)*0x10*8)

# 2. Your instructions here...
prog += mins(enable_scalar=0x1, s_op=MOVI, s_x=0, imm_scalar=0xabab)

# 3. Halt
prog += mins(branch=2>>1, enable_scalar=1)

# 4. Four NOP padding (required)
prog += mins(enable_scalar=1)
prog += mins(enable_scalar=1)
prog += mins(enable_scalar=1)
prog += mins(enable_scalar=1)

# 5. End marker
prog += mins(branch=0x3e>>1, enable_scalar=1, imm_size=0x10*8)
```

### Compiler-Generated Program Sections (for simple element-wise ops)

All simple ops (add1, mul2, div2, div4, relu, relu_2, relu_4) generate **179 instructions** with this structure:

```
Section 1 (0x000-0x030): BRANCH_START + DMA/vector init headers
Section 2 (0x040-0x070): Scalar register initialization (MOVI s0-s3 = 0)
Section 3 (0x080-0x0C0): First data path setup (v_op=0xa, branch=0x4, etc.)
Section 4 (0x0D0-0x150): Second data path + sync barriers
Section 5 (0x160-0x2A0): Parameter descriptor (MOVI s11,s12 + address computation + v_op=0xa output setup + v_op=0xc trigger)
Section 6 (0x2B0-0x340): Vector pipeline configuration (v_op_2 constants)
Section 7 (0x350-0x3C0): DMA/sync section (branch=0x1a, branch=0x17)
Section 8 (0x3D0-0x6B0): Core computation section — **this is where the operation differs**
Section 9 (0x6C0-0x8E0): Output path (address computation + v_op=0xa + v_op=0xc)
Section 10 (0x8F0-0xAC0): Final output section (repeat output pattern)
Section 11 (0xAD0-0xB30): HALT + NOP padding + END_MARKER
```

### Key Finding: Only 1 Instruction Differs

For the simple operations, **only instruction at offset 0x6B0** changes between programs. Everything else is identical boilerplate for data movement, synchronization, and output.

---

## 8. Vector Operation Catalog <a name="8-vector-ops"></a>

### v_op (Primary Vector Operation, 5 bits)

| v_op | Count | Confirmed Meaning | Notes |
|------|-------|-------------------|-------|
| 0x01 | 43 | Init/config | Often in BRANCH_START or branch=0x1a sections |
| 0x02 | 76 | Data path control | Often with v_cmd, appears in sync sections |
| 0x03 | 39 | Program init | At offset 0x0000 for sub1, sigmoid |
| 0x04 | 32 | Barrier/sync | Appears after computation sections |
| 0x05 | 13 | Program init | At offset 0x0000 for add1, mul2, div2, relu |
| 0x06 | 1 | Program init | At offset 0x0000 for mul2_add10 only |
| 0x08 | 31 | DMA trigger | Often with branch=0x6, s_op=0x3e |
| 0x09 | 1 | Program init | relu_sub_1 only |
| **0x0a** | **195** | **Load vector register** | **Output descriptor loading (confirmed)** |
| 0x0b | 1 | Sigmoid-specific | Complex vector op |
| **0x0c** | **45** | **USB output trigger** | **"Send output tensor via USB" (confirmed)** |
| 0x0f | 3 | matmul-specific | |
| 0x10 | 6 | Vector compute | relu_sub_1, matmul |
| 0x11 | 9 | Vector compute | sigmoid, relu_sub_1 |
| 0x12 | 3 | Vector compute | sigmoid, dense |
| 0x13 | 11 | Requantize? | Common in simple ops at 0x590, v_offset=0x92 |
| 0x15 | 1 | Sigmoid-specific | |
| 0x1a | 1 | Sigmoid-specific | |
| 0x1c | 2 | Weight copy | weight_copy_in_0x100, sigmoid |
| 0x1d | 1 | Sigmoid-specific | |
| 0x1e | 16 | Pipeline config | In BRANCH headers, weight_copy |
| 0x1f | 26 | Config/compute | In computation sections, variable uses |

### v_op_2 (Secondary Vector Operation, 3 bits)

| v_op_2 | Count | Meaning |
|--------|-------|---------|
| 0x1 | 41 | Reset/zero load (imm often 0x0) |
| 0x2 | 21 | Constant load (imm=0x8000000 common) |
| 0x3 | 3 | Float constant (sigmoid-specific, e.g., 192.0) |
| 0x4 | 36 | Constant load (imm=0x20000000 common) |
| 0x5 | 4 | Float constant (relu_sub_1 specific) |
| 0x6 | 4 | Float constant (sigmoid, mul2_add10) |
| 0x7 | 46 | Config constant (imm=0xfc056600 universal across simple ops) |

---

## 9. Weight & Quantization Encoding <a name="9-weight-encoding"></a>

### How Operations Are Actually Encoded

The operation type is determined by a **three-layer encoding**:

#### Layer 1: Instruction Bitstream (.coral file)
- Generic "apply quantized affine transform" template
- Nearly identical across add1, mul2, div2, div4, relu
- Selects operation *class* (e.g., with/without activation, bias)

#### Layer 2: Parameter Data (sent via USB tag 2)
- Raw int8 weight values, no metadata header
- For all simple scaling ops: **weight = 127** (max int8 value)
- Same value for mul2, div2, div4

#### Layer 3: Quantization Metadata (host-side TFLite flatbuffer)
- **This is where the actual math is determined**

### Quantization Parameters by Operation

| Operation | weight_val | weight_scale | real_weight | input_scale | output_scale | input_zp | output_zp | out/in ratio |
|-----------|-----------|--------------|-------------|-------------|--------------|----------|-----------|-------------|
| mul2 | 127 | 0.00784314 | **2.0** | 0.00783911 | 0.01567822 | 0 | 0 | 2.000 |
| div2 | 127 | 0.00196078 | **0.5** | 0.00784227 | 0.00392114 | -1 | -1 | 0.500 |
| div4 | 127 | 0.00098039 | **0.25** | 0.00784118 | 0.00196029 | 0 | 0 | 0.250 |
| add1 | 127 | 0.00392157 | **1.0** | 0.00784017 | 0.00784160 | 0 | **-128** | 1.000 |
| sub1 | 127 | 0.00392157 | **1.0** | 0.00782977 | 0.00783181 | -1 | **+127** | 1.000 |
| relu | N/A | N/A | N/A | 0.00783250 | 0.00392039 | -1 | **-128** | 0.500 |

### Key Formulas

**Weight dequantization:**
```
real_weight = (weight_int8 - weight_zero_point) * weight_scale
            = (127 - (-128)) * weight_scale
            = 255 * weight_scale
```

**Scaling operations (mul2, div2, div4):**
- `weight_scale` encodes the multiplier
- `output_scale / input_scale` = real multiplier
- The int8 output is the same for all — only host-side dequantization differs

**Offset operations (add1, sub1):**
- `real_weight = 1.0` (identity multiply)
- Offset encoded in `output_zero_point`:
  - add1: `output_zp = -128` → `real_out = (int8_out + 128) * scale ≈ int8_out * scale + 1.0`
  - sub1: `output_zp = +127` → `real_out = (int8_out - 127) * scale ≈ int8_out * scale - 1.0`

**Activation operations (relu):**
- No weight tensor
- `output_zp = -128` clamps minimum at zero
- `output_scale ≈ input_scale / 2` (range halved)

### Critical Implication

**div2.coral and div4.coral are byte-identical programs** with identical weights. The difference exists only in the TFLite metadata that tells the host how to interpret the int8 output. The Edge TPU hardware does not know what operation it's performing — it just runs int8 multiply-accumulate.

---

## 10. Systematic Program Comparison Results <a name="10-program-comparison"></a>

### Same-Size Programs (179 instructions each, 2864 bytes)

| Comparison | Differing Instructions | What Differs |
|------------|----------------------|--------------|
| mul2 vs div2 | **1** | `imm_size` at 0x6B0: 0x406→0x6 |
| mul2 vs div4 | **1** | `imm_size` at 0x6B0: 0x406→0x6 |
| div2 vs div4 | **0** | **Byte-identical** |
| relu vs relu_4 | **0** | **Byte-identical** |
| relu vs relu_2 | **1** | `s_x`, `s_y`, `imm_scalar` at 0x6B0 |
| add1 vs mul2 | **1** | Multiple fields at 0x6B0 |
| add1 vs relu | **1** | Multiple fields at 0x6B0 |

### The Critical Instruction at Offset 0x6B0

| Field | add1 | mul2 | div2/div4 | relu | relu_2 |
|-------|------|------|-----------|------|--------|
| pred_reg | 4 | 4 | 4 | 4 | 4 |
| vs_reg_v1 | 3 | 3 | 3 | 3 | 3 |
| **imm_size** | **0xc06** | **0x406** | **0x6** | **0x6** | **0x6** |
| **v_op** | **0x1f** | **0** | **0** | **0** | **0** |
| **v_offset** | **0xff** | **0xfe** | **0xfe** | **0** | **0** |
| **v_cmd** | **0x1d** | **0x1d** | **0x1d** | **0** | **0** |
| **vs_reg** | **0x17** | **0xf** | **0xf** | **0x18** | **0x18** |
| s_op | 0x1f | 0x1f | 0x1f | 0x1f | 0x1f |
| s_x | 0x1e | 0 | 0 | 0 | 0x1e |
| s_y | 0x1f | 0 | 0 | 0 | 0x1f |
| imm_scalar | 0x21bf7f | 0x21bf80 | 0x21bf80 | 0x21bf80 | 0x21bf7f |

### Size Variation Programs (div2_8 through div2_80)

As input tensor size increases (8→10→20→40→80 elements), changes occur at:
- Offset 0x0270: `imm_scalar` scales with tensor size (0x8, 0x10, 0x20, 0x40, 0x80)
- DMA configuration fields change (v_cmd, imm_scalar at various offsets)
- Branch targets shift to accommodate larger data transfers
- Number of total instructions stays similar until div2_80 (216 vs 179)

### Architecture Differences (x86_64 vs aarch64)

div2_8 vs div2_8_aarch64: **18 differing instructions**, but the differences are in instruction ordering and DMA configuration, not in the core computation. The scalar operations are equivalent — the compiler optimizes instruction scheduling differently per architecture.

---

## 11. v_op_2 Float Constant Loading <a name="11-vop2"></a>

The `v_op_2` field loads 32-bit values into vector storage registers via `vs_reg_w`.

### Universal Constants (identical across ALL simple ops)

| Offset | v_op_2 | vs_reg_w | imm_scalar | Notes |
|--------|--------|----------|------------|-------|
| 0x02E0 | 2 | 0x0 | 0x08000000 | Pipeline config |
| 0x02F0 | 4 | 0x13 | 0x20000000 | Pipeline config |
| 0x0310 | 1 | 0x0 | 0x00000000 | Zero/reset |
| 0x0340 | 7 | 0x1 | 0xfc056600 | **Key config constant** |

### Sigmoid-Specific Constants (additional)

Sigmoid uses 12 additional v_op_2 instructions with varied float constants. These likely encode the piecewise linear approximation of the sigmoid function as lookup table values or polynomial coefficients.

### mul2_add10-Specific Constants

mul2_add10 has extra v_op_2 instructions related to its two-operation pipeline (multiply then add), including NaN-valued constants that may serve as sentinel values.

---

## 12. Output Descriptor Pattern <a name="12-output-pattern"></a>

Every program uses this exact sequence to trigger output via USB:

```
v_op=0xa  vs=6  v_off=0x0  (NOP scalar)     # Base register setup
v_op=0xa  vs=7  v_off=0x1  MOVI s8=OFFSET   # Output byte offset
v_op=0xa  vs=8  v_off=0x2  MOVI s9=TAG      # Descriptor tag
v_op=0xa  vs=9  v_off=0x3  (NOP scalar)     # End of descriptor
v_op=0xc  vs=0  v_off=0x0  (NOP scalar)     # TRIGGER: send output via USB
```

### Descriptor Tags (s9 values)
- `s9 = 1`: kInputActivations
- `s9 = 3`: kOutputActivations
- `s9 = 4`: kInterrupt0 (status/completion signal)

### Output Offsets (s8 values)
- `s8 = 0x8`: First output chunk (commonly 8 bytes)
- `s8 = 0x0`: Final output with interrupt

Programs typically have 3 output sections:
1. First: tag=1 (input ack), offset=8
2. Second: tag=3 (output data), offset=8
3. Final: tag=4 (interrupt/done), offset=0

---

## 13. Data Flow Protocol <a name="13-data-flow"></a>

### Complete Execution Sequence (from simple.py and connect.py)

```
1. USB Device Setup
   ├── Find device (VID=0x18d1, PID=0x9302)
   ├── Download firmware if needed (VID=0x1a6e → apex_latest_single_ep.bin)
   └── Claim interface, set configuration

2. Hardware Register Configuration (~40 registers)
   ├── System control: scu_ctrl_0, scu_ctrl_3
   ├── Tile config: tileconfig0 = 0x7f
   ├── Power: idleRegister, deepSleep
   ├── Data paths: avDataPop, parameterPop, infeed, outfeed, op RunControl
   ├── Width converters: narrowToWide, wideToNarrow RunControl
   ├── Interconnect: meshBus0-3, ringBusConsumer0-1, ringBusProducer RunControl
   ├── Interrupts: fatal_err, top_level_int_0-3 control
   └── Output: descr_ep, outfeed_chunk_length, omc0_d4, omc0_d8

3. Program Download
   └── llsend(dev, program_bytes, tag=0)  # kInstructions

4. Start Execution
   └── write_register('scalarCoreRunControl', 0x01)

5. Data Transfer (program waits at specific PCs for data)
   ├── llsend(dev, weight_bytes, tag=2)      # kParameters
   ├── llsend(dev, input_bytes, tag=1)       # kInputActivations
   └── (program processes data using scalar+vector units)

6. Output Retrieval
   ├── dev.read(0x82, ...) → status packet (8 bytes)
   └── dev.read(0x81, ...) → output tensor data

7. Post-Execution Debugging (optional)
   ├── write_register('scalarCoreRunControl', 0x02)  # halt
   ├── Read all 32 scalar registers
   ├── Read all 8 predicate registers
   └── Read program counter
```

### FieldOffsets: Runtime Patching

The DarwiNN executable contains `FieldOffset` entries that tell the runtime where to patch the instruction bitstream with actual memory addresses at load time. From `compare.py`'s `special_bits`:

| Byte Offset | Instruction | Desc | Name | Purpose |
|-------------|------------|------|------|---------|
| 0x048 | @0x40 | 2 | (param) | Parameter DMA address, slot 1 |
| 0x058 | @0x50 | 2 | (param) | Parameter DMA address, slot 2 |
| 0x068 | @0x60 | 3 | (param) | Parameter DMA config, slot 1 |
| 0x078 | @0x70 | 3 | (param) | Parameter DMA config, slot 2 |
| 0x0E8 | @0xE0 | 1 | "x" | Input activation address, slot 1 |
| 0x0F8 | @0xF0 | 1 | "x" | Input activation address, slot 2 |
| 0x718 | @0x710 | 0 | "Identity" | Output activation address, slot 1 |
| 0x728 | @0x720 | 0 | "Identity" | Output activation address, slot 2 |

All patch positions are at bit offset 70 within their respective 128-bit instructions, which corresponds to the `imm_scalar` field. The values are zero in the .coral files and patched with actual DMA addresses at runtime.

---

## 14. Patent-Derived Architecture Details <a name="14-patent-details"></a>

Multiple Google patents describe the Edge TPU architecture. Cross-referencing these with the empirically determined instruction fields significantly clarifies the unknowns.

### Key Patents

| Patent | Title | Key Content |
|--------|-------|-------------|
| US20190050717A1 | Methods and Systems for Neural Network Accelerator | Tile architecture, MAC array, memory hierarchy, instruction distribution |
| US20180197068A1 / US9836691 | Neural Network Instruction Set Architecture | TTU details, TensorOp/DMAOp encoding, sync flags, opcode table |
| GB2558980A | Neural Network Instruction Set Architecture | Opcode table (Table 300), TTU loop nest, layer type encoding |
| US20210373895A1 | Tensor Traversal Engine for Strided Memory Access | TTU counter/stride/limit programming, address generation algorithm |

### High-Level Opcode Table (from GB2558980A Table 300)

| Opcode | Operation | Category |
|--------|-----------|----------|
| 0 | Convolution / Fully Connected layers | TensorOp |
| 1 | Max pooling | TensorOp |
| 2 | Average pooling | TensorOp |
| 3 | Depth-wise conv / element-wise multiply | TensorOp |
| 4 | DMAOp.In — Input activation reception | DMAOp |
| 5 | DMAOp.Out — Output data writing | DMAOp |
| 6 | DMAOp.W-N — Wide→Narrow memory transfer | DMAOp |
| 7 | DMAOp.N-W — Narrow→Wide memory transfer | DMAOp |
| 8 | DMAOp.R-bus — Ring bus data operations | DMAOp |
| 9 | DMAOp.InFeed — External activation distribution | DMAOp |
| 10 | DMAOp.OutFeed — Result movement to I/O | DMAOp |
| 11 | TileFenceOp — Tile-level synchronization | Sync |
| 12 | ScalarFenceOp — Scalar-level synchronization | Sync |

### Mapping Patent Opcodes to Empirically Determined Fields

The `branch` field in the 128-bit instruction likely encodes these opcodes:

| branch value | Occurrences | Probable Patent Opcode | Evidence |
|-------------|-------------|----------------------|----------|
| 0x00 | (most) | Scalar/Vector compute | Normal ALU execution |
| 0x01 (HALT) | 22 | Control flow | Program termination |
| 0x04 | 148 | **DMAOp.In (opcode 4)** | Most common DMA op; aligns with input data movement |
| 0x05 | 21 | **DMAOp.Out (opcode 5)** | Output data path; often with `enable_vector=1` |
| 0x06 | 16 | **DMAOp.W-N (opcode 6)** | Wide→Narrow transfer; often with `v_op=0x8` (DMA trigger) |
| 0x07 | 19 | **DMAOp.N-W (opcode 7)** | Narrow→Wide transfer; often predicated on p6/p7 |
| 0x08 | 6 | **DMAOp.R-bus (opcode 8)** | Ring bus operations |
| 0x10 | 17 | **DMAOp.OutFeed (opcode 10)** | Result movement; `v_cmd` varies |
| 0x11 | 17 | **TileFenceOp (opcode 11)** | Tile sync; `v_cmd=0x4` or `v_cmd=0x10` |
| 0x13 | 28 | TensorOp-related | Possibly scatter/gather with TTU |
| 0x14 | 20 | TensorOp variant | |
| 0x17 | 44 | Section boundary/barrier | Program structure marker |
| 0x1A | 165 | **DMAOp.InFeed (opcode 9)** or TTU config | Dominates data-path setup sections |
| 0x1E (START) | per program | Control flow | Program entry point |
| 0x1F (END) | per program | Control flow | Program end marker |

### Tensor Traversal Unit (TTU) Architecture

From patents US20180197068A1 and US20210373895A1:

**TTU Register Arrays (4 tensor registers per TTU):**
```
Counters tensor [M×N]: Current position in each dimension of each tensor
Stride tensor   [M×N]: Address increment per dimension step
Limit tensor    [M×N]: Boundary condition (loop bound) per dimension
Init tensor     [M×N]: Initial counter values (reset targets)
```
Where M = number of tensors tracked simultaneously, N = max nesting depth.

**Address Generation:**
```
address = sum(counters[selected_by_TTU_loop_mask])
```
The TTU loop mask (part of TensorOp encoding) specifies which counters are summed. On each cycle, the innermost dimension counter increments by its stride. When a counter reaches its limit, it resets to its init value and the next outer dimension increments (rollover propagation).

**Mapping to Empirically Determined Fields:**
- **`unk_3` (18 bits)**: Very likely **TTU configuration** — encodes stride/limit/init values or TTU loop mask bits. Values like 0x3ffa0, 0x3ffe0 (with many high bits set) resemble TTU loop masks. Values like 0xc00, 0x1f00 resemble packed stride/limit pairs.
- **`imm_size` (12 bits)** in DMA contexts: Likely the **transfer size** or **block count** for DMA operations (the TTE patent describes "source block count" as a key register).
- **`v_offset` (8 bits)** in DMA contexts: Likely a **stride dimension selector** or **memory bank offset**.
- **`v_cmd` (5 bits)**: Likely the **DMA subcommand** or **TTU control signal** selecting which TTU registers to program or which DMA channel to activate.

### MAC Array Architecture

From US20190050717A1:

- **Estimated 64×64 systolic array** at ~480 MHz (Q-Engineering estimate for Edge TPU's 4 TOPS)
- Each MAC cell: receives one activation from **narrow memory** via input activation bus, one parameter from **wide memory**
- Partial sums accumulate in **sum registers** within each cell
- Results flow to **output activation pipeline** (shift register) then to **NLU**

**Memory widths (from patent):**
- **Narrow memory**: < 16 bits per unit (stores activations) — ~8 MB total (see Section 4 for measured per-tile usage)
- **Wide memory**: 16-64 bits per unit (stores weights/parameters) — ~8 MB total (confirmed by compiler: 7.73-7.88 MiB available)
- DMA engines move data between them: `narrowToWideRunControl`, `wideToNarrowRunControl`

### Non-Linear Unit (NLU)

The NLU applies activation functions after MAC computation:
- Configured via TensorOp instruction fields
- Supports at minimum: ReLU, sigmoid (via piecewise linear approximation), softmax (on classifier tile)
- Output written back to narrow memory

This explains why **sigmoid.coral** has many extra `v_op_2` instructions with float constants — these are the **piecewise linear approximation coefficients** for the sigmoid lookup table loaded into the NLU.

### Sync Flag Mechanism

From US20180197068A1:
- One sync flag register per virtual write port
- **Sync watcher**: Specifies which loop iteration to synchronize on, required count threshold
- **Sync producer**: Increments counter on specified tensor dimension completion
- **Stall conditions**: TTU stalls when sync flag count < threshold

This maps to the `branch=0x04` instructions (148 occurrences) which likely encode **sync barriers** where the scalar core waits for DMA or vector operations to complete.

### Instruction Distribution

Instructions arrive via instruction bus with a **7-bit header** containing tile bitmap. Each tile inspects the bitmap to determine if it should consume the instruction, then forwards to the next tile in the ring. This explains:
- `branch=0x1A` instructions (165 occurrences): These are likely **ring bus instruction distribution** commands that program which tiles receive which data/instructions.
- `vs_reg_v1`, `vs_reg_w`: May encode tile bitmap or destination tile selectors in some instruction types.

---

## 15. Refined Unknowns (Post-Patent Analysis) <a name="15-unknowns"></a>

### Now Understood (via patent cross-reference)
- **`unk_3` (18 bits)**: Almost certainly **TTU configuration** — loop mask, stride, or limit values for tensor traversal. The high bit patterns (0x3ffa0, etc.) match TTU loop mask encoding.
- **`branch` values**: Map closely to the patent opcode table (4=DMAOp.In through 12=ScalarFenceOp).
- **`v_op=0x13`** at offset 0x590 with `v_offset=0x92`: Likely the **requantization operation** — the NLU applying scale/offset conversion after MAC accumulation.
- **`v_op_2` float constants**: NLU configuration — piecewise linear function coefficients (sigmoid), scale factors, or bias values loaded into the non-linear unit.

### Still Unknown
- **`unk_0` (1 bit)**: Could be instruction pipeline hint, or DMA direction bit (appears in branch=0x7 which is DMAOp.N-W).
- **Exact bit packing within `unk_3`**: Which sub-fields encode stride vs. limit vs. mask — would need systematic experimentation with different tensor shapes.
- **`v_cmd` sub-commands**: 20 distinct values; likely DMA channel selector or TTU register selector, but exact mapping needs more program variants.
- **Requantization multiplier location**: The fixed-point scale factor that combines input_scale × weight_scale / output_scale must be embedded somewhere in the bitstream (possibly in the `v_op_2=7` constant `0xfc056600` or in the `imm_scalar` at 0x6B0).

---

## 16. Compiled Model Structure & Edge TPU / CPU Split <a name="16-compiled-model"></a>

### How the edgetpu_compiler Produces Output

The `edgetpu_compiler` takes a quantized INT8 TFLite model and produces a new TFLite model where supported operations are replaced with a custom op:

**Original TFLite (e.g., 256×256 Dense):**
```
[0] QUANTIZE
[1] FULLY_CONNECTED
[2] QUANTIZE
```

**Compiled TFLite (fully mapped):**
```
[0] edgetpu-custom-op    ← Contains DarwiNN binary (instructions + weights)
```

### The `edgetpu-custom-op`

- The `edgetpu-custom-op` is a TFLite custom operator that **always runs on the Edge TPU hardware**
- Its opaque data blob contains the complete DarwiNN executable: instruction bitstream + weight parameters
- At runtime, libedgetpu intercepts this custom op, sends the DarwiNN binary to the Edge TPU over USB, and retrieves results
- The string `"edgetpu-custom-op"` appears at a fixed offset (~100 bytes) in the compiled TFLite file

### Partial Mapping (CPU + Edge TPU Split)

When a model contains ops the Edge TPU cannot handle, the compiler creates a **split model**:

**Example: Dense → Dense → SIN → Dense**
```
Compiled ops:
  [0] edgetpu-custom-op    ← First 2 Dense layers on Edge TPU
  [1] DEQUANTIZE           ← CPU
  [2] SIN                  ← CPU (unsupported on Edge TPU)
  [3] QUANTIZE             ← CPU
  [4] FULLY_CONNECTED      ← CPU (stranded after the break)
  [5] QUANTIZE             ← CPU
```

**Key rules for the split:**
1. The compiler creates **at most one `edgetpu-custom-op` subgraph** — it cannot split execution across multiple Edge TPU segments
2. Once an unsupported op breaks the chain, **all subsequent ops run on CPU**, even if individually supported (the compiler reports: "More than one subgraph is not supported")
3. The compiler log clearly reports the status of each operator: "Mapped to Edge TPU" vs. reason for CPU fallback
4. Ops that fall back to CPU include reasons like: "Operation is working on an unsupported data type", "Operation is otherwise supported, but not mapped due to some unspecified limitation"

### How to Check the Split

1. **Compiler log** (most reliable): Shows every operator and whether it mapped
2. **TFLite op inspection**: In the compiled model, `edgetpu-custom-op` = Edge TPU, everything else = CPU
3. **Netron**: Visually shows the `edgetpu-custom-op` node and any remaining standard TFLite ops

### Compiler Version

All tests performed with: `Edge TPU Compiler version 16.0.384591198`

---

## 17. Matrix Size Limits & Weight Caching vs. Streaming <a name="17-matrix-limits"></a>

### Compilation Test Results

Systematically tested Dense(N, N) layers (no bias) with INT8 quantization:

| Matrix Size | Weights | Compiled Size | On-Chip Cached | Streamed | Status |
|---|---|---|---|---|---|
| 256×256 | 64 KB | 101 KB | 66 KB | 0 B | Fully cached |
| 512×512 | 256 KB | 297 KB | — | — | OK |
| 1024×1024 | 1.0 MB | 1.1 MB | — | — | OK |
| 2048×2048 | 4.0 MB | 4.1 MB | — | — | OK |
| 4096×4096 | 16.0 MB | 16.1 MB | — | — | OK |
| 8192×8192 | 64.0 MB | 64.0 MB | 0 B | 64 MB | Fully streamed |
| 12288×12288 | 144.0 MB | — | — | — | OK (compiled) |

All sizes compiled successfully and all ops mapped to Edge TPU. The compiler has no apparent matrix size limit.

### Two Parameter Modes: Cached vs. Streamed

The Edge TPU has **~8 MB of on-chip SRAM** for parameter storage. The compiler reports how parameters are handled:

**Small model (256×256, 64 KB weights):**
```
On-chip memory used for caching model parameters: 66.00KiB
On-chip memory remaining for caching model parameters: 7.67MiB
Off-chip memory used for streaming uncached model parameters: 0.00B
```
→ Weights loaded once into SRAM, reused across inferences. **Fast.**

**Large model (8192×8192, 64 MB weights):**
```
On-chip memory used for caching model parameters: 0.00B
On-chip memory remaining for caching model parameters: 7.88MiB
Off-chip memory used for streaming uncached model parameters: 64.00MiB
```
→ **Zero** bytes cached. All 64 MB streamed over USB **every inference call**. **Slow.**

### Performance Implications

| Mode | Behavior | Bottleneck |
|---|---|---|
| **Cached** (weights ≤ ~8 MB) | Weights loaded to SRAM once at model load. MAC array reads from on-chip memory. | Compute-bound (MAC array throughput) |
| **Streamed** (weights > ~8 MB) | Host pushes weights over USB every inference. | I/O-bound (USB 2.0 ~40 MB/s effective) |

**Practical maximum for cached execution:**
- ~8 MB of weights → **~2,896×2,896 square matrix** (8,388,608 bytes)
- This is where the Edge TPU delivers its full 4 TOPS performance
- Beyond this, USB transfer time dominates and performance degrades proportionally to weight size

**Streaming overhead estimate for 8192×8192:**
- 64 MB over USB 2.0 at ~40 MB/s → ~1.6 seconds per inference just for weight transfer
- The actual MAC computation at 4 TOPS would take ~17 ms
- So the model is **~100× slower** than its compute potential due to I/O

### Recommendation for Custom Computation

For maximum throughput, **keep weight matrices under ~8 MB** (e.g., 2048×2048 or smaller). If you need larger transforms, consider:
- Breaking them into multiple cached sub-matrices and combining results on the host
- Using batch inference to amortize the overhead (process many inputs per weight load)
- Accepting the streaming penalty if throughput isn't critical

---

## 18. Partial Model Loading & On-the-Fly Weight Updates <a name="18-partial-loading"></a>

### Architecture Supports Independent Data Streams

The Edge TPU architecture explicitly separates instructions, parameters, and activations at every level — from the USB protocol to the DarwiNN executable format. This makes partial model loading and on-the-fly weight updates architecturally supported.

### USB Protocol: Separate Tags for Each Data Type

```
Tag 0: Instructions (program code)
Tag 1: Input activations (input tensor)
Tag 2: Parameters (weights)
Tag 3: Output activations (output tensor)
```

These are sent as independent USB transfers via `llsend()`. From `simple.py`:
```python
llsend(dev, prog, 0)              # Send instructions (once)
llsend(dev, b"\xaa"*0x80, 2)      # Send weights (can be repeated with new data)
llsend(dev, input_data, 1)        # Send input (each inference)
```

The hardware processes these independently — the `parameterPopRunControl` data path handles weight ingestion separately from the `infeedRunControl` activation path and the instruction queue.

### DarwiNN Executable: Two-Phase Loading

For cached models (weights ≤ ~8 MB), the compiler creates **two separate executables**:

1. **`PARAMETER_CACHING` executable** — Contains only the weight data and a small instruction program (e.g., 843 instructions for SSDLite) that loads weights into wide memory SRAM. Run once at model load time.

2. **`EXECUTION_ONLY` executable** — Contains only the inference instructions (e.g., 14,629 instructions for SSDLite) with zero embedded parameters. Assumes weights are already in SRAM. Run every inference.

This split means the runtime **already knows how to update weights without reloading the instruction program**.

### FieldOffset Patching: Relocatable Base Addresses

The `FieldOffset` mechanism in the DarwiNN executable specifies where the runtime patches base addresses into the instruction bitstream at load time:

| Description | What it patches |
|---|---|
| `BASE_ADDRESS_PARAMETER (2)` | Where parameter data lives in memory |
| `BASE_ADDRESS_INPUT_ACTIVATION (1)` | Where input tensor lives in memory |
| `BASE_ADDRESS_OUTPUT_ACTIVATION (0)` | Where output tensor lives in memory |
| `BASE_ADDRESS_SCRATCH (3)` | Where scratch buffer lives in memory |

These are patched independently via `Bundle::Alu::MOVI` instructions at specific bit positions in the bitstream. This means the same instruction program can be pointed at different parameter memory regions.

### Parameter Caching Token: Shared Weight Management

The `parameter_caching_token` (uint64) field in the DarwiNN executable enables multiple models to share cached parameters:

```
// Parameter-caching executables with the same token can cache their
// parameters together on the TPU SRAM.
parameter_caching_token:uint64;
```

This was designed for model versioning — if two model versions share the same backbone weights, they can share the cached parameters by using the same token.

### Practical Weight Swapping Approaches

#### Approach 1: Raw USB (via edgetpuxray)
1. Compile a Dense(N→M) model once with `edgetpu_compiler`
2. Send instructions (tag=0) once at startup
3. Before each inference, send new weight bytes via `llsend(dev, new_weights, 2)`
4. Send input and read output as normal

This is essentially what `STAND_ALONE` mode does every inference — it sends all parameters over USB each time. The hardware treats each tag=2 packet as the current parameter data.

#### Approach 2: Rerun PARAMETER_CACHING executable
1. Load a cached model normally (runs both PARAMETER_CACHING and EXECUTION_ONLY)
2. When weights need updating, re-run only the PARAMETER_CACHING executable with modified weight data
3. Continue running EXECUTION_ONLY for inference with the new weights

#### Approach 3: Full model swap via libedgetpu
1. Pre-compile multiple models with different weights
2. Use libedgetpu to switch between loaded models
3. Models with the same `parameter_caching_token` may share cache space

### Caveats and Limitations

**Quantization scale mismatch:** The requantization multiplier (combining input_scale × weight_scale / output_scale) is baked into the instruction bitstream — likely in the `v_op_2=7` float constant or the `imm_scalar` field at offset 0x6B0. If your new weights have a very different value range, the on-chip requantization will produce incorrect output. Weight updates should stay within the original quantization range, or you must also patch the requantization constant.

**Address patching:** For Approach 1 (raw USB), the parameter base address in the instructions must match where the hardware expects the data. The `FieldOffset` patching is normally done by libedgetpu at load time. When using raw USB, the instruction program already has addresses baked in from a previous session.

**No partial weight update:** You cannot update just a subset of the weight matrix — the entire parameter blob is sent as one contiguous stream via tag=2. To update a single layer's weights in a multi-layer model, you'd need to resend all parameters.

**Instruction program compatibility:** The instruction program encodes the exact dimensions and memory layout for the weight matrix. You can swap the weight VALUES freely, but the weight DIMENSIONS must match the compiled model exactly.

### Implications for Robotics

This weight-swapping capability enables several robotics use cases:
- **Adaptive controllers:** Train multiple policy versions offline, swap weights on the Edge TPU as conditions change
- **Online learning:** Update a small MLP's weights after each episode (quantize new weights on host, send to Edge TPU)
- **Dynamic transforms:** Change rotation/projection matrices by updating the Dense layer weights
- **Multi-task switching:** Pre-compile one model architecture, load different weight sets for different tasks
- **Kalman filter updates:** Update the state transition matrix as the system model changes

---

## 19. libredgetpu: libedgetpu-Informed Driver & Parameter Caching <a name="19-libredgetpu-driver"></a>

### Overview

The `libredgetpu/` package provides a pure-Python Edge TPU runtime (no libedgetpu, no tflite_runtime). After studying the libedgetpu C++ source code, three key improvements were made to the driver layer:

1. **Proper hardware initialization sequence** matching libedgetpu's phased approach
2. **Inference-only execution** that skips parameter upload when weights are already cached
3. **Register polling** with timeouts instead of blind reads

### Hardware Init: libedgetpu's 8-Phase Sequence

The original `init_hardware()` used hardcoded magic bytes captured from USB traces. The improved version follows libedgetpu's named phases with proper bitfield manipulation:

| Phase | What It Does | Key Registers |
|---|---|---|
| 1. Open | Clear USB inactive PHY mode bits [13:11] in `scu_ctrl_0` | `scu_ctrl_0`, `scu_ctrl_2` |
| 2. EnableReset | Force sleep (`rg_force_sleep`=0b11), **poll** until `cur_pwr_state`=sleeping, pulse `gcbb_credit0` | `scu_ctrl_3`, `gcbb_credit0` |
| 3. QuitReset | Set max clocks (500/250 MHz), **poll** until running, **poll** `scalarCoreRunControl`=0, write idle/tile/deepsleep, **poll** `tileconfig0` | `scu_ctrl_3`, `scalarCoreRunControl`, `tileconfig0` |
| 4. EnableHardwareClockGate | Set `rg_gated_gcb` bits [19:18]=0b01 | `scu_ctrl_2` |
| 5. InitializeChip | Read e-fuse, configure USB endpoints | `omc0_00`, `descr_ep`, `outfeed_chunk_length` |
| 6. DoRunControl | All RunControl registers = 1 (scalar, DMA, mesh, ring) | 13 RunControl registers |
| 7. Interrupts | Enable fatal + top-level interrupts | `fatal_err_int_control`, `top_level_int_*` |
| 8. Misc | USB-trace-only registers (not in libedgetpu open source) | `omc0_d4`, `rambist_ctrl_1`, ABM enables |

Key improvements over the original:
- **Polling with timeouts** via `transport.poll_register()` — reads a register in a loop until `(value & mask) == expected`, with configurable timeout. Replaces blind reads that could miss state transitions.
- **Bitfield manipulation** for `scu_ctrl_3` instead of hardcoded `b"\x5c\x02\x85\x50"` — named constants like `_SCU3_RG_FORCE_SLEEP_SLEEP`, `_SCU3_CUR_PWR_STATE_MASK` make the code self-documenting and robust to different initial register states.

### scu_ctrl_3 Bitfield Decode

| Bits | Field | Values |
|---|---|---|
| [9:8] | `cur_pwr_state` | 0x0=running, 0x2=sleeping (read-only status) |
| [16] | `rg_axi_clk_125m` | 0=250 MHz AXI, 1=125 MHz |
| [17] | `rg_8051_clk_250m` | 0=500 MHz 8051, 1=250 MHz |
| [21:20] | `rg_gcb_clkdiv` | 0=500 MHz GCB (max perf) |
| [23:22] | `rg_force_sleep` | 0b11=force sleep, 0b10=exit sleep |

### Smart Parameter Caching: Inference-Only Mode

The DarwiNN `parameter_caching_token` (uint64) identifies which parameter set is cached in the Edge TPU's wide memory SRAM. `SimpleInvoker.invoke_raw()` now tracks this token and skips the parameter upload on repeated inferences:

```
First call:   execute_cached() → send PC instructions + params + EO instructions + input
Second+ call: execute_inference_only() → send EO instructions + input only
```

The caching logic:
- If `token == 0` or `token != driver._cached_token` → full `execute_cached()`, then store the token
- If `token == driver._cached_token` → `execute_inference_only()` (skip param upload)
- `init_hardware()` and `reset_cached_parameters()` both reset the token to 0

### Hardware-Verified Performance Results

All tests performed on USB-connected Coral Edge TPU (Beagle chip), measuring wall-clock time per `invoke_raw()` call:

| Model | Params | 1st Call (cache) | 2nd+ Call (inference-only) | Speedup | Outputs Match |
|---|---|---|---|---|---|
| Dense 256x256 | 66 KB | 1.32 ms | 0.28 ms | **4.7x** | Yes |
| Dense 1024x1024 | 1.0 MB | 5.55 ms | 0.29 ms | **19x** | Yes |
| Dense 2048x2048 | 4.0 MB | 13.64 ms | 0.27 ms | **50x** | Yes |
| SSD MobileDet | 4.8 MB | 27.0 ms | 12.25 ms | **2.2x** | Yes |

**Key observations:**
- The Dense models show dramatic speedups because their inference is fast (~0.3 ms) but param upload is slow (proportional to weight size over USB 2.0)
- SSD MobileDet shows a smaller speedup because its inference is compute-heavy (~12 ms for 320x320 input through 134 ops), so the ~15 ms param upload is a smaller fraction of total time
- Steady-state inference time for Dense models is ~0.27-0.29 ms regardless of matrix size (256 to 2048) — the MAC array is underutilized for these single-layer models
- All output bytes are identical between cached and inference-only paths, confirming the hardware correctly reuses SRAM-cached parameters

### Timing Stability (10 consecutive inference-only calls)

| Model | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| Dense 256 | 0.25 ms | 0.32 ms | 0.28 ms | ~0.02 ms |
| Dense 1024 | 0.26 ms | 0.36 ms | 0.29 ms | ~0.03 ms |
| Dense 2048 | 0.26 ms | 0.30 ms | 0.27 ms | ~0.01 ms |
| SSD MobileDet | 12.03 ms | 12.92 ms | 12.25 ms | ~0.25 ms |

Very low jitter — suitable for real-time control loops.

### Implications for Custom Computation Throughput

With parameter caching, the Edge TPU can sustain:
- **~3,500 Dense matmuls/sec** (256-2048 range) for cached models — limited by USB round-trip, not compute
- **~80 SSD inferences/sec** — limited by actual MAC compute
- **Weight swapping + caching**: Load new weights once (~5-14 ms for 1-4 MB), then run many inferences at ~0.3 ms each. Ideal for the byte-decomposition technique where 4 weight sets are cycled.

### DMA Hints: The Missing Piece for Complex Models

Initial testing showed that simple models (Dense, SSD MobileDet) worked, but PoseNet and DeepLabV3 produced no output at all — the Edge TPU would accept instructions and parameters but hang silently during execution.

**Root cause**: The DarwiNN executable contains a `DmaHints` table specifying the **exact order and sizes** of USB transfers. Complex models require split-input DMAs and multi-chunk instruction sequences that differ from the simple "send all instructions, send all input" pattern.

#### DMA Hint Types (from executable.fbs)

| Hint Type | Purpose |
|---|---|
| `InstructionHint` | Send instruction bitstream chunk N (via tag=0) |
| `DmaDescriptorHint` | Send/receive data: INPUT (tag=1), PARAMETER (tag=2), OUTPUT (tag=3) with offset+size |
| `InterruptHint` | Read status/completion (EP 0x82) |
| `FenceHint` | Barrier — all prior DMAs must complete before continuing |

#### Decoded DMA Hint Sequences

**Dense 256 (simple — works with naive approach):**
```
[0] INSTRUCTION chunk=0          → send all instructions
[1] INPUT offset=0 size=256      → send all input (one shot)
[2] OUTPUT size=256              → read output
[3] INTERRUPT                    → read status
```

**SSD MobileDet (simple — also works with naive approach):**
```
[0] INSTRUCTION chunk=0          → send instructions
[1] INPUT offset=0 size=307200   → send input (one shot)
[2] OUTPUT size=8136             → read boxes
[3] OUTPUT size=187128           → read scores
[4] INTERRUPT                    → read status
```

**PoseNet (split input — FAILS with naive approach):**
```
[0] INSTRUCTION chunk=0          → send instructions
[1] INPUT offset=0 size=490368   → send first ~490KB of 925KB input
[2] INPUT offset=453824 size=471144 → send overlapping second chunk
[3] OUTPUT size=25424            → read heatmaps
[4] OUTPUT size=45760            → read short offsets
[5] OUTPUT size=81344            → read mid offsets
[6] INTERRUPT                    → read status
```

**DeepLabV3 (split input + multi-chunk instructions — FAILS with naive approach):**
```
[0] INSTRUCTION chunk=0          → send first instruction bitstream (262KB)
[1] INPUT offset=0 size=415536   → send first ~416KB of 790KB input
[2] INPUT offset=386288 size=403224 → send overlapping second chunk
[3] INSTRUCTION chunk=1          → send SECOND instruction bitstream (137KB)
[4] OUTPUT size=278784           → read ASPP features
[5] OUTPUT size=256              → read pooling features
[6] INTERRUPT                    → read status
```

#### Why Split-Input DMAs?

The two input DMAs have **overlapping byte ranges** (e.g., PoseNet: 0–490,368 and 453,824–924,968 overlap by ~36KB). This is because the Edge TPU processes the input image in a **streaming pipeline** — the first DMA feeds the first stage while the MAC array starts computing. The second DMA then feeds the remainder, but the overlap ensures the pipeline doesn't stall at the boundary between stages. The narrow memory per tile isn't large enough to hold the entire input, so it must be streamed in.

#### Implementation

The `execute_dma_hints()` method in `driver.py` walks the hint sequence step-by-step:
- `instruction` → `transport.send(bitstreams[chunk_index], TAG_INSTRUCTIONS)`
- `input` → `transport.send(input_data[offset:offset+size], TAG_INPUT_ACTIVATIONS)`
- `parameter` → `transport.send(params[offset:offset+size], TAG_PARAMETERS)`
- `output` → `transport.read_output(max_size=size)`
- `interrupt` → `transport.read_status()`
- `fence` → no-op (our synchronous USB protocol is inherently fenced)

### Hardware-Verified Results: Complex Models

| Model | Input | 1st Call (cache) | Cached Calls | Output | Result |
|---|---|---|---|---|---|
| PoseNet MobileNet v1 | 481x641x3 | 22.9 ms | 17.4 ms | 152 KB | 17/17 keypoints detected |
| DeepLabV3 MNv2 | 513x513x3 | 34.5 ms | 26.5 ms | 279 KB | Segmentation map produced |

**Notes:**
- PoseNet's Edge TPU output contains raw heatmaps + offset maps. The `PosenetDecoderOp` (CPU custom op) would normally decode these into keypoint coordinates — that post-processing must be reimplemented in Python. Note: the shipped PoseNet model is **single-person only**; while `max_detections=20` in the FlexBuffer options defines the decoder's output capacity, the model reliably detects one person per image.
- DeepLabV3's Edge TPU output is a 33x33x256 feature map. The original TFLite model has 8 additional CPU ops (CONV_2D, RESIZE_BILINEAR, CONCATENATION, ARG_MAX) that upsample and decode the segmentation — these must be reimplemented in Python/NumPy.
- Both models are fully mapped to the Edge TPU for their `edgetpu-custom-op` portion. No partial mapping issues.

---

## 20. Higher-Precision Arithmetic via Byte Decomposition <a name="20-byte-decomposition"></a>

### The Problem

The Edge TPU only performs int8 × int8 multiply-accumulate. Many robotics and signal processing applications need 16-bit or higher precision. Can we achieve exact higher-precision results using only 8-bit operations?

### The Solution: Byte Decomposition

Any 16-bit integer can be split into two 8-bit halves:

```
x = x_high × 256 + x_low
```

where `x_high` and `x_low` are both uint8 (0–255).

For matrices A and B with 16-bit entries, the product C = A × B decomposes as:

```
A = A_h × 256 + A_l
B = B_h × 256 + B_l

C = A × B
  = (A_h × B_h) × 65536  +  (A_h × B_l) × 256  +  (A_l × B_h) × 256  +  (A_l × B_l)
  = (C_hh << 16)  +  ((C_hl + C_lh) << 8)  +  C_ll
```

This requires **4 uint8 matrix multiplies** on the Edge TPU, then a trivial combination (shifts + additions) on the host CPU. The result is **mathematically exact** — no approximation.

### Generalization to N-Byte Precision

| Target Precision | Bytes per Value | Edge TPU Matmuls | Host Post-Processing |
|---|---|---|---|
| 8-bit | 1 | 1 | None |
| 16-bit | 2 | 4 | 3 additions + shifts |
| 24-bit | 3 | 9 | 8 additions + shifts |
| 32-bit | 4 | 16 | 15 additions + shifts |

The pattern is **N² matmuls** for N-byte precision. Each additional byte of precision costs quadratically more compute.

### Why It Works on the Edge TPU

1. **Int32 accumulation**: The MAC array accumulates int8 × int8 products into int32 sum registers internally. For an N×N matrix multiply, each output element is a sum of N products of int8 × int8 (max 127 × 127 = 16,129 per product). With N up to ~2,896 (8 MB cache limit), the accumulated sum fits well within int32 range (max ~47M vs int32 max ~2.1B).

2. **Weight swapping**: Using the partial loading capability (Section 18), the same compiled Dense(N→M) model can be reused for all 4 matmuls — just swap between W_h and W_l weights via tag=2 USB packets. No recompilation needed.

3. **Input splitting**: The input vector is also split into x_h and x_l, sent as two different input activations (tag=1) across the 4 inferences.

### Implementation

```python
import numpy as np

# --- Setup (once) ---
# Split 16-bit weight matrix W into byte halves
W_h = (W >> 8).astype(np.uint8)    # high bytes
W_l = (W & 0xFF).astype(np.uint8)  # low bytes
# Compile ONE Dense(N→M) model, or two (one per weight half)

# --- Per inference ---
# Split 16-bit input x into byte halves
x_h = (x >> 8).astype(np.uint8)
x_l = (x & 0xFF).astype(np.uint8)

# Run 4 inferences on Edge TPU (reuse same instructions, swap weights):
C_hh = edgetpu_inference(x_h, weights=W_h)  # A_h × B_h
C_hl = edgetpu_inference(x_h, weights=W_l)  # A_h × B_l
C_lh = edgetpu_inference(x_l, weights=W_h)  # A_l × B_h
C_ll = edgetpu_inference(x_l, weights=W_l)  # A_l × B_l

# Combine on host CPU (exact 32-bit result):
C = (C_hh << 16) + (C_hl << 8) + (C_lh << 8) + C_ll
```

### Signed Values

For signed 16-bit values (-32768 to +32767):
- **Option A**: Convert to unsigned by adding 32768 before splitting, compute, subtract the bias from the result
- **Option B**: Treat the high byte as signed int8 and the low byte as unsigned uint8. The math works out with a sign correction term on the high-high product

### Performance Estimate

For a 2048×2048 matrix (fits in 8 MB cache):
- **Single 8-bit matmul**: ~1 ms on Edge TPU (compute-bound, cached)
- **16-bit via 4 matmuls**: ~4 ms + weight swap overhead (~1 ms per swap via USB) ≈ **7–8 ms total**
- **CPU 16-bit matmul (2048×2048)**: ~50–200 ms on a typical ARM Cortex-A CPU
- **Speedup**: ~10–25× faster than CPU, at 2W power

### Implications

This technique means the Edge TPU's int8 limitation is not a hard precision wall — it's a **throughput tradeoff**. Any precision is achievable at the cost of N² more matmuls. For many robotics applications (Kalman filters, coordinate transforms, control), 16-bit precision is sufficient, and 4× the compute on a 4 TOPS engine still vastly outperforms a CPU.

Combined with on-the-fly weight swapping (Section 18), this becomes practical: compile once, cache instructions, and cycle through the byte-decomposed weight halves at runtime.

---

## 21. Implications for Custom Computation <a name="21-implications"></a>

### What This Means for Non-ML Edge TPU Usage

1. **The coral/ approach (TF → TFLite → edgetpu_compiler) works today** for any algorithm expressible as TensorFlow tensor operations. The FFT example proves this.

2. **The edgetpuxray approach is viable for simple operations** — scalar-only programs work, and the output path is fully understood. Custom programs can be written with `mins()` and executed via `simple.py`.

3. **For vector/tensor operations, the compiler is still needed** because the vector pipeline configuration (v_cmd, v_offset, unk_3, etc.) is not sufficiently decoded to generate from scratch.

4. **Hybrid approach possible**: Use the compiler to generate a template program for a known operation, then modify specific instructions (especially at 0x6B0) and weights to change the computation, then use `simple.py` to execute.

5. **The quantization insight is powerful**: Since the Edge TPU is just an int8 MAC engine, and the actual operation is determined by weights + host-side dequantization, you can potentially implement arbitrary linear transforms by:
   - Using a matrix multiply model (Dense layer)
   - Setting the weight matrix to encode your desired linear transform
   - Adjusting quantization parameters for the desired input/output ranges

### Recommended Next Steps
1. Generate more model variants and compare their .coral programs to decode remaining v_cmd and branch values
2. Focus on the matmul_256 and dense_1_8_mul programs to understand the vector MAC pipeline
3. Use single-stepping (`scalarCoreRunControl = 0x03`) to trace execution and correlate PC values with register state changes
4. Try modifying individual fields in known-working programs and observing hardware behavior
5. Cross-reference with Google patent US20190050717A1 (referenced in decompiler.py) for TTU architecture details

---

## CPU Post-Processing: TFLite Op Graphs for Models with CPU Ops

Models that include unsupported ops (custom or standard) are split by the compiler into an Edge TPU portion (`edgetpu-custom-op`) and CPU-side ops. The CPU ops must be reimplemented to get final results. Analysis below is from `parse_full()` introspection of compiled `*_edgetpu.tflite` models.

### DeepLabV3 MNv2 PASCAL — 9 operators (1 Edge TPU + 8 CPU)

```
Op 0: edgetpu-custom-op
  in:  T[0]  MobilenetV2/input     [1,513,513,3]   scale=0.00781250 zp=128
  out: T[6]  ASPP features         [1,33,33,256]    scale=0.03520937 zp=0
  out: T[7]  image_pooling          [1,1,1,256]     scale=0.01242244 zp=0

Op 1: RESIZE_BILINEAR
  T[7] [1,1,1,256] → T[8] [1,33,33,256]  (tile image pooling to match ASPP)
  Scale preserved: 0.01242244

Op 2: HARD_SWISH (actually requantization — scales differ, both zp=0)
  T[8] scale=0.01242244 → T[9] scale=0.03520937  (match ASPP scale for concat)

Op 3: CONCATENATION
  [T[9] resized_pool, T[6] ASPP] → T[10] [1,33,33,512]  scale=0.03520937

Op 4: CONV_2D (concat_projection, 1×1)
  T[10] [33,33,512] × T[11] [256,1,1,512] + T[5] [256] → T[12] [33,33,256]
  Weight: scale=0.00222522 zp=132, Bias: int32 scale=0.00007835
  Output: scale=0.02648678 zp=0 (ReLU6 activation)

Op 5: CONV_2D (logits, 1×1)
  T[12] [33,33,256] × T[13] [21,1,1,256] + T[4] [21] → T[14] [33,33,21]
  Weight: scale=0.00400870 zp=124, Bias: int32 scale=0.00010618
  Output: scale=0.20149837 zp=36

Op 6-7: RESIZE_BILINEAR ×2
  T[14] [33,33,21] → T[15] [33,33,21] → T[16] [513,513,21]

Op 8: ARG_MAX
  T[16] [513,513,21] → T[17] [513,513] (class indices 0-20)
```

**Key findings:**
- The HARD_SWISH (op 2) is not the activation function — it's a requantization. Both input and output have zp=0; only scales differ. In float space it's near-identity.
- Weight extraction from `parse_full()` buffers is identical to TensorFlow schema approach (verified 0.0 max diff).
- For efficiency, argmax can be applied at [33,33] without the two RESIZE_BILINEAR upscaling ops.

### PoseNet MNv1 — 2 operators (1 Edge TPU + 1 custom CPU op)

```
Op 0: edgetpu-custom-op
  in:  T[0]  input                  [1,481,641,3]   scale=0.00784314 zp=128
  out: T[1]  heatmaps               [1,31,41,17]    (score logits, pre-sigmoid)
  out: T[2]  short_offsets           [1,31,41,34]    (sub-pixel y,x per keypoint)
  out: T[3]  mid_offsets             [1,31,41,64]    (inter-keypoint displacements)

Op 1: PosenetDecoderOp (custom)
  in:  T[1], T[2], T[3]
  out: T[4]  pose_keypoints          [1,20,17,2]    (y,x in pixel coords)
  out: T[5]  pose_keypoint_scores    [1,20,17]      (per-keypoint scores)
  out: T[6]  pose_scores             [1,20]          (instance-level scores)
  out: T[7]  pose_count              [1]             (number of detected poses)

  FlexBuffer custom_options:
    max_detections = 20
    score_threshold = 0.2
    stride = 16
    nms_radius = 10.0
    _output_quantized = ...
    _support_output_type_float_in_quantized_op = ...
```

**Key findings:**
- Dequantization of offsets uses `extra_scale = 1.0/stride` (from C++ `DequantizeTensor`). This converts from pixel-space offsets to block-space for the decoder.
- The PersonLab algorithm (arXiv:1803.08225): build local maxima queue → greedy pose decode via mid-range offsets + short-range refinement → soft keypoint NMS → instance rescoring.
- Edge list is 32 entries: 16 forward + 16 backward edges. Backward edge IDs (≥16) must be offset by +16 when indexing into mid_offsets (which has layout [fwd_y, fwd_x, bwd_y, bwd_x] × 16 edges = 64 channels).
- The nms_radius parameter (10.0 pixels) is divided by stride (16) to get block-space radius (0.625) for NMS comparisons.

### TFLite Flatbuffer Schema Field Indices

Corrected field indices discovered during implementation:

```
Model:     version=0, operator_codes=1, subgraphs=2, description=3, buffers=4
Tensor:    shape=0, type=1, buffer=2, name=3, quantization=4
Buffer:    data=0
OperatorCode: deprecated_builtin_code=0(int8), custom_code=1(old)/4(new), builtin_code=3(int32)
Operator:  opcode_index=0, inputs=1, outputs=2, builtin_options=3, builtin_options_type=4, custom_options=5
SubGraph:  tensors=0, inputs=1, outputs=2, operators=3
QuantizationParameters: min=0, max=1, scale=2, zero_point=3
```

**Common pitfall:** Model.buffers is field 4, NOT field 3. Field 3 is the `description` string (e.g. "Exported from Subgraph.").

Builtin opcode enum values (subset): ADD=0, CONCATENATION=2, CONV_2D=3, DEPTHWISE_CONV_2D=4, RESIZE_BILINEAR=23, CUSTOM=32, ARG_MAX=56, QUANTIZE=80, HARD_SWISH=114.

---

## Appendix: Reference Files

### Key Source Files
- `edgetpuxray/decompiler.py` — The assembler/disassembler, use `mins()` to build instructions
- `edgetpuxray/simple.py` — Hardware interface, `open_device()`, `write_register()`, `llsend()`
- `edgetpuxray/connect.py` — Full inference without libedgetpu
- `edgetpuxray/beagle_csr_offsets.h` — Complete hardware register map
- `edgetpuxray/executable.fbs` — DarwiNN executable FlatBuffers schema
- `coral/scripts/fft_model.py` — FFT as TensorFlow graph example

### Google Patent References
- [US20190050717A1](https://patents.google.com/patent/US20190050717A1/en): "Methods and Systems for Neural Network Accelerator" — tile architecture, MAC array, memory hierarchy, instruction bus distribution
- [US20180197068A1 / US9836691](https://patents.google.com/patent/US20180197068A1/en): "Neural Network Instruction Set Architecture" — TTU registers, TensorOp/DMAOp format, sync flags, opcode table
- [GB2558980A](https://patents.google.com/patent/GB2558980A/en): "Neural Network Instruction Set Architecture" — opcode table (Table 300: opcodes 0-12), TTU loop nest field, layer type encoding
- [US20210373895A1](https://patents.google.com/patent/US20210373895A1/en): "Tensor Traversal Engine for Strided Memory Access" — TTU counter/stride/limit programming, address generation formula, pointer array mode

### External References
- [Q-Engineering: Google Coral Edge TPU Explained in Depth](https://qengineering.eu/google-corals-tpu-explained.html): Estimated 64×64 systolic array, 480 MHz, 4 TOPS
- DarwiNN namespace visible in `libedgetpu.so` as `platforms::darwinn::*`
- Biology-themed naming: Coral (brand), Beagle (chip codename from CSR header), Mendel (OS), DarwiNN (platform)

---

*Document generated from analysis of the CustomEdgeTPU repository.*
*All instruction decoding verified against 23 .coral binary programs.*
*Patent cross-references added from US20190050717A1, US20180197068A1, GB2558980A, US20210373895A1.*
*Matrix size limits and caching/streaming behavior verified with edgetpu_compiler v16.0.384591198.*
*Compiled model structure verified by TFLite op inspection across fully-mapped and partially-mapped models.*
*Memory architecture confirmed by parsing DarwiNN MultiExecutable flatbuffers (executable.fbs schema) from compiled models.*
*Narrow memory per-tile usage measured from UsedNarrowMemoryBytesPerTile field across 4 models.*
*Partial loading analysis based on executable.fbs schema (ExecutableType, FieldOffset, parameter_caching_token), USB protocol tags, and simple.py/connect.py data flow patterns.*
*libredgetpu driver improvements informed by libedgetpu C++ source analysis. Parameter caching performance verified on hardware with Dense 256/1024/2048 and SSD MobileDet models.*
*DMA hint discovery and split-input execution verified on hardware with PoseNet MobileNet v1 and DeepLabV3 MNv2 PASCAL models.*
*CPU post-processing op graphs mapped via parse_full() TFLite introspection. DeepLabV3 weight extraction verified identical to TF-schema approach. PoseNet PersonLab decoder ported from C++ reference implementation (google-coral/edgetpu). FlexBuffer custom_options parsed for decoder parameters.*

---

## Systolic Array Tiling Behavior (from qengineering.eu analysis)

### Input Vector Tiling Along Array Width
The 64×64 MAC array has fixed dimensions. When an input vector exceeds the 64-column array width, the compiler **splits the input into 64-element chunks** processed sequentially. Intermediate partial sums accumulate in on-chip buffers, with weights reloaded for each partition.

Example: Dense(256) → 256-wide input split into 4 passes of 64 elements each.

### Implications for Large-Kernel Conv2D (SpotTracker)
A Conv2D with an H×W kernel has H×W weights per filter. For SpotTracker's coordinate-weighted soft argmax:
- **16×16 input**: 256 weights/filter → ~4 tiling passes → compiles and runs
- **64×64 input**: 4096 weights/filter → ~64 tiling passes → compiler fails

This suggests the compiler has a **tiling pass limit or memory budget** for a single Conv2D operation. The failure at ≥64×64 is likely a compiler limitation on the number of weight-reload passes, not a fundamental hardware constraint.

### Activation Unit is Hardwired
The post-MAC activation function is implemented as ROM (lookup table), not programmable logic. Likely ReLU. The compiler controls whether the activation is applied or bypassed, but cannot change the function itself. This is consistent with our finding that the requantization multiplier is baked into EO instructions — the multiplier is applied before the fixed activation lookup.

### Weight Loading Cost
Each tiling pass requires a full weight reload to the MAC array. This means operations with large weight counts (like full-image Conv2D kernels) are bottlenecked not just by compute but by **weight transfer bandwidth**. The EfficientNet design principle — avoid doubling weight loads by fusing 1×1+3×3 convolutions — directly reflects this hardware cost.

*Source: https://qengineering.eu/google-corals-tpu-explained.html — architectural overview cross-referenced with our independent analysis findings.*

---

## Critical Finding: Edge TPU Always Outputs uint8 (Experiment 2, 2026-02-05)

### Discovery
The Edge TPU hardware **always outputs bytes in uint8 format** (0-255), regardless of the TFLite model's specified output tensor type. For models with `int8` output type (TFLite TensorType=9), the raw bytes from USB have their **sign bit flipped** compared to what CPU TFLite produces.

### The Relationship
```
hw_byte = cpu_byte XOR 0x80
```
Equivalently: `hw_int8 = cpu_int8 + 128` (with wrapping), or reading hardware bytes as uint8 and using `zp_uint8 = zp_int8 + 128`.

### Verification
Tested across 8 different input images on a SpotTracker int8-output model (16×16, all ops mapped to Edge TPU). Every single byte matched perfectly:
```
Image          CPU_int8    HW_int8     CPU+128
uniform        [-42, +45]  [+86, -83]  [+86, -83]  ✓
pixel_UL       [-117, -31] [+11, +97]  [+11, +97]  ✓
pixel_UR       [+33, -31]  [-95, +97]  [-95, +97]  ✓
... (all 8 images: 100% match)
```

### Correct Dequantization for int8 Models
```python
raw_uint8 = np.frombuffer(hw_bytes, dtype=np.uint8)
raw_int8 = (raw_uint8 ^ 0x80).view(np.int8)  # XOR sign bit
float_output = (raw_int8.astype(np.float32) - zp_int8) * scale
```

### Why This Wasn't Noticed Before
All previously tested models (MobileNet, SSD, DeepLabV3, PoseNet, Looming) use **uint8 output** type. The libredgetpu code already reads output as uint8, so dequantization was correct for those models. The SpotTracker was the first model built with `inference_output_type = tf.int8`, exposing the bug.

### Impact on libredgetpu
- `simple_invoker.py:invoke()`: Added dtype check — when `output_tensor.dtype == 9` (INT8), XOR bytes with 0x80 before dequantization.
- `spot_tracker.py:track()`: Removed incorrect `negate_outputs` hack. Now uses proper uint8→int8 conversion.
- The `negate_outputs` field in JSON sidecar was entirely unnecessary — the perceived "negation" was the uint8/int8 sign-bit difference.

*Verified on hardware: 26 SpotTracker tests + 28 existing hardware tests = 54 total, all pass.*
