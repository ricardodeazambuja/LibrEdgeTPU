# MatMulEngine — Systematic Design Notes

## Goal
Runtime weight-swapping matrix multiplication on Edge TPU.
Ship pre-compiled Dense(N×N) templates; swap weight values at runtime.

## Architecture Assumptions (need validation)

### A1: TFLite weight buffer can be patched in-place
- **Claim**: Find the NxN weight buffer in the uncompiled TFLite flatbuffer, overwrite bytes, recompile.
- **Risk**: Flatbuffer stores offsets/lengths. If the buffer region is length-prefixed internally, in-place patching of same-size data should be fine. But if the compiler uses checksums or if the buffer isn't contiguous, this breaks.
- **Test**: Patch known values, recompile, verify compilation succeeds and parameters change.

### A2: Recompilation produces same instruction structure
- **Claim**: Compiling Dense(N) with different weight values produces the same DarwiNN instructions but different parameter blobs.
- **Evidence**: We already proved div2.coral and div4.coral are byte-identical programs. Same Dense(N) model with different weights should behave similarly.
- **Risk**: Compiler could choose different quantization ranges or tiling strategies based on weight statistics.
- **Test**: Compile same Dense(N) with 3 different weight distributions, diff the instruction bitstreams.

### A3: Recompiled params can replace original params via USB
- **Claim**: Extract PARAMETER_CACHING params from recompiled model, send via tag=2, run EXECUTION_ONLY from original template.
- **Risk**: Instructions contain baked-in requantization multipliers. If the recompiled model has different quant params, the original EXECUTION_ONLY instructions won't match.
- **Critical**: The requant multiplier = (input_scale * weight_scale) / output_scale. If TF quantization picks different scales for different weight values, the instructions won't match the new params.
- **Test**: Compare quant scales across recompilations.

### A4: We can force consistent quantization across recompilations
- **Claim**: By using the same representative dataset and model structure, TF lite converter will pick the same input/output scales.
- **Risk**: Weight scale is derived from the actual weight values. Different weights → different weight_scale → different requant multiplier → instructions don't match.
- **THIS IS THE CRITICAL QUESTION**: Can we swap params extracted from a recompiled model into the original template's instruction stream?

## Key Hypothesis to Test First

**H1: The recompiled model's EXECUTION_ONLY instructions are identical to the template's.**

If H1 is TRUE: We can extract params from recompilation and use them with the original template instructions. This is the `set_weights()` path.

If H1 is FALSE: We need to also swap the EXECUTION_ONLY instructions, not just params. This means `set_weights()` must update both instructions and params — effectively loading a whole new model.

**H2: The PARAMETER_CACHING instructions are identical across recompilations of the same architecture.**

If TRUE: We only need to swap the param blob. If FALSE: We need to send new PC instructions too.

## Experiment Plan

### Experiment 1: Recompilation consistency (OFFLINE, no hardware)
1. Generate Dense(256) with weight values = all zeros
2. Generate Dense(256) with weight values = random uniform [-1, 1]
3. Generate Dense(256) with weight values = random uniform [-0.1, 0.1]
4. Compile all three with edgetpu_compiler
5. Extract DarwiNN executables from each
6. Compare:
   a. PC instruction bitstreams (byte-for-byte)
   b. EO instruction bitstreams (byte-for-byte)
   c. Parameter blob sizes
   d. TFLite quantization scales (input, weight, output)

### Experiment 2: Weight patching round-trip (OFFLINE)
1. Create Dense(256) TFLite with known weights W1
2. Patch weights to W2 in the uncompiled TFLite
3. Recompile patched model
4. Extract params from recompiled model
5. Verify params differ from original
6. Load both in TF interpreter, run same input, verify outputs differ

### Experiment 3: Cross-model param injection (HARDWARE)
Only if Experiments 1-2 succeed:
1. Compile template with weights W1
2. Compile separately with weights W2
3. Load template (W1) on Edge TPU
4. Replace params with W2's extracted params
5. Run inference, compare output against:
   a. W2 compiled model run natively
   b. CPU reference W2 @ x

## Current Code Issues Found

1. **Missing uncompiled TFLite**: `templates/` only has `*_edgetpu.tflite`, but `set_weights()` needs `dense_{n}.tflite` (uncompiled). Template gen saves it but the files aren't in the directory.

2. **auto_calibrate() not implemented**: Referenced in test but method doesn't exist in MatMulEngine.

3. **Assumption A3 is untested**: The entire `set_weights()` path assumes recompiled params are compatible with original instructions. This is the riskiest assumption.

## Experiment 1 Results (2026-02-04)

### Setup
- Dense(256) with 4 weight variants: zeros, identity, uniform[-1,1], uniform[-0.01,0.01]
- zeros and identity failed edgetpu_compiler (degenerate weight distributions)
- uniform_wide and uniform_narrow compiled successfully

### Findings

**CRITICAL: Type label bug found in initial analysis**
The type mapping {0: "PARAMETER_CACHING"} was wrong. Correct: {0: STAND_ALONE, 1: PARAMETER_CACHING, 2: EXECUTION_ONLY}.

**Corrected results:**

| Component | Template | uniform_wide | uniform_narrow |
|---|---|---|---|
| PC instructions | f801dadfbd170b86 | f801dadfbd170b86 | f801dadfbd170b86 |
| EO instructions | 48e5c6833578dcbc | 58f1c2d9d977b77f | c710819a73298e2d |
| PC param size | 67,584 | 67,584 | 67,584 |
| weight_scale | ~0.005 (random) | 0.00785 | 0.0000787 |
| output_scale | 5.912 | 57.705 | 0.564 |

**H2 CONFIRMED**: PARAMETER_CACHING instructions are **identical** across all weight variants.
**H1 REJECTED**: EXECUTION_ONLY instructions **differ** (embed requant multiplier).

### Implications for set_weights()

**If weight_scale changes** (different TFLite quant metadata), set_weights() would need to update both
PC params AND EO instructions — effectively loading a whole new model.

**But Experiment 1c showed this is avoidable**: by patching int8 values while preserving the
TFLite quant metadata, all instruction streams stay identical. Only the PC param blob changes.
This is the approach used in the actual implementation.

## Experiment 1c Results (2026-02-04)

### The winning strategy: patch int8 weights, keep quant metadata

Patched 5 different int8 weight sets into the same uncompiled TFLite (preserving
the original weight_scale in the flatbuffer). All compiled to:
- **Identical EO instructions** (hash: 8dc946ed1bd75be3)
- **Identical PC instructions** (hash: f801dadfbd170b86)
- **Identical output_scale** (5.4954)
- **Different PC params** (weight blobs — confirming weights ARE encoded)

This confirms:
1. The edgetpu_compiler reads weight_scale from the TFLite quantization metadata
2. It computes requant_multiplier from (input_scale * weight_scale / output_scale)
3. As long as these scales are unchanged, the instructions are deterministic
4. Only the DarwiNN parameter blob changes with different weight values

### Correct set_weights() algorithm:
1. Quantize float32 weights to int8 using **template's fixed weight_scale**
2. Patch int8 bytes into uncompiled TFLite at the weight buffer offset
3. Recompile with edgetpu_compiler (scales preserved → same instructions)
4. Extract PC params from compiled model (the DarwiNN weight blob)
5. Send new PC params via USB tag=2 (instructions stay the same)

### Key constraint:
- User's float32 weights must fit in the representable range: [-128 * scale, 127 * scale]
- Values outside this range get clipped to int8 extremes
- For the template with scale=0.00085, range is approximately [-0.109, 0.108]
- Templates with different weight initializations have different scales

## Input Quantization Bug (found during hardware testing)

### Symptom
`matmul()` returned all zeros for float inputs in [-1, 1].

### Root cause
The representative dataset in `template_gen.py` used `np.zeros([1, n]) + i` for `i in range(256)`,
producing values [0, 255]. TF's quantizer chose `input_scale=1.0, zp=0`, meaning float 0.0 → uint8 0
and float 255.0 → uint8 255. So float inputs in [-1, 1] mapped to uint8 [0, 1] — essentially all zeros.

The old `output_scale` was ~5.0 (only ~50 distinct output values across the typical range), making
the float API nearly useless.

### Fix
Changed representative dataset to `rng.uniform(-1.0, 1.0, [1, n])`. New quantization:
- `input_scale ≈ 0.00784, input_zp ≈ 127` → float [-1, 1] maps to uint8 [0, 254]
- `output_scale ≈ 0.019` → 260x finer output resolution than before
- `weight_scale ≈ 0.00085` (unchanged — determined by random Keras init)

### Lesson
The representative dataset determines the ENTIRE quantization profile. It must cover the actual
input range the user will provide. For a general-purpose matmul engine with float [-1, 1] inputs,
the representative dataset must use [-1, 1] values.

Also: hardware tests that passed before were passing with huge tolerances that hid the bug.
Tests must validate actual numerical correctness, not just "didn't crash."

## Status
- [x] Experiment 1: Recompilation consistency — H1 rejected, H2 confirmed
- [x] Experiment 1c: Fixed-scale patching — CONFIRMED: only PC params change
- [x] Experiment 2: Full hardware round-trip — completed via 6 hardware tests:
      test_identity_matrix, test_weight_swap_changes_output, test_reset_weights,
      test_multiple_swaps, test_known_matmul, test_set_weights_recompile.
      All exercise the full pipeline: quantize → patch → recompile → extract → upload → infer.
- [x] Hardware tests: 19/19 pass (13 offline + 6 hardware)
- [x] Full test suite: 40/40 pass (10 parse + 19 matmul + 11 inference)
- [x] Fix: Input quantization (representative dataset [-1,1])
- [x] Fix: Generate uncompiled TFLite alongside compiled templates
- [x] Fix: Replaced auto_calibrate() test with set_weights_recompile test
