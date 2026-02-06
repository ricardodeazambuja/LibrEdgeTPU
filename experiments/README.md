# libredgetpu Experiments

Systematic experiments validating Edge TPU behavior. Each script is self-contained
with a docstring documenting the question, method, expected outcome, and findings.

## Naming Convention

`exp{N}[letter]_short_description.py` — sequential numbering, letters for variants.

## Experiment Index

### Weight Swapping (MatMulEngine)
| Script | Question | Finding |
|--------|----------|---------|
| `exp1_recompilation_consistency.py` | Do different weights produce the same DarwiNN instructions? | PC instructions are architecture-determined. EO instructions encode requant multiplier — differ when weight_scale changes. |
| `exp1b_detailed_check.py` | Detailed comparison of instruction bitstreams across weight variants. | Confirmed exp1 findings with finer granularity. |
| `exp1c_fixed_scale.py` | If we patch int8 weights but preserve quant metadata, are instructions identical? | YES — EO and PC instructions identical. Only PC param blob changes. Validates set_weights() algorithm. |

### SpotTracker (CPU vs Edge TPU)
| Script | Question | Finding |
|--------|----------|---------|
| `exp2_spot_tracker_cpu_vs_tpu.py` | How exactly do Edge TPU outputs differ from CPU TFLite for the soft argmax tracker? | Edge TPU always outputs uint8. For int8 models: `hw_byte = cpu_byte XOR 0x80`. Fix: XOR before interpreting as int8. The perceived "negation" was a uint8/int8 sign-bit difference, not a computation error. |

## Running

Most experiments require TensorFlow and `edgetpu_compiler` (for model creation/compilation).
Hardware experiments additionally require a USB Edge TPU.

```bash
# From repo root:
python -m experiments.exp2_spot_tracker_cpu_vs_tpu
```

## Recording Results

After running, append findings to this README and update the experiment's docstring.
Save raw output logs as `exp{N}_results.txt` in this directory.
