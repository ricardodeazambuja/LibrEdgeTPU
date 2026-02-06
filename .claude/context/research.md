# Research Notes
> Note: Some file paths below reference the old multi-project repository (coral/, edgetpuxray/, edgetpu_compiler_binaries/) which is no longer part of this repo.

## Optical Flow Research (2026-02-05)
- Full analysis in `opticalflow_edgetpu.md`
- **Classical methods not viable**: Lucas-Kanade (per-pixel 2x2 solve), Horn-Schunck (iterative), Block matching (ArgMax not supported)
- **Training-free design possible** via **Soft Argmax**: `flow = sum(displacement * softmax(correlation))`
- Key insight: ArgMax replaced by Softmax + Mul + Sum (all Edge TPU compatible)
- Design: Gabor features → Pad+Slice shifts → correlation volume → soft argmax
- **EdgeFlowNet** (neural): 100 FPS, but requires QAT training. Repo: github.com/pearwpi/EdgeFlowNet

## Image Compression Assessment (2026-02-05)
- **NOT recommended** for Edge TPU
- Entropy coding (Huffman/RLE) is 30-50% of JPEG encoding — fundamentally incompatible with systolic arrays
- USB overhead (150ms) exceeds CPU time (2-30ms) for DCT/DWT transforms
- Realistic speedup vs CPU: 0.1x-0.5x (slower, not faster)

## edgetpu_compiler Binary Analysis (2026-02-05)
Detailed findings in `edgetpu_compiler_binaries/edgetpu_compiler_bin/edgetpu_compiler_advanced.md`.
- **Binary**: 26.5 MB stripped ELF x86-64, version 16.0.384591198, built from `blaze-out/k8-opt/bin/platforms/darwinn/tflite/edgetpu_compiler`
- **11 CLI flags** (complete, no hidden flags): `-h`, `-v`, `-o <dir>`, `-s`, `-m <10-14>`, `-t <sec>`, `-n <segments>`, `-d`, `-k <step>`, `-i <tensors>`, `-a`
- **Useful flags**: `-m 10` forces standalone/streamed mode (no param caching); `-a` enables multiple Edge TPU subgraphs (experimental); `-i` controls tensor boundaries between TPU/CPU; `-d -k` searches for maximal delegatable op set
- **73 MLIR passes** in 3 dialects: `dwg-*` (15, graph-level), `dwl-*` (53, layer-level), `dwgt-*`/`legalize-*` (5). Pipeline: TFLite → dwg-fbs-import → shape inference → legalize-tfl → cluster/fitter → optimize → buffer-placement → dwg-fbs-translate → compiled TFLite
- **Internal options NOT accessible via CLI**: `dump-tflite-custom-op`, `set_testing_flags`, `pass_pipeline`, `enable_parameter_caching`, `data_parallelism_x/y`, `compile_target`, `device_type`, `input-shapes`
- **Debug file patterns exist but are inaccessible**: `*-instructions-lst.txt` (assembly listing), `*-chips-debug.txt`, `*-instructions-debug.txt`, `*-parameter-caching-instructions-lst.txt`, etc. — gated by internal compiler_options proto, not CLI flags
- **Process model**: parent (CLI) → child (MLIR compilation, ~12 threads) via `clone()` + timeout watcher. Environment variables have no effect (child isolated from parent env).
- **Binary patching attempted and failed**: (1) Redirected "Invalid option" error → crashes on unknown long opts (getopt_long corrupts state). (2) Flipped boolean init sites for `dump-tflite-custom-op` and `set_testing_flags` → MLIR pass config updated but actual dump gated by **second check** at compiler_options proto field `[rsi+0x184]` which is only set by Google's internal build. Two-level gate makes debug output inaccessible without analyzing the proto layout.
- **Conclusion**: No further compiler debug extraction is feasible. Our own disassembler (`decompiler.py`) and hardware experimentation are more productive paths for remaining unknowns.
- **Analysis scripts**: `analyze_compiler.py` (string extraction), `patch_compiler.py` (binary patching + xref), `test_patched.py` (21-test suite for patched binary) — all in `edgetpu_compiler_binaries/edgetpu_compiler_bin/`
