"""Shared constants for the libredgetpu package.

Hardware-determined values from Edge TPU architecture and DarwiNN format.
"""

# DarwiNN parameter blob encoding: int8 weight values are XOR'd with this
# mask before storage.  The same transform applies to Edge TPU output bytes
# when the TFLite output dtype is int8 (dtype == 9).
SIGN_BIT_FLIP = 0x80

# Epsilon used to prevent division by zero during quantization.
QUANT_EPSILON = 1e-9

# Edge TPU systolic array dimensions (64x64 MAC array).
# Parameters are stored in 64-row groups in the DarwiNN blob.
MAC_ARRAY_ROWS = 64

# Wide memory bus width in bytes.  Within each 64-row group, weight bytes
# are arranged in 4-column tiles.
WIDE_BUS_WIDTH = 4
