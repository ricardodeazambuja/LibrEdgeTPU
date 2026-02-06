#!/usr/bin/env python3
"""Offline tests: parse Edge TPU models without hardware.

Verifies TFLite parsing, DarwiNN extraction, quantization params, and
executable structure for models from the EdgeTPUModelZoo.

No Edge TPU hardware required.
"""

import pytest

from libredgetpu.tflite_parser import parse as parse_tflite
from libredgetpu.delegate import (
    parse_darwinn,
    apply_field_offsets,
    FieldOffsetInfo,
    TYPE_PARAMETER_CACHING,
    TYPE_EXECUTION_ONLY,
    TYPE_STAND_ALONE,
)
from tests.model_zoo import get_model, MODELS


@pytest.fixture(params=sorted(MODELS.keys()))
def model_name(request):
    return request.param


def test_parse_model(model_name):
    """Parse a model and verify its structure."""
    path = get_model(model_name)

    with open(path, "rb") as f:
        data = f.read()

    # TFLite parsing
    model = parse_tflite(data)
    assert len(model.custom_op_data) > 0, "customOptions should not be empty"
    assert model.input_tensor.scale > 0, "Input scale should be positive"

    # DarwiNN parsing
    executables = parse_darwinn(model.custom_op_data)
    assert len(executables) in (1, 2), f"Expected 1 or 2 executables, got {len(executables)}"

    # Verify executable types
    types = {exe.exec_type for exe in executables}
    if len(executables) == 2:
        assert TYPE_PARAMETER_CACHING in types, "Expected PARAMETER_CACHING executable"
        assert TYPE_EXECUTION_ONLY in types, "Expected EXECUTION_ONLY executable"
    elif len(executables) == 1:
        assert TYPE_STAND_ALONE in types, "Single executable should be STAND_ALONE"

    # Every executable should have at least one bitstream
    for exe in executables:
        assert len(exe.bitstreams) > 0, "Executable should have at least one bitstream"

    # The execution executable (EO or STAND_ALONE) should have output layers
    eo = [e for e in executables if e.exec_type != TYPE_PARAMETER_CACHING]
    assert len(eo) > 0, "No execution executable found"
    assert len(eo[0].output_layers) > 0, "Execution executable should have output layers"


def test_apply_field_offsets_returns_patched():
    """Verify apply_field_offsets returns a patched bytearray, not None."""
    # Create a 16-byte zeroed bitstream
    bitstream = bytes(16)

    # FieldOffset at bit 0 with desc=1 (input)
    fo = FieldOffsetInfo(desc=1, batch=0, name="test", offset_bit=0)

    # Patch base address 0x12345678 at bit offset 0
    bases = {1: 0x12345678}
    result = apply_field_offsets(bitstream, [fo], bases)

    assert isinstance(result, bytearray), "Should return bytearray"
    assert len(result) == 16, "Length should be preserved"
    # Verify the patched bytes (little-endian)
    assert result[0] == 0x78
    assert result[1] == 0x56
    assert result[2] == 0x34
    assert result[3] == 0x12


def test_apply_field_offsets_empty_returns_copy():
    """Verify empty field_offsets returns unmodified bytearray."""
    bitstream = b"\xAA\xBB\xCC\xDD"
    result = apply_field_offsets(bitstream, [], {})
    assert isinstance(result, bytearray)
    assert result == bytearray(b"\xAA\xBB\xCC\xDD")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
