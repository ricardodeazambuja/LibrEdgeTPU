"""DarwiNN executable extraction from edgetpu-custom-op data.

Parses the Package → MultiExecutable → Executable flatbuffer hierarchy
embedded inside the customOptions blob of a compiled TFLite model.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .darwinn.Package import Package
from .darwinn.MultiExecutable import MultiExecutable
from .darwinn.Executable import Executable
from .darwinn.ExecutableType import ExecutableType
from .darwinn.AnyHint import AnyHint
from .darwinn.DmaDescriptorHint import DmaDescriptorHint
from .darwinn.InstructionHint import InstructionHint
from .darwinn.InterruptHint import InterruptHint
from .darwinn.FenceHint import FenceHint
from .darwinn.OutputLayer import OutputLayer as OutputLayerFB

__all__ = [
    "parse_darwinn",
    "apply_field_offsets",
    "relayout_output",
    "DarwiNNExecutable",
    "BitstreamInfo",
    "FieldOffsetInfo",
    "LayerInfo",
    "TileLayout",
    "DmaStep",
    "TYPE_STAND_ALONE",
    "TYPE_PARAMETER_CACHING",
    "TYPE_EXECUTION_ONLY",
]

# ExecutableType enum values
TYPE_STAND_ALONE = ExecutableType.STAND_ALONE          # 0
TYPE_PARAMETER_CACHING = ExecutableType.PARAMETER_CACHING  # 1
TYPE_EXECUTION_ONLY = ExecutableType.EXECUTION_ONLY    # 2


@dataclass
class FieldOffsetInfo:
    desc: int       # Description enum
    batch: int
    name: str
    offset_bit: int


@dataclass
class BitstreamInfo:
    data: bytes
    field_offsets: List[FieldOffsetInfo]


@dataclass
class TileLayout:
    """Output tile layout for de-scattering TYXZ → YXZ."""
    y_tile_id_map: List[int]
    x_tile_id_map: List[int]
    tile_byte_offsets: List[int]
    x_local_byte_offset: List[int]
    y_local_y_offset: List[int]
    x_local_y_row_size: List[int]


@dataclass
class LayerInfo:
    name: str
    size_bytes: int
    y_dim: int
    x_dim: int
    z_dim: int
    zero_point: int = 0
    dequant_factor: float = 0.0
    tile_layout: Optional[TileLayout] = None


@dataclass
class DmaStep:
    """One step in the DMA hint sequence."""
    kind: str          # "instruction", "input", "output", "parameter", "interrupt", "fence"
    chunk_index: int = 0      # for "instruction": which bitstream chunk
    offset: int = 0           # for "input"/"output"/"parameter": byte offset in buffer
    size: int = 0             # for "input"/"output"/"parameter": transfer size
    name: str = ""            # layer name (for input/output)
    direction: int = 0        # 0=INFEED (host→dev), 1=OUTFEED (dev→host)


@dataclass
class DarwiNNExecutable:
    exec_type: int               # TYPE_STAND_ALONE / TYPE_PARAMETER_CACHING / TYPE_EXECUTION_ONLY
    bitstreams: List[BitstreamInfo]
    parameters: Optional[bytes]  # raw parameter blob (weights)
    input_layers: List[LayerInfo]
    output_layers: List[LayerInfo]
    parameter_caching_token: int = 0
    scratch_size_bytes: int = 0
    dma_steps: List[DmaStep] = field(default_factory=list)


def _parse_tile_layout(layer_fb) -> Optional[TileLayout]:
    """Extract OutputLayout (tile de-scatter tables) if present."""
    any_type = layer_fb.AnyLayerType()
    if any_type != 1:  # 1 = OutputLayer
        return None
    any_data = layer_fb.AnyLayer()
    if any_data is None:
        return None
    ol = OutputLayerFB()
    ol.Init(any_data.Bytes, any_data.Pos)
    layout = ol.Layout()
    if layout is None:
        return None

    def _vec(getter, length_getter):
        return [getter(i) for i in range(length_getter())]

    return TileLayout(
        y_tile_id_map=_vec(layout.YCoordinateToLinearTileIdMap,
                           layout.YCoordinateToLinearTileIdMapLength),
        x_tile_id_map=_vec(layout.XCoordinateToLinearTileIdMap,
                           layout.XCoordinateToLinearTileIdMapLength),
        tile_byte_offsets=_vec(layout.LinearizedTileByteOffset,
                               layout.LinearizedTileByteOffsetLength),
        x_local_byte_offset=_vec(layout.XCoordinateToLocalByteOffset,
                                  layout.XCoordinateToLocalByteOffsetLength),
        y_local_y_offset=_vec(layout.YCoordinateToLocalYOffset,
                               layout.YCoordinateToLocalYOffsetLength),
        x_local_y_row_size=_vec(layout.XCoordinateToLocalYRowSize,
                                 layout.XCoordinateToLocalYRowSizeLength),
    )


def _parse_layer(layer_fb) -> LayerInfo:
    """Extract LayerInfo from a DarwiNN Layer flatbuffer object."""
    name = ""
    raw_name = layer_fb.Name()
    if raw_name is not None:
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

    zp = 0
    df = 0.0
    numerics = layer_fb.Numerics()
    if numerics is not None:
        zp = numerics.ZeroPoint()
        df = numerics.DequantizationFactor()

    tile_layout = _parse_tile_layout(layer_fb)

    size_bytes = layer_fb.SizeBytes()
    if size_bytes < 0 or size_bytes > 100_000_000:
        raise ValueError(f"Output layer size_bytes={size_bytes} out of safe range")

    return LayerInfo(
        name=name,
        size_bytes=size_bytes,
        y_dim=layer_fb.YDim(),
        x_dim=layer_fb.XDim(),
        z_dim=layer_fb.ZDim(),
        zero_point=zp,
        dequant_factor=df,
        tile_layout=tile_layout,
    )


def _parse_bitstream(bs_fb) -> BitstreamInfo:
    """Extract BitstreamInfo from an InstructionBitstream flatbuffer object."""
    data = bytes(bs_fb.BitstreamAsNumpy())

    field_offsets = []
    for i in range(bs_fb.FieldOffsetsLength()):
        fo = bs_fb.FieldOffsets(i)
        meta = fo.Meta()
        name = ""
        if meta is not None:
            raw = meta.Name()
            if raw is not None:
                name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        field_offsets.append(FieldOffsetInfo(
            desc=meta.Desc() if meta else 0,
            batch=meta.Batch() if meta else 0,
            name=name,
            offset_bit=fo.OffsetBit(),
        ))

    return BitstreamInfo(data=data, field_offsets=field_offsets)


_DESC_TO_KIND = {0: "output", 1: "input", 2: "parameter", 3: "scratch"}


def _parse_dma_hints(exe) -> List[DmaStep]:
    """Extract DMA hint sequence from a DarwiNN Executable flatbuffer."""
    dma_hints = exe.DmaHints()
    if dma_hints is None:
        return []

    steps = []
    for i in range(dma_hints.HintsLength()):
        hint = dma_hints.Hints(i)
        hint_type = hint.AnyHintType()
        direction = hint.Direction()

        if hint_type == AnyHint.InstructionHint:
            ih = InstructionHint()
            ih.Init(hint.AnyHint().Bytes, hint.AnyHint().Pos)
            steps.append(DmaStep(
                kind="instruction",
                chunk_index=ih.InstructionChunkIndex(),
                direction=direction,
            ))
        elif hint_type == AnyHint.DmaDescriptorHint:
            dh = DmaDescriptorHint()
            dh.Init(hint.AnyHint().Bytes, hint.AnyHint().Pos)
            meta = dh.Meta()
            kind = _DESC_TO_KIND.get(meta.Desc(), "unknown") if meta else "unknown"
            name = ""
            if meta and meta.Name():
                raw = meta.Name()
                name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            steps.append(DmaStep(
                kind=kind,
                offset=dh.OffsetInBytes(),
                size=dh.SizeInBytes(),
                name=name,
                direction=direction,
            ))
        elif hint_type == AnyHint.InterruptHint:
            irh = InterruptHint()
            irh.Init(hint.AnyHint().Bytes, hint.AnyHint().Pos)
            steps.append(DmaStep(kind="interrupt", direction=direction))
        elif hint_type == AnyHint.FenceHint:
            steps.append(DmaStep(kind="fence", direction=direction))

    return steps


def _parse_executable(exe_bytes) -> DarwiNNExecutable:
    """Parse a single DarwiNN Executable from its serialized flatbuffer bytes."""
    exe = Executable.GetRootAsExecutable(exe_bytes, 0)

    bitstreams = []
    for i in range(exe.InstructionBitstreamsLength()):
        bitstreams.append(_parse_bitstream(exe.InstructionBitstreams(i)))

    params = None
    if exe.ParametersLength() > 0:
        params = bytes(exe.ParametersAsNumpy())

    input_layers = []
    for i in range(exe.InputLayersLength()):
        input_layers.append(_parse_layer(exe.InputLayers(i)))

    output_layers = []
    for i in range(exe.OutputLayersLength()):
        output_layers.append(_parse_layer(exe.OutputLayers(i)))

    dma_steps = _parse_dma_hints(exe)

    # Reorder output_layers to match the physical byte order in the raw
    # USB output buffer.  The DMA hint steps with kind="output" define the
    # true order in which output tensors are read from the Edge TPU.
    # The flatbuffer output_layers order does NOT always match.
    output_dma_names = [s.name for s in dma_steps if s.kind == "output"]
    if output_dma_names and len(output_dma_names) == len(output_layers):
        name_to_layer = {layer.name: layer for layer in output_layers}
        if all(n in name_to_layer for n in output_dma_names):
            output_layers = [name_to_layer[n] for n in output_dma_names]

    return DarwiNNExecutable(
        exec_type=exe.Type(),
        bitstreams=bitstreams,
        parameters=params,
        input_layers=input_layers,
        output_layers=output_layers,
        parameter_caching_token=exe.ParameterCachingToken(),
        scratch_size_bytes=exe.ScratchSizeBytes(),
        dma_steps=dma_steps,
    )


def parse_darwinn(custom_op_data: bytes) -> List[DarwiNNExecutable]:
    """Extract DarwiNN executables from edgetpu-custom-op customOptions.

    Returns a list of DarwiNNExecutable (typically 2 for cached models:
    PARAMETER_CACHING + EXECUTION_ONLY, or 1 for STAND_ALONE).
    """
    buf = custom_op_data if isinstance(custom_op_data, (bytes, bytearray)) else bytes(custom_op_data)

    # Find DWN1 magic — Package flatbuffer starts 4 bytes before the identifier.
    # Scan for the magic to skip over any prefix bytes in the custom op data.
    magic_pos = buf.find(b"DWN1")
    if magic_pos < 4:
        raise ValueError("DWN1 magic not found in custom op data")
    package_offset = magic_pos - 4

    # Validate that the root offset at package_offset points back into the
    # buffer and that the flatbuffer identifier matches.  This guards against
    # false-positive matches (e.g., "DWN1" appearing inside a data payload).
    if package_offset + 8 > len(buf):
        raise ValueError(
            f"DWN1 magic at offset {magic_pos} too close to end of buffer"
        )
    root_offset = int.from_bytes(buf[package_offset:package_offset + 4], "little")
    root_table_pos = package_offset + root_offset
    if root_table_pos < package_offset or root_table_pos >= len(buf):
        raise ValueError(
            f"DWN1 magic at offset {magic_pos} has invalid root offset "
            f"({root_offset}), likely a false match"
        )

    pkg = Package.GetRootAsPackage(buf, package_offset)

    # Package → serialized MultiExecutable bytes
    me_bytes = pkg.SerializedMultiExecutableAsNumpy()
    if isinstance(me_bytes, int) and me_bytes == 0:
        raise ValueError("Package has no SerializedMultiExecutable")

    me = MultiExecutable.GetRootAsMultiExecutable(me_bytes, 0)

    executables = []
    for i in range(me.SerializedExecutablesLength()):
        # SerializedExecutables returns raw bytes for each executable
        exe_bytes = me.SerializedExecutables(i)
        if isinstance(exe_bytes, (bytes, bytearray)):
            executables.append(_parse_executable(exe_bytes))
        elif isinstance(exe_bytes, np.ndarray):
            executables.append(_parse_executable(bytes(exe_bytes)))
        else:
            # It's a string (flatbuffers returns bytes-as-string)
            executables.append(_parse_executable(exe_bytes))

    return executables


def apply_field_offsets(bitstream_data: bytes, field_offsets: List[FieldOffsetInfo],
                        bases: dict) -> bytearray:
    """Patch base addresses into instruction bitstream at specified bit offsets.

    *bases* maps Description enum values to uint32 addresses:
      {0: parameter_base, 1: input_base, 2: output_base, 3: scratch_base}

    Returns a new bytearray with patches applied.
    """
    if not field_offsets:
        return bytearray(bitstream_data)

    result = bytearray(bitstream_data)

    for fo in field_offsets:
        if fo.desc not in bases:
            warnings.warn(f"FieldOffset desc={fo.desc} name={fo.name!r} has no base address, skipping")
            continue

        base = bases[fo.desc]
        bit_off = fo.offset_bit

        # Write 32-bit value at the specified bit offset (little-endian bit order)
        byte_off = bit_off // 8
        bit_shift = bit_off % 8

        if bit_shift == 0 and byte_off + 4 <= len(result):
            # Aligned case — simple write
            result[byte_off : byte_off + 4] = base.to_bytes(4, "little")
        elif byte_off + 5 <= len(result):
            # Unaligned — read 5 bytes, insert 32 bits, write back
            val = int.from_bytes(result[byte_off : byte_off + 5], "little")
            mask = ((1 << 40) - 1) & ~(0xFFFFFFFF << bit_shift)
            val = (val & mask) | (base << bit_shift)
            result[byte_off : byte_off + 5] = (val & ((1 << 40) - 1)).to_bytes(5, "little")
        else:
            warnings.warn(f"FieldOffset at bit {bit_off} extends past bitstream end, skipping")

    return result


def relayout_output(raw_bytes: bytes, layer: LayerInfo) -> np.ndarray:
    """De-scatter raw Edge TPU output from TYXZ tile format to standard YXZ.

    The Edge TPU stores output activations in a tiled memory layout (TYXZ).
    This function uses the OutputLayout tables from the DarwiNN executable
    to rearrange the bytes into standard [y_dim, x_dim, z_dim] order.

    If no tile layout is present (simple outputs), the raw bytes are reshaped
    directly.

    Returns a uint8 numpy array of shape [y_dim, x_dim, z_dim].
    """
    y_dim, x_dim, z_dim = layer.y_dim, layer.x_dim, layer.z_dim
    tl = layer.tile_layout

    if tl is None:
        # No tiling — assume simple packed layout
        return np.frombuffer(raw_bytes[:y_dim * x_dim * z_dim],
                             dtype=np.uint8).reshape(y_dim, x_dim, z_dim)

    src = np.frombuffer(raw_bytes, dtype=np.uint8)
    dest = np.zeros((y_dim, x_dim, z_dim), dtype=np.uint8)

    # Validate tile layout array lengths match tensor dimensions
    if len(tl.y_tile_id_map) < y_dim or len(tl.y_local_y_offset) < y_dim:
        raise ValueError(
            f"Tile layout Y maps too short for y_dim={y_dim}: "
            f"y_tile_id_map={len(tl.y_tile_id_map)}, "
            f"y_local_y_offset={len(tl.y_local_y_offset)}"
        )
    if (len(tl.x_tile_id_map) < x_dim
            or len(tl.x_local_byte_offset) < x_dim
            or len(tl.x_local_y_row_size) < x_dim):
        raise ValueError(
            f"Tile layout X maps too short for x_dim={x_dim}: "
            f"x_tile_id_map={len(tl.x_tile_id_map)}, "
            f"x_local_byte_offset={len(tl.x_local_byte_offset)}, "
            f"x_local_y_row_size={len(tl.x_local_y_row_size)}"
        )

    for y in range(y_dim):
        y_tile = tl.y_tile_id_map[y]
        y_local = tl.y_local_y_offset[y]
        for x in range(x_dim):
            tile_id = y_tile + tl.x_tile_id_map[x]
            if tile_id < 0 or tile_id >= len(tl.tile_byte_offsets):
                raise ValueError(
                    f"Tile ID {tile_id} out of range (y={y}, x={x}, "
                    f"num_tiles={len(tl.tile_byte_offsets)})"
                )
            base = (tl.tile_byte_offsets[tile_id]
                    + y_local * tl.x_local_y_row_size[x]
                    + tl.x_local_byte_offset[x])
            if base < 0:
                raise ValueError(
                    f"Negative base offset {base} at (y={y}, x={x})"
                )
            if base + z_dim > len(src):
                raise ValueError(
                    f"Tile data at (y={y}, x={x}) extends past buffer end: "
                    f"offset {base}+{z_dim} > {len(src)}"
                )
            dest[y, x, :] = src[base:base + z_dim]

    return dest
