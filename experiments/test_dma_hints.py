#!/usr/bin/env python3
"""Inspect DMA hints of the Gabor model to understand output tiling.

Also test: what if we use relayout_output() to reorder the output?
The _base.py has a relayout_output method that might handle this.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # Load the standard template
    from libredgetpu.optical_flow.templates import get_template
    tflite_path, json_path = get_template(64)

    print(f"Model: {tflite_path}")

    # Parse the TFLite model
    from libredgetpu.tflite_parser import parse_full
    with open(tflite_path, 'rb') as f:
        tflite_bytes = f.read()
    model = parse_full(tflite_bytes)

    print(f"\nModel description: {model.description}")
    print(f"Subgraph inputs: {model.graph_inputs}")
    print(f"Subgraph outputs: {model.graph_outputs}")

    print(f"\nTensors:")
    for i, t in enumerate(model.tensors):
        print(f"  {i}: name={t.name}, shape={t.shape}, dtype={t.dtype}, "
              f"scale={t.scale}, zp={t.zero_point}")

    print(f"\nOperators:")
    for i, op in enumerate(model.operators):
        print(f"  {i}: {op.opcode_name} inputs={op.inputs} outputs={op.outputs}")

    # Parse the DarwiNN executable
    from libredgetpu.delegate import parse_delegate
    delegate = parse_delegate(tflite_path)

    print(f"\nDarwiNN executable:")
    print(f"  Input layers: {len(delegate.input_layers)}")
    for i, layer in enumerate(delegate.input_layers):
        print(f"    {i}: name={layer.name}, shape={layer.shape}, "
              f"data_type={layer.data_type}")

    print(f"  Output layers: {len(delegate.output_layers)}")
    for i, layer in enumerate(delegate.output_layers):
        print(f"    {i}: name={layer.name}, shape={layer.shape}, "
              f"data_type={layer.data_type}")

    print(f"\n  DMA hints: {len(delegate.dma_hints)} steps")
    for i, hint in enumerate(delegate.dma_hints):
        print(f"    {i}: kind={hint.kind}, offset={hint.offset}, "
              f"size={hint.size}")

    # Check relayout_output
    print(f"\n  Relayout output info:")
    if hasattr(delegate, 'output_relayout'):
        print(f"    relayout: {delegate.output_relayout}")
    else:
        print(f"    No relayout info found")

    # Now check the actual output tensor shape from the output layers
    if delegate.output_layers:
        out_layer = delegate.output_layers[0]
        print(f"\n  Output layer details:")
        print(f"    shape: {out_layer.shape}")
        if hasattr(out_layer, 'y_dim'):
            print(f"    y_dim: {out_layer.y_dim}")
        if hasattr(out_layer, 'x_dim'):
            print(f"    x_dim: {out_layer.x_dim}")
        if hasattr(out_layer, 'z_dim'):
            print(f"    z_dim: {out_layer.z_dim}")

    # Check if there are output relayout parameters in the model
    from libredgetpu._base import EdgeTPUModelBase

    class InspectModel(EdgeTPUModelBase):
        def _default_output_size(self):
            return 64 * 64 * 8

    from libredgetpu.optical_flow_module import OpticalFlow
    flow = OpticalFlow.from_template(64, pooled=False)

    # Don't open (don't need USB), just look at parsed data
    print(f"\n  Parsed model info:")
    print(f"    Input: shape={flow._input_info.shape}, "
          f"scale={flow._input_info.scale}, zp={flow._input_info.zero_point}")
    print(f"    Output: shape={flow._output_info.shape}, "
          f"scale={flow._output_info.scale}, zp={flow._output_info.zero_point}")

    # Check the delegate for relayout info
    print(f"\n  Delegate output_layers:")
    for i, ol in enumerate(flow._delegate.output_layers):
        print(f"    {i}: {ol}")
        # Print all attributes
        for attr in dir(ol):
            if not attr.startswith('_'):
                try:
                    val = getattr(ol, attr)
                    if not callable(val):
                        print(f"      {attr} = {val}")
                except:
                    pass

    # Check if relayout is applied
    print(f"\n  Relayout info:")
    if hasattr(flow, '_relayout_params'):
        print(f"    relayout_params: {flow._relayout_params}")
    if hasattr(flow._delegate, '_relayout_info'):
        print(f"    relayout_info: {flow._delegate._relayout_info}")

    # Check the _read_output method
    print(f"\n  Output size: {flow._default_output_size()}")

    # Look at the actual relayout_output code
    import inspect
    if hasattr(flow, '_relayout_output'):
        print(f"\n  _relayout_output source:")
        try:
            src = inspect.getsource(flow._relayout_output)
            print(src[:500])
        except:
            print("    Could not get source")

    print("\nDone.")


if __name__ == "__main__":
    main()
