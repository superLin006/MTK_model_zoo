#!/usr/bin/env python3
"""
Convert SenseVoice PyTorch/TorchScript model to TFLite/MLIR

Usage:
    python3 pt2tflite.py -i model/sensevoice_complete.pt -o model/sensevoice_complete.tflite --float 1
"""

import argparse
import torch
import mtk_converter
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description='Convert SenseVoice TorchScript to TFLite')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input TorchScript (.pt) file path')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output TFLite/MLIR file path')
    parser.add_argument('--float', type=int, default=1,
                        help='Use float32 (1) or quantize (0). Default: 1')
    parser.add_argument('--input_shapes', type=str, default="[[1,166,560],[1],[1],[1],[1]]",
                        help='Input shapes as string. Default: [[1,166,560],[1],[1],[1],[1]] for 10s audio with 4 scalar prompt inputs')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"SenseVoice TorchScript to TFLite Conversion")
    print(f"{'='*80}\n")

    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Float mode: {args.float}")
    print(f"Input shapes: {args.input_shapes}")

    # Parse input shapes
    import ast
    input_shapes = ast.literal_eval(args.input_shapes)
    print(f"Parsed input shapes: {input_shapes}")

    # Determine output format
    output_format = "tflite" if args.output.endswith('.tflite') else "mlir"
    print(f"Output format: {output_format}")

    try:
        # Create converter
        print("\nCreating MTK converter...")
        converter = mtk_converter.PyTorchConverter.from_script_module_file(
            args.input,
            input_shapes=input_shapes,
            input_types=[torch.float32, torch.int32, torch.int32, torch.int32, torch.int32],  # 5 inputs: features + 4 prompt scalars
        )

        # Set quantization mode
        if args.float:
            converter.quantize = False
            print("Quantization: Disabled (float32 mode)")
        else:
            converter.quantize = True
            print("Quantization: Enabled (int8 mode)")

        # Convert to TFLite
        print("\nConverting to TFLite...")
        if output_format == "tflite":
            tflite_model = converter.convert_to_tflite()

            # Save TFLite model
            print(f"Saving TFLite model to: {args.output}")
            with open(args.output, 'wb') as f:
                f.write(tflite_model)
        else:
            # Convert to MLIR
            print("Converting to MLIR...")
            mlir_file = converter.convert_to_mlir(args.output)
            print(f"Saved MLIR model to: {mlir_file}")

        print(f"\n✅ Conversion completed successfully")
        print(f"Output saved to: {args.output}")

        # Print model info
        print(f"\nModel Information:")
        print(f"  Input shapes: {input_shapes}")
        print(f"  Input types: [float32, int32, int32, int32, int32]")
        print(f"  Output: CTC logits [1, T+4, 25055]")

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'='*80}")
    print(f"Conversion completed")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
