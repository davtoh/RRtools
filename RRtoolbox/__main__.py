"""
Main Command-Line Interface entrypoint for RRtools.
Usage:
    python -m RRtoolbox <command> [options]

Commands:
    super-res    Restore images with single/multi-frame super-resolution
    hardware     Display hardware acceleration and compute profile
"""
import sys
import argparse
import logging
from RRtoolbox.hardware import detect_hardware

def cmd_hardware(args):
    profile = detect_hardware()
    print(profile.summary())

def cmd_super_res(args):
    from RRtoolbox.restoration.super_resolution import SuperResolutionEngine, Backend
    
    engine = SuperResolutionEngine(
        scale=args.scale,
        backend=Backend(args.backend),
        model_path=args.model_path,
    )
    print(f"Executing Super-Resolution (Scale: {args.scale}x, Inputs: {len(args.images)})...")
    res = engine.restore(args.images, scale=args.scale)
    saved = res.save(args.output)
    
    print("\n================ SUPER-RESOLUTION RESULTS ================")
    print(f"  Mode:            {res.mode.upper()} ({res.backend})")
    print(f"  Input Dim:       {res.input_shape[1]}x{res.input_shape[0]}")
    print(f"  Output Dim:      {res.output_shape[1]}x{res.output_shape[0]} ({res.scale}x)")
    print(f"  Input Sharpness: {res.sharpness_input:.2f}")
    print(f"  Output Sharpness:{res.sharpness_output:.2f} (+{res.sharpness_gain_pct:.1f}% vs baseline)")
    print(f"  Execution Time:  {res.elapsed_ms:.2f} ms")
    print(f"  Saved Image:     {saved}")
    print("==========================================================\n")

def main():
    parser = argparse.ArgumentParser(
        prog="python -m RRtoolbox",
        description="RRtools: Advanced Scene & Image Restoration Platform"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: hardware
    p_hw = subparsers.add_parser("hardware", help="Show hardware and GPU acceleration profile")
    p_hw.set_defaults(func=cmd_hardware)

    # Command: super-res
    p_sr = subparsers.add_parser("super-res", help="Super-resolution single/multi-frame enhancement")
    p_sr.add_argument("images", nargs="+", help="Input image paths or glob pattern")
    p_sr.add_argument("-s", "--scale", type=int, default=2, choices=[2, 3, 4], help="Upscale factor (default: 2)")
    p_sr.add_argument("-o", "--output", type=str, default="super_restored.jpg", help="Output destination path")
    p_sr.add_argument("-b", "--backend", type=str, default="auto", choices=["auto", "edge_directed", "dnn_superres", "onnx"])
    p_sr.add_argument("-m", "--model-path", type=str, default=None, help="Path to optional model weights")
    p_sr.set_defaults(func=cmd_super_res)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
