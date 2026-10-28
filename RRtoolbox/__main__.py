"""
Main Command-Line Interface entrypoint for RRtools.
Usage:
    python -m RRtoolbox <command> [options]

Commands:
    super-res    Restore images with single/multi-frame super-resolution
    panorama     Multi-angle 360 spherical/cylindrical panoramic stitcher
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

def cmd_panorama(args):
    from RRtoolbox.geometry.panoramic import PanoramicStitcher, ProjectionType, BlendMode
    
    stitcher = PanoramicStitcher(
        projection=ProjectionType(args.projection),
        blend_mode=BlendMode(args.blend),
        fov_degrees=args.fov,
        pyramid_levels=args.levels,
        crop_borders=not args.no_crop,
        backend=args.backend,
    )
    print(f"Executing Panoramic Stitch ({args.projection.upper()}, Blend: {args.blend.upper()}, Views: {len(args.images)})...")
    res = stitcher.stitch(args.images)
    saved = res.save(args.output)
    
    print("\n================= PANORAMIC STITCH RESULTS =================")
    print(f"  Projection:      {res.projection.upper()}")
    print(f"  Blend Mode:      {res.blend_mode.upper()}")
    print(f"  Input Views:     {res.num_images}")
    print(f"  Output Dim:      {res.output_shape[1]}x{res.output_shape[0]}")
    print(f"  RANSAC Inliers:  {res.inlier_count} ({res.inlier_ratio:.1%} inlier ratio)")
    print(f"  Execution Time:  {res.elapsed_ms:.2f} ms")
    print(f"  Saved Panorama:  {saved}")
    print("============================================================\n")

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

    # Command: panorama
    p_pa = subparsers.add_parser("panorama", help="Multi-angle 360 spherical/cylindrical panoramic stitcher")
    p_pa.add_argument("images", nargs="+", help="Input image paths across overlapping viewing angles")
    p_pa.add_argument("-o", "--output", type=str, default="panorama_restored.jpg", help="Output panorama destination")
    p_pa.add_argument("-p", "--projection", type=str, default="spherical", choices=["spherical", "cylindrical", "planar"], help="Geometric projection model (default: spherical)")
    p_pa.add_argument("-b", "--blend", type=str, default="multiband", choices=["multiband", "feather", "linear", "none"], help="Seam blending method (default: multiband)")
    p_pa.add_argument("-f", "--fov", type=float, default=65.0, help="Horizontal Field-of-View in degrees (default: 65.0)")
    p_pa.add_argument("-l", "--levels", type=int, default=4, help="Laplacian pyramid levels for multi-band blending (default: 4)")
    p_pa.add_argument("--backend", type=str, default="auto", choices=["auto", "native", "opencv"], help="Stitching backend engine (default: auto)")
    p_pa.add_argument("--no-crop", action="store_true", help="Disable automatic outer black border trimming")
    p_pa.set_defaults(func=cmd_panorama)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
