"""Command line interface.

    lapscale calib board  --input VIDEO|DIR --target charuco --cols 11 --rows 8 --square-mm 5 --marker-mm 3.7 \
                          --out configs/camera/lap.yaml
    lapscale calib import --file stereo_calib.xml --side left --image-size 1280x960 --out configs/camera/x.yaml
    lapscale calib moge   --video surgery.mp4 --out configs/camera/x_moge.yaml       (last resort)
    lapscale calib mask   --camera configs/camera/x.yaml --video surgery.mp4          (add the optical disc)
    lapscale calib show   --camera configs/camera/x.yaml
    lapscale calib target --target charuco --cols 11 --rows 8 --square-mm 5 --marker-mm 3.7 --out board.png

The other stages take ``--camera PATH`` (bypass: any camera YAML, e.g. written by hand from known
intrinsics) or ``--camera moge``.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from . import __version__
from .config import load_config, section


def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", choices=["checkerboard", "charuco"], default="charuco")
    p.add_argument("--cols", type=int, required=False,
                   help="checkerboard: inner corners per row; charuco: squares per row")
    p.add_argument("--rows", type=int, required=False)
    p.add_argument("--square-mm", type=float, required=False)
    p.add_argument("--marker-mm", type=float, default=None, help="charuco marker side")
    p.add_argument("--dictionary", default="DICT_5X5_100", help="charuco ArUco dictionary")
    p.add_argument("--legacy", action="store_true", help="charuco printed with the pre-OpenCV-4.6 layout")


def _spec(a):
    from .calib.targets import TargetSpec
    missing = [n for n in ("cols", "rows", "square_mm") if getattr(a, n) is None]
    if missing:
        sys.exit(f"missing target parameters: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    return TargetSpec(a.target, a.cols, a.rows, a.square_mm, a.marker_mm, a.dictionary, a.legacy)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lapscale", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"lapscale {__version__}")
    p.add_argument("--config", default=None, help="YAML with per-stage defaults (see configs/default.yaml)")
    sub = p.add_subparsers(dest="stage", required=True)

    # ------------------------------------------------------------------ calib
    c = sub.add_parser("calib", help="stage 1: camera intrinsics")
    cs = c.add_subparsers(dest="action", required=True)

    b = cs.add_parser("board", help="calibrate from a video or folder of images of a target")
    b.add_argument("--input", required=True)
    b.add_argument("--out", required=True, help="camera YAML to write")
    _add_target_args(b)
    b.add_argument("--dist-model", choices=["k1k2", "standard", "rational"], default="standard")
    b.add_argument("--zero-tangent", action="store_true")
    b.add_argument("--stride", type=int, default=None, help="scan every Nth frame (default: --max-candidates evenly)")
    b.add_argument("--max-candidates", type=int, default=600)
    b.add_argument("--max-views", type=int, default=60)
    b.add_argument("--holdout", type=float, default=0.2, dest="holdout_fraction")
    b.add_argument("--no-circle", action="store_true", help="do not detect the optical disc")
    b.add_argument("--report-dir", default=None)

    i = cs.add_parser("import", help="bypass: convert intrinsics you already have")
    i.add_argument("--file", required=True, help="OpenCV XML/YAML/JSON")
    i.add_argument("--out", required=True)
    i.add_argument("--side", choices=["mono", "left", "right"], default="mono")
    i.add_argument("--image-size", default=None, help="WxH the intrinsics refer to, if not in the file")
    i.add_argument("--crop-offset", default=None, help="x0,y0 of the crop the intrinsics were computed on")
    i.add_argument("--key-K", default=None)
    i.add_argument("--key-D", default=None)
    i.add_argument("--video", default=None, help="adapt to this video's resolution and detect its optical disc")

    m = cs.add_parser("moge", help="last resort: focal length estimated by MoGe")
    m.add_argument("--video", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--n-frames", type=int, default=20)
    m.add_argument("--moge-version", choices=["v2", "v3"], default="v2")
    m.add_argument("--checkpoint", default=None)
    m.add_argument("--device", default="cuda")

    k = cs.add_parser("mask", help="detect the optical disc of a video and store it in a camera YAML")
    k.add_argument("--camera", required=True)
    k.add_argument("--video", required=True)
    k.add_argument("--out", default=None, help="default: overwrite --camera")

    s = cs.add_parser("show", help="print a camera YAML")
    s.add_argument("--camera", required=True)

    t = cs.add_parser("target", help="render a calibration target to print")
    _add_target_args(t)
    t.add_argument("--out", required=True)
    t.add_argument("--px-per-square", type=int, default=200)

    # ------------------------------------------------------------------ later stages
    for name, text in [("depth", "stage 2: MoGe depth"), ("tool", "stage 3: instrument geometry"),
                       ("scale", "stage 4: scale recovery"), ("run", "all stages on a video (offline / stream)"),
                       ("eval", "evaluation against stereo ground truth"), ("viz", "overlays")]:
        q = sub.add_parser(name, help=f"{text} (not implemented yet)")
        q.add_argument("rest", nargs=argparse.REMAINDER)
    p.calib_actions = {"board": b, "import": i, "moge": m, "mask": k, "show": s, "target": t}
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    cfg = load_config(pre.config)
    if cfg:                                   # config values become defaults; explicit flags still win
        for name, subp in parser.calib_actions.items():
            subp.set_defaults(**section(cfg, "calib", name))
    a = parser.parse_args(argv)

    if a.stage != "calib":
        print(f"'{a.stage}' is not implemented yet (see docs/roadmap.md)", file=sys.stderr)
        return 2

    from .calib import run as R
    if a.action == "board":
        from .calib.calibrate import CalibConfig
        cfg_c = CalibConfig(dist_model=a.dist_model, zero_tangent=a.zero_tangent, max_views=a.max_views,
                            holdout_fraction=a.holdout_fraction)
        R.run_board(a.input, _spec(a), a.out, cfg_c, stride=a.stride, max_candidates=a.max_candidates,
                    detect_circle=not a.no_circle, report_dir=a.report_dir)
    elif a.action == "import":
        R.run_import(a.file, a.out, a.side, R.parse_size(a.image_size), R.parse_pair(a.crop_offset),
                     a.key_K, a.key_D, a.video)
    elif a.action == "moge":
        R.run_moge(a.video, a.out, a.n_frames,
                   dict(version=a.moge_version, checkpoint=a.checkpoint, device=a.device))
    elif a.action == "mask":
        R.run_mask(a.camera, a.video, a.out)
    elif a.action == "show":
        from .camera import Camera
        print(Camera.load(a.camera).summary())
    elif a.action == "target":
        from .calib.targets import render_board
        spec = _spec(a)
        img = render_board(spec, a.px_per_square)
        cv2.imwrite(a.out, img)
        dpi_note = (f"print so that one square measures exactly {spec.square_mm} mm, then MEASURE it: "
                    "the measured value is what goes in --square-mm")
        print(f"saved {a.out} ({img.shape[1]}x{img.shape[0]} px). {dpi_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
