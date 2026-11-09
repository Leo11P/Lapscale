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


def _add_seg_args(p: argparse.ArgumentParser, required: bool = True) -> None:
    p.add_argument("--seg-backend", choices=["sam3", "yolo", "masks", "surgenet"], required=required)
    p.add_argument("--prompt", nargs="+", default=["surgical instrument"], help="sam3 text prompt(s)")
    p.add_argument("--seg-confidence", type=float, default=None)
    p.add_argument("--sam3-checkpoint", default=None, help="local SAM 3 checkpoint (default: Hugging Face)")
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--surgenet-checkpoint", default=None, help="fine-tuned SurgeNetXL segmentation checkpoint")
    p.add_argument("--surgenet-repo", default=None, help="clone of github.com/TimJaspers0801/SurgeNet")
    p.add_argument("--masks-dir", default=None, help="masks backend: folder of precomputed masks")


def _add_depth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--depth-backend", choices=["moge", "da3", "folder"], default="moge")
    p.add_argument("--moge-version", choices=["v2", "v3"], default="v3")
    p.add_argument("--depth-checkpoint", default=None,
                   help="moge: Ruicheng/moge-3-vitl (v3) / Ruicheng/moge-2-vitl (v2); "
                        "da3: depth-anything/DA3MONO-LARGE")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--resolution-level", type=int, default=9)
    p.add_argument("--num-tokens", type=int, default=None)
    p.add_argument("--refine-steps", type=int, default=3, help="moge v3 only")
    p.add_argument("--process-res", type=int, default=504, help="da3 only")
    p.add_argument("--depth-dir", default=None, help="folder backend: precomputed maps")
    p.add_argument("--depth-kind", choices=["depth", "inverse_depth"], default="depth", help="folder backend")
    p.add_argument("--depth-unit-mm", type=float, default=None, help="folder backend: mm per unit if metric")


def _add_frames_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stride", type=int, default=None, help="every Nth frame")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None, help="evenly spread in [start, stop)")


def _indices(a, src):
    if a.max_frames:
        return src.sample_indices(n=a.max_frames, start=a.start, stop=a.stop)
    return src.sample_indices(stride=a.stride or 1, start=a.start, stop=a.stop)


def _seg_backend(a):
    from .tool.backends import make_seg_backend
    if a.seg_backend == "sam3":
        kw = dict(prompts=a.prompt, checkpoint=a.sam3_checkpoint, device=a.device)
    elif a.seg_backend == "yolo":
        if not a.yolo_weights:
            sys.exit("--yolo-weights is required for the yolo backend")
        kw = dict(weights=a.yolo_weights, device=a.device)
    elif a.seg_backend == "surgenet":
        if not a.surgenet_checkpoint:
            sys.exit("--surgenet-checkpoint is required (train one with `lapscale segtrain surgenet`)")
        kw = dict(checkpoint=a.surgenet_checkpoint, repo=a.surgenet_repo, device=a.device)
    else:
        if not a.masks_dir:
            sys.exit("--masks-dir is required for the masks backend")
        kw = dict(masks_dir=a.masks_dir)
    if getattr(a, "seg_confidence", None) is not None and a.seg_backend in ("sam3", "yolo"):
        kw["confidence"] = a.seg_confidence
    return make_seg_backend(a.seg_backend, **kw)


def _depth_backend(a):
    from .depth import make_backend
    if a.depth_backend == "moge":
        kw = dict(version=a.moge_version, checkpoint=a.depth_checkpoint, device=a.device, fp16=not a.no_fp16,
                  resolution_level=a.resolution_level, num_tokens=a.num_tokens, refine_steps=a.refine_steps)
    elif a.depth_backend == "da3":
        kw = dict(checkpoint=a.depth_checkpoint or "depth-anything/DA3MONO-LARGE", device=a.device,
                  process_res=a.process_res)
    else:
        if not a.depth_dir:
            sys.exit("--depth-dir is required for the folder backend")
        kw = dict(depth_dir=a.depth_dir, output_kind=a.depth_kind, native_unit_mm=a.depth_unit_mm)
    return make_backend(a.depth_backend, **kw), kw


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
    calib_actions = {"board": b, "import": i, "moge": m, "mask": k, "show": s, "target": t}

    # ------------------------------------------------------------------ segmentation benchmark
    g = sub.add_parser("segbench", help="benchmark instrument segmentation by shaft edge accuracy")
    gs = g.add_subparsers(dest="action", required=True)
    gp = gs.add_parser("sample", help="sample frames from videos into a benchmark folder")
    gp.add_argument("--video", nargs="+", required=True)
    gp.add_argument("--bench", required=True)
    gp.add_argument("--n-per-video", type=int, default=25)
    ga = gs.add_parser("annotate", help="click the two edges of each measurable shaft (GUI)")
    ga.add_argument("--bench", required=True)
    ga.add_argument("--display-scale", type=float, default=1.0)
    ga.add_argument("--diameter-mm", type=float, default=None, help="default diameter of the annotated shafts")
    ga.add_argument("--all", action="store_true", help="revisit frames already annotated")
    gr = gs.add_parser("run", help="evaluate one segmentation backend")
    gr.add_argument("--bench", required=True)
    gr.add_argument("--camera", required=True, help="camera YAML of the videos (edges are compared undistorted)")
    gr.add_argument("--name", default=None, help="results folder name (default: backend name)")
    gr.add_argument("--device", default="cuda")
    gr.add_argument("--no-overlays", action="store_true")
    _add_seg_args(gr)
    gm = gs.add_parser("annotate-masks", help="ground-truth instrument masks (GUI), optional proposals")
    gm.add_argument("--bench", required=True)
    gm.add_argument("--display-scale", type=float, default=1.0)
    gm.add_argument("--all", action="store_true")
    gm.add_argument("--device", default="cuda")
    _add_seg_args(gm, required=False)
    gy = gs.add_parser("export-yolo", help="YOLO-seg dataset from the ground-truth masks (split by video)")
    gy.add_argument("--bench", required=True)
    gy.add_argument("--out", required=True)
    gy.add_argument("--val-fraction", type=float, default=0.2)
    gc = gs.add_parser("compare", help="table of all evaluated backends")
    gc.add_argument("--bench", required=True)

    # ------------------------------------------------------------------ depth
    d = sub.add_parser("depth", help="stage 2: per-frame depth (MoGe, DA3, precomputed), cached in the run folder")
    d.add_argument("--video", required=True)
    d.add_argument("--camera", required=True, help="camera YAML, or 'moge' (no calibration: focal estimated)")
    d.add_argument("--run-dir", default=None, help="default: runs/<video stem>")
    d.add_argument("--tag", default=None, help="cache name (default: backend name); change it for a new config")
    d.add_argument("--device", default="cuda")
    _add_depth_args(d)
    _add_frames_args(d)

    # ------------------------------------------------------------------ tool
    t = sub.add_parser("tool", help="stage 3 over a video: segmentation, shaft edges, metric samples")
    t.add_argument("--video", required=True)
    t.add_argument("--camera", required=True)
    t.add_argument("--run-dir", default=None)
    t.add_argument("--tag", default=None)
    t.add_argument("--diameter-mm", type=float, required=True, help="shaft diameter (measure it)")
    t.add_argument("--mc", type=int, default=100, help="Monte Carlo draws for the uncertainty (0 = off)")
    t.add_argument("--bias-sd-px", type=float, default=1.0, help="assumed edge bias of the segmentation")
    t.add_argument("--device", default="cuda")
    _add_seg_args(t)
    _add_frames_args(t)

    # ------------------------------------------------------------------ scale
    sc = sub.add_parser("scale", help="stage 4: fit the depth to the shaft (both variants)")
    sc.add_argument("--run-dir", required=True)
    sc.add_argument("--depth-tag", required=True)
    sc.add_argument("--tool-tag", required=True)
    sc.add_argument("--max-rel-sd", type=float, default=5.0, help="gate on the shaft depth uncertainty (%%)")
    sc.add_argument("--smooth-window", type=int, default=9)

    # ------------------------------------------------------------------ run
    r = sub.add_parser("run", help="all stages on a video: offline (cached) or stream (real time)")
    r.add_argument("--video", required=True, help="video file (or folder of images)")
    r.add_argument("--camera", required=True)
    r.add_argument("--mode", choices=["offline", "stream"], default="offline")
    r.add_argument("--run-dir", default=None)
    r.add_argument("--diameter-mm", type=float, required=True)
    r.add_argument("--mc", type=int, default=100, help="offline: Monte Carlo draws (stream: off)")
    r.add_argument("--variant", choices=["scale_only", "affine"], default="scale_only", help="stream only")
    r.add_argument("--out-video", default=None, help="stream: write the overlay video here")
    r.add_argument("--show", action="store_true", help="stream: show a window")
    r.add_argument("--overlay", action="store_true", help="offline: also render the overlay video")
    r.add_argument("--device", default="cuda")
    _add_seg_args(r)
    _add_depth_args(r)
    _add_frames_args(r)

    # ------------------------------------------------------------------ viz
    v = sub.add_parser("viz", help="overlay video of an offline run")
    v.add_argument("--run-dir", required=True)
    v.add_argument("--video", required=True)
    v.add_argument("--camera", required=True)
    v.add_argument("--depth-tag", required=True)
    v.add_argument("--tool-tag", required=True)
    v.add_argument("--variant", choices=["scale_only", "affine"], default="scale_only")
    v.add_argument("--out", default=None)
    v.add_argument("--range-mm", default="20,200")

    # ------------------------------------------------------------------ later stages
    # ------------------------------------------------------------------ depth benchmark
    db = sub.add_parser("depthbench", help="benchmark depth networks against depth ground truth")
    dbs = db.add_subparsers(dest="action", required=True)
    ps = dbs.add_parser("prepare-stereo", help="rectified left images + SGBM depth ground truth from a stereo pair")
    ps.add_argument("--left", required=True)
    ps.add_argument("--right", required=True)
    ps.add_argument("--calib", required=True, help="OpenCV file with M_l, D_l, M_r, D_r, R, T")
    ps.add_argument("--bench", required=True)
    ps.add_argument("--n", type=int, default=50)
    ps.add_argument("--offset", type=int, default=0, help="right frame = left frame + offset")
    ps.add_argument("--crop", default=None, help="y0,y1,x0,x1 applied to both videos (e.g. 0,960,320,1600)")
    ps.add_argument("--calib-frame", choices=["full", "crop"], default="full")
    ps.add_argument("--z-range-mm", default="20,300")
    pf = dbs.add_parser("prepare-folder", help="any undistorted images + depth maps")
    pf.add_argument("--images", required=True)
    pf.add_argument("--gt", required=True)
    pf.add_argument("--camera", required=True)
    pf.add_argument("--bench", required=True)
    pf.add_argument("--gt-scale-mm", type=float, default=1.0, help="multiply the GT files by this to get mm")
    pr = dbs.add_parser("run", help="evaluate one depth backend")
    pr.add_argument("--bench", required=True)
    pr.add_argument("--name", default=None)
    pr.add_argument("--device", default="cuda")
    _add_depth_args(pr)
    dbs.add_parser("compare", help="table of all depth backends").add_argument("--bench", required=True)

    # ------------------------------------------------------------------ eval
    ev = sub.add_parser("eval", help="whole pipeline (instrument scale) against depth ground truth")
    ev.add_argument("--bench", required=True, help="depth benchmark folder (the run used <bench>/images)")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--depth-tag", required=True)
    ev.add_argument("--tool-tag", required=True)
    ev.add_argument("--variant", choices=["scale_only", "affine"], default="scale_only")

    # ------------------------------------------------------------------ segtrain
    st = sub.add_parser("segtrain", help="fine-tune instrument segmentation on ground-truth masks")
    sts = st.add_subparsers(dest="action", required=True)
    sn = sts.add_parser("surgenet", help="SurgeNetXL CaFormer-S18 + FPN")
    sn.add_argument("--bench", nargs="+", required=True, help="benchmark folder(s) with masks_gt")
    sn.add_argument("--out", required=True)
    sn.add_argument("--surgenet-repo", required=True)
    sn.add_argument("--pretrained", required=True, help="SurgeNetXL teacher checkpoint (.pth)")
    sn.add_argument("--epochs", type=int, default=30)
    sn.add_argument("--batch", type=int, default=8)
    sn.add_argument("--size", default="480x640", help="HxW of the network input (multiple of 32)")
    sn.add_argument("--lr", type=float, default=1e-4)
    sn.add_argument("--device", default="cuda")
    yo = sts.add_parser("yolo", help="Ultralytics YOLO-seg on a dataset from `segbench export-yolo`")
    yo.add_argument("--data", required=True, help="data.yaml")
    yo.add_argument("--model", default="yolo26n-seg.pt")
    yo.add_argument("--epochs", type=int, default=100)
    yo.add_argument("--imgsz", type=int, default=960)
    yo.add_argument("--batch", type=int, default=8)
    yo.add_argument("--device", default=None)

    # ------------------------------------------------------------------ check
    ck = sub.add_parser("check", help="load each backend with its weights and run it once on this machine")
    ck.add_argument("--image", default=None, help="test frame (default: synthetic)")
    ck.add_argument("--device", default="cuda")
    ck.add_argument("--moge", nargs="*", default=None, help="v2 / v3 / v3=<checkpoint>")
    ck.add_argument("--da3", nargs="*", default=None, help="Hugging Face ids or local paths")
    ck.add_argument("--sam3", nargs="?", const=True, default=None, help="optional local checkpoint")
    ck.add_argument("--yolo", default=None, help="weights")
    ck.add_argument("--surgenet-repo", default=None)
    ck.add_argument("--surgenet-pretrained", default=None, help="teacher checkpoint: tests the backbone loading")
    ck.add_argument("--surgenet-checkpoint", default=None, help="fine-tuned checkpoint: tests inference")
    p.calib_actions = calib_actions
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    cfg = load_config(pre.config)
    if cfg:                                   # config values become defaults; explicit flags still win
        for name, subp in parser.calib_actions.items():
            subp.set_defaults(**section(cfg, "calib", name))
    a = parser.parse_args(argv)

    if a.stage == "segbench":
        return _segbench(a, cfg)
    if a.stage in ("depth", "tool", "scale", "run", "viz", "depthbench", "eval", "segtrain", "check"):
        return {"depth": _depth, "tool": _tool, "scale": _scale, "run": _run, "viz": _viz, "depthbench": _depthbench,
                "eval": _eval, "segtrain": _segtrain, "check": _check}[a.stage](a)

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


def _run_dir(a):
    from pathlib import Path
    return a.run_dir or str(Path("runs") / Path(a.video).stem)


def _depth(a) -> int:
    from .camera import resolve_camera
    from .depth import run_depth
    from .io.video import FrameSource
    src = FrameSource(a.video)
    camera = resolve_camera(a.camera, src.size)
    backend, kw = _depth_backend(a)
    run_depth(a.video, camera, backend, _run_dir(a), a.tag, _indices(a, src), config=kw)
    return 0


def _tool(a) -> int:
    from .camera import Camera
    from .io.video import FrameSource
    from .tool.run import ToolConfig, run_tool
    src = FrameSource(a.video)
    run_tool(a.video, Camera.load(a.camera), _seg_backend(a), _run_dir(a),
             ToolConfig(diameter_mm=a.diameter_mm, mc_draws=a.mc, bias_sd_px=a.bias_sd_px), tag=a.tag,
             indices=_indices(a, src))
    return 0


def _scale(a) -> int:
    from .scale.fit import ScaleConfig
    from .scale.run import run_scale
    run_scale(a.run_dir, a.depth_tag, a.tool_tag,
              ScaleConfig(max_instrument_rel_sd_pct=a.max_rel_sd, smooth_window=a.smooth_window))
    return 0


def _run(a) -> int:
    from .camera import Camera
    from .io.video import FrameSource
    from .tool.run import ToolConfig
    src = FrameSource(a.video)
    camera = Camera.load(a.camera)
    seg = _seg_backend(a)
    depth, kw = _depth_backend(a)
    if a.mode == "offline":
        from .pipeline import run_offline
        res = run_offline(a.video, camera, seg, depth, _run_dir(a), _indices(a, src),
                          ToolConfig(diameter_mm=a.diameter_mm, mc_draws=a.mc), depth_config=kw)
        if a.overlay:
            from .viz import render_run
            render_run(_run_dir(a), a.video, camera, res["depth"]["tag"], res["tool"]["tag"])
    else:
        from pathlib import Path

        from .pipeline import run_stream
        Path(_run_dir(a)).mkdir(parents=True, exist_ok=True)
        run_stream(a.video, camera, seg, depth, a.variant, ToolConfig(diameter_mm=a.diameter_mm, mc_draws=0),
                   out_video=a.out_video, log_csv=Path(_run_dir(a)) / "stream_log.csv", show=a.show,
                   max_frames=a.max_frames)
    return 0


def _viz(a) -> int:
    from .camera import Camera
    from .viz import render_run
    lo, hi = (float(x) for x in a.range_mm.split(","))
    render_run(a.run_dir, a.video, Camera.load(a.camera), a.depth_tag, a.tool_tag, a.variant, a.out, (lo, hi))
    return 0


def _depthbench(a) -> int:
    from . import depthbench as dbm
    if a.action == "prepare-stereo":
        from .depthbench.stereo import StereoConfig
        zlo, zhi = (float(x) for x in a.z_range_mm.split(","))
        crop = tuple(int(x) for x in a.crop.split(",")) if a.crop else None
        dbm.prepare_stereo(a.left, a.right, a.calib, a.bench, a.n, a.offset, crop, a.calib_frame,
                           stereo_cfg=StereoConfig(z_min_mm=zlo, z_max_mm=zhi))
    elif a.action == "prepare-folder":
        dbm.prepare_folder(a.images, a.gt, a.camera, a.bench, a.gt_scale_mm)
    elif a.action == "run":
        backend, _ = _depth_backend(a)
        dbm.run_depthbench(a.bench, backend, a.name)
    else:
        dbm.compare_depth(a.bench)
    return 0


def _eval(a) -> int:
    from .eval import evaluate
    evaluate(a.bench, a.run_dir, a.depth_tag, a.tool_tag, a.variant)
    return 0


def _segtrain(a) -> int:
    if a.action == "surgenet":
        from .segtrain.surgenet import train_surgenet
        hh, ww = (int(x) for x in a.size.lower().split("x"))
        if hh % 32 or ww % 32:
            sys.exit("--size must be a multiple of 32 in both dimensions")
        train_surgenet(a.bench, a.out, a.surgenet_repo, a.pretrained, a.epochs, a.batch, (hh, ww), a.lr,
                       device=a.device)
    else:
        from .segtrain.yolo import train_yolo
        train_yolo(a.data, a.model, a.epochs, a.imgsz, a.batch, a.device)
    return 0


def _check(a) -> int:
    from .check import run_checks
    res = run_checks(a.image, a.device, a.moge, a.da3, a.sam3, a.yolo, a.surgenet_checkpoint, a.surgenet_repo,
                     a.surgenet_pretrained)
    return 0 if all(ok for _, ok, _ in res) else 1


def _segbench(a, cfg: dict) -> int:
    from . import segbench as sb
    if a.action == "sample":
        sb.sample_frames(a.video, a.bench, a.n_per_video)
    elif a.action == "annotate":
        from .segbench.annotate import annotate
        annotate(a.bench, a.display_scale, a.diameter_mm, only_todo=not a.all)
    elif a.action == "run":
        from .camera import Camera
        from .tool.edges import EdgeConfig
        edge_cfg = EdgeConfig(**section(cfg, "tool", "edges"))       # --config tool.edges, else defaults
        sb.run_benchmark(a.bench, _seg_backend(a), Camera.load(a.camera), a.name,
                         edge_cfg=edge_cfg, overlays=not a.no_overlays)
    elif a.action == "annotate-masks":
        from .segbench.masks import annotate_masks
        annotate_masks(a.bench, _seg_backend(a) if a.seg_backend else None, a.display_scale, only_todo=not a.all)
    elif a.action == "export-yolo":
        from .segbench.masks import export_yolo
        export_yolo(a.bench, a.out, a.val_fraction)
    elif a.action == "compare":
        sb.compare(a.bench)
    return 0


if __name__ == "__main__":
    sys.exit(main())
