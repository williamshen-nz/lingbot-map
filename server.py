"""LingBot-Map reconstruction server.

Streaming 3D reconstruction as a service: POST a zip of images, get back an npz
of camera poses, dense depth and intrinsics. The service is capture-device
agnostic — it takes images and nothing else.

Running
-------
    uv sync --all-extras
    uv run uvicorn server:app --host 0.0.0.0 --port 5464

    GET  /health       -> {"ok": bool, "weights": str}
    POST /reconstruct  -> npz, see Output below

    curl --data-binary @scan.zip http://HOST:5464/reconstruct -o scan.npz

Pass --port explicitly: uvicorn defaults to 8000 otherwise, which collides with
half the dev servers in existence. 5464 is unassigned in /etc/services and sits
below the 32768-60999 ephemeral range, so it will not clash with an outbound
connection that happened to grab the same number. Use --host 0.0.0.0 to accept
connections from other machines; there is no authentication, so keep it on a
trusted network.

Input
-----
The raw request body is a zip archive of images. That is the entire interface:
no tuning parameters, query strings or headers to set.

  * JPEG and PNG only. Other entries, dotfiles, and the ``__MACOSX/._*``
    resource forks macOS adds to zips are skipped.
  * At least 2 images; the frame count is otherwise unbounded.
  * Every image must share one aspect ratio, else 400. (Mixed sizes would be
    silently white-padded to a common shape, which is worse than failing.)
  * Frames are ordered by **filename**, ascending — the same rule demo.py uses.
    Name them so lexicographic order matches capture order: zero-pad indices,
    or use timestamps. Beware that "9.jpg" sorts after "10.jpg".
  * Archive paths are flattened to basenames, so those basenames must be unique
    across any subdirectories; a collision is a 400 rather than a lost frame.
  * EXIF orientation is honoured, so portrait captures are uprighted first.

Requests are served strictly one at a time. The model carries KV-cache state
through the streaming loop, so a second request waits rather than interleaving.

Output
------
An uncompressed .npz. For N frames at model resolution HxW — a 1024x768 input
yields 518x392, and roughly 120 MB for N=147:

    names             (N,)      <U    source filename, in array order
    c2w               (N,4,4)   f64   world-from-camera, OpenCV axes
    K                 (N,3,3)   f64   intrinsics in model pixel space
    K_upright         (N,3,3)   f64   the same camera at full resolution
    depth             (N,H,W)   f16   camera-z
    conf              (N,H,W)   f16   depth confidence, >= 1
    orig_wh           (N,2)     i32   image size as stored on disk
    upright_wh        (N,2)     i32   size after EXIF rotation
    resize_wh         (N,2)     i32   size after resize, before crop
    crop_xy           (N,2)     i32   top-left of the crop within resize_wh
    exif_orientation  (N,)      i32   EXIF tag 274 (1 = none)
    metadata          ()        <U    JSON: conventions and settings used

Conventions and caveats
-----------------------
* OpenCV camera axes — x right, y down, z forward. ``c2w`` maps camera
  coordinates into the world.
* **Scale is arbitrary and per-scan.** Poses and depth share one unknown
  factor, fixed by the first ``num_scale_frames``; it is not comparable across
  scans. Recover metres by fitting ``c2w[:, :3, 3]`` against a metrically
  scaled trajectory from the capture device (Umeyama over the shared frames).
* The world frame is **not gravity-aligned** and its origin is not meaningful.
* ``K`` is predicted per frame by the model's field-of-view head, not measured.
  It varies a few percent frame to frame even for a fixed lens. Prefer the
  capture device's intrinsics when you have them; ``K`` is here so the returned
  depth is self-consistently unprojectable.

Consuming
---------
Unproject one frame into world points::

    d = np.load("scan.npz", allow_pickle=True)
    metadata = json.loads(str(d["metadata"].item()))

    i = 0
    z = d["depth"][i].astype(np.float32)
    K, c2w = d["K"][i], d["c2w"][i]
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    cam = np.stack([(xx - K[0, 2]) * z / K[0, 0],
                    (yy - K[1, 2]) * z / K[1, 1], z], -1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    world = world[d["conf"][i] > 1.5]        # 1.5 is demo.py's default cutoff

Map a model pixel back to the full-resolution image — to colour a point, or to
place a detection made on the original photo::

    u_full = (u_model + crop_xy[i, 0]) * upright_wh[i, 0] / resize_wh[i, 0]
    v_full = (v_model + crop_xy[i, 1]) * upright_wh[i, 1] / resize_wh[i, 1]

The x and y factors differ — 1.97683 vs 1.95918 for 1024x768 — because width is
pinned to 518 while height is rounded to a multiple of the patch size. Never
collapse them into one scalar. ``K_upright`` already accounts for both, so if
you are working at full resolution, use it and skip this arithmetic entirely.
"""

import argparse
import asyncio
import io
import json
import os
import tempfile
import zipfile
from contextlib import asynccontextmanager

import anyio
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageOps

from demo import load_model
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

IMAGE_SIZE = 518
PATCH_SIZE = 14
NUM_SCALE_FRAMES = 8
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

WEIGHTS = os.environ.get(
    "LINGBOT_WEIGHTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/lingbot-map.pt"),
)

# Mirrors demo.py's argparse defaults; load_model() reads these off the namespace.
MODEL_ARGS = argparse.Namespace(
    mode="streaming",
    model_path=WEIGHTS,
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    enable_3d_rope=True,
    max_frame_num=1024,
    num_scale_frames=NUM_SCALE_FRAMES,
    kv_cache_sliding_window=64,
    camera_num_iterations=4,
    use_sdpa=False,
)

state = {}
# The model carries KV-cache state across the streaming loop, so two concurrent
# reconstructions would corrupt each other. One at a time.
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_ARGS, device)
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model.aggregator = model.aggregator.to(dtype=dtype)
    state.update(model=model, device=device, dtype=dtype)
    yield
    state.clear()


app = FastAPI(title="lingbot-map", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": "model" in state, "weights": WEIGHTS}


def extract_images(blob, dest):
    """Unzip to dest, returning image paths sorted by name."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        raise HTTPException(400, "body is not a zip archive")

    paths, seen = [], set()
    for info in zf.infolist():
        name = os.path.basename(info.filename)
        # macOS zips carry __MACOSX/._foo.jpg resource forks that are not images.
        if info.is_dir() or name.startswith(".") or "__MACOSX" in info.filename:
            continue
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            continue
        # Archive paths are flattened to basenames, so a/1.jpg and b/1.jpg would
        # silently overwrite each other and drop a frame. Refuse instead.
        if name in seen:
            raise HTTPException(
                400,
                f"duplicate image filename in archive: {name!r}. Frames are ordered "
                "by filename, so names must be unique across subdirectories.",
            )
        seen.add(name)
        out = os.path.join(dest, name)
        with open(out, "wb") as fh:
            fh.write(zf.read(info))
        paths.append(out)

    if len(paths) < 2:
        raise HTTPException(400, f"need at least 2 images, found {len(paths)}")
    return sorted(paths)


def frame_geometry(paths):
    """Recompute load_fn.py's resize/crop per frame so model pixels map home.

    load_and_preprocess_images does not report the transform it applied, so the
    arithmetic from load_fn.py:170-180 is repeated here. The caller asserts the
    result against the tensor the loader actually produced.
    """
    orig_wh, upright_wh, resize_wh, crop_xy, orientation = [], [], [], [], []
    for path in paths:
        img = Image.open(path)
        orig_wh.append(img.size)
        orientation.append(img.getexif().get(274, 1))

        img = ImageOps.exif_transpose(img)
        w, h = img.size
        upright_wh.append((w, h))

        new_w = IMAGE_SIZE
        new_h = round(h * (new_w / w) / PATCH_SIZE) * PATCH_SIZE
        resize_wh.append((new_w, new_h))
        crop_xy.append((0, (new_h - IMAGE_SIZE) // 2 if new_h > IMAGE_SIZE else 0))

    return {
        "orig_wh": np.array(orig_wh, np.int32),
        "upright_wh": np.array(upright_wh, np.int32),
        "resize_wh": np.array(resize_wh, np.int32),
        "crop_xy": np.array(crop_xy, np.int32),
        "exif_orientation": np.array(orientation, np.int32),
    }


def intrinsics_to_upright(K, geom):
    """Map intrinsics from model pixel space back to the upright full-res image."""
    sx = geom["upright_wh"][:, 0] / geom["resize_wh"][:, 0]
    sy = geom["upright_wh"][:, 1] / geom["resize_wh"][:, 1]
    out = K.copy()
    out[:, 0, 0] = K[:, 0, 0] * sx
    out[:, 1, 1] = K[:, 1, 1] * sy
    out[:, 0, 2] = (K[:, 0, 2] + geom["crop_xy"][:, 0]) * sx
    out[:, 1, 2] = (K[:, 1, 2] + geom["crop_xy"][:, 1]) * sy
    return out


def reconstruct(paths):
    model, device, dtype = state["model"], state["device"], state["dtype"]

    images = load_and_preprocess_images(paths, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
    n, _, h, w = images.shape

    # FlashInfer's KV cache manager is built lazily on first use and sized from
    # that frame shape (aggregator/stream.py:207). clean_kv_cache() resets its
    # contents but not its geometry, so a differently-shaped scan would assert
    # deep in append_frame. Drop it whenever the shape changes and let the next
    # forward rebuild it.
    if state.get("shape") != (h, w):
        model.aggregator.kv_cache_manager = None
        state["shape"] = (h, w)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Matches demo.py: keep the KV cache inside the 320-view video-RoPE range.
    keyframe_interval = max(1, (n + 319) // 320) if n > 320 else 1

    images = images.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        preds = model.inference_streaming(
            images, num_scale_frames=min(NUM_SCALE_FRAMES, n), keyframe_interval=keyframe_interval
        )

    pose_enc = preds["pose_enc"].float()
    # pose_encoding_to_extri_intri returns world_from_camera (c2w) despite its
    # docstring; demo.py inverts it and the viewer inverts it back. Verified by
    # trajectory alignment against ARKit poses. Flip here if that ever changes.
    c2w_34, K = pose_encoding_to_extri_intri(pose_enc, (h, w))
    c2w = torch.zeros((*c2w_34.shape[:-2], 4, 4), dtype=c2w_34.dtype)
    c2w[..., :3, :4] = c2w_34.cpu()
    c2w[..., 3, 3] = 1.0

    depth = preds["depth"].squeeze(-1)
    out = {
        "c2w": c2w[0].numpy().astype(np.float64),
        "K": K[0].cpu().numpy().astype(np.float64),
        "depth": depth[0].cpu().numpy().astype(np.float16),
        "conf": preds["depth_conf"][0].cpu().numpy().astype(np.float16),
    }
    return out, keyframe_interval, (h, w)


def run_job(blob):
    with tempfile.TemporaryDirectory() as tmp:
        paths = extract_images(blob, tmp)
        geom = frame_geometry(paths)

        shapes = {tuple(row) for row in geom["resize_wh"]}
        if len(shapes) > 1:
            raise HTTPException(
                400, f"all images must share an aspect ratio, got model shapes {shapes}"
            )

        out, keyframe_interval, (h, w) = reconstruct(paths)

    expected = (int(geom["resize_wh"][0][0]), min(int(geom["resize_wh"][0][1]), IMAGE_SIZE))
    if (w, h) != expected:
        raise HTTPException(500, f"geometry mismatch: model {(w, h)} vs computed {expected}")

    out.update(geom)
    out["names"] = np.array([os.path.basename(p) for p in paths])
    out["K_upright"] = intrinsics_to_upright(out["K"], geom)
    out["metadata"] = json.dumps(
        {
            "pose": "c2w, world_from_camera, OpenCV (x-right, y-down, z-forward)",
            "units": "arbitrary — poses and depth share one unknown scale factor",
            "world": "arbitrary orientation, not gravity-aligned",
            "depth": "camera-z, (N,H,W), same arbitrary scale as poses",
            "conf": "1 + exp(x); >= 1, and exactly 1.0 where float16 underflows",
            "K": "predicted, model pixel space (H,W below)",
            "K_upright": "K mapped to the EXIF-upright full-resolution image",
            "model_hw": [h, w],
            "n_frames": len(out["names"]),
            "keyframe_interval": keyframe_interval,
            "num_scale_frames": NUM_SCALE_FRAMES,
            "kv_cache_sliding_window": MODEL_ARGS.kv_cache_sliding_window,
            "camera_num_iterations": MODEL_ARGS.camera_num_iterations,
            "mode": "streaming",
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "weights": os.path.basename(WEIGHTS),
        }
    )

    buf = io.BytesIO()
    np.savez(buf, **out)
    return buf.getvalue()


@app.post("/reconstruct")
async def reconstruct_endpoint(request: Request):
    """Reconstruct a scan. Body is a zip of images; returns an npz.

    See the module docstring for the archive rules and the full output schema.
    Serialised against other requests — expect to queue behind an in-flight
    scan rather than run concurrently with it.
    """
    blob = await request.body()
    if not blob:
        raise HTTPException(400, "empty body; POST a zip of images")
    async with gpu_lock:
        return Response(
            content=await anyio.to_thread.run_sync(run_job, blob),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="reconstruction.npz"'},
        )
