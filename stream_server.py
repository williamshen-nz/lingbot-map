"""LingBot-Map streaming session server.

A WebSocket front end for live capture: open a connection, push frames as they
arrive, get a pose and a depth map back for each one. Where server.py takes a
whole scan at once, this exposes the model's native causal loop incrementally,
so a client can display results while still scanning.

    uv sync --all-extras
    uv run python stream_server.py           # 0.0.0.0:5465

Host, port and checkpoint come from LINGBOT_HOST, LINGBOT_STREAM_PORT and
LINGBOT_WEIGHTS. This loads its own copy of the model (~4.4 GB), so running it
alongside server.py costs twice the GPU memory.

Protocol
--------
One session per connection. A second connection while one is active is closed
with code 1013; there is a single KV cache, so sessions cannot overlap.

    C->S  {"type": "start", "budget": 300}
    S->C  {"type": "ready", "keyframe_interval": 1}

Then, repeatedly — a JSON header naming the frame, followed by its JPEG bytes:

    C->S  {"type": "frame", "frame_id": 8412}
    C->S  <binary JPEG>

The first NUM_SCALE_FRAMES frames are buffered rather than reconstructed; the
model fixes scale over them bidirectionally before the causal loop can start.
Each one is acknowledged so the client can keep a uniform send-one-wait-one
loop:

    S->C  {"type": "buffered", "frame_id": 8412}

Once the buffer fills, those frames are reconstructed together and emitted in
order, and every later frame produces exactly one result. Each result is a JSON
header followed by a binary payload:

    S->C  {"type": "result", "frame_id": 8412, "index": 0, "hw": [392, 518],
           "c2w": [[...]], "K": [[...]], "K_upright": [[...]],
           "keyframe": true, "ms": 129.4}
    S->C  <binary: depth then conf, both float16, hw[0]*hw[1] each>

    C->S  {"type": "end"}
    S->C  {"type": "done", "frames": 147}

``frame_id`` is echoed rather than inferred. Clients are expected to drop frames
upstream — a live capture pipeline that keeps only the newest frame will send a
sparse subsequence — so position in the stream says nothing about which capture
a result belongs to. ``index`` is the model's own sequence position.

Conventions match server.py exactly: ``c2w`` is world-from-camera on OpenCV
axes, depth is camera-z, and both share one arbitrary per-session scale factor
in a world frame that is not gravity-aligned. Anchor them client-side against
whatever metric pose your capture device provides.

Constraints
-----------
* Every frame in a session must share one aspect ratio; the KV cache is sized
  from the first and cannot be resized mid-session.
* ``keyframe_interval`` is fixed at ``start`` from ``budget`` (the expected
  frame count), because the model cannot change it partway through. Omitting
  budget assumes 1. Sessions longer than 320 frames at interval 1 run past the
  range the video RoPE was trained on.
* Poses are never revised. The model is causal with no loop closure, so a
  result is final when emitted and drift accumulates uncorrected.
"""

import asyncio
import json
import os
import tempfile
import time

import anyio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from demo import load_model
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from server import (
    HOST,
    IMAGE_SIZE,
    MODEL_ARGS,
    NUM_SCALE_FRAMES,
    PATCH_SIZE,
    WEIGHTS,
    geometry_for,
    intrinsics_to_upright,
)

PORT = int(os.environ.get("LINGBOT_STREAM_PORT", "5465"))

state = {}
session_lock = asyncio.Lock()


class SessionError(Exception):
    """Client-side protocol or input error; closes the session with a reason."""


class Session:
    """Drives the model's causal loop one frame at a time.

    Mirrors gct_stream.py:inference_streaming — buffer NUM_SCALE_FRAMES frames,
    run them as one bidirectional block, then step frame by frame — but split so
    each step can be awaited independently.
    """

    def __init__(self, model, device, dtype, tmpdir, budget=None):
        self.model, self.device, self.dtype = model, device, dtype
        self.tmpdir = tmpdir
        self.budget = budget
        self.keyframe_interval = (budget + 319) // 320 if budget and budget > 320 else 1

        self.pending = []       # (frame_id, path, geom) buffered for the scale block
        self.n = 0              # frames reconstructed so far
        self.scale_frames = 0
        self.started = False
        self.hw = None
        self.first_shape = None

        model.clean_kv_cache()

    def _to_tensor(self, paths):
        images = load_and_preprocess_images(paths, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
        return images.unsqueeze(0).to(self.device)

    def _forward(self, images, num_frames):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            return self.model.forward(
                images,
                num_frame_for_scale=self.scale_frames,
                num_frame_per_block=num_frames,
                causal_inference=True,
            )

    def _emit(self, preds, frame_ids, geoms):
        """Split a prediction dict into one result per frame."""
        h, w = self.hw
        c2w_34, K = pose_encoding_to_extri_intri(preds["pose_enc"].float(), (h, w))
        c2w = np.tile(np.eye(4), (len(frame_ids), 1, 1))
        c2w[:, :3, :4] = c2w_34[0].cpu().numpy()
        K = K[0].cpu().numpy().astype(np.float64)
        depth = preds["depth"].squeeze(-1)[0].cpu().numpy().astype(np.float16)
        conf = preds["depth_conf"][0].cpu().numpy().astype(np.float16)

        out = []
        for j, fid in enumerate(frame_ids):
            g = geoms[j]
            stacked = {k: np.array([v]) for k, v in g.items()}
            index = self.n + j
            is_kf = self.keyframe_interval <= 1 or (
                index < self.scale_frames
                or (index - self.scale_frames) % self.keyframe_interval == 0
            )
            out.append((
                {
                    "type": "result",
                    "frame_id": fid,
                    "index": index,
                    "hw": [h, w],
                    "c2w": c2w[j].tolist(),
                    "K": K[j].tolist(),
                    "K_upright": intrinsics_to_upright(K[j:j + 1], stacked)[0].tolist(),
                    "keyframe": bool(is_kf),
                    **{k: list(v) for k, v in g.items() if k != "exif_orientation"},
                    "exif_orientation": g["exif_orientation"],
                },
                depth[j].tobytes() + conf[j].tobytes(),
            ))
        self.n += len(frame_ids)
        return out

    def push(self, frame_id, jpeg):
        """Accept one frame. Returns [] while buffering, else a list of results."""
        path = os.path.join(self.tmpdir, f"{self.n + len(self.pending):06d}.jpg")
        with open(path, "wb") as fh:
            fh.write(jpeg)

        try:
            geom = geometry_for(Image.open(path))
        except (OSError, ValueError) as exc:
            # PIL raises UnidentifiedImageError (an OSError) for junk bytes.
            raise SessionError(f"frame {frame_id} is not a readable image: {exc}")

        shape = tuple(geom["resize_wh"])
        if self.first_shape is None:
            self.first_shape = shape
            # The KV cache is sized from the first frame and cannot be resized
            # (aggregator/stream.py:207), so any manager left by a previous
            # session must be dropped before this one starts.
            self.model.aggregator.kv_cache_manager = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        elif shape != self.first_shape:
            raise SessionError(
                f"frame {frame_id} has model shape {shape}, session is {self.first_shape}; "
                "all frames in a session must share one aspect ratio"
            )

        self.pending.append((frame_id, path, geom))

        if not self.started:
            if len(self.pending) < NUM_SCALE_FRAMES:
                return []
            return self._start()

        fid, p, g = self.pending.pop()
        # Non-keyframes attend to the cache but must not be written into it,
        # exactly as gct_stream.py:452-464 does inside the batch loop.
        is_kf = self.keyframe_interval <= 1 or (self.n - self.scale_frames) % self.keyframe_interval == 0
        t0 = time.perf_counter()
        if not is_kf:
            self.model._set_skip_append(True)
        try:
            preds = self._forward(self._to_tensor([p]), 1)
        finally:
            if not is_kf:
                self.model._set_skip_append(False)
        return _stamp(self._emit(preds, [fid], [g]), t0)

    def _start(self):
        """Run the bidirectional scale block over everything buffered so far."""
        frame_ids = [f for f, _, _ in self.pending]
        paths = [p for _, p, _ in self.pending]
        geoms = [g for _, _, g in self.pending]
        self.pending = []

        images = self._to_tensor(paths)
        self.hw = (int(images.shape[-2]), int(images.shape[-1]))
        self.first_shape = tuple(geoms[0]["resize_wh"])
        self.scale_frames = len(paths)

        t0 = time.perf_counter()
        preds = self._forward(images, self.scale_frames)
        self.started = True
        return _stamp(self._emit(preds, frame_ids, geoms), t0)

    def flush(self):
        """Finish a session that ended before the scale block was reached."""
        if self.started or not self.pending:
            return []
        return self._start()

    def close(self):
        self.model.clean_kv_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _stamp(results, t0):
    ms = round((time.perf_counter() - t0) * 1000 / max(1, len(results)), 1)
    for header, _ in results:
        header["ms"] = ms
    return results


def _lifespan_factory():
    from contextlib import asynccontextmanager

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

    return lifespan


app = FastAPI(title="lingbot-map-stream", lifespan=_lifespan_factory())


@app.get("/health")
def health():
    return {"ok": "model" in state, "weights": WEIGHTS, "busy": session_lock.locked()}


async def _send(ws, results):
    for header, payload in results:
        await ws.send_json(header)
        await ws.send_bytes(payload)


@app.websocket("/stream")
async def stream(ws: WebSocket):
    if session_lock.locked():
        await ws.accept()
        await ws.close(code=1013, reason="a session is already running")
        return

    async with session_lock:
        await ws.accept()
        session = None
        try:
            start = await ws.receive_json()
            if start.get("type") != "start":
                raise SessionError("first message must be {'type': 'start'}")

            with tempfile.TemporaryDirectory() as tmp:
                session = Session(
                    state["model"], state["device"], state["dtype"],
                    tmp, budget=start.get("budget"),
                )
                await ws.send_json({
                    "type": "ready",
                    "keyframe_interval": session.keyframe_interval,
                    "scale_frames": NUM_SCALE_FRAMES,
                })

                frame_id = None
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    if (text := msg.get("text")) is not None:
                        header = json.loads(text)
                        if header.get("type") == "end":
                            await _send(ws, await anyio.to_thread.run_sync(session.flush))
                            break
                        if header.get("type") != "frame":
                            raise SessionError(f"unexpected message {header.get('type')!r}")
                        frame_id = header.get("frame_id")
                        if frame_id is None:
                            raise SessionError("frame header needs a frame_id")
                        continue

                    if frame_id is None:
                        raise SessionError("binary frame arrived before its header")
                    results = await anyio.to_thread.run_sync(session.push, frame_id, msg["bytes"])
                    if results:
                        await _send(ws, results)
                    else:
                        await ws.send_json({"type": "buffered", "frame_id": frame_id})
                    frame_id = None

                await ws.send_json({"type": "done", "frames": session.n})
        except WebSocketDisconnect:
            pass
        except SessionError as exc:
            await ws.close(code=1008, reason=str(exc)[:120])
        finally:
            if session is not None:
                session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
