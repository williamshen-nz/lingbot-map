# LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction
### [project website](https://technology.robbyant.com/lingbot-map) &emsp; &emsp; [arxiv paper](https://arxiv.org/abs/2604.14141) &emsp; &emsp; [model weights](https://huggingface.co/robbyant/lingbot-map)
<img src="assets/teaser.webp" width="500">

This repository is a fork of LingBot-Map. Please see [README_OLD.md](README_OLD.md) for the original README. This fork replaces the conda/pip install with [uv](https://docs.astral.sh/uv/) and adds a weight download script.

## Installation

**Prerequisites**:

- NVIDIA GPU (tested on RTX 3090)
- Linux (tested on Ubuntu)
- NVIDIA driver supporting CUDA 12.8 runtime
- `ffmpeg` (offline rendering only): `sudo apt install ffmpeg`

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. If you don't already have it installed, you can run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart or source your shell.

Once you have uv installed, follow the instructions below to set up LingBot-Map:

```bash
# Clone the repository
git clone https://github.com/williamshen-nz/lingbot-map.git
cd lingbot-map

# Install dependencies (this may take a few minutes, ~9 GB)
uv sync --all-extras

# Download pretrained weights into weights/ (~4.8 GB)
uv run scripts/download_weights.py
```

uv pins Python 3.12, `torch==2.8.0+cu128`, and `kaolin==0.18.0` from their respective indexes automatically — no manual `--index-url` juggling. Python 3.12 is the ceiling because Kaolin publishes no cp313 wheels for `torch-2.8.0_cu128`.

Next, run the demo script to ensure everything is set up correctly.

```bash
# Reconstruct one of the bundled example scenes
uv run python demo.py --model_path weights/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```

Go to the viser viewer at http://localhost:8080 in your browser to see the reconstruction stream in.

> **Tip:** to specify which GPU to use, prefix the command with `CUDA_VISIBLE_DEVICES=0` (or your desired GPU ID):
> ```bash
> CUDA_VISIBLE_DEVICES=0 uv run python demo.py --model_path weights/lingbot-map.pt --image_folder example/courthouse
> ```

Three example scenes ship with the repo and need no extra download: `example/courthouse`, `example/university`, and `example/loop`.

> **Note:** `--mask_sky` downloads `skyseg.onnx` to the HuggingFace cache (`~/.cache/huggingface`) on first use. If your home partition is short on space, export `HF_HOME` to somewhere roomier first.

## Inference Modes

`demo.py` runs in one of two modes, set with `--mode`:

| Mode | When to use | Command |
|---|---|---|
| `streaming` (default) | Frame-by-frame with a paged KV cache. The normal path. | `--mode streaming` |
| `windowed` | Sequences beyond ~3000 frames. Overlapping windows, KV cache reset per window. | `--mode windowed --window_size 128 --overlap_keyframes 8` |

Input is either a folder of images or a video:

```bash
--image_folder path/to/images/         # sorted image files
--video_path path/to/video.mp4 --fps 10  # decoded at the given FPS
```

`--stride N` subsamples input frames, `--first_k N` truncates to the first N.

### Keyframe interval

`--keyframe_interval` controls which frames stay resident in the KV cache. Non-keyframes still produce full predictions — they just don't grow the cache. The model was trained with video RoPE over 320 views, so quality degrades once the cache exceeds that.

- **Streaming**: auto-selected if unset — `1` when the sequence is ≤ 320 frames, otherwise `ceil(num_frames / 320)`.
- **Windowed**: defaults to `1`. `--window_size` counts *keyframes*, not actual frames, so a value above 1 expands each window's real coverage to `num_scale_frames + (window_size - num_scale_frames) * keyframe_interval`.

Use `--overlap_keyframes` rather than `--overlap_size` whenever `keyframe_interval > 1`; it is converted internally to `max(num_scale_frames, overlap_keyframes * keyframe_interval)` actual frames.

### Speed and memory knobs

| Flag | Default | Effect |
|---|---|---|
| `--compile` | **off** | `torch.compile` on hot modules with a CUDA-graph warmup. Off by default — pass it explicitly for the accelerated path. |
| `--use_sdpa` | off | Falls back to PyTorch SDPA attention. FlashInfer paged KV is the default and is roughly 2× faster. |
| `--camera_num_iterations` | `4` | Camera-head refinement passes per frame. `1` is faster and shrinks the camera KV cache 4×, at some pose accuracy. |
| `--num_scale_frames` | `8` | Bidirectional scale frames at startup. `2` reduces the initial activation peak. |
| `--offload_to_cpu` | **off** | Moves per-frame predictions to CPU to cut GPU peak memory, at a throughput cost. Enable with `--offload_to_cpu`, disable with `--no-offload_to_cpu`. (Upstream's help text claims this is on by default — it is not.) |
| `--kv_cache_sliding_window` | `64` | Pose-reference window size. |

Precision is chosen automatically: bf16 on compute capability ≥ 8.0, otherwise fp16.

### Measured throughput

Benchmark the model on your own hardware with the bundled profiler (it builds random weights, so no checkpoint is needed):

```bash
uv run python gct_profile.py --backend flashinfer --dtype bf16 --compile
```

On an **RTX 3090** at 378×504, 500 frames, `sliding_window=64`, `keyframe_interval=1`, FlashInfer + bf16 + compile: **4.48 FPS** (223 ms/frame), flat across the sequence.

End-to-end `demo.py` streaming runs on the Feijoa scene (147 frames, 518×392, `keyframe_interval=1`):

| GPU | `--compile` | no compile | compile gain |
|---|---|---|---|
| RTX 3090 | 27.5 s (5.3 FPS) | 30.6 s (4.8 FPS) | 1.11× |
| RTX 4090 | 16.6 s (8.9 FPS) | 19.0 s (7.7 FPS) | 1.14× |

The paper reports ~20 FPS at 518×378 under the same configuration but **never states which GPU** it was measured on, so treat that figure as hardware-dependent. Throughput here is a constant per-frame cost, not sequence-length degradation — window and keyframe settings will not change it.

`--fa3` selects FlashInfer's FA3 kernel, which is **Hopper (SM90) only**. It fails to compile on Ampere and Blackwell consumer cards.

## Offline Rendering Pipeline

The batch renderer (`demo_render/batch_demo.py`) produces point-cloud flythrough MP4s for sequences too long for the interactive viewer. It needs two CUDA extensions built in place first:

```bash
cd demo_render/render_cuda_ext
uv run --project ../.. python setup.py build_ext --inplace
cd ../..
```

This builds `voxel_morton_ext` and `frustum_cull_ext` against your local CUDA toolkit. A minor-version mismatch between your local `nvcc` and the CUDA 12.8 that torch was built with is expected and harmless.

Then render:

```bash
uv run python demo_render/batch_demo.py \
    --video_path /path/to/video.mp4 \
    --output_folder /path/to/output/ \
    --model_path weights/lingbot-map.pt \
    --config demo_render/config/indoor.yaml \
    --mode windowed --window_size 128 \
    --keyframe_interval 10 --overlap_keyframes 8 \
    --camera_vis default --keyframes_only_points
```

See [README_OLD.md](README_OLD.md) for the full flag-by-flag rationale, camera path YAML reference, and worked examples.

## Downloading Other Checkpoints

`scripts/download_weights.py` fetches `lingbot-map.pt` (balanced, used by the paper and benchmark) and `skyseg_batch.onnx` (batched sky segmentation for the renderer). To grab the long-sequence or stage-1 checkpoints as well, add them to `FILES` in that script, or pull one directly:

```bash
uv run --with huggingface_hub python -c \
  "from huggingface_hub import hf_hub_download; \
   hf_hub_download('robbyant/lingbot-map', 'lingbot-map-long.pt', local_dir='weights')"
```

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```
