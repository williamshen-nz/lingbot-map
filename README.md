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
