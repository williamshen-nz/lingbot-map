# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.34"]
# ///
"""Download LingBot-Map weights into weights/ (gitignored)."""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "robbyant/lingbot-map"
FILES = ["lingbot-map.pt", "skyseg_batch.onnx"]
DEST = Path(__file__).resolve().parent.parent / "weights"

DEST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    print(f"{name} -> {hf_hub_download(REPO, name, local_dir=DEST)}")
