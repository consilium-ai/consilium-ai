#!/usr/bin/env python3
"""Download the Consilium model (2.2GB, one-time setup)."""
import os
import sys


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "model")

    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        print("Model already downloaded.")
        return

    print("Downloading Consilium model (Qwen3.5-4B + Claude Opus reasoning)...")
    print("Size: ~2.2 GB. This is a one-time download.\n")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit",
            local_dir=model_dir,
        )
        print(f"\nDone! Model saved to: {model_dir}")

    except ImportError:
        print("Installing huggingface-hub...")
        os.system(f"{sys.executable} -m pip install huggingface-hub")
        from huggingface_hub import snapshot_download

        snapshot_download(
            "Jackrong/MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-4bit",
            local_dir=model_dir,
        )
        print(f"\nDone! Model saved to: {model_dir}")


if __name__ == "__main__":
    main()
