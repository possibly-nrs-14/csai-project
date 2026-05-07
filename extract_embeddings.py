"""
Stage 1 — GPU-intensive: extract V-JEPA embeddings for all video clips.

Usage:
    python extract_embeddings.py --config path/to/config.json
"""

import argparse
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pipeline_utils import (
    build_vjepa_bundle,
    extract_video_features,
    load_config,
    natural_key,
)


def worker_process(gpu_id: int, video_list: list, config: dict):
    """Runs in a separate spawned process; one process per GPU."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    bundle = build_vjepa_bundle(config["vjepa_model_name"], device, config)

    for root, file in tqdm(video_list, desc=f"GPU {gpu_id}", position=int(gpu_id)):
        video_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, config["video_input_folder"])
        try:
            extract_video_features(video_path, relative_path, bundle, device, config)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error on {file}: {e}")
            traceback.print_exc()
            raise


def run_embedding_collection(config: dict):
    # Collect all videos
    all_videos = []
    for root, _, files in os.walk(config["video_input_folder"]):
        for f in files:
            if f.endswith((".mkv", ".mp4")):
                all_videos.append((root, f))
    all_videos.sort(key=lambda x: natural_key(x[1]))

    if not all_videos:
        raise FileNotFoundError(f"No videos found in {config['video_input_folder']}")

    gpus = config["gpu_ids"] if config.get("parallelize_across_gpus", False) else [config["gpu_ids"][0]]
    shards = np.array_split(all_videos, len(gpus))

    print(f"Extracting embeddings for {len(all_videos)} videos across GPUs {gpus}")
    print(f"  sample_fraction = {config['sample_fraction']}")
    print(f"  output_root     = {config['output_root']}")

    processes = []
    for i, gpu_id in enumerate(gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, shards[i].tolist(), config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Embedding worker (GPU {gpus[processes.index(p)]}) exited with code {p.exitcode}")

    print("Embedding extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA embeddings (GPU stage)")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(config["output_root"]).mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)
    run_embedding_collection(config)


if __name__ == "__main__":
    main()