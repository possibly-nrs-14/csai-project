import os
os.environ.setdefault("HF_HOME", "/scratch/saigum/models")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
import math
import traceback
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModel
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

CONFIG = {
    # Dataset roots
    "data_root": "/scratch/arihantr/CSAI/algonauts_2025.competitors",
    "video_input_folder": "/scratch/arihantr/CSAI/algonauts_2025.competitors/stimuli/movies/friends/s1_resized_224",
    "output_root": "/scratch/arihantr/CSAI/algonauts_outputs/s1_ijepa_10_480f",
    "cache_dir": "/scratch/arihantr/models/hub",

    # Embedding extraction (I-JEPA specific)
    "ijepa_model_name": "facebook/ijepa_vith14_22k", 
    "feature_key_for_decoding": "enc-last-ln",
    "tr": 1.49,
    "chunk_length": 8.0,
    "seconds_before_chunk": 6.0,

    # How many frames per clip to pass to I-JEPA for feature extraction
    "ijepa_frames_per_clip": 240,  

    # Speed / sampling
    "sample_fraction": 0.10,
    "minimum_sampled_trs_per_clip": 8,
    "batch_size": 8,
    "parallelize_across_gpus": True,
    "gpu_ids": [1],
    "decord_num_threads": 12,
    "video_width": 224,
    "video_height": 224,
    "model_dtype": "bfloat16",
    "save_dtype": "float16",
    "allow_tf32": False,

    # OOM control (Reduces frames passed per clip if OOM occurs)
    "oom_frame_retry_schedule": [16, 12, 8, 4, 1],
    "force_reextract": False,
}

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def to_torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]

def clip_id_from_name(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(s\d{2}e\d{2}[a-z])", stem.lower())
    if not m: raise ValueError(f"Could not parse clip id from {name}")
    return m.group(1)

def get_chunk_intervals(video_duration, tr, chunk_length, seconds_before_chunk):
    chunks, interests = [], []
    start_time = 0.0
    while start_time < video_duration:
        chunk_start = max(0.0, start_time - seconds_before_chunk)
        chunk_end = min(chunk_start + chunk_length, video_duration)
        rel_start = start_time - chunk_start
        rel_end = min(rel_start + tr, chunk_end - chunk_start)
        chunks.append((chunk_start, chunk_end))
        interests.append((rel_start, rel_end))
        start_time += tr
    return chunks, interests

def uniform_sample_indices(num_items: int, fraction: float, minimum_items: int):
    if num_items <= 0: return np.array([], dtype=np.int64)
    k = max(minimum_items, int(round(num_items * fraction)))
    k = min(num_items, k)
    if k == num_items: return np.arange(num_items, dtype=np.int64)
    
    idx = np.unique(np.round(np.linspace(0, num_items - 1, num=k)).astype(np.int64))
    if len(idx) < k:
        remaining = np.setdiff1d(np.arange(num_items), idx, assume_unique=True)
        idx = np.sort(np.concatenate([idx, remaining[:k - len(idx)]]))
    return idx

def crop_ijepa_by_time(tensor, rel_start, rel_end, clip_seconds):
    if tensor.dim() < 3:
        return tensor.squeeze()

    num_tokens = tensor.shape[1]
    spatial_tokens = num_tokens - 1 if int(math.sqrt(num_tokens - 1))**2 == (num_tokens - 1) else num_tokens
    
    if num_tokens > spatial_tokens:
        tensor = tensor[:, 1:, :]
        
    grid_size = int(math.sqrt(spatial_tokens))
    tensor = tensor.reshape(-1, grid_size, grid_size, tensor.shape[-1])
    
    pooled = F.adaptive_avg_pool2d(
        tensor.permute(0, 3, 1, 2),
        (3, 3),
    ).permute(0, 2, 3, 1)
    
    num_frames = tensor.size(0)
    tok0 = min(num_frames - 1, int(round(rel_start / clip_seconds * num_frames)))
    tok1 = min(num_frames, max(tok0 + 1, int(round(rel_end / clip_seconds * num_frames))))
    
    return pooled[tok0:tok1].mean(0).flatten()

def get_feature_module(model, feature_key):
    mapping = {
        "enc-10layer-fc2": model.encoder.layer[10].intermediate.dense,
        "enc-last-ln": model.layernorm, 
    }
    if feature_key not in mapping:
        print(f"Warning: {feature_key} not in default map. Attempting dynamic search.")
        return model.encoder.layer[-1].layernorm_before
    return mapping[feature_key]

def register_single_hook(module):
    store = []
    def _hook(_, __, out):
        store.append(out.detach() if isinstance(out, torch.Tensor) else out[0].detach())
    handle = module.register_forward_hook(_hook)
    return store, handle

def build_ijepa_bundle(model_name, device, config):
    model_dtype = to_torch_dtype(config["model_dtype"])
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=config["cache_dir"])
    model = AutoModel.from_pretrained(model_name, dtype=model_dtype, cache_dir=config["cache_dir"]).to(device).eval()
    
    feature_key = config["feature_key_for_decoding"]
    feature_module = get_feature_module(model, feature_key)
    hook_store, hook_handle = register_single_hook(feature_module)
    return {
        "processor": processor,
        "model": model,
        "hook_store": hook_store,
        "hook_handle": hook_handle,
        "feature_key": feature_key,
        "model_dtype": model_dtype,
    }

def temporal_subsample(video_t: torch.Tensor, max_frames: int) -> torch.Tensor:
    if video_t.shape[0] <= max_frames: return video_t
    idx = torch.linspace(0, video_t.shape[0] - 1, steps=max_frames).round().long()
    return video_t.index_select(0, idx)

def empty_cuda_cache():
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def forward_batch_with_frame_retry(batch_video_t, bundle, device, valid_items, retry_schedule):
    last_error = None
    for max_frames in retry_schedule:
        try:
            # --- INSIDE forward_batch_with_frame_retry ---

            batched_clips = [temporal_subsample(v, max_frames) for v in batch_video_t]
            
            # 1. Track the EXACT number of frames each clip has
            clip_lengths = [clip.shape[0] for clip in batched_clips]
            
            bundle["hook_store"].clear()

            flat_frames = []
            for clip in batched_clips:
                for f in range(clip.shape[0]):
                    flat_frames.append(clip[f])

            x_hf = bundle["processor"](
                images=flat_frames,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )["pixel_values"].to(device=device, dtype=bundle["model_dtype"], non_blocking=True)

            _ = bundle["model"](pixel_values=x_hf)

            if len(bundle["hook_store"]) == 0:
                raise RuntimeError(f"Hook for {bundle['feature_key']} did not fire")

            flat_out = bundle["hook_store"][0] 
            
            # 2. INSTEAD of .view(), split the flat tensor back into a list of tensors 
            # based on their original, actual lengths.
            batch_out = torch.split(flat_out, clip_lengths, dim=0)
            
            feats = []
            for i, (tr_idx, c_start, c_end, i_start, i_end) in enumerate(valid_items):
                # batch_out[i] is now safely sized (e.g., 240, or 202) for this specific clip
                feat = crop_ijepa_by_time(batch_out[i], i_start, i_end, clip_seconds=(c_end - c_start))
                feats.append(feat)

            del x_hf, flat_out, batch_out, batched_clips, flat_frames
            bundle["hook_store"].clear()
            empty_cuda_cache()
            return feats, max_frames

        except torch.OutOfMemoryError as e:
            last_error = e
            bundle["hook_store"].clear()
            empty_cuda_cache()
            continue

    raise RuntimeError(f"OOM persisted on batch even after retries. Last CUDA OOM: {last_error}")

@torch.no_grad()
def extract_video_features(video_path, relative_path, bundle, device, config):
    video_path = str(video_path)
    stem = Path(video_path).stem
    feature_root = Path(config["output_root"]) / "embeddings"
    feature_key = bundle["feature_key"]

    tensor_path = feature_root / feature_key / relative_path / f"{stem}.pt"
    tr_idx_path = feature_root / "sampled_tr_indices" / relative_path / f"{stem}.npy"
    manifest_path = feature_root / "manifests" / relative_path / f"{stem}.json"
    success_file = feature_root / "_SUCCESS" / relative_path / f"{stem}.json"

    for p in [tensor_path, tr_idx_path, manifest_path, success_file]:
        p.parent.mkdir(parents=True, exist_ok=True)

    if success_file.exists() and not config.get("force_reextract", False):
        return

    vr = VideoReader(
        video_path, ctx=cpu(0), 
        width=config["video_width"], height=config["video_height"], 
        num_threads=config["decord_num_threads"]
    )
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration = total_frames / fps

    chunks, interests = get_chunk_intervals(duration, config["tr"], config["chunk_length"], config["seconds_before_chunk"])
    sampled_tr_indices = uniform_sample_indices(len(chunks), config["sample_fraction"], config["minimum_sampled_trs_per_clip"])
    selected = [(int(tr_idx), chunks[tr_idx], interests[tr_idx]) for tr_idx in sampled_tr_indices]

    saved_features = []
    rows = []

    retry_schedule = config["oom_frame_retry_schedule"]
    if retry_schedule[0] != config["ijepa_frames_per_clip"]:
        retry_schedule = [config["ijepa_frames_per_clip"]] + [x for x in retry_schedule if x != config["ijepa_frames_per_clip"]]

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    for batch_selected in tqdm(list(chunker(selected, config["batch_size"])), desc=f"Batches: {Path(video_path).name}", leave=False):
        batch_video_t = []
        valid_items = []

        for tr_idx, (c_start, c_end), (i_start, i_end) in batch_selected:
            start_f = int(round(c_start * fps))
            target_duration_f = int(round((c_end - c_start) * fps))
            end_f = min(start_f + target_duration_f, total_frames)
            
            if start_f >= end_f: continue
            frames_t = vr.get_batch(list(range(start_f, end_f)))
            if frames_t.shape[0] == 0: continue

            video_t = frames_t.permute(0, 3, 1, 2).float()
            batch_video_t.append(video_t)
            valid_items.append((tr_idx, c_start, c_end, i_start, i_end))

        if not batch_video_t: continue

        batch_feats, used_frames = forward_batch_with_frame_retry(
            batch_video_t=batch_video_t, bundle=bundle, device=device,
            valid_items=valid_items, retry_schedule=retry_schedule,
        )

        for feat, (tr_idx, c_start, c_end, i_start, i_end) in zip(batch_feats, valid_items):
            saved_features.append(feat.cpu().to(to_torch_dtype(config["save_dtype"])))
            rows.append({
                "tr_index": int(tr_idx), "tr_start_sec": float(tr_idx * config["tr"]),
                "tr_end_sec": float(tr_idx * config["tr"] + config["tr"]),
                "clip_start_sec": float(c_start), "clip_end_sec": float(c_end),
                "feature_key": feature_key, "used_max_frames": int(used_frames),
            })

    tensor = torch.stack(saved_features, dim=0)
    torch.save(tensor, tensor_path)
    np.save(tr_idx_path, sampled_tr_indices.astype(np.int64))

    manifest = {
        "video_path": video_path, "relative_path": relative_path, "stem": stem,
        "clip_id": clip_id_from_name(stem), "fps": fps, "total_frames": int(total_frames),
        "duration_sec": float(duration), "feature_key": feature_key,
        "video_size": [config["video_width"], config["video_height"]],
        "num_total_tr_windows": int(len(chunks)), "num_sampled_tr_windows": int(len(sampled_tr_indices)),
        "sample_fraction": float(config["sample_fraction"]),
        "feature_tensor_path": str(tensor_path), "sampled_tr_indices_path": str(tr_idx_path), "rows": rows,
    }
    with open(manifest_path, "w") as f: json.dump(manifest, f, indent=2)
    with open(success_file, "w") as f: json.dump({"status": "ok", "video": video_path}, f)

def worker_process(gpu_id, video_list, config):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    bundle = build_ijepa_bundle(config["ijepa_model_name"], device, config)

    for root, file in tqdm(video_list, desc=f"GPU {gpu_id} videos", position=int(gpu_id)):
        video_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, config["video_input_folder"])
        try:
            extract_video_features(video_path, relative_path, bundle, device, config)
        except Exception as e:
            print(f"Error on GPU {gpu_id} processing {file}: {e}")
            traceback.print_exc()
            raise

def run_embedding_collection(config):
    all_videos = []
    for root, _, files in os.walk(config["video_input_folder"]):
        for f in files:
            if f.endswith((".mkv", ".mp4")): all_videos.append((root, f))
    all_videos.sort(key=lambda x: natural_key(x[1]))
    
    gpus = config["gpu_ids"] if config.get("parallelize_across_gpus", False) else [0]
    shards = np.array_split(all_videos, len(gpus))

    processes = []
    for i, gpu_id in enumerate(gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, shards[i].tolist(), config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0: raise RuntimeError(f"Embedding worker failed with exit code {p.exitcode}")

def main():
    mp.set_start_method("spawn", force=True)
    Path(CONFIG["output_root"]).mkdir(parents=True, exist_ok=True)
    run_embedding_collection(CONFIG)

if __name__ == "__main__":
    main()