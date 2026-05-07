"""
Shared utilities for the V-JEPA fMRI encoding pipeline.
"""

import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def to_torch_dtype(name: str):
    mapping = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[name]


def clip_id_from_name(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(s\d{2}e\d{2}[a-z])", stem.lower())
    if not m:
        raise ValueError(f"Could not parse clip id from {name}")
    return m.group(1)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Chunking / sampling
# ---------------------------------------------------------------------------

def get_chunk_intervals(video_duration, tr, chunk_length, seconds_before_chunk):
    if not (seconds_before_chunk < chunk_length):
        raise ValueError("seconds_before_chunk must be shorter than chunk_length")
    if not (seconds_before_chunk + tr <= chunk_length):
        raise ValueError("chunk_length must be >= seconds_before_chunk + tr")
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
    if num_items <= 0:
        return np.array([], dtype=np.int64)
    k = max(minimum_items, min(num_items, int(round(num_items * fraction))))
    if k == num_items:
        return np.arange(num_items, dtype=np.int64)
    idx = np.unique(np.round(np.linspace(0, num_items - 1, num=k)).astype(np.int64))
    if len(idx) < k:
        full = np.arange(num_items, dtype=np.int64)
        remaining = np.setdiff1d(full, idx, assume_unique=True)
        idx = np.sort(np.concatenate([idx, remaining[: k - len(idx)]]))
    return idx


# ---------------------------------------------------------------------------
# V-JEPA model helpers
# ---------------------------------------------------------------------------

def crop_vjepa_by_time(tensor, rel_start, rel_end, clip_seconds):
    if tensor.dim() < 3:
        return tensor.squeeze()
    tensor = tensor.reshape(-1, 16, 16, tensor.shape[-1])
    pooled = F.adaptive_avg_pool2d(
        tensor.permute(0, 3, 1, 2), (3, 3)
    ).permute(0, 2, 3, 1)
    T = tensor.size(0)
    tok0 = min(T - 1, int(round(rel_start / clip_seconds * T)))
    tok1 = min(T, max(tok0 + 1, int(round(rel_end / clip_seconds * T))))
    return pooled[tok0:tok1].mean(0).flatten()


def get_feature_module(model, feature_key):
    mapping = {
        "enc-10layer-fc2":        model.encoder.layer[10].mlp.fc2,
        "enc-18layer-norm2":      model.encoder.layer[18].norm2,
        "enc-20layer-fc2":        model.encoder.layer[20].mlp.fc2,
        "enc-last-ln":            model.encoder.layernorm,
        "predictor-5-layer-norm": model.predictor.layer[5].norm1,
        "pred-fc1":               model.predictor.layer[11].mlp.fc1,
        "final-features":         model.predictor.proj,
    }
    if feature_key not in mapping:
        raise ValueError(f"Unsupported feature_key_for_decoding: {feature_key}")
    return mapping[feature_key]


def needs_predictor_forward(feature_key):
    return feature_key in {"predictor-5-layer-norm", "pred-fc1", "final-features"}


def register_single_hook(module):
    store = []

    def _hook(_, __, out):
        store.append(out.detach())

    handle = module.register_forward_hook(_hook)
    return store, handle


def temporal_subsample(video_t: torch.Tensor, max_frames: int) -> torch.Tensor:
    if video_t.shape[0] <= max_frames:
        return video_t
    idx = torch.linspace(0, video_t.shape[0] - 1, steps=max_frames).round().long()
    return video_t.index_select(0, idx)


def build_vjepa_bundle(model_name: str, device: torch.device, config: dict) -> dict:
    model_dtype = to_torch_dtype(config["model_dtype"])
    processor = AutoVideoProcessor.from_pretrained(
        model_name, cache_dir="/scratch/arihantr/models/hub/", local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_name, dtype=model_dtype,
        cache_dir="/scratch/arihantr/models/hub/", local_files_only=True
    ).to(device).eval()

    # ── torch.compile ────────────────────────────────────────────────────
    # On L40S (Ada Lovelace) this eliminates Python/kernel-launch overhead
    # and typically gives 20-40% wall-clock speedup for repeated inference.
    # We compile only the encoder so that the hook on layernorm still fires.
    if config.get("use_compile", True):
        compile_mode = config.get("compile_mode", "reduce-overhead")
        try:
            model.encoder = torch.compile(
                model.encoder, mode=compile_mode, fullgraph=False
            )
            print(f"[build_vjepa_bundle] torch.compile applied  mode={compile_mode}")
        except Exception as e:
            print(f"[build_vjepa_bundle] torch.compile skipped: {e}")

    feature_key = config["feature_key_for_decoding"]
    feature_module = get_feature_module(model, feature_key)
    hook_store, hook_handle = register_single_hook(feature_module)

    return {
        "processor":    processor,
        "model":        model,
        "hook_store":   hook_store,
        "hook_handle":  hook_handle,
        "feature_key":  feature_key,
        "use_predictor": needs_predictor_forward(feature_key),
        "model_dtype":  model_dtype,
    }


# ---------------------------------------------------------------------------
# Batched GPU forward
# ---------------------------------------------------------------------------

def _forward_one_batch(bundle, x_hf):
    """Run a single batched forward; return the hooked activation [B, tokens, dim]."""
    bundle["hook_store"].clear()
    if bundle["use_predictor"]:
        bundle["model"](pixel_values_videos=x_hf, skip_predictor=False)
    else:
        bundle["model"].get_vision_features(x_hf)
    if not bundle["hook_store"]:
        raise RuntimeError(f"Hook for {bundle['feature_key']} did not fire")
    out = bundle["hook_store"][0]
    bundle["hook_store"].clear()
    return out


@torch.no_grad()
def batched_forward(clips_data: list, bundle: dict, device: torch.device, config: dict) -> list:
    """
    Process N pre-decoded clips in mini-batches of `batch_size`.

    clips_data entries:
        { video_t: Tensor[T,C,H,W], tr_idx, c_start, c_end, i_start, i_end }

    Returns a list of cpu feature tensors in the same order.
    """
    batch_size  = config.get("batch_size", 16)
    max_frames  = config["max_frames_per_clip"]
    model_dtype = bundle["model_dtype"]
    save_dtype  = to_torch_dtype(config["save_dtype"])

    results = [None] * len(clips_data)

    for b0 in tqdm(range(0, len(clips_data), batch_size), desc="  GPU batches", leave=False):
        batch = clips_data[b0: b0 + batch_size]

        # Subsample every clip to the same frame count, then stack → [B, T, C, H, W].
        # pin_memory() lets the H2D copy run via DMA without blocking the CPU.
        subsampled = [temporal_subsample(c["video_t"], max_frames) for c in batch]
        stacked    = torch.stack(subsampled).pin_memory()

        x_hf = (
            bundle["processor"](
                list(stacked),
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )["pixel_values_videos"]
            .to(device=device, dtype=model_dtype, non_blocking=True)
        )

        try:
            batch_out = _forward_one_batch(bundle, x_hf)  # [B, tokens, dim]
        except torch.OutOfMemoryError:
            # Halve the batch and retry recursively once.
            empty_cuda_cache()
            half = len(batch) // 2
            if half == 0:
                raise RuntimeError("OOM even on batch_size=1; reduce max_frames_per_clip.")
            print(f"  OOM on B={len(batch)}, retrying as {half}+{len(batch)-half}")
            left  = batched_forward(batch[:half],   bundle, device, config)
            right = batched_forward(batch[half:],   bundle, device, config)
            for i, feat in enumerate(left + right):
                results[b0 + i] = feat
            del stacked, x_hf
            empty_cuda_cache()
            continue

        for i, clip in enumerate(batch):
            clip_len = clip["c_end"] - clip["c_start"]
            feat = crop_vjepa_by_time(batch_out[i], clip["i_start"], clip["i_end"], clip_len)
            results[b0 + i] = feat.cpu().to(save_dtype)

        del stacked, x_hf, batch_out, subsampled
        empty_cuda_cache()

    return results


# ---------------------------------------------------------------------------
# Per-video extraction (called once per video file)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_video_features(video_path, relative_path, bundle, device, config):
    video_path = str(video_path)
    stem = Path(video_path).stem
    feature_root = Path(config["output_root"]) / "embeddings"
    feature_key  = bundle["feature_key"]

    tensor_path   = feature_root / feature_key / relative_path / f"{stem}.pt"
    tr_idx_path   = feature_root / "sampled_tr_indices" / relative_path / f"{stem}.npy"
    manifest_path = feature_root / "manifests" / relative_path / f"{stem}.json"
    success_file  = feature_root / "_SUCCESS" / relative_path / f"{stem}.json"

    for p in (tensor_path, tr_idx_path, manifest_path, success_file):
        p.parent.mkdir(parents=True, exist_ok=True)

    if success_file.exists() and not config.get("force_reextract", False):
        return

    vr = VideoReader(
        video_path,
        ctx=cpu(0),
        width=config["video_width"],
        height=config["video_height"],
        num_threads=config["decord_num_threads"],
    )
    fps          = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration     = total_frames / fps

    chunks, interests = get_chunk_intervals(
        duration, config["tr"], config["chunk_length"], config["seconds_before_chunk"]
    )
    sampled_tr_indices = uniform_sample_indices(
        len(chunks), config["sample_fraction"], config["minimum_sampled_trs_per_clip"]
    )

    # ── Phase 1: CPU decode – all selected TR clips for this video ────────
    # Decode everything first so the GPU forward loop is never stalled by I/O.
    # Phase 1: decode entire video into RAM once, then slice — no redundant disk reads
    all_frames = vr.get_batch(list(range(total_frames)))          # [T, H, W, C]
    all_frames = all_frames.permute(0, 3, 1, 2).float()          # [T, C, H, W]

    clips_data = []
    for tr_idx in tqdm(sampled_tr_indices, desc=f"Slice {stem}", leave=False):
        tr_idx  = int(tr_idx)
        c_start, c_end = chunks[tr_idx]
        i_start, i_end = interests[tr_idx]

        start_f = int(round(c_start * fps))
        end_f   = min(start_f + int(round((c_end - c_start) * fps)), total_frames)
        if start_f >= end_f:
            continue

        clip = all_frames[start_f:end_f]   # zero-copy slice, no disk I/O
        if clip.shape[0] == 0:
            continue

        clips_data.append({
            "tr_idx":  tr_idx,
            "video_t": clip,
            "c_start": float(c_start), "c_end": float(c_end),
            "i_start": float(i_start), "i_end": float(i_end),
        })

    del all_frames  # free after slicing

    if not clips_data:
        raise RuntimeError(f"No frames decoded for {video_path}")

    # ── Phase 2: batched GPU forward passes ──────────────────────────────
    features = batched_forward(clips_data, bundle, device, config)

    saved_features = [f for f in features if f is not None]
    if not saved_features:
        raise RuntimeError(f"No features produced for {video_path}")

    torch.save(torch.stack(saved_features, dim=0), tensor_path)
    np.save(tr_idx_path, sampled_tr_indices.astype(np.int64))

    rows = [
        {
            "tr_index":       c["tr_idx"],
            "tr_start_sec":   float(c["tr_idx"] * config["tr"]),
            "tr_end_sec":     float(c["tr_idx"] * config["tr"] + config["tr"]),
            "clip_start_sec": c["c_start"],
            "clip_end_sec":   c["c_end"],
            "feature_key":    feature_key,
        }
        for c in clips_data
    ]

    with open(manifest_path, "w") as f:
        json.dump({
            "video_path":              video_path,
            "relative_path":           relative_path,
            "stem":                    stem,
            "clip_id":                 clip_id_from_name(stem),
            "fps":                     fps,
            "total_frames":            int(total_frames),
            "duration_sec":            float(duration),
            "feature_key":             feature_key,
            "video_size":              [config["video_width"], config["video_height"]],
            "num_total_tr_windows":    int(len(chunks)),
            "num_sampled_tr_windows":  int(len(sampled_tr_indices)),
            "sample_fraction":         float(config["sample_fraction"]),
            "feature_tensor_path":     str(tensor_path),
            "sampled_tr_indices_path": str(tr_idx_path),
            "rows":                    rows,
        }, f, indent=2)

    with open(success_file, "w") as f:
        json.dump({"status": "ok", "video": video_path}, f)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def load_embedding_map(config):
    feature_root = Path(config["output_root"]) / "embeddings" / config["feature_key_for_decoding"]
    files = sorted(feature_root.rglob("*.pt"), key=lambda p: natural_key(p.name))
    if not files:
        raise FileNotFoundError(f"No embedding files found under {feature_root}")

    clip_map, manifest = {}, []
    for path in tqdm(files, desc="Loading embeddings"):
        stem     = path.stem
        clip_id  = clip_id_from_name(stem)
        x        = torch.load(path, map_location="cpu", weights_only=True)
        x        = x.float().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
        tr_path  = (
            Path(config["output_root"]) / "embeddings" / "sampled_tr_indices"
            / path.parent.relative_to(feature_root) / f"{stem}.npy"
        )
        clip_map[clip_id] = {"X": x, "tr_indices": np.load(tr_path).astype(np.int64)}
        manifest.append({"clip_id": clip_id, "file": str(path), "tr_idx_file": str(tr_path),
                          "num_sampled_trs": int(x.shape[0]), "dim": int(x.shape[1])})
    return clip_map, manifest


def find_h5_key_for_clip(h5_file, clip_id: str) -> str:
    matches = [k for k in h5_file.keys() if f"task-{clip_id}" in k.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one HDF5 key for {clip_id}, got {matches}")
    return matches[0]


def align_sparse_clip_xy(X_clip, tr_indices, Y_clip, lag_trs):
    X_clip     = np.asarray(X_clip,     dtype=np.float32)
    tr_indices = np.asarray(tr_indices, dtype=np.int64)
    Y_clip     = np.asarray(Y_clip,     dtype=np.float32)
    if Y_clip.ndim == 1:
        Y_clip = Y_clip[:, None]
    elif Y_clip.ndim > 2:
        Y_clip = Y_clip.reshape(Y_clip.shape[0], -1)
    target_idx = tr_indices + lag_trs
    valid = (target_idx >= 0) & (target_idx < Y_clip.shape[0])
    if valid.sum() <= 1:
        return None, None, None
    return X_clip[valid], Y_clip[target_idx[valid]], tr_indices[valid]


def parcelwise_pearson(y_true, y_pred):
    yt    = y_true - y_true.mean(axis=0, keepdims=True)
    yp    = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    valid = denom > 0
    corr  = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    corr[valid] = (yt[:, valid] * yp[:, valid]).sum(axis=0) / denom[valid]
    return corr


def split_train_test_clips(clip_ids, train_fraction, minimum_train_clips, minimum_test_clips):
    n = len(clip_ids)
    if n < (minimum_train_clips + minimum_test_clips):
        raise RuntimeError(f"Need >= {minimum_train_clips + minimum_test_clips} clips, got {n}")
    n_train = max(minimum_train_clips, min(int(np.floor(n * train_fraction)), n - minimum_test_clips))
    return clip_ids[:n_train], clip_ids[n_train:]


def load_subject_clip_data(subject, embedding_map, config):
    h5_path = (
        Path(config["fmri_root"]) / subject / "func"
        / config["friends_h5_pattern"].format(subject=subject)
    )
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")

    clip_data, missing = [], []
    with h5py.File(h5_path, "r") as f:
        all_h5_keys = list(f.keys())
        for clip_id in sorted(embedding_map.keys(), key=natural_key):
            try:
                key = find_h5_key_for_clip(f, clip_id)
            except Exception:
                missing.append(clip_id)
                continue
            Y = f[key][()]
            X, tr_indices = embedding_map[clip_id]["X"], embedding_map[clip_id]["tr_indices"]
            X_al, Y_al, kept = align_sparse_clip_xy(X, tr_indices, Y, config["lag_trs"])
            if X_al is None:
                missing.append(clip_id)
                continue
            clip_data.append({
                "clip_id": clip_id, "h5_key": key,
                "X": X_al, "Y": Y_al, "sampled_tr_indices_kept": kept,
                "num_embedding_trs_raw": int(len(X)),
                "num_fmri_trs_raw":      int(np.asarray(Y).shape[0]),
                "num_aligned_trs":       int(len(X_al)),
                "feature_dim":           int(X_al.shape[1]),
                "num_targets":           int(Y_al.shape[1]),
            })
        matched_keys      = {row["h5_key"] for row in clip_data}
        unmatched_h5_keys = [k for k in all_h5_keys if k not in matched_keys and "task-s01" in k.lower()]

    clip_data.sort(key=lambda row: natural_key(row["clip_id"]))
    return h5_path, clip_data, missing, unmatched_h5_keys


def concatenate_clip_rows(rows):
    X = np.concatenate([row["X"] for row in rows], axis=0)
    Y = np.concatenate([row["Y"] for row in rows], axis=0)
    return X.reshape(X.shape[0], -1), Y


def decode_one_subject(subject, embedding_map, config):
    h5_path, clip_data, missing_clip_ids, unmatched_h5_keys = load_subject_clip_data(
        subject, embedding_map, config
    )
    if not clip_data:
        raise RuntimeError(f"No matched season-1 clips for {subject}")

    clip_ids = [row["clip_id"] for row in clip_data]
    train_clip_ids, test_clip_ids = split_train_test_clips(
        clip_ids, config["train_fraction"], config["minimum_train_clips"], config["minimum_test_clips"]
    )
    train_rows = [row for row in clip_data if row["clip_id"] in set(train_clip_ids)]
    test_rows  = [row for row in clip_data if row["clip_id"] in set(test_clip_ids)]

    X_train, Y_train = concatenate_clip_rows(train_rows)
    X_test,  Y_test  = concatenate_clip_rows(test_rows)

    mdl = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=np.asarray(config["ridge_alphas"], dtype=np.float64)),
    )
    mdl.fit(X_train, Y_train)
    Y_pred = mdl.predict(X_test)
    corr   = parcelwise_pearson(Y_test, Y_pred)
    r2     = r2_score(Y_test, Y_pred, multioutput="raw_values")
    ridge  = mdl.named_steps["ridgecv"]

    per_clip_test_metrics, offset = [], 0
    for row in test_rows:
        n = row["Y"].shape[0]
        yt, yp = Y_test[offset:offset+n], Y_pred[offset:offset+n]
        per_clip_test_metrics.append({
            "clip_id": row["clip_id"], "h5_key": row["h5_key"], "num_test_trs": int(n),
            "mean_test_pearson": float(np.nanmean(parcelwise_pearson(yt, yp))),
            "mean_test_r2":      float(np.nanmean(r2_score(yt, yp, multioutput="raw_values"))),
        })
        offset += n

    result = {
        "subject":               subject,
        "h5_path":               str(h5_path),
        "feature_key":           config["feature_key_for_decoding"],
        "lag_trs":               int(config["lag_trs"]),
        "train_fraction":        float(config["train_fraction"]),
        "sample_fraction":       float(config["sample_fraction"]),
        "alpha":                 float(ridge.alpha_),
        "n_train_samples":       int(X_train.shape[0]),
        "n_test_samples":        int(X_test.shape[0]),
        "feature_dim":           int(X_train.shape[1]),
        "num_targets":           int(Y_train.shape[1]),
        "num_matched_clips":     int(len(clip_data)),
        "train_clip_ids":        train_clip_ids,
        "test_clip_ids":         test_clip_ids,
        "missing_embedding_or_h5_match": missing_clip_ids,
        "unmatched_h5_keys_for_s1":      unmatched_h5_keys,
        "mean_test_pearson":     float(np.nanmean(corr)),
        "median_test_pearson":   float(np.nanmedian(corr)),
        "mean_test_r2":          float(np.nanmean(r2)),
        "median_test_r2":        float(np.nanmedian(r2)),
        "per_clip_alignment": [
            {"clip_id": row["clip_id"], "h5_key": row["h5_key"],
             "num_embedding_trs_raw": row["num_embedding_trs_raw"],
             "num_fmri_trs_raw":      row["num_fmri_trs_raw"],
             "num_aligned_trs":       row["num_aligned_trs"],
             "sampled_tr_indices_kept_preview": row["sampled_tr_indices_kept"][:10].tolist()}
            for row in clip_data
        ],
        "per_clip_test_metrics": per_clip_test_metrics,
        "note": "Predicts fMRI from uniformly sampled V-JEPA embeddings (encoding model).",
    }
    return result, Y_pred, Y_test