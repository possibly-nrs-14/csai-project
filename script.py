import os
os.environ.setdefault("HF_HOME", "/scratch/saigum/models")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
import traceback
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoVideoProcessor, AutoModel
import decord
from decord import VideoReader, cpu
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

decord.bridge.set_bridge("torch")

CONFIG = {
    # Dataset roots
    "data_root": "/scratch/arihantr/CSAI/algonauts_2025.competitors",
    "video_input_folder": "/scratch/arihantr/CSAI/algonauts_2025.competitors/stimuli/movies/friends/s1",
    "fmri_root": "/scratch/arihantr/CSAI/algonauts_2025.competitors/fmri",
    "output_root": "/scratch/arihantr/CSAI/algonauts_outputs/s1_vjepa_decode_uniform10pct_oomfixed",

    # Embedding extraction
    "vjepa_model_name": "facebook/vjepa2-vitl-fpc64-256",
    "feature_key_for_decoding": "enc-last-ln",
    "tr": 1.49,
    "chunk_length": 8.0,
    "seconds_before_chunk": 6.0,

    # Speed / sampling
    "sample_fraction": 0.10,
    "minimum_sampled_trs_per_clip": 8,
    "batch_size": 8,
    "parallelize_across_gpus": True,
    "gpu_ids": [0, 1, 2],
    "decord_num_threads": 12,
    "video_width": 224,
    "video_height": 224,
    "model_dtype": "bfloat16",
    "save_dtype": "float16",
    "allow_tf32": False,

    # OOM control
    "max_frames_per_clip": 32,
    "oom_frame_retry_schedule": [32, 24, 16, 12, 8],

    # Clip-matched prediction of fMRI from embeddings
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-05"],
    "friends_h5_pattern": "{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5",
    "lag_trs": 3,
    "train_fraction": 0.8,
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],
    "minimum_train_clips": 4,
    "minimum_test_clips": 1,
    "force_reextract": False,
    "force_redecode": False,
}


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def to_torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[name]


def clip_id_from_name(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(s\d{2}e\d{2}[a-z])", stem.lower())
    if not m:
        raise ValueError(f"Could not parse clip id from {name}")
    return m.group(1)


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
    k = int(round(num_items * fraction))
    k = max(minimum_items, k)
    k = min(num_items, k)
    if k == num_items:
        return np.arange(num_items, dtype=np.int64)
    idx = np.linspace(0, num_items - 1, num=k)
    idx = np.unique(np.round(idx).astype(np.int64))
    if len(idx) < k:
        full = np.arange(num_items, dtype=np.int64)
        remaining = np.setdiff1d(full, idx, assume_unique=True)
        need = k - len(idx)
        idx = np.sort(np.concatenate([idx, remaining[:need]]))
    return idx


def crop_vjepa_by_time(tensor, rel_start, rel_end, clip_seconds):
    if tensor.dim() < 3:
        return tensor.squeeze()

    tensor = tensor.reshape(-1, 16, 16, tensor.shape[-1])
    pooled = F.adaptive_avg_pool2d(
        tensor.permute(0, 3, 1, 2),
        (3, 3),
    ).permute(0, 2, 3, 1)
    T = tensor.size(0)
    tok0 = min(T - 1, int(round(rel_start / clip_seconds * T)))
    tok1 = min(T, max(tok0 + 1, int(round(rel_end / clip_seconds * T))))
    return pooled[tok0:tok1].mean(0).flatten()


def get_feature_module(model, feature_key):
    mapping = {
        "enc-10layer-fc2": model.encoder.layer[10].mlp.fc2,
        "enc-18layer-norm2": model.encoder.layer[18].norm2,
        "enc-20layer-fc2": model.encoder.layer[20].mlp.fc2,
        "enc-last-ln": model.encoder.layernorm,
        "predictor-5-layer-norm": model.predictor.layer[5].norm1,
        "pred-fc1": model.predictor.layer[11].mlp.fc1,
        "final-features": model.predictor.proj,
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


def build_vjepa_bundle(model_name, device, config):
    model_dtype = to_torch_dtype(config["model_dtype"])
    processor = AutoVideoProcessor.from_pretrained(model_name, cache_dir="/scratch/arihantr/models/hub/", local_files_only=True)
    model = AutoModel.from_pretrained(model_name, dtype=model_dtype, cache_dir="/scratch/arihantr/models/hub/", local_files_only=True).to(device).eval()
    feature_key = config["feature_key_for_decoding"]
    feature_module = get_feature_module(model, feature_key)
    hook_store, hook_handle = register_single_hook(feature_module)
    return {
        "processor": processor,
        "model": model,
        "hook_store": hook_store,
        "hook_handle": hook_handle,
        "feature_key": feature_key,
        "use_predictor": needs_predictor_forward(feature_key),
        "model_dtype": model_dtype,
    }


def temporal_subsample(video_t: torch.Tensor, max_frames: int) -> torch.Tensor:
    if video_t.shape[0] <= max_frames:
        return video_t
    idx = torch.linspace(0, video_t.shape[0] - 1, steps=max_frames)
    idx = idx.round().long()
    return video_t.index_select(0, idx)


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def forward_with_frame_retry(video_t, bundle, device, clip_seconds, rel_start, rel_end, retry_schedule):
    last_error = None
    for max_frames in retry_schedule:
        try:
            clip_t = temporal_subsample(video_t, max_frames)
            bundle["hook_store"].clear()

            x_hf = bundle["processor"](
                [clip_t],
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )["pixel_values_videos"].to(device=device, dtype=bundle["model_dtype"], non_blocking=True)

            if bundle["use_predictor"]:
                _ = bundle["model"](pixel_values_videos=x_hf, skip_predictor=False)
            else:
                _ = bundle["model"].get_vision_features(x_hf)

            if len(bundle["hook_store"]) == 0:
                raise RuntimeError(f"Hook for {bundle['feature_key']} did not fire")

            batch_out = bundle["hook_store"][0]
            feat = crop_vjepa_by_time(batch_out[0], rel_start, rel_end, clip_seconds)

            del x_hf, batch_out, clip_t
            bundle["hook_store"].clear()
            empty_cuda_cache()
            return feat, max_frames

        except torch.OutOfMemoryError as e:
            last_error = e
            bundle["hook_store"].clear()
            empty_cuda_cache()
            continue

    raise RuntimeError(
        f"OOM persisted even after retries with max_frames schedule {retry_schedule}. "
        f"Last CUDA OOM: {last_error}"
    )


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

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tr_idx_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    success_file.parent.mkdir(parents=True, exist_ok=True)

    if success_file.exists() and not config.get("force_reextract", False):
        return

    vr = VideoReader(
        video_path,
        ctx=cpu(0),
        width=config["video_width"],
        height=config["video_height"],
        num_threads=config["decord_num_threads"],
    )
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    duration = total_frames / fps

    chunks, interests = get_chunk_intervals(
        duration,
        config["tr"],
        config["chunk_length"],
        config["seconds_before_chunk"],
    )

    sampled_tr_indices = uniform_sample_indices(
        len(chunks),
        config["sample_fraction"],
        config["minimum_sampled_trs_per_clip"],
    )

    selected = [(int(tr_idx), chunks[tr_idx], interests[tr_idx]) for tr_idx in sampled_tr_indices]

    saved_features = []
    rows = []

    retry_schedule = config["oom_frame_retry_schedule"]
    if retry_schedule[0] != config["max_frames_per_clip"]:
        retry_schedule = [config["max_frames_per_clip"]] + [x for x in retry_schedule if x != config["max_frames_per_clip"]]

    for tr_idx, (c_start, c_end), (i_start, i_end) in tqdm(
        selected,
        desc=f"TRs: {Path(video_path).name}",
        leave=False,
    ):
        start_f = int(round(c_start * fps))
        target_duration_f = int(round((c_end - c_start) * fps))
        end_f = min(start_f + target_duration_f, total_frames)
        if start_f >= end_f:
            continue

        frames_t = vr.get_batch(list(range(start_f, end_f)))
        if frames_t.shape[0] == 0:
            continue

        video_t = frames_t.permute(0, 3, 1, 2).float()
        feat, used_frames = forward_with_frame_retry(
            video_t=video_t,
            bundle=bundle,
            device=device,
            clip_seconds=(c_end - c_start),
            rel_start=i_start,
            rel_end=i_end,
            retry_schedule=retry_schedule,
        )

        saved_features.append(feat.cpu().to(to_torch_dtype(config["save_dtype"])))
        rows.append(
            {
                "tr_index": int(tr_idx),
                "tr_start_sec": float(tr_idx * config["tr"]),
                "tr_end_sec": float(tr_idx * config["tr"] + config["tr"]),
                "clip_start_sec": float(c_start),
                "clip_end_sec": float(c_end),
                "feature_key": feature_key,
                "used_max_frames": int(used_frames),
            }
        )

        del frames_t, video_t, feat

    if not saved_features:
        raise RuntimeError(f"No features extracted for {video_path}")

    tensor = torch.stack(saved_features, dim=0)
    torch.save(tensor, tensor_path)
    np.save(tr_idx_path, sampled_tr_indices.astype(np.int64))

    manifest = {
        "video_path": video_path,
        "relative_path": relative_path,
        "stem": stem,
        "clip_id": clip_id_from_name(stem),
        "fps": fps,
        "total_frames": int(total_frames),
        "duration_sec": float(duration),
        "feature_key": feature_key,
        "video_size": [config["video_width"], config["video_height"]],
        "num_total_tr_windows": int(len(chunks)),
        "num_sampled_tr_windows": int(len(sampled_tr_indices)),
        "sample_fraction": float(config["sample_fraction"]),
        "feature_tensor_path": str(tensor_path),
        "sampled_tr_indices_path": str(tr_idx_path),
        "rows": rows,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    with open(success_file, "w") as f:
        json.dump({"status": "ok", "video": video_path}, f)


def worker_process(gpu_id, video_list, config):
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
            if f.endswith((".mkv", ".mp4")):
                all_videos.append((root, f))
    all_videos.sort(key=lambda x: natural_key(x[1]))
    if not all_videos:
        raise FileNotFoundError(f"No videos found in {config['video_input_folder']}")

    gpus = config["gpu_ids"] if config.get("parallelize_across_gpus", False) else [0]
    shards = np.array_split(all_videos, len(gpus))

    print(f"Embedding extraction over {len(all_videos)} videos using GPUs {gpus}")
    processes = []
    for i, gpu_id in enumerate(gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, shards[i].tolist(), config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Embedding worker failed with exit code {p.exitcode}")


def load_embedding_map(config):
    feature_root = Path(config["output_root"]) / "embeddings" / config["feature_key_for_decoding"]
    files = sorted(feature_root.rglob("*.pt"), key=lambda p: natural_key(p.name))
    if not files:
        raise FileNotFoundError(f"No embedding files found under {feature_root}")

    clip_map = {}
    manifest = []
    for path in tqdm(files, desc="Loading embeddings"):
        stem = path.stem
        clip_id = clip_id_from_name(stem)

        x = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(x, torch.Tensor):
            x = x.float().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)

        tr_idx_path = Path(config["output_root"]) / "embeddings" / "sampled_tr_indices" / path.parent.relative_to(feature_root) / f"{stem}.npy"
        tr_indices = np.load(tr_idx_path).astype(np.int64)

        clip_map[clip_id] = {"X": x, "tr_indices": tr_indices}
        manifest.append(
            {
                "clip_id": clip_id,
                "file": str(path),
                "tr_idx_file": str(tr_idx_path),
                "num_sampled_trs": int(x.shape[0]),
                "dim": int(x.shape[1]),
            }
        )
    return clip_map, manifest


def find_h5_key_for_clip(h5_file, clip_id: str) -> str:
    matches = [k for k in h5_file.keys() if f"task-{clip_id}" in k.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one HDF5 key for {clip_id}, got {matches}")
    return matches[0]


def align_sparse_clip_xy(X_clip, tr_indices, Y_clip, lag_trs):
    X_clip = np.asarray(X_clip, dtype=np.float32)
    tr_indices = np.asarray(tr_indices, dtype=np.int64)
    Y_clip = np.asarray(Y_clip, dtype=np.float32)

    if Y_clip.ndim == 1:
        Y_clip = Y_clip[:, None]
    elif Y_clip.ndim > 2:
        Y_clip = Y_clip.reshape(Y_clip.shape[0], -1)

    target_idx = tr_indices + lag_trs
    valid = (target_idx >= 0) & (target_idx < Y_clip.shape[0])

    if valid.sum() <= 1:
        return None, None, None

    X_aligned = X_clip[valid]
    Y_aligned = Y_clip[target_idx[valid]]
    kept_tr_indices = tr_indices[valid]
    return X_aligned, Y_aligned, kept_tr_indices


def parcelwise_pearson(y_true, y_pred):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    valid = denom > 0
    corr = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    corr[valid] = (yt[:, valid] * yp[:, valid]).sum(axis=0) / denom[valid]
    return corr


def split_train_test_clips(clip_ids, train_fraction, minimum_train_clips, minimum_test_clips):
    n = len(clip_ids)
    if n < (minimum_train_clips + minimum_test_clips):
        raise RuntimeError(
            f"Need at least {minimum_train_clips + minimum_test_clips} matched clips, got {n}"
        )

    n_train = int(np.floor(n * train_fraction))
    n_train = max(minimum_train_clips, n_train)
    n_train = min(n_train, n - minimum_test_clips)
    train_ids = clip_ids[:n_train]
    test_ids = clip_ids[n_train:]
    return train_ids, test_ids


def load_subject_clip_data(subject, embedding_map, config):
    h5_path = Path(config["fmri_root"]) / subject / "func" / config["friends_h5_pattern"].format(subject=subject)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")

    clip_data = []
    missing = []
    unmatched_h5_keys = []

    with h5py.File(h5_path, "r") as f:
        all_h5_keys = list(f.keys())
        for clip_id in sorted(embedding_map.keys(), key=natural_key):
            try:
                key = find_h5_key_for_clip(f, clip_id)
            except Exception:
                missing.append(clip_id)
                continue

            Y = f[key][()]
            X = embedding_map[clip_id]["X"]
            tr_indices = embedding_map[clip_id]["tr_indices"]

            X_aligned, Y_aligned, kept_tr_indices = align_sparse_clip_xy(
                X,
                tr_indices,
                Y,
                config["lag_trs"],
            )
            if X_aligned is None:
                missing.append(clip_id)
                continue

            clip_data.append(
                {
                    "clip_id": clip_id,
                    "h5_key": key,
                    "X": X_aligned,
                    "Y": Y_aligned,
                    "sampled_tr_indices_kept": kept_tr_indices,
                    "num_embedding_trs_raw": int(len(X)),
                    "num_fmri_trs_raw": int(np.asarray(Y).shape[0]),
                    "num_aligned_trs": int(len(X_aligned)),
                    "feature_dim": int(X_aligned.shape[1]),
                    "num_targets": int(Y_aligned.shape[1]),
                }
            )

        matched_keys = {row["h5_key"] for row in clip_data}
        unmatched_h5_keys = [k for k in all_h5_keys if k not in matched_keys and "task-s01" in k.lower()]

    clip_data.sort(key=lambda row: natural_key(row["clip_id"]))
    return h5_path, clip_data, missing, unmatched_h5_keys


def concatenate_clip_rows(rows):
    X = np.concatenate([row["X"] for row in rows], axis=0)
    Y = np.concatenate([row["Y"] for row in rows], axis=0)
    X = X.reshape(X.shape[0], -1)
    return X, Y


def decode_one_subject(subject, embedding_map, config):
    h5_path, clip_data, missing_clip_ids, unmatched_h5_keys = load_subject_clip_data(subject, embedding_map, config)
    if not clip_data:
        raise RuntimeError(f"No matched season-1 clips for {subject}")

    clip_ids = [row["clip_id"] for row in clip_data]
    train_clip_ids, test_clip_ids = split_train_test_clips(
        clip_ids,
        config["train_fraction"],
        config["minimum_train_clips"],
        config["minimum_test_clips"],
    )

    train_set = set(train_clip_ids)
    test_set = set(test_clip_ids)
    train_rows = [row for row in clip_data if row["clip_id"] in train_set]
    test_rows = [row for row in clip_data if row["clip_id"] in test_set]

    X_train, Y_train = concatenate_clip_rows(train_rows)
    X_test, Y_test = concatenate_clip_rows(test_rows)

    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=np.asarray(config["ridge_alphas"], dtype=np.float64)),
    )
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    corr = parcelwise_pearson(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred, multioutput="raw_values")
    ridge = model.named_steps["ridgecv"]

    per_clip_test_metrics = []
    offset = 0
    for row in test_rows:
        n = row["Y"].shape[0]
        y_true_clip = Y_test[offset : offset + n]
        y_pred_clip = Y_pred[offset : offset + n]
        offset += n
        clip_corr = parcelwise_pearson(y_true_clip, y_pred_clip)
        clip_r2 = r2_score(y_true_clip, y_pred_clip, multioutput="raw_values")
        per_clip_test_metrics.append(
            {
                "clip_id": row["clip_id"],
                "h5_key": row["h5_key"],
                "num_test_trs": int(n),
                "mean_test_pearson": float(np.nanmean(clip_corr)),
                "mean_test_r2": float(np.nanmean(clip_r2)),
            }
        )

    result = {
        "subject": subject,
        "h5_path": str(h5_path),
        "feature_key": config["feature_key_for_decoding"],
        "lag_trs": int(config["lag_trs"]),
        "train_fraction": float(config["train_fraction"]),
        "sample_fraction": float(config["sample_fraction"]),
        "alpha": float(ridge.alpha_),
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "num_targets": int(Y_train.shape[1]),
        "num_matched_clips": int(len(clip_data)),
        "train_clip_ids": train_clip_ids,
        "test_clip_ids": test_clip_ids,
        "missing_embedding_or_h5_match": missing_clip_ids,
        "unmatched_h5_keys_for_s1": unmatched_h5_keys,
        "mean_test_pearson": float(np.nanmean(corr)),
        "median_test_pearson": float(np.nanmedian(corr)),
        "mean_test_r2": float(np.nanmean(r2)),
        "median_test_r2": float(np.nanmedian(r2)),
        "per_clip_alignment": [
            {
                "clip_id": row["clip_id"],
                "h5_key": row["h5_key"],
                "num_embedding_trs_raw": row["num_embedding_trs_raw"],
                "num_fmri_trs_raw": row["num_fmri_trs_raw"],
                "num_aligned_trs": row["num_aligned_trs"],
                "sampled_tr_indices_kept_preview": row["sampled_tr_indices_kept"][:10].tolist(),
            }
            for row in clip_data
        ],
        "per_clip_test_metrics": per_clip_test_metrics,
        "note": "This predicts fMRI from uniformly sampled season-1 V-JEPA embeddings. It is an encoding model.",
    }
    return result, Y_pred, Y_test


def run_decoding(config):
    out_dir = Path(config["output_root"]) / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_map, embedding_manifest = load_embedding_map(config)
    with open(out_dir / "season1_embedding_manifest.json", "w") as f:
        json.dump(
            {
                "feature_key": config["feature_key_for_decoding"],
                "num_clips": len(embedding_manifest),
                "clips": embedding_manifest,
                "note": "Season-1 embeddings kept per clip with sampled TR indices and matched to HDF5 datasets via task-s01eXX? clip ids.",
            },
            f,
            indent=2,
        )

    summary = []
    for subject in config["subjects"]:
        report_path = out_dir / f"{subject}_report.json"
        if report_path.exists() and not config.get("force_redecode", False):
            with open(report_path) as f:
                summary.append(json.load(f))
            continue

        result, y_pred, y_true = decode_one_subject(subject, embedding_map, config)
        np.save(out_dir / f"{subject}_y_pred.npy", y_pred)
        np.save(out_dir / f"{subject}_y_true.npy", y_true)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Clip-matched prediction summary:")
    for row in summary:
        print(
            f"  {row['subject']}: mean Pearson={row['mean_test_pearson']:.4f}, "
            f"mean R2={row['mean_test_r2']:.4f}, alpha={row['alpha']}, "
            f"matched_clips={row['num_matched_clips']}"
        )


def main():
    mp.set_start_method("spawn", force=True)
    Path(CONFIG["output_root"]).mkdir(parents=True, exist_ok=True)
    run_embedding_collection(CONFIG)
    run_decoding(CONFIG)


if __name__ == "__main__":
    main()