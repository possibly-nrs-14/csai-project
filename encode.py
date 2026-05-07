import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# The Dual Ridge OOM Fix
# ---------------------------------------------------------------------------
# When D >> N (e.g., D=3.2M, N=1140), computing X^T @ X (the Primal solution) 
# causes catastrophic OOM. 
# We switch to Dual Ridge Regression (Kernel Ridge with a Linear Kernel).
# We solve A = (X @ X^T + alpha * I)^-1 @ Y.
# The N x N kernel fits easily in memory, and the entire train feature matrix
# (~14.6 GB for 1140 x 3.2M at fp32) fits comfortably in the L40S 48GB VRAM.
# ---------------------------------------------------------------------------

CONFIG = {
    # Dataset roots
    "fmri_root": "/scratch/arihantr/CSAI/algonauts_2025.competitors/fmri",
    "output_root": "/scratch/arihantr/CSAI/algonauts_outputs/s1_vjepa_decode_uniform8pct_oomfixed",

    # Feature target mapping
    "feature_key_for_decoding": "enc-last-ln",
    "sample_fraction": 0.08,

    # Clip-matched prediction of fMRI from embeddings
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-05"],
    "friends_h5_pattern": (
        "{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18"
        "_parcel-1000Par7Net_desc-s123456_bold.h5"
    ),
    "lag_trs": 3,
    "train_fraction": 0.8,
    "ridge_alpha": 1000.0,
    "minimum_train_clips": 4,
    "minimum_test_clips": 1,
    "force_redecode": False,

    # --- Loader ---
    "loader_workers": 8,
    "convert_pt_to_npy": True,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def clip_id_from_name(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(s\d{2}e\d{2}[a-z])", stem.lower())
    if not m:
        raise ValueError(f"Could not parse clip id from {name}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Embedding loading -- parallel + optional npy memory-map
# ---------------------------------------------------------------------------

_torch_load_lock = threading.Lock()


def _convert_pt_to_npy_if_needed(pt_path: Path) -> Path:
    npy_path = pt_path.with_suffix(".npy")
    if not npy_path.exists():
        with _torch_load_lock:
            x = torch.load(pt_path, map_location="cpu", weights_only=True)
        if isinstance(x, torch.Tensor):
            x = x.float().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        np.save(npy_path, x)
    return npy_path


def _load_one_clip(args):
    pt_path, feature_root, output_root, convert_to_npy = args
    stem    = pt_path.stem
    clip_id = clip_id_from_name(stem)

    if convert_to_npy:
        npy_path = _convert_pt_to_npy_if_needed(pt_path)
        x = np.load(npy_path, mmap_mode="r").astype(np.float32)
    else:
        with _torch_load_lock:
            raw = torch.load(pt_path, map_location="cpu", weights_only=True)
        x = raw.float().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw, dtype=np.float32)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(-1, 1)

    rel = pt_path.parent.relative_to(feature_root)
    tr_idx_path = (
        Path(output_root) / "embeddings" / "sampled_tr_indices" / rel / f"{stem}.npy"
    )
    tr_indices = np.load(tr_idx_path).astype(np.int64)
    return clip_id, pt_path, x, tr_indices


def load_embedding_map(config):
    feature_root = Path(config["output_root"]) / "embeddings" / config["feature_key_for_decoding"]
    files = sorted(feature_root.rglob("*.pt"), key=lambda p: natural_key(p.name))
    if not files:
        raise FileNotFoundError(f"No embedding files found under {feature_root}")

    args = [
        (p, feature_root, config["output_root"], config.get("convert_pt_to_npy", True))
        for p in files
    ]

    clip_map, manifest = {}, []
    with ThreadPoolExecutor(max_workers=config.get("loader_workers", 8)) as pool:
        for clip_id, path, x, tr_indices in tqdm(
            pool.map(_load_one_clip, args),
            total=len(files),
            desc="Loading embeddings (parallel)",
        ):
            clip_map[clip_id] = {"X": x, "tr_indices": tr_indices}
            manifest.append({
                "clip_id":         clip_id,
                "file":            str(path),
                "num_sampled_trs": int(x.shape[0]),
                "dim":             int(x.shape[1]),
            })

    return clip_map, manifest


# ---------------------------------------------------------------------------
# fMRI / alignment helpers
# ---------------------------------------------------------------------------

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


def load_subject_clip_data(subject, embedding_map, config):
    h5_path = (
        Path(config["fmri_root"]) / subject / "func"
        / config["friends_h5_pattern"].format(subject=subject)
    )
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing {h5_path}")

    clip_data, missing, unmatched_h5_keys = [], [], []

    with h5py.File(h5_path, "r") as f:
        all_h5_keys = list(f.keys())
        for clip_id in tqdm(
            sorted(embedding_map.keys(), key=natural_key),
            desc=f"Loading fMRI for {subject}",
            leave=False,
        ):
            try:
                key = find_h5_key_for_clip(f, clip_id)
            except Exception:
                missing.append(clip_id)
                continue

            Y = f[key][()]
            X_aligned, Y_aligned, kept_tr = align_sparse_clip_xy(
                embedding_map[clip_id]["X"],
                embedding_map[clip_id]["tr_indices"],
                Y,
                config["lag_trs"],
            )
            if X_aligned is None:
                missing.append(clip_id)
                continue

            clip_data.append({
                "clip_id":                 clip_id,
                "h5_key":                  key,
                "X":                       X_aligned,
                "Y":                       Y_aligned,
                "sampled_tr_indices_kept": kept_tr,
                "num_embedding_trs_raw":   int(embedding_map[clip_id]["X"].shape[0]),
                "num_fmri_trs_raw":        int(np.asarray(Y).shape[0]),
                "num_aligned_trs":         int(len(X_aligned)),
                "feature_dim":             int(X_aligned.shape[1]),
                "num_targets":             int(Y_aligned.shape[1]),
            })

        matched_keys      = {row["h5_key"] for row in clip_data}
        unmatched_h5_keys = [
            k for k in all_h5_keys
            if k not in matched_keys and "task-s01" in k.lower()
        ]

    clip_data.sort(key=lambda r: natural_key(r["clip_id"]))
    return h5_path, clip_data, missing, unmatched_h5_keys


def split_train_test_clips(clip_ids, train_fraction, min_train, min_test):
    n = len(clip_ids)
    if n < (min_train + min_test):
        raise RuntimeError(f"Need >= {min_train + min_test} matched clips, got {n}")
    n_train = max(min_train, min(int(np.floor(n * train_fraction)), n - min_test))
    return clip_ids[:n_train], clip_ids[n_train:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def parcelwise_pearson(y_true, y_pred):
    yt    = y_true - y_true.mean(axis=0, keepdims=True)
    yp    = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0))
    valid = denom > 0
    corr  = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    corr[valid] = (yt[:, valid] * yp[:, valid]).sum(0) / denom[valid]
    return corr


# ---------------------------------------------------------------------------
# Dual Ridge Regresion solver (Fast GPU Matmul)
# ---------------------------------------------------------------------------

def dual_ridge_solve(train_rows, test_rows, device, alpha: float = 1000.0):
    """
    Computes Dual Ridge Regression on the GPU.
    Loads memory-mapped chunks directly into a pre-allocated GPU tensor
    to avoid massive CPU RAM spikes.
    """
    n_train = sum(r["X"].shape[0] for r in train_rows)
    dim = train_rows[0]["X"].shape[1]
    n_targets = train_rows[0]["Y"].shape[1]
    
    tqdm.write(f"  -> Pre-allocating and streaming train data to GPU (N={n_train}, D={dim}) ...")
    
    # Pre-allocate on GPU
    X_tr = torch.empty((n_train, dim), device=device, dtype=torch.float32)
    Y_tr = torch.empty((n_train, n_targets), device=device, dtype=torch.float32)
    
    # Stream data directly onto GPU to bypass CPU concatenation overhead
    offset = 0
    for r in tqdm(train_rows, desc="  Loading chunks to VRAM", leave=False):
        n = r["X"].shape[0]
        X_tr[offset:offset + n] = torch.from_numpy(r["X"]).to(device, dtype=torch.float32)
        Y_tr[offset:offset + n] = torch.from_numpy(r["Y"]).to(device, dtype=torch.float32)
        offset += n

    tqdm.write("  -> Centering data on GPU...")
    Y_mean = Y_tr.mean(dim=0, keepdim=True)
    Y_tr.sub_(Y_mean)

    X_mean = X_tr.mean(dim=0, keepdim=True)
    X_tr.sub_(X_mean)

    tqdm.write(f"  -> Computing Dual Kernel K = X @ X^T (Shape: {n_train}x{n_train}) ...")
    K = X_tr @ X_tr.T

    tqdm.write(f"  -> Solving A = (K + alpha*I)^-1 @ Y ...")
    I = torch.eye(n_train, device=device, dtype=torch.float32)
    A = torch.linalg.solve(K + alpha * I, Y_tr)
    
    del K, I
    torch.cuda.empty_cache()

    tqdm.write("  -> Predicting on test set...")
    all_preds, all_trues = [], []
    for row in test_rows:
        X_te = torch.from_numpy(row["X"]).to(device, dtype=torch.float32)
        X_te.sub_(X_mean)
        
        # Dual prediction: Y_pred = (X_test @ X_train^T) @ A + Y_mean
        K_te = X_te @ X_tr.T
        pred = torch.addmm(Y_mean, K_te, A)
        
        all_preds.append(pred.cpu().numpy())
        all_trues.append(row["Y"])

    del X_tr, Y_tr, A, X_mean, Y_mean, X_te, K_te, pred
    torch.cuda.empty_cache()

    return np.concatenate(all_preds, 0), np.concatenate(all_trues, 0)


# ---------------------------------------------------------------------------
# Per-subject decoding
# ---------------------------------------------------------------------------

def decode_one_subject(subject, embedding_map, config):
    h5_path, clip_data, missing_clip_ids, unmatched_h5_keys = \
        load_subject_clip_data(subject, embedding_map, config)

    if not clip_data:
        raise RuntimeError(f"No matched season-1 clips for {subject}")

    clip_ids = [row["clip_id"] for row in clip_data]
    train_clip_ids, test_clip_ids = split_train_test_clips(
        clip_ids,
        config["train_fraction"],
        config["minimum_train_clips"],
        config["minimum_test_clips"],
    )

    train_set  = set(train_clip_ids)
    test_set   = set(test_clip_ids)
    train_rows = [row for row in clip_data if row["clip_id"] in train_set]
    test_rows  = [row for row in clip_data if row["clip_id"] in test_set]

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    tqdm.write(
        f"\n  Subject: {subject}  |  device: {device}  "
        f"|  train clips: {len(train_rows)}  |  test clips: {len(test_rows)}"
    )

    raw_dim = train_rows[0]["X"].shape[1]
    alpha   = float(config["ridge_alpha"])

    # Fast Dual-Ridge execution
    Y_pred, Y_test = dual_ridge_solve(train_rows, test_rows, device, alpha=alpha)

    corr = parcelwise_pearson(Y_test, Y_pred)
    r2   = r2_score(Y_test, Y_pred, multioutput="raw_values")

    per_clip_test_metrics = []
    offset = 0
    for row in test_rows:
        n = row["Y"].shape[0]
        clip_corr = parcelwise_pearson(Y_test[offset:offset + n], Y_pred[offset:offset + n])
        clip_r2   = r2_score(
            Y_test[offset:offset + n], Y_pred[offset:offset + n], multioutput="raw_values"
        )
        per_clip_test_metrics.append({
            "clip_id":           row["clip_id"],
            "h5_key":            row["h5_key"],
            "num_test_trs":      int(n),
            "mean_test_pearson": float(np.nanmean(clip_corr)),
            "mean_test_r2":      float(np.nanmean(clip_r2)),
        })
        offset += n

    result = {
        "subject":               subject,
        "h5_path":               str(h5_path),
        "feature_key":           config["feature_key_for_decoding"],
        "lag_trs":               int(config["lag_trs"]),
        "train_fraction":        float(config["train_fraction"]),
        "sample_fraction":       float(config["sample_fraction"]),
        "alpha":                 alpha,
        "raw_feature_dim":       int(raw_dim),
        "n_train_samples":       int(sum(r["X"].shape[0] for r in train_rows)),
        "n_test_samples":        int(sum(r["X"].shape[0] for r in test_rows)),
        "num_targets":           int(train_rows[0]["Y"].shape[1]),
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
            {
                "clip_id":                         row["clip_id"],
                "h5_key":                          row["h5_key"],
                "num_embedding_trs_raw":           row["num_embedding_trs_raw"],
                "num_fmri_trs_raw":                row["num_fmri_trs_raw"],
                "num_aligned_trs":                 row["num_aligned_trs"],
                "sampled_tr_indices_kept_preview": row["sampled_tr_indices_kept"][:10].tolist(),
            }
            for row in clip_data
        ],
        "per_clip_test_metrics": per_clip_test_metrics,
        "note": (
            "Dual Ridge Regression used to fix OOM. X @ X^T computed natively on GPU "
            "avoiding Primal space (D >> N) bottlenecks."
        ),
    }
    return result, Y_pred, Y_test


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_decoding(config):
    out_dir = Path(config["output_root"]) / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_map, embedding_manifest = load_embedding_map(config)

    with open(out_dir / "season1_embedding_manifest.json", "w") as f:
        json.dump(
            {
                "feature_key": config["feature_key_for_decoding"],
                "num_clips":   len(embedding_manifest),
                "clips":       embedding_manifest,
            },
            f, indent=2,
        )

    summary = []
    for subject in tqdm(config["subjects"], desc="Processing subjects"):
        report_path = out_dir / f"{subject}_report.json"
        if report_path.exists() and not config.get("force_redecode", False):
            tqdm.write(f"  Skipping {subject} (cached)")
            with open(report_path) as f:
                summary.append(json.load(f))
            continue

        tqdm.write(f"\n{'='*60}\nStarting {subject}\n{'='*60}")
        result, y_pred, y_true = decode_one_subject(subject, embedding_map, config)

        np.save(out_dir / f"{subject}_y_pred.npy", y_pred)
        np.save(out_dir / f"{subject}_y_true.npy", y_true)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nClip-matched prediction summary:")
    for row in summary:
        print(
            f"  {row['subject']}: "
            f"Pearson={row['mean_test_pearson']:.4f}  "
            f"R2={row['mean_test_r2']:.4f}  "
            f"alpha={row['alpha']}  "
            f"clips={row['num_matched_clips']}"
        )


def main():
    run_decoding(CONFIG)


if __name__ == "__main__":
    main()