import os
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


CONFIG = {
    # Input embeddings from your existing extraction
    "embedding_dir": "/scratch/saigum/algonauts_outputs/s1_vjepa_decode_uniform10pct_oomfixed/embeddings/enc-last-ln",
    "sampled_tr_dir": "/scratch/saigum/algonauts_outputs/s1_vjepa_decode_uniform10pct_oomfixed/embeddings/sampled_tr_indices",

    # fMRI roots
    "fmri_root": "/scratch/saigum/algonauts_2025.competitors/fmri",
    "friends_h5_pattern": "{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5",
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-05"],

    # Output
    "output_root": "/scratch/saigum/algonauts_outputs/s1_postpooled_encoding",

    # Pooling mode for saved tensors shaped like [num_sampled_trs, 3136, 1024]
    # Options:
    #   "mean_tokens"      -> [N, 1024]
    #   "spatial_3x3"      -> infer T,H,W from tokens, avg over time, adaptive pool spatially to 3x3 -> [N, 9*D]
    "pooling_mode": "mean_tokens",

    # Alignment / model
    "lag_trs": 3,
    "train_fraction": 0.8,
    "minimum_train_clips": 4,
    "minimum_test_clips": 1,
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],

    # Save pooled per-clip embeddings
    "save_pooled_embeddings": True,

    # Recompute controls
    "force_repool": False,
    "force_reencode": False,
}


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def clip_id_from_name(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(s\d{2}e\d{2}[a-z])", stem.lower())
    if not m:
        raise ValueError(f"Could not parse clip id from {name}")
    return m.group(1)


def infer_token_grid(num_tokens: int):
    """
    Try to infer (T, H, W) from flattened token count.
    For your current tensors, 3136 = 16 * 14 * 14 is the likely match.
    """
    for T in (16, 8, 32, 12, 24):
        if num_tokens % T != 0:
            continue
        spatial = num_tokens // T
        side = int(round(spatial ** 0.5))
        if side * side == spatial:
            return T, side, side
    raise ValueError(f"Could not infer token grid from num_tokens={num_tokens}")


def adaptive_pool_2d_lastdim(x: np.ndarray, out_h: int = 3, out_w: int = 3) -> np.ndarray:
    """
    x: [H, W, D] -> [out_h, out_w, D]
    Simple average pooling by evenly splitting bins.
    """
    H, W, D = x.shape
    pooled = np.zeros((out_h, out_w, D), dtype=np.float32)

    hs = np.linspace(0, H, out_h + 1).astype(int)
    ws = np.linspace(0, W, out_w + 1).astype(int)

    for i in range(out_h):
        for j in range(out_w):
            patch = x[hs[i]:hs[i + 1], ws[j]:ws[j + 1], :]
            pooled[i, j] = patch.mean(axis=(0, 1))
    return pooled


def pool_saved_tensor(x: np.ndarray, pooling_mode: str) -> np.ndarray:
    """
    Input examples:
      [N, 3136, 1024]  raw saved token features
      [N, D]           already pooled
    Output:
      [N, D'] suitable for ridge regression
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        return x

    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D embedding tensor, got shape {x.shape}")

    N, num_tokens, dim = x.shape

    if pooling_mode == "mean_tokens":
        return x.mean(axis=1)

    if pooling_mode == "spatial_3x3":
        T, H, W = infer_token_grid(num_tokens)
        x = x.reshape(N, T, H, W, dim)
        # average over time -> [N, H, W, D]
        x = x.mean(axis=1)

        pooled_rows = []
        for i in range(N):
            pooled = adaptive_pool_2d_lastdim(x[i], 3, 3)   # [3, 3, D]
            pooled_rows.append(pooled.reshape(-1))          # [9*D]
        return np.stack(pooled_rows, axis=0).astype(np.float32)

    raise ValueError(f"Unsupported pooling_mode={pooling_mode}")


def load_and_pool_embedding(path: Path, pooling_mode: str):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, torch.Tensor):
        x = x.float().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
    return pool_saved_tensor(x, pooling_mode)


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


def load_embedding_map(config):
    embedding_dir = Path(config["embedding_dir"])
    sampled_tr_dir = Path(config["sampled_tr_dir"])
    pooled_dir = Path(config["output_root"]) / "pooled_embeddings" / config["pooling_mode"]
    pooled_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(embedding_dir.glob("*.pt"), key=lambda p: natural_key(p.name))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {embedding_dir}")

    clip_map = {}
    manifest = []

    for path in files:
        stem = path.stem
        clip_id = clip_id_from_name(stem)
        tr_idx_path = sampled_tr_dir / f"{stem}.npy"
        if not tr_idx_path.exists():
            raise FileNotFoundError(f"Missing sampled TR indices for {stem}: {tr_idx_path}")

        pooled_path = pooled_dir / f"{stem}.pt"
        if pooled_path.exists() and not config["force_repool"]:
            pooled = torch.load(pooled_path, map_location="cpu")
            if isinstance(pooled, torch.Tensor):
                pooled = pooled.float().numpy()
            else:
                pooled = np.asarray(pooled, dtype=np.float32)
        else:
            pooled = load_and_pool_embedding(path, config["pooling_mode"])
            if config["save_pooled_embeddings"]:
                torch.save(torch.from_numpy(pooled), pooled_path)

        tr_indices = np.load(tr_idx_path).astype(np.int64)

        clip_map[clip_id] = {
            "X": pooled,
            "tr_indices": tr_indices,
            "source_file": str(path),
            "pooled_file": str(pooled_path),
        }
        manifest.append(
            {
                "clip_id": clip_id,
                "raw_file": str(path),
                "pooled_file": str(pooled_path),
                "tr_idx_file": str(tr_idx_path),
                "num_sampled_trs": int(pooled.shape[0]),
                "feature_dim": int(pooled.shape[1]),
            }
        )

    return clip_map, manifest


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
                X, tr_indices, Y, config["lag_trs"]
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
    return X, Y


def encode_one_subject(subject, embedding_map, config):
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
        y_true_clip = Y_test[offset:offset + n]
        y_pred_clip = Y_pred[offset:offset + n]
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
        "pooling_mode": config["pooling_mode"],
        "lag_trs": int(config["lag_trs"]),
        "train_fraction": float(config["train_fraction"]),
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
        "note": "This is a post-processed encoding model: raw saved token tensors are pooled after extraction, then aligned clip-wise and fit with ridge regression.",
    }
    return result, Y_pred, Y_test


def main():
    output_root = Path(CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    embedding_map, embedding_manifest = load_embedding_map(CONFIG)
    with open(output_root / "pooled_embedding_manifest.json", "w") as f:
        json.dump(
            {
                "pooling_mode": CONFIG["pooling_mode"],
                "num_clips": len(embedding_manifest),
                "clips": embedding_manifest,
            },
            f,
            indent=2,
        )

    summary = []
    for subject in CONFIG["subjects"]:
        report_path = reports_dir / f"{subject}_report.json"
        if report_path.exists() and not CONFIG["force_reencode"]:
            with open(report_path) as f:
                summary.append(json.load(f))
            continue

        result, y_pred, y_true = encode_one_subject(subject, embedding_map, CONFIG)
        np.save(reports_dir / f"{subject}_y_pred.npy", y_pred)
        np.save(reports_dir / f"{subject}_y_true.npy", y_true)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

    with open(reports_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Post-pooled encoding summary:")
    for row in summary:
        print(
            f"  {row['subject']}: mean Pearson={row['mean_test_pearson']:.4f}, "
            f"mean R2={row['mean_test_r2']:.4f}, alpha={row['alpha']}, "
            f"feature_dim={row['feature_dim']}"
        )


if __name__ == "__main__":
    main()
