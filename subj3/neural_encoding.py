"""
Neural encoding: predict fMRI responses from V-JEPA embeddings
using Ridge (linear) and Lasso regression, one parcel at a time.

Inputs
------
- Embeddings produced by collect_embeddings.py  (embeddings/ folder)
- fMRI .h5 file for sub-03

Outputs
-------
- results/<model>/<feature_key>/pearson_r.npy   shape (1000,)  per-parcel r
- results/<model>/<feature_key>/pearson_r.json  summary stats
"""

import os
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ===============================================================
# Config — edit before running
# ===============================================================
CONFIG = {
    # Path to the embeddings folder produced by collect_embeddings.py
    # (the folder name encodes tr/len/before params)
    "embeddings_folder": "embeddings/_tr1.49_len8_before6_vjepa1_ijepa0",

    # Which V-JEPA feature layer(s) to use for encoding.
    # Set to None to run all layers found in embeddings_folder.
    "feature_keys": [
        "v-jepa2-vitl-enc-last-ln",
    ],

    # fMRI
    "fmri_h5": "algonauts_2025.competitors/fmri/sub-03/func/"
               "sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5",
    "subject": "sub-03",

    # Which episodes to use (must match downloaded + embedded videos)
    # Format: session_key as stored in the .h5 file
    # e.g. "ses-004_task-s01e01a"
    # Set to None to use all keys found in the h5 that also have embeddings.
    "episode_keys": None,

    # Regression
    "models": ["ridge", "lasso"],   # which models to run
    "ridge_alpha": 1000.0,
    "lasso_alpha": 0.01,
    "n_folds": 5,                   # cross-validation folds
    "n_parcels": 1000,

    # Output
    "output_folder": "results/",
}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def load_embeddings_for_episode(embeddings_folder, feature_key, episode_file_stem):
    """Load the .pt tensor for one episode split, shape (T, D)."""
    # embeddings are stored under <feature_key>/<relative_path>/<stem>.pt
    # relative_path mirrors stimuli/movies structure, e.g. movies/friends/s1
    # We search recursively for the stem.
    search_root = os.path.join(embeddings_folder, feature_key)
    for root, _, files in os.walk(search_root):
        for f in files:
            if f == episode_file_stem + ".pt":
                return torch.load(os.path.join(root, f), weights_only=True).float().numpy()
    return None


def load_fmri_for_key(h5_path, key):
    """Load fMRI array for one episode key, shape (T, 1000)."""
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            return None
        return f[key][:]


def find_episode_keys(h5_path, embeddings_folder, feature_key):
    """Return h5 keys that have both fMRI data and a matching embedding."""
    with h5py.File(h5_path, "r") as f:
        h5_keys = list(f.keys())

    matched = []
    for key in tqdm(h5_keys, desc="Matching episodes", unit="ep", leave=False):
        # key looks like "ses-004_task-s01e01a"
        # episode_split is the last part after "task-"
        parts = key.split("_task-")
        if len(parts) != 2:
            continue
        episode_split = parts[1]                        # e.g. "s01e01a"
        # collect_embeddings saves as friends_s01e01a.pt (mkv stem without extension)
        mkv_stem = f"friends_{episode_split}"           # friends_s01e01a

        emb = load_embeddings_for_episode(embeddings_folder, feature_key, mkv_stem)
        if emb is not None:
            matched.append((key, mkv_stem))

    return matched


def align_lengths(X, Y):
    """Trim X and Y to the same number of timepoints."""
    n = min(len(X), len(Y))
    return X[:n], Y[:n]


def run_encoding(X, Y, model_type, ridge_alpha, lasso_alpha, n_folds):
    """
    Cross-validated encoding: predict Y (T x P) from X (T x D).
    Returns per-parcel Pearson r, shape (P,).
    """
    n_parcels = Y.shape[1]
    r_scores = np.zeros(n_parcels)

    kf = KFold(n_splits=n_folds, shuffle=False)
    pred_all = np.zeros_like(Y)

    scaler_X = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(kf.split(X), total=n_folds, desc=f"  CV folds [{model_type}]", unit="fold")
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]

        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s = scaler_X.transform(X_test)

        if model_type == "ridge":
            reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
        elif model_type == "lasso":
            reg = Lasso(alpha=lasso_alpha, fit_intercept=True, max_iter=5000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        reg.fit(X_train_s, Y_train)
        pred_all[test_idx] = reg.predict(X_test_s)

    for p in tqdm(range(n_parcels), desc=f"  Pearson r [{model_type}]", unit="parcel"):
        r, _ = pearsonr(Y[:, p], pred_all[:, p])
        r_scores[p] = r if np.isfinite(r) else 0.0

    return r_scores


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    cfg = CONFIG
    embeddings_folder = cfg["embeddings_folder"]
    fmri_h5 = os.path.realpath(cfg["fmri_h5"])  # resolve git-annex symlink
    output_folder = cfg["output_folder"]

    feature_keys = cfg["feature_keys"]
    if feature_keys is None:
        feature_keys = [
            d for d in os.listdir(embeddings_folder)
            if os.path.isdir(os.path.join(embeddings_folder, d)) and not d.startswith(".")
        ]
        print(f"Found feature keys: {feature_keys}")

    for feature_key in tqdm(feature_keys, desc="Feature keys", unit="key"):
        print(f"\n=== Feature key: {feature_key} ===")

        episode_keys = cfg["episode_keys"]
        if episode_keys is None:
            matched = find_episode_keys(fmri_h5, embeddings_folder, feature_key)
        else:
            matched = []
            for key in episode_keys:
                parts = key.split("_task-")
                mkv_stem = f"friends_{parts[1]}" if len(parts) == 2 else key
                matched.append((key, mkv_stem))

        if not matched:
            print(f"  No matching episodes found for {feature_key}. Skipping.")
            continue

        print(f"  Episodes: {[k for k, _ in matched]}")

        # Concatenate across episodes
        X_all, Y_all = [], []
        for h5_key, mkv_stem in tqdm(matched, desc="Loading episodes", unit="ep"):
            emb = load_embeddings_for_episode(embeddings_folder, feature_key, mkv_stem)
            fmri = load_fmri_for_key(fmri_h5, h5_key)

            if emb is None:
                tqdm.write(f"  [SKIP] No embedding for {mkv_stem}")
                continue
            if fmri is None:
                tqdm.write(f"  [SKIP] No fMRI for {h5_key}")
                continue

            emb, fmri = align_lengths(emb, fmri)
            X_all.append(emb)
            Y_all.append(fmri)
            tqdm.write(f"  Loaded {h5_key}: X={emb.shape}, Y={fmri.shape}")

        if not X_all:
            print("  Nothing to encode. Skipping.")
            continue

        X = np.concatenate(X_all, axis=0)
        Y = np.concatenate(Y_all, axis=0)
        print(f"  Total: X={X.shape}, Y={Y.shape}")

        for model_type in tqdm(cfg["models"], desc="Models", unit="model", leave=False):
            print(f"  Running {model_type}...")
            r_scores = run_encoding(
                X, Y,
                model_type=model_type,
                ridge_alpha=cfg["ridge_alpha"],
                lasso_alpha=cfg["lasso_alpha"],
                n_folds=cfg["n_folds"],
            )

            save_dir = os.path.join(output_folder, model_type, feature_key)
            os.makedirs(save_dir, exist_ok=True)

            np.save(os.path.join(save_dir, "pearson_r.npy"), r_scores)

            summary = {
                "model": model_type,
                "feature_key": feature_key,
                "subject": cfg["subject"],
                "n_timepoints": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_parcels": int(Y.shape[1]),
                "n_folds": cfg["n_folds"],
                "mean_r": float(np.mean(r_scores)),
                "median_r": float(np.median(r_scores)),
                "max_r": float(np.max(r_scores)),
                "pct_positive": float(np.mean(r_scores > 0) * 100),
            }
            with open(os.path.join(save_dir, "pearson_r.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print(f"    mean r={summary['mean_r']:.4f}  median r={summary['median_r']:.4f}  "
                  f"max r={summary['max_r']:.4f}  %positive={summary['pct_positive']:.1f}%")
            print(f"    Saved to {save_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
