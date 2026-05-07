import os
import re
from pathlib import Path
from itertools import combinations

import h5py
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "fmri_root": "/scratch/arihantr/CSAI/algonauts_2025.competitors/fmri",
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-05"],
    "friends_h5_pattern": (
        "{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18"
        "_parcel-1000Par7Net_desc-s123456_bold.h5"
    ),
    "train_fraction": 0.8,  # Used to isolate the final 20% of clips as the Test Set
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]

def extract_clip_id_from_h5_key(key: str) -> str:
    # Matches patterns like "task-s01e02a"
    m = re.search(r"(s\d{2}e\d{2}[a-z])", key.lower())
    if not m:
        return None
    return m.group(1)

# ---------------------------------------------------------------------------
# Core Math Functions
# ---------------------------------------------------------------------------
def parcelwise_pearson(y_a, y_b):
    """Computes Pearson r for each parcel independently across time."""
    yt = y_a - y_a.mean(axis=0, keepdims=True)
    yp = y_b - y_b.mean(axis=0, keepdims=True)
    
    denom = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0))
    valid = denom > 0
    
    corr = np.full(y_a.shape[1], np.nan, dtype=np.float32)
    corr[valid] = (yt[:, valid] * yp[:, valid]).sum(0) / denom[valid]
    return corr

def compute_isc_for_clip(subject_y_dict):
    """
    Computes Inter-Subject Correlation (Noise Ceiling) for a single clip using
    Fisher z-transformation.
    """
    subjects = list(subject_y_dict.keys())
    if len(subjects) < 2:
        return None

    pairwise_z_corrs = []
    
    # Calculate for every unique pair (e.g., Sub01-Sub02, Sub01-Sub03, etc.)
    for sub_a, sub_b in combinations(subjects, 2):
        y_a = subject_y_dict[sub_a]
        y_b = subject_y_dict[sub_b]
        
        # Guardrail: fMRI recordings might differ slightly by a few TRs at the end
        min_trs = min(y_a.shape[0], y_b.shape[0])
        
        # Calculate raw r
        raw_r = parcelwise_pearson(y_a[:min_trs], y_b[:min_trs])
        
        # Fisher z-transform (clip to prevent infinity on perfect correlations)
        clipped_r = np.clip(raw_r, -0.9999, 0.9999)
        z_score = np.arctanh(clipped_r)
        
        pairwise_z_corrs.append(z_score)
        
    # Average the z-scores across all pairs
    mean_z = np.nanmean(pairwise_z_corrs, axis=0)
    
    # Inverse transform back to r
    mean_isc_r = np.tanh(mean_z)
    
    return mean_isc_r

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    print("--- Algonauts 2025: Inter-Subject Correlation (Noise Ceiling) ---")
    
    # 1. Map all available clips per subject
    subject_h5_paths = {}
    subject_clip_keys = {}
    
    for subject in CONFIG["subjects"]:
        h5_path = Path(CONFIG["fmri_root"]) / subject / "func" / CONFIG["friends_h5_pattern"].format(subject=subject)
        subject_h5_paths[subject] = h5_path
        
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing {h5_path}")
            
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            clip_mapping = {}
            for k in keys:
                clip_id = extract_clip_id_from_h5_key(k)
                if clip_id:
                    clip_mapping[clip_id] = k
            subject_clip_keys[subject] = clip_mapping

    # 2. Find clips that ALL subjects share
    shared_clip_ids = set.intersection(*[set(d.keys()) for d in subject_clip_keys.values()])
    shared_clip_ids = sorted(list(shared_clip_ids), key=natural_key)
    
    print(f"Found {len(shared_clip_ids)} clips shared across all {len(CONFIG['subjects'])} subjects.")
    
    # 3. Apply exactly the same Train/Test split as your ML pipeline
    n_total = len(shared_clip_ids)
    n_train = int(np.floor(n_total * CONFIG["train_fraction"]))
    test_clip_ids = shared_clip_ids[n_train:]
    
    print(f"Isolating {len(test_clip_ids)} Test Set clips for ISC calculation...\n")
    
    # 4. Calculate ISC for the test clips
    clip_iscs = []
    
    # Keep files open to avoid repeated I/O overhead
    files = {sub: h5py.File(path, "r") for sub, path in subject_h5_paths.items()}
    
    try:
        for clip_id in tqdm(test_clip_ids, desc="Computing ISC per Test Clip"):
            
            # Extract Y_true data for this clip for all subjects
            subject_y_dict = {}
            for subject, f in files.items():
                h5_key = subject_clip_keys[subject][clip_id]
                subject_y_dict[subject] = f[h5_key][()]
                
            # Compute ISC
            isc_array = compute_isc_for_clip(subject_y_dict)
            if isc_array is not None:
                clip_iscs.append(isc_array)
                
    finally:
        # Clean up file handles
        for f in files.values():
            f.close()

    # 5. Aggregate Results
    # clip_iscs is a list of arrays, each shape (num_parcels,)
    all_clips_isc = np.stack(clip_iscs, axis=0) # shape: (num_test_clips, num_parcels)
    
    # Mean across all clips
    final_parcel_isc = np.nanmean(all_clips_isc, axis=0)
    
    # Global scalar average
    global_noise_ceiling = np.nanmean(final_parcel_isc)
    
    print("\n" + "="*60)
    print("NOISE CEILING RESULTS")
    print("="*60)
    print(f"Global Test Set ISC (Noise Ceiling): {global_noise_ceiling:.4f}")
    print("\nIf your Dual Ridge Regression model achieves an R2 or Pearson")
    print(f"score close to ~{global_noise_ceiling:.4f}, your model is effectively")
    print("capturing all the theoretical true signal available in the data.")
    print("="*60)

if __name__ == "__main__":
    main()