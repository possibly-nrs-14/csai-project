import os
import json
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def parcels_to_volume(r_scores, atlas_path):
    """Map 1000-parcel r scores into the atlas 3D volume."""
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)  # parcel IDs 1–1000, 0=background
    vol = np.zeros_like(atlas_data, dtype=float)
    for parcel_id in range(1, len(r_scores) + 1):
        mask = atlas_data == parcel_id
        vol[mask] = r_scores[parcel_id - 1]
    return nib.Nifti1Image(vol, atlas_img.affine)

def plot_brain_slices(nifti_img, title, output_path, vmax=None, cmap="RdBu_r"):
    data = nifti_img.get_fdata()
    vmax = vmax or np.percentile(np.abs(data[data != 0]), 95)
    vmin = -vmax

    slices = {
        "Axial (z)": data[:, :, data.shape[2] // 2],
        "Coronal (y)": data[:, data.shape[1] // 2, :],
        "Sagittal (x)": data[data.shape[0] // 2, :, :],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, sl) in zip(axes, slices.items()):
        im = ax.imshow(sl.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="Pearson r")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot fMRI encoding results.")
    parser.add_argument("--subject", type=str, default="sub-02", help="Subject ID (e.g., sub-02)")
    parser.add_argument("--feature_key", type=str, default="enc-last-ln", help="Feature key (e.g., enc-last-ln)")
    parser.add_argument("--results_dir", type=str, default="/scratch/arihantr/CSAI/results_encoding/", help="Base results directory")
    parser.add_argument("--fmri_dir", type=str, default="/scratch/arihantr/CSAI/algonauts_2025.competitors/fmri/", help="Base fMRI directory for atlases")
    args = parser.parse_args()

    MODELS = ["ridge", "lasso"]
    SUBJECT = args.subject
    FEATURE_KEY = args.feature_key
    RESULTS_FOLDER = args.results_dir
    
    # Dynamically format the atlas path based on the subject
    ATLAS_PATH = os.path.join(
        args.fmri_dir, 
        SUBJECT, 
        "atlas",
        f"{SUBJECT}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    )

    # Ensure the output directory for the subject exists
    output_dir = os.path.join(RESULTS_FOLDER, SUBJECT)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    print(f"--- Loading data for {SUBJECT} | Feature: {FEATURE_KEY} ---")
    
    for model in MODELS:
        r_path = os.path.join(RESULTS_FOLDER, SUBJECT, model, FEATURE_KEY, "pearson_r.npy")
        j_path = os.path.join(RESULTS_FOLDER, SUBJECT, model, FEATURE_KEY, "pearson_r.json")
        if not os.path.isfile(r_path) or not os.path.isfile(j_path):
            print(f"[SKIP] Data for {model} not found at {r_path} or {j_path}")
            continue
        
        results[model] = {
            "r": np.load(r_path),
            "summary": json.load(open(j_path)),
        }
        s = results[model]["summary"]
        print(f"{model:10s}  mean r={s['mean_r']:.4f}  median r={s['median_r']:.4f}  "
              f"max r={s['max_r']:.4f}  %positive={s['pct_positive']:.1f}%")

    if not results:
        print("No results found. Exiting.")
        return

    # 1. Plot Distribution
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (model, data) in zip(axes, results.items()):
        r = data["r"]
        ax.hist(r, bins=60, color="steelblue" if model == "ridge" else "tomato",
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(np.mean(r), color="black", linestyle="--", linewidth=1.2, label=f"mean={np.mean(r):.3f}")
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_title(f"{model.capitalize()} — {FEATURE_KEY}", fontsize=11)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Number of parcels")
        ax.legend(fontsize=9)

    plt.tight_layout()
    dist_path = os.path.join(output_dir, f"{FEATURE_KEY}_pearson_r_distribution.png")
    plt.savefig(dist_path, dpi=150)
    print(f"Saved: {dist_path}")
    plt.show()
    plt.close()

    # 2. Scatter Plot: Ridge vs Lasso
    if "ridge" in results and "lasso" in results:
        r_ridge = results["ridge"]["r"]
        r_lasso = results["lasso"]["r"]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(r_ridge, r_lasso, s=4, alpha=0.4, color="mediumslateblue")
        lim = max(abs(r_ridge).max(), abs(r_lasso).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="y=x")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Ridge — Pearson r")
        ax.set_ylabel("Lasso — Pearson r")
        ax.set_title(f"Ridge vs Lasso per parcel ({SUBJECT})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        
        scatter_path = os.path.join(output_dir, f"{FEATURE_KEY}_ridge_vs_lasso_scatter.png")
        plt.savefig(scatter_path, dpi=150)
        print(f"Saved: {scatter_path}")
        plt.show()
        plt.close()
    else:
        print("Need both ridge and lasso results for the scatter plot. Skipping.")

    # 3. Top N Parcels Bar Chart
    TOP_N = 20
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (model, data) in zip(axes, results.items()):
        r = data["r"]
        top_idx = np.argsort(r)[::-1][:TOP_N]
        ax.barh(
            [f"Parcel {i+1}" for i in top_idx], # Added +1 so parcels are 1-indexed (matching atlas)
            r[top_idx],
            color="steelblue" if model == "ridge" else "tomato",
            alpha=0.85,
        )
        ax.set_xlabel("Pearson r")
        ax.set_title(f"{model.capitalize()} — Top {TOP_N} parcels")
        ax.invert_yaxis()

    plt.tight_layout()
    top_parcels_path = os.path.join(output_dir, f"{FEATURE_KEY}_top_parcels.png")
    plt.savefig(top_parcels_path, dpi=150)
    print(f"Saved: {top_parcels_path}")
    plt.show()
    plt.close()

    # 4. Brain Maps
    if os.path.isfile(ATLAS_PATH):
        for model, data in results.items():
            brain_img = parcels_to_volume(data["r"], ATLAS_PATH)
            
            # Save NIfTI
            nii_path = os.path.join(output_dir, f"{model}_{FEATURE_KEY}_pearson_r.nii.gz")
            nib.save(brain_img, nii_path)
            print(f"Saved NIfTI: {nii_path}")
            
            # Plot and save slices
            title = f"{model.capitalize()} Pearson r — {SUBJECT} ({FEATURE_KEY})"
            save_name = f"{model}_{FEATURE_KEY}_brain_slices.png"
            slice_path = os.path.join(output_dir, save_name)
            
            plot_brain_slices(brain_img, title=title, output_path=slice_path)
            print(f"Saved Brain Slices: {slice_path}")
    else:
        print(f"\n[WARNING] Atlas not found at {ATLAS_PATH} — skipping brain map.")
        print(f"Make sure you ran: datalad get fmri/{SUBJECT}/atlas/")

    # 5. Final Summary Table
    print("\n--- Summary ---")
    print(f"{'Model':<10} {'Mean r':>8} {'Median r':>9} {'Max r':>7} {'% positive':>11}")
    print("-" * 50)
    for model, data in results.items():
        s = data["summary"]
        print(f"{model:<10} {s['mean_r']:>8.4f} {s['median_r']:>9.4f} {s['max_r']:>7.4f} {s['pct_positive']:>10.1f}%")

if __name__ == "__main__":
    main()