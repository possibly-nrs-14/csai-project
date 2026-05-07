#!/usr/bin/env python3
"""
visualize_brain.py  —  Stage 3 of the V-JEPA → fMRI encoding pipeline.

Reads the outputs of encode.py:
  {subject}_y_pred.npy  (n_test_trs × 1000)
  {subject}_y_true.npy  (n_test_trs × 1000)

Computes per-parcel Pearson correlations, projects them onto the fsaverage
cortical surface via the Schaefer-1000 atlas, and renders a 4-view brain
map (Left Lateral / Left Medial / Right Lateral / Right Medial).

Dependencies:
  pip install nilearn nibabel matplotlib Pillow scipy numpy

Usage:
  python visualize_brain.py                      # uses CONFIG below
  python visualize_brain.py --subject sub-01     # single subject
  python visualize_brain.py --subject mean       # mean across all subjects (default)
"""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from PIL import Image

from nilearn import datasets, surface, plotting


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  update to match your encode.py output_root
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Directory containing {subject}_y_pred.npy / {subject}_y_true.npy
    "decoding_root": (
        "/scratch/arihantr/CSAI/algonauts_outputs/"
        "s1_vjepa_decode_uniform8pct_oomfixed/decoding"
    ),

    # Subjects to load (used when aggregate == "mean")
    "subjects": ["sub-01", "sub-02", "sub-03", "sub-05"],

    # "mean"   → average correlations across all subjects before plotting
    # "sub-01" → plot only that subject
    "aggregate": "mean",

    # Schaefer atlas parameters  (must match encode.py)
    "n_parcels":   1000,
    "n_networks":  7,

    # Output figure
    "plot_title":  "All Prediction Correlations",
    "output_fig":  "all_prediction_correlations.png",

    # Color scale
    "vmin": -0.3,
    "vmax":  0.6,
    "dpi":   150,
}


# ─────────────────────────────────────────────────────────────────────────────
# Colormap  —  mirrors the screenshot (cyan → black → yellow)
# ─────────────────────────────────────────────────────────────────────────────
def make_hot_cold_cmap() -> LinearSegmentedColormap:
    """
    Diverging colormap centred at zero:
      negative: cyan  → blue → near-black
      positive: near-black → dark-red → orange → bright-yellow
    """
    anchors = [
        (0.00, (0.00, 0.90, 1.00)),   # cyan          (most negative)
        (0.20, (0.00, 0.20, 0.90)),   # blue
        (0.42, (0.00, 0.00, 0.25)),   # deep blue-black
        (0.50, (0.04, 0.00, 0.04)),   # near-black    (zero)
        (0.58, (0.25, 0.00, 0.00)),   # deep red-black
        (0.72, (0.80, 0.10, 0.00)),   # dark red / orange
        (0.87, (1.00, 0.50, 0.00)),   # orange
        (1.00, (1.00, 1.00, 0.20)),   # bright yellow (most positive)
    ]
    positions, colors = zip(*anchors)
    return LinearSegmentedColormap.from_list(
        "hot_cold", list(zip(positions, colors)), N=512
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def parcelwise_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Vectorised Pearson r for each parcel (column).

    Parameters
    ----------
    y_true, y_pred : (n_samples, n_parcels)

    Returns
    -------
    r : (n_parcels,)  float32
    """
    y_true = y_true - y_true.mean(0, keepdims=True)
    y_pred = y_pred - y_pred.mean(0, keepdims=True)
    num   = (y_true * y_pred).sum(0)
    denom = np.sqrt((y_true ** 2).sum(0) * (y_pred ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, num / denom, 0.0)
    return r.astype(np.float32)


def load_correlations(config: dict) -> np.ndarray:
    """
    Load y_pred / y_true and compute per-parcel Pearson r.
    Averages across subjects when aggregate == "mean".

    Returns
    -------
    correlations : (n_parcels,)
    """
    root = Path(config["decoding_root"])
    aggregate = config["aggregate"]
    subjects  = config["subjects"] if aggregate == "mean" else [aggregate]

    all_corr = []
    for subject in subjects:
        pred_path = root / f"{subject}_y_pred.npy"
        true_path = root / f"{subject}_y_true.npy"
        if not pred_path.exists():
            print(f"  [WARN] Missing {pred_path.name} — skipping {subject}")
            continue
        y_pred = np.load(pred_path)           # (n_test, n_parcels)
        y_true = np.load(true_path)
        corr   = parcelwise_pearson(y_true, y_pred)
        all_corr.append(corr)
        print(
            f"  {subject}: mean r = {np.nanmean(corr):.4f}  "
            f"max = {np.nanmax(corr):.4f}  min = {np.nanmin(corr):.4f}"
        )

    if not all_corr:
        raise RuntimeError("No subject data could be loaded — check decoding_root and subject ids.")

    return np.nanmean(np.stack(all_corr, 0), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Atlas → surface projection
# ─────────────────────────────────────────────────────────────────────────────
def build_surface_map(
    correlations: np.ndarray,
    atlas_img,
    fsaverage,
    hemi: str,
):
    """
    Project Schaefer-1000 correlation values onto fsaverage surface vertices.

    Schaefer 1000-parcel NIfTI label convention (1-indexed):
      labels   1 –  500  →  left-hemisphere parcels   (correlations[  0:500])
      labels 501 – 1000  →  right-hemisphere parcels  (correlations[500:1000])

    Parameters
    ----------
    hemi : "left" or "right"

    Returns
    -------
    surf_map : (n_vertices,) float32
    sulc     : path to sulcal depth file (used as background texture)
    mesh     : path to pial surface
    """
    if hemi == "left":
        mesh         = fsaverage.pial_left
        sulc         = fsaverage.sulc_left
        label_start  = 1          # 1-indexed atlas labels for LH
        corr_slice   = slice(0,   500)
    else:
        mesh         = fsaverage.pial_right
        sulc         = fsaverage.sulc_right
        label_start  = 501
        corr_slice   = slice(500, 1000)

    hemi_corr = correlations[corr_slice]                          # (500,)

    # Project volumetric atlas labels to surface vertices
    labels_on_surf = surface.vol_to_surf(atlas_img, mesh)         # (n_vertices,) float

    # Map each parcel label → correlation value
    rounded = np.round(labels_on_surf).astype(np.int32)
    surf_map = np.zeros(len(rounded), dtype=np.float32)
    for local_idx in range(500):
        atlas_label = label_start + local_idx
        mask = rounded == atlas_label
        if mask.any():
            surf_map[mask] = hemi_corr[local_idx]

    return surf_map, sulc, mesh


# ─────────────────────────────────────────────────────────────────────────────
# Per-view rendering  (nilearn → BytesIO → PIL → numpy)
# ─────────────────────────────────────────────────────────────────────────────
VIEWS = [
    ("left",  "lateral", "Left Lateral"),
    ("left",  "medial",  "Left Medial"),
    ("right", "lateral", "Right Lateral"),
    ("right", "medial",  "Right Medial"),
]


def render_view(surf_map, sulc, mesh, hemi, view, cmap, vmin, vmax) -> np.ndarray:
    """
    Render one brain view to a numpy RGB array via nilearn + BytesIO.
    Works with nilearn ≥ 0.9 (returns Figure) and older versions
    (returns SurfaceView with a .figure attribute).
    """
    # Build kwargs defensively — some args were removed in nilearn ≥ 0.10
    kwargs = dict(
        surf_mesh=mesh,
        stat_map=surf_map,
        hemi=hemi,
        view=view,
        colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,
        bg_map=sulc,
        engine="matplotlib",
    )
    import inspect
    sig = inspect.signature(plotting.plot_surf_stat_map)
    for deprecated in ("bg_on_stat", "darkness"):
        if deprecated in sig.parameters:
            kwargs[deprecated] = {"bg_on_stat": True, "darkness": 0.5}[deprecated]

    result = plotting.plot_surf_stat_map(**kwargs)

    # Normalise return type across nilearn versions
    fig = result if isinstance(result, plt.Figure) else result.figure

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure composition
# ─────────────────────────────────────────────────────────────────────────────
def compose_figure(
    view_images: list,
    view_labels: list,
    cmap,
    vmin: float,
    vmax: float,
    title: str,
    dpi: int,
    out_path: str,
):
    """
    Stitch four brain-view images into a 2 × 2 grid with a shared colorbar.
    """
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(title, fontsize=16, y=0.98, fontfamily="DejaVu Sans")

    # 2 rows × 2 cols + narrow colorbar column
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[5, 5, 0.35],
        wspace=0.02,
        hspace=0.08,
    )

    for idx, (img, label) in enumerate(zip(view_images, view_labels)):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, aspect="auto")
        ax.set_title(label, fontsize=11, pad=5)
        ax.axis("off")

    # Shared colorbar
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax  = fig.add_subplot(gs[:, 2])
    cb   = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"\n✓  Saved → {Path(out_path).resolve()}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(config: dict = CONFIG):
    sep = "─" * 60
    print(f"{sep}\nStage 3: Brain Surface Visualisation\n{sep}")

    # ── 1. Load correlations ──────────────────────────────────────────────────
    print("\n[1/4]  Loading correlations …")
    correlations = load_correlations(config)
    print(
        f"\n       Aggregated ({config['aggregate']}): "
        f"mean r = {np.nanmean(correlations):.4f}  "
        f"max = {np.nanmax(correlations):.4f}  "
        f"min = {np.nanmin(correlations):.4f}"
    )

    # ── 2. Fetch atlas + fsaverage surface ────────────────────────────────────
    print("\n[2/4]  Fetching Schaefer atlas and fsaverage5 surface …")
    atlas     = datasets.fetch_atlas_schaefer_2018(
        n_rois=config["n_parcels"],
        yeo_networks=config["n_networks"],
        resolution_mm=1,
    )
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")   # lower-res, faster
    print(f"       Atlas: {config['n_parcels']} parcels, "
          f"{config['n_networks']} networks")

    # ── 3. Project to surface ─────────────────────────────────────────────────
    print("\n[3/4]  Projecting parcels onto surface …")
    surf_data = {}
    for hemi in ("left", "right"):
        surf_map, sulc, mesh = build_surface_map(
            correlations, atlas.maps, fsaverage, hemi
        )
        surf_data[hemi] = (surf_map, sulc, mesh)
        print(f"       {hemi.capitalize()}: {surf_map.shape[0]:,} vertices  "
              f"non-zero = {(surf_map != 0).sum():,}")

    # ── 4. Render views and compose figure ────────────────────────────────────
    print("\n[4/4]  Rendering views …")
    cmap = make_hot_cold_cmap()
    vmin, vmax = config["vmin"], config["vmax"]

    view_images = []
    for hemi, view, label in VIEWS:
        print(f"       {label} …", end=" ", flush=True)
        surf_map, sulc, mesh = surf_data[hemi]
        img = render_view(surf_map, sulc, mesh, hemi, view, cmap, vmin, vmax)
        view_images.append(img)
        print(f"✓  {img.shape[1]}×{img.shape[0]} px")

    view_labels = [v[2] for v in VIEWS]
    compose_figure(
        view_images, view_labels, cmap,
        vmin, vmax,
        config["plot_title"],
        config["dpi"],
        config["output_fig"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise per-parcel encoding correlations on the brain surface."
    )
    parser.add_argument(
        "--decoding_root",
        default=None,
        help="Path to the decoding/ output directory from encode.py",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help='Subject to plot, e.g. "sub-01", or "mean" to average all subjects.',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: all_prediction_correlations.png)",
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Colorscale minimum (default: -0.7)",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Colorscale maximum (default:  0.7)",
    )
    args = parser.parse_args()

    cfg = dict(CONFIG)                          # copy so we don't mutate the global
    if args.decoding_root is not None:
        cfg["decoding_root"] = args.decoding_root
    if args.subject is not None:
        cfg["aggregate"] = args.subject
    if args.output is not None:
        cfg["output_fig"] = args.output
    if args.vmin is not None:
        cfg["vmin"] = args.vmin
    if args.vmax is not None:
        cfg["vmax"] = args.vmax

    main(cfg)