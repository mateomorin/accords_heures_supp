"""
Fetch the `population_concernee` embeddings from Qdrant, reduce them to 2D with UMAP
and plot the resulting topic cloud.

UMAP is always computed on ALL points. Produces (in --output_dir):
    - population_concernee_umap.parquet : UMAP coordinates + all metadata (all points)
    - umap_heures_heatmap.png           : global hexbin heatmap, cell colour = mean nombre_heures_fixe
    - umap_by_year.png                  : per-year panel grid, colour = type_substitution
                                           (single marker size, no size legend; shared axes)
"""
import argparse
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Upper clip so a few absurd `nombre_heures_fixe` values don't blow the heatmap colormap
HEURES_CLIP = 600

# Fixed, consistent palette for the type_substitution categories (stable across panels)
TYPE_PAL = {
    "fixation_directe": "#4C72B0",
    "remplacement_convention_collective": "#DD8452",
    "remplacement_accord_anterieur": "#55A868",
    "remplacement_contingent_legal": "#C44E52",
}


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def fetch_all(collection: str) -> pd.DataFrame:
    """Scroll the whole collection (vectors + payload) into a DataFrame."""
    points = []
    offset = None
    while True:
        res, offset = config.qdrant_client.scroll(
            collection_name=collection,
            limit=5000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        points.extend(res)
        if offset is None:
            break

    records = []
    for p in points:
        payload = p.payload or {}
        meta = payload.get("metadata", {}) or {}
        rec = {"id": str(p.id), "page_content": payload.get("page_content")}
        rec.update(meta)
        records.append(rec)

    df = pd.DataFrame(records)
    df.attrs["vectors"] = np.asarray([p.vector for p in points], dtype=np.float32)
    logger.info(f"Fetched {len(df)} points from '{collection}'.")
    return df


# --------------------------------------------------------------------------- #
# UMAP
# --------------------------------------------------------------------------- #
def run_umap(vectors: np.ndarray, n_neighbors: int, min_dist: float, metric: str, random_state: int) -> np.ndarray:
    logger.info(
        f"Running UMAP (n={len(vectors)}, n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})..."
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def heatmap_heures(df: pd.DataFrame, gridsize: int, output_dir: Path) -> None:
    """Global hexbin heatmap: each cell is coloured by the mean nombre_heures_fixe of its points."""
    c = df["nombre_heures_fixe"].clip(upper=HEURES_CLIP)

    fig, ax = plt.subplots(figsize=(11, 9))
    hb = ax.hexbin(
        df["umap_1"],
        df["umap_2"],
        C=c,
        gridsize=gridsize,
        cmap="YlOrRd",
        reduce_C_function=np.nanmean,
        mincnt=1,
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(f"nombre_heures_fixe (moyenne par cellule, clip à {HEURES_CLIP} h)", fontsize=11)

    ax.set_title("UMAP des populations concernées — heatmap de nombre_heures_fixe", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_dir / "umap_heures_heatmap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {out}")


def scatter_by_year(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Per-year panel grid, each panel coloured by type_substitution.
    All panels share the same (global) UMAP axes so they are directly comparable.
    Single marker size, no size legend, one shared colour legend.
    """
    years = sorted(df["anneeSignature"].dropna().unique())
    n = len(years)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    # Shared limits so every panel sits in the same coordinate frame
    x0, x1 = df["umap_1"].min(), df["umap_1"].max()
    y0, y1 = df["umap_2"].min(), df["umap_2"].max()
    mx = (x1 - x0) * 0.04 or 1.0
    my = (y1 - y0) * 0.04 or 1.0
    xlim = (x0 - mx, x1 + mx)
    ylim = (y0 - my, y1 + my)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, year in enumerate(years):
        ax = axes[i]
        sub = df[pd.col("anneeSignature") == year]
        sns.scatterplot(
            data=sub,
            x="umap_1",
            y="umap_2",
            hue="type_substitution",
            palette=TYPE_PAL,
            s=14,
            alpha=0.6,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{year}  (n={len(sub)})", fontsize=12, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused panels
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # One shared colour legend for the whole figure (bottom)
    vals = [v for v in sorted(df["type_substitution"].dropna().unique())]
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=TYPE_PAL.get(v, "gray"), markeredgecolor="none", label=v)
        for v in vals
    ]
    fig.subplots_adjust(bottom=0.10)
    fig.legend(handles=handles, loc="lower center", ncol=min(4, max(1, len(vals))), frameon=True, fontsize=10)

    fig.suptitle("UMAP des populations concernées — par année, coloré par type de substitution",
                 fontsize=15, fontweight="bold", y=1.0)

    out = output_dir / "umap_by_year.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Reduce population_concernee embeddings with UMAP and plot the 2D topic cloud."
    )
    parser.add_argument("--collection", type=str, default=config.COLLECTION, help="Qdrant collection name.")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--metric", type=str, default="cosine", help="UMAP distance metric.")
    parser.add_argument("--random_state", type=int, default=42, help="UMAP random state (reproducibility).")
    parser.add_argument("--gridsize", type=int, default=50, help="Hexbin gridsize for the hours heatmap.")
    parser.add_argument("--output_dir", type=str, default=str(config.DEFAULT_OUTPUT_DIR),
                        help="Where to write the UMAP parquet and PNGs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_all(args.collection)
    if df.empty:
        logger.warning("Collection is empty; run embed_population.py first.")
        return

    coords = run_umap(
        df.attrs["vectors"],
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    df["umap_1"] = coords[:, 0]
    df["umap_2"] = coords[:, 1]
    del df.attrs["vectors"]

    parquet_path = output_dir / "population_concernee_umap.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {parquet_path} ({len(df)} rows, {len(df.columns)} cols)")

    heatmap_heures(df, gridsize=args.gridsize, output_dir=output_dir)
    scatter_by_year(df, output_dir=output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
