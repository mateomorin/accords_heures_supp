"""
Shared setup for the extraction benchmark scripts.

Imports the extraction package after putting `src/extraction` on `sys.path`, and
re-exports the pure context helpers so the scripts only import from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "extraction_benchmark"
EXTRACTION_DIR = ROOT / "src" / "extraction"
sys.path.insert(0, str(EXTRACTION_DIR))

load_dotenv(ROOT / ".env", override=True)

from context import (  # noqa: E402
    SelectedSections,
    build_context,
    build_outline,
    expand_subtrees,
    find_anchor_ids,
    preamble_ids,
    render_document,
)

from rag import load_parts_index, similarity_section_ids  # noqa: E402

# One benchmark run per full-document ceiling: same protocol, different sample.
CAPS = (20_000, 30_000, 40_000)

# Benchmark-only prompts: the production prompt minus every section-citation
# requirement (the flat arms receive chunks without `[section_id]` headers).
PROMPTS_DIR = BENCH_DIR / "prompts"

LEGACY_COLLECTION = "accords_heures_supp"
V2_COLLECTION = "accords_heures_supp_parts"

ARMS = ("full", "structured", "flat_legacy", "flat_v2")

COLUMNS = [
    "cap",
    "reference",
    "anneeSignature",
    "arm",
    "mode",
    "full_chars",
    "n_parts_doc",
    "context_chars",
    "n_context_parts",
    "tokens_in",
    "tokens_out",
    "selector_tokens_in",
    "selector_tokens_out",
    "context_s",
    "llm_s",
    "latency_s",
    "n_mentions",
    "n_regles",
    "n_rappels",
    "n_indeterminees",
    "n_with_rate",
    "n_paliers",
    "n_citations",
    "n_citations_grounded",
    "erreur",
    "mentions_json",
]


def samples_path(cap: int) -> Path:
    """Sample of the documents whose rendered length is at most `cap`."""
    return BENCH_DIR / f"samples_{cap // 1000}k.parquet"


def results_path(cap: int) -> Path:
    """One row per (document, arm) for a given ceiling."""
    return BENCH_DIR / f"results_{cap // 1000}k.jsonl"


def review_path(cap: int) -> Path:
    """Mentions that the matcher could not align, left for manual adjudication."""
    return BENCH_DIR / f"manual_review_{cap // 1000}k.md"


REPORT_PATH = BENCH_DIR / "REPORT.md"


__all__ = [
    "ARMS",
    "BENCH_DIR",
    "CAPS",
    "COLUMNS",
    "EXTRACTION_DIR",
    "LEGACY_COLLECTION",
    "PROMPTS_DIR",
    "REPORT_PATH",
    "ROOT",
    "SelectedSections",
    "V2_COLLECTION",
    "build_context",
    "build_outline",
    "build_services",
    "expand_subtrees",
    "find_anchor_ids",
    "load_config",
    "load_parts_index",
    "preamble_ids",
    "render_document",
    "results_path",
    "review_path",
    "samples_path",
    "similarity_section_ids",
]


def load_config(profile_type: str = "majoration", cap: int = 30_000) -> tuple[
    DictConfig, DictConfig
]:
    """Pipeline config + retrieval profile, with both budgets set to the tested ceiling."""
    cfg = OmegaConf.load(EXTRACTION_DIR / "configs" / "config.yaml")
    cfg.args.type = profile_type
    profile = OmegaConf.load(EXTRACTION_DIR / "configs" / "retrieval" / f"{profile_type}.yaml")
    # Every arm is bounded by the ceiling under test: a full reading would have been
    # allowed up to `cap`, so the structured and flat contexts get the same budget.
    profile.max_document_chars = cap
    profile.max_context_chars = cap
    cfg.vector_storage.qdrant_collection = profile.collection
    return cfg, profile


def build_services(cfg: DictConfig):
    """Import here so service clients are built after dotenv is loaded."""
    from services import ServiceContainer

    return ServiceContainer(cfg)
