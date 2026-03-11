"""Build final model-ready ECG datasets for AFIB and abnormal ECG tasks.

This script reproduces the non-exploratory dataset assembly from the notebook.
It:
1. Loads subset metadata, signal feature tables, and HuBERT embedding tables.
2. Filters metadata and embeddings to the ECGs that survived signal-feature QC.
3. Builds two final combined task datasets:
   - data/modeling_datasets/afib_features.parquet
   - data/modeling_datasets/abnorm_features.parquet

The final datasets retain metadata columns for traceability. Downstream modeling
code should drop identifier / metadata / label columns before fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


METADATA_COLS = [
    "subject_id",
    "study_id",
    "file_name",
    "ecg_time",
    "path",
    "machine_report",
    "is_af",
    "is_normal_strict",
    "is_clearly_abnormal",
    "waveform_path",
    "label",
]

FEATURE_DUPLICATE_DROP_COLS = ["subject_id_dup", "study_id_dup", "path_dup", "label_dup"]
EMBED_DUPLICATE_DROP_COLS = ["file_name_dup", "embedding"]
UNIQUE_KEY = "waveform_path"


def build_task_dataset(meta: pd.DataFrame, feat: pd.DataFrame, emb: pd.DataFrame) -> pd.DataFrame:
    """Build a single final combined dataset for one task."""
    valid_waveforms = set(feat[UNIQUE_KEY])

    # Restrict all tables to ECGs that survived classical feature extraction.
    meta = meta[meta[UNIQUE_KEY].isin(valid_waveforms)].copy()
    emb = emb[emb[UNIQUE_KEY].isin(valid_waveforms)].copy()

    # Merge metadata into signal feature table, then remove duplicated metadata columns.
    signal = meta.merge(feat, on=UNIQUE_KEY, how="inner", suffixes=("", "_dup"))
    drop_signal = [c for c in FEATURE_DUPLICATE_DROP_COLS if c in signal.columns]
    signal = signal.drop(columns=drop_signal).copy()

    # Merge signal+metadata with embeddings. Drop nested embedding list column and duplicate file_name.
    emb_for_merge = emb.drop(columns=[c for c in ["file_name", "embedding"] if c in emb.columns]).copy()
    combined = signal.merge(emb_for_merge, on=UNIQUE_KEY, how="inner")

    # Stable ordering: metadata first, then signal features, then emb_* columns.
    emb_cols = [c for c in combined.columns if c.startswith("emb_")]
    signal_feature_cols = [
        c for c in signal.columns
        if c not in METADATA_COLS and c != UNIQUE_KEY
    ]
    ordered_cols = [
        *[c for c in METADATA_COLS if c in combined.columns],
        *[c for c in signal_feature_cols if c in combined.columns],
        *emb_cols,
    ]
    ordered_cols = list(dict.fromkeys(ordered_cols))
    combined = combined[ordered_cols].copy()

    # Sanity check: one row per waveform.
    combined = combined.drop_duplicates(subset=[UNIQUE_KEY], keep="first").copy()
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final AFIB and abnormal ECG datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing data/processed. Defaults to parent of this script.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Optional explicit path to data/processed.",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=None,
        help="Optional explicit path to output modeling_datasets directory.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.processed_dir is not None:
        processed = args.processed_dir.resolve()
        project_root = processed.parent.parent
    else:
        if args.project_root is not None:
            project_root = args.project_root.resolve()
        else:
            cwd = Path.cwd().resolve()
            project_root = cwd.parent if cwd.name == "src" else cwd
        processed = project_root / "data" / "processed"

    final_dir = args.final_dir.resolve() if args.final_dir is not None else project_root / "data" / "modeling_datasets"
    return project_root, processed, final_dir


def main() -> None:
    args = parse_args()
    project_root, processed, final_dir = resolve_paths(args)
    emb_dir = processed / "embeddings"

    # Load subset tables.
    afib_meta = pd.read_parquet(processed / "afib_subset_metadata.parquet")
    norm_meta = pd.read_parquet(processed / "norm_subset_metadata.parquet")
    afib_feat = pd.read_parquet(processed / "afib_subset_features.parquet")
    norm_feat = pd.read_parquet(processed / "norm_subset_features.parquet")
    afib_emb = pd.read_parquet(emb_dir / "afib_subset_embeddings_exp.parquet")
    norm_emb = pd.read_parquet(emb_dir / "norm_subset_embeddings_exp.parquet")

    # Build final task datasets.
    afib_features = build_task_dataset(afib_meta, afib_feat, afib_emb)
    abnorm_features = build_task_dataset(norm_meta, norm_feat, norm_emb)

    final_dir.mkdir(parents=True, exist_ok=True)
    afib_out = final_dir / "afib_features.parquet"
    abnorm_out = final_dir / "abnorm_features.parquet"
    afib_features.to_parquet(afib_out, index=False)
    abnorm_features.to_parquet(abnorm_out, index=False)

    print(f"Project root: {project_root}")
    print(f"Processed dir: {processed}")
    print(f"Final dir: {final_dir}")
    print(f"AFIB final dataset: {afib_features.shape} -> {afib_out}")
    print(f"Abnormal final dataset: {abnorm_features.shape} -> {abnorm_out}")
    print("\nReminder: drop identifier, metadata, and label columns before model training.")


if __name__ == "__main__":
    main()
