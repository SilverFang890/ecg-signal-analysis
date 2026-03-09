from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


REPORT_COLS = [f"report_{i}" for i in range(18)]


# -----------------------------
# Core loading / label creation
# -----------------------------
def load_metadata(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load record list and machine measurements tables."""
    records = pd.read_csv(data_dir / "record_list.csv")
    mm = pd.read_csv(data_dir / "machine_measurements.csv", low_memory=False)
    return records, mm


def build_machine_report(mm: pd.DataFrame) -> pd.DataFrame:
    """Combine report_0 ... report_17 into a single lowercased machine_report."""
    missing = [c for c in REPORT_COLS if c not in mm.columns]
    if missing:
        raise ValueError(f"Missing expected report columns: {missing}")

    mm = mm.copy()
    mm["machine_report"] = (
        mm[REPORT_COLS]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.lower()
    )
    return mm


def create_labels(mm: pd.DataFrame) -> pd.DataFrame:
    """Create final boolean label columns from machine_report text."""
    mm = mm.copy()
    report = mm["machine_report"].fillna("").str.lower()

    mm["is_af"] = (
        report.str.contains(r"\batrial fibrillation\b", regex=True, na=False)
        & ~report.str.contains(r"\batrial flutter\b", regex=True, na=False)
    )

    mm["is_normal_strict"] = (
        (
            report.str.contains(r"\bnormal ecg\b", regex=True, na=False)
            | report.str.contains(r"\bwithin normal limits\b", regex=True, na=False)
        )
        & ~report.str.contains(
            r"\babnormal ecg\b|"
            r"\bborderline ecg\b|"
            r"\bexcept for\b|"
            r"\bprobable\b|"
            r"\bpossible\b|"
            r"\bpvc\b|"
            r"\bpac\b|"
            r"\bblock\b|"
            r"\binfarct\b|"
            r"\bst-t\b|"
            r"\bt wave\b|"
            r"\batrial\b|"
            r"\bventricular\b|"
            r"\btachycardia\b|"
            r"\bbradycardia\b|"
            r"\barrhythmia\b",
            regex=True,
            na=False,
        )
    )

    mm["is_clearly_abnormal"] = (
        report.str.contains(r"\babnormal ecg\b", regex=True, na=False)
        & ~mm["is_af"]
    )

    return mm


# -----------------------------
# Dataset construction
# -----------------------------
def build_meta(records: pd.DataFrame, mm: pd.DataFrame, waveform_root: str) -> pd.DataFrame:
    """Merge record paths with label columns and create a local waveform_path."""
    cols = [
        "subject_id",
        "study_id",
        "machine_report",
        "is_af",
        "is_normal_strict",
        "is_clearly_abnormal",
    ]
    meta = records.merge(mm[cols], on=["subject_id", "study_id"], how="inner")
    meta["waveform_path"] = str(Path(waveform_root)) + "/" + meta["path"].astype(str)
    return meta


def sanity_checks(meta: pd.DataFrame) -> None:
    overlap_af_normal = int((meta["is_af"] & meta["is_normal_strict"]).sum())
    overlap_af_abn = int((meta["is_af"] & meta["is_clearly_abnormal"]).sum())
    overlap_norm_abn = int((meta["is_normal_strict"] & meta["is_clearly_abnormal"]).sum())
    overlap_any = int((meta[["is_af", "is_normal_strict", "is_clearly_abnormal"]].sum(axis=1) > 1).sum())

    if any([overlap_af_normal, overlap_af_abn, overlap_norm_abn, overlap_any]):
        raise ValueError(
            "Label groups are not mutually exclusive. "
            f"AF∩Normal={overlap_af_normal}, "
            f"AF∩Abnormal={overlap_af_abn}, "
            f"Normal∩Abnormal={overlap_norm_abn}, "
            f"Any-overlap={overlap_any}"
        )


def build_task_datasets(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two binary task datasets.

    AF task: AF vs strict normal
    Normal/abnormal task: clearly abnormal vs strict normal
    """
    afib = meta[meta["is_af"] | meta["is_normal_strict"]].copy()
    afib["label"] = afib["is_af"].astype(int)

    norm = meta[meta["is_normal_strict"] | meta["is_clearly_abnormal"]].copy()
    norm["label"] = norm["is_clearly_abnormal"].astype(int)

    return afib, norm


# -----------------------------
# Subset construction
# -----------------------------
def sample_balanced_groups(
    meta: pd.DataFrame,
    n_af: int,
    n_normal: int,
    n_abnormal: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sample three disjoint label cohorts; normal can be reused across both tasks."""
    af_sub = meta.loc[meta["is_af"]].sample(n=n_af, random_state=random_state).copy()
    normal_sub = meta.loc[meta["is_normal_strict"]].sample(n=n_normal, random_state=random_state).copy()
    abnormal_sub = meta.loc[meta["is_clearly_abnormal"]].sample(n=n_abnormal, random_state=random_state).copy()
    return af_sub, normal_sub, abnormal_sub


def build_subset_tasks(
    af_sub: pd.DataFrame,
    normal_sub: pd.DataFrame,
    abnormal_sub: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    afib_sub = pd.concat([af_sub, normal_sub]).sort_index()
    afib_sub["label"] = afib_sub["is_af"].astype(int)
    
    norm_sub = pd.concat([abnormal_sub, normal_sub]).sort_index()
    norm_sub["label"] = norm_sub["is_clearly_abnormal"].astype(int)

    return afib_sub, norm_sub


# -----------------------------
# Outputs
# -----------------------------
def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def make_cohort_pie(meta: pd.DataFrame, out_path: Path) -> None:
    af = int(meta["is_af"].sum())
    normal = int(meta["is_normal_strict"].sum())
    abnormal = int(meta["is_clearly_abnormal"].sum())
    included = meta["is_af"] | meta["is_normal_strict"] | meta["is_clearly_abnormal"]
    excluded = int((~included).sum())

    labels = [
        "Strict Normal",
        "Clearly Abnormal",
        "Atrial Fibrillation",
        "Excluded / Ambiguous"
    ]
    sizes = [normal, abnormal, af, excluded]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    colors = [
        "#2ecc71",   # Normal
        "#faf023",   # Abnormal
        "#e74c3c",   # AF
        "#c8c8c8"    # Excluded
    ]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Distribution of ECG Cohorts")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_aria2_input(needed_paths: Iterable[str], out_file: Path, waveform_root: Path) -> None:
    """Create aria2 input with output paths rooted under waveform_root.

    This preserves the relative MIMIC path structure under data/raw_waveforms/.
    """
    base_url = "https://physionet.org/files/mimic-iv-ecg/1.0"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sorted_paths = sorted(set(needed_paths))
    for p in sorted_paths:
        (waveform_root / p).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for p in sorted_paths:
            for ext in (".hea", ".dat"):
                rel = f"{p}{ext}"
                url = f"{base_url}/{rel}"
                out_path = waveform_root / rel
                f.write(url + "\n")
                f.write(f"  out={out_path.as_posix()}\n")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labeled MIMIC-IV-ECG metadata and subset manifests.")
    parser.add_argument("--data-dir", type=Path, default=Path("../data/metadata"), help="Root data directory containing the CSVs.")
    parser.add_argument(
        "--waveform-root",
        type=Path,
        default=Path("../data/raw_waveforms"),
        help="Root directory where downloaded waveforms live / will be written.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("../data/processed"),
        help="Directory for processed parquet outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../data/figures"),
        help="Directory for optional plots.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000,
        help="Number of rows to sample from each of AF, strict normal, and clearly abnormal cohorts.",
    )
    parser.add_argument("--random-state", type=int, default=67)
    parser.add_argument("--make-plots", action="store_true", help="Save cohort pie chart.")
    parser.add_argument("--write-aria2", action="store_true", help="Write aria2 input file for the subset waveforms.")
    parser.add_argument(
        "--aria2-file",
        type=Path,
        default=Path("../utils/aria2_input.txt"),
        help="Output path for aria2 input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dirs([args.processed_dir, args.figures_dir, args.waveform_root])

    records, mm = load_metadata(args.data_dir)
    mm = build_machine_report(mm)
    mm = create_labels(mm)
    meta = build_meta(records, mm, waveform_root=args.waveform_root.as_posix())
    sanity_checks(meta)

    afib, norm = build_task_datasets(meta)

    af_sub, normal_sub, abnormal_sub = sample_balanced_groups(
        meta,
        n_af=args.subset_size,
        n_normal=args.subset_size,
        n_abnormal=args.subset_size,
        random_state=args.random_state,
    )
    afib_sub, norm_sub = build_subset_tasks(af_sub, normal_sub, abnormal_sub)

    # Save canonical outputs
    save_parquet(meta, args.processed_dir / "meta_labels.parquet")
    save_parquet(afib, args.processed_dir / "afib_task_metadata.parquet")
    save_parquet(norm, args.processed_dir / "norm_task_metadata.parquet")
    save_parquet(afib_sub, args.processed_dir / "afib_task_subset_metadata.parquet")
    save_parquet(norm_sub, args.processed_dir / "norm_task_subset_metadata.parquet")

    if args.make_plots:
        make_cohort_pie(meta, args.figures_dir / "ecg_cohort_pie.png")

    needed_paths = set(afib_sub["path"]).union(set(norm_sub["path"]))
    pd.Series(sorted(needed_paths), name="path").to_csv(
        args.processed_dir / "needed_paths_subset.csv", index=False
    )

    if args.write_aria2:
        write_aria2_input(needed_paths, args.aria2_file, args.waveform_root)

    # Console summary
    print("Saved:")
    print(f"  META:         {args.processed_dir / 'meta_labels.parquet'}")
    print(f"  AF full:      {args.processed_dir / 'afib_task_metadata.parquet'}")
    print(f"  Norm full:    {args.processed_dir / 'norm_task_metadata.parquet'}")
    print(f"  AF subset:    {args.processed_dir / 'afib_task_subset_metadata.parquet'}")
    print(f"  Norm subset:  {args.processed_dir / 'norm_task_subset_metadata.parquet'}")
    print(f"  Needed paths: {args.processed_dir / 'needed_paths_subset.csv'}")
    print()
    print("Counts:")
    print(meta[["is_af", "is_normal_strict", "is_clearly_abnormal"]].sum())
    print()
    print("Task counts:")
    print("AF full")
    print(afib["label"].value_counts().sort_index())
    print("\nNorm/abnormal full")
    print(norm["label"].value_counts().sort_index())
    print("\nAF subset")
    print(afib_sub["label"].value_counts().sort_index())
    print("\nNorm/abnormal subset")
    print(norm_sub["label"].value_counts().sort_index())
    print()
    print(f"Unique subset waveform paths: {len(needed_paths)}")

    if args.write_aria2:
        print()
        print("aria2 command:")
        print(
            f"aria2c -i {args.aria2_file.as_posix()} -j 32 "
            "--enable-http-pipelining=true --min-split-size=1M "
            "--continue=true --auto-file-renaming=false"
        )


if __name__ == "__main__":
    main()
