from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sps
from joblib import Parallel, delayed
from scipy.stats import entropy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.load_ecg import load_ecg_with_metadata


def bandpass_filter_ecg(signal: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = sps.butter(order, [low / nyq, high / nyq], btype="band")
    return sps.filtfilt(b, a, signal, axis=0)


def get_lead(signal: np.ndarray, sig_names: List[str], lead_name: str = "II") -> np.ndarray:
    if lead_name not in sig_names:
        raise ValueError(f"Lead {lead_name} not found. Available leads: {sig_names}")
    idx = sig_names.index(lead_name)
    return signal[:, idx]


def detect_r_peaks_lead2(lead_signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    qrs_band = bandpass_filter_ecg(lead_signal[:, None], fs, low=5.0, high=20.0, order=2).squeeze()
    squared = qrs_band ** 2
    win = max(1, int(0.15 * fs))
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")
    distance = int(0.25 * fs)
    height = np.percentile(integrated, 90)
    peaks, _ = sps.find_peaks(integrated, distance=distance, height=height)
    return peaks, integrated


def compute_rr_features(peaks: np.ndarray, fs: int) -> Dict[str, float]:
    if len(peaks) < 2:
        return {
            "n_r_peaks": len(peaks),
            "rr_mean": np.nan,
            "rr_std": np.nan,
            "rmssd": np.nan,
            "heart_rate_bpm": np.nan,
        }

    rr = np.diff(peaks) / fs
    rr_diff = np.diff(rr)

    return {
        "n_r_peaks": len(peaks),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rmssd": float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else np.nan,
        "heart_rate_bpm": float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
    }


def compute_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def compute_energy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(np.square(x)))


def compute_peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.ptp(x))


def spectral_entropy(psd: np.ndarray) -> float:
    psd = np.asarray(psd, dtype=np.float64)
    psd_sum = psd.sum()
    if psd_sum <= 0:
        return np.nan
    p = psd / psd_sum
    return float(entropy(p))


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    denom = np.sum(psd)
    if denom <= 0:
        return np.nan
    return float(np.sum(freqs * psd) / denom)


def bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_spectral_features(x: np.ndarray, fs: int, prefix: str = "signal") -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    freqs, psd = sps.welch(x, fs=fs, nperseg=min(1024, len(x)))

    if len(psd) == 0 or np.all(psd == 0):
        return {
            f"{prefix}_dom_freq": np.nan,
            f"{prefix}_spec_entropy": np.nan,
            f"{prefix}_spec_centroid": np.nan,
            f"{prefix}_bp_0_5_5": np.nan,
            f"{prefix}_bp_5_15": np.nan,
            f"{prefix}_bp_15_40": np.nan,
        }

    return {
        f"{prefix}_dom_freq": float(freqs[np.argmax(psd)]),
        f"{prefix}_spec_entropy": spectral_entropy(psd),
        f"{prefix}_spec_centroid": spectral_centroid(freqs, psd),
        f"{prefix}_bp_0_5_5": bandpower(freqs, psd, 0.5, 5.0),
        f"{prefix}_bp_5_15": bandpower(freqs, psd, 5.0, 15.0),
        f"{prefix}_bp_15_40": bandpower(freqs, psd, 15.0, 40.0),
    }


def hjorth_parameters(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx) if len(dx) > 0 else np.nan
    var_ddx = np.var(ddx) if len(ddx) > 0 else np.nan

    activity = float(var_x)

    if var_x <= 0 or np.isnan(var_dx):
        return {
            "hjorth_activity": activity,
            "hjorth_mobility": np.nan,
            "hjorth_complexity": np.nan,
        }

    mobility = float(np.sqrt(var_dx / var_x))

    if var_dx <= 0 or np.isnan(var_ddx) or mobility <= 0:
        return {
            "hjorth_activity": activity,
            "hjorth_mobility": mobility,
            "hjorth_complexity": np.nan,
        }

    complexity = float(np.sqrt(var_ddx / var_dx) / mobility)
    return {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def aggregate_lead_features(values: Iterable[float], prefix: str) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
    }


def compute_multilead_basic_features(ecg_signal: np.ndarray) -> Dict[str, float]:
    lead_means, lead_stds, lead_mins, lead_maxs = [], [], [], []
    lead_ptp, lead_rms, lead_energy = [], [], []

    for i in range(ecg_signal.shape[1]):
        x = ecg_signal[:, i]
        lead_means.append(np.mean(x))
        lead_stds.append(np.std(x))
        lead_mins.append(np.min(x))
        lead_maxs.append(np.max(x))
        lead_ptp.append(compute_peak_to_peak(x))
        lead_rms.append(compute_rms(x))
        lead_energy.append(compute_energy(x))

    features = {}
    features.update(aggregate_lead_features(lead_means, "lead_mean"))
    features.update(aggregate_lead_features(lead_stds, "lead_std"))
    features.update(aggregate_lead_features(lead_mins, "lead_min"))
    features.update(aggregate_lead_features(lead_maxs, "lead_max"))
    features.update(aggregate_lead_features(lead_ptp, "lead_ptp"))
    features.update(aggregate_lead_features(lead_rms, "lead_rms"))
    features.update(aggregate_lead_features(lead_energy, "lead_energy"))
    return features


def compute_multilead_spectral_features(ecg_signal: np.ndarray, fs: int) -> Dict[str, float]:
    dom_freqs, spec_entropies, spec_centroids = [], [], []
    bp_low, bp_mid, bp_high = [], [], []

    for i in range(ecg_signal.shape[1]):
        x = ecg_signal[:, i]
        feats = compute_spectral_features(x, fs, prefix=f"lead{i}")
        dom_freqs.append(feats[f"lead{i}_dom_freq"])
        spec_entropies.append(feats[f"lead{i}_spec_entropy"])
        spec_centroids.append(feats[f"lead{i}_spec_centroid"])
        bp_low.append(feats[f"lead{i}_bp_0_5_5"])
        bp_mid.append(feats[f"lead{i}_bp_5_15"])
        bp_high.append(feats[f"lead{i}_bp_15_40"])

    features = {}
    features.update(aggregate_lead_features(dom_freqs, "dom_freq"))
    features.update(aggregate_lead_features(spec_entropies, "spec_entropy"))
    features.update(aggregate_lead_features(spec_centroids, "spec_centroid"))
    features.update(aggregate_lead_features(bp_low, "bp_0_5_5"))
    features.update(aggregate_lead_features(bp_mid, "bp_5_15"))
    features.update(aggregate_lead_features(bp_high, "bp_15_40"))
    return features


def extract_ecg_features(signal: np.ndarray, fs: int, sig_names: List[str]) -> Dict[str, float]:
    signal_filt = bandpass_filter_ecg(signal, fs)

    features = {}
    features.update(compute_multilead_basic_features(signal_filt))
    features.update(compute_multilead_spectral_features(signal_filt, fs))

    lead2 = get_lead(signal_filt, sig_names, lead_name="II")
    peaks, _ = detect_r_peaks_lead2(lead2, fs)
    rr_features = compute_rr_features(peaks, fs)

    rr = np.diff(peaks) / fs if len(peaks) >= 2 else np.array([])
    rr_range = float(np.max(rr) - np.min(rr)) if len(rr) > 0 else np.nan

    features.update(hjorth_parameters(lead2))
    features.update(rr_features)
    features["rr_range"] = rr_range

    return features


def process_ecg_row(row: pd.Series) -> Dict[str, object]:
    waveform_path = row["waveform_path"]
    signal, fs, sig_names, units = load_ecg_with_metadata(waveform_path)
    features = extract_ecg_features(signal, fs, sig_names)
    features["signal_length"] = len(signal)

    output = {
        "subject_id": row["subject_id"],
        "study_id": row["study_id"],
        "path": row["path"],
        "waveform_path": row["waveform_path"],
        "label": row["label"],
    }
    output.update(features)
    return output


def process_ecg_row_safe(row: pd.Series) -> Dict[str, object]:
    try:
        return {"ok": True, "result": process_ecg_row(row)}
    except Exception as e:
        return {
            "ok": False,
            "error": {
                "subject_id": row.get("subject_id", None),
                "study_id": row.get("study_id", None),
                "waveform_path": row.get("waveform_path", None),
                "label": row.get("label", None),
                "error": str(e),
            },
        }


def build_feature_table_parallel(df: pd.DataFrame, n_jobs: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outputs = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_ecg_row_safe)(row) for _, row in df.iterrows()
    )
    results = [x["result"] for x in outputs if x["ok"]]
    errors = [x["error"] for x in outputs if not x["ok"]]
    return pd.DataFrame(results), pd.DataFrame(errors)


def choose_artifact_leads(signal: np.ndarray, top_k: int = 3) -> List[int]:
    ptp = np.ptp(signal, axis=0)
    order = np.argsort(ptp)[::-1]
    return sorted(order[:top_k].tolist())


def plot_ecg(signal: np.ndarray, fs: int, leads: List[int], seconds: int = 10, title: str = "ECG preview", save_path: Path | None = None) -> None:
    n_samples = min(len(signal), seconds * fs)
    t = np.arange(n_samples) / fs

    plt.figure(figsize=(12, 5))
    for lead in leads:
        plt.plot(t, signal[:n_samples, lead], label=f"Lead {lead}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_invalid_ecg_figures(features_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    invalid_df = features_df[features_df["rr_mean"].isna()].copy()
    invalid_records = []

    for _, row in invalid_df.iterrows():
        signal, fs, sig_names, units = load_ecg_with_metadata(row["waveform_path"])
        leads = choose_artifact_leads(signal, top_k=3)
        fname = f"study_{row['study_id']}_label_{row['label']}.png"
        save_path = out_dir / fname
        title = f"Invalid ECG | study_id={row['study_id']} | label={row['label']} | leads={leads}"
        plot_ecg(signal, fs, leads=leads, seconds=10, title=title, save_path=save_path)

        invalid_records.append({
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "label": row["label"],
            "waveform_path": row["waveform_path"],
            "saved_plot": str(save_path),
            "chosen_leads": str(leads),
        })

    return pd.DataFrame(invalid_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ECG signal processing feature tables.")
    parser.add_argument("--afib-input", default="../data/processed/afib_subset_metadata.parquet")
    parser.add_argument("--norm-input", default="../data/processed/norm_subset_metadata.parquet")
    parser.add_argument("--afib-output", default="../data/processed/afib_subset_features.parquet")
    parser.add_argument("--norm-output", default="../data/processed/norm_subset_features.parquet")
    parser.add_argument("--figures-dir", default="../data/figures/invalid_ecg_records")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() // 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    afib = pd.read_parquet(args.afib_input)
    norm = pd.read_parquet(args.norm_input)

    print(f"Loaded AF subset: {afib.shape}")
    print(f"Loaded Normal/Abnormal subset: {norm.shape}")
    print(f"Using n_jobs={args.n_jobs}")

    features_af_df, errors_af_df = build_feature_table_parallel(afib, n_jobs=args.n_jobs)
    print("AF features shape:", features_af_df.shape)
    print("AF errors:", len(errors_af_df))

    save_invalid_ecg_figures(features_af_df, Path(args.figures_dir))
    
    features_af_clean = features_af_df.dropna(subset=["rr_mean"]).reset_index(drop=True)
    features_af_clean.to_parquet(args.afib_output, index=False)
    print("AF invalid RR rows dropped:", len(features_af_df) - len(features_af_clean))

    features_norm_df, errors_norm_df = build_feature_table_parallel(norm, n_jobs=args.n_jobs)
    print("Norm features shape:", features_norm_df.shape)
    print("Norm errors:", len(errors_norm_df))

    save_invalid_ecg_figures(features_norm_df, Path(args.figures_dir))

    features_norm_clean = features_norm_df.dropna(subset=["rr_mean"]).reset_index(drop=True)
    features_norm_clean.to_parquet(args.norm_output, index=False)
    print("Norm invalid RR rows dropped:", len(features_norm_df) - len(features_norm_clean))

    print("Done.")
    print(f"Saved AF features to: {args.afib_output}")
    print(f"Saved Norm/Abnormal features to: {args.norm_output}")
    print(f"Saved invalid ECG figures under: {args.figures_dir}")


if __name__ == "__main__":
    main()
