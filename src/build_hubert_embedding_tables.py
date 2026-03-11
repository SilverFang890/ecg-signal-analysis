"""Extract HuBERT-ECG embeddings for AFIB and abnormal-vs-normal subsets.

This script reproduces the non-exploratory outputs from the notebook:
- data/processed/embeddings/afib_subset_embeddings.parquet
- data/processed/embeddings/norm_subset_embeddings.parquet
- data/processed/embeddings/afib_subset_embeddings_exp.parquet
- data/processed/embeddings/norm_subset_embeddings_exp.parquet

Expected inputs:
- data/processed/afib_subset_metadata.parquet
- data/processed/norm_subset_metadata.parquet
- waveform_path column in each metadata parquet
- src/load_ecg.py exposing load_ecg(path)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from tqdm.auto import tqdm
from transformers import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.load_ecg import load_ecg


AFIB_META_PATH = PROJECT_ROOT / "data" / "processed" / "afib_subset_metadata.parquet"
NORM_META_PATH = PROJECT_ROOT / "data" / "processed" / "norm_subset_metadata.parquet"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HuBERT-ECG embeddings for task subset metadata tables."
    )
    parser.add_argument(
        "--task",
        choices=["afib", "norm", "both"],
        default="both",
        help="Which task subset to process.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--model-name",
        default="Edoardo-BS/hubert-ecg-small",
        help="Hugging Face model identifier for HuBERT-ECG.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Checkpoint frequency for parquet writes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet instead of resuming.",
    )
    return parser.parse_args()


def _zscore_per_lead(x_5s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize one 5-second clip leadwise.

    Parameters
    ----------
    x_5s : np.ndarray
        Array with shape (500, 12) after resampling to 100 Hz.

    Returns
    -------
    np.ndarray
        Normalized clip with shape (12, 500), dtype float32.
    """
    x_5s = np.nan_to_num(x_5s, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x_5s.mean(axis=0, keepdims=True)
    sd = x_5s.std(axis=0, keepdims=True) + eps
    x_norm = (x_5s - mu) / sd
    return x_norm.T.astype(np.float32)



def preprocess_for_hubert(x_10s: np.ndarray, fs: float) -> np.ndarray:
    """Convert a raw 10-second ECG into two HuBERT-ready halves.

    Parameters
    ----------
    x_10s : np.ndarray
        Raw ECG with shape (time, leads), expected around (5000, 12).
    fs : float
        Input sampling rate.

    Returns
    -------
    np.ndarray
        Array with shape (2, 12, 500), representing two contiguous 5-second
        clips resampled to 100 Hz and z-scored per lead.
    """
    target_fs = 100.0
    resamp_n = int(round(x_10s.shape[0] * target_fs / fs))
    resamp_x = resample(x_10s, resamp_n, axis=0).astype(np.float32)

    crop_len = int(5 * target_fs)  # 500 samples
    if resamp_x.shape[0] < 2 * crop_len:
        raise ValueError(
            f"Resampled ECG too short for two 5-second clips: {resamp_x.shape}"
        )

    x_5s_a = resamp_x[:crop_len]
    x_5s_b = resamp_x[crop_len : 2 * crop_len]

    clip_a = _zscore_per_lead(x_5s_a)
    clip_b = _zscore_per_lead(x_5s_b)
    return np.stack([clip_a, clip_b], axis=0)



def load_hubert_ecg(model_name: str, device: str = "cpu") -> AutoModel:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return model


@torch.no_grad()
def hubert_encode(x_half: np.ndarray, hubert_model: AutoModel, device: str = "cpu") -> np.ndarray:
    """Encode one 5-second HuBERT-ready ECG clip.

    Parameters
    ----------
    x_half : np.ndarray
        Shape (12, 500).

    Returns
    -------
    np.ndarray
        1D embedding vector.
    """
    x = torch.tensor(
        x_half.reshape(1, -1),
        dtype=torch.float32,
        device=device,
    )
    out = hubert_model(x)
    z = out.last_hidden_state.mean(dim=1).squeeze(0)
    return z.detach().cpu().numpy().astype(np.float32)



def encode_ecg_record(row: pd.Series, encoder: AutoModel, device: str = "cpu") -> np.ndarray:
    signal, fs = load_ecg(row["waveform_path"])
    x_halves = preprocess_for_hubert(signal, fs)
    emb1 = hubert_encode(x_halves[0], encoder, device)
    emb2 = hubert_encode(x_halves[1], encoder, device)
    return ((emb1 + emb2) / 2.0).astype(np.float32)



def flatten_embedding(x: object) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    return arr.reshape(-1)



def add_expanded_embedding_columns(
    df: pd.DataFrame, embedding_col: str = "embedding", prefix: str = "emb_"
) -> pd.DataFrame:
    emb_matrix = np.vstack(df[embedding_col].apply(flatten_embedding).values)
    emb_cols = [f"{prefix}{i}" for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    return pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)



def load_subset_metadata(task: str) -> pd.DataFrame:
    if task == "afib":
        return pd.read_parquet(AFIB_META_PATH)
    if task == "norm":
        return pd.read_parquet(NORM_META_PATH)
    raise ValueError(f"Unknown task: {task}")



def output_paths(task: str) -> tuple[Path, Path]:
    base = EMBEDDING_DIR / f"{task}_subset_embeddings.parquet"
    expanded = EMBEDDING_DIR / f"{task}_subset_embeddings_exp.parquet"
    return base, expanded



def _buffer_to_parquet(buffer: list[dict], out_path: Path) -> None:
    if not buffer:
        return

    chunk = pd.DataFrame(buffer)
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        merged = pd.concat([prev, chunk], ignore_index=True)
        merged = merged.drop_duplicates(subset=["waveform_path"], keep="last")
        merged.to_parquet(out_path, index=False)
    else:
        chunk.to_parquet(out_path, index=False)



def build_embeddings_table(
    df: pd.DataFrame,
    out_path: Path,
    model: AutoModel,
    device: str = "cpu",
    save_every: int = 50,
    overwrite: bool = False,
) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and out_path.exists():
        out_path.unlink()

    done_paths: set[str] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if "waveform_path" in existing.columns:
            done_paths = set(existing["waveform_path"].astype(str).tolist())

    if done_paths:
        todo = df[~df["waveform_path"].astype(str).isin(done_paths)].copy()
    else:
        todo = df.copy()

    buffer: list[dict] = []

    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"Encoding {out_path.stem}"):
        record = {
            "file_name": row.get("file_name", None),
            "waveform_path": row["waveform_path"],
        }

        try:
            emb = encode_ecg_record(row, encoder=model, device=device)
            record["embedding"] = emb.tolist()
            record["embedding_dim"] = int(len(emb))
            record["status"] = "ok"
            record["error"] = None
        except Exception as exc:  # noqa: BLE001
            record["embedding"] = None
            record["embedding_dim"] = None
            record["status"] = "failed"
            record["error"] = str(exc)

        buffer.append(record)

        if len(buffer) >= save_every:
            _buffer_to_parquet(buffer, out_path)
            print(f"Saved {len(buffer)} rows to {out_path.name}")
            buffer = []

    if buffer:
        _buffer_to_parquet(buffer, out_path)
        print(f"Saved final {len(buffer)} rows to {out_path.name}")

    result = pd.read_parquet(out_path)
    print(f"Done. Final output: {out_path}")
    return result



def build_expanded_embeddings_table(base_df: pd.DataFrame, expanded_out_path: Path) -> pd.DataFrame:
    keep = [col for col in ["file_name", "waveform_path", "embedding"] if col in base_df.columns]
    expanded_input = base_df[keep].copy()
    expanded_df = add_expanded_embedding_columns(expanded_input, embedding_col="embedding", prefix="emb_")
    expanded_out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_df.to_parquet(expanded_out_path, index=False)
    print(f"Saved expanded embeddings to {expanded_out_path}")
    return expanded_df



def process_task(
    task: str,
    model: AutoModel,
    device: str,
    save_every: int,
    overwrite: bool,
) -> None:
    df = load_subset_metadata(task)
    out_path, expanded_out_path = output_paths(task)

    embeddings_df = build_embeddings_table(
        df=df,
        out_path=out_path,
        model=model,
        device=device,
        save_every=save_every,
        overwrite=overwrite,
    )

    ok_df = embeddings_df[embeddings_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise RuntimeError(f"No successful embeddings found for task '{task}'.")

    build_expanded_embeddings_table(ok_df, expanded_out_path)



def main() -> None:
    args = parse_args()
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    tasks: Iterable[str]
    if args.task == "both":
        tasks = ("afib", "norm")
    else:
        tasks = (args.task,)

    model = load_hubert_ecg(model_name=args.model_name, device=args.device)
    for task in tasks:
        process_task(
            task=task,
            model=model,
            device=args.device,
            save_every=args.save_every,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
