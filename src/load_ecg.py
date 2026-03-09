from pathlib import Path
from typing import Tuple, Union

import numpy as np
import wfdb

PathLike = Union[str, Path]

def resolve_record_path(record_path: PathLike) -> str:
    """
    Resolve a WFDB record path.

    Accepts either:
    - a path without extension, e.g. data/raw_waveforms/files/.../40689238
    - a .hea path
    - a .dat path

    Returns the record stem path as a string, which is what wfdb.rdrecord expects.
    """
    p = Path(record_path)

    if p.suffix in {".hea", ".dat"}:
        p = p.with_suffix("")

    return str(p)


def load_ecg(record_path: PathLike) -> Tuple[np.ndarray, int]:
    """
    Load a single ECG waveform from a WFDB record.

    Parameters
    ----------
    record_path : str or Path
        Path to the record stem, or to a .hea/.dat file.

    Returns
    -------
    signal : np.ndarray
        ECG waveform array of shape (n_samples, n_leads), typically (5000, 12).
    fs : int
        Sampling frequency.
    """
    record_stem = resolve_record_path(record_path)
    record = wfdb.rdrecord(record_stem)

    signal = np.asarray(record.p_signal, dtype=np.float32)
    fs = int(record.fs)

    return signal, fs


def load_ecg_with_metadata(record_path: PathLike):
    """
    Load a single ECG waveform and return useful WFDB metadata.

    Returns
    -------
    signal : np.ndarray
        ECG waveform array of shape (n_samples, n_leads).
    fs : int
        Sampling frequency.
    sig_names : list[str]
        Lead names.
    units : list[str]
        Signal units.
    """
    record_stem = resolve_record_path(record_path)
    record = wfdb.rdrecord(record_stem)

    signal = np.asarray(record.p_signal, dtype=np.float32)
    fs = int(record.fs)
    sig_names = list(record.sig_name)
    units = list(record.units)

    return signal, fs, sig_names, units


if __name__ == "__main__":
    example = "../data/raw_waveforms/files/p1000/p10001860/s45808859/45808859"

    try:
        signal, fs, sig_names, units = load_ecg_with_metadata(example)
        print("Loaded ECG successfully.")
        print("Shape:", signal.shape)
        print("Sampling frequency:", fs)
        print("Lead names:", sig_names)
        print("Units:", units)
    except Exception as e:
        print(f"Failed to load ECG: {e}")