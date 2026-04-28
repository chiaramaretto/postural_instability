"""Preprocessing utilities per GRUCNN: EMD denoise, downsample, scaling, padding, augmentation."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import signal
from PyEMD import EMD


def emd_denoise(sequence: np.ndarray, n_imf: int = 7) -> np.ndarray:
    """Applica EMD e ricostruisce usando i primi `n_imf` IMF (low-frequency).

    sequence: array shape (timesteps, channels) or (timesteps,)
    """
    seq = np.asarray(sequence)
    if seq.ndim == 1:
        imf = EMD().emd(seq)
        if imf.size == 0:
            return seq
        n = min(n_imf, imf.shape[0])
        return np.sum(imf[:n, :], axis=0)

    # multichannel: apply per channel
    out = np.zeros_like(seq, dtype=float)
    for c in range(seq.shape[1]):
        channel = seq[:, c]
        imf = EMD().emd(channel)
        if imf.size == 0:
            out[:, c] = channel
        else:
            n = min(n_imf, imf.shape[0])
            out[:, c] = np.sum(imf[:n, :], axis=0)
    return out


def downsample_sequence(sequence: np.ndarray, orig_fs: int = 100, target_fs: int = 25) -> np.ndarray:
    """Downsample using decimation preserving shape (timesteps, channels)."""
    if target_fs >= orig_fs:
        return sequence
    q = orig_fs // target_fs
    if q <= 1:
        return sequence
    # use decimate along axis 0 per channel
    if sequence.ndim == 1:
        return signal.decimate(sequence, q, ftype='iir', zero_phase=True)
    return signal.decimate(sequence, q, axis=0, ftype='iir', zero_phase=True)


def minmax_scale(sequence: np.ndarray, feature_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    lo, hi = feature_range
    seq = np.asarray(sequence, dtype=float)
    minv = np.nanmin(seq, axis=0)
    maxv = np.nanmax(seq, axis=0)
    denom = (maxv - minv)
    denom[denom == 0] = 1.0
    scaled = (seq - minv) / denom
    return scaled * (hi - lo) + lo


def zero_pad_sequences(sequences: List[np.ndarray], max_len: int = None) -> np.ndarray:
    """Pad sequences (list of (timesteps, channels)) to same length with zeros at end."""
    lengths = [s.shape[0] for s in sequences]
    if max_len is None:
        max_len = max(lengths)
    channels = sequences[0].shape[1] if sequences[0].ndim > 1 else 1
    out = np.zeros((len(sequences), max_len, channels), dtype=float)
    for i, s in enumerate(sequences):
        L = min(s.shape[0], max_len)
        if channels == 1 and s.ndim == 1:
            out[i, :L, 0] = s[:L]
        else:
            out[i, :L, :] = s[:L, :]
    return out


def augment_oversample(X: np.ndarray, y: np.ndarray, target_count: int = None, jitter_scale: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Semplice oversampling: duplica campioni minoritari con jitter.

    X: (n_samples, timesteps, channels)
    y: (n_samples,)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    if target_count is None:
        target_count = int(np.max(counts))
    out_X = [X]
    out_y = [y]
    for cls in unique:
        cls_idx = np.where(y == cls)[0]
        need = target_count - counts[unique.tolist().index(cls)]
        if need <= 0:
            continue
        choices = np.random.choice(cls_idx, size=need, replace=True)
        dup = X[choices] + np.random.normal(scale=jitter_scale, size=X[choices].shape)
        out_X.append(dup)
        out_y.append(np.full(len(dup), cls))

    X_aug = np.concatenate(out_X, axis=0)
    y_aug = np.concatenate(out_y, axis=0)
    return X_aug, y_aug
