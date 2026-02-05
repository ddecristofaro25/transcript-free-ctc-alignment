# src/plotting.py
import os
from typing import List, Dict, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
import torchaudio.transforms as T  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.gridspec import GridSpec  # type: ignore

from src.datasets import normalize_phoneme


def plot_alignment_comparison(
    waveform: torch.Tensor,
    wav_path: str,
    sr: int,
    gold: List[Dict],
    strategies_all: Dict[str, List[Dict]],
    output_path: Optional[str] = None,
) -> None:
    """
    Multi-tier plot: spectrogram + lanes (GOLD + strategies).
    Draws waveform in each lane and colored spans per segment.
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(torch.float32).squeeze(0)

    dur = waveform.shape[0] / sr
    wav_np = (waveform.numpy() / (np.abs(waveform.numpy()).max() + 1e-8)) * 0.8

    tiers = {"GOLD": gold, **{k.upper(): v for k, v in strategies_all.items()}}
    colors = plt.cm.Pastel1.colors

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(len(tiers) + 1, 1, height_ratios=[2] + [1] * len(tiers), hspace=0.15)

    # --- top: spectrogram ---
    ax_spec = fig.add_subplot(gs[0])
    mel_spec = T.MelSpectrogram(sample_rate=sr)(waveform.unsqueeze(0))
    db_spec = T.AmplitudeToDB()(mel_spec)
    ax_spec.imshow(
        db_spec[0],
        origin="lower",
        aspect="auto",
        extent=[0, dur, 0, db_spec.shape[1]],
    )
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_xticks([])

    # --- lanes ---
    for i, (name, segs) in enumerate(tiers.items(), start=1):
        ax = fig.add_subplot(gs[i], sharex=ax_spec)
        ax.set_yticks([])
        ax.set_ylabel("")

        ax.text(
            -0.01,
            0.5,
            name,
            transform=ax.transAxes,
            rotation=0,
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

        times = np.linspace(0, dur, len(wav_np))
        ax.plot(times, wav_np, color="gray", linewidth=0.5, alpha=0.6)

        for seg in segs:
            start, end, lab = seg["start"], seg["end"], seg["label"]
            ax.axvspan(start, end, facecolor=colors[i % len(colors)], alpha=0.3)
            ax.text((start + end) / 2, 0, normalize_phoneme(lab), ha="center", va="center", fontsize=7)

        # boundary lines
        show_end = name.upper() in {"BASELINE", "CONF_RATIO_CONTEXT", "RECURSIVE_ADJUST"}
        for seg in segs:
            ax.axvline(seg["start"], color="k", lw=0.5)
            if show_end:
                ax.axvline(seg["end"], color="k", lw=0.5)

        ax.set_ylim(-1, 1)
        if i < len(tiers):
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (s)")

    fig.subplots_adjust(left=0.12)
    fig.suptitle(os.path.basename(wav_path))

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_alignment_explained(
    waveform: torch.Tensor,
    wav_path: str,
    sr: int,
    gold: List[Dict],
    baseline: List[Dict],
    conf_ratio: List[Dict],
    recursive: List[Dict],
    output_path: Optional[str] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> None:
    """
    Focused plot to visually explain strategies (zoomable).
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(torch.float32).squeeze(0)

    dur = waveform.shape[0] / sr
    wav_np = (waveform.numpy() / (np.abs(waveform.numpy()).max() + 1e-8)) * 0.8
    times = np.linspace(0, dur, len(wav_np))

    if t_min is None:
        t_min = 0.0
    if t_max is None:
        t_max = dur

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, wav_np, color="gray", lw=0.6, alpha=0.6)

    def draw_segments(segments: List[Dict], color: str, label: str, yoff: float):
        for seg in segments:
            start, end, lab = seg["start"], seg["end"], seg["label"]
            if end < t_min or start > t_max:
                continue
            ax.axvspan(start, end, ymin=yoff, ymax=yoff + 0.15, facecolor=color, alpha=0.4, edgecolor="none")
            ax.text((start + end) / 2, yoff + 0.075, normalize_phoneme(lab), ha="center", va="center", fontsize=7)
        ax.plot([], [], color=color, lw=6, alpha=0.5, label=label)

    draw_segments(gold, "black", "Gold", 0.65)
    draw_segments(baseline, "red", "Baseline", 0.45)
    draw_segments(conf_ratio, "blue", "Conf-ratio", 0.25)
    draw_segments(recursive, "orange", "Recursive", 0.05)

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
    ax.set_title(os.path.basename(wav_path))

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
