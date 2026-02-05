# src/evaluation.py
import os
import csv
import glob
from typing import List, Dict, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Questi moduli li creiamo nei prossimi step
from src.datasets import load_timit_phn, load_textgrid_phn
from src.decoding import decode_segments_from_logits, merge_no_snap
from src.metrics import (
    compute_abd_pbe_pdur,
    compute_pbe_label_aware,
    boundary_precision_recall,
    compute_per,
)
from src.plotting import plot_alignment_comparison, plot_alignment_explained


def evaluate_folder(
    wav_root: str,
    phn_root: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    tokenizer: Wav2Vec2Tokenizer,
    blank_id: int,
    device: str = "cpu",
    tolerance_ms: float = 20.0,
    make_plots: bool = False,
    plot_out_dir: Optional[str] = None,
    csv_out_path: Optional[str] = None,
    ratio_thresh: float = 0.2,
    abs_thresh: float = 0.1,
    window: int = 2,
) -> List[Dict]:
    """
    Evaluate transcript-free alignment strategies over a folder of WAV files.

    Expects paired annotations under phn_root:
      - TIMIT:    same relpath basename with .PHN
      - TextGrid: same relpath basename with .TextGrid

    Writes per-file metrics to csv_out_path if provided.
    """

    if make_plots and plot_out_dir:
        os.makedirs(plot_out_dir, exist_ok=True)

    results: List[Dict] = []

    strategies = [
        "baseline",
        "conf_ratio_context",
        "recursive_adjust",
    ]

    wav_paths = sorted(glob.glob(os.path.join(wav_root, "**/*.wav"), recursive=True))

    for wav_path in wav_paths:
        rel = os.path.relpath(wav_path, wav_root)

        # --- load + resample wav ---
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0, keepdim=True)  # mono
        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        wav_np = wav.squeeze(0).numpy()

        # --- locate gold annotation ---
        phn_path = os.path.join(phn_root, os.path.splitext(rel)[0] + ".PHN")
        tg_path = os.path.join(phn_root, os.path.splitext(rel)[0] + ".TextGrid")

        if os.path.exists(phn_path):
            gold = load_timit_phn(phn_path, sr=sr)
        elif os.path.exists(tg_path):
            gold = load_textgrid_phn(tg_path)
        else:
            continue

        if not gold:
            continue

        # --- model forward ---
        with torch.no_grad():
            inputs = processor(wav_np, sampling_rate=sr, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            logits = outputs.logits[0].cpu()  # (T, V)

        total_dur = wav.shape[-1] / sr
        T = logits.shape[0]
        frame_dur = total_dur / max(T, 1)

        strategy_segments: Dict[str, List[Dict]] = {}

        for strat in strategies:
            if strat == "recursive_adjust":
                segs = decode_segments_from_logits(
                    logits=logits,
                    frame_dur=frame_dur,
                    tokenizer=tokenizer,
                    blank_id=blank_id,
                    strategy="recursive_adjust",
                    ratio_thresh=ratio_thresh,
                    abs_thresh=abs_thresh,
                    window=window,
                )
            elif strat == "conf_ratio_context":
                segs = decode_segments_from_logits(
                    logits=logits,
                    frame_dur=frame_dur,
                    tokenizer=tokenizer,
                    blank_id=blank_id,
                    strategy="conf_ratio_context",
                    ratio_thresh=ratio_thresh,
                    abs_thresh=abs_thresh,
                    window=window,
                )
            else:  # baseline
                segs = decode_segments_from_logits(
                    logits=logits,
                    frame_dur=frame_dur,
                    tokenizer=tokenizer,
                    blank_id=blank_id,
                    strategy="baseline",
                    ratio_thresh=ratio_thresh,
                    abs_thresh=abs_thresh,
                    window=window,
                )

            segs = merge_no_snap(segs)
            strategy_segments[strat] = segs

            abd_ms, pbe_pos_ms, pdur_ms = compute_abd_pbe_pdur(segs, gold)
            pbe_lab_ms = compute_pbe_label_aware(segs, gold)
            P, R, F1 = boundary_precision_recall(segs, gold, tolerance_ms=tolerance_ms)
            per = compute_per(segs, gold)

            results.append(
                {
                    "file": rel,
                    "strategy": strat,
                    "ABD_ms": abd_ms,
                    "PBE_pos_ms": pbe_pos_ms,
                    "PBE_label_ms": pbe_lab_ms,
                    "PDUR_ms": pdur_ms,
                    "Prec": P,
                    "Rec": R,
                    "F1": F1,
                    "PER": per,
                }
            )

        # --- plots (optional) ---
        if make_plots and plot_out_dir:
            base = os.path.splitext(os.path.basename(wav_path))[0]
            out_png = os.path.join(plot_out_dir, base + "_align.png")
            plot_alignment_comparison(wav, wav_path, sr, gold, strategy_segments, output_path=out_png)

            # optional zoom for common TIMIT sentences
            if base in ["SA1", "SA2"]:
                out_zoom = os.path.join(plot_out_dir, base + "_zoom.png")
                plot_alignment_explained(
                    wav,
                    wav_path,
                    sr,
                    gold,
                    strategy_segments["baseline"],
                    strategy_segments["conf_ratio_context"],
                    strategy_segments["recursive_adjust"],
                    output_path=out_zoom,
                    t_min=0.5,
                    t_max=1.5,
                )

    # --- write CSV (optional) ---
    if csv_out_path:
        keys = ["file", "strategy", "ABD_ms", "PBE_pos_ms", "PBE_label_ms", "PDUR_ms", "Prec", "Rec", "F1", "PER"]
        with open(csv_out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)

    return results
