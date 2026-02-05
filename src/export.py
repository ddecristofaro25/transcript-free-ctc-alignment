# src/export.py
import os
import glob
from typing import Dict, List, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Tokenizer,
    Wav2Vec2ForCTC,
)

from src.decoding import decode_segments_from_logits


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def merge_no_snap(segments: List[Dict]) -> List[Dict]:
    """Merge consecutive identical labels, do NOT snap boundaries."""
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for s in segments[1:]:
        if s["label"] == merged[-1]["label"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return [x for x in merged if (x["end"] - x["start"]) > 1e-4]


def save_segments_csv(csv_path: str, rel_wav: str, strategy: str, segs: List[Dict]) -> None:
    """
    Append segments to a single CSV.
    Columns: file,strategy,start,end,label
    """
    header = "file,strategy,start,end,label\n"
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write(header)
        for s in segs:
            f.write(f"{rel_wav},{strategy},{s['start']:.6f},{s['end']:.6f},{s['label']}\n")


def write_textgrid(out_path: str, tiers: Dict[str, List[Dict]], total_dur: float) -> None:
    """
    Write Praat TextGrid (text format), one IntervalTier per tier name.
    tiers[name] is a list of dicts: {"start": float, "end": float, "label": str}
    """
    tier_names = list(tiers.keys())
    n_tiers = len(tier_names)

    def esc(s: str) -> str:
        return s.replace('"', '""')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write("xmin = 0\n")
        f.write(f"xmax = {total_dur:.6f}\n")
        f.write("tiers? <exists>\n")
        f.write(f"size = {n_tiers}\n")
        f.write("item []:\n")

        for idx, name in enumerate(tier_names, start=1):
            segs = tiers[name]
            f.write(f"    item [{idx}]:\n")
            f.write('        class = "IntervalTier"\n')
            f.write(f'        name = "{esc(name)}"\n')
            f.write("        xmin = 0\n")
            f.write(f"        xmax = {total_dur:.6f}\n")
            f.write(f"        intervals: size = {len(segs)}\n")

            for j, seg in enumerate(segs, start=1):
                start = max(0.0, float(seg["start"]))
                end = min(total_dur, float(seg["end"]))
                lab = "" if seg.get("label") is None else str(seg["label"])
                f.write(f"        intervals [{j}]:\n")
                f.write(f"            xmin = {start:.6f}\n")
                f.write(f"            xmax = {end:.6f}\n")
                f.write(f'            text = "{esc(lab)}"\n')


def export_alignments_folder(
    wav_root: str,
    out_dir: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    tokenizer: Wav2Vec2Tokenizer,
    strategies: List[str],
    device: str = "cpu",
    csv_name: str = "alignments_segments.csv",
    one_textgrid_per_file: bool = True,
    ratio_thresh: float = 0.2,
    abs_thresh: float = 0.1,
    window: int = 2,
) -> None:
    """
    Export alignments for all wav files under wav_root:
      - one global CSV with all segments
      - TextGrid(s) under out_dir (mirrors wav_root structure)

    If one_textgrid_per_file=True:
      produces one TextGrid per wav with multiple tiers (one per strategy).
    Else:
      produces one TextGrid per (wav,strategy).
    """
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, csv_name)

    wav_paths = sorted(glob.glob(os.path.join(wav_root, "**/*.wav"), recursive=True))
    print(f"[INFO] Found {len(wav_paths)} wav files")

    blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for wav_path in wav_paths:
        rel = os.path.relpath(wav_path, wav_root)
        base = os.path.splitext(os.path.basename(wav_path))[0]

        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0, keepdim=True)
        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        wav_np = wav.squeeze(0).numpy()

        with torch.no_grad():
            inputs = processor(wav_np, sampling_rate=sr, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            attn_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
            outputs = model(input_values=input_values, attention_mask=attn_mask)
            logits = outputs.logits[0].cpu()

        total_dur = wav.shape[-1] / sr
        T = logits.shape[0]
        frame_dur = total_dur / max(T, 1)

        decoded: Dict[str, List[Dict]] = {}

        for strat in strategies:
            segs = decode_segments_from_logits(
                logits=logits,
                frame_dur=frame_dur,
                tokenizer=tokenizer,
                blank_id=blank_id,
                strategy=strat,
                use_ipa_map=False,
                ratio_thresh=ratio_thresh,
                abs_thresh=abs_thresh,
                window=window,
            )
            segs = merge_no_snap(segs)

            decoded[strat] = segs
            save_segments_csv(csv_path, rel, strat, segs)

        out_subdir = os.path.join(out_dir, os.path.dirname(rel))
        ensure_dir(out_subdir)

        if one_textgrid_per_file:
            tg_out = os.path.join(out_subdir, f"{base}.TextGrid")
            write_textgrid(tg_out, decoded, total_dur)
        else:
            for strat in strategies:
                tg_out = os.path.join(out_subdir, f"{base}.{strat}.TextGrid")
                write_textgrid(tg_out, {strat: decoded[strat]}, total_dur)

        print(f"[OK] {rel}")

    print(f"\n[DONE] CSV saved to: {csv_path}")
    print(f"[DONE] TextGrids saved under: {out_dir}")
