#!/usr/bin/env python3
import argparse
from collections import defaultdict

import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Tokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from src.evaluation import evaluate_folder


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(args.model)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()
  
    blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    wav_root = args.wav_root
    phn_root = args.phn_root if args.phn_root else args.wav_root

    results = evaluate_folder(
        wav_root=wav_root,
        phn_root=phn_root,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        blank_id=blank_id,
        device=device,
        tolerance_ms=args.tolerance_ms,
        make_plots=args.make_plots,
        plot_out_dir=args.plot_dir,
        csv_out_path=args.out_csv,
        ratio_thresh=args.ratio_thresh,
        abs_thresh=args.abs_thresh,
        window=args.window,
    )

    by_strat = defaultdict(list)
    for r in results:
        by_strat[r["strategy"]].append(r)

    print("\n=== Averages per strategy ===")
    metrics = ["ABD_ms","PBE_pos_ms","PBE_label_ms","PDUR_ms","Prec","Rec","F1","PER"]
    for s, rows in by_strat.items():
        avg = {}
        for m in metrics:
            vals = [x[m] for x in rows if x[m] is not None]
            avg[m] = sum(vals)/len(vals) if vals else float("nan")
        print(f"{s:>20s} | " + " ".join([f"{k}:{avg[k]:.3f}" for k in metrics]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", required=True, help="Root with .wav files (recursively searched)")
    parser.add_argument("--phn_root", default=None, help="Root with matching .PHN or .TextGrid (defaults to wav_root)")
    parser.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    parser.add_argument("--out_csv", default="alignment_results.csv")
    parser.add_argument("--plot_dir", default=None, help="If set, save alignment plots here")
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--tolerance_ms", type=float, default=20.0, help="Boundary tolerance for P/R/F1")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    parser.add_argument("--ratio_thresh", type=float, default=0.2, help="tau for confidence-ratio substitution")
    parser.add_argument("--abs_thresh", type=float, default=0.1, help="optional absolute threshold for blank substitution")
    parser.add_argument("--window", type=int, default=2, help="context window for recursive adjustment")

    args = parser.parse_args()
    main(args)
