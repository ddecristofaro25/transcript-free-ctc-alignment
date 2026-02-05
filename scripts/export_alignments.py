#!/usr/bin/env python3
import argparse
import torch  # type: ignore

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Tokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from src.export import export_alignments_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    ap.add_argument("--strategies", default="baseline,conf_ratio_context,recursive_adjust")
    ap.add_argument("--csv_name", default="alignments_segments.csv")
    ap.add_argument("--one_textgrid_per_file", action="store_true")
    ap.add_argument("--ratio_thresh", type=float, default=0.2)
    ap.add_argument("--abs_thresh", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    # --- same loading pattern ---
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(args.model)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)
    model.eval()
    # ---------------------------

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    export_alignments_folder(
        wav_root=args.wav_root,
        out_dir=args.out_dir,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        strategies=strategies,
        device=device,
        csv_name=args.csv_name,
        one_textgrid_per_file=args.one_textgrid_per_file,
        ratio_thresh=args.ratio_thresh,
        abs_thresh=args.abs_thresh,
        window=args.window,
    )


if __name__ == "__main__":
    main()
