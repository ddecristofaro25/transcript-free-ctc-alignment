# Transcript-Free Phoneme Alignment with CTC Blank Resolution

This repository contains the code accompanying the paper  
**“Making Transcript-Free Phoneme Alignment Usable for Low-Resource Annotation”**.

The code implements and evaluates **training-free blank-resolution strategies**
for transcript-free phoneme alignment using CTC-based speech models, with a
focus on low-resource and dialectal scenarios.

---

## Overview

CTC-based speech models expose frame-level posterior distributions without
requiring transcripts, but greedy decoding is dominated by blank symbols,
resulting in unstable phoneme boundaries.

This repository provides:

- **Decoding strategies**
  - Greedy CTC baseline
  - Confidence-ratio blank substitution
  - Recursive context adjustment
- **Evaluation metrics**
  - Average Boundary Deviation (ABD)
  - Phoneme Duration Error (PDUR)
  - Boundary Precision / Recall / F1
  - Phoneme Error Rate (PER)
- **Dataset support**
  - TIMIT (`.PHN`)
  - Praat TextGrid (`phone`, `MAU`, `PHO`, `PHON` tiers)
- **Visualization utilities**
  - Multi-tier alignment plots
  - Zoomed explanatory figures

All methods operate **without transcripts at decoding time** and **without retraining**.

---

## Requirements

Main dependencies:

- Python ≥ 3.8
- PyTorch
- torchaudio
- transformers
- numpy
- librosa
- matplotlib
- textgrids *(optional, for TextGrid parsing)*
- praatio *(fallback TextGrid parser)*

The code was tested with the HuggingFace model: facebook/wav2vec2-xlsr-53-espeak-cv-ft


---

## Running the Evaluation

All experiments are launched via the script:

bash
scripts/run_alignment_eval.py

Basic usage

python scripts/run_alignment_eval.py \
  --wav_root /path/to/wav_files \
  --phn_root /path/to/annotations \
  --out_csv alignment_results.csv

If Python cannot find the src package, run instead:

PYTHONPATH=. python scripts/run_alignment_eval.py \
  --wav_root /path/to/wav_files \
  --phn_root /path/to/annotations


---

## Audio

.wav files

Automatically resampled to 16 kHz if needed

---

## Annotations

The script looks for annotations matching the audio filenames:
TIMIT: .PHN files
Other corpora: .TextGrid files
Supported TextGrid tiers (first match is used):
phone, MAU, PHO, PHON

---

## Command-line Arguments

| Argument         | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `--wav_root`     | Root directory containing `.wav` files (recursive)         |
| `--phn_root`     | Root directory with annotations (defaults to `wav_root`)   |
| `--model`        | HuggingFace model name or path                             |
| `--out_csv`      | Output CSV with evaluation metrics                         |
| `--make_plots`   | Enable alignment plots                                     |
| `--plot_dir`     | Directory where plots are saved                            |
| `--tolerance_ms` | Boundary tolerance for Precision / Recall (default: 20 ms) |
| `--cpu`          | Force CPU inference                                        |


---

## Exporting Alignments (CSV / TextGrid)

Predicted phoneme alignments can be exported for manual inspection or
annotation using:
scripts/export_alignments.py

Example usage
PYTHONPATH=. python scripts/export_alignments.py \
  --wav_root /path/to/wav_files \
  --out_dir exported_alignments \
  --one_textgrid_per_file

---

## Notes on Reproducibility
No model retraining is performed.
Hyperparameters are selected via grid search on decoding outputs.
The code is intended for annotation bootstrapping, not as a replacement
for transcript-conditioned forced aligners.

---

## License
This code is released for research purposes.
Please cite the accompanying paper if you use it.

---

## Contact
This repository is anonymized for peer review.

