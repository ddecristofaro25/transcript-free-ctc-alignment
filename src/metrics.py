# src/metrics.py
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = prev if a[i - 1] == b[j - 1] else prev + 1
            cur = min(cur, dp[j] + 1, dp[j - 1] + 1)
            prev, dp[j] = dp[j], cur
    return dp[m]


def merge_length_marks(seq: List[str]) -> List[str]:
    merged: List[str] = []
    for tok in seq:
        if tok == "ː" and merged:
            merged[-1] += "ː"
        else:
            merged.append(tok)
    return merged


def compute_per(pred_segments: List[Dict], gold_segments: List[Dict]) -> float:
    """
    Phoneme Error Rate using Levenshtein distance on label sequences.
    Drops <unk>. Merges length marks (ː) onto previous token.
    """
    pred_seq = [p["label"] for p in pred_segments if p["label"] != "<unk>"]
    gold_seq = [g["label"] for g in gold_segments if g["label"] != "<unk>"]
    pred_seq = merge_length_marks(pred_seq)
    gold_seq = merge_length_marks(gold_seq)

    if len(gold_seq) == 0:
        return 0.0 if len(pred_seq) == 0 else 1.0

    dist = _levenshtein(gold_seq, pred_seq)
    return dist / len(gold_seq)


def compute_abd_pbe_pdur(
    pred: List[Dict], gold: List[Dict]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    ABD: average boundary deviation (ms) using positional pairing (min len).
    PBE_pos: here identical to ABD (kept for compatibility with your tables).
    PDUR: average duration error (ms).
    """
    if not pred or not gold:
        return None, None, None

    N = min(len(pred), len(gold))
    abd_ms = 0.0
    pdur_ms = 0.0

    for i in range(N):
        g, p = gold[i], pred[i]
        abd_ms += (abs(p["start"] - g["start"]) + abs(p["end"] - g["end"])) * 0.5 * 1000.0
        pdur_ms += abs((p["end"] - p["start"]) - (g["end"] - g["start"])) * 1000.0

    abd_ms /= N
    pbe_pos_ms = abd_ms
    pdur_ms /= N
    return abd_ms, pbe_pos_ms, pdur_ms


def compute_pbe_label_aware(pred: List[Dict], gold: List[Dict]) -> Optional[float]:
    """
    Label-aware boundary deviation:
    For each label, align occurrences by order and average boundary deviation.
    """
    gold_by = defaultdict(list)
    pred_by = defaultdict(list)

    for g in gold:
        if g["label"] != "<unk>":
            gold_by[g["label"]].append(g)
    for p in pred:
        if p["label"] != "<unk>":
            pred_by[p["label"]].append(p)

    errs: List[float] = []
    for lab, gsegs in gold_by.items():
        psegs = pred_by.get(lab, [])
        M = min(len(gsegs), len(psegs))
        for i in range(M):
            g, p = gsegs[i], psegs[i]
            errs.append((abs(p["start"] - g["start"]) + abs(p["end"] - g["end"])) * 0.5 * 1000.0)

    if not errs:
        return None
    return sum(errs) / len(errs)


def boundary_events_from_segments(segs: List[Dict]) -> List[Tuple[float, str]]:
    """
    Return boundary events at each segment start (excluding the first segment).
    Each event is (time, label_of_segment_start).
    """
    if not segs:
        return []
    events: List[Tuple[float, str]] = []
    for i in range(1, len(segs)):
        if segs[i]["label"] != "<unk>":
            events.append((segs[i]["start"], segs[i]["label"]))
    return events


def boundary_precision_recall(
    pred: List[Dict], gold: List[Dict], tolerance_ms: float = 20.0
) -> Tuple[float, float, float]:
    """
    Boundary P/R/F1: match predicted boundary events to gold by (label, time)
    within tolerance.
    """
    tol = tolerance_ms / 1000.0

    P_events = boundary_events_from_segments(pred)
    G_events = boundary_events_from_segments(gold)

    matched_g = set()
    tp = 0

    for t_p, lab_p in P_events:
        candidates = [
            (j, abs(t_p - t_g))
            for j, (t_g, lab_g) in enumerate(G_events)
            if lab_g == lab_p and abs(t_p - t_g) <= tol and j not in matched_g
        ]
        if candidates:
            j, _ = min(candidates, key=lambda x: x[1])
            matched_g.add(j)
            tp += 1

    P = tp / max(len(P_events), 1)
    R = tp / max(len(G_events), 1)
    F1 = 2 * P * R / max(P + R, 1e-8)
    return P, R, F1
