# src/decoding.py
from typing import List, Dict

import torch  # type: ignore


def merge_no_snap(segments: List[Dict]) -> List[Dict]:
    """
    Merge consecutive identical labels without snapping boundaries.
    Keeps original boundaries and only merges adjacent same-label segments.
    """
    if not segments:
        return segments

    merged = [segments[0].copy()]
    for s in segments[1:]:
        if s["label"] == merged[-1]["label"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s.copy())

    return [x for x in merged if (x["end"] - x["start"]) > 1e-4]


def decode_segments_from_logits(
    logits: torch.Tensor,
    frame_dur: float,
    tokenizer,
    blank_id: int,
    strategy: str = "baseline",
    ratio_thresh: float = 0.2,
    abs_thresh: float = 0.1,
    window: int = 2,
) -> List[Dict]:
    """
    Decode frame-level logits into phoneme segments.

    Strategies:
      - baseline: greedy top-1, drop blanks
      - conf_ratio_context: if top-1 is blank, promote a top-k candidate that
        matches neighbor top-1 and has p_k / p_blank > ratio_thresh
      - recursive_adjust: proto-segmentation + iterative blank promotion using
        context window (±window segments), choosing best candidate among top-k
        that appears in neighbors

    Returns list of segments:
      [{"label": str, "start": float, "end": float}, ...]
    """
    probs = torch.softmax(logits, dim=-1)
    topk = min(4, probs.shape[-1])
    topk_probs, topk_idx = probs.topk(k=topk, dim=-1)

    T = logits.shape[0]

    def token_of(idx: int) -> str:
        tok = tokenizer.convert_ids_to_tokens(int(idx))
        if tok in ("<pad>", getattr(tokenizer, "pad_token", None)):
            return ""
        return tok

    # -------------------------
    # Strategy: recursive_adjust
    # -------------------------
    if strategy == "recursive_adjust":
        proto = []
        seen_nonblank = False

        for t in range(T):
            preds = [(token_of(int(i)), float(p)) for i, p in zip(topk_idx[t], topk_probs[t])]
            top_label = preds[0][0]
            top_id = int(topk_idx[t, 0])

            # skip leading blanks until first non-blank
            if not seen_nonblank and (top_label == "" or top_id == blank_id):
                continue
            seen_nonblank = True

            if not proto or top_label != proto[-1]["Predictions"][0][0]:
                if proto:
                    proto[-1]["End"] = t * frame_dur
                proto.append({"Start": t * frame_dur, "End": None, "Predictions": preds})

        if proto:
            proto[-1]["End"] = T * frame_dur

        def adjust_once(i: int) -> bool:
            # only adjust blanks
            if proto[i]["Predictions"][0][0] != "":
                return False

            # collect neighbor top-1 labels in a window of segments
            targets = []
            for off in range(1, window + 1):
                for d in (-1, 1):
                    j = i + d * off
                    if 0 <= j < len(proto):
                        targets.append(proto[j]["Predictions"][0][0])

            # promote first best candidate (among top-2..top-k) that matches targets
            for alt in proto[i]["Predictions"][1:]:
                lab = alt[0]
                prob = alt[1]
                if lab != "" and lab in targets:
                    # (optional) you can enforce abs_thresh too, but keep behavior close to your code
                    if prob >= abs_thresh:
                        k = proto[i]["Predictions"].index(alt)
                        proto[i]["Predictions"].insert(0, proto[i]["Predictions"].pop(k))
                        return True
            return False

        changed = True
        while changed:
            changed = False
            for i in range(len(proto)):
                if adjust_once(i):
                    changed = True

        out = []
        for s in proto:
            lab = s["Predictions"][0][0]
            if lab == "" or lab is None:
                continue
            out.append({"label": lab, "start": s["Start"], "end": s["End"]})

        return out

    # -------------------------
    # Framewise strategies
    # -------------------------
    segments: List[Dict] = []
    prev_label = None

    for t in range(T):
        top1 = int(topk_idx[t, 0])
        p1 = float(topk_probs[t, 0])

        # baseline
        chosen = top1

        if strategy == "baseline":
            if top1 == blank_id:
                prev_label = None
                continue

        elif strategy == "conf_ratio_context":
            if top1 == blank_id:
                # neighbor set from top-1 prev/next
                neighbors = set()
                if t > 0:
                    neighbors.add(int(topk_idx[t - 1, 0]))
                if t + 1 < T:
                    neighbors.add(int(topk_idx[t + 1, 0]))

                # check top-2..top-3 candidates for neighbor match + ratio threshold
                chosen = blank_id
                for k in range(1, min(3, topk)):  # top2 and top3
                    cand = int(topk_idx[t, k])
                    pk = float(topk_probs[t, k])

                    if cand in neighbors:
                        ratio = pk / max(p1, 1e-8)
                        if ratio > ratio_thresh and pk >= abs_thresh:
                            chosen = cand
                            break

                if chosen == blank_id:
                    prev_label = None
                    continue

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # materialize chosen
        tok = token_of(chosen)
        if tok == "":
            prev_label = None
            continue

        if tok != prev_label:
            segments.append({"label": tok, "start": t * frame_dur, "end": (t + 1) * frame_dur})
            prev_label = tok
        else:
            segments[-1]["end"] = (t + 1) * frame_dur

    return segments
