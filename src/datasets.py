# src/datasets.py
from typing import List, Dict

# ---------- TIMIT closures ----------
CLOSURE_TO_STOP = {"pcl": "p", "bcl": "b", "tcl": "t", "dcl": "d", "kcl": "k", "gcl": "g"}


def normalize_timit_closures(segs: List[Dict]) -> List[Dict]:
    """
    If a closure (e.g., 'tcl') is followed by its matching stop ('t'),
    extend the stop start to the closure start and drop the closure.
    Orphan closures are dropped.
    """
    out: List[Dict] = []
    i = 0
    while i < len(segs):
        s = segs[i]
        lab = s["label"].strip().lower()

        if lab in CLOSURE_TO_STOP:
            want = CLOSURE_TO_STOP[lab]
            if i + 1 < len(segs):
                nxt = segs[i + 1]
                nxt_lab = nxt["label"].strip().lower()
                if nxt_lab == want:
                    nxt = dict(nxt)
                    nxt["start"] = min(nxt["start"], s["start"])
                    segs[i + 1] = nxt
            i += 1
            continue

        out.append(s)
        i += 1

    return out


def load_timit_phn(phn_path: str, sr: int = 16000) -> List[Dict]:
    """
    TIMIT .PHN format: start_sample end_sample label
    Returns segments in seconds, labels lowercased.
    """
    segs: List[Dict] = []
    with open(phn_path, "r") as f:
        for line in f:
            a, b, lab = line.strip().split()
            a_i, b_i = int(a), int(b)
            segs.append({"label": lab.lower(), "start": a_i / sr, "end": b_i / sr})
    return normalize_timit_closures(segs)


# ---------- Praat hack codes (your LABEL_MAP) ----------
LABEL_MAP = {
    r"\ct": "ɔ",
    r"\ae": "æ",
    r"\sw": "ə",
    r"\sh": "ʃ",
    r"\ef": "ɛ",
    r"\efa": "ɛa",
    r"\hs": "ʊ",
    r"\fh": "ɾ",
    r"\oe": "œ",
    r"\er": "ɜ",
    r"\ic": "ɪ",
    r"\n.": "ŋ",
    r"\ng": "ŋ",
    r"\ct\~^": "ɔ̃",
    r"ɔ\~^": "ɔ̃",
    r"a\hs": "aʊ",
    r"a\ct": "aɔ",
    r"a\ic": "aɪ",
    r"\ct\ef": "ɔɛ",
    r"u\ef": "uɛ",
    r"\cti": "ɔi",
    r"\cta": "ɔa",
    r"\tf": "θ",
    r"\as": "ɑ",
    r"\o-": "ɵ",
}


def normalize_phoneme(label: str) -> str:
    """
    Map Praat hack codes (e.g. \\ct) to real phoneme symbols.
    If no mapping exists, returns the label unchanged.
    """
    return LABEL_MAP.get(label, label)


# ---------- TextGrid loader ----------
def load_textgrid_phn(tg_path: str) -> List[Dict]:
    """
    Robust TextGrid loader:
    - tries `textgrids` first
    - if it fails, falls back to `praatio`
    Supports tiers: phone, MAU, PHO, PHON (first match).
    Returns segments: [{"label","start","end"}]
    """
    def pick_tier_name(names):
        for cand in ("phone", "MAU", "PHO", "PHON"):
            if cand in names:
                return cand
        return None

    # ---- 1) Try textgrids ----
    try:
        import textgrids  # type: ignore

        tg = textgrids.TextGrid(tg_path)
        tier_name = pick_tier_name(tg.keys())
        if tier_name is None:
            print(f"[WARN] No phone/MAU/PHO tier in {tg_path}, skipping.")
            return []

        segs: List[Dict] = []
        for interval in tg[tier_name]:
            lab = interval.text.strip()
            if not lab:
                continue
            lab = normalize_phoneme(lab)
            segs.append({"label": lab, "start": float(interval.xmin), "end": float(interval.xmax)})

        return segs

    except Exception as e:
        print(f"[WARN] textgrids failed on {tg_path}: {type(e).__name__}: {e}")

    # ---- 2) Fallback: praatio ----
    try:
        from praatio import textgrid as praatio_textgrid  # type: ignore

        tg = praatio_textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
        tier_name = pick_tier_name(tg.tierNames)
        if tier_name is None:
            print(f"[WARN] No phone/MAU/PHO tier in {tg_path} (praatio), skipping.")
            return []

        tier = tg.getTier(tier_name)

        segs: List[Dict] = []
        for start, end, lab in tier.entries:
            lab_s = str(lab).strip()
            if not lab_s:
                continue
            lab_s = normalize_phoneme(lab_s)
            segs.append({"label": lab_s, "start": float(start), "end": float(end)})

        return segs

    except Exception as e:
        print(f"[ERROR] praatio failed on {tg_path}: {type(e).__name__}: {e}")
        return []
