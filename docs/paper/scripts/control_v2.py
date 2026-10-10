#!/usr/bin/env python3
"""Negative control for judge leniency (Section 4): labels on a context that is known to be insufficient.

Context ctrl-<hybrid>: for every dev turn, the hybrid top-5 retrieved for a turn of another dialogue
from the same domain (the dialogue offset by half the dialogue list, same turn number, else the first
turn), with the turn's own gold removed (colab/build.py, contexts()). The judge sees the current
turn, its history and its reference answer, exactly as for the retrieved contexts.

For each judge: yes / partial / no rates on the control (CI: bootstrap over turns stratified by
domain, 1000 resamples, seed 0, as in label_v2.py) and the same rate on qrels-none contexts of every
retriever (false miss fm and fm_or_partial). Also counts how many control yes come from documents of the
turn's own dialogue (n_own_dialogue); doc_id is inspected only here and was never shown to the judge.
Grounding of yes in the documents: share of rationale words with 6+ letters (excluding the STOP list
of generic words) that occur in the text of the five documents; < .5 means the rationale paraphrases
the reference answer rather than the documents.
Leniency correction: if the judge also says yes wrongly at rate l on each retriever's qrels-none
contexts, a lower bound on the false-miss rate is fm_adj = (fm - l) / (1 - l).

Output: docs/paper/out/v2/control.csv, control.json.
"""
import csv
import json
import os
import re
from collections import Counter

import numpy as np

from label_v2 import DATA, DISP, DOMAINS, OUT, ST, SUFF, load_judges, lab_of

CTRL = "ctrl-hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
B, SEED = 1000, 0
STOP = {"documents", "document", "reference", "answer", "answers", "discuss", "explain", "explains", "mention",
        "provide", "supports", "support", "claims", "aligns", "specifically", "information", "regarding"}


def support_overlap(recs, lab):
    """For each yes: share of rationale words (6+ letters, excluding STOP) found in the context text."""
    need = {}
    for r, l in zip(recs, lab):
        if l == "yes":
            need.setdefault(r["domain"], set()).update(r["context_ids"])
    text = {}
    for dom, ids in need.items():
        for ln in open(os.path.join(DATA, dom, "documents.jsonl")):
            d = json.loads(ln)
            if d["doc_id"] in ids:
                text[(dom, d["doc_id"])] = d["content"].lower()
    cov = []
    for r, l in zip(recs, lab):
        if l != "yes":
            continue
        t = " ".join(text.get((r["domain"], c), "") for c in r["context_ids"])
        w = set(re.findall(r"[a-z]{6,}", r["rationale"].lower())) - STOP
        cov.append(sum(x in t for x in w) / max(1, len(w)))
    return cov


def boot(vals, dom, rng):
    """Mean and 95% CI, bootstrap stratified by domain."""
    idx = [np.where(dom == d)[0] for d in DOMAINS if (dom == d).any()]
    est = []
    for _ in range(B):
        s = np.concatenate([rng.choice(i, len(i)) for i in idx])
        est.append(vals[s].mean())
    return float(vals.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main():
    judges, _ = load_judges()
    rows, js = [], {}
    for judge in judges:
        path = os.path.join(SUFF, f"{judge}_{CTRL}_dev.jsonl")
        if not os.path.exists(path):
            continue
        recs = [json.loads(ln) for ln in open(path) if ln.strip()]
        if len(recs) != 858:
            print(f"{judge}: control incomplete ({len(recs)} rows), skipping")
            continue
        rng = np.random.default_rng(SEED)
        lab = np.array([lab_of(r) for r in recs])
        dom = np.array([r["domain"] for r in recs])
        assert all(r["gold_in_ctx"] == 0 and r["suff_qrels"] == "none" for r in recs)
        own = np.array([any(re.search(rf"(?i)_{re.escape(r['topic_id'].rsplit('_turn_', 1)[0])}_", f"_{c}_")
                            for c in r["context_ids"]) for r in recs])
        v = lab != "invalid"
        out = {"judge": judge, "stage": "control", "n": int(v.sum()), "n_invalid": int((~v).sum()),
               "n_own_dialogue": int(own.sum()), "n_yes_own_dialogue": int((own & (lab == "yes")).sum()),
               "yes_by_domain": dict(Counter(map(str, dom[lab == "yes"])))}
        for k, sel in (("yes", lab == "yes"), ("partial", lab == "partial"),
                       ("yes_or_partial", np.isin(lab, ["yes", "partial"]))):
            out[k], out[k + "_lo"], out[k + "_hi"] = boot(sel[v].astype(float), dom[v], rng)
        cov = support_overlap(recs, lab)
        out["n_yes_rationale_overlap_lt_half"] = int(sum(c < .5 for c in cov))
        l = out["yes"]
        rows.append({k: out[k] for k in ("judge", "stage", "n", "yes", "yes_lo", "yes_hi", "yes_or_partial",
                                         "yes_or_partial_lo", "yes_or_partial_hi")})
        stages = {}
        for st in ST:
            S = judges[judge].get(st)
            if not S:
                continue
            q = np.array([r["suff_qrels"] for r in S.values()])
            lb = np.array([lab_of(r) for r in S.values()])
            d = np.array([k[0] for k in S])
            m = (q == "none") & (lb != "invalid")
            fm = boot((lb[m] == "yes").astype(float), d[m], rng)
            fmp = boot(np.isin(lb[m], ["yes", "partial"]).astype(float), d[m], rng)
            stages[st] = {"n_none": int(m.sum()), "fm": fm, "fm_or_partial": fmp,
                          "fm_adj": (fm[0] - l) / (1 - l)}
            rows.append({"judge": judge, "stage": DISP[st], "n": int(m.sum()), "yes": fm[0], "yes_lo": fm[1],
                         "yes_hi": fm[2], "yes_or_partial": fmp[0], "yes_or_partial_lo": fmp[1],
                         "yes_or_partial_hi": fmp[2]})
        out["stages"] = stages
        js[judge] = out
        print(f"{judge}: control yes {out['yes']:.3f} [{out['yes_lo']:.3f},{out['yes_hi']:.3f}], "
              f"yes|partial {out['yes_or_partial']:.3f}; own dialogue in {out['n_own_dialogue']} contexts; "
              f"yes by domain {out['yes_by_domain']}; yes rationales with < half of words in documents: "
              f"{out['n_yes_rationale_overlap_lt_half']} of {len(cov)}")
        for st, s in stages.items():
            print(f"   {DISP[st]:<10} qrels-none n={s['n_none']:>3}  yes {s['fm'][0]:.3f}  "
                  f"yes|partial {s['fm_or_partial'][0]:.3f}  corrected lower bound {s['fm_adj']:.3f}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "control.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(x, 4) if isinstance(x, float) else x for k, x in r.items()} for r in rows)
    json.dump(js, open(os.path.join(OUT, "control.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
