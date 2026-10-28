#!/usr/bin/env python3
"""Replication on MTRAG (Section 4.4 of the paper; the human check of the judge is in Section 4.2):
what do false misses rest on -- passages the annotators rejected, or passages they never saw? Input is part of what mtrag.ipynb writes to reteco/mtrag
(local copy in data/drive/reteco/mtrag): turns.jsonl (gold, human annotator judgments, top-5) and
suff/phi-4_<retriever>.jsonl. The notebook's passages.jsonl (passage texts) is not read here and
is not released.

A false miss is a context without gold (suff_qrels none) that the judge labels yes. Its supporting
passages are the judge's evidence_docs (indices 1..5 into context_ids). Passage class for the turn:
  other_turn  -- gold of another turn in the same conversation;
  same_page   -- a chunk of the same page as the turn's gold (MTRAG chunks overlap);
  rejected    -- the annotator saw the passage on this turn and marked it not relevant (human = no);
  unjudged    -- the annotator did not see the passage on this turn (or there is no judgment).
A context is "contested" if the annotator marked all of its supporting passages not relevant on this
turn (human = no, whatever the passage class): the judge's yes may then be a mistake; otherwise the
context holds support that humans did not reject, i.e. a hole in the qrels. rejected_any is the share of
top-5 passages in qrels-none contexts with human = no on this turn.
fm_if_contested_wrong = (false misses - contested) / qrels-none: the false-miss rate if the judge was
wrong on every contested case.
Bootstrap: over turns, stratified by domain, 1000 resamples, seed 0.
Output: docs/paper/out/v2/mtrag_anatomy.csv, mtrag_anatomy.json.
"""
import csv
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MT = os.path.join(ROOT, "data", "drive", "reteco", "mtrag")
OUT = os.path.join(ROOT, "docs", "paper", "out", "v2")
RET = ["bm25_questions", "bm25_lastturn", "bm25_rewrite", "hybrid_rewrite", "dense_rewrite"]
CLS = ["other_turn", "same_page", "rejected", "unjudged"]
B, SEED = 1000, 0


def jl(p):
    return [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]


def page_of(dom, pid):
    """As in mtrag.ipynb: the id without chunk offsets; for clapnq also without the paragraph offset."""
    p = pid.split("-")[0]
    return p.split("_")[0] if dom == "clapnq" else p


def main():
    turns = {t["task_id"]: t for t in jl(os.path.join(MT, "turns.jsonl")) if t.get("use") == "label"}
    conv_gold = {}
    for x, t in turns.items():
        conv_gold.setdefault(t["conversation_id"], {}).update({p: x for p in t["gold"]})

    def cls(x, p):
        t = turns[x]
        if conv_gold.get(t["conversation_id"], {}).get(p, x) != x:
            return "other_turn"
        if page_of(t["domain"], p) in {page_of(t["domain"], g) for g in t["gold"]}:
            return "same_page"
        return "rejected" if t.get("human", {}).get(p) == "no" else "unjudged"

    rng = np.random.default_rng(SEED)
    rows, js = [], {}
    for job in RET:
        path = os.path.join(MT, "suff", f"phi-4_{job}.jsonl")
        if not os.path.exists(path):
            print("missing", path)
            continue
        recs = [r for r in jl(path) if r["task_id"] in turns]
        none = [r for r in recs if r["suff_qrels"] == "none" and r.get("label") in ("yes", "partial", "no")]
        fm = [r for r in none if r["label"] == "yes"]
        best, contested, top_cls = Counter(), [], Counter()
        n_top, n_rej = 0, 0
        for r in none:
            for p in r["context_ids"]:
                top_cls[cls(r["task_id"], p)] += 1
                n_top += 1
                n_rej += turns[r["task_id"]].get("human", {}).get(p) == "no"
        for r in fm:
            ev = [r["context_ids"][e - 1] for e in r.get("evidence_docs") or [] if 1 <= e <= len(r["context_ids"])]
            cs = [cls(r["task_id"], p) for p in ev]
            best[min(cs, key=CLS.index) if cs else "no_evidence"] += 1
            contested.append(bool(ev) and all(turns[r["task_id"]].get("human", {}).get(p) == "no" for p in ev))
        # bootstrap over qrels-none turns, stratified by domain
        dom = np.array([r["domain"] for r in none])
        isfm = np.array([r["label"] == "yes" for r in none], float)
        con = np.zeros(len(none))
        k = 0
        for i, r in enumerate(none):
            if r["label"] == "yes":
                con[i] = contested[k]
                k += 1
        idx = [np.where(dom == d)[0] for d in np.unique(dom)]
        est = {"fm": [], "fm_if_contested_wrong": [], "contested_share": []}
        for _ in range(B):
            s = np.concatenate([rng.choice(i, len(i)) for i in idx])
            est["fm"].append(isfm[s].mean())
            est["fm_if_contested_wrong"].append((isfm[s] - con[s]).mean())
            est["contested_share"].append(con[s].sum() / max(1, isfm[s].sum()))
        point = {"fm": isfm.mean(), "fm_if_contested_wrong": (isfm - con).mean(),
                 "contested_share": con.sum() / max(1, isfm.sum())}
        lab_all = [r.get("label") for r in recs]
        q_all = [r["suff_qrels"] for r in recs]
        n_ok = sum(l in ("yes", "partial", "no") for l in lab_all)
        tp = sum(q == "none" and l == "no" for q, l in zip(q_all, lab_all))
        fp = sum(q == "none" and l == "yes" for q, l in zip(q_all, lab_all))
        out = {"job": job, "n_none": len(none), "n_fm": len(fm), "n_contested": int(con.sum()),
               # unrounded shares for tab_mtrag (summary.csv is rounded to 4 decimals, which would round twice)
               "share_qrels_none": sum(q == "none" for q, l in zip(q_all, lab_all) if l in ("yes", "partial", "no")) / n_ok,
               "share_judge_no": sum(l == "no" for l in lab_all) / n_ok,
               "precision": tp / (tp + fp) if tp + fp else float("nan"),
               "rejected_any": n_rej / max(1, n_top),
               "best_evidence": {c: best[c] for c in CLS + ["no_evidence"]},
               "top5_in_none": {c: top_cls[c] for c in CLS},
               "top5_in_none_share": {c: top_cls[c] / max(1, sum(top_cls.values())) for c in CLS}}
        for m, v in point.items():
            out[m] = float(v)
            out[m + "_lo"], out[m + "_hi"] = (float(np.percentile(est[m], q)) for q in (2.5, 97.5))
        js[job] = out
        rows.append({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in out.items()
                     if not isinstance(v, dict)} | {f"best_{c}": best[c] for c in CLS + ["no_evidence"]}
                    | {f"top5_{c}": round(out["top5_in_none_share"][c], 4) for c in CLS})
        print(f"{job:<15} none {len(none):>3}  false misses {out['fm']:.3f} [{out['fm_lo']:.3f},{out['fm_hi']:.3f}]  "
              f"contested {out['n_contested']:>3}/{len(fm):<3} -> if all are judge errors: "
              f"{out['fm_if_contested_wrong']:.3f} [{out['fm_if_contested_wrong_lo']:.3f},"
              f"{out['fm_if_contested_wrong_hi']:.3f}]")
        print(f"{'':<15} best evidence: {dict(best)};  human=no among top-5 of none: {out['rejected_any']:.3f}; top-5 in none: "
              + ", ".join(f"{c} {out['top5_in_none_share'][c]:.2f}" for c in CLS))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mtrag_anatomy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(js, open(os.path.join(OUT, "mtrag_anatomy.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
