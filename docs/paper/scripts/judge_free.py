"""A judge-free lower bound on false misses: copies of gold in "no gold" contexts (Section 4.2 of
the paper, paragraph "Without any judge").

A qrels-none context (no gold document of the turn in the top-5) contains a near-verbatim copy of
gold if one of its documents covers >=80% of the word 8-grams of a gold document of the turn, or
matches it with character 5-gram Jaccard >= .7 (the near_dup thresholds of
false_miss_anatomy_v2.py). No judge or model is involved: only the run, the qrels and the corpus
texts. Contexts and the qrels flag are read from the phi-4 files (context_ids and suff_qrels do
not depend on the judge's label). Shares are among none contexts; 95% CI from a bootstrap over
turns (1000 resamples, seed 0).

    python3 docs/paper/scripts/judge_free.py     # -> docs/paper/out/v2/judge_free.{csv,json}
"""
import csv, json, os, re
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
SUFF = os.path.join(ROOT, "data", "drive", "reteco", "suff")
OUT = os.path.join(ROOT, "docs", "paper", "out", "v2")
HYB = "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
STAGES = [("BM25-turn", "bm25_query"), ("BM25-hist", "bm25_hist"), ("Fusion", "bm25_fuse3"), ("Hybrid", HYB)]
JAC, CONT, B, SEED = 0.7, 0.8, 1000, 0


def load(stage):
    return [json.loads(x) for x in open(os.path.join(SUFF, f"phi-4_2c-{stage}_dev.jsonl")) if x.strip()]


def main():
    rows = {name: load(st) for name, st in STAGES}
    gold = defaultdict(set)
    for d in {r["domain"] for r in rows["BM25-turn"]}:
        for ln in open(os.path.join(DATA, d, "qrels_dev.txt")):
            p = ln.split()
            if len(p) == 4 and int(p[3]) > 0:
                gold[(d, p[0])].add(p[2])
    need = defaultdict(set)
    for rs in rows.values():
        for r in rs:
            if r["suff_qrels"] == "none":
                need[r["domain"]] |= set(r["context_ids"]) | gold[(r["domain"], r["topic_id"])]
    text = {}
    for d, ids in need.items():
        for ln in open(os.path.join(DATA, d, "documents.jsonl"), encoding="utf-8"):
            o = json.loads(ln)
            if o["doc_id"] in ids:
                text[(d, o["doc_id"])] = o["content"]

    sh, wg = {}, {}

    def shingles(k):
        if k not in sh:
            t = re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", text.get(k, "").lower())).strip()
            sh[k] = {t[i:i + 5] for i in range(len(t) - 4)}
        return sh[k]

    def wgrams(k, n=8):
        if k not in wg:
            w = re.sub(r"[^0-9a-z]+", " ", text.get(k, "").lower()).split()
            wg[k] = {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))} if w else set()
        return wg[k]

    def is_copy(d, doc, g):
        a, b = shingles((d, doc)), shingles((d, g))
        jac = len(a & b) / len(a | b) if a and b else 0.0
        wd, wgold = wgrams((d, doc)), wgrams((d, g))
        gold_inside = len(wd & wgold) / len(wgold) if wgold else 0.0
        return jac >= JAC or gold_inside >= CONT

    out, rng = [], np.random.default_rng(SEED)
    keys = [(r["domain"], r["topic_id"]) for r in rows["BM25-turn"]]
    BI = rng.integers(0, len(keys), size=(B, len(keys)))
    for name, _ in STAGES:
        by = {(r["domain"], r["topic_id"]): r for r in rows[name]}
        none = np.array([by[k]["suff_qrels"] == "none" for k in keys])
        copy = np.array([bool(none[i]) and any(is_copy(k[0], doc, g) for doc in by[k]["context_ids"]
                                               for g in gold[k]) for i, k in enumerate(keys)])
        share = copy.sum() / none.sum()
        bs = [copy[b].sum() / none[b].sum() for b in BI if none[b].sum()]
        out.append({"stage": name, "n_none": int(none.sum()), "n_copy": int(copy.sum()), "share": round(float(share), 4),
                    "lo": round(float(np.percentile(bs, 2.5)), 4), "hi": round(float(np.percentile(bs, 97.5)), 4)})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "judge_free.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    json.dump(out, open(os.path.join(OUT, "judge_free.json"), "w"), indent=1)
    for r in out:
        print(f"{r['stage']:<10} none {r['n_none']:>4}  with a gold copy {r['n_copy']:>3}  {r['share']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")


if __name__ == "__main__":
    main()
