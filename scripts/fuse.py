#!/usr/bin/env python3
"""Fuse retrieval runs: RRF or weighted score sum, and/or conversation memory.

    # hybrid of BM25 + history and a dense run
    python scripts/fuse.py --runs runs/bm25_hist runs/dense --out runs/hybrid_rrf

    # conversation memory: the ranking for turn t is blended with the rankings
    # of PREVIOUS turns of the same conversation only, weighted decay**(t - t')
    python scripts/fuse.py --runs runs/rewrite --memory-decay 0.5 --out runs/rewrite_memory

    # score-based hybrid (DIVER: 0.5/0.5), reranker on top of retrieval (DIVER: 0.6/0.4)
    python scripts/fuse.py --mode score --runs runs/A runs/B --weights 0.5 0.5 --out runs/...

RRF:   score(d) = Σ_i w_i / (k + rank_i(d)), k=60 by default (Cormack et al.).
score: score(d) = Σ_i w_i · minmax_i(d) — each run's scores are normalised to
       [0, 1] within a turn; a document missing from a list gets 0.
Weights and decay are not tuned on dev: either defaults or values fit on train.
Topic ids have the form <conv>_turn_<n>; the turn number is parsed from the id,
so the benchmark file is not needed.
"""
import argparse
import datetime as dt
import json
import os
import re
from collections import defaultdict

DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics",
           "sustainable_living"]
TOPIC = re.compile(r"^(.*)_turn_(\d+)$")


def load_run(path):
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) == 6:
                run[p[0]].append((p[2], int(p[3]), float(p[4])))
    return {q: sorted(v, key=lambda x: x[1]) for q, v in run.items()}


def rrf(lists, k):
    """lists: list of (weight, [(doc, rank, ...), ...])."""
    s = defaultdict(float)
    for w, ranked in lists:
        for d, r, *_ in ranked:
            s[d] += w / (k + r)
    return sorted(s.items(), key=lambda x: -x[1])


def score_fuse(lists):
    """lists: list of (weight, [(doc, rank, score), ...]); min-max within each list."""
    s = defaultdict(float)
    for w, ranked in lists:
        if not ranked:
            continue
        lo, hi = min(x[2] for x in ranked), max(x[2] for x in ranked)
        for d, _, v in ranked:
            s[d] += w * ((v - lo) / (hi - lo) if hi > lo else 1.0)
    return sorted(s.items(), key=lambda x: -x[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories")
    ap.add_argument("--mode", choices=["rrf", "score"], default="rrf")
    ap.add_argument("--weights", nargs="*", type=float, help="one per run, default 1")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--memory-decay", type=float, default=None,
                    help="weight of previous turns is decay**distance; memory is off without this flag")
    ap.add_argument("--memory-max-back", type=int, default=10)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="fuse")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.runs)
    if len(weights) != len(args.runs):
        raise SystemExit("--weights: exactly one per run")
    os.makedirs(args.out, exist_ok=True)

    for dom in args.domains:
        runs = []
        for rd in args.runs:
            for cand in (os.path.join(rd, dom, "run.trec"), os.path.join(rd, f"{dom}.trec")):
                if os.path.isfile(cand):
                    runs.append(load_run(cand)); break
            else:
                raise SystemExit(f"no run for {dom} in {rd}")
        qids = sorted(set().union(*[r.keys() for r in runs]))

        # 1) fuse the runs for each turn
        combine = (lambda ls: rrf(ls, args.k)) if args.mode == "rrf" else score_fuse
        fused = {q: combine([(w, r.get(q, [])) for w, r in zip(weights, runs)]) for q in qids}

        # 2) conversation memory: blend in the rankings of previous turns
        if args.memory_decay is not None:
            by_conv = defaultdict(dict)
            for q in qids:
                m = TOPIC.match(q)
                if m:
                    by_conv[m.group(1)][int(m.group(2))] = q
            out = {}
            for q in qids:
                m = TOPIC.match(q)
                if not m:
                    out[q] = fused[q]; continue
                conv, t = m.group(1), int(m.group(2))
                lists = [(1.0, [(d, i + 1) for i, (d, _) in enumerate(fused[q])])]
                for back in range(1, args.memory_max_back + 1):
                    prev = by_conv[conv].get(t - back)
                    if prev is None:
                        continue
                    lists.append((args.memory_decay ** back,
                                  [(d, i + 1) for i, (d, _) in enumerate(fused[prev])]))
                out[q] = rrf(lists, args.k)
            fused = out

        path = os.path.join(args.out, dom, "run.trec")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for q in qids:
                for rank, (d, s) in enumerate(fused[q][:args.topk], 1):
                    f.write(f"{q}\tQ0\t{d}\t{rank}\t{s:.6f}\t{args.tag}\n")
        print(f"  {dom}: {len(qids)} turns")

    cfg = {"script": "scripts/fuse.py", "args": vars(args),
           "date": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
