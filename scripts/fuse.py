#!/usr/bin/env python3
"""Слияние прогонов: RRF или взвешенная сумма баллов, и/или память диалога.

    # B4: гибрид BM25+история и плотного
    $PY scripts/fuse.py --runs runs/..._b1_bm25_hist runs/..._b3_dense --out runs/..._b4_rrf

    # S3: память диалога — к ранжированию реплики t подмешиваются ранжирования
    #     ТОЛЬКО прошлых реплик того же диалога с весом decay**(t - t')
    $PY scripts/fuse.py --runs runs/..._s2 --memory-decay 0.5 --out runs/..._s3

    # гибрид по баллам (DIVER: 0.5/0.5), реранк поверх поиска (DIVER: 0.6/0.4)
    $PY scripts/fuse.py --mode score --runs runs/A runs/B --weights 0.5 0.5 --out runs/...

RRF:   score(d) = Σ_i w_i / (k + rank_i(d)), k=60 по умолчанию (Cormack et al.).
score: score(d) = Σ_i w_i · minmax_i(d) — баллы каждого прогона нормируются в
       [0, 1] внутри реплики; документа нет в списке — 0.
Веса и decay не подбираются на dev: либо значения по умолчанию, либо из train.
Топики вида <conv>_turn_<n>; номер реплики берётся из id, benchmark не нужен.
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
    """lists: список (weight, [(doc, rank, ...), ...])."""
    s = defaultdict(float)
    for w, ranked in lists:
        for d, r, *_ in ranked:
            s[d] += w / (k + r)
    return sorted(s.items(), key=lambda x: -x[1])


def score_fuse(lists):
    """lists: список (weight, [(doc, rank, score), ...]); min-max внутри списка."""
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
    ap.add_argument("--runs", nargs="+", required=True, help="папки прогонов")
    ap.add_argument("--mode", choices=["rrf", "score"], default="rrf")
    ap.add_argument("--weights", nargs="*", type=float, help="по одному на прогон, по умолчанию 1")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--memory-decay", type=float, default=None,
                    help="вес прошлых реплик decay**расстояние; без флага память выключена")
    ap.add_argument("--memory-max-back", type=int, default=10)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="fuse")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.runs)
    if len(weights) != len(args.runs):
        raise SystemExit("--weights: по одному на прогон")
    os.makedirs(args.out, exist_ok=True)

    for dom in args.domains:
        runs = []
        for rd in args.runs:
            for cand in (os.path.join(rd, dom, "run.trec"), os.path.join(rd, f"{dom}.trec")):
                if os.path.isfile(cand):
                    runs.append(load_run(cand)); break
            else:
                raise SystemExit(f"нет прогона {dom} в {rd}")
        qids = sorted(set().union(*[r.keys() for r in runs]))

        # 1) слияние прогонов для каждой реплики
        combine = (lambda ls: rrf(ls, args.k)) if args.mode == "rrf" else score_fuse
        fused = {q: combine([(w, r.get(q, [])) for w, r in zip(weights, runs)]) for q in qids}

        # 2) память диалога: подмешать ранжирования прошлых реплик
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
        print(f"  {dom}: {len(qids)} реплик")

    cfg = {"script": "scripts/fuse.py", "args": vars(args),
           "date": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
