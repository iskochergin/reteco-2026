#!/usr/bin/env python3
"""nDCG@10 по прогону — та же метрика, что на лидерборде.

pytrec_eval ndcg_cut_10, среднее по репликам внутри домена, затем среднее по
доменам. Реплика из qrels, которой нет в прогоне, получает ноль.

    $PY scripts/score.py --runs runs/2026-09-28_b1_bm25_hist            # свой прогон
    $PY scripts/score.py --runs runs/... --domains law drones           # часть доменов
    $PY scripts/score.py --check-baseline    # runs/baseline против docs/baseline_bm25.csv

Прогон — папка с <домен>/run.trec (или <домен>.trec).
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import pytrec_eval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
REF = os.path.join(ROOT, "docs", "baseline_bm25.csv")
DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics", "sustainable_living"]
KEYS = ["ndcg_cut_10", "recall_10", "recall_100", "recip_rank"]


def load_qrels(path):
    q = defaultdict(dict)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) == 4:
            q[p[0]][p[2]] = int(p[3])
    return dict(q)


def load_run(path):
    r = defaultdict(dict)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) == 6 and p[2] not in r[p[0]]:
            r[p[0]][p[2]] = float(p[4])
    return dict(r)


def score_domain(run_path, qrels_path):
    qrels = load_qrels(qrels_path)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10,100", "recip_rank"})
    per_topic = ev.evaluate(load_run(run_path))
    n = len(qrels)
    return {k: sum(v.get(k, 0.0) for v in per_topic.values()) / n for k in KEYS} | {"n": n}


def baseline(split="dev", config="2a_hist"):
    """Эталон по доменам из docs/baseline_bm25.csv."""
    return {r["domain"]: float(r["ndcg_10"]) for r in csv.DictReader(open(REF, encoding="utf-8"))
            if r["split"] == split and r["config"] == config}


def score_runs(args):
    ref = baseline(args.split)
    rows = []
    for dom in args.domains:
        for p in (os.path.join(args.runs, dom, "run.trec"), os.path.join(args.runs, f"{dom}.trec")):
            if os.path.isfile(p):
                rows.append((dom, score_domain(p, os.path.join(DATA, dom, f"qrels_{args.split}.txt"))))
                break
        else:
            print(f"{dom:<20} нет прогона, пропуск")
    if not rows:
        sys.exit("нечего оценивать")

    print(f"{'домен':<20}{'nDCG@10':>9}{'R@10':>8}{'R@100':>8}{'MRR':>8}{'реплик':>8}{'BM25+hist':>11}")
    for dom, m in rows:
        print(f"{dom:<20}{m['ndcg_cut_10']:>9.4f}{m['recall_10']:>8.3f}{m['recall_100']:>8.3f}"
              f"{m['recip_rank']:>8.3f}{m['n']:>8d}{ref.get(dom, float('nan')):>11.4f}")
    macro = {k: sum(m[k] for _, m in rows) / len(rows) for k in KEYS}
    base = sum(ref[d] for d, _ in rows) / len(rows)
    print(f"{'МАКРО':<20}{macro['ndcg_cut_10']:>9.4f}{macro['recall_10']:>8.3f}{macro['recall_100']:>8.3f}"
          f"{macro['recip_rank']:>8.3f}{sum(m['n'] for _, m in rows):>8d}{base:>11.4f}")
    print(f"\nдельта к BM25+история: {macro['ndcg_cut_10'] - base:+.4f}")
    if args.out:
        json.dump({"per_domain": dict(rows), "macro": macro, "split": args.split},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def check_baseline():
    """runs/baseline/summary.json (посчитал кит) против опубликованных чисел."""
    path = os.path.join(ROOT, "runs", "baseline", "summary.json")
    if not os.path.isfile(path):
        sys.exit("нет runs/baseline/summary.json — сначала official_baseline.py (README)")
    got = json.load(open(path, encoding="utf-8"))["per_domain"]["track2_recor"]
    bad = 0
    for cfg in ("2a", "2a_hist"):
        ref = baseline("dev", cfg)
        for dom in DOMAINS:
            mine, want = got[dom][f"{cfg}_dev"]["NDCG@10"], ref[dom]
            ok = abs(mine - want) < 1e-4
            bad += not ok
            print(f"{dom:<20}{cfg:<9}{mine:>8.4f}{want:>8.4f}  {'ok' if ok else 'MISMATCH'}")
    print("\nвсё сошлось" if not bad else f"\nрасхождений: {bad}, окружение собрано неверно")
    sys.exit(bool(bad))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="папка прогона")
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--out", help="сохранить сводку в json")
    ap.add_argument("--check-baseline", action="store_true")
    a = ap.parse_args()
    check_baseline() if a.check_baseline else score_runs(a) if a.runs else ap.print_help()
