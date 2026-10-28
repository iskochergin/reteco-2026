#!/usr/bin/env python3
"""The 2b/2c contrast on MTRAG (Section 4.4 of the paper): for Qwen3-4B on the hybrid retriever, refusal on
an insufficient retrieved context minus refusal on the gold context of the same turn, under the qrels label
and under the phi-4 content label, with responses without a status line excluded (as on RECOR, Section 3).
Also the share of Qwen3-4B responses without a status line per context set (the paper's 13--23%).

Input is part of what mtrag.ipynb writes to reteco/mtrag (local copy in data/drive/reteco/mtrag; the
environment variable RETECO_MTRAG points elsewhere): turns.jsonl, gen/Qwen3-4B-Instruct-2507_<job>.jsonl,
suff/phi-4_<job>.jsonl, summary.csv.

Definitions, as in the summary cell of mtrag.ipynb ("contrast_q", "contrast_c"):
  turns     -- the 777 ANSWERABLE + PARTIAL turns (use == label), sorted by task_id;
  refusal   -- status starts with UN. An empty status (no status line) is handled in two ways:
               "excluded" drops the turn if either of its two responses has no status (the paper);
               "answer" counts it as no refusal (mtrag.ipynb; must reproduce summary.csv, checked below);
  contrast  -- on the turns whose hybrid top-5 is insufficient (qrels: none; content: phi-4 label no),
               mean refusal on the hybrid context minus mean refusal on the gold context of the same turns.
               The gold context's own label is not required to be sufficient (in consequences_v2.py,
               def=same_turn also requires it; on MTRAG every gold context is qrels all).
Bootstrap: as in mtrag.ipynb, over turns within a domain, 1000 resamples, seed 0, 95% percentile CI;
the same resamples for both labels and both conventions.
Output: docs/paper/out/v2/mtrag_contrast.json. CPU only, runs in seconds.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MT = Path(os.environ["RETECO_MTRAG"]) if os.environ.get("RETECO_MTRAG") else ROOT / "data/drive/reteco/mtrag"
OUT = ROOT / "docs/paper/out/v2"
READER = "Qwen3-4B-Instruct-2507"
JUDGE = "phi-4"
JOB = "hybrid_rewrite"
JOBS = ["gold", "bm25_questions", "bm25_lastturn", "bm25_rewrite", "hybrid_rewrite", "dense_rewrite"]
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]  # MT_DOMAINS of mtrag.ipynb: the order of the strata
B, SEED = 1000, 0


def keyed(p):
    return {r["key"]: r for r in (json.loads(x) for x in open(p, encoding="utf-8") if x.strip())}


turns = {t["task_id"]: t for t in (json.loads(x) for x in open(MT / "turns.jsonl", encoding="utf-8"))
         if t.get("use") == "label"}
KEYS = sorted(turns)
N = len(KEYS)
assert N == 777
DOMS = np.array([turns[x]["domain"] for x in KEYS])
STRATA = [np.where(DOMS == d)[0] for d in DOMAINS if (DOMS == d).any()]
assert sum(map(len, STRATA)) == N

gen = {j: keyed(MT / f"gen/{READER}_{j}.jsonl") for j in JOBS}
STATUS = {j: np.array([gen[j][x]["status"] or "" for x in KEYS]) for j in JOBS}
suff = {j: keyed(MT / f"suff/{JUDGE}_{j}.jsonl") for j in ("gold", JOB)}
Q = np.array([suff[JOB][x]["suff_qrels"] for x in KEYS])
C = np.array([suff[JOB][x].get("label") or "" for x in KEYS])  # "" = judge label not parsed
assert all(gen[JOB][x]["suff"] == suff[JOB][x]["suff_qrels"] for x in KEYS)
assert all(suff["gold"][x]["suff_qrels"] == "all" for x in KEYS)


def refusal(j, empty):
    """1 = refusal, 0 = answer; an empty status is nan (excluded) or 0 (answer)."""
    st = STATUS[j]
    r = np.array([float(s.startswith("UN")) for s in st])
    return np.where(st == "", np.nan, r) if empty == "excluded" else r


INSUFF = {"qrels": Q == "none", "content": C == "no"}
CONV = ("excluded", "answer")
REF = {e: (refusal(JOB, e), refusal("gold", e)) for e in CONV}


def contrast(idx):
    out = {}
    for e, (r, rg) in REF.items():
        r, rg = r[idx], rg[idx]
        ok = ~np.isnan(r) & ~np.isnan(rg)
        for L, m in INSUFF.items():
            mm = ok & m[idx]
            out[e, L] = float(r[mm].mean() - rg[mm].mean()) if mm.any() else np.nan
            out[e, L, "n"] = float(mm.sum())
    return out


# bootstrap as boot() in mtrag.ipynb: a fresh generator, resampling within each domain
point = contrast(np.arange(N))
rng, acc = np.random.default_rng(SEED), {k: [] for k in point}
for _ in range(B):
    idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in STRATA])
    for k, v in contrast(idx).items():
        acc[k].append(v)
res = {}
for e in CONV:
    for L in INSUFF:
        a = np.asarray(acc[e, L], float)
        a = a[~np.isnan(a)]
        lo, hi = np.percentile(a, [2.5, 97.5])
        res[e, L] = {"n": int(point[e, L, "n"]), "contrast": point[e, L], "lo": float(lo), "hi": float(hi),
                     "ci_excludes_0": bool(lo > 0 or hi < 0)}

# check: with empty status counted as an answer, the numbers must equal summary.csv of mtrag.ipynb
summ = {}
for row in csv.reader(open(MT / "summary.csv", encoding="utf-8")):
    if row[:3] == ["readers", JOB, READER] and row[3] in ("contrast_q", "contrast_c"):
        summ["qrels" if row[3] == "contrast_q" else "content"] = tuple(map(float, row[4:7]))
for L, t in summ.items():
    r = res["answer", L]
    assert np.allclose(t, (r["contrast"], r["lo"], r["hi"]), atol=5e-5), (L, t, r)

empty = {j: {"n_empty": int((STATUS[j] == "").sum()), "n": N, "share": float((STATUS[j] == "").mean())}
         for j in JOBS}

print(f"{READER}, {JOB}: refusal on insufficient 2c context minus on gold, same turns "
      f"({B} resamples within domain, seed {SEED})")
for e in CONV:
    tag = "paper: no-status responses excluded" if e == "excluded" else "mtrag.ipynb: no status = answer"
    print(f"  {tag}")
    for L in INSUFF:
        r = res[e, L]
        print(f"    {L:<8} n={r['n']:>3}  {r['contrast']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]"
              f"  CI excludes 0: {r['ci_excludes_0']}")
print("  summary.csv (contrast_q, contrast_c) reproduced by the 'answer' rows:", bool(summ))
print(f"{READER}: responses without a status line, per context set (of {N})")
for j, v in empty.items():
    print(f"    {j:<15} {v['n_empty']:>3}  {100 * v['share']:.1f}%")
sh = [v["share"] for v in empty.values()]
print(f"    range {100 * min(sh):.1f}-{100 * max(sh):.1f}%")

OUT.mkdir(parents=True, exist_ok=True)
json.dump({"reader": READER, "judge": JUDGE, "retriever": JOB, "n_turns": N, "n_boot": B, "seed": SEED,
           "contrast": {e: {L: res[e, L] for L in INSUFF} for e in CONV},
           "summary_csv_reproduced": bool(summ), "no_status": empty},
          open(OUT / "mtrag_contrast.json", "w"), ensure_ascii=False, indent=1)
