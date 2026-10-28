#!/usr/bin/env python3
"""The 2b/2c contrast on MTRAG (Section 4.4 of the paper): for Qwen3-4B on the hybrid retriever, refusal on
an insufficient retrieved context minus refusal on a sufficient gold context of the same turn, under the qrels
label and under the phi-4 content label. Also the share of Qwen3-4B responses without a status line per
context set (the paper's 13--23%).

Input is part of what mtrag.ipynb writes to reteco/mtrag (local copy in data/drive/reteco/mtrag; the
environment variable RETECO_MTRAG points elsewhere): turns.jsonl, gen/Qwen3-4B-Instruct-2507_<job>.jsonl,
suff/phi-4_<job>.jsonl, summary.csv.

The paper's numbers (primary output): def=same_turn of consequences_v2.py, as on RECOR.
  turns     -- the 777 ANSWERABLE + PARTIAL turns (use == label), sorted by task_id;
  refusal   -- status starts with UN; a turn is dropped if either of its two responses has no status line;
  contrast  -- on the turns whose hybrid top-5 is insufficient (qrels none; phi-4 no) AND whose gold context
               is sufficient under the same label (qrels all, which every MTRAG gold context is; phi-4 yes):
               mean refusal on the hybrid context minus mean refusal on the gold context.
  bootstrap -- as consequences_v2.py: over all turns, unstratified, default_rng(0).integers(0, N, (1000, N)),
               point estimate on the full sample, 95% percentile CI.
Secondary check, NOT the paper's definition: the contrast of the summary cell of mtrag.ipynb ("contrast_q",
"contrast_c"), which does not require the gold context to be sufficient and resamples turns within a domain
(1000, seed 0). With a missing status counted as an answer, as in the notebook, it must reproduce summary.csv
(asserted). With no-status turns dropped it gives the +.09 / +.01 of an earlier version of the paper.
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
SUFF, INSUFF = {"qrels": "all", "content": "yes"}, {"qrels": "none", "content": "no"}
B, SEED = 1000, 0


def keyed(p):
    return {r["key"]: r for r in (json.loads(x) for x in open(p, encoding="utf-8") if x.strip())}


# ── data ──────────────────────────────────────────────────────────────────
turns = {t["task_id"]: t for t in (json.loads(x) for x in open(MT / "turns.jsonl", encoding="utf-8"))
         if t.get("use") == "label"}
KEYS = sorted(turns)
N = len(KEYS)
assert N == 777
gen = {j: keyed(MT / f"gen/{READER}_{j}.jsonl") for j in JOBS}
STATUS = {j: np.array([gen[j][x]["status"] or "" for x in KEYS]) for j in JOBS}
suff = {j: keyed(MT / f"suff/{JUDGE}_{j}.jsonl") for j in ("gold", JOB)}
LAB = {(L, j): np.array([(suff[j][x]["suff_qrels"] if L == "qrels" else suff[j][x].get("label")) or ""
                         for x in KEYS])  # "" = judge label not parsed
       for L in SUFF for j in ("gold", JOB)}
assert all(gen[j][x]["suff"] == suff[j][x]["suff_qrels"] for j in ("gold", JOB) for x in KEYS)
assert (LAB["qrels", "gold"] == "all").all()


def refusal(j, empty):
    """1 = refusal, 0 = answer; an empty status is nan (excluded) or 0 (counted as an answer)."""
    st = STATUS[j]
    r = np.array([float(s.startswith("UN")) for s in st])
    return np.where(st == "", np.nan, r) if empty == "excluded" else r


def ci(v):
    """Point estimate (v[0]) and percentile 95% CI of the resamples (v[1:]), nan resamples dropped."""
    v = np.asarray(v, float)
    lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
    return {"contrast": float(v[0]), "lo": float(lo), "hi": float(hi), "ci_excludes_0": bool(lo > 0 or hi < 0)}


# ── (1) the paper: def=same_turn of consequences_v2.py ─────────────────────
IDX = np.random.default_rng(SEED).integers(0, N, size=(B, N))
W = np.vstack([np.ones(N)] + [np.bincount(ix, minlength=N) for ix in IDX]).astype(float)  # row 0 = point


def wmean(mask, val):
    m = mask.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ (m * np.nan_to_num(val))) / (W @ m)


r2c, rg = refusal(JOB, "excluded"), refusal("gold", "excluded")
paper = {}
for L in SUFF:
    T = (LAB[L, JOB] == INSUFF[L]) & (LAB[L, "gold"] == SUFF[L]) & ~np.isnan(r2c) & ~np.isnan(rg)
    paper[L] = {"n": int(T.sum()), **ci(wmean(T, r2c) - wmean(T, rg)),
                "refusal_2c": float(r2c[T].mean()), "refusal_gold": float(rg[T].mean())}

# ── (2) secondary check, not the paper's definition: the summary cell of mtrag.ipynb ──
DOMS = np.array([turns[x]["domain"] for x in KEYS])
STRATA = [np.where(DOMS == d)[0] for d in DOMAINS if (DOMS == d).any()]
CONV = ("answer", "excluded")  # answer: as the notebook; excluded: no-status turns dropped
REF = {e: (refusal(JOB, e), refusal("gold", e)) for e in CONV}


def nb_contrast(idx):
    out = {}
    for e, (r, g) in REF.items():
        r, g = r[idx], g[idx]
        for L in SUFF:
            mm = ~np.isnan(r) & ~np.isnan(g) & (LAB[L, JOB][idx] == INSUFF[L])
            out[e, L] = float(r[mm].mean() - g[mm].mean()) if mm.any() else np.nan
            out[e, L, "n"] = float(mm.sum())
    return out


rng = np.random.default_rng(SEED)  # boot() of mtrag.ipynb: a fresh generator, resampling within each domain
acc = [nb_contrast(np.arange(N))] + [nb_contrast(np.concatenate([g[rng.integers(0, len(g), len(g))]
                                                                  for g in STRATA])) for _ in range(B)]
notebook = {e: {L: {"n": int(acc[0][e, L, "n"]), **ci([a[e, L] for a in acc])} for L in SUFF} for e in CONV}
summ = {}
for row in csv.reader(open(MT / "summary.csv", encoding="utf-8")):
    if row[:3] == ["readers", JOB, READER] and row[3] in ("contrast_q", "contrast_c"):
        summ["qrels" if row[3] == "contrast_q" else "content"] = tuple(map(float, row[4:7]))
assert set(summ) == set(SUFF)
for L, t in summ.items():
    r = notebook["answer"][L]
    assert np.allclose(t, (r["contrast"], r["lo"], r["hi"]), atol=5e-5), (L, t, r)

# ── (3) responses without a status line ───────────────────────────────────
empty = {j: {"n_empty": int((STATUS[j] == "").sum()), "n": N, "share": float((STATUS[j] == "").mean())}
         for j in JOBS}


def line(L, r):
    return (f"    {L:<8} n={r['n']:>3}  {r['contrast']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]"
            f"  CI excludes 0: {r['ci_excludes_0']}")


print(f"{READER}, {JOB}: refusal on insufficient 2c context minus on gold, same turns")
print("  the paper (def=same_turn of consequences_v2.py: gold sufficient under the same label, "
      "no-status turns dropped, unstratified bootstrap)")
for L in SUFF:
    print(line(L, paper[L]) + f"  (refusal 2c {paper[L]['refusal_2c']:.3f}, gold {paper[L]['refusal_gold']:.3f})")
print("  secondary check, NOT the paper's definition: mtrag.ipynb (gold label not required, bootstrap within domain)")
for e in CONV:
    print(f"   no status = {'answer (as the notebook; equals summary.csv)' if e == 'answer' else 'dropped'}")
    for L in SUFF:
        print(line(L, notebook[e][L]))
print(f"{READER}: responses without a status line, per context set (of {N})")
for j, v in empty.items():
    print(f"    {j:<15} {v['n_empty']:>3}  {100 * v['share']:.1f}%")
sh = [v["share"] for v in empty.values()]
print(f"    range {100 * min(sh):.1f}-{100 * max(sh):.1f}%")

OUT.mkdir(parents=True, exist_ok=True)
json.dump({"reader": READER, "judge": JUDGE, "retriever": JOB, "n_turns": N, "n_boot": B, "seed": SEED,
           "paper_same_turn": paper,
           "notebook_check_not_paper": {"no_status_answer_equals_summary_csv": notebook["answer"],
                                        "no_status_dropped": notebook["excluded"]},
           "no_status": empty},
          open(OUT / "mtrag_contrast.json", "w"), ensure_ascii=False, indent=1)
