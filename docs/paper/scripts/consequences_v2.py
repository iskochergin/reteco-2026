"""Consequences of the label swap for refusals and detectors (Section 4.3 of the paper: Figure 3, the
2b/2c contrast, AUROC on the contexts that both labels keep, and the counts of changed conclusions):
readers x (gold 2b context + 4 retrievers).

The paper uses three Qwen3 readers; run from the repository root:
    RETECO_EXCLUDE=OCC-RAG-1.7B RETECO_OUT=docs/paper/out/three_readers python docs/paper/scripts/consequences_v2.py
Without these variables the script also includes OCC-RAG-1.7B, a fourth reader from an earlier version
of the analysis that the paper does not use, and writes to docs/paper/out/v2 (not reported in the paper).
Removing a reader does not change the numbers of the other readers.

Extends docs/paper/scripts/refusal_flips.py, the first version of this analysis (2 readers, OCC and
Qwen3-4B; gold and the 3 BM25 retrievers; not reported in the paper). Names containing "v1" (verdict_v1,
v1_subset, def=v1 in contrast.csv, the "v1 subset" counts in the summary) refer to that analysis. A
"stage" is the gold context or one of the four retrievers (BM25-turn, BM25-hist, BM25-fusion = Fusion,
hybrid = Hybrid).
Definitions:
  refusal = reader verdict (status starts with UN). An empty status is excluded from the refusal
  rate (verdict), as in Section 3 of the paper. verdict_v1, which counts an empty status as no refusal
  (as refusal_flips.py does), is kept for comparison with refusal_flips.py. With four readers, on the
  87-comparison subset (2 readers, 3 BM25 retrievers) with the phi-4 content label, verdict_v1 gives
  20 changed conclusions and verdict gives 17 (not reported in the paper).
  Judge-based refusal: response_type == refused (Qwen3-8B, as in refusal_flips.py; plus phi-4),
  available only for OCC and Qwen3-4B on gold / BM25-turn / BM25-hist / BM25-fusion.
  Labels: qrels all/part/none (S/P/I); content-based yes/partial/no (phi-4; Qwen3-14B if its file
  for that stage is complete, 858 rows); label null -> invalid, excluded from the sample. partial is
  excluded from binary metrics.
  Bootstrap over turns, 1000 resamples, seed 0, percentile 95% CI; the same resamples are used
  for all metrics -> differences are paired. Implemented via a multiplicity matrix (point estimate =
  row 0); with verdict_v1 on the v1 subset it reproduces the 20 of 87 changed conclusions of
  refusal_flips.py.
  auroc_common.csv: AUROC under both labels on the common set (qrels all/none AND content
  yes/no), paired difference. flips_judge_vs_judge in the summary: the same family of comparisons
  (258 with three readers, 384 with four), phi-4 vs Qwen3-14B, i.e. the "noise" threshold for the
  number of qrels-vs-content flips.
Where the numbers of Section 4.3 are (three readers, docs/paper/out/three_readers): Figure 3 from
auroc.csv (drawn by paper_figs.py); the 2b/2c contrast from contrast.csv (def=v1: pooled over turns;
def=same_turn: within turns); the counts of changed conclusions from consequences_summary.json,
flips -> phi-4 -> "extended, verdict" (258 comparisons, 75 changes, 13 by sign; survive_judge_*: the
refusal-based changes that an answer judge can check) and flips_judge_vs_judge (25); responses without
a status line per context set in n_empty_status.
Output: <RETECO_OUT, default docs/paper/out/v2>/{refusal,auroc,auroc_common,contrast,decomp,flips}.csv,
consequences_summary.json. CPU only, runs in seconds.
"""
import csv
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pytrec_eval

ROOT = Path(__file__).resolve().parents[3]
DRV = ROOT / "data/drive/reteco"
OUT = Path(os.environ["RETECO_OUT"]) if os.environ.get("RETECO_OUT") else ROOT / "docs/paper/out/v2"
READERS = {"OCC-RAG-1.7B": "OCC-RAG-1.7B", "Qwen3-1.7B": "Qwen3-1.7B",
           "Qwen3-4B": "Qwen3-4B-Instruct-2507", "Qwen3-8B": "Qwen3-8B"}
# the paper (three Qwen3 readers): RETECO_EXCLUDE=OCC-RAG-1.7B RETECO_OUT=docs/paper/out/three_readers
for _x in filter(None, os.environ.get("RETECO_EXCLUDE", "").split(",")):
    READERS.pop(_x)
RD = list(READERS)
OLD = {"OCC-RAG-1.7B", "Qwen3-4B"}  # readers of refusal_flips.py (the v1 subset)
HYB = "2c-hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
STAGES = ["2b", "2c-bm25_query", "2c-bm25_hist", "2c-bm25_fuse3", HYB]
S2C = STAGES[1:]
V1_STAGES = set(STAGES[:4])
NAME = {"2b": "gold", "2c-bm25_query": "BM25-turn", "2c-bm25_hist": "BM25-hist",
        "2c-bm25_fuse3": "Fusion", HYB: "Hybrid"}
RUN = {"2c-bm25_query": "bm25_query_dev", "2c-bm25_hist": "bm25_hist_dev",
       "2c-bm25_fuse3": "bm25_fuse3_dev", HYB: f"{HYB[3:]}_dev"}
SUFF_JUDGES = ["phi-4", "Qwen3-14B"]
ANS_JUDGES = ["Qwen3-8B", "phi-4"]
GRP = {"qrels": {"all": 0, "part": 1, "none": 2}, "content": {"yes": 0, "partial": 1, "no": 2}}
CLS = {"qrels": ("all", "part", "none"), "content": ("yes", "partial", "no")}
B, SEED = 1000, 0
N_TURNS = 858


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def complete(p):
    """A file counts as complete only if it has exactly 858 parseable lines."""
    if not p.exists():
        return None
    try:
        rs = jl(p)
    except json.JSONDecodeError:
        return None
    return rs if len(rs) == N_TURNS else None


# ── data ──────────────────────────────────────────────────────────────────
qsuff = {s: {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"suff/phi-4_{s}_dev.jsonl")} for s in STAGES}
KEYS = sorted(qsuff["2b"])
N = len(KEYS)
assert N == N_TURNS
LAB = {}  # (label, stage) -> int: 0 S, 1 P, 2 I, -1 invalid
for s in STAGES:
    LAB["qrels", s] = np.array([GRP["qrels"][qsuff[s][k]["suff_qrels"]] for k in KEYS])
CONTENT = []  # sufficiency judges with at least one complete stage
N_INVALID = {}
for J in SUFF_JUDGES:
    got = False
    for s in STAGES:
        rs = complete(DRV / f"suff/{J}_{s}_dev.jsonl")
        if rs is None:
            continue
        d = {(r["domain"], r["topic_id"]): r for r in rs}
        if set(d) != set(KEYS):
            continue
        LAB[J, s] = np.array([GRP["content"].get(d[k].get("label"), -1) for k in KEYS])
        N_INVALID[J, s] = int((LAB[J, s] < 0).sum())
        assert all(d[k]["suff_qrels"] == qsuff[s][k]["suff_qrels"] for k in KEYS)
        got = True
    if got:
        CONTENT.append(J)
LABELS = ["qrels"] + CONTENT


def has(L, *ss):
    return all((L, s) in LAB for s in ss)


GEN, REF = {}, defaultdict(dict)  # REF[D][(rd, s)] -> float array, nan = excluded from the refusal rate
for rd, f in READERS.items():
    for s in STAGES:
        g = {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"gen/{f}_{s}_dev.jsonl")}
        rs = [g[k] for k in KEYS]
        assert all(r["suff"] == qsuff[s][k]["suff_qrels"] for r, k in zip(rs, KEYS))
        st = [r["status"] or "" for r in rs]
        GEN[rd, s] = {
            "p": np.array([np.nan if r["p_answerable"] is None else r["p_answerable"] for r in rs]),
            "verdict": np.array([np.nan if r["p_answerable"] is None else float(not r["status"].startswith("UN"))
                                 for r in rs]),
            "n_empty": sum(x == "" for x in st),
            "n_odd": sum(x not in ("", "ANSWERABLE", "UNANSWERABLE") for x in st),
        }
        REF["verdict"][rd, s] = np.array([np.nan if x == "" else float(x.startswith("UN")) for x in st])
        REF["verdict_v1"][rd, s] = np.array([float(x.startswith("UN")) for x in st])
        for J in ANS_JUDGES:
            jr = complete(DRV / f"judge/{J}/{f}_{s}_dev.jsonl")
            if jr is not None:
                jd = {(r["domain"], r["topic_id"]): r for r in jr}
                REF[f"judge_{J}"][rd, s] = np.array([jd[k].get("response_type") == "refused" for k in KEYS], float)
DEFS = list(REF)  # verdict, verdict_v1, judge_*
VERDICT_DEFS = ("verdict", "verdict_v1")


def ndcg_macro(run_dir):
    per = []
    for dom in sorted(p.name for p in (ROOT / "data/reteco/track2_recor").iterdir() if p.is_dir()):
        qrels = defaultdict(dict)
        for line in open(ROOT / f"data/reteco/track2_recor/{dom}/qrels_dev.txt"):
            q, _, d, rel = line.split()
            qrels[q][d] = int(rel)
        run = defaultdict(dict)
        for line in open(DRV / f"runs/{run_dir}/{dom}/run.trec"):
            q, _, d, _, sc, _ = line.split()
            run[q][d] = float(sc)
        ev = pytrec_eval.RelevanceEvaluator(dict(qrels), {"ndcg_cut.10"})
        res = ev.evaluate({q: run.get(q, {}) for q in qrels})
        per.append(np.mean([res.get(q, {}).get("ndcg_cut_10", 0.0) for q in qrels]))
    return float(np.mean(per))


NDCG = {s: ndcg_macro(r) for s, r in RUN.items()}
NDCG["2b"] = 1.0

# ── bootstrap: multiplicity matrix W (row 0 is the point estimate) ─────────
IDX = np.random.default_rng(SEED).integers(0, N, size=(B, N))
W = np.vstack([np.ones(N)] + [np.bincount(ix, minlength=N) for ix in IDX]).astype(float)


def wmean(mask, val):
    """Weighted mean of val over mask for each row of W; nan if empty."""
    m = mask.astype(float)
    cnt = W @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ (m * np.nan_to_num(val))) / cnt, cnt


def wauroc(score, g):
    """Mann-Whitney with multiplicities: S (g=0) vs I (g=2); ties count as 1/2 (as with rankdata in refusal_flips.py)."""
    ok = ~np.isnan(score)
    P, Q = (g == 0) & ok, (g == 2) & ok
    sp, sq = score[P], score[Q]
    C = (sp[:, None] > sq[None, :]).astype(float) + .5 * (sp[:, None] == sq[None, :])
    wp, wq = W[:, P], W[:, Q]
    num = np.einsum("bq,bq->b", wp @ C, wq)
    den = wp.sum(1) * wq.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan), int(P.sum()), int(Q.sum())


def kitagawa(wa, ra, wb, rb):
    """ΔR = Σ Δw·r̄ (composition) + Σ Δr·w̄ (rate); an empty group takes the rate of the other stage."""
    ra = np.where(np.isnan(ra), rb, ra)
    rb = np.where(np.isnan(rb), ra, rb)
    ra, rb = np.nan_to_num(ra), np.nan_to_num(rb)
    return np.sum((wb - wa) * (ra + rb) / 2, 1), np.sum((rb - ra) * (wa + wb) / 2, 1)


def est(v):
    v = np.asarray(v, float)
    if np.isnan(v[0]) or np.all(np.isnan(v[1:])):
        return {"est": float(v[0]), "lo": np.nan, "hi": np.nan, "p_le0": np.nan}
    lo, hi = np.nanpercentile(v[1:], [2.5, 97.5])
    return {"est": float(v[0]), "lo": float(lo), "hi": float(hi), "p_le0": float(np.nanmean(v[1:] <= 0))}


def sig(e):
    return e["lo"] > 0 or e["hi"] < 0


M, NN = {}, {}
# class shares (independent of the reader)
for L in LABELS:
    for s in STAGES:
        if has(L, s):
            g = LAB[L, s]
            for c in range(3):
                M["share", L, s, c] = wmean(g >= 0, (g == c).astype(float))[0]
# refusals by class, weights and rates for the decomposition
for D in DEFS:
    for (rd, s), r in REF[D].items():
        ok = ~np.isnan(r)
        M["ref", D, rd, s, "all"], NN[D, rd, s] = wmean(ok, r)
        for L in LABELS:
            if not has(L, s):
                continue
            g = LAB[L, s]
            rates, cnts = [], []
            for c, gn in enumerate("SPI"):
                v, n = wmean((g == c) & ok, r)
                M["ref", D, rd, s, L, gn] = v
                rates.append(v)
                cnts.append(n)
            cnts = np.stack(cnts, 1)
            M["w", D, rd, s, L] = cnts / cnts.sum(1, keepdims=True)
            M["r", D, rd, s, L] = np.stack(rates, 1)
            M["tot", D, rd, s, L] = wmean((g >= 0) & ok, r)[0]
# p_answerable by class, AUROC
for rd in RD:
    for s in STAGES:
        p = GEN[rd, s]["p"]
        M["p", rd, s, "all"] = wmean(~np.isnan(p), p)[0]
        for L in LABELS:
            if not has(L, s):
                continue
            for c, gn in enumerate("SPI"):
                M["p", rd, s, L, gn] = wmean((LAB[L, s] == c) & ~np.isnan(p), p)[0]
            for sc in ("p", "verdict"):
                M["auc", rd, s, L, sc], NN["auc", rd, s, L, sc, 1], NN["auc", rd, s, L, sc, 0] = \
                    wauroc(GEN[rd, s][sc], LAB[L, s])
# Kitagawa decomposition over stage pairs
PAIRS = list(combinations(S2C, 2)) + [("2b", "2c-bm25_fuse3"), ("2b", HYB)]
for D in DEFS:
    for rd in RD:
        for a, b in PAIRS:
            if (rd, a) not in REF[D] or (rd, b) not in REF[D]:
                continue
            for L in LABELS:
                if not has(L, a, b):
                    continue
                c, e = kitagawa(M["w", D, rd, a, L], M["r", D, rd, a, L], M["w", D, rd, b, L], M["r", D, rd, b, L])
                M["dec", D, rd, a, b, L, "dR"] = M["tot", D, rd, b, L] - M["tot", D, rd, a, L]
                M["dec", D, rd, a, b, L, "comp"] = c
                M["dec", D, rd, a, b, L, "rate"] = e
# 2b/2c contrast: (def=v1, as in refusal_flips.py) refusal on I contexts of the retrieved stage minus
# refusal on S contexts of gold; (same_turn) the same difference, computed on the same turns
for D in DEFS:
    for rd in RD:
        for s in S2C:
            if (rd, s) not in REF[D] or (rd, "2b") not in REF[D]:
                continue
            r2c, rg = REF[D][rd, s], REF[D][rd, "2b"]
            for L in LABELS:
                if not has(L, s, "2b"):
                    continue
                M["con", D, rd, s, L, "v1"] = M["ref", D, rd, s, L, "I"] - M["ref", D, rd, "2b", L, "S"]
                T = (LAB[L, s] == 2) & (LAB[L, "2b"] == 0) & ~np.isnan(r2c) & ~np.isnan(rg)
                a, n = wmean(T, r2c)
                b, _ = wmean(T, rg)
                M["con", D, rd, s, L, "same_turn"] = a - b
                M["con_parts", D, rd, s, L] = (a, b, n)

OUT.mkdir(parents=True, exist_ok=True)


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def wcsv(name, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def ecols(e, pre=""):
    return {f"{pre}est" if pre else "est": r3(e["est"]), f"{pre}lo": r3(e["lo"]), f"{pre}hi": r3(e["hi"])}


# ── (1) refusal.csv: long format ───────────────────────────────────────────
rows_ref = []
for rd in RD:
    for s in STAGES:
        D = "verdict"
        r = REF[D][rd, s]
        base = {"reader": rd, "stage": NAME[s], "ndcg10": round(NDCG[s], 4), "n_empty": GEN[rd, s]["n_empty"],
                "n_odd_status": GEN[rd, s]["n_odd"]}
        e, pe = est(M["ref", D, rd, s, "all"]), est(M["p", rd, s, "all"])
        rows_ref.append({**base, "label": "overall", "class": "all", "n": N, "n_valid": int(NN[D, rd, s][0]),
                         "refusal": r3(e["est"]), "lo": r3(e["lo"]), "hi": r3(e["hi"]),
                         "refusal_v1": r3(M["ref", "verdict_v1", rd, s, "all"][0]),
                         "p_ans": r3(pe["est"]), "p_lo": r3(pe["lo"]), "p_hi": r3(pe["hi"])})
        for L in LABELS:
            if not has(L, s):
                continue
            g = LAB[L, s]
            for c, gn in enumerate("SPI"):
                n = int((g == c).sum())
                if n == 0:
                    continue
                e, pe = est(M["ref", D, rd, s, L, gn]), est(M["p", rd, s, L, gn])
                rows_ref.append({**base, "label": L, "class": CLS["qrels" if L == "qrels" else "content"][c], "n": n,
                                 "n_valid": int(((g == c) & ~np.isnan(r)).sum()),
                                 "refusal": r3(e["est"]), "lo": r3(e["lo"]), "hi": r3(e["hi"]),
                                 "refusal_v1": r3(M["ref", "verdict_v1", rd, s, L, gn][0]),
                                 "p_ans": r3(pe["est"]), "p_lo": r3(pe["lo"]), "p_hi": r3(pe["hi"])})
            if (g < 0).any():
                rows_ref.append({**base, "label": L, "class": "invalid", "n": int((g < 0).sum())})
wcsv("refusal.csv", rows_ref)

# ── (2) auroc.csv ──────────────────────────────────────────────────────────
rows_auc = []
for rd in RD:
    for s in STAGES:
        for L in LABELS:
            if not has(L, s) or (s == "2b" and L == "qrels"):
                continue
            for sc in ("p", "verdict"):
                e = est(M["auc", rd, s, L, sc])
                row = {"reader": rd, "stage": NAME[s], "ndcg10": round(NDCG[s], 4), "label": L, "score": sc,
                       "n_pos": NN["auc", rd, s, L, sc, 1], "n_neg": NN["auc", rd, s, L, sc, 0],
                       "auroc": r3(e["est"]), "lo": r3(e["lo"]), "hi": r3(e["hi"])}
                if L != "qrels" and s != "2b":
                    d = est(M["auc", rd, s, L, sc] - M["auc", rd, s, "qrels", sc])
                    row.update({"delta": r3(d["est"]), "d_lo": r3(d["lo"]), "d_hi": r3(d["hi"]), "d_sig": sig(d)})
                rows_auc.append(row)
wcsv("auroc.csv", rows_auc)

# ── (2b) auroc_common.csv: both labels on the same set of contexts ─────────
# qrels and content labels drop different middle classes (part / partial), so in auroc.csv
# the two AUROCs are computed on different contexts. Here we keep only contexts that are both
# all-or-none under qrels and yes-or-no under the content judge; same W -> paired difference.
rows_common = []
for rd in RD:
    for s in S2C:
        for CL in CONTENT:
            if not has(CL, s):
                continue
            q, c = LAB["qrels", s], LAB[CL, s]
            com = np.isin(q, (0, 2)) & np.isin(c, (0, 2))
            for sc in ("p", "verdict"):
                aq, nq1, nq0 = wauroc(GEN[rd, s][sc], np.where(com, q, -1))
                ac, nc1, nc0 = wauroc(GEN[rd, s][sc], np.where(com, c, -1))
                eq, ec, d = est(aq), est(ac), est(ac - aq)
                d0 = est(M["auc", rd, s, CL, sc] - M["auc", rd, s, "qrels", sc])
                rows_common.append({
                    "reader": rd, "stage": NAME[s], "ndcg10": round(NDCG[s], 4), "content_label": CL, "score": sc,
                    "n_common": int(com.sum()), "n_common_scored": int((com & ~np.isnan(GEN[rd, s][sc])).sum()),
                    "qrels_n_pos": nq1, "qrels_n_neg": nq0, "content_n_pos": nc1, "content_n_neg": nc0,
                    "auroc_qrels": r3(eq["est"]), "q_lo": r3(eq["lo"]), "q_hi": r3(eq["hi"]),
                    "auroc_content": r3(ec["est"]), "c_lo": r3(ec["lo"]), "c_hi": r3(ec["hi"]),
                    "delta": r3(d["est"]), "d_lo": r3(d["lo"]), "d_hi": r3(d["hi"]), "d_sig": sig(d),
                    "delta_own_masks": r3(d0["est"]), "own_lo": r3(d0["lo"]), "own_hi": r3(d0["hi"])})
wcsv("auroc_common.csv", rows_common)

# ── (3) contrast.csv ───────────────────────────────────────────────────────
rows_con = []
for D in DEFS:
    if D == "verdict_v1":
        continue
    for rd in RD:
        for s in S2C:
            for L in LABELS:
                for df in ("v1", "same_turn"):
                    k = ("con", D, rd, s, L, df)
                    if k not in M:
                        continue
                    e = est(M[k])
                    a, b, n = M["con_parts", D, rd, s, L]
                    row = {"refusal": D, "reader": rd, "stage": NAME[s], "label": L, "def": df,
                           "n_I": int((LAB[L, s] == 2).sum()) if df == "v1" else int(n[0]),
                           "ref_I": r3(M["ref", D, rd, s, L, "I"][0] if df == "v1" else a[0]),
                           "ref_gold": r3(M["ref", D, rd, "2b", L, "S"][0] if df == "v1" else b[0]),
                           "contrast": r3(e["est"]), "lo": r3(e["lo"]), "hi": r3(e["hi"]), "sig": sig(e)}
                    if L != "qrels":
                        q = est(M["con", D, rd, s, "qrels", df])
                        d = est(M[k] - M["con", D, rd, s, "qrels", df])
                        row.update({"sig_qrels": sig(q), "sig_flip": sig(q) != sig(e),
                                    "d_vs_qrels": r3(d["est"]), "d_lo": r3(d["lo"]), "d_hi": r3(d["hi"])})
                    rows_con.append(row)
wcsv("contrast.csv", rows_con)

# ── (4) decomp.csv: BM25-turn -> Fusion, BM25-turn -> Hybrid ───────────────
rows_dec = []
for D in DEFS:
    if D == "verdict_v1":
        continue
    for rd in RD:
        for a, b in (("2c-bm25_query", "2c-bm25_fuse3"), ("2c-bm25_query", HYB)):
            for L in LABELS:
                k = ("dec", D, rd, a, b, L)
                if k + ("dR",) not in M:
                    continue
                row = {"refusal": D, "reader": rd, "transition": f"{NAME[a]}->{NAME[b]}", "label": L}
                for part in ("dR", "comp", "rate"):
                    e = est(M[k + (part,)])
                    row.update({part: r3(e["est"]), f"{part}_lo": r3(e["lo"]), f"{part}_hi": r3(e["hi"]),
                                f"{part}_sig": sig(e)})
                row["comp_share"] = r3(row["comp"] / row["dR"]) if abs(row["dR"]) > .01 else None
                if L != "qrels":
                    for part in ("comp", "rate"):
                        d = est(M[k + (part,)] - M[("dec", D, rd, a, b, "qrels", part)])
                        row.update({f"{part}_c_minus_q": r3(d["est"]), f"{part}_cq_lo": r3(d["lo"]),
                                    f"{part}_cq_hi": r3(d["hi"])})
                rows_dec.append(row)
wcsv("decomp.csv", rows_dec)

# ── (5) conclusion flips: the comparison family of refusal_flips.py on 4 readers x 4 retrievers ──
AUCD = {"AUROC p_answerable": "p", "AUROC verdict": "verdict"}


def avail(D, rd, *ss):
    return all((rd, s) in REF[D] for s in ss)


def base(rd, s, D):
    """Quantities of one (reader, stage) cell as fn(L) -> vector."""
    out = {}
    if D in VERDICT_DEFS:
        for desc, sc in AUCD.items():
            out[desc] = lambda L, rd=rd, s=s, sc=sc: M["auc", rd, s, L, sc]
    out["refusal | insufficient"] = lambda L, rd=rd, s=s: M["ref", D, rd, s, L, "I"]
    out["refusal | sufficient"] = lambda L, rd=rd, s=s: M["ref", D, rd, s, L, "S"]
    out["refusal gap I-S"] = lambda L, rd=rd, s=s: M["ref", D, rd, s, L, "I"] - M["ref", D, rd, s, L, "S"]
    return out


comps = []  # (family, refusal, description, readers, stages, fn(L))
for D in DEFS:
    for s in S2C:
        for ra, rb in combinations(RD, 2):
            if not avail(D, ra, s) or not avail(D, rb, s):
                continue
            ba, bb = base(ra, s, D), base(rb, s, D)
            for desc in ba:
                comps.append(("reader", D, f"{rb} - {ra}, {desc}, {NAME[s]}", {ra, rb}, {s},
                              lambda L, fa=ba[desc], fb=bb[desc]: fb(L) - fa(L)))
    for rd in RD:
        for a, b in combinations(S2C, 2):
            if not avail(D, rd, a, b):
                continue
            ba, bb = base(rd, a, D), base(rd, b, D)
            for desc in ba:
                comps.append(("stage", D, f"{rd}: {NAME[b]} - {NAME[a]}, {desc}", {rd}, {a, b},
                              lambda L, fa=ba[desc], fb=bb[desc]: fb(L) - fa(L)))
        for s in S2C:
            if not avail(D, rd, s):
                continue
            if D in VERDICT_DEFS:
                comps.append(("chance", D, f"{rd}: AUROC p_answerable - 0.5, {NAME[s]}", {rd}, {s},
                              lambda L, rd=rd, s=s: M["auc", rd, s, L, "p"] - .5))
            comps.append(("chance", D, f"{rd}: refusal gap I-S, {NAME[s]}", {rd}, {s}, base(rd, s, D)["refusal gap I-S"]))
            if avail(D, rd, "2b"):
                comps.append(("gold-vs-2c", D, f"{rd}: refusal {NAME[s]}|I - gold|S", {rd}, {s, "2b"},
                              lambda L, rd=rd, s=s, D=D: M["ref", D, rd, s, L, "I"] - M["ref", D, rd, "2b", L, "S"]))
        for a, b in PAIRS:
            if not avail(D, rd, a, b):
                continue
            k = ("dec", D, rd, a, b)
            tr = f"{NAME[a]}->{NAME[b]}"
            comps.append(("decomp", D, f"{rd}: rate effect {tr}", {rd}, {a, b}, lambda L, k=k: M[k + (L, "rate")]))
            comps.append(("decomp", D, f"{rd}: composition effect {tr}", {rd}, {a, b},
                          lambda L, k=k: M[k + (L, "comp")]))
            comps.append(("decomp", D, f"{rd}: |comp| - |rate| {tr}", {rd}, {a, b},
                          lambda L, k=k: np.abs(M[k + (L, "comp")]) - np.abs(M[k + (L, "rate")])))


def kind(q, c, dd=None):
    sq, sc = np.sign(q["est"]), np.sign(c["est"])
    if sq != sc and (sig(q) or sig(c)):
        return "sign flip" + (" (both sig.)" if sig(q) and sig(c) else "")
    if sq != sc:
        return "sign flip (n.s.)"
    if sig(q) != sig(c):
        return "significance flip"
    if dd is not None and sig(dd):
        return "magnitude shift"
    return "consistent"


FLIPS = ("sign flip", "sign flip (both sig.)", "significance flip")
rows_flip = []
for fam, D, desc, rds, sts, fn in comps:
    for CL in CONTENT:
        try:
            vq, vc = fn("qrels"), fn(CL)
        except KeyError:  # this judge lacks the required stage
            continue
        q, c = est(vq), est(vc)
        if np.isnan(q["est"]) or np.isnan(c["est"]):
            continue
        dd = est(vc - vq)
        rows_flip.append({"family": fam, "refusal": D, "label": CL, "comparison": desc, "kind": kind(q, c, dd),
                          "refusal_based": "AUROC" not in desc,
                          "v1_subset": rds <= OLD and sts <= V1_STAGES,
                          "new_reader": bool(rds - OLD), "hybrid": HYB in sts,
                          "qrels": r3(q["est"]), "qrels_lo": r3(q["lo"]), "qrels_hi": r3(q["hi"]),
                          "content": r3(c["est"]), "content_lo": r3(c["lo"]), "content_hi": r3(c["hi"]),
                          "c_minus_q": r3(dd["est"]), "cq_lo": r3(dd["lo"]), "cq_hi": r3(dd["hi"])})
# noise threshold: the same comparison family, but phi-4 vs Qwen3-14B
# (same rule, different content judge)
JJ, JJ_FAM = Counter(), Counter()
if len(CONTENT) >= 2:
    for fam, D, desc, rds, sts, fn in comps:
        if D != "verdict":
            continue
        try:
            v1, v2 = fn(CONTENT[0]), fn(CONTENT[1])
        except KeyError:
            continue
        a, b = est(v1), est(v2)
        if np.isnan(a["est"]) or np.isnan(b["est"]):
            continue
        kk = kind(a, b, est(v2 - v1))
        JJ[kk] += 1
        if kk in ("sign flip", "sign flip (both sig.)", "significance flip"):
            JJ_FAM[fam] += 1
JK = {(r["family"], r["comparison"], r["label"], r["refusal"]): r["kind"] for r in rows_flip}
JDEFS = [D for D in DEFS if D.startswith("judge_")]
for r in rows_flip:
    if r["refusal"] in VERDICT_DEFS:
        for JD in JDEFS:
            r[f"kind_{JD}"] = JK.get((r["family"], r["comparison"], r["label"], JD))
wcsv("flips.csv", rows_flip)


def count(rows):
    fl = [r for r in rows if r["kind"] in FLIPS]
    rb = [r for r in fl if r["refusal_based"]]
    out = {"n": len(rows), "flips": len(fl), "sign": sum(r["kind"].startswith("sign flip") for r in fl),
           "significance": sum(r["kind"] == "significance flip" for r in fl),
           "magnitude_shift": sum(r["kind"] == "magnitude shift" for r in rows),
           "by_family": dict(Counter(r["family"] for r in fl)), "refusal_based_flips": len(rb)}
    for JD in JDEFS:
        el = [r for r in rb if r.get(f"kind_{JD}") is not None]
        out[f"survive_{JD}"] = f"{sum(r[f'kind_{JD}'] in FLIPS for r in el)}/{len(el)}"
    return out


SUMMARY = {"ndcg10": {NAME[s]: round(v, 4) for s, v in NDCG.items()}, "content_labels": CONTENT,
           "label_stages": {L: [NAME[s] for s in STAGES if has(L, s)] for L in LABELS},
           "n_invalid_labels": {f"{J}|{NAME[s]}": n for (J, s), n in N_INVALID.items() if n},
           "refusal_defs": DEFS, "flips": {}}
for CL in CONTENT:
    F = [r for r in rows_flip if r["label"] == CL]
    SUMMARY["flips"][CL] = {
        "v1_compat (verdict_v1, v1 subset)": count([r for r in F if r["refusal"] == "verdict_v1" and r["v1_subset"]]),
        "v1 subset, verdict (empty excluded)": count([r for r in F if r["refusal"] == "verdict" and r["v1_subset"]]),
        "extended, verdict": count([r for r in F if r["refusal"] == "verdict"]),
        "extended, verdict, outside v1 subset": count([r for r in F if r["refusal"] == "verdict" and not r["v1_subset"]]),
    }
if JJ:
    SUMMARY["flips_judge_vs_judge"] = {
        "labels": f"{CONTENT[0]} vs {CONTENT[1]} (same family as 'extended, verdict')",
        "n": sum(JJ.values()), "flips": sum(JJ[k] for k in FLIPS),
        "sign": JJ["sign flip"] + JJ["sign flip (both sig.)"], "significance": JJ["significance flip"],
        "magnitude_shift": JJ["magnitude shift"], "by_family": dict(JJ_FAM), "kinds": dict(JJ)}
for L in LABELS:
    SUMMARY.setdefault("insuff_share", {})[L] ={NAME[s]: r3(M["share", L, s, 2][0]) for s in STAGES if has(L, s)}
SUMMARY["overall_refusal"] = {rd: {NAME[s]: r3(M["ref", "verdict", rd, s, "all"][0]) for s in STAGES} for rd in RD}
SUMMARY["n_empty_status"] = {rd: {NAME[s]: GEN[rd, s]["n_empty"] for s in STAGES} for rd in RD}
json.dump(SUMMARY, open(OUT / "consequences_summary.json", "w"), ensure_ascii=False, indent=1)

print(json.dumps({k: SUMMARY[k] for k in ("ndcg10", "content_labels", "overall_refusal", "n_empty_status")}))
for CL, d in SUMMARY["flips"].items():
    for k, v in d.items():
        print(CL, "|", k, "|", json.dumps(v))
if "flips_judge_vs_judge" in SUMMARY:
    print("judge vs judge |", json.dumps(SUMMARY["flips_judge_vs_judge"]))
for r in rows_common:
    if r["score"] == "p":
        print(f"common {r['reader']:13s} {r['stage']:9s} {r['content_label']:9s} n={r['n_common']:3d} "
              f"q={r['auroc_qrels']} c={r['auroc_content']} d={r['delta']:+.3f} [{r['d_lo']:+.3f},{r['d_hi']:+.3f}] "
              f"(own masks {r['delta_own_masks']:+.3f})")
