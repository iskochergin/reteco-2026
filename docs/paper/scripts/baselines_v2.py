#!/usr/bin/env python3
"""Baseline answerability detectors under two labels, plus selective answering.

2c stages (first-stage top-5 = reader context): BM25-turn, BM25-hist, Fusion, Hybrid; gold = 2b.
Labels: qrels label: all = 1 (sufficient) / none = 0, part excluded;
content label of <judge>: yes = 1 / no = 0, partial excluded, null/other = invalid (excluded, counted in json).
Judges: phi-4 (primary) and Qwen3-14B; a judge's content label is used for a stage only if
suff/<judge>_<stage>_dev.jsonl has all 858 rows.

Detectors (higher = "context is sufficient"):
 (a) retrieval confidence from the stage's run.trec: top1 = score of the 1st document, mean5 = mean top-5 score,
     margin12 = top1 − top2. Normalisation: raw = as in run.trec; dz = z-score within (stage × domain)
     over dev turns, label-free. Why not per query: Fusion and Hybrid are already min-max normalised within
     each query (Hybrid top1 is almost always 1), while AUROC is computed over all 11 domains pooled; the BM25
     scale differs across domains, and dz removes that.
 (b) readers (OCC, Qwen3-1.7B, Qwen3-4B, Qwen3-8B): verdict (ANSWERABLE = 1) and p_answerable.
     Empty status (p_answerable = None) is left out of the sample; the number of such rows is reported in the json.
 (c) transfer probes from analysis/transfer_content.csv (OCC and the 3 BM25 stages only; point estimates only,
     no per-turn features, hence no CI).
Selective answering: the reader answers only if its verdict is ANSWERABLE. coverage = share answered among rows
with a verdict; risk = share insufficient (label 0) among answered rows with a binary label; base = the same
when answering everything (same rows with a verdict); reduction = base − risk.
Bootstrap over turns: 1000 resamples, seed 0, the same indices for all quantities (differences are paired),
percentile 95% CI, as in docs/paper/scripts/refusal_flips.py. Flip type: kind(), a copy of refusal_flips.kind.
CPU only, runs in seconds. Output: docs/paper/out/v2/{detectors.csv, detectors_pairs.csv, selective.csv,
baselines.json}.
"""
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pytrec_eval
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[3]
DRV = Path(os.environ.get("RETECO_DRV", ROOT / "data/drive/reteco"))  # override for testing only
DATA = ROOT / "data/reteco/track2_recor"
OUT = Path(os.environ.get("RETECO_OUT", ROOT / "docs/paper/out/v2"))
HYB = "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
STAGES = ["2b", "2c-bm25_query", "2c-bm25_hist", "2c-bm25_fuse3", f"2c-{HYB}"]
S2C = STAGES[1:]
NAME = {"2b": "gold", "2c-bm25_query": "BM25-turn", "2c-bm25_hist": "BM25-hist", "2c-bm25_fuse3": "Fusion",
        f"2c-{HYB}": "Hybrid"}
RUN = {"2c-bm25_query": "bm25_query_dev", "2c-bm25_hist": "bm25_hist_dev", "2c-bm25_fuse3": "bm25_fuse3_dev",
       f"2c-{HYB}": f"{HYB}_dev"}
READERS = {"OCC": "OCC-RAG-1.7B", "Qwen3-1.7B": "Qwen3-1.7B", "Qwen3-4B": "Qwen3-4B-Instruct-2507",
           "Qwen3-8B": "Qwen3-8B"}
JUDGES = ["phi-4", "Qwen3-14B"]
DOMAINS = sorted(p.name for p in DATA.iterdir() if p.is_dir())
B, SEED, NT = 1000, 0, 858
NAN = float("nan")


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def auroc(score, y):
    """Mann–Whitney: y=1 sufficient, y=0 insufficient; ties get the average rank."""
    m = ~np.isnan(score)
    s, y = score[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return NAN
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def complete(p):
    """A judge file counts as complete only if it has 858 parseable rows."""
    if not p.exists():
        return False
    try:
        return len(jl(p)) == NT
    except json.JSONDecodeError:
        return False


# ── labels ────────────────────────────────────────────────────────────────
phi = {s: {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"suff/phi-4_{s}_dev.jsonl")} for s in STAGES}
KEYS = sorted(phi["2b"])
N = len(KEYS)
assert N == NT and all(sorted(phi[s]) == KEYS for s in STAGES)
lab, counts, LABELS = {}, {}, {s: ["qrels"] for s in STAGES}
for s in STAGES:
    q = np.array([phi[s][k]["suff_qrels"] for k in KEYS])
    lab["qrels", s] = np.where(q == "all", 1.0, np.where(q == "none", 0.0, NAN))
    counts["qrels", s] = {"pos": int((q == "all").sum()), "neg": int((q == "none").sum()),
                          "partial": int((q == "part").sum()), "invalid": 0}
    for j in JUDGES:
        p = DRV / f"suff/{j}_{s}_dev.jsonl"
        if not complete(p):
            continue
        rows = {(r["domain"], r["topic_id"]): r for r in jl(p)}
        assert sorted(rows) == KEYS
        assert all(rows[k]["suff_qrels"] == phi[s][k]["suff_qrels"] for k in KEYS)
        c = np.array([str(rows[k].get("label")).strip().lower() for k in KEYS])
        L = f"content_{j}"
        lab[L, s] = np.where(c == "yes", 1.0, np.where(c == "no", 0.0, NAN))
        counts[L, s] = {"pos": int((c == "yes").sum()), "neg": int((c == "no").sum()),
                        "partial": int((c == "partial").sum()),
                        "invalid": int((~np.isin(c, ["yes", "no", "partial"])).sum())}
        LABELS[s].append(L)
CONTENT = sorted({L for s in STAGES for L in LABELS[s] if L != "qrels"}, key=lambda L: JUDGES.index(L[8:]))

# ── readers ───────────────────────────────────────────────────────────────
gen, ctx = {}, {}
for rd, f in READERS.items():
    for s in STAGES:
        g = {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"gen/{f}_{s}_dev.jsonl")}
        rs = [g[k] for k in KEYS]
        assert all(r["suff"] == phi[s][k]["suff_qrels"] for r, k in zip(rs, KEYS))
        gen[rd, s] = {
            "p": np.array([NAN if r["p_answerable"] is None else r["p_answerable"] for r in rs]),
            # empty status (p_answerable None) = no verdict; otherwise UN* = refusal
            "verdict": np.array([NAN if r["p_answerable"] is None else float(not r["status"].startswith("UN"))
                                 for r in rs]),
            "n_empty": int(sum(r["p_answerable"] is None for r in rs)),
            "n_nonstd": int(sum(r["status"] not in ("ANSWERABLE", "UNANSWERABLE", "") for r in rs)),
        }
        ctx[rd, s] = [tuple(r["context_ids"]) for r in rs]


# ── first stage: scores, nDCG check, and top-5 vs reader context check ────
def load_run(run_dir):
    sc = defaultdict(list)
    for d in DOMAINS:
        for ln in open(DRV / f"runs/{run_dir}/{d}/run.trec"):
            q, _, doc, rk, s, _ = ln.split()
            sc[d, q].append((int(rk), float(s), doc))
    return {k: sorted(v) for k, v in sc.items()}


def ndcg_macro(run):
    """pytrec_eval nDCG@10, a turn missing from the run = 0, macro over domains."""
    per = []
    for d in DOMAINS:
        qrels = defaultdict(dict)
        for ln in open(DATA / d / "qrels_dev.txt"):
            q, _, doc, rel = ln.split()
            qrels[q][doc] = int(rel)
        rn = {q: {} for q in qrels}
        for (dd, q), lst in run.items():
            if dd == d and q in rn:
                for _, s, doc in lst:
                    rn[q].setdefault(doc, s)
        res = pytrec_eval.RelevanceEvaluator(dict(qrels), {"ndcg_cut.10"}).evaluate(rn)
        per.append(np.mean([res.get(q, {}).get("ndcg_cut_10", 0.0) for q in qrels]))
    return float(np.mean(per))


DOM = np.array([k[0] for k in KEYS])
DET, FAMILY, NDCG, CTX_MISMATCH = {}, {}, {}, {}
for s in S2C:
    run = load_run(RUN[s])
    NDCG[NAME[s]] = round(ndcg_macro(run), 4)
    top = [[x[1] for x in run.get(k, [])] for k in KEYS]
    CTX_MISMATCH[NAME[s]] = int(sum(tuple(x[2] for x in run.get(k, [])[:5]) != ctx["OCC", s][i]
                                    for i, k in enumerate(KEYS)))
    feats = {"top1": np.array([t[0] if t else NAN for t in top]),
             "mean5": np.array([np.mean(t[:5]) if len(t) >= 5 else NAN for t in top]),
             "margin12": np.array([t[0] - t[1] if len(t) >= 2 else NAN for t in top])}
    for fn, x in feats.items():
        DET[f"ret_{fn}_raw", s] = x
        z = np.full(N, NAN)
        for d in DOMAINS:
            m = DOM == d
            sd = np.nanstd(x[m])
            z[m] = (x[m] - np.nanmean(x[m])) / sd if sd > 0 else 0.0
        DET[f"ret_{fn}_dz", s] = z
        FAMILY[f"ret_{fn}_raw"] = FAMILY[f"ret_{fn}_dz"] = "retrieval"
for rd in READERS:
    for s in STAGES:
        DET[f"{rd}_p", s] = gen[rd, s]["p"]
        DET[f"{rd}_verdict", s] = gen[rd, s]["verdict"]
        FAMILY[f"{rd}_p"] = FAMILY[f"{rd}_verdict"] = "reader"

# set used for detector ranking and pairwise comparisons: dz retrieval + both reader scores
RANKSET = [f"ret_{f}_dz" for f in ("top1", "mean5", "margin12")] + \
          [f"{rd}_{sc}" for rd in READERS for sc in ("p", "verdict")]

# ── metrics on an index sample ────────────────────────────────────────────
IDX = np.random.default_rng(SEED).integers(0, N, size=(B, N))
ALL = np.arange(N)


def metrics(idx):
    out = {}
    for (dn, s), x in DET.items():
        xs = x[idx]
        for L in LABELS[s]:
            y = lab[L, s][idx]
            m = ~np.isnan(y)
            out[f"auc|{dn}|{s}|{L}"] = auroc(xs[m], y[m])
    for rd in READERS:
        for s in STAGES:
            v = gen[rd, s]["verdict"][idx]
            valid = ~np.isnan(v)
            ans = valid & (v == 1)
            out[f"cov|{rd}|{s}"] = ans.sum() / valid.sum()
            for L in LABELS[s]:
                y = lab[L, s][idx]
                b = ~np.isnan(y)
                a, al = y[ans & b], y[valid & b]
                out[f"risk|{rd}|{s}|{L}"] = float((a == 0).mean()) if len(a) else NAN
                out[f"base|{rd}|{s}|{L}"] = float((al == 0).mean()) if len(al) else NAN
    return out


PT = metrics(ALL)
BOOT = defaultdict(list)
for i in range(B):
    for k, v in metrics(IDX[i]).items():
        BOOT[k].append(v)
BOOT = {k: np.array(v, float) for k, v in BOOT.items()}


def est(key_or_fn):
    """Point estimate and 95% CI for a metric or a difference (fn(dict) -> float)."""
    f = (lambda d: d[key_or_fn]) if isinstance(key_or_fn, str) else key_or_fn
    pt, bs = float(f(PT)), f(BOOT)
    if np.isnan(pt) or np.all(np.isnan(bs)):
        return {"est": NAN, "lo": NAN, "hi": NAN, "p_le0": NAN}
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": pt, "lo": float(lo), "hi": float(hi), "p_le0": float(np.nanmean(bs <= 0))}


def sig(e):
    return e["lo"] > 0 or e["hi"] < 0


def kind(q, c, dd=None):
    """Flip type; a copy of refusal_flips.kind."""
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


r3 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

# ── (1) detectors.csv ─────────────────────────────────────────────────────
rows_det = []
for (dn, s) in DET:
    for L in LABELS[s]:
        c = counts[L, s]
        if dn.endswith("_verdict") or dn.endswith("_p"):
            nvalid = ~np.isnan(DET[dn, s]) & ~np.isnan(lab[L, s])
            n_pos, n_neg = int((lab[L, s][nvalid] == 1).sum()), int((lab[L, s][nvalid] == 0).sum())
        else:
            n_pos, n_neg = c["pos"], c["neg"]
        e = est(f"auc|{dn}|{s}|{L}")
        if np.isnan(e["est"]):
            continue
        row = {"detector": dn, "family": FAMILY[dn], "stage": NAME[s], "label": L, "n_pos": n_pos, "n_neg": n_neg,
               "auroc": r3(e["est"]), "lo": r3(e["lo"]), "hi": r3(e["hi"]),
               "delta_vs_qrels": None, "d_lo": None, "d_hi": None, "p_d_le0": None, "source": "computed"}
        if L != "qrels" and not np.isnan(PT[f"auc|{dn}|{s}|qrels"]):
            d = est(lambda D, a=f"auc|{dn}|{s}|{L}", b=f"auc|{dn}|{s}|qrels": D[a] - D[b])
            row.update(delta_vs_qrels=r3(d["est"]), d_lo=r3(d["lo"]), d_hi=r3(d["hi"]), p_d_le0=r3(d["p_le0"]))
        rows_det.append(row)
# (c) transfer probes: point estimates only, from analysis/transfer_content.csv
# (the keys below must match the detector/train/label values stored in that CSV)
XNAME = {("OCC verdict", "—"): "xfer_OCC_verdict", ("p_status", "—"): "xfer_OCC_p_status",
         ("trace", "synthetic"): "xfer_trace_synth", ("probe", "synthetic"): "xfer_probe_synth",
         ("trace", "RETECO train"): "xfer_trace_retecotrain", ("probe", "RETECO train"): "xfer_probe_retecotrain"}
XLAB = {"qrels": "qrels", "content": "content_phi-4"}
XST = {"bm25_query": "2c-bm25_query", "bm25_hist": "2c-bm25_hist", "bm25_fuse3": "2c-bm25_fuse3"}
xt = defaultdict(dict)
for r in csv.DictReader(open(DRV / "analysis/transfer_content.csv", encoding="utf-8")):
    if r["label"] in XLAB and r["stage"] in XST:
        dn = XNAME[r["detector"], r["train"]] + (f"_L{r['layer']}" if r["layer"] else "")
        xt[dn, XST[r["stage"]]][XLAB[r["label"]]] = r
for (dn, s), v in xt.items():
    for L, r in v.items():
        dlt = float(r["auroc"]) - float(v["qrels"]["auroc"]) if L != "qrels" else None
        rows_det.append({"detector": dn, "family": "transfer", "stage": NAME[s], "label": L,
                         "n_pos": int(r["n_pos"]), "n_neg": int(r["n_neg"]), "auroc": r3(float(r["auroc"])),
                         "lo": None, "hi": None, "delta_vs_qrels": r3(dlt), "d_lo": None, "d_hi": None,
                         "p_d_le0": None, "source": "analysis/transfer_content.csv (point only)"})

# ── (2) detector ranking and pairwise flips ───────────────────────────────
rank_info, rows_pair = [], []
for s in S2C:
    for L in LABELS[s]:
        pts = {dn: PT[f"auc|{dn}|{s}|{L}"] for dn in RANKSET}
        bs = np.array([BOOT[f"auc|{dn}|{s}|{L}"] for dn in RANKSET])  # detector × resample
        best = np.nanargmax(bs, axis=0)
        order = sorted(pts, key=lambda d: -pts[d])
        info = {"stage": NAME[s], "label": L, "order": [f"{d}={pts[d]:.3f}" for d in order],
                "best": order[0], "p_best": round(float(np.mean(best == RANKSET.index(order[0]))), 3),
                "best_retrieval": max([d for d in RANKSET if d.startswith("ret_")], key=pts.get),
                "best_reader": max([d for d in RANKSET if not d.startswith("ret_")], key=pts.get)}
        if L != "qrels":
            q = np.array([PT[f"auc|{d}|{s}|qrels"] for d in RANKSET])
            c = np.array([PT[f"auc|{d}|{s}|{L}"] for d in RANKSET])
            bq = np.array([BOOT[f"auc|{d}|{s}|qrels"] for d in RANKSET])
            taus = [kendalltau(bq[:, i], bs[:, i]).statistic for i in range(B)]
            lo, hi = np.nanpercentile(taus, [2.5, 97.5])
            info["kendall_tau_vs_qrels"] = [round(float(kendalltau(q, c).statistic), 3), round(float(lo), 3), round(float(hi), 3)]
        rank_info.append(info)
    for L in [x for x in LABELS[s] if x != "qrels"]:
        for a, b in combinations(RANKSET, 2):
            mk = lambda LL, a=a, b=b: (lambda D: D[f"auc|{a}|{s}|{LL}"] - D[f"auc|{b}|{s}|{LL}"])
            q, c = est(mk("qrels")), est(mk(L))
            dd = est(lambda D, mk=mk: mk(L)(D) - mk("qrels")(D))
            fam = "retrieval-vs-reader" if (a.startswith("ret_") != b.startswith("ret_")) else \
                ("retrieval-vs-retrieval" if a.startswith("ret_") else "reader-vs-reader")
            rows_pair.append({"metric": "auroc", "family": fam, "stage": NAME[s], "label": L,
                              "comparison": f"{a} - {b}", "kind": kind(q, c, dd),
                              "qrels": r3(q["est"]), "qrels_lo": r3(q["lo"]), "qrels_hi": r3(q["hi"]),
                              "content": r3(c["est"]), "content_lo": r3(c["lo"]), "content_hi": r3(c["hi"]),
                              "c_minus_q": r3(dd["est"]), "cq_lo": r3(dd["lo"]), "cq_hi": r3(dd["hi"])})

# ── (3) selective answering ───────────────────────────────────────────────
rows_sel = []
for rd in READERS:
    for s in STAGES:
        cov = est(f"cov|{rd}|{s}")
        v = gen[rd, s]["verdict"]
        for L in LABELS[s]:
            y = lab[L, s]
            ans = (v == 1) & ~np.isnan(y)
            r, bse = est(f"risk|{rd}|{s}|{L}"), est(f"base|{rd}|{s}|{L}")
            red = est(lambda D, rd=rd, s=s, L=L: D[f"base|{rd}|{s}|{L}"] - D[f"risk|{rd}|{s}|{L}"])
            row = {"reader": rd, "stage": NAME[s], "label": L, "n_valid": int((~np.isnan(v)).sum()),
                   "n_empty_status": gen[rd, s]["n_empty"], "n_nonstd_status": gen[rd, s]["n_nonstd"],
                   "coverage": r3(cov["est"]), "cov_lo": r3(cov["lo"]), "cov_hi": r3(cov["hi"]),
                   "n_answered_labelled": int(ans.sum()), "n_answered_insuff": int((ans & (y == 0)).sum()),
                   "risk": r3(r["est"]), "risk_lo": r3(r["lo"]), "risk_hi": r3(r["hi"]),
                   "base_risk": r3(bse["est"]), "base_lo": r3(bse["lo"]), "base_hi": r3(bse["hi"]),
                   "reduction": r3(red["est"]), "red_lo": r3(red["lo"]), "red_hi": r3(red["hi"])}
            if L != "qrels":
                dr = est(lambda D, rd=rd, s=s, L=L: D[f"risk|{rd}|{s}|{L}"] - D[f"risk|{rd}|{s}|qrels"])
                rq = est(lambda D, rd=rd, s=s: D[f"base|{rd}|{s}|qrels"] - D[f"risk|{rd}|{s}|qrels"])
                ddr = est(lambda D, rd=rd, s=s, L=L: (D[f"base|{rd}|{s}|{L}"] - D[f"risk|{rd}|{s}|{L}"])
                          - (D[f"base|{rd}|{s}|qrels"] - D[f"risk|{rd}|{s}|qrels"]))
                row.update(d_risk_vs_qrels=r3(dr["est"]), drisk_lo=r3(dr["lo"]), drisk_hi=r3(dr["hi"]),
                           d_red_vs_qrels=r3(ddr["est"]), dred_lo=r3(ddr["lo"]), dred_hi=r3(ddr["hi"]),
                           kind_reduction=kind(rq, red, ddr) if s != "2b" else None)
            rows_sel.append(row)
# reader-order flips by risk (comparable coverage is not guaranteed; this is raw risk)
for s in S2C:
    for L in [x for x in LABELS[s] if x != "qrels"]:
        for a, b in combinations(READERS, 2):
            mk = lambda LL, a=a, b=b: (lambda D: D[f"risk|{a}|{s}|{LL}"] - D[f"risk|{b}|{s}|{LL}"])
            q, c = est(mk("qrels")), est(mk(L))
            dd = est(lambda D, mk=mk: mk(L)(D) - mk("qrels")(D))
            rows_pair.append({"metric": "selective_risk", "family": "reader-vs-reader", "stage": NAME[s], "label": L,
                              "comparison": f"{a} - {b}", "kind": kind(q, c, dd),
                              "qrels": r3(q["est"]), "qrels_lo": r3(q["lo"]), "qrels_hi": r3(q["hi"]),
                              "content": r3(c["est"]), "content_lo": r3(c["lo"]), "content_hi": r3(c["hi"]),
                              "c_minus_q": r3(dd["est"]), "cq_lo": r3(dd["lo"]), "cq_hi": r3(dd["hi"])})

# ── write outputs ──────────────────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)


def wcsv(name, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


wcsv("detectors.csv", rows_det)
wcsv("detectors_pairs.csv", rows_pair)
wcsv("selective.csv", rows_sel)
flip_counts = defaultdict(lambda: defaultdict(int))
for r in rows_pair:
    flip_counts[f"{r['metric']}|{r['family']}|{r['label']}"][r["kind"]] += 1
json.dump({
    "ndcg10_macro": NDCG, "n_turns": N, "bootstrap": B, "seed": SEED, "judges_used": CONTENT,
    "label_counts": {f"{NAME[s]}|{L}": counts[L, s] for s in STAGES for L in LABELS[s]},
    "reader_empty_status": {f"{rd}|{NAME[s]}": gen[rd, s]["n_empty"] for rd in READERS for s in STAGES},
    "reader_nonstd_status": {f"{rd}|{NAME[s]}": gen[rd, s]["n_nonstd"] for rd in READERS for s in STAGES
                             if gen[rd, s]["n_nonstd"]},
    "ctx_vs_run_top5_mismatch": CTX_MISMATCH,
    "rankings": rank_info, "flip_counts": {k: dict(v) for k, v in flip_counts.items()},
    "notes": ("AUROC: label 1 = sufficient (qrels all / content yes), 0 = insufficient (qrels none / content no); "
              "partial excluded; null content label = invalid, excluded. Retrieval scores: raw = run.trec score; "
              "dz = z-score within stage x domain over dev turns (label-free). Verdict: ANSWERABLE = 1; empty "
              "status excluded; non-standard status strings not starting with UN count as ANSWERABLE. "
              "Selective: answer iff verdict ANSWERABLE; risk = share insufficient among answered "
              "with a binary label; base = same when answering all rows with a verdict. Bootstrap over turns "
              "(1000, seed 0), percentile 95% CI, paired. Transfer rows: points from "
              "analysis/transfer_content.csv (subset: rows w/o status or > MAXLEN dropped), no CI."),
}, open(OUT / "baselines.json", "w"), ensure_ascii=False, indent=1)

# ── print summary ──────────────────────────────────────────────────────────
print("nDCG@10:", NDCG, "| ctx≠run top5:", CTX_MISMATCH, "| judges:", CONTENT)
for r in rank_info:
    print(r["stage"], r["label"], "best", r["best"], f"P(best)={r['p_best']}", "| ret", r["best_retrieval"],
          "| rd", r["best_reader"], "| tau", r.get("kendall_tau_vs_qrels"))
for k, v in flip_counts.items():
    print(k, dict(v))
