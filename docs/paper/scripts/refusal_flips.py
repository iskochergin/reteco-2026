"""Reader refusals along the retriever ladder (OCC and Qwen3-4B; gold and the three BM25 retrievers):
Kitagawa decomposition, detector AUROC under the qrels and content (phi-4) labels, and the list of
changed conclusions ("conclusion flips"). Not reported in the paper: consequences_v2.py extends this
analysis to more readers and four retrievers and is the version the paper uses. tab_audit_auroc, written
by tables_v2.py (also not in the paper), uses this script's audit output.
Stage names here: gold = 2b, query = BM25-turn, hist = BM25-hist, fuse3 = BM25-fusion.

Refusal = reader verdict UNANSWERABLE (the status field of the reader output);
an empty verdict counts as no refusal. Sensitivity check: refusal by the Qwen3-8B judge (response_type == refused).
Labels: qrels  all / part / none  (S / P / I);  content  yes / partial / no  (phi-4);
clean = no and the answer is not in the history. Binary metrics exclude P, but shares and refusal rates for P are reported.
Bootstrap over turns (1000 resamples, seed 0); the same resamples are used for all metrics -> differences are paired.
CPU only, runs in seconds. Output: docs/paper/out/refusal_flips.{json,tex}, refusal_{rates,decomp,auroc,flips}.csv
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
DRV = ROOT / "data/drive/reteco"
OUT = ROOT / "docs/paper/out"
READERS = {"OCC": "OCC-RAG-1.7B", "Qwen": "Qwen3-4B-Instruct-2507"}
STAGES = ["2b", "2c-bm25_query", "2c-bm25_hist", "2c-bm25_fuse3"]
NAME = {"2b": "gold", "2c-bm25_query": "query", "2c-bm25_hist": "hist", "2c-bm25_fuse3": "fuse3"}
RUN = {"2c-bm25_query": "bm25_query_dev", "2c-bm25_hist": "bm25_hist_dev", "2c-bm25_fuse3": "bm25_fuse3_dev"}
PAIRS = [("2c-bm25_query", "2c-bm25_hist"), ("2c-bm25_hist", "2c-bm25_fuse3"),
         ("2c-bm25_query", "2c-bm25_fuse3"), ("2b", "2c-bm25_fuse3")]
GRP = {"qrels": {"all": 0, "part": 1, "none": 2}, "content": {"yes": 0, "partial": 1, "no": 2}}
B, SEED = 1000, 0


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def auroc(score, y):
    """Mann–Whitney: y=1 sufficient, y=0 not; ties get the average rank."""
    m = ~np.isnan(score)
    s, y = score[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ── data ──────────────────────────────────────────────────────────────────
suff = {s: {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"suff/phi-4_{s}_dev.jsonl")} for s in STAGES}
KEYS = sorted(suff["2b"])
N = len(KEYS)
lab = {}  # (labeling, stage) -> int array: 0 S, 1 P, 2 I, -1 excluded (clean only)
for s in STAGES:
    rs = [suff[s][k] for k in KEYS]
    lab["qrels", s] = np.array([GRP["qrels"][r["suff_qrels"]] for r in rs])
    lab["content", s] = np.array([GRP["content"][r["label"]] for r in rs])
    # clean: "no" but the answer is in the history -> excluded
    lab["clean", s] = np.array([-1 if (r["label"] == "no" and r["answered_in_history"]) else GRP["content"][r["label"]]
                                for r in rs])
gen, jref = {}, {}
for rd, f in READERS.items():
    for s in STAGES:
        g = {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"gen/{f}_{s}_dev.jsonl")}
        j = {(r["domain"], r["topic_id"]): r for r in jl(DRV / f"judge/Qwen3-8B/{f}_{s}_dev.jsonl")}
        rs = [g[k] for k in KEYS]
        assert all(r["suff"] == suff[s][k]["suff_qrels"] for r, k in zip(rs, KEYS))
        gen[rd, s] = {
            "ref": np.array([r["status"].startswith("UN") for r in rs], float),
            "p": np.array([np.nan if r["p_answerable"] is None else r["p_answerable"] for r in rs]),
            "verdict": np.array([np.nan if r["p_answerable"] is None else float(not r["status"].startswith("UN"))
                                 for r in rs]),
            "n_noverdict": int(sum(r["p_answerable"] is None for r in rs)),
        }
        jref[rd, s] = np.array([j[k].get("response_type") == "refused" for k in KEYS], float)


# nDCG@10 per stage (macro over domains), for cross-checking
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
NDCG["2b"] = 1.0  # gold context

# ── metrics on a resample of indices ──────────────────────────────────────
IDX = np.random.default_rng(SEED).integers(0, N, size=(B, N))
ALL = np.arange(N)


def rates(refv, g, idx):
    """Group shares and refusal rate per group (0 S, 1 P, 2 I), plus overall refusal."""
    gi, ri = g[idx], refv[idx]
    m = gi >= 0
    cnt = np.bincount(gi[m], minlength=3).astype(float)
    sm = np.bincount(gi[m], weights=ri[m], minlength=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sm / cnt
    return cnt / cnt.sum(), r, ri.mean()


def kitagawa(wa, ra, wb, rb):
    """ΔR = Σ Δw·r̄ (composition) + Σ Δr·w̄ (rate); an empty group takes the other stage's rate."""
    ra = np.where(np.isnan(ra), rb, ra)
    rb = np.where(np.isnan(rb), ra, rb)
    ra, rb = np.nan_to_num(ra), np.nan_to_num(rb)
    comp = np.sum((wb - wa) * (ra + rb) / 2)
    rate = np.sum((rb - ra) * (wa + wb) / 2)
    return comp, rate


def metrics(idx):
    out = {}
    for s in STAGES:
        for L in ("qrels", "content", "clean"):
            w, _, _ = rates(np.zeros(N), lab[L, s], idx)
            for gi, gn in enumerate("SPI"):
                out[f"share|{L}|{s}|{gn}"] = w[gi]
    for rd in READERS:
        for s in STAGES:
            for D, refv in (("verdict", gen[rd, s]["ref"]), ("judge", jref[rd, s])):
                for L in ("qrels", "content", "clean"):
                    w, r, tot = rates(refv, lab[L, s], idx)
                    out[f"ref|{rd}|{s}|{D}|all"] = tot
                    for gi, gn in enumerate("SPI"):
                        out[f"ref|{rd}|{s}|{D}|{L}|{gn}"] = r[gi]
            for L in ("qrels", "content", "clean", "content3"):
                g = lab["content" if L == "content3" else L, s][idx]
                y = np.where(g == 0, 1.0, np.where((g == 2) | ((g == 1) & (L == "content3")), 0.0, np.nan))
                m = ~np.isnan(y)
                for sc in ("p", "verdict"):
                    out[f"auc|{rd}|{s}|{L}|{sc}"] = auroc(gen[rd, s][sc][idx][m], y[m])
        for D in ("verdict", "judge"):
            refv = {s: (gen[rd, s]["ref"] if D == "verdict" else jref[rd, s]) for s in STAGES}
            for a, b in PAIRS:
                for L in ("qrels", "content"):
                    wa, ra, ta = rates(refv[a], lab[L, a], idx)
                    wb, rb, tb = rates(refv[b], lab[L, b], idx)
                    c, e = kitagawa(wa, ra, wb, rb)
                    out[f"dec|{rd}|{a}>{b}|{D}|{L}|dR"] = tb - ta
                    out[f"dec|{rd}|{a}>{b}|{D}|{L}|comp"] = c
                    out[f"dec|{rd}|{a}>{b}|{D}|{L}|rate"] = e
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
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"est": pt, "lo": float(lo), "hi": float(hi), "p_le0": float(np.nanmean(bs <= 0))}


# ── (1) refusals by label and decomposition ───────────────────────────────
rows_rates = []
for rd in READERS:
    for s in STAGES:
        for D in ("verdict", "judge"):
            row = {"reader": rd, "stage": NAME[s], "refusal": D, "ndcg10": round(NDCG[s], 4),
                   "n_noverdict": gen[rd, s]["n_noverdict"], **{f"all": round(PT[f"ref|{rd}|{s}|{D}|all"], 3)}}
            for L in ("qrels", "content", "clean"):
                for gn in "SPI":
                    if L == "clean" and gn != "I":
                        continue
                    n = int(np.sum(lab[L, s] == "SPI".index(gn)))
                    e = est(f"ref|{rd}|{s}|{D}|{L}|{gn}") if n else None
                    row[f"{L}_{gn}_n"] = n
                    row[f"{L}_{gn}"] = None if e is None else round(e["est"], 3)
                    row[f"{L}_{gn}_ci"] = None if e is None else f"[{e['lo']:.3f}, {e['hi']:.3f}]"
            rows_rates.append(row)

rows_dec = []
for rd in READERS:
    for D in ("verdict", "judge"):
        for a, b in PAIRS:
            row = {"reader": rd, "refusal": D, "from": NAME[a], "to": NAME[b]}
            for L in ("qrels", "content"):
                for part in ("dR", "comp", "rate"):
                    e = est(f"dec|{rd}|{a}>{b}|{D}|{L}|{part}")
                    row[f"{L}_{part}"], row[f"{L}_{part}_lo"], row[f"{L}_{part}_hi"] = (
                        round(e["est"], 3), round(e["lo"], 3), round(e["hi"], 3))
                row[f"{L}_comp_share"] = round(row[f"{L}_comp"] / row[f"{L}_dR"], 2) if abs(row[f"{L}_dR"]) > .01 else None
            e = est(lambda d, rd=rd, a=a, b=b, D=D: d[f"dec|{rd}|{a}>{b}|{D}|content|rate"] - d[f"dec|{rd}|{a}>{b}|{D}|qrels|rate"])
            row["rate_c_minus_q"], row["rate_c_minus_q_ci"] = round(e["est"], 3), f"[{e['lo']:.3f}, {e['hi']:.3f}]"
            rows_dec.append(row)

# ── (2) AUROC ──────────────────────────────────────────────────────────────
rows_auc = []
for rd in READERS:
    for s in STAGES:
        for sc in ("p", "verdict"):
            row = {"reader": rd, "stage": NAME[s], "score": sc, "ndcg10": round(NDCG[s], 4)}
            for L in ("qrels", "content", "clean", "content3"):
                e = est(f"auc|{rd}|{s}|{L}|{sc}")
                row[L], row[f"{L}_lo"], row[f"{L}_hi"] = (round(e["est"], 3), round(e["lo"], 3), round(e["hi"], 3)) \
                    if not np.isnan(e["est"]) else (None, None, None)
            if s != "2b":
                e = est(lambda d, rd=rd, s=s, sc=sc: d[f"auc|{rd}|{s}|content|{sc}"] - d[f"auc|{rd}|{s}|qrels|{sc}"])
                row["d_c_q"], row["d_c_q_lo"], row["d_c_q_hi"], row["p_d_le0"] = (
                    round(e["est"], 3), round(e["lo"], 3), round(e["hi"], 3), round(e["p_le0"], 3))
            rows_auc.append(row)

# ── (3) conclusion flips: the same difference under qrels and under content ─
S2C = STAGES[1:]
AUC = {"auc_p": ("p", "AUROC p_answerable"), "auc_v": ("verdict", "AUROC verdict")}
REF = {"ref_I": ("I", "refusal | insufficient"), "ref_S": ("S", "refusal | sufficient")}


def gap(rd, s, L, D):  # how sensitive refusal is to sufficiency
    return lambda d: d[f"ref|{rd}|{s}|{D}|{L}|I"] - d[f"ref|{rd}|{s}|{D}|{L}|S"]


def base(rd, s, L, D):
    """All compared quantities of one (reader, stage) cell as fn(d)."""
    out = {desc: (lambda d, k=f"auc|{rd}|{s}|{L}|{sc}": d[k]) for sc, desc in AUC.values()} if D == "verdict" else {}
    for g, desc in REF.values():
        out[desc] = lambda d, k=f"ref|{rd}|{s}|{D}|{L}|{g}": d[k]
    out["refusal gap I-S"] = gap(rd, s, L, D)
    return out


comps = []  # (family, refusal, description, fn(L) -> fn(d))
for D in ("verdict", "judge"):
    for s in S2C:
        for desc in base("OCC", s, "qrels", D):
            comps.append(("reader", D, f"Qwen - OCC, {desc}, {NAME[s]}",
                          lambda L, s=s, desc=desc, D=D: lambda d: base("Qwen", s, L, D)[desc](d)
                          - base("OCC", s, L, D)[desc](d)))
    for rd in READERS:
        for a, b in PAIRS[:3]:
            for desc in base(rd, a, "qrels", D):
                comps.append(("stage", D, f"{rd}: {NAME[b]} - {NAME[a]}, {desc}",
                              lambda L, rd=rd, a=a, b=b, desc=desc, D=D: lambda d: base(rd, b, L, D)[desc](d)
                              - base(rd, a, L, D)[desc](d)))
        for s in S2C:
            if D == "verdict":
                comps.append(("chance", D, f"{rd}: AUROC p_answerable - 0.5, {NAME[s]}",
                              lambda L, rd=rd, s=s: lambda d: d[f"auc|{rd}|{s}|{L}|p"] - .5))
            comps.append(("chance", D, f"{rd}: refusal gap I-S, {NAME[s]}",
                          lambda L, rd=rd, s=s, D=D: gap(rd, s, L, D)))
            # 2b/2c pair from the original design: refusal on an "insufficient" retrieved context vs the gold context
            comps.append(("gold-vs-2c", D, f"{rd}: refusal {NAME[s]}|I - gold|S",
                          lambda L, rd=rd, s=s, D=D: lambda d: d[f"ref|{rd}|{s}|{D}|{L}|I"]
                          - d[f"ref|{rd}|2b|{D}|{L}|S"]))
        for a, b in PAIRS:
            k = f"dec|{rd}|{a}>{b}|{D}"
            comps.append(("decomp", D, f"{rd}: rate effect {NAME[a]}->{NAME[b]}",
                          lambda L, k=k: lambda d: d[f"{k}|{L}|rate"]))
            comps.append(("decomp", D, f"{rd}: composition effect {NAME[a]}->{NAME[b]}",
                          lambda L, k=k: lambda d: d[f"{k}|{L}|comp"]))
            comps.append(("decomp", D, f"{rd}: |comp| - |rate| {NAME[a]}->{NAME[b]}",
                          lambda L, k=k: lambda d: np.abs(d[f"{k}|{L}|comp"]) - np.abs(d[f"{k}|{L}|rate"])))


def sig(e):
    return e["lo"] > 0 or e["hi"] < 0


def kind(q, c, dd=None):
    sq, sc = np.sign(q["est"]), np.sign(c["est"])
    if sq != sc and (sig(q) or sig(c)):
        return "sign flip" + (" (both sig.)" if sig(q) and sig(c) else "")
    if sq != sc:
        return "sign flip (n.s.)"
    if sig(q) != sig(c):
        return "significance flip"
    if dd is not None and sig(dd):  # same sign and significance, but a different magnitude (paired c-q CI excludes 0)
        return "magnitude shift"
    return "consistent"


rows_flip = []
for fam, D, desc, mk in comps:
    q, c = est(mk("qrels")), est(mk("content"))
    try:
        cl = est(mk("clean"))
    except KeyError:  # no decomposition under clean (clean is not a partition)
        cl = {"est": np.nan, "lo": np.nan, "hi": np.nan}
    if np.isnan(q["est"]) or np.isnan(c["est"]):
        continue
    dd = est(lambda d, mk=mk: mk("content")(d) - mk("qrels")(d))
    rows_flip.append({"family": fam, "refusal": D, "comparison": desc, "kind": kind(q, c, dd),
                      "kind_clean": None if np.isnan(cl["est"]) else kind(q, cl),
                      "qrels": round(q["est"], 3), "qrels_ci": f"[{q['lo']:.3f}, {q['hi']:.3f}]",
                      "content": round(c["est"], 3), "content_ci": f"[{c['lo']:.3f}, {c['hi']:.3f}]",
                      "clean": round(cl["est"], 3), "c_minus_q": round(dd["est"], 3),
                      "c_minus_q_ci": f"[{dd['lo']:.3f}, {dd['hi']:.3f}]"})
# robustness to the refusal definition: same kind when refusal is taken from the Qwen3-8B judge
J = {(r["family"], r["comparison"]): r["kind"] for r in rows_flip if r["refusal"] == "judge"}
for r in rows_flip:
    if r["refusal"] == "verdict":
        r["kind_judge"] = J.get((r["family"], r["comparison"]))
# stage order by AUROC p_answerable for each reader
rank_rows = []
for rd in READERS:
    for L in ("qrels", "content"):
        order = sorted(S2C, key=lambda s: -PT[f"auc|{rd}|{s}|{L}|p"])
        rank_rows.append({"reader": rd, "label": L, "order": " > ".join(NAME[s] for s in order)})
for s in S2C:
    for L in ("qrels", "content"):
        best = max(READERS, key=lambda rd: PT[f"auc|{rd}|{s}|{L}|p"])
        rank_rows.append({"reader": "best", "label": L, "order": f"{NAME[s]}: {best}"})

# ── blind audit (100 fuse3 contexts): a third label on the same subset ─────
key = {r["n"]: (r["domain"], r["topic_id"]) for r in csv.DictReader(open(ROOT / "data/drive/audit/sample_key.csv"))}
aud = list(csv.DictReader(open(ROOT / "docs/audit_llm.csv")))
ak = [key[r["n"]] for r in aud]
pos = {k: i for i, k in enumerate(KEYS)}
ai = np.array([pos[k] for k in ak])
alab = {"qrels": lab["qrels", "2c-bm25_fuse3"][ai], "phi4": lab["content", "2c-bm25_fuse3"][ai],
        "claude": np.array([GRP["content"][r["final"]] for r in aud])}
AIDX = np.random.default_rng(SEED).integers(0, len(ai), size=(B, len(ai)))
audit, ABS, APT = [], {}, {}
for rd in READERS:
    g = gen[rd, "2c-bm25_fuse3"]
    for L, a in alab.items():
        def f(ix, a=a, g=g):
            y = np.where(a[ix] == 0, 1.0, np.where(a[ix] == 2, 0.0, np.nan))
            m = ~np.isnan(y)
            return auroc(g["p"][ai][ix][m], y[m]), g["ref"][ai][ix][a[ix] == 2].mean() if (a[ix] == 2).any() else np.nan
        pt = f(np.arange(len(ai)))
        bs = np.array([f(ix) for ix in AIDX])
        ABS[rd, L] = bs[:, 0]
        lo_, hi_ = np.nanpercentile(bs[:, 0], [2.5, 97.5])
        audit.append({"reader": rd, "label": L, "n_S": int((a == 0).sum()), "n_P": int((a == 1).sum()),
                      "n_I": int((a == 2).sum()), "auc_p": round(pt[0], 3),
                      "auc_ci": "[%.3f, %.3f]" % (lo_, hi_),
                      "ref_I": round(pt[1], 3),
                      # unrounded, for tables (avoids double rounding: .4645 -> .465 -> .47)
                      "auc_exact": float(pt[0]), "lo_exact": float(lo_), "hi_exact": float(hi_)})
        APT[rd, L] = pt[0]
for L in alab:
    dlt = ABS["Qwen", L] - ABS["OCC", L]
    lo_, hi_ = np.nanpercentile(dlt, [2.5, 97.5])
    audit.append({"reader": "Qwen-OCC", "label": L, "auc_p": round(APT["Qwen", L] - APT["OCC", L], 3),
                  "auc_ci": "[%.3f, %.3f]" % (lo_, hi_),
                  "auc_exact": float(APT["Qwen", L] - APT["OCC", L]), "lo_exact": float(lo_), "hi_exact": float(hi_)})

# ── output ─────────────────────────────────────────────────────────────────
OUT.mkdir(parents=True, exist_ok=True)


def wcsv(name, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


wcsv("refusal_rates.csv", rows_rates)
wcsv("refusal_decomp.csv", rows_dec)
wcsv("refusal_auroc.csv", rows_auc)
wcsv("refusal_flips.csv", rows_flip)
shares = {f"{NAME[s]}|{L}": {gn: round(PT[f"share|{L}|{s}|{gn}"], 3) for gn in "SPI"}
          for s in STAGES for L in ("qrels", "content")}
json.dump({"ndcg10": {NAME[s]: round(v, 4) for s, v in NDCG.items()}, "n_turns": N, "bootstrap": B, "seed": SEED,
           "shares": shares, "rates": rows_rates, "decomp": rows_dec, "auroc": rows_auc, "flips": rows_flip,
           "rankings": rank_rows, "audit100": audit},
          open(OUT / "refusal_flips.json", "w"), ensure_ascii=False, indent=1)


# ── LaTeX ──────────────────────────────────────────────────────────────────
def ci(e, nd=2):
    f = lambda x: f"{x:.{nd}f}".replace("0.", ".").replace("-.", "$-$.")
    return f"{f(e['est'])} {{\\scriptsize[{f(e['lo'])}, {f(e['hi'])}]}}"


def num(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    t = f"{x:.{nd}f}"
    t = t[1:] if t.startswith("-") and float(t) == 0 else t  # -0.00 -> 0.00
    return t.replace("0.", ".").replace("-.", "$-$.")


T = []
T.append(r"""% generated by docs/paper/scripts/refusal_flips.py — do not edit by hand
\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{llr cc cc cc ccc}\toprule
& & & \multicolumn{2}{c}{Insuff.\ share} & \multicolumn{2}{c}{Refusal $\mid$ suff.} & \multicolumn{2}{c}{Refusal $\mid$ insuff.} & \multicolumn{3}{c}{AUROC $p_{\text{ans}}$ (suff.\ vs insuff.)}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}
Reader & Stage & nDCG & qrels & content & qrels & content & qrels & content & qrels & content & $\Delta$ (c$-$q)\\\midrule""")
for rd in READERS:
    for s in STAGES:
        r = PT
        fs = lambda L, g: num(r[f"share|{L}|{s}|{g}"])
        rf = lambda L, g: num(r[f"ref|{rd}|{s}|verdict|{L}|{g}"])
        aq = est(f"auc|{rd}|{s}|qrels|p")
        ac = est(f"auc|{rd}|{s}|content|p")
        if s == "2b":
            dq, aqs = "--", "--"
            rq_i = "--"
        else:
            d = est(lambda dd, rd=rd, s=s: dd[f"auc|{rd}|{s}|content|p"] - dd[f"auc|{rd}|{s}|qrels|p"])
            dq, aqs, rq_i = ci(d), ci(aq), rf("qrels", "I")
        T.append(f"{rd if s == '2b' else ''} & {NAME[s]} & {num(NDCG[s])} & {fs('qrels', 'I')} & {fs('content', 'I')} & "
                 f"{rf('qrels', 'S')} & {rf('content', 'S')} & {rq_i} & {rf('content', 'I')} & {aqs} & {ci(ac)} & {dq}\\\\")
    T.append(r"\midrule" if rd == "OCC" else r"\bottomrule")
T.append(r"""\end{tabular}
\caption{Refusal (reader verdict \textsc{unanswerable}) and abstention-detector AUROC by retrieval stage under the
qrels label (\emph{all} vs \emph{none} gold in top-5) and the content label (phi-4: context \emph{yes}/\emph{no}
supports the reference; \emph{partial} excluded). gold = 2b (gold passages). 95\% bootstrap CIs over turns
(1000 resamples). $\Delta$ is paired.}\label{tab:refusal-auroc}
\end{table*}
""")
T.append(r"""\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll r rr rr}\toprule
& & & \multicolumn{2}{c}{qrels label} & \multicolumn{2}{c}{content label}\\\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Reader & Transition & $\Delta R$ & comp. & rate & comp. & rate\\\midrule""")
for rd in READERS:
    for i, (a, b) in enumerate(PAIRS):
        g = lambda L, p: est(f"dec|{rd}|{a}>{b}|verdict|{L}|{p}")
        cell = lambda e: num(e["est"]) + ("$^{*}$" if sig(e) else "")
        T.append(f"{rd if i == 0 else ''} & {NAME[a]}$\\to${NAME[b]} & {cell(g('qrels', 'dR'))} & "
                 f"{cell(g('qrels', 'comp'))} & {cell(g('qrels', 'rate'))} & {cell(g('content', 'comp'))} & "
                 f"{cell(g('content', 'rate'))}\\\\")
    T.append(r"\midrule" if rd == "OCC" else r"\bottomrule")
T.append(r"""\end{tabular}
\caption{Kitagawa decomposition of the change in refusal rate between stages into a composition effect (shares of
sufficient / partial / insufficient contexts) and a rate effect (refusal given the label). $^{*}$: 95\% bootstrap CI
excludes 0. comp.\ + rate $= \Delta R$.}\label{tab:refusal-decomp}
\end{table}
""")
FLIPS = ("sign flip", "sign flip (both sig.)", "significance flip")
flips = [r for r in rows_flip if r["refusal"] == "verdict" and r["kind"] in FLIPS]
# on the audit subset, does the blind-audit label reverse the reader order relative to phi-4?
_au = {(r["reader"], r["label"]): r["auc_p"] for r in audit}
AUDIT_REV = np.sign(_au["Qwen-OCC", "claude"]) != np.sign(_au["Qwen-OCC", "phi4"])
FAM = ["gold-vs-2c", "reader", "stage", "decomp", "chance"]
flips.sort(key=lambda r: (FAM.index(r["family"]), not r["kind"].startswith("sign"), r["comparison"]))


def cis(txt):
    lo, hi = (float(x) for x in txt.strip("[]").split(","))
    return f"{{\\scriptsize[{num(lo)}, {num(hi)}]}}"


T.append(r"""\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lll lll}\toprule
Family & Comparison & Type & qrels label & content label & content $-$ qrels\\\midrule""")
tx = lambda s: s.replace("_", r"\_").replace("|", r"$\mid$").replace("->", r"$\to$").replace(" - ", " $-$ ")
for r in flips:
    rob = "$^{\\dagger}$" if r.get("kind_judge") not in (None, *FLIPS) and "AUROC" not in r["comparison"] else ""
    if r["comparison"] == "Qwen - OCC, AUROC p_answerable, fuse3" and AUDIT_REV:
        rob += "$^{\\ddagger}$"
    T.append(f"{r['family']} & {tx(r['comparison'])}{rob} & {r['kind']} & "
             f"{num(r['qrels'])} {cis(r['qrels_ci'])} & {num(r['content'])} {cis(r['content_ci'])} & "
             f"{num(r['c_minus_q'])} {cis(r['c_minus_q_ci'])}\\\\")
T.append(r"""\bottomrule\end{tabular}
\caption{Conclusion flips: the same comparison (paired 95\% bootstrap CI over turns) changes sign or significance
when the qrels label (\emph{none} gold in top-5 = insufficient) is replaced by the content label (phi-4).
Refusal = reader verdict; $^{\dagger}$: not a flip when refusal is taken from the Qwen3-8B judge
(response\_type = refused); $^{\ddagger}$: reader order reverses under the blind-audit label on the 100-context
subset (Table~\ref{tab:refusal-audit}). ``sign flip'' requires at least one CI to exclude 0. gold = 2b.}\label{tab:refusal-flips}
\end{table*}
""")
# audit subset: same fuse3 stage, three labels (qrels / phi-4 / blind audit)
T.append(r"""\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{l ccc}\toprule
AUROC $p_{\text{ans}}$ & qrels & phi-4 & blind audit\\\midrule""")
AU = {(r["reader"], r["label"]): r for r in audit}
nI = " / ".join(str(AU["OCC", L]["n_I"]) for L in ("qrels", "phi4", "claude"))
for rd in ("OCC", "Qwen", "Qwen-OCC"):
    T.append(f"{rd.replace('-', ' $-$ ')} & " + " & ".join(
        f"{num(AU[rd, L]['auc_p'])} {cis(AU[rd, L]['auc_ci'])}" for L in ("qrels", "phi4", "claude")) + "\\\\")
T.append(r"""\bottomrule\end{tabular}
\caption{Audit subset: 100 fuse3 contexts (50 \emph{all}, 50 \emph{none} by qrels), detector AUROC under three
labels; insufficient contexts per label: """ + nI + r""". 95\% bootstrap CIs over the 100 contexts.}
\label{tab:refusal-audit}
\end{table}
""")
(OUT / "refusal_flips.tex").write_text("\n".join(T), encoding="utf-8")

# ── print summary ──────────────────────────────────────────────────────────
print("nDCG@10:", {NAME[s]: round(v, 4) for s, v in NDCG.items()})
print("flip kinds:", {k: sum(r["kind"] == k for r in rows_flip) for k in sorted({r["kind"] for r in rows_flip})})
for r in rank_rows:
    print("rank", r)
