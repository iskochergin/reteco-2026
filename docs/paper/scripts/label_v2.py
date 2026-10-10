#!/usr/bin/env python3
"""Table 1, Table 5 and Figure 2: qrels-label bias on the four retrievers (BM25-turn, BM25-hist,
BM25-fusion, hybrid), both judges and the blind audit.

In code and output files, a "stage" is one of the four retrievers (bm25_query = BM25-turn,
bm25_hist = BM25-hist, bm25_fuse3 = BM25-fusion, hybrid), or "gold" for the gold (2b) context.
The annotator "claude" is the blind audit (Claude Opus 5.5; runs A and B, disagreements adjudicated).

Definitions:
  qrels "insufficient" = suff_qrels == none (no gold in top 5); "actually insufficient" = judge says no.
  P, R, F1: partial excluded; P_partial_insuff / P_partial_suff: partial counted as "insufficient" / "sufficient".
  fm (false miss) = share of none that the judge labels yes; denominator = all none (partial included).
  ratio = share_none / share_no.
  fpr (false alarm) = share of yes that qrels labels none: n_fm / n_yes (partial excluded);
  youden = R - fpr; n_disagree = (none & yes) + (not none & no): label disagreements without partial.
  share_no_or_partial / ratio_partial_insuff: partial counted as "insufficient".
  P_at_turn_rates (label_v2.json base_rate_only_precision): precision at the BM25-turn R and fpr with
  this stage's share of no among yes/no, i.e. how much of the precision drop the base rate alone explains.
  A null judge label is invalid: such turns are dropped from the labels (n_invalid in the table).
Judges: every suff/<judge>_<job>_dev.jsonl; a file is used only if it has exactly 858 rows
(incomplete files are skipped and reported).
Bootstrap: paired over turns, stratified by domain, 1000 replicates, seed 0;
p_ge0 / p_le0 = share of replicates with Δ >= 0 / Δ <= 0. Audit and agreement: bootstrap over items.
"Gold present but judge says no": unstratified bootstrap over turns, with the same resamples as
false_miss_anatomy_v2.py (turns in the row order of the phi-4 BM25-turn file).

Output (docs/paper/out/v2/): label_stage.csv, label_deltas.csv, label_domains.csv,
gold_present_no.csv, audit_stage.csv, agreement.csv, label_v2.json.
"""
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pytrec_eval

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
RUNS = os.path.join(ROOT, "data", "drive", "reteco", "runs")
SUFF = os.path.join(ROOT, "data", "drive", "reteco", "suff")
OUT = os.path.join(ROOT, "docs", "paper", "out", "v2")
DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics", "sustainable_living"]
HYB = "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
# stage, job, run dir, display
STAGES = [("bm25_query", "2c-bm25_query", "bm25_query_dev", "BM25-turn"),
          ("bm25_hist", "2c-bm25_hist", "bm25_hist_dev", "BM25-hist"),
          ("bm25_fuse3", "2c-bm25_fuse3", "bm25_fuse3_dev", "Fusion"),
          ("hybrid", f"2c-{HYB}", f"{HYB}_dev", "Hybrid")]
GOLD = ("gold", "2b", None, "gold")
ST = [s[0] for s in STAGES]
JOB = {s[0]: s[1] for s in STAGES + [GOLD]}
DISP = {s[0]: s[3] for s in STAGES + [GOLD]}
PAIRS = [("bm25_hist", "bm25_query"), ("bm25_fuse3", "bm25_hist"), ("hybrid", "bm25_fuse3"),
         ("bm25_fuse3", "bm25_query"), ("hybrid", "bm25_query")]
L3 = ["yes", "partial", "no"]
B, SEED, N = 1000, 0, 858
NAN = float("nan")


# ───────────────────────── data ─────────────────────────
def per_turn_ndcg(run_dir):
    """Per-turn nDCG@10 as in scripts/score.py: a turn missing from the run scores 0."""
    out = {}
    for d in DOMAINS:
        qrels = defaultdict(dict)
        for ln in open(os.path.join(DATA, d, "qrels_dev.txt")):
            p = ln.split()
            if len(p) == 4:
                qrels[p[0]][p[2]] = int(p[3])
        run = defaultdict(dict)
        for ln in open(os.path.join(RUNS, run_dir, d, "run.trec")):
            p = ln.split()
            if len(p) == 6 and p[2] not in run[p[0]]:
                run[p[0]][p[2]] = float(p[4])
        ev = pytrec_eval.RelevanceEvaluator(dict(qrels), {"ndcg_cut.10"}).evaluate(dict(run))
        for t in qrels:
            out[(d, t)] = ev.get(t, {}).get("ndcg_cut_10", 0.0)
    return out


def load_judges():
    """{judge: {stage: {key: row}}}; only files with exactly 858 rows."""
    jobs = {v: k for k, v in JOB.items()}
    found, skipped = defaultdict(dict), []
    for path in sorted(glob.glob(os.path.join(SUFF, "*_dev.jsonl"))):
        base = os.path.basename(path)[:-len("_dev.jsonl")]
        hit = [(j, s) for j, s in jobs.items() if base.endswith("_" + j)]
        if not hit:
            continue
        job, stage = max(hit, key=lambda x: len(x[0]))
        judge = base[:-len(job) - 1]
        try:
            rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        except json.JSONDecodeError:
            skipped.append((judge, job, "bad json"))
            continue
        if len(rows) != N:
            skipped.append((judge, job, len(rows)))
            continue
        found[judge][stage] = {(r["domain"], r["topic_id"]): r for r in rows}
    order = sorted(found, key=lambda j: (j != "phi-4", j))
    return {j: found[j] for j in order}, skipped


def lab_of(r):
    x = r.get("label")
    return x if x in L3 else "invalid"


# ───────────────────────── metrics (invalid labels dropped) ─────────────────────────
def metrics(q, lab, nd, dom):
    v = lab != "invalid"
    n_inv = int((~v).sum())
    q, lab, dv = q[v], lab[v], dom[v]
    none = q == "none"
    yes, par, no = lab == "yes", lab == "partial", lab == "no"
    tp, fp, fn = (none & no).sum(), (none & yes).sum(), (~none & no).sum()
    p = tp / (tp + fp) if tp + fp else NAN
    r = tp / (tp + fn) if tp + fn else NAN
    n_none = none.sum()
    n_yes = yes.sum()
    fpr = fp / n_yes if n_yes else NAN
    nop = no | par
    m = {"n": int(len(q)), "n_invalid": n_inv, "n_none": int(n_none), "n_no": int(no.sum()),
         "n_partial": int(par.sum()), "n_fm": int(fp), "n_yes": int(n_yes), "n_tp": int(tp), "n_fn": int(fn),
         "n_disagree": int(fp + fn), "fpr": fpr, "youden": r - fpr,
         "base_no": no.sum() / (no.sum() + n_yes) if no.sum() + n_yes else NAN,
         "share_no_or_partial": nop.mean() if len(q) else NAN,
         "ratio_partial_insuff": none.mean() / nop.mean() if nop.any() and none.any() else NAN,
         "share_none": none.mean() if len(q) else NAN, "share_no": no.mean() if len(q) else NAN,
         "share_partial": par.mean() if len(q) else NAN,
         "ratio": none.mean() / no.mean() if no.any() and none.any() else NAN,
         "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r > 0 else (0.0 if tp + fn else NAN),
         "P_partial_insuff": (none & (no | par)).sum() / n_none if n_none else NAN,
         "P_partial_suff": tp / n_none if n_none else NAN,
         "fm": fp / n_none if n_none else NAN,
         "fm_or_partial": (none & (yes | par)).sum() / n_none if n_none else NAN}
    if nd is not None:
        m["ndcg"] = float(np.mean([nd[dom == d].mean() for d in np.unique(dom)]))
    return {k: (x if isinstance(x, int) else float(x)) for k, x in m.items()}


def kappa(a, b, cats):
    k, idx = len(cats), {c: i for i, c in enumerate(cats)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return NAN
    w = np.ones((k, k)) - np.eye(k)
    de = (w * np.outer(m.sum(1), m.sum(0)) / n).sum()
    return float(1 - (w * m).sum() / de) if de else NAN


def gwet_ac1(a, b, cats):
    n, k = len(a), len(cats)
    if n == 0:
        return NAN
    po = np.mean([x == y for x, y in zip(a, b)])
    pi = [(a.count(c) + b.count(c)) / (2 * n) for c in cats]
    pe = sum(p * (1 - p) for p in pi) / (k - 1)
    return float((po - pe) / (1 - pe)) if pe != 1 else NAN


def agree_stats(a, b):
    a, b = list(a), list(b)
    ba = ["yes" if x == "yes" else "not" for x in a]
    bb = ["yes" if x == "yes" else "not" for x in b]
    st = [(x, y) for x, y in zip(a, b) if x != "partial" and y != "partial"]
    sa, sb = [x for x, _ in st], [y for _, y in st]
    return {"n": len(a), "agree3": float(np.mean([x == y for x, y in zip(a, b)])) if a else NAN,
            "kappa3": kappa(a, b, L3), "ac1_3": gwet_ac1(a, b, L3),
            "agree_yes_rest": float(np.mean([x == y for x, y in zip(ba, bb)])) if a else NAN,
            "kappa_yes_rest": kappa(ba, bb, ["yes", "not"]), "ac1_yes_rest": gwet_ac1(ba, bb, ["yes", "not"]),
            "n_strict": len(st), "agree_yes_no": float(np.mean([x == y for x, y in st])) if st else NAN,
            "kappa_yes_no": kappa(sa, sb, ["yes", "no"]), "ac1_yes_no": gwet_ac1(sa, sb, ["yes", "no"])}


def pct(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [NAN, NAN]


R4 = lambda x: round(x, 4) if isinstance(x, float) else x


def write_csv(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: R4(x) for k, x in r.items()})


# ───────────────────────── main ─────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    judges, skipped = load_judges()
    assert "phi-4" in judges and all(s in judges["phi-4"] for s in ST + ["gold"]), "phi-4 must be complete"
    keys = sorted(judges["phi-4"]["gold"])
    assert len(keys) == N
    dom = np.array([k[0] for k in keys])
    conv = np.array([f"{k[0]}|{k[1].rsplit('_turn_', 1)[0]}" for k in keys])
    nd = {s: per_turn_ndcg(run) for s, _, run, _ in STAGES}
    for s in ST:
        assert set(nd[s]) == set(keys)
    ndarr = {s: np.array([nd[s][k] for k in keys]) for s in ST}

    # there is one qrels label per stage: check that it is identical across judges
    qarr, labarr, gic, ngold = {}, defaultdict(dict), {}, {}
    for j, by in judges.items():
        for s, rows in by.items():
            assert sorted(rows) == keys, (j, s)
            q = np.array([rows[k]["suff_qrels"] for k in keys])
            if s in qarr:
                assert (qarr[s] == q).all(), f"suff_qrels differ: {j} {s}"
            qarr[s] = q
            gic[s] = np.array([rows[k]["gold_in_ctx"] for k in keys])
            ngold[s] = np.array([rows[k]["n_gold"] for k in keys])
            labarr[j][s] = np.array([lab_of(rows[k]) for k in keys])
    JS = [(j, s) for j in judges for s in ST if s in labarr[j]]

    point = {(j, s): metrics(qarr[s], labarr[j][s], ndarr[s], dom) for j, s in JS}
    gold_pt = {j: metrics(qarr["gold"], labarr[j]["gold"], None, dom) for j in judges if "gold" in labarr[j]}

    # paired bootstrap over turns, stratified by domain (one draw per domain per replicate, in DOMAINS order)
    rng = np.random.default_rng(SEED)
    by_dom = [np.where(dom == d)[0] for d in DOMAINS]
    DK = ("P", "R", "F1", "fm", "ratio", "share_none", "share_no", "ndcg", "fpr", "youden")
    boot, dboot = defaultdict(list), defaultdict(list)
    for _ in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in by_dom])
        res = {(j, s): metrics(qarr[s][idx], labarr[j][s][idx], ndarr[s][idx], dom[idx]) for j, s in JS}
        for js, m in res.items():
            for k, x in m.items():
                boot[js + (k,)].append(x)
        for j in judges:
            for a, b in PAIRS:
                if (j, a) in res and (j, b) in res:
                    for k in DK:
                        dboot[(j, a, b, k)].append(res[(j, a)][k] - res[(j, b)][k])
    ci = {key: pct(v) for key, v in boot.items()}

    # conversation-cluster bootstrap: robustness of the CIs for P and fm
    rng = np.random.default_rng(SEED)
    convs = np.unique(conv)
    members = {c: np.where(conv == c)[0] for c in convs}
    cboot = defaultdict(list)
    for _ in range(B):
        idx = np.concatenate([members[c] for c in rng.choice(convs, len(convs))])
        for j, s in JS:
            m = metrics(qarr[s][idx], labarr[j][s][idx], None, dom[idx])
            for k in ("P", "fm", "fpr", "youden"):
                cboot[(j, s, k)].append(m[k])
    ci_conv = {key: pct(v) for key, v in cboot.items()}

    # precision at the BM25-turn R and fpr with this stage's base rate of no (P = Rπ / (Rπ + fpr(1-π)))
    brp = {}
    for j, s in JS:
        t = point.get((j, "bm25_query"))
        pi = point[(j, s)]["base_no"]
        brp[(j, s)] = (t["R"] * pi / (t["R"] * pi + t["fpr"] * (1 - pi))) if t else NAN

    # conversation-cluster vs per-turn bootstrap: how far the CI bounds shift
    def conv_shift(k):
        sh = []
        for j, s in JS:
            for b, (x, y) in enumerate(zip(ci[(j, s, k)], ci_conv[(j, s, k)])):
                sh.append((abs(y - x), j, s, ("lo", "hi")[b], x, y))
        mx = max(sh)
        adj = []
        for j in judges:
            for a, b in zip(ST, ST[1:]):
                if (j, a) in point and (j, b) in point:
                    dis = lambda c: c[(j, a, k)][0] > c[(j, b, k)][1] or c[(j, b, k)][0] > c[(j, a, k)][1]
                    adj.append({"judge": j, "pair": f"{a}|{b}", "disjoint_turn": bool(dis(ci)),
                                "disjoint_conv": bool(dis(ci_conv))})
        return {"max_abs_shift": mx[0], "at": {"judge": mx[1], "stage": mx[2], "bound": mx[3], "turn": mx[4],
                                                 "conv": mx[5]},
                "max_abs_shift_by_judge": {j: max(x[0] for x in sh if x[1] == j) for j in judges},
                "adjacent_stages": adj,
                "disjoint_turn_stays_disjoint_conv": all(a["disjoint_conv"] for a in adj if a["disjoint_turn"])}
    conv_json = {k: conv_shift(k) for k in ("P", "fm")}

    # ── label_stage.csv
    rows = []
    for j, s in JS:
        m = point[(j, s)]
        r = {"judge": j, "stage": s, "display": DISP[s], "ndcg": m["ndcg"],
             "ndcg_lo": ci[(j, s, "ndcg")][0], "ndcg_hi": ci[(j, s, "ndcg")][1],
             "n": m["n"], "n_invalid": m["n_invalid"], "n_none": m["n_none"], "n_no": m["n_no"],
             "n_partial": m["n_partial"], "n_fm": m["n_fm"]}
        for k in ("share_none", "share_no"):
            r.update({k: m[k], k + "_lo": ci[(j, s, k)][0], k + "_hi": ci[(j, s, k)][1]})
        r["share_partial"] = m["share_partial"]
        for k in ("ratio", "fm", "P", "R", "F1"):
            r.update({k: m[k], k + "_lo": ci[(j, s, k)][0], k + "_hi": ci[(j, s, k)][1]})
        r.update({"P_partial_insuff": m["P_partial_insuff"], "P_partial_suff": m["P_partial_suff"],
                  "fm_or_partial": m["fm_or_partial"],
                  "P_conv_lo": ci_conv[(j, s, "P")][0], "P_conv_hi": ci_conv[(j, s, "P")][1]})
        # false alarm on yes, Youden index, counts, partial counted as "insufficient"
        r.update({"n_yes": m["n_yes"], "n_tp": m["n_tp"], "n_fn": m["n_fn"], "n_disagree": m["n_disagree"]})
        for k in ("fpr", "youden"):
            r.update({k: m[k], k + "_lo": ci[(j, s, k)][0], k + "_hi": ci[(j, s, k)][1]})
        r.update({"base_no": m["base_no"], "P_at_turn_rates": brp[(j, s)],
                  "share_no_or_partial": m["share_no_or_partial"],
                  "ratio_partial_insuff": m["ratio_partial_insuff"],
                  "fm_conv_lo": ci_conv[(j, s, "fm")][0], "fm_conv_hi": ci_conv[(j, s, "fm")][1]})
        rows.append(r)
    write_csv("label_stage.csv", rows)

    # ── label_deltas.csv
    drows = []
    for j in judges:
        for a, b in PAIRS:
            if (j, a) not in point or (j, b) not in point:
                continue
            for k in DK:
                v = np.array(dboot[(j, a, b, k)], dtype=float)
                v = v[~np.isnan(v)]
                lo, hi = pct(v)
                drows.append({"judge": j, "comparison": f"{a} - {b}", "metric": k,
                              "delta": point[(j, a)][k] - point[(j, b)][k], "lo": lo, "hi": hi,
                              "p_ge0": float(np.mean(v >= 0)), "p_le0": float(np.mean(v <= 0)), "n_boot": len(v)})
    write_csv("label_deltas.csv", drows)

    # ── label_domains.csv: P by domain × stage
    dm, domrows = defaultdict(dict), []
    for j, s in JS:
        for d in DOMAINS:
            sel = dom == d
            m = metrics(qarr[s][sel], labarr[j][s][sel], ndarr[s][sel], dom[sel])
            dm[j][(d, s)] = m
            domrows.append({"judge": j, "domain": d, "stage": s, "n": m["n"], "ndcg": m["ndcg"],
                            "n_none": m["n_none"], "n_no": m["n_no"], "share_none": m["share_none"],
                            "share_no": m["share_no"], "P": m["P"], "fm": m["fm"]})
    write_csv("label_domains.csv", domrows)
    dom_summary = {}
    for j in judges:
        def cnt(a, b):
            if (DOMAINS[0], a) not in dm[j] or (DOMAINS[0], b) not in dm[j]:
                return None
            pairs = [(dm[j][(d, b)]["P"], dm[j][(d, a)]["P"]) for d in DOMAINS]
            ok = [(x, y) for x, y in pairs if not (np.isnan(x) or np.isnan(y))]
            return {"decrease": sum(y < x for x, y in ok), "n_domains": len(ok),
                    "not_decreasing": [d for d, (x, y) in zip(DOMAINS, pairs) if not (y < x)]}
        mono = None
        if all((DOMAINS[0], s) in dm[j] for s in ST):
            ok_d = [d for d in DOMAINS if not any(np.isnan([dm[j][(d, s)]["P"] for s in ST]))]
            mono_d = [d for d in ok_d if all(dm[j][(d, x)]["P"] >= dm[j][(d, y)]["P"] for x, y in zip(ST, ST[1:]))]
            mono = {"domains_P_nonincreasing_over_4_stages": len(mono_d), "n_domains": len(ok_d),
                    "exceptions": [d for d in ok_d if d not in mono_d]}
        dom_summary[j] = {"turn_to_hybrid": cnt("hybrid", "bm25_query"),
                          "turn_to_fusion": cnt("bm25_fuse3", "bm25_query"),
                          "fusion_to_hybrid": cnt("hybrid", "bm25_fuse3"), "monotone": mono}

    # ── gold_present_no.csv: among judge-"no" turns, those with >=1 / all gold passages in context
    # turns are resampled in the row order of the phi-4 bm25_query file (v1order), the order that
    # false_miss_anatomy_v2.py uses, so that the CIs of the two scripts match
    pos = {k: i for i, k in enumerate(keys)}
    v1order = np.array([pos[(r["domain"], r["topic_id"])] for r in map(
        json.loads, open(os.path.join(SUFF, "phi-4_2c-bm25_query_dev.jsonl")))])
    rng = np.random.default_rng(SEED)
    BOOT = v1order[rng.integers(0, N, size=(B, N))]

    def boot_share(sel, pred):
        sel, pred = np.asarray(sel), np.asarray(pred) & np.asarray(sel)
        bs = [pred[i].sum() / s for i in BOOT if (s := sel[i].sum()) > 0]
        lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (NAN, NAN)
        return float(pred.sum() / max(sel.sum(), 1)), float(lo), float(hi)

    gprows = []
    for j, s in JS:
        no = labarr[j][s] == "no"
        ge1 = qarr[s] != "none"
        allg = qarr[s] == "all"
        sh, lo, hi = boot_share(no, ge1)
        sha, loa, hia = boot_share(no, allg)
        g2b = Counter(labarr[j]["gold"][no & ge1]) if "gold" in labarr[j] else Counter()
        gprows.append({"judge": j, "stage": s, "n_no": int(no.sum()), "gold_ge1": int((no & ge1).sum()),
                       "share_ge1": sh, "share_ge1_lo": lo, "share_ge1_hi": hi,
                       "gold_all": int((no & allg).sum()), "share_all": sha, "share_all_lo": loa,
                       "share_all_hi": hia, "ge1_2b_yes": g2b["yes"], "ge1_2b_partial": g2b["partial"],
                       "ge1_2b_no": g2b["no"], "ge1_2b_invalid": g2b["invalid"]})
    write_csv("gold_present_no.csv", gprows)

    # ───────────────────────── blind audit ─────────────────────────
    key = {r["n"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data/drive/audit/sample_key.csv")))}
    items = []  # (stage, domain, topic_id, qrels, claude = adjudicated audit label, A, B = the two runs)
    for r in csv.DictReader(open(os.path.join(ROOT, "docs/audit_llm.csv"))):
        k = key[r["n"]]
        assert k["domain"] == r["domain"] and k["suff"] == r["suff_qrels"] and k["first_stage"] == "bm25_fuse3"
        items.append({"src": "audit1", "stage": "bm25_fuse3", "key": (k["domain"], k["topic_id"]),
                      "qrels": k["suff"], "claude": r["final"], "A": r["A"], "B": r["B"]})
    smap = {"bm25_query": "bm25_query", "bm25_hist": "bm25_hist", HYB: "hybrid", "hybrid": "hybrid"}
    for r in csv.DictReader(open(os.path.join(ROOT, "docs/audit_llm2.csv"))):
        items.append({"src": "audit2", "stage": smap[r["stage"]], "key": (r["domain"], r["topic_id"]),
                      "qrels": None, "claude": r["final"], "A": r["A"], "B": r["B"]})
    assert len(items) == 250
    for it in items:
        it["qrels"] = it["qrels"] or str(qarr[it["stage"]][keys.index(it["key"])])
        assert it["qrels"] == qarr[it["stage"]][keys.index(it["key"])], it
        assert it["claude"] in L3, it
        for j in judges:
            if it["stage"] in labarr[j]:
                it[j] = str(labarr[j][it["stage"]][keys.index(it["key"])])
    assert all(it["qrels"] == "none" for it in items if it["src"] == "audit2")
    ann = ["claude"] + list(judges)

    arows, aud_json = [], {}
    for s in ST:
        sub = [it for it in items if it["stage"] == s and it["qrels"] == "none"]
        rng = np.random.default_rng(SEED)
        BI = rng.integers(0, len(sub), size=(B, len(sub)))
        boot_p, pt_p = {}, {}
        for a in ann:
            if not all(a in it for it in sub):
                continue
            lab = np.array([it[a] for it in sub])

            def pf(lb):  # invalid is excluded from the fm denominator
                no, yes, nv = (lb == "no").sum(), (lb == "yes").sum(), (lb != "invalid").sum()
                return (no / (no + yes) if no + yes else NAN), (yes / nv if nv else NAN)
            P, fm = pf(lab)
            bs = np.array([pf(lab[i]) for i in BI], dtype=float)
            boot_p[a], pt_p[a] = bs, (P, fm)
            c = Counter(lab)
            row = {"stage": s, "display": DISP[s], "ndcg": point[("phi-4", s)]["ndcg"], "annotator": a,
                   "n_none_audited": len(sub), "yes": c["yes"], "partial": c["partial"], "no": c["no"],
                   "invalid": c["invalid"], "P": P, "P_lo": pct(bs[:, 0])[0], "P_hi": pct(bs[:, 0])[1],
                   "fm": fm, "fm_lo": pct(bs[:, 1])[0], "fm_hi": pct(bs[:, 1])[1],
                   "judge_P_all858": point[(a, s)]["P"] if (a, s) in point else NAN,
                   "judge_fm_all858": point[(a, s)]["fm"] if (a, s) in point else NAN}
            arows.append(row)
        aud_json[s] = {f"claude - {a}": {"dP": pt_p["claude"][0] - pt_p[a][0], "dfm": pt_p["claude"][1] - pt_p[a][1],
                                          "dP_ci": pct(boot_p["claude"][:, 0] - boot_p[a][:, 0]),
                                          "dfm_ci": pct(boot_p["claude"][:, 1] - boot_p[a][:, 1])}
                       for a in boot_p if a != "claude"}
    write_csv("audit_stage.csv", arows)
    # blind-audit trend across retrievers (independent samples, hence an independent bootstrap)
    aud_trend = {}
    for a, b in PAIRS:
        la = np.array([it["claude"] for it in items if it["stage"] == a and it["qrels"] == "none"])
        lb = np.array([it["claude"] for it in items if it["stage"] == b and it["qrels"] == "none"])
        rng = np.random.default_rng(SEED)
        P = lambda lb_: ((lb_ == "no").sum() / ((lb_ == "no").sum() + (lb_ == "yes").sum())
                         if ((lb_ == "no") | (lb_ == "yes")).any() else NAN)
        F = lambda lb_: (lb_ == "yes").mean()
        dP, dF = [], []
        for _ in range(B):
            xa, xb = la[rng.integers(0, len(la), len(la))], lb[rng.integers(0, len(lb), len(lb))]
            dP.append(P(xa) - P(xb))
            dF.append(F(xa) - F(xb))
        dP = np.array(dP, dtype=float)
        aud_trend[f"{a} - {b}"] = {"dP": P(la) - P(lb), "dP_ci": pct(dP), "p_ge0": float(np.nanmean(dP >= 0)),
                                   "dfm": F(la) - F(lb), "dfm_ci": pct(dF)}

    # ───────────────────────── agreement ─────────────────────────
    AK = ["agree3", "kappa3", "ac1_3", "agree_yes_rest", "kappa_yes_rest", "ac1_yes_rest",
          "agree_yes_no", "kappa_yes_no", "ac1_yes_no"]

    def agree_row(pair, subset, a, b):
        ok = [(x, y) for x, y in zip(a, b) if x != "invalid" and y != "invalid"]
        n_inv = len(a) - len(ok)
        a, b = [x for x, _ in ok], [y for _, y in ok]
        if not a:
            return None
        s = agree_stats(a, b)
        rng = np.random.default_rng(SEED)
        bt = defaultdict(list)
        for _ in range(B):
            i = rng.integers(0, len(a), len(a))
            r = agree_stats([a[t] for t in i], [b[t] for t in i])
            for k in AK:
                bt[k].append(r[k])
        row = {"pair": pair, "subset": subset, "n": s["n"], "n_dropped_invalid": n_inv,
               "n_strict": s["n_strict"]}
        for k in AK:
            row.update({k: s[k], k + "_lo": pct(bt[k])[0], k + "_hi": pct(bt[k])[1]})
        return row

    subsets = [("all_audited", lambda it: True), ("none_audited", lambda it: it["qrels"] == "none")]
    subsets += [(f"{s}_audited", (lambda s_: lambda it: it["stage"] == s_)(s)) for s in ST]
    subsets += [("bm25_fuse3_none_audited", lambda it: it["stage"] == "bm25_fuse3" and it["qrels"] == "none"),
                ("bm25_fuse3_all_audited", lambda it: it["stage"] == "bm25_fuse3" and it["qrels"] == "all")]
    agrows = []
    pairs = [("claude", j) for j in judges] + [(a, b) for i, a in enumerate(judges) for b in list(judges)[i + 1:]]
    for a, b in pairs:
        for name, f in subsets:
            sub = [it for it in items if f(it)]
            if not sub or not all(a in it and b in it for it in sub):
                continue
            r = agree_row(f"{a} vs {b}", name, [it[a] for it in sub], [it[b] for it in sub])
            if r:
                agrows.append(r)
    # the two audit runs (A, B) before adjudication
    r = agree_row("claude_A vs claude_B", "all_audited", [it["A"] for it in items], [it["B"] for it in items])
    agrows.append(r)
    # judge vs judge on all contexts of each job
    jl = list(judges)
    for i, a in enumerate(jl):
        for b in jl[i + 1:]:
            pooled_a, pooled_b = [], []
            for s in ST + ["gold"]:
                if s in labarr[a] and s in labarr[b]:
                    xa, xb = list(labarr[a][s]), list(labarr[b][s])
                    pooled_a += xa
                    pooled_b += xb
                    r = agree_row(f"{a} vs {b}", f"{s}_all858", xa, xb)
                    if r:
                        agrows.append(r)
            if pooled_a:
                r = agree_row(f"{a} vs {b}", "all_jobs_pooled", pooled_a, pooled_b)
                if r:
                    agrows.append(r)
    write_csv("agreement.csv", agrows)

    # ── json + printout
    res = {"judges": {j: sorted(v) for j, v in judges.items()}, "skipped_files": skipped,
           "point": {f"{j}|{s}": m for (j, s), m in point.items()},
           "gold_2b": gold_pt, "domains": dom_summary, "audit_claude_vs_judge": aud_json,
           "audit_claude_trend": aud_trend,
           "base_rate_only_precision": {j: {DISP[s]: brp[(j, s)] for s in ST if (j, s) in brp} for j in judges},
           "conv_bootstrap": conv_json,
           "counts": {j: {DISP[s]: {k: point[(j, s)][k] for k in ("n_yes", "n_fm", "n_no", "n_tp", "n_fn", "n_disagree",
                                                                 "n_partial", "fpr", "youden", "R", "P", "base_no",
                                                                 "share_no_or_partial", "ratio_partial_insuff",
                                                                 "P_partial_insuff", "fm_or_partial", "share_partial")}
                              for s in ST if (j, s) in point} for j in judges},
           "note": ("P/R/F1: qrels none predicts judge no, partial excluded; "
                    "fm = yes among all none; ratio = share_none/share_no. Paired bootstrap over turns stratified "
                    "by domain, 1000, seed 0; P_conv_* = conversation-cluster bootstrap. Audit P = no/(no+yes) "
                    "on 50 qrels-none contexts per stage (fusion: none half of the stratified audit_llm.csv); "
                    "bootstrap over items, seed 0; stage trend = independent bootstrap. Agreement: bootstrap "
                    "over items; yes_rest = yes vs partial/no; yes_no = partial from either side dropped. "
                    "fpr = n_fm / n_yes (false alarm on content-yes, partial excluded); youden = R - fpr; "
                    "n_disagree = none&yes + not-none&no; base_rate_only_precision = precision with BM25-turn "
                    "R and fpr at this stage's base rate no/(no+yes); ratio_partial_insuff = share_none / "
                    "share(no or partial); conv_bootstrap = |dialogue-cluster CI bound - turn CI bound|.")}
    json.dump(res, open(os.path.join(OUT, "label_v2.json"), "w"), indent=1, default=float)

    print("judges:", {j: sorted(v) for j, v in judges.items()}, "skipped:", skipped)
    for r in rows:
        print(f"{r['judge']:10s} {r['stage']:10s} nDCG {r['ndcg']:.4f} none {r['share_none']:.3f} no {r['share_no']:.3f} "
              f"ratio {r['ratio']:.2f} [{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}] fm {r['fm']:.3f} [{r['fm_lo']:.3f},{r['fm_hi']:.3f}] "
              f"P {r['P']:.3f} [{r['P_lo']:.3f},{r['P_hi']:.3f}] R {r['R']:.3f} F1 {r['F1']:.3f} "
              f"Pins {r['P_partial_insuff']:.3f} Psuf {r['P_partial_suff']:.3f} inv {r['n_invalid']}")
    for r in rows:
        print(f"  {r['judge']:10s} {r['display']:9s} fpr {r['n_fm']}/{r['n_yes']} = {r['fpr']:.3f} "
              f"[{r['fpr_lo']:.3f},{r['fpr_hi']:.3f}] J {r['youden']:.3f} [{r['youden_lo']:.3f},{r['youden_hi']:.3f}] "
              f"disagree {r['n_disagree']} no {r['n_no']} P@turn {r['P_at_turn_rates']:.3f} "
              f"no|partial {r['share_no_or_partial']:.3f} ratio_pi {r['ratio_partial_insuff']:.3f} "
              f"partial {r['share_partial']:.3f}")
    print("conv bootstrap:", json.dumps({k: {x: v[x] for x in ("max_abs_shift", "at", "max_abs_shift_by_judge",
                                                               "disjoint_turn_stays_disjoint_conv")}
                                          for k, v in conv_json.items()}, default=float))
    for r in drows:
        if r["metric"] in ("P", "fm", "youden"):
            print(f"  {r['judge']} {r['comparison']:24s} d{r['metric']} {r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
                  f"p_ge0 {r['p_ge0']:.3f} p_le0 {r['p_le0']:.3f}")
    print("domains:", json.dumps(dom_summary))
    for r in arows:
        print(f"audit {r['stage']:10s} {r['annotator']:10s} y/p/n {r['yes']}/{r['partial']}/{r['no']} P {r['P']:.3f} "
              f"[{r['P_lo']:.3f},{r['P_hi']:.3f}] fm {r['fm']:.2f} [{r['fm_lo']:.2f},{r['fm_hi']:.2f}]")
    print("audit trend:", json.dumps(aud_trend))
    for r in agrows:
        print(f"agree {r['pair']:22s} {r['subset']:26s} n {r['n']} raw {r['agree3']:.2f} k3 {r['kappa3']:.2f} "
              f"[{r['kappa3_lo']:.2f},{r['kappa3_hi']:.2f}] kYN {r['kappa_yes_no']:.2f} AC1 {r['ac1_3']:.2f}")
    for r in gprows:
        print(f"goldpres {r['judge']} {r['stage']:10s} no {r['n_no']} ge1 {r['gold_ge1']} ({r['share_ge1']:.2f} "
              f"[{r['share_ge1_lo']:.2f},{r['share_ge1_hi']:.2f}]) all {r['gold_all']} 2b y/p/n "
              f"{r['ge1_2b_yes']}/{r['ge1_2b_partial']}/{r['ge1_2b_no']}")


if __name__ == "__main__":
    main()
