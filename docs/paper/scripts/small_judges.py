"""Small open judges as a stand-in for human annotators: agreement and the decline in precision.

Four models of 3-8B parameters from different families (colab/suff_small.ipynb) label the same
858x5 contexts and the gold contexts with the same prompt as phi-4. In the paper: the rows of the four
small judges and of their panel in Table 1, the paragraph "Other labellers" and the negative control in
Section 4.2, and the grey lines in Figure 2(b) (paper_figs.py with RETECO_FIG_SMALL=1). Computed here:
  * per judge and stage: precision of the qrels label "no gold" as a detector of "no" (partial
    excluded), the false-miss share (none & yes among none), 95% CI from a bootstrap over turns
    (seed 0); the change in precision from BM25-turn to the hybrid with a paired CI;
  * the panel of small judges: majority label (tie -> partial), the same precision;
  * agreement: Fleiss' kappa of the four small judges on three labels, pairwise Cohen's kappa of all
    judges, and each judge against the blind audit (250 contexts);
  * negative control (contexts of other dialogues): share of "yes" for each judge.
Precision and false misses are defined as in label_v2.metrics.

    python3 docs/paper/scripts/small_judges.py   # -> docs/paper/out/v2/small_judges.{csv,json}
"""
import csv, json, os, sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pytrec_eval  # noqa: F401  label_v2 needs it only for nDCG, not called here
except ImportError:
    import types
    sys.modules["pytrec_eval"] = types.ModuleType("pytrec_eval")
import label_v2 as L

SMALL = ["Qwen3-4B-Instruct-2507", "Phi-4-mini-instruct", "SmolLM3-3B", "granite-3.3-8b-instruct"]
BIG = ["phi-4", "Qwen3-14B"]
CTRL = f"ctrl-{L.HYB}"
OUT = L.OUT


def panel(labels):
    """Majority of the valid labels; a tie or no majority -> partial."""
    c = Counter(x for x in labels if x in L.L3)
    if not c:
        return "invalid"
    (top, n), *rest = c.most_common()
    return top if not rest or rest[0][1] < n else "partial"


def fleiss(rows, cats):
    """Fleiss' kappa: rows are the labels of one item (the same number of raters for every item)."""
    m = len(rows[0])
    cnt = np.array([[r.count(c) for c in cats] for r in rows], float)
    p_i = ((cnt * (cnt - 1)).sum(1)) / (m * (m - 1))
    p_j = cnt.sum(0) / cnt.sum()
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe))


def prec_fm(q, lab):
    """Precision of none as a detector of no (partial excluded) and the false-miss share among none."""
    v = lab != "invalid"
    q, lab = q[v], lab[v]
    none = q == "none"
    tp, fp = (none & (lab == "no")).sum(), (none & (lab == "yes")).sum()
    return (tp / (tp + fp) if tp + fp else np.nan), (fp / none.sum() if none.sum() else np.nan)


def main():
    judges, skipped = L.load_judges()
    keys = sorted(judges["phi-4"]["gold"])
    have = [j for j in SMALL if j in judges and all(s in judges[j] for s in L.ST + ["gold"])]
    if skipped:
        print("skipped (not 858 rows):", skipped)
    print("small judges with complete labels:", have or "none")
    names = BIG + have
    lab = {j: {s: np.array([L.lab_of(judges[j][s][k]) for k in keys]) for s in L.ST + ["gold"]} for j in names}
    q = {s: np.array([judges["phi-4"][s][k]["suff_qrels"] for k in keys]) for s in L.ST}
    if len(have) >= 2:
        names.append("panel")
        lab["panel"] = {s: np.array([panel([lab[j][s][i] for j in have]) for i in range(len(keys))])
                        for s in L.ST + ["gold"]}

    rng = np.random.default_rng(L.SEED)
    BI = rng.integers(0, len(keys), size=(L.B, len(keys)))
    rows = []
    for j in names:
        boots = {}
        for s in L.ST:
            p, fm = prec_fm(q[s], lab[j][s])
            bp = np.array([prec_fm(q[s][b], lab[j][s][b])[0] for b in BI])
            boots[s] = bp
            n_inv = int((lab[j][s] == "invalid").sum())
            c = Counter(lab[j][s])
            rows.append({"judge": j, "stage": L.DISP[s], "P": p, "P_lo": np.nanpercentile(bp, 2.5),
                         "P_hi": np.nanpercentile(bp, 97.5), "fm": fm, "yes": c["yes"], "partial": c["partial"],
                         "no": c["no"], "invalid": n_inv})
        d = boots["hybrid"] - boots["bm25_query"]
        p0 = prec_fm(q["bm25_query"], lab[j]["bm25_query"])[0]
        p1 = prec_fm(q["hybrid"], lab[j]["hybrid"])[0]
        rows.append({"judge": j, "stage": "Hybrid - Turn", "P": p1 - p0, "P_lo": np.nanpercentile(d, 2.5),
                     "P_hi": np.nanpercentile(d, 97.5), "fm": np.nan, "yes": "", "partial": "", "no": "",
                     "invalid": ""})

    # agreement: all contexts (gold + 4 stages) where both labels of a pair are valid
    allst = ["gold"] + L.ST
    flat = {j: np.concatenate([lab[j][s] for s in allst]) for j in names}
    agree = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = (flat[a] != "invalid") & (flat[b] != "invalid")
            st = L.agree_stats(flat[a][v], flat[b][v])
            agree.append({"pair": f"{a} vs {b}", "n": st["n"], "agree": st["agree3"], "kappa": st["kappa3"],
                          "ac1": st["ac1_3"], "kappa_yes_no": st["kappa_yes_no"]})
    fl = None
    if len(have) >= 2:
        v = np.all([flat[j] != "invalid" for j in have], axis=0)
        items = [[flat[j][i] for j in have] for i in np.where(v)[0]]
        fl = {"n": len(items), "fleiss3": fleiss(items, L.L3),
              "unanimous": float(np.mean([len(set(r)) == 1 for r in items]))}

    # blind audit: 250 contexts (keys as in label_v2)
    aud = []
    key_rows = {r["n"]: r for r in csv.DictReader(open(os.path.join(L.ROOT, "data/drive/audit/sample_key.csv")))}
    for r in csv.DictReader(open(os.path.join(L.ROOT, "docs/audit_llm.csv"))):
        k = key_rows[r["n"]]
        aud.append(("bm25_fuse3", (k["domain"], k["topic_id"]), r["final"]))
    smap = {"bm25_query": "bm25_query", "bm25_hist": "bm25_hist", L.HYB: "hybrid", "hybrid": "hybrid"}
    for r in csv.DictReader(open(os.path.join(L.ROOT, "docs/audit_llm2.csv"))):
        aud.append((smap[r["stage"]], (r["domain"], r["topic_id"]), r["final"]))
    assert len(aud) == 250
    idx = {k: i for i, k in enumerate(keys)}
    audit = []
    for j in names:
        a = [x for _, _, x in aud]
        b = [str(lab[j][s][idx[k]]) for s, k, _ in aud]
        v = [x != "invalid" for x in b]
        st = L.agree_stats([x for x, ok in zip(a, v) if ok], [x for x, ok in zip(b, v) if ok])
        audit.append({"pair": f"audit vs {j}", "n": st["n"], "agree": st["agree3"], "kappa": st["kappa3"],
                      "ac1": st["ac1_3"], "kappa_yes_no": st["kappa_yes_no"]})

    # negative control: share of yes / partial on contexts of other dialogues
    ctrl, ctrl_lab = [], {}
    for j in BIG + have:
        path = os.path.join(L.SUFF, f"{j}_{CTRL}_dev.jsonl")
        if not os.path.exists(path):
            continue
        rs = [json.loads(x) for x in open(path) if x.strip()]
        ctrl_lab[j] = {(r["domain"], r["topic_id"]): L.lab_of(r) for r in rs}
        c = Counter(L.lab_of(r) for r in rs)
        ctrl.append({"judge": j, "n": len(rs), "yes": c["yes"], "partial": c["partial"], "no": c["no"],
                     "invalid": c["invalid"], "share_yes": c["yes"] / len(rs)})
    if len(have) >= 2 and all(j in ctrl_lab for j in have):
        ks = sorted(set.intersection(*(set(ctrl_lab[j]) for j in have)))
        c = Counter(panel([ctrl_lab[j][k] for j in have]) for k in ks)
        ctrl.append({"judge": "panel", "n": len(ks), "yes": c["yes"], "partial": c["partial"], "no": c["no"],
                     "invalid": c["invalid"], "share_yes": c["yes"] / len(ks)})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "small_judges.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: L.R4(float(x)) if isinstance(x, (float, np.floating)) else x for k, x in r.items()}
                    for r in rows)
    res = {"judges": names, "small": have, "precision": rows, "agreement": agree, "fleiss_small": fl,
           "audit": audit, "control": ctrl}
    json.dump(res, open(os.path.join(OUT, "small_judges.json"), "w"), indent=1,
              default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\n{'judge':<26}{'stage':<15}{'P':>6}  95% CI        {'fm':>6}{'inv':>5}")
    for r in rows:
        fm = "" if r["fm"] != r["fm"] else f"{r['fm']:.2f}"
        print(f"{r['judge']:<26}{r['stage']:<15}{r['P']:>6.2f}  [{r['P_lo']:.2f}, {r['P_hi']:.2f}]  {fm:>6}{r['invalid']!s:>5}")
    print(f"\n{'pair':<52}{'n':>6}{'agree':>7}{'kappa':>7}{'AC1':>7}")
    for r in agree + audit:
        print(f"{r['pair']:<52}{r['n']:>6}{r['agree']:>7.2f}{r['kappa']:>7.2f}{r['ac1']:>7.2f}")
    if fl:
        print(f"\nFleiss kappa (4 small, 3 labels): {fl['fleiss3']:.3f} on {fl['n']} contexts; unanimous {fl['unanimous']:.2f}")
    for r in ctrl:
        print(f"control {r['judge']:<26} yes {r['yes']}/{r['n']} ({r['share_yes']:.3f}) partial {r['partial']} invalid {r['invalid']}")


if __name__ == "__main__":
    main()
