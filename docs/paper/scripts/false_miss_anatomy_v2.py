#!/usr/bin/env python3
"""False-miss anatomy (Section 5, Figure 3, Appendix H) on the four retrievers (BM25-turn,
BM25-hist, BM25-fusion, hybrid; called stages in the code, with display names BM25-turn,
BM25-hist, Fusion, Hybrid).

Definitions and priority:
a false miss is suff_qrels == none (no qrels gold in the top-5) with judge label yes
(partial/no/null are not misses). The evidence document is classified relative to the
turn's gold, with priority near_dup > other_turn > same_source > independent; independent
is split into pool_neg (a labelled negative from the same dialogue's pool) and other (the
rest: gold of another dialogue, everything else). The context class is the best among its
evidence documents. A "yes" context without evidence (no_evidence) goes to other, so the
five class shares sum to 1 and n is all false misses; the number of such contexts is in
anatomy_summary.json.
Bootstrap over the 858 turns (1000 resamples, seed 0, percentile 95% CI).

Judges: phi-4 (primary; anatomy.csv, anatomy_ci.csv) and Qwen3-14B (anatomy_by_judge.csv); a
judge's file for a retriever is used only if it has all 858 lines.
Comparison with the blind audit (annotator "Claude" in anatomy_claude.csv): batch 1 (the 50
qrels-none contexts among the 100 audited BM25-fusion contexts) and batch 2 (50 qrels-none contexts
each for BM25-turn, BM25-hist and the hybrid). For contexts the audit labels yes, classes follow the
audit's evidence_docs; each judge is shown on the same contexts with its own evidence_docs.
Batch 1 is also checked against data/drive/audit/sample.csv (the audited texts) to confirm that the
documents were shown in context_ids order, so that the audit's evidence_docs indices point into
context_ids. That file contains benchmark text and is not released; without it the check is
skipped and anatomy_summary.json records "not checked (sample.csv not released)".
Appendix examples: 3 near-duplicates with a verbatim shared sentence (anatomy_examples.csv).
Base rate (anatomy_base.csv, base_rate in anatomy_summary.json): classes of ALL top-5 documents
in qrels-none contexts (gold_elsewhere -> other), the share of none contexts with at least one
tied document, the share of evidence by class, and qrels-none precision after removing
all dialogue-tied false misses (no / (no + untied false misses)).

Output: docs/paper/out/v2/anatomy{,_ci,_by_judge,_claude,_examples,_docs,_base}.csv,
anatomy_summary.json. Usage: python docs/paper/scripts/false_miss_anatomy_v2.py
The RETECO_SUFF / RETECO_OUT environment variables override the label and output folders (for code checks only).
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pytrec_eval

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
RUNS = os.path.join(ROOT, "data", "drive", "reteco", "runs")
SUFF = os.environ.get("RETECO_SUFF") or os.path.join(ROOT, "data", "drive", "reteco", "suff")
OUT = os.environ.get("RETECO_OUT") or os.path.join(ROOT, "docs", "paper", "out", "v2")
SE = ["drones", "politics", "law", "medicalsciences", "hardware"]
BR = ["economics", "psychology", "biology", "sustainable_living", "robotics", "earth_science"]
DOMAINS = SE + BR
HYB = "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
STAGES = [("BM25-turn", "bm25_query"), ("BM25-hist", "bm25_hist"), ("Fusion", "bm25_fuse3"), ("Hybrid", HYB)]
JUDGES = ["phi-4", "Qwen3-14B"]
CLASSES = ["near_dup", "other_turn", "same_source", "independent"]
DSUB = ["pool_neg", "gold_elsewhere", "other"]
OUT_CLASSES = ["near_dup", "other_turn", "same_source", "pool_neg", "other"]
ROWS = ["BM25-turn", "BM25-hist", "Fusion", "Hybrid",
        "Fusion: StackExchange", "Fusion: BRIGHT", "Hybrid: StackExchange", "Hybrid: BRIGHT"]
JAC, CONT, B, SEED, N_TURNS = 0.7, 0.8, 1000, 0, 858
SE_RE = re.compile(r"_(ex_\d+)_(doc|neg|google_neg)_(\d+)$")
# Appendix examples were picked by hand from the candidates (set CANDIDATES to list them): short
# documents whose shared sentence carries the answer; different stages and domains; the Figure 1
# example (medicalsciences, glucose oxidase, Fusion) is excluded; together with it all 4 stages are covered.
EXAMPLES = [("Hybrid", "hardware", "ex_18337_turn_4", "hardware_ex_18337_neg_9"),
            ("BM25-turn", "biology", "28_turn_2", "hot_water_bacteria/Hand_washing_0_13.txt"),
            ("BM25-hist", "medicalsciences", "ex_34329_turn_2", "medicalsciences_ex_34329_neg_1")]


def group(dom):
    return "StackExchange" if dom in SE else "BRIGHT"


def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def suff_path(judge, stage):
    return os.path.join(SUFF, f"{judge}_2c-{stage}_dev.jsonl")


def available(judge, stage):
    p = suff_path(judge, stage)
    return os.path.exists(p) and sum(1 for x in open(p, encoding="utf-8") if x.strip()) == N_TURNS


# ── gold: per turn, per dialogue, per domain (dev+train) ──
gold = defaultdict(set)
dial_gold = defaultdict(lambda: defaultdict(set))
any_gold = defaultdict(lambda: defaultdict(set))
question = {}
for d in DOMAINS:
    for split in ("dev", "train"):
        for ln in open(os.path.join(DATA, d, f"qrels_{split}.txt")):
            p = ln.split()
            if len(p) == 4 and int(p[3]) > 0:
                dial = p[0].rsplit("_turn_", 1)[0]
                any_gold[d][p[2]].add((split, dial))
                if split == "dev":
                    gold[(d, p[0])].add(p[2])
                    dial_gold[(d, dial)][p[0]].add(p[2])
    for c in json.load(open(os.path.join(DATA, d, "benchmark_dev.json"), encoding="utf-8")):
        for t in c["turns"]:
            tid = f"{c['id']}_turn_{t['turn_id']}"
            dial_gold[(d, c["id"])][tid] |= set(t.get("gold_doc_ids") or [])
            question[(d, tid)] = t["query"]


def dial_of(topic):
    return topic.rsplit("_turn_", 1)[0]


def page_key(dom, doc):
    name = doc.split("/", 1)[1][:-4] if doc.endswith(".txt") else doc.split("/", 1)[1]
    k = re.sub(r"[_\d]+$", "", name) if dom == "earth_science" else re.sub(r"(_\d+)+$", "", name)
    return k or name


def se_parts(doc):
    m = SE_RE.search(doc)
    return (m.group(1), m.group(2)) if m else (None, None)


# ── judge labels ──
labels = {}                                  # (judge, stage) -> {(dom, topic): row}
for j in JUDGES:
    for _, st in STAGES:
        if available(j, st):
            labels[(j, st)] = {(r["domain"], r["topic_id"]): r for r in load_jsonl(suff_path(j, st))}
assert all(("phi-4", st) in labels for _, st in STAGES), "phi-4 labels incomplete"
keys = [(r["domain"], r["topic_id"]) for r in load_jsonl(suff_path("phi-4", "bm25_query"))]   # this order defines the bootstrap draws (label_v2.py uses it too)
N = len(keys)
assert N == N_TURNS

# ── texts: none contexts and their gold (plus audit 1 contexts for the order check) ──
need = defaultdict(set)
for (j, st), rows in labels.items():
    for k, r in rows.items():
        if r["suff_qrels"] == "none" or st == "bm25_fuse3":
            need[k[0]] |= set(r["context_ids"]) | gold[k]
text = {}
for d in DOMAINS:
    for ln in open(os.path.join(DATA, d, "documents.jsonl"), encoding="utf-8"):
        o = json.loads(ln)
        if o["doc_id"] in need[d]:
            text[(d, o["doc_id"])] = o["content"]

_sh, _wg = {}, {}


def shingles(d, doc):
    if (d, doc) not in _sh:
        t = re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", text.get((d, doc), "").lower())).strip()
        _sh[(d, doc)] = {t[i:i + 5] for i in range(len(t) - 4)}
    return _sh[(d, doc)]


def wgrams(d, doc, n=8):
    if (d, doc) not in _wg:
        w = re.sub(r"[^0-9a-z]+", " ", text.get((d, doc), "").lower()).split()
        _wg[(d, doc)] = {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))} if w else set()
    return _wg[(d, doc)]


def sim(d, x, y):
    a, b = shingles(d, x), shingles(d, y)
    jac = len(a & b) / len(a | b) if a and b else 0.0
    wa, wb = wgrams(d, x), wgrams(d, y)
    i = len(wa & wb)
    return jac, (i / len(wa) if wa else 0.0), (i / len(wb) if wb else 0.0)


def best_gold(d, topic, doc):
    """The turn's gold document closest to doc (by containment, then by Jaccard)."""
    return max(gold[(d, topic)], key=lambda x: (max(sim(d, doc, x)[1:]), sim(d, doc, x)[0]), default=None)


def classify(d, topic, doc):
    """Flags and the mutually exclusive class of an evidence document."""
    g, dial = gold[(d, topic)], dial_of(topic)
    other = set().union(*[v for t, v in dial_gold[(d, dial)].items() if t != topic] or [set()]) - g
    jac, c_in, c_out = max((sim(d, doc, x) for x in g), key=lambda s: (max(s[1:]), s[0]), default=(0.0, 0.0, 0.0))
    cont = max(c_in, c_out)
    f = {"near_dup": jac >= JAC or cont >= CONT, "other_turn": doc in other,
         "ev_inside_gold": c_in >= CONT, "gold_inside_ev": c_out >= CONT}
    if d in SE:
        ex, typ = se_parts(doc)
        dial_docs = set().union(*dial_gold[(d, dial)].values())
        f["same_source"] = typ == "doc" and ex == dial and doc not in dial_docs and doc not in g
        f["pool_neg"] = ex == dial and typ in ("neg", "google_neg")
    else:
        pk = page_key(d, doc)
        src = [x for x in g if page_key(d, x) == pk]
        f["same_source"] = bool(src) and doc not in g
        f["pool_neg"] = doc.split("/")[0] in {x.split("/")[0] for x in g}
    f["gold_elsewhere"] = any(dl != dial for _, dl in any_gold[d].get(doc, ()))
    cls = next(c for c in CLASSES[:3] if f[c]) if any(f[c] for c in CLASSES[:3]) else "independent"
    sub = (next(s for s in DSUB[:2] if f[s]) if any(f[s] for s in DSUB[:2]) else "other") \
        if cls == "independent" else ""
    return cls, sub, f, jac, cont


RANK = {c: i for i, c in enumerate(CLASSES)}


def ctx_class(items):
    if not items:
        return "no_evidence", ""
    best = min(items, key=lambda x: (RANK[x[0]], DSUB.index(x[1]) if x[1] else 0))
    return best[0], best[1]


def out_class(c):
    """(class, subclass) from classify() -> one of the five reported classes (OUT_CLASSES, column class_v2)."""
    cls, sub = c
    return cls if cls in CLASSES[:3] else "pool_neg" if sub == "pool_neg" else "other"


def ev_index(ev, n_ctx):
    """evidence_docs -> valid 1-based indices (Qwen3-14B may emit junk); number discarded."""
    ok, bad = [], 0
    for i in ev or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            bad += 1
            continue
        if 1 <= i <= n_ctx:
            ok.append(i)
        else:
            bad += 1
    return ok, bad


def macro_ndcg(run_dir):
    vals, top5 = [], {}
    for d in DOMAINS:
        q = defaultdict(dict)
        for ln in open(os.path.join(DATA, d, "qrels_dev.txt")):
            p = ln.split()
            q[p[0]][p[2]] = int(p[3])
        run, order = defaultdict(dict), defaultdict(list)
        for ln in open(os.path.join(RUNS, run_dir, d, "run.trec")):
            p = ln.split()
            if p[2] not in run[p[0]]:
                run[p[0]][p[2]] = float(p[4])
                order[p[0]].append((int(p[3]), p[2]))
        for t, o in order.items():
            top5[(d, t)] = [x for _, x in sorted(o)[:5]]
        res = pytrec_eval.RelevanceEvaluator(q, {"ndcg_cut.10"}).evaluate(run)
        vals.append(np.mean([res.get(t, {}).get("ndcg_cut_10", 0.0) for t in q]))
    return float(np.mean(vals)), top5


BOOT = np.random.default_rng(SEED).integers(0, N, size=(B, N))


def boot_share(sel, pred, boot=BOOT):
    """Share of pred among sel + bootstrap 95% CI."""
    sel, pred = np.asarray(sel), np.asarray(pred) & np.asarray(sel)
    pt = pred.sum() / max(sel.sum(), 1)
    bs = [pred[i].sum() / s for i in boot if (s := sel[i].sum()) > 0]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan"))
    return {"share": round(float(pt), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "k": int(pred.sum()), "n": int(sel.sum())}


def anatomy(judge, stage, rows, doc_rows=None):
    """False misses of a stage under judge labels -> {key: (class, subclass)}, count of discarded indices."""
    ctx, bad = {}, 0
    for k in keys:
        r = rows[k]
        if not (r["suff_qrels"] == "none" and r.get("label") == "yes"):
            continue
        items = []
        ev, b = ev_index(r.get("evidence_docs"), len(r["context_ids"]))
        bad += b
        for i in ev:
            doc = r["context_ids"][i - 1]
            cls, sub, f, jac, cont = classify(k[0], k[1], doc)
            items.append((cls, sub))
            if doc_rows is not None:
                doc_rows.append({"judge": judge, "stage": stage, "domain": k[0], "group": group(k[0]),
                                 "topic_id": k[1], "evidence_rank": i, "doc_id": doc, "class": cls, "d_sub": sub,
                                 "class_v2": out_class((cls, sub)),
                                 **{fk: int(fv) for fk, fv in f.items()}, "jaccard": round(jac, 3),
                                 "containment": round(cont, 3), "gold": " ".join(sorted(gold[k]))})
        ctx[k] = ctx_class(items)
    return ctx, bad


_CLS = {}


def doc_class(d, topic, doc):
    """Reported class (class_v2) of any context document (classification does not depend on the judge, so it is cached)."""
    if (d, topic, doc) not in _CLS:
        _CLS[(d, topic, doc)] = out_class(classify(d, topic, doc)[:2])
    return _CLS[(d, topic, doc)]


def base_rate(judge, stage, rows, ctx):
    """Class base rate: all top-5 documents in qrels-none contexts (not only evidence).
    Evidence is the judge's evidence_docs in its false misses (label yes), as in anatomy().
    Returns the summary and the rows for anatomy_base.csv."""
    allc, evc, groups = Counter(), Counter(), {"StackExchange": Counter(), "BRIGHT": Counter()}
    none_sel, tied_ctx, tied_ctx_g = [], [], {"StackExchange": [0, 0], "BRIGHT": [0, 0]}
    for k in keys:
        r = rows[k]
        isnone = r["suff_qrels"] == "none"
        none_sel.append(isnone)
        if not isnone:
            tied_ctx.append(False)
            continue
        cl = [doc_class(k[0], k[1], doc) for doc in r["context_ids"]]
        allc.update(cl)
        groups[group(k[0])].update(cl)
        t = any(c != "other" for c in cl)
        tied_ctx.append(t)
        tied_ctx_g[group(k[0])][0] += t
        tied_ctx_g[group(k[0])][1] += 1
        if r.get("label") == "yes":
            idx, _ = ev_index(r.get("evidence_docs"), len(r["context_ids"]))
            evc.update(cl[i - 1] for i in idx)
    A, E = sum(allc.values()), sum(evc.values())
    tA, tE = A - allc["other"], E - evc["other"]
    # qrels-none precision after removing all dialogue-tied false misses:
    # numerator is none & judge no, denominator is the same plus untied false misses
    no = [rows[k]["suff_qrels"] == "none" and rows[k].get("label") == "no" for k in keys]
    untied = [k in ctx and out_class(ctx[k]) == "other" for k in keys]
    sel_raw = [a or (k in ctx) for a, k in zip(no, keys)]
    sel_disc = [a or b for a, b in zip(no, untied)]
    rows_out = []
    for c in OUT_CLASSES + ["tied", "all"]:
        na = tA if c == "tied" else A if c == "all" else allc[c]
        ne = tE if c == "tied" else E if c == "all" else evc[c]
        rows_out.append({"judge": judge, "stage": stage, "class": c, "n_docs": na,
                         "share_of_docs": round(na / A, 4) if A else None, "n_evidence": ne,
                         "share_of_evidence": round(ne / E, 4) if E else None,
                         "evidence_rate": round(ne / na, 4) if na else None})
    summ = {"definition": "all top-5 docs of qrels-none contexts, v2 classes (gold_elsewhere counted as other); "
                          "evidence = judge evidence_docs in its false misses (label yes); "
                          "evidence_rate = evidence docs of a class / all docs of that class; "
                          "tied = any class except other",
            "n_none_contexts": int(sum(none_sel)), "n_docs": A, "n_evidence_docs": E,
            "docs_by_class": {c: allc[c] for c in OUT_CLASSES},
            "docs_by_class_group": {g: {c: cc[c] for c in OUT_CLASSES} for g, cc in groups.items()},
            "evidence_by_class": {c: evc[c] for c in OUT_CLASSES},
            "tied_share_of_docs": round(tA / A, 4) if A else None,
            "tied_share_of_evidence": round(tE / E, 4) if E else None,
            "evidence_rate": {c: (round(evc[c] / allc[c], 4) if allc[c] else None) for c in OUT_CLASSES},
            "evidence_rate_tied": round(tE / tA, 4) if tA else None,
            "evidence_rate_untied": round(evc["other"] / allc["other"], 4) if allc["other"] else None,
            "none_ctx_with_tied_doc": boot_share(none_sel, tied_ctx),
            "none_ctx_with_tied_doc_by_group": {g: {"k": v[0], "n": v[1], "share": round(v[0] / v[1], 4) if v[1] else None}
                                                for g, v in tied_ctx_g.items()},
            "precision_raw": boot_share(sel_raw, no),
            "precision_untied_fm_only": boot_share(sel_disc, no),
            "n_no": int(sum(no)), "n_false_miss": len(ctx), "n_false_miss_untied": int(sum(untied))}
    return summ, rows_out


def table(ctx, row_name, gdoms):
    sel = [k in ctx and k[0] in gdoms for k in keys]
    out = []
    for c in OUT_CLASSES:
        s = boot_share(sel, [k in ctx and k[0] in gdoms and out_class(ctx[k]) == c for k in keys])
        out.append({"row": row_name, "class": c, **s})
    return out


os.makedirs(OUT, exist_ok=True)
summary = {"definitions": {
    "false_miss": "suff_qrels == none (no qrels gold in top-5 context) and judge label == yes",
    "classes": "classes and priority near_dup > other_turn > same_source > independent; independent split "
               "into pool_neg (labelled negative from the dialogue's own pool: SE _neg_/_google_neg_ of the "
               "same ex_N, BRIGHT same question folder) and other (gold of another dialogue + rest); "
               "context class = best among its evidence docs; yes without evidence docs -> other",
    "near_dup": f"word 8-gram containment >= {CONT} or char 5-gram Jaccard >= {JAC} vs a gold doc of the turn",
    "bootstrap": f"over {N} dev turns, {B} resamples, seed {SEED}, 95% percentile CI; audits: over audited items",
    "judges": "phi-4 primary; Qwen3-14B included per stage only when its file has 858 lines"},
    "judges_used": {j: [n for n, st in STAGES if (j, st) in labels] for j in JUDGES}, "stages": {}}
tidy, by_judge, doc_rows, ctx_all, base_rows = [], [], [], {}, []
for name, st in STAGES:
    nd, top5 = macro_ndcg(f"{st}_dev")
    for j in JUDGES:
        if (j, st) not in labels:
            continue
        rows = labels[(j, st)]
        ctx, bad = anatomy(j, name, rows, doc_rows)
        ctx_all[(j, st)] = ctx
        none_mask = [rows[k]["suff_qrels"] == "none" for k in keys]
        lab = Counter(rows[k].get("label") for k in keys if rows[k]["suff_qrels"] == "none")
        groups = [(name, DOMAINS)] + ([(f"{name}: StackExchange", SE), (f"{name}: BRIGHT", BR)]
                                      if name in ("Fusion", "Hybrid") else [])
        tab = [x for g, gd in groups for x in table(ctx, g, gd)]
        by_judge += [{"judge": j, **x} for x in tab]
        tied = boot_share([k in ctx for k in keys], [k in ctx and out_class(ctx[k]) != "other" for k in keys])
        s = {"false_miss": len(ctx), "none": int(sum(none_mask)),
             "none_labels": {str(a): b for a, b in lab.items()},
             "false_miss_rate_among_none": boot_share(none_mask, [k in ctx for k in keys]),
             "false_miss_rate_among_none_yes_or_no": boot_share(
                 [m and rows[k].get("label") in ("yes", "no") for m, k in zip(none_mask, keys)],
                 [k in ctx for k in keys]),
             "no_evidence": sum(v[0] == "no_evidence" for v in ctx.values()),
             "invalid_evidence_indices": bad, "tied_to_dialogue": tied,
             "context_eq_top5": f"{sum(rows[k]['context_ids'] == top5.get(k) for k in keys)}/{N}"}
        dr = [x for x in doc_rows if x["judge"] == j and x["stage"] == name]
        nd_docs = [x for x in dr if x["class"] == "near_dup"]
        s["doc_level"] = {"n_docs": len(dr), "classes": dict(Counter(x["class_v2"] for x in dr)),
                          "near_dup": {"n": len(nd_docs), "gold_inside_ev": sum(x["gold_inside_ev"] for x in nd_docs),
                                       "ev_inside_gold": sum(x["ev_inside_gold"] for x in nd_docs),
                                       "jaccard_hits": sum(x["jaccard"] >= JAC for x in nd_docs),
                                       "doc_type": dict(Counter(se_parts(x["doc_id"])[1] or "bright"
                                                                for x in nd_docs))}}
        s["base_rate"], br = base_rate(j, name, rows, ctx)
        base_rows += br
        summary["stages"].setdefault(name, {"run": f"{st}_dev", "ndcg10_macro": round(nd, 4)})[j] = s
        if j == "phi-4":
            tidy += tab

# ── comparison with the blind audit ──
akey = {r["n"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data", "drive", "audit", "sample_key.csv")))}
# sample.csv holds the audited benchmark texts and is not released: without it, skip the order check
SAMPLE = os.path.join(ROOT, "data", "drive", "audit", "sample.csv")
smp = {r["n"]: r for r in csv.DictReader(open(SAMPLE))} if os.path.exists(SAMPLE) else None
if smp is None:
    print("data/drive/audit/sample.csv not found (not released): skipping the audit-1 document order check")
audit = defaultdict(list)                     # stage -> [(key, final, evidence)]
order_ok = 0
for a in csv.DictReader(open(os.path.join(ROOT, "docs", "audit_llm.csv"))):
    kk = akey[a["n"]]
    k = (kk["domain"], kk["topic_id"])
    r = labels[("phi-4", "bm25_fuse3")][k]
    if smp is not None:
        blk, pos, ok = re.sub(r"\s+", " ", smp[a["n"]]["top5"]), 0, True
        for jx, doc in enumerate(r["context_ids"], 1):
            q = f"[{jx}] " + re.sub(r"\s+", " ", text[(k[0], doc)]).strip()[:60]
            pos = blk.find(q, pos)
            ok, pos = ok and pos >= 0, max(pos, 0)
        order_ok += ok
    if r["suff_qrels"] == "none":
        audit["bm25_fuse3"].append((k, a["final"], a["evidence_docs"].split()))
for a in csv.DictReader(open(os.path.join(ROOT, "docs", "audit_llm2.csv"))):
    k = (a["domain"], a["topic_id"])
    assert labels[("phi-4", a["stage"])][k]["suff_qrels"] == "none", (a["stage"], k)
    audit[a["stage"]].append((k, a["final"], a["evidence_docs"].split()))
claude_rows, summary["claude"] = [], {"audit1_top5_order_matches_context_ids": f"{order_ok}/100" if smp is not None
                                      else "not checked (sample.csv not released)"}
for name, st in STAGES:
    items = audit[st]
    M = len(items)
    boot = np.random.default_rng(SEED).integers(0, M, size=(B, M))
    ann = {"Claude": [(f == "yes", ev) for _, f, ev in items]}
    for j in JUDGES:
        if (j, st) in labels:
            ann[f"{j} (same items)"] = [(labels[(j, st)][k].get("label") == "yes", labels[(j, st)][k].get("evidence_docs"))
                                        for k, _, _ in items]
    summary["claude"][name] = {"items": M}
    for an, lst in ann.items():
        cls = []
        for (k, _, _), (yes, ev) in zip(items, lst):
            if not yes:
                cls.append(None)
                continue
            ctx_ids = labels[("phi-4", st)][k]["context_ids"]
            idx, _ = ev_index(ev, len(ctx_ids))
            cls.append(out_class(ctx_class([classify(k[0], k[1], ctx_ids[i - 1])[:2] for i in idx])))
        sel = [c is not None for c in cls]
        for c in OUT_CLASSES:
            claude_rows.append({"row": name, "annotator": an, "class": c,
                                **boot_share(sel, [x == c for x in cls], boot)})
        summary["claude"][name][an] = {"yes": sum(sel), "tied_to_dialogue": boot_share(
            sel, [x not in (None, "other") for x in cls], boot)}

# ── appendix examples ──


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", t.lower())).strip()


def shared_sentence(d, ev, g):
    """Longest sentence of the evidence document (8-30 words) that occurs verbatim
    (after normalisation) in the gold document."""
    gn = " " + norm(text[(d, g)]) + " "
    best = ""
    for s in sentences(text[(d, ev)]):
        w = norm(s).split()
        if 8 <= len(w) <= 30 and f" {' '.join(w)} " in gn and len(w) > len(norm(best).split()):
            best = s
    return best


cands = []
seen = set()
for x in doc_rows:
    if x["judge"] != "phi-4" or x["class"] != "near_dup" or (x["stage"], x["domain"], x["topic_id"]) in seen:
        continue
    d, ev = x["domain"], x["doc_id"]
    g = best_gold(d, x["topic_id"], ev)
    s = shared_sentence(d, ev, g)
    if s:
        seen.add((x["stage"], x["domain"], x["topic_id"]))
        cands.append({**x, "gold_doc": g, "sentence": s, "ev_words": len(text[(d, ev)].split()),
                      "gold_words": len(text[(d, g)].split())})
ex_rows = []
for st, d, t, ev in EXAMPLES:
    c = next(c for c in cands if (c["stage"], c["domain"], c["topic_id"], c["doc_id"]) == (st, d, t, ev))
    ex_rows.append({"stage": st, "domain": d, "topic_id": t, "question": question[(d, t)], "evidence_doc_id": ev,
                    "gold_doc_id": c["gold_doc"], "class": c["class"], "containment": c["containment"],
                    "evidence_words": c["ev_words"], "gold_words": c["gold_words"],
                    "shared_sentence": re.sub(r"\s+", " ", c["sentence"])})
summary["example_candidates"] = len(cands)


def write(fn, rows, fields=None):
    with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


tidy.sort(key=lambda x: (ROWS.index(x["row"]), OUT_CLASSES.index(x["class"])))
by_judge.sort(key=lambda x: (JUDGES.index(x["judge"]), ROWS.index(x["row"]), OUT_CLASSES.index(x["class"])))
write("anatomy.csv", tidy, ["row", "class", "share", "n"])
write("anatomy_ci.csv", tidy, ["row", "class", "share", "lo", "hi", "k", "n"])
write("anatomy_by_judge.csv", by_judge, ["judge", "row", "class", "share", "lo", "hi", "k", "n"])
write("anatomy_claude.csv", claude_rows, ["row", "annotator", "class", "share", "lo", "hi", "k", "n"])
write("anatomy_docs.csv", doc_rows)
write("anatomy_base.csv", base_rows)
if ex_rows:
    write("anatomy_examples.csv", ex_rows)
json.dump(summary, open(os.path.join(OUT, "anatomy_summary.json"), "w"), indent=1, ensure_ascii=False)

if __name__ == "__main__" and os.environ.get("CANDIDATES"):
    for c in sorted(cands, key=lambda c: c["ev_words"] + c["gold_words"]):
        print(f"{c['stage']}|{c['domain']}|{c['topic_id']}|{c['doc_id']}|gold {c['gold_doc']}|"
              f"{c['ev_words']}/{c['gold_words']}w|{c['containment']}\n  Q: {question[(c['domain'], c['topic_id'])]}"
              f"\n  S: {c['sentence'][:300]}")
else:
    for r in ROWS:
        print(f"{r:<22}", " ".join(f"{x['class']} {x['share']:.2f}" for x in tidy if x["row"] == r),
              f"n={next(x['n'] for x in tidy if x['row'] == r)}")
    print(json.dumps({n: {"ndcg": s["ndcg10_macro"], **{j: {"fm": v["false_miss"], "none": v["none"],
                                                              "rate": v["false_miss_rate_among_none"]["share"],
                                                              "tied": v["tied_to_dialogue"]["share"],
                                                              "noev": v["no_evidence"], "ctx": v["context_eq_top5"],
                                                              "base": {x: v["base_rate"][x] for x in (
                                                                  "tied_share_of_docs", "tied_share_of_evidence",
                                                                  "evidence_rate_tied", "evidence_rate_untied",
                                                                  "evidence_rate")},
                                                              "none_ctx_tied": v["base_rate"]["none_ctx_with_tied_doc"],
                                                              "P_raw": v["base_rate"]["precision_raw"],
                                                              "P_untied": v["base_rate"]["precision_untied_fm_only"]}
                                                          for j, v in s.items() if isinstance(v, dict) and "none" in v}}
                      for n, s in summary["stages"].items()}, indent=1))
