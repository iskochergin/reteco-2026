"""Collect blind-audit batch 2 (Claude Opus 5.5; qrels-none contexts of BM25-turn, BM25-hist and the
hybrid, built by audit_build.py) into docs/audit_llm2.csv and compare it with phi-4 on the same contexts.
<annotation_runs.json> is a list of batches (or {"result": [...]}), each of the form
{"a": {"items": [...]}, "bb": {"items": [...]}, "adj": {"items": [...]}}: run A, run B and the
adjudication run for items where A and B disagree (optional). Every item has n, label,
answered_in_history, evidence_docs and rationale. <key.csv> is the key written by audit_build.py.
    python audit2_collect.py <annotation_runs.json> <key.csv>"""
import csv, json, os, sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
res = json.load(open(sys.argv[1]))
res = res.get("result", res) if isinstance(res, dict) else res
key = {int(r["n"]): r for r in csv.DictReader(open(sys.argv[2]))}
rows = {}
for b in res:
    A = {x["n"]: x for x in b["a"]["items"]}
    B = {x["n"]: x for x in b["bb"]["items"]}
    J = {x["n"]: x for x in (b.get("adj") or {}).get("items", [])}
    for n in A:
        fin = J.get(n, A[n]) if A[n]["label"] != B[n]["label"] else A[n]
        rows[n] = {"n": n, "stage": key[n]["stage"], "domain": key[n]["domain"], "topic_id": key[n]["topic_id"],
                   "A": A[n]["label"], "B": B[n]["label"], "adjudicated": n in J, "final": fin["label"],
                   "answered_in_history": fin["answered_in_history"],
                   "evidence_docs": " ".join(map(str, fin["evidence_docs"])), "rationale": fin["rationale"]}
assert len(rows) == len(key), (len(rows), len(key))
out = f"{ROOT}/docs/audit_llm2.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
    w.writeheader(); w.writerows(rows[n] for n in sorted(rows))
agree = sum(r["A"] == r["B"] for r in rows.values()) / len(rows)
print(f"{len(rows)} → {out}; A=B {agree:.2f}")
phi = {}
for st in {r["stage"] for r in rows.values()}:
    p = f"{ROOT}/data/drive/reteco/suff/phi-4_2c-{st}_dev.jsonl"
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l); phi[(st, d["domain"], d["topic_id"])] = d["label"]
for st in ["bm25_query", "bm25_hist", "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"]:
    rs = [r for r in rows.values() if r["stage"] == st]
    c = Counter(r["final"] for r in rs)
    prec = c["no"] / (c["no"] + c["yes"]) if c["no"] + c["yes"] else float("nan")
    ph = [phi.get((st, r["domain"], r["topic_id"])) for r in rs]
    cp = Counter(ph)
    pp = cp["no"] / (cp["no"] + cp["yes"]) if cp["no"] + cp["yes"] else float("nan")
    same = sum(a == r["final"] for a, r in zip(ph, rs) if a)
    print(f"{st[:14]:<15} audit yes/partial/no {c['yes']}/{c['partial']}/{c['no']}  P(none→no) {prec:.2f}"
          f" | phi-4 {cp['yes']}/{cp['partial']}/{cp['no']} P {pp:.2f} (missing {cp[None]}) | same {same}/{len(rs)}"
          f" | hist {sum(r['answered_in_history'] for r in rs)}")
