"""Answers come from the context, not from memory (generation check of the content label):
Table 2 and the paragraph "A reader answers from false-miss contexts" in Section 4.2 of the paper.

On false-miss contexts (qrels: no gold document; phi-4: "yes") a reader should answer correctly
much more often than on the same turn without documents (nodoc). Control: true misses
(qrels: no gold; phi-4: "no"), where the difference from nodoc should be small.

A correct answer: the reader answered (status ANSWERABLE), the answer judge did not mark a refusal,
and correctness >= 4 (scale 1-5). Answer judges: phi-4 and Qwen3-8B; readers: OCC-RAG-1.7B and
Qwen3-4B-Instruct-2507; stages: the three BM25 retrievers (answer judges were not run on the hybrid).
The paper reports Qwen3-4B-Instruct-2507 (Table 2: phi-4 answer judge; the Qwen3-8B judge is within .04).
OCC-RAG-1.7B is a reader from an earlier version of the analysis; its rows are not reported in the paper.

    python3 docs/paper/scripts/leak_check.py      # -> docs/paper/out/v2/leak_check.{csv,json}
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
D = os.path.join(ROOT, "data", "drive", "reteco")
OUT = os.path.join(HERE, "..", "out", "v2")
READERS = ["OCC-RAG-1.7B", "Qwen3-4B-Instruct-2507"]
JUDGES = ["phi-4", "Qwen3-8B"]
STAGES = {"bm25_query": "BM25-turn", "bm25_hist": "BM25-hist", "bm25_fuse3": "BM25-fusion"}
B, SEED = 1000, 0
# Without documents OCC leaves its output format: the status is empty on 857 of 858 turns and 391
# answers hit max_tokens. Its no-document control is therefore undefined.
NODOC_INVALID = {"OCC-RAG-1.7B"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return {(r["domain"], r["topic_id"]): r for r in map(json.loads, filter(str.strip, f))}


def correct(gen, jud, grounded=False):
    """1: correct answer; 0: not (refusal or error); None: the judge output could not be parsed.
    grounded: also supported by the shown documents (faithfulness >= 4, no unsupported claims)."""
    if gen is None or jud is None or not jud.get("ok", True):
        return None
    if gen.get("status") != "ANSWERABLE" or jud.get("response_type") == "refused":
        return 0
    ok = (jud.get("correctness") or 0) >= 4
    if grounded:
        ok = ok and (jud.get("faithfulness") or 0) >= 4 and not jud.get("unsupported_claims")
    return int(ok)


def boot(x, y):
    """Share correct with and without context, their difference and a 95% CI over turns (paired bootstrap)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), (B, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return x.mean(), y.mean(), x.mean() - y.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    content = {s: load(os.path.join(D, "suff", f"phi-4_2c-{s}_dev.jsonl")) for s in STAGES}
    rows = []
    for reader in READERS:
        gen_nodoc = load(os.path.join(D, "gen", f"{reader}_nodoc_dev.jsonl"))
        for judge in JUDGES:
            jud_nodoc = load(os.path.join(D, "judge", judge, f"{reader}_nodoc_dev.jsonl"))
            for s, name in STAGES.items():
                gen = load(os.path.join(D, "gen", f"{reader}_2c-{s}_dev.jsonl"))
                jud = load(os.path.join(D, "judge", judge, f"{reader}_2c-{s}_dev.jsonl"))
                for cls, want in [("false miss", "yes"), ("true miss", "no")]:
                    nodoc_ok = reader not in NODOC_INVALID
                    x, y, g, ans_nodoc = [], [], [], []
                    for k, lab in content[s].items():
                        if lab.get("suff_qrels") != "none" or lab.get("label") != want:
                            continue
                        a = correct(gen.get(k), jud.get(k))
                        b = correct(gen_nodoc.get(k), jud_nodoc.get(k)) if nodoc_ok else None
                        if a is None or (nodoc_ok and b is None):
                            continue
                        x.append(a)
                        g.append(correct(gen.get(k), jud.get(k), grounded=True) or 0)
                        if nodoc_ok:
                            y.append(b)
                            ans_nodoc.append(int(gen_nodoc[k].get("status") == "ANSWERABLE"))
                    if len(x) < 10:
                        continue
                    r4 = lambda v: None if v is None else round(float(v), 4)
                    if nodoc_ok:
                        pc, pn, d, lo, hi = boot(x, y)
                        only_ctx = float(np.mean([a == 1 and b == 0 for a, b in zip(x, y)]))
                        an = float(np.mean(ans_nodoc))
                    else:   # control undefined: only the share correct with context
                        pc, pn, d, lo, hi, only_ctx, an = float(np.mean(x)), None, None, None, None, None, None
                    rows.append({"reader": reader, "judge": judge, "stage": name, "class": cls, "n": len(x),
                                 "correct_ctx": r4(pc), "correct_nodoc": r4(pn), "diff": r4(d), "lo": r4(lo),
                                 "hi": r4(hi), "only_with_ctx": r4(only_ctx),
                                 "grounded_ctx": r4(np.mean(g)), "answered_nodoc": r4(an)})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "leak_check.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(OUT, "leak_check.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{'reader':<24}{'judge':<10}{'stage':<13}{'class':<12}{'n':>5}{'ctx':>7}{'grnd':>7}{'nodoc':>7}{'ans0':>7}{'diff':>7}  95% CI")
    f = lambda v, fmt: "     --" if v is None else format(v, fmt)
    for r in rows:
        ci = "" if r["lo"] is None else f"  [{r['lo']:+.2f},{r['hi']:+.2f}]"
        print(f"{r['reader']:<24}{r['judge']:<10}{r['stage']:<13}{r['class']:<12}{r['n']:>5}{r['correct_ctx']:>7.2f}"
              f"{r['grounded_ctx']:>7.2f}{f(r['correct_nodoc'], '>7.2f')}{f(r['answered_nodoc'], '>7.2f')}"
              f"{f(r['diff'], '>+7.2f')}{ci}")


if __name__ == "__main__":
    main()
