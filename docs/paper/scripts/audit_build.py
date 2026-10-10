"""Build the second blind-audit batch: 50 contexts each for BM25-turn, BM25-hist and the hybrid
with no gold document in the top 5 (qrels label none). Items are written as item_NNN.md in the
same format as the first audit batch (BM25-fusion); the key n -> (stage, domain, topic_id) is
written separately to key.csv.
    python audit_build.py <out_dir>"""
import csv, json, os, random, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src = open(f"{ROOT}/colab/build.py", encoding="utf-8").read()
worker = re.search(r"GEN_WORKER = r'''(.*?)'''", src, re.S).group(1)
G = {}
exec(worker.split("def main(")[0], G)          # load_turns, contexts, sufficiency, corpus_texts

DATA = f"{ROOT}/data/reteco/track2_recor"
DOMAINS = sorted(d for d in os.listdir(DATA) if os.path.isdir(f"{DATA}/{d}"))
STAGES = ["bm25_query", "bm25_hist", "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"]
N = 50
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
C = {"drive": f"{ROOT}/data/drive/reteco", "domains": DOMAINS, "k_ctx": 5}
turns = G["load_turns"](DATA, DOMAINS, "dev")
pick = []
for st in STAGES:
    ctx = G["contexts"](C, turns, "dev", f"2c-{st}")
    pool = [t for t in turns if G["sufficiency"](t["gold"], ctx[(t["domain"], t["topic_id"])])[1] == "none"]
    pick += [(st, t, ctx[(t["domain"], t["topic_id"])]) for t in random.Random(0).sample(pool, N)]
random.Random(1).shuffle(pick)                  # mix stages so the annotator cannot tell which retriever produced a context
need = {}
for _, t, c in pick:
    need.setdefault(t["domain"], set()).update(c)
text = G["corpus_texts"](DATA, need)
with open(f"{out}/key.csv", "w", newline="", encoding="utf-8") as f:
    k = csv.writer(f)
    k.writerow(["n", "stage", "domain", "topic_id", "n_gold", "gold_in_prior", "context_ids"])
    for n, (st, t, c) in enumerate(pick, 1):
        b = f"{out}/batch_{(n - 1) // 10:02d}"
        os.makedirs(b, exist_ok=True)
        hist = "\n".join(f"Q: {q}\nA: {a}" for q, a in t["history"]) or "(none)"
        docs = "\n\n".join(f"[{i}] {text[(t['domain'], d)]}" for i, d in enumerate(c, 1))
        open(f"{b}/item_{n:03d}.md", "w", encoding="utf-8").write(
            f"# Item {n} (domain: {t['domain']})\n\n## Conversation history\n{hist}\n\n## Latest question\n"
            f"{t['query']}\n\n## Reference answer\n{t['reference']}\n\n## Top-5 retrieved documents\n{docs}\n")
        k.writerow([n, st, t["domain"], t["topic_id"], len(t["gold"]), t["gold_in_prior"], " ".join(c)])
print(len(pick), "items →", out)
