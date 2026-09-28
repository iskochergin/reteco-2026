#!/usr/bin/env python3
"""Первая стадия поиска RETECO трек 2: BM25 (официальный, из кита) или плотный.

Один прогон = один способ поиска × одно представление запроса × сплит.
Пишет runs/<имя>/<домен>/run.trec (формат сабмита) и runs/<имя>/config.json.

    # B1, воспроизведение официального: BM25 на реплике + истории
    $PY scripts/retrieve.py --method bm25 --query hist --out runs/2026-09-28_b1_bm25_hist

    # B3: плотный zero-shot на переписанном запросе + истории
    $PY scripts/retrieve.py --method dense --model Qwen/Qwen3-Embedding-0.6B \
        --query rewrite_hist --rewrites runs/rewrites/llama70b/dev.jsonl --out runs/...

Представления запроса (--query):
    query         только текущая реплика                      (официальный 2a)
    hist          реплика + "Conversation History:\\n" + история (официальный 2a_hist)
    rewrite       LLM-переписанная самостоятельная реплика   (нужен --rewrites)
    rewrite_hist  rewrite + история

Эмбеддинги корпуса кэшируются в data/index/<модель>/<домен>.npy (fp16) и
переиспользуются между сплитами и представлениями запроса.
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JDK для pyserini. Задано явно: /usr/libexec/java_home -v 21 на этой машине
# возвращает JRE 1.8 с кодом 0, а pyserini падает позже и невнятно.
_JDK = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JDK):
    os.environ.setdefault("JAVA_HOME", _JDK)
    os.environ.setdefault("JVM_PATH", os.path.join(_JDK, "lib", "server", "libjvm.dylib"))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics",
           "sustainable_living"]

# Промпты запроса/документа по моделям. Diver — как в retrievers.py RECOR;
# Qwen3-Embedding — как в карточке модели (документ без префикса).
PREFIXES = {
    "Qwen/Qwen3-Embedding": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        ""),
    "AQ-MedAI/Diver-Retriever": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "Represent this text:"),
}


def official_kit():
    spec = importlib.util.spec_from_file_location(
        "official_baseline",
        os.path.join(ROOT, "vendor", "RETECO", "starter_kit", "official_baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rewrites(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                out[r["topic_id"]] = r["rewrite"]
    return out


def topics(domain, split, variant, rewrites):
    """Как _recor_topics в ките: реплики без золота пропускаются, история
    приклеивается ровно тем же разделителем."""
    convs = json.load(open(os.path.join(DATA, domain, f"benchmark_{split}.json"),
                           encoding="utf-8"))
    ids, texts = [], []
    for c in convs:
        for t in c["turns"]:
            if not (t.get("gold_doc_ids") or t.get("supporting_doc_ids")):
                continue
            qid = f"{c['id']}_turn_{t['turn_id']}"
            base = t["query"]
            if variant.startswith("rewrite"):
                if qid not in rewrites:
                    sys.exit(f"нет переписывания для {qid} в --rewrites")
                base = rewrites[qid]
            parts = [base]
            if variant.endswith("hist"):
                h = t.get("conversation_history", "")
                if h and h != "No previous conversation.":
                    parts.append(f"Conversation History:\n{h}")
            ids.append(qid)
            texts.append("\n\n".join(parts))
    return ids, texts


def write_trec(path, ranked, tag):
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in ranked.items():
            for rank, (d, s) in enumerate(docs, 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{s:.6f}\t{tag}\n")


# ---------------------------------------------------------------- BM25 ------
def run_bm25(kit, doc_ids, docs, qids, qtexts, topk):
    bm25 = kit.OfficialBM25(docs, doc_ids)
    scores = bm25.search(qtexts, qids, desc="  bm25")
    return {q: sorted(v.items(), key=lambda x: -x[1])[:topk] for q, v in scores.items()}


# --------------------------------------------------------------- dense ------
class Dense:
    def __init__(self, model_name, max_doc_len, max_q_len, batch, fp16, q_prefix, d_prefix):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        kw = {"dtype": torch.float16} if fp16 else {}   # transformers 5: dtype, не torch_dtype
        self.model = SentenceTransformer(model_name, device=self.device,
                                         model_kwargs=kw, trust_remote_code=True)
        self.max_doc_len, self.max_q_len, self.batch = max_doc_len, max_q_len, batch
        self.q_prefix, self.d_prefix = q_prefix, d_prefix

    def encode(self, texts, prefix, max_len, desc):
        self.model.max_seq_length = max_len
        if prefix:
            # Qwen3-Embedding: "...\nQuery:{text}" без пробела, как в карточке модели
            sep = "" if prefix.endswith((":", "\n")) else " "
            texts = [prefix + sep + t for t in texts]
        return self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)

    def corpus(self, cache, docs):
        import numpy as np
        if os.path.isfile(cache):
            return np.load(cache)
        emb = self.encode(docs, self.d_prefix, self.max_doc_len, "docs").astype("float16")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, emb)
        return emb

    def search(self, emb, doc_ids, qids, qtexts, topk):
        import numpy as np
        q = self.encode(qtexts, self.q_prefix, self.max_q_len, "queries").astype("float32")
        D = self.torch.from_numpy(emb.astype("float32")).to(self.device)
        Q = self.torch.from_numpy(q).to(self.device)
        out = {}
        for i in range(0, len(qids), 64):
            sims = Q[i:i + 64] @ D.T
            vals, idx = sims.topk(min(topk, D.shape[0]), dim=1)
            vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
            for j, qid in enumerate(qids[i:i + 64]):
                out[qid] = [(doc_ids[k], float(v)) for k, v in zip(idx[j], vals[j])]
        return out


# ---------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["bm25", "dense"], required=True)
    ap.add_argument("--query", choices=["query", "hist", "rewrite", "rewrite_hist"], required=True)
    ap.add_argument("--rewrites", help="JSONL с полями topic_id, rewrite")
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--out", required=True, help="runs/<дата>_<имя>")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--tag", default=None)
    # dense
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max-doc-len", type=int, default=512)
    ap.add_argument("--max-q-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--doc-prefix", default=None)
    args = ap.parse_args()

    if args.query.startswith("rewrite") and not args.rewrites:
        sys.exit("--query rewrite* требует --rewrites")
    rewrites = load_rewrites(args.rewrites) if args.rewrites else {}
    tag = args.tag or f"{args.method}_{args.query}"
    os.makedirs(args.out, exist_ok=True)

    kit = official_kit()
    dense = None
    if args.method == "dense":
        qp, dp = next((v for k, v in PREFIXES.items() if args.model.startswith(k)), ("", ""))
        qp = args.query_prefix if args.query_prefix is not None else qp
        dp = args.doc_prefix if args.doc_prefix is not None else dp
        dense = Dense(args.model, args.max_doc_len, args.max_q_len, args.batch,
                      not args.no_fp16, qp, dp)
        print(f"модель {args.model} на {dense.device}, префикс запроса {qp!r}, документа {dp!r}")

    t0 = time.time()
    for dom in args.domains:
        out_path = os.path.join(args.out, dom, "run.trec")
        if os.path.isfile(out_path):
            print(f"[cached] {dom}"); continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        qids, qtexts = topics(dom, args.split, args.query, rewrites)
        doc_ids, docs = kit.load_corpus(os.path.join(DATA, dom, "documents.jsonl"), "doc_id")
        print(f"\n=== {dom}: {len(docs)} док, {len(qids)} реплик ===", flush=True)
        t = time.time()
        if args.method == "bm25":
            ranked = run_bm25(kit, doc_ids, docs, qids, qtexts, args.topk)
        else:
            slug = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
            cache = os.path.join(ROOT, "data", "index", slug, f"{dom}.npy")
            emb = dense.corpus(cache, docs)
            ranked = dense.search(emb, doc_ids, qids, qtexts, args.topk)
        write_trec(out_path, ranked, tag)
        print(f"  {dom}: {time.time()-t:.0f}s", flush=True)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    cfg = {"script": "scripts/retrieve.py", "args": vars(args), "tag": tag,
           "date": dt.datetime.now().isoformat(timespec="seconds"), "git": commit,
           "seconds": round(time.time() - t0)}
    if dense is not None:
        import torch, sentence_transformers, transformers
        cfg["versions"] = {"torch": torch.__version__,
                           "sentence_transformers": sentence_transformers.__version__,
                           "transformers": transformers.__version__}
        cfg["device"] = dense.device
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\nготово за {cfg['seconds']}s → {args.out}")


if __name__ == "__main__":
    main()
