#!/usr/bin/env python3
"""First-stage retrieval for RETECO Track 2: BM25 (starter kit or Lucene, optional RM3) or dense.

One run = one retrieval method x one query representation x one split.
Writes runs/<name>/<domain>/run.trec (submission format) and runs/<name>/config.json.

    # reproduce the official baseline: BM25 on the turn + history
    python scripts/retrieve.py --method bm25 --query hist --out runs/bm25_hist

    # the same BM25 via Lucene (pyserini), and with RM3 feedback
    python scripts/retrieve.py --method lucene --query hist --out runs/...
    python scripts/retrieve.py --method lucene --rm3 --query hist --out runs/...

    # a query variant produced by rewrite.py
    python scripts/retrieve.py --method bm25 --queries runs/rewrites/q2d/dev.jsonl --out runs/...

Query representations:
    --query query   current turn only                              (official 2a)
    --query hist    turn + "Conversation History:" + history         (official 2a_hist)
    --queries F     precomputed texts from JSONL (domain, topic_id, text), e.g. rewrite.py output

Methods:
    bm25     the starter kit's official BM25 (gensim LuceneBM25Model, k1=0.9, b=0.4);
             reproduces the published numbers
    lucene   BM25 with the same analyzer but over a Lucene index (pyserini); required
             for RM3. The index is built once in data/index/lucene/<domain>
    dense    dense retrieval (sentence-transformers); corpus embeddings are cached
             in data/index/<model>/<domain>.npy (fp16), best computed on a GPU
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JDK for pyserini: Homebrew JDK 21 on macOS, used only if present; elsewhere set
# JAVA_HOME/JVM_PATH yourself. Set explicitly because `/usr/libexec/java_home -v 21`
# may return an older JRE with exit code 0, and pyserini then fails later with an unclear error.
_JDK = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JDK):
    os.environ.setdefault("JAVA_HOME", _JDK)
    os.environ.setdefault("JVM_PATH", os.path.join(_JDK, "lib", "server", "libjvm.dylib"))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics",
           "sustainable_living"]

# Query/document prompts come from the model's own config_sentence_transformers.json
# (prompt_name="query"/"document"); the encoding notebooks from colab/build.py do the same.
# For Diver-Retriever-4B the document prompt there is empty, whereas the original RECOR code
# prepends "Represent this text:" to documents.


def official_kit():
    spec = importlib.util.spec_from_file_location(
        "official_baseline",
        os.path.join(ROOT, "vendor", "RETECO", "starter_kit", "official_baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_queries(path):
    """Keyed by (domain, topic_id): in the BRIGHT domains topic_ids repeat across domains."""
    with open(path, encoding="utf-8") as f:
        return {(r["domain"], r["topic_id"]): r["text"] for r in map(json.loads, filter(str.strip, f))}


def topics(domain, split, variant, given):
    """Mirrors _recor_topics in the starter kit: turns without gold documents are skipped
    and the history is appended with exactly the same separator. given: texts from --queries."""
    convs = json.load(open(os.path.join(DATA, domain, f"benchmark_{split}.json"),
                           encoding="utf-8"))
    ids, texts = [], []
    for c in convs:
        for t in c["turns"]:
            if not (t.get("gold_doc_ids") or t.get("supporting_doc_ids")):
                continue
            qid = f"{c['id']}_turn_{t['turn_id']}"
            if given is not None:
                if (domain, qid) not in given:
                    sys.exit(f"no query text for {domain}/{qid} in --queries")
                text = given[(domain, qid)]
            else:
                parts = [t["query"]]
                h = t.get("conversation_history", "")
                if variant == "hist" and h and h != "No previous conversation.":
                    parts.append(f"Conversation History:\n{h}")
                text = "\n\n".join(parts)
            ids.append(qid)
            texts.append(text)
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


# -------------------------------------------------------------- Lucene ------
def lucene_index(dom, doc_ids, docs):
    """Lucene index with stored document vectors (needed for RM3). Built once."""
    idx = os.path.join(ROOT, "data", "index", "lucene", dom)
    if os.path.isdir(idx) and any(f.startswith("segments") for f in os.listdir(idx)):
        return idx
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "docs.jsonl"), "w", encoding="utf-8") as f:
            for i, t in zip(doc_ids, docs):
                f.write(json.dumps({"id": i, "contents": t}, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, "-m", "pyserini.index.lucene",
                        "--collection", "JsonCollection", "--input", tmp, "--index", idx,
                        "--generator", "DefaultLuceneDocumentGenerator", "--threads", "4",
                        "--storeDocvectors"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return idx


def run_lucene(idx, qids, qtexts, topk, rm3):
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(idx)
    s.set_bm25(0.9, 0.4)                     # same as the starter kit
    if rm3:
        s.set_rm3(fb_terms=rm3[0], fb_docs=rm3[1], original_query_weight=rm3[2])
    hits = s.batch_search(qtexts, qids, k=topk, threads=8)
    return {q: [(h.docid, h.score) for h in hits.get(q, [])] for q in qids}


# --------------------------------------------------------------- dense ------
class Dense:
    def __init__(self, model_name, max_doc_len, max_q_len, batch, fp16, q_prompt, d_prompt):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        kw = {"dtype": torch.float16} if fp16 else {}   # transformers 5: dtype, not torch_dtype
        self.model = SentenceTransformer(model_name, device=self.device,
                                         model_kwargs=kw, trust_remote_code=True)
        self.max_doc_len, self.max_q_len, self.batch = max_doc_len, max_q_len, batch
        prompts = self.model.prompts or {}
        self.q_prompt = prompts.get("query", "") if q_prompt is None else q_prompt
        self.d_prompt = prompts.get("document", "") if d_prompt is None else d_prompt

    def encode(self, texts, prompt, max_len, desc):
        self.model.max_seq_length = max_len
        return self.model.encode(texts, prompt=prompt, batch_size=self.batch,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=True)

    def corpus(self, cache, docs):
        import numpy as np
        if os.path.isfile(cache):
            return np.load(cache)
        emb = self.encode(docs, self.d_prompt, self.max_doc_len, "docs").astype("float16")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, emb)
        return emb

    def search(self, emb, doc_ids, qids, qtexts, topk):
        q = self.encode(qtexts, self.q_prompt, self.max_q_len, "queries").astype("float32")
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
    ap.add_argument("--method", choices=["bm25", "lucene", "dense"], required=True)
    ap.add_argument("--query", choices=["query", "hist"], default="hist")
    ap.add_argument("--queries", help="JSONL with fields domain, topic_id, text; overrides --query")
    ap.add_argument("--rm3", action="store_true", help="only with --method lucene")
    ap.add_argument("--rm3-params", nargs=3, type=float, default=[10, 10, 0.5],
                    metavar=("TERMS", "DOCS", "ORIG_W"), help="Anserini defaults: 10 10 0.5")
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--out", required=True, help="runs/<date>_<name>")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--tag", default=None)
    # dense
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max-doc-len", type=int, default=512)
    ap.add_argument("--max-q-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--query-prompt", default=None, help="default: from the model config")
    ap.add_argument("--doc-prompt", default=None, help="default: from the model config")
    args = ap.parse_args()

    given = load_queries(args.queries) if args.queries else None
    qname = os.path.basename(os.path.dirname(args.queries)) if args.queries else args.query
    rm3 = (int(args.rm3_params[0]), int(args.rm3_params[1]), args.rm3_params[2]) if args.rm3 else None
    if rm3 and args.method != "lucene":
        sys.exit("--rm3 requires --method lucene")
    tag = args.tag or f"{args.method}{'_rm3' if rm3 else ''}_{qname}"
    os.makedirs(args.out, exist_ok=True)

    kit = official_kit()
    dense = None
    if args.method == "dense":
        dense = Dense(args.model, args.max_doc_len, args.max_q_len, args.batch,
                      not args.no_fp16, args.query_prompt, args.doc_prompt)
        print(f"model {args.model} on {dense.device}, query prompt {dense.q_prompt!r}, "
              f"document prompt {dense.d_prompt!r}")

    t0 = time.time()
    for dom in args.domains:
        out_path = os.path.join(args.out, dom, "run.trec")
        if os.path.isfile(out_path):
            print(f"[cached] {dom}"); continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        qids, qtexts = topics(dom, args.split, args.query, given)
        doc_ids, docs = kit.load_corpus(os.path.join(DATA, dom, "documents.jsonl"), "doc_id")
        print(f"\n=== {dom}: {len(docs)} docs, {len(qids)} turns ===", flush=True)
        t = time.time()
        if args.method == "bm25":
            ranked = run_bm25(kit, doc_ids, docs, qids, qtexts, args.topk)
        elif args.method == "lucene":
            ranked = run_lucene(lucene_index(dom, doc_ids, docs), qids, qtexts, args.topk, rm3)
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
    print(f"\ndone in {cfg['seconds']}s -> {args.out}")


if __name__ == "__main__":
    main()
