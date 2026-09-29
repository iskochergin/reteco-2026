#!/usr/bin/env python3
"""Первая стадия поиска RETECO трек 2: BM25 (кит или Lucene, опц. RM3) или плотный.

Один прогон = один способ поиска × одно представление запроса × сплит.
Пишет runs/<имя>/<домен>/run.trec (формат сабмита) и runs/<имя>/config.json.

    # B1, воспроизведение официального: BM25 на реплике + истории
    $PY scripts/retrieve.py --method bm25 --query hist --out runs/2026-09-28_b1_bm25_hist

    # тот же BM25 через Lucene (pyserini) и с обратной связью RM3
    $PY scripts/retrieve.py --method lucene --query hist --out runs/...
    $PY scripts/retrieve.py --method lucene --rm3 --query hist --out runs/...

    # вариант запроса из rewrite.py
    $PY scripts/retrieve.py --method bm25 --queries runs/rewrites/q2d/dev.jsonl --out runs/...

Представления запроса:
    --query query   только текущая реплика                      (официальный 2a)
    --query hist    реплика + "Conversation History:" + история  (официальный 2a_hist)
    --queries F     готовые тексты из JSONL (domain, topic_id, text) — варианты rewrite.py

Методы:
    bm25     официальный BM25 кита (gensim LuceneBM25Model, k1=0.9, b=0.4);
             числа совпадают с опубликованными
    lucene   BM25 с тем же анализатором, но через индекс Lucene (pyserini); нужен
             для RM3. Индекс строится один раз в data/index/lucene/<домен>
    dense    плотный поиск (sentence-transformers); эмбеддинги корпуса кэшируются
             в data/index/<модель>/<домен>.npy (fp16) — считать их в Colab
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

# Промпты запроса/документа берутся из config_sentence_transformers.json самой
# модели (prompt_name="query"/"document"): так кодирует и Colab-ноутбук.
# У Diver там документ БЕЗ префикса; RECOR в своём коде добавлял
# "Represent this text:" — см. docs/gotchas.md.


def official_kit():
    spec = importlib.util.spec_from_file_location(
        "official_baseline",
        os.path.join(ROOT, "vendor", "RETECO", "starter_kit", "official_baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_queries(path):
    """Ключ — (домен, topic_id): в доменах BRIGHT topic_id повторяются между доменами."""
    with open(path, encoding="utf-8") as f:
        return {(r["domain"], r["topic_id"]): r["text"] for r in map(json.loads, filter(str.strip, f))}


def topics(domain, split, variant, given):
    """Как _recor_topics в ките: реплики без золота пропускаются, история
    приклеивается ровно тем же разделителем. given — тексты из --queries."""
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
                    sys.exit(f"нет текста запроса для {domain}/{qid} в --queries")
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
    """Индекс Lucene с векторами документов (нужны RM3). Строится один раз."""
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
    s.set_bm25(0.9, 0.4)                     # как у кита
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
        kw = {"dtype": torch.float16} if fp16 else {}   # transformers 5: dtype, не torch_dtype
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
    ap.add_argument("--queries", help="JSONL с полями domain, topic_id, text — вместо --query")
    ap.add_argument("--rm3", action="store_true", help="только для --method lucene")
    ap.add_argument("--rm3-params", nargs=3, type=float, default=[10, 10, 0.5],
                    metavar=("TERMS", "DOCS", "ORIG_W"), help="по умолчанию Anserini: 10 10 0.5")
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
    ap.add_argument("--query-prompt", default=None, help="по умолчанию из конфига модели")
    ap.add_argument("--doc-prompt", default=None, help="по умолчанию из конфига модели")
    args = ap.parse_args()

    given = load_queries(args.queries) if args.queries else None
    qname = os.path.basename(os.path.dirname(args.queries)) if args.queries else args.query
    rm3 = (int(args.rm3_params[0]), int(args.rm3_params[1]), args.rm3_params[2]) if args.rm3 else None
    if rm3 and args.method != "lucene":
        sys.exit("--rm3 только с --method lucene")
    tag = args.tag or f"{args.method}{'_rm3' if rm3 else ''}_{qname}"
    os.makedirs(args.out, exist_ok=True)

    kit = official_kit()
    dense = None
    if args.method == "dense":
        dense = Dense(args.model, args.max_doc_len, args.max_q_len, args.batch,
                      not args.no_fp16, args.query_prompt, args.doc_prompt)
        print(f"модель {args.model} на {dense.device}, промпт запроса {dense.q_prompt!r}, "
              f"документа {dense.d_prompt!r}")

    t0 = time.time()
    for dom in args.domains:
        out_path = os.path.join(args.out, dom, "run.trec")
        if os.path.isfile(out_path):
            print(f"[cached] {dom}"); continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        qids, qtexts = topics(dom, args.split, args.query, given)
        doc_ids, docs = kit.load_corpus(os.path.join(DATA, dom, "documents.jsonl"), "doc_id")
        print(f"\n=== {dom}: {len(docs)} док, {len(qids)} реплик ===", flush=True)
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
    print(f"\nготово за {cfg['seconds']}s → {args.out}")


if __name__ == "__main__":
    main()
