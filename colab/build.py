"""Builds colab/*.ipynb from a single source: the shared cells (Drive, data,
topics, ndcg10, write_run) are identical across all notebooks.

    python3 colab/build.py        # overwrites colab/*.ipynb

Edit the notebooks HERE, not in the .ipynb files; otherwise the next build overwrites the edit.
"""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))

def nb(cells, gpu="T4"):
    colab = {"gpuType": gpu, "provenance": []} if gpu != "CPU" else {"provenance": []}
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {**({"accelerator": "GPU"} if gpu != "CPU" else {}), "colab": colab,
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "cells": [{"cell_type": t, "metadata": {}, "source": src.strip("\n").splitlines(True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, src in cells]}

COMMON_SETUP = r'''
# ── 2. Drive, packages, GPU ──────────────────────────────────────────────────
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"   # the HF Xet transport sometimes hangs; plain HTTP is more reliable
from google.colab import drive
drive.mount("/content/drive")
!pip -q install -U sentence-transformers pytrec-eval-terrier
import torch, transformers
from packaging.version import Version
assert torch.cuda.is_available(), "No GPU: Runtime → Change runtime type → T4 GPU"
DTYPE_KEY = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB,",
      "transformers", transformers.__version__)
'''

COMMON_DATA = r'''
# ── 3. RETECO data, track 2 (≈270 MB, about a minute) ───────────────────────
from huggingface_hub import snapshot_download
snapshot_download("DataScience-UIBK/RETECO-SemEval2027", repo_type="dataset",
                  local_dir="/content/reteco",
                  allow_patterns=["track2_recor/*/documents.jsonl",
                                  "track2_recor/*/benchmark_*.json",
                                  "track2_recor/*/qrels_*.txt"])
'''

COMMON_CODE = r'''
# ── 4. Shared code. Mirrors scripts/retrieve.py and score.py in the repository ──
import os, re, json, time, glob
import numpy as np, pytrec_eval
from tqdm.auto import tqdm

DATA = "/content/reteco/track2_recor"
DOMAINS = ["drones", "politics", "law", "medicalsciences", "hardware", "economics",
           "psychology", "biology", "sustainable_living", "robotics", "earth_science"]  # smallest to largest
# official BM25 "turn + history", dev, nDCG@10, for comparison
BM25_DEV = {"biology": .6065, "drones": .2954, "earth_science": .6264, "economics": .4739,
            "hardware": .3378, "law": .3437, "medicalsciences": .1730, "politics": .3219,
            "psychology": .5726, "robotics": .5123, "sustainable_living": .5531}
BM25_MACRO = {"dev": 0.4379, "train": 0.4539}
RUNS, QUERIES = f"{DRIVE}/runs", f"{DRIVE}/queries"

# First run: the BM25 runs and LLM query rewrites sit next to the notebooks as parts
# of the archive reteco_bundle.tar.xz.part*; they are unpacked into DRIVE once.
_parts = sorted(glob.glob(f"{os.path.dirname(DRIVE)}/reteco_bundle.tar.xz.part*"))
if _parts and not os.path.exists(f"{DRIVE}/.bundle_ok"):
    import tarfile
    with open("/content/bundle.tar.xz", "wb") as _out:
        for _p in _parts:
            _out.write(open(_p, "rb").read())
    with tarfile.open("/content/bundle.tar.xz", "r:xz") as _tf:
        try:
            _tf.extractall(os.path.dirname(DRIVE), filter="data")
        except TypeError:
            _tf.extractall(os.path.dirname(DRIVE))
    open(f"{DRIVE}/.bundle_ok", "w").write("ok")
    print(f"project data unpacked from {len(_parts)} parts → {DRIVE}")
    try:                 # flush the unpacked files to Drive now, before the next notebooks start
        from google.colab import drive as _gd
        _gd.flush_and_unmount(); _gd.mount("/content/drive")
    except Exception as _e:
        print("flush Drive:", _e)
assert os.path.isdir(f"{RUNS}/bm25_hist_dev"), (
    f"no project data in {DRIVE}: the notebooks and reteco_bundle.tar.xz.part00…03 must be "
    "in the parent folder of DRIVE (default: MyDrive/RETECO 2026) of the account mounted by drive.mount")
os.makedirs(RUNS, exist_ok=True); os.makedirs(QUERIES, exist_ok=True)

def load_corpus(dom):
    ids, texts = [], []
    for ln in open(f"{DATA}/{dom}/documents.jsonl", encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln); ids.append(d["doc_id"]); texts.append(d["content"])
    return ids, texts

def topics(dom, split, qset):
    """Turns without gold documents are skipped; history is appended as in the organisers' baseline.
    qset: "query", "hist" or a folder name in Drive/reteco/queries (variants from rewrite.py)."""
    given = None
    if qset not in ("query", "hist"):
        with open(f"{QUERIES}/{qset}/{split}.jsonl", encoding="utf-8") as f:
            # key (domain, topic_id): in the BRIGHT domains topic_ids repeat across domains
            given = {(r["domain"], r["topic_id"]): r["text"] for r in map(json.loads, filter(str.strip, f))}
    ids, texts = [], []
    for c in json.load(open(f"{DATA}/{dom}/benchmark_{split}.json", encoding="utf-8")):
        for t in c["turns"]:
            if not (t.get("gold_doc_ids") or t.get("supporting_doc_ids")):
                continue
            qid = f"{c['id']}_turn_{t['turn_id']}"
            if given is not None:
                text = given[(dom, qid)]
            else:
                text, h = t["query"], t.get("conversation_history", "")
                if qset == "hist" and h and h != "No previous conversation.":
                    text += f"\n\nConversation History:\n{h}"
            ids.append(qid); texts.append(text)
    return ids, texts

def ndcg10(run, dom, split):
    """run: {qid: {doc: score}}. A turn missing from the run scores 0, as on the leaderboard."""
    qrels = {}
    for ln in open(f"{DATA}/{dom}/qrels_{split}.txt"):
        p = ln.split()
        if len(p) == 4:
            qrels.setdefault(p[0], {})[p[2]] = int(p[3])
    res = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    return sum(v["ndcg_cut_10"] for v in res.values()) / len(qrels)

def read_run(path):
    run = {}
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) == 6:
            run.setdefault(p[0], {})[p[2]] = float(p[4])
    return run

def write_run(path, run, tag):
    """Writes to a temporary file first, then renames it, so an interrupted write never
    leaves a partial file. Scores are written strictly decreasing and with spare digits:
    pytrec_eval breaks score ties by docno in reverse order, which in five domains would
    systematically rank _neg_ above _doc_. Tied scores keep the dict order (that is,
    the first-stage order)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for q, docs in run.items():
            last = None
            for r, (d, sc) in enumerate(sorted(docs.items(), key=lambda x: -x[1]), 1):
                sc = float(sc)
                if last is not None and sc >= last:
                    sc = last - 1e-9 * max(1.0, abs(last))
                last = sc
                f.write(f"{q}\tQ0\t{d}\t{r}\t{sc:.12g}\t{tag}\n")
    os.replace(path + ".tmp", path)

def strata(rs):
    """Slices for the generation tables. All rows ("total"); for 2c, by the qrels label (all/part/none)
    plus "clean" misses (none, and the gold never appeared in earlier turns, so the answer cannot be
    recovered from the history); for 2b and nodoc, the first turn (no history) and the rest."""
    yield "total", rs
    if rs and rs[0]["mode"] == "2c":
        for x in ("all", "part", "none"):
            yield x, [r for r in rs if r["suff"] == x]
        yield "clean", [r for r in rs if r["suff"] == "none" and r.get("gold_in_prior") == 0]
    else:
        yield "T1", [r for r in rs if r.get("turn_id") == 1]
        yield "T2+", [r for r in rs if (r.get("turn_id") or 0) > 1]

# ── log: everything the cells print, timestamped → Drive/reteco/logs ────────
import sys
class _Tee:
    def __init__(self, out, path):
        self.out, self.f, self.bol = out, open(path, "a", encoding="utf-8"), True
    def write(self, x):
        self.out.write(x)
        try:
            for part in x.splitlines(True):
                if self.bol:
                    self.f.write(time.strftime("%m-%d %H:%M:%S "))
                self.f.write(part)
                self.bol = part.endswith("\n")
            self.f.flush()
        except Exception:
            pass
        return len(x)
    def flush(self):
        self.out.flush()
    def __getattr__(self, k):
        return getattr(self.out, k)
os.makedirs(f"{DRIVE}/logs", exist_ok=True)
if not isinstance(getattr(sys.stdout, "_reteco_log", None), str):
    LOG_PATH = f"{DRIVE}/logs/{globals().get('NB', 'notebook')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = _Tee(sys.stdout, LOG_PATH)
    sys.stdout._reteco_log = LOG_PATH
LOG_PATH = sys.stdout._reteco_log
print("log:", LOG_PATH)
'''

LLM_SETUP = r'''
# ── 2. Drive, vLLM, GPU ─────────────────────────────────────────────────────
import os, sys, subprocess
os.environ["HF_HUB_DISABLE_XET"] = "1"
# JAX (pulled in by bm25s) preallocates 75% of GPU memory by default, and vLLM then fails to start
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from google.colab import drive
drive.mount("/content/drive")
!pip -q install vllm openai pytrec-eval-terrier 2>&1 | tail -2
import torch

# vllm installs its own torch, while Colab's torchvision/torchaudio stay built for a different
# CUDA, and import vllm fails. Fix: uninstall torchaudio and reinstall torchvision
# for the new torch's CUDA. A runtime restart is needed afterwards.
def _imports_ok(mod):
    try:
        __import__(mod); return True
    except RuntimeError:
        return False       # installed, but built for a different CUDA
    except ImportError:
        return True        # not installed, so no conflict
fixed = False
if not _imports_ok("torchaudio"):
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchaudio"], check=True)
    fixed = True
if not _imports_ok("torchvision") and "+" in torch.__version__:
    base, cu = torch.__version__.split("+")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "--force-reinstall", "--no-deps",
                    f"torchvision==0.{int(base.split('.')[1]) + 15}.*",
                    "--index-url", f"https://download.pytorch.org/whl/{cu}"], check=True)
    fixed = True
if fixed:
    raise RuntimeError("Dependencies fixed. Runtime → Restart session, "
                       "then Run all again.")
if globals().get("NEED_GPU", True):
    assert torch.cuda.is_available(), "No GPU: Runtime → Change runtime type → A100 or L4"
MEM = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
DTYPE = "bfloat16" if MEM and torch.cuda.get_device_capability(0)[0] >= 8 else "half"   # T4 (sm75): no bf16
BIG = MEM >= 39                                                             # A100 40/80 GB
MID = MEM >= 22                                                             # L4 24 GB and above
print(torch.cuda.get_device_name(0) if MEM else "no GPU", f"{MEM:.0f} GB, {DTYPE}, torch", torch.__version__)

def load_llm(model, maxlen, util=0.88):
    """Uses the real stdout/stderr while the constructor runs: in a Jupyter kernel vLLM fails
    on sys.stdout.fileno() before the weights are even loaded."""
    from vllm import LLM
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        return LLM(model=model, dtype=DTYPE, max_model_len=maxlen, gpu_memory_utilization=util, seed=0)
    finally:
        sys.stdout, sys.stderr = out, err

def chat(tok, msgs):
    """Prompt built with the model's chat template; Qwen3 thinking is disabled (fast, deterministic)."""
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
'''

# ============================================================== embed ========
def embed_nb(model, title, note):
    cells = [
("markdown", f"""
# RETECO · {title}

Encodes all 507K track 2 documents with `{model}` and retrieves the
top 100 for every turn. Everything is saved to Google Drive under `reteco/`.

**How to run.** Runtime → Change runtime type → **T4 GPU**. Then
Runtime → **Run all**. Allow Drive access, then leave it running.

**If the session drops**, choose "Run all" again. Finished domains are on Drive
and are skipped; the run resumes where it stopped. Nothing is computed twice.

**How long.** On a T4, an estimated 2–4 hours for the whole corpus; the notebook
prints the exact speed after the first domain. An A100 is 5–8× faster. About
2.6 GB of free space is needed on Drive.

**Output.**
- `reteco/index/<model>/<domain>.npy`: document vectors (computed once)
- `reteco/runs/dense_<model>_<query>_<split>/<domain>/run.trec`: runs

**Next.** Copy `reteco/runs/` from Drive to `data/drive/reteco/runs/` of the repo.

**New query variants** (`query.ipynb` writes them to `reteco/queries`): "Run all"
again. The corpus vectors are already there, so only the queries are encoded (minutes).

{note}
"""),
("code", f'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
MODEL = "{model}"
BATCH = 32            # "CUDA out of memory" → 16, then 8
MAX_DOC_LEN = 512     # tokens per document; the median document is 20 words
MAX_Q_LEN = 1024      # a query with history can reach ~600 tokens
SPLITS = ["dev", "train"]
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Drive (configurable)
'''),
("code", COMMON_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. Model. Query and document prompts come from the model's own config ─────
from sentence_transformers import SentenceTransformer
if "model" not in globals():      # re-running "Run all" does not load a second copy onto the GPU
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={DTYPE_KEY: torch.float16},
                                trust_remote_code=True)
assert next(model.parameters()).dtype == torch.float16, "the model was not loaded in fp16"
Q_PROMPT = (model.prompts or {}).get("query", "")
D_PROMPT = (model.prompts or {}).get("document", "")
print("query prompt:   ", repr(Q_PROMPT))
print("document prompt:", repr(D_PROMPT))
SLUG = re.sub(r"[^A-Za-z0-9._-]", "_", MODEL)
IDX = f"{DRIVE}/index/{SLUG}"
os.makedirs(IDX, exist_ok=True)

def encode(texts, prompt, max_len):
    model.max_seq_length = max_len
    e = model.encode(texts, prompt=prompt, batch_size=BATCH, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True)
    assert not np.isnan(e).any(), "NaN in vectors: fp16 overflowed; use an A100 and bfloat16"
    return e

def search(emb, ids, qids, qtexts, topk=100):
    # scores in fp32: in fp16 close cosines collide, and pytrec_eval breaks ties by docno
    Q = torch.from_numpy(encode(qtexts, Q_PROMPT, MAX_Q_LEN)).cuda().float()
    E = torch.from_numpy(emb).cuda().float()
    run = {}
    for i in range(0, len(qids), 64):
        v, ix = (Q[i:i + 64] @ E.T).topk(min(topk, E.shape[0]), dim=1)
        for j, q in enumerate(qids[i:i + 64]):
            run[q] = {ids[k]: float(s) for k, s in zip(ix[j].tolist(), v[j].tolist())}
    del E, Q; torch.cuda.empty_cache()
    return run

def run_dir(qset, split):
    return f"{RUNS}/dense_{SLUG}_{qset}_{split}"
'''),
("code", r'''
# ── 6. Corpus vectors → Drive. A dev check after each domain ───────────────
sizes = {d: sum(1 for _ in open(f"{DATA}/{d}/documents.jsonl")) for d in DOMAINS}
rate = None
for dom in DOMAINS:
    out = f"{IDX}/{dom}.npy"
    if os.path.exists(out):
        print(f"[done] {dom}"); continue
    ids, docs = load_corpus(dom)
    t = time.time()
    parts = []
    for b in range(0, len(docs), 20000):     # in blocks: an interruption loses at most one block
        pp = f"{IDX}/{dom}.part{b // 20000:03d}.npy"
        if not os.path.exists(pp):
            with open(pp + ".tmp", "wb") as f:
                np.save(f, encode(docs[b:b + 20000], D_PROMPT, MAX_DOC_LEN).astype(np.float16))
            os.replace(pp + ".tmp", pp)
            print(f"  {dom}: {min(b + 20000, len(docs))}/{len(docs)} docs")
        parts.append(pp)
    emb = np.concatenate([np.load(p) for p in parts])
    sec = time.time() - t
    json.dump(ids, open(f"{IDX}/{dom}.ids.json", "w"))
    with open(out + ".tmp", "wb") as f:
        np.save(f, emb)
    os.replace(out + ".tmp", out)
    for p in parts:
        os.remove(p)
    rate = len(docs) / sec
    left = sum(sizes[d] for d in DOMAINS if not os.path.exists(f"{IDX}/{d}.npy"))
    # check: dense "turn + history" retrieval on dev against the official BM25
    qids, qtexts = topics(dom, "dev", "hist")
    run = search(emb, ids, qids, qtexts)
    write_run(f"{run_dir('hist', 'dev')}/{dom}/run.trec", run, "dense_hist")
    print(f"{dom}: {len(docs)} docs in {sec/60:.1f} min ({rate:.0f} docs/s), ≈{left/rate/60:.0f} min left · "
          f"nDCG@10 dev {ndcg10(run, dom, 'dev'):.4f} vs BM25 {BM25_DEV[dom]:.4f}")
print("corpus vectors ready:", IDX)
'''),
("code", r'''
# ── 7. Runs: all query sets × splits → Drive/reteco/runs ────────────────────
# Domain is the outer loop: each domain's vectors are read from Drive once for all query sets.
extra = sorted(os.path.basename(p) for p in glob.glob(f"{QUERIES}/*")
               if os.path.isdir(p) and os.path.basename(p) != "raw")
JOBS = [(q, s) for q in ["hist", "query"] + extra for s in SPLITS
        if q in ("query", "hist") or os.path.exists(f"{QUERIES}/{q}/{s}.jsonl")]
print("query sets × splits:", JOBS)
for dom in DOMAINS:
    todo = [(q, s) for q, s in JOBS if not os.path.exists(f"{run_dir(q, s)}/{dom}/run.trec")]
    if not todo:
        continue
    ids, emb = json.load(open(f"{IDX}/{dom}.ids.json")), np.load(f"{IDX}/{dom}.npy")
    for qset, split in todo:
        qids, qtexts = topics(dom, split, qset)
        write_run(f"{run_dir(qset, split)}/{dom}/run.trec", search(emb, ids, qids, qtexts), f"dense_{qset}")
    print(f"{dom}: {len(todo)} runs")
    del emb

summary = []
for qset, split in JOBS:
    rd = run_dir(qset, split)
    macro = sum(ndcg10(read_run(f"{rd}/{d}/run.trec"), d, split) for d in DOMAINS) / len(DOMAINS)
    json.dump({"model": MODEL, "query_prompt": Q_PROMPT, "doc_prompt": D_PROMPT,
               "max_doc_len": MAX_DOC_LEN, "max_q_len": MAX_Q_LEN, "dtype": "float16",
               "query_set": qset, "split": split, "topk": 100,
               "gpu": torch.cuda.get_device_name(0),
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
               "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{rd}/config.json", "w"), ensure_ascii=False, indent=2)
    summary.append((os.path.basename(rd), macro, BM25_MACRO[split]))
'''),
("code", r'''
# ── 8. Summary ──────────────────────────────────────────────────────────────
print(f"{'run':<60}{'nDCG@10':>9}{'BM25':>8}{'delta':>9}")
for name, m, b in summary:
    print(f"{name:<60}{m:>9.4f}{b:>8.4f}{m - b:>+9.4f}")
print("\nNext: copy", RUNS, "from Drive to data/drive/reteco/runs/ of the repo.")
'''),
    ]
    return nb(cells)

# ============================================================= rerank ========
def rerank_nb():
    cells = [
("markdown", r"""
# RETECO · top-100 reranking (Qwen3-Reranker-4B)

Takes a ready top-100 list for each turn (a first-stage run) and
reorders it: the model reads each "query + document" pair in full and judges
whether the document answers the query. The model never sees `doc_id`, only the text.

**When to run.** After the first retrieval stage has been chosen. The first-stage
run is in `reteco/runs/<name>/`: BM25 runs come from the project archive, dense runs
are written by `diver` and `reason_embed`, hybrids by `analysis`.

**How to run.** T4 GPU → set `FIRST_STAGE` and `QUERY_SET` in cell 1 →
"Run all". If it is interrupted, "Run all" again; finished domains
are skipped.

**How long.** 858 turns × 100 documents = 86K pairs. With short queries
(`qd`), an estimated 1–2 hours on a T4; with query + history (`hist`), 3–4× longer.
The exact speed is printed after the first domain.

**Output.** `reteco/runs/rerank_<first stage>_<query>/<domain>/run.trec`,
score = log-odds of "yes" vs "no". Fusion with the first stage (0.6 rerank + 0.4 retrieval)
is done locally: `scripts/fuse.py --mode score --weights 0.6 0.4`.

**Next.** Copy `reteco/runs/` from Drive to `data/drive/reteco/runs/` of the repo.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-Reranker-4B"
FIRST_STAGE = "bm25_fuse3_dev"   # folder in Drive/reteco/runs; BM25-fusion (dev nDCG@10 .550), the first stage of the reranker runs in Appendix B
QUERY_SET = "qd"      # what the reranker sees: "qd" = the rewritten standalone question (fast);
                      # "qd_hist" = with history (possibly more accurate, 3-4x slower); "query", "hist" = as provided by the organisers
SPLIT = "dev"
TOPK = 100            # how many candidates to rerank
BATCH = 8             # "CUDA out of memory" → 4
MAX_LEN = 1024        # tokens per pair; long documents are truncated from the end
TASK = "Given a web search query, retrieve relevant passages that answer the query"
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Drive (configurable)
'''),
("code", COMMON_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. Model. Format taken verbatim from the Qwen3-Reranker README ──────────
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if "lm" not in globals():         # re-running "Run all" does not load a second copy onto the GPU
    lm = AutoModelForCausalLM.from_pretrained(MODEL, **{DTYPE_KEY: torch.float16}).cuda().eval()
YES, NO = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PRE, SUF = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)

@torch.no_grad()
def score_pairs(pairs):
    """Log-odds of "yes" for each (query, document) pair. Pairs are sorted by length
    to reduce padding within a batch; the original order is restored afterwards."""
    texts = [f"<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
    enc = tok(texts, padding=False, truncation="longest_first", return_attention_mask=False,
              max_length=MAX_LEN - len(PRE) - len(SUF))
    seqs = [PRE + x + SUF for x in enc["input_ids"]]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    for b in tqdm(range(0, len(order), BATCH), leave=False):
        idx = order[b:b + BATCH]
        batch = tok.pad({"input_ids": [seqs[i] for i in idx]}, padding=True,
                        return_tensors="pt").to("cuda")
        try:
            logits = lm(**batch, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            logits = lm(**batch).logits[:, -1, :]
        s = (logits[:, YES] - logits[:, NO]).float()   # log-odds of "yes": same ranking, no saturation near 1.0
        for i, v in zip(idx, s.tolist()):
            out[i] = v
    return out
'''),
("code", r'''
# ── 6. Rerank by domain → Drive ─────────────────────────────────────────────
src = f"{RUNS}/{FIRST_STAGE}"
assert os.path.isdir(src), f"first-stage run not found: {src}"
dst = f"{RUNS}/rerank_{FIRST_STAGE}_{QUERY_SET}"
rows = []
for dom in DOMAINS:
    path = f"{dst}/{dom}/run.trec"
    first = read_run(f"{src}/{dom}/run.trec")
    if not os.path.exists(path):
        ids, texts = load_corpus(dom)
        text = dict(zip(ids, texts))
        qids, qtexts = topics(dom, SPLIT, QUERY_SET)
        qtext = dict(zip(qids, qtexts))
        cand = {q: sorted(first.get(q, {}).items(), key=lambda x: -x[1])[:TOPK] for q in qids}
        pairs = [(q, d) for q in qids for d, _ in cand[q]]
        assert pairs, f"{FIRST_STAGE} does not cover the {SPLIT} turns in {dom}"
        t = time.time()
        scores = score_pairs([(qtext[q], text[d]) for q, d in pairs])
        run = {}
        for (q, d), s in zip(pairs, scores):
            run.setdefault(q, {})[d] = s
        write_run(path, run, "rerank")
        print(f"{dom}: {len(pairs)} pairs in {(time.time() - t)/60:.1f} min "
              f"({len(pairs)/(time.time() - t):.0f} pairs/s)")
    a, b = ndcg10(first, dom, SPLIT), ndcg10(read_run(path), dom, SPLIT)
    rows.append((dom, a, b))
    print(f"  {dom:<20} first stage {a:.4f} → rerank {b:.4f}  ({b - a:+.4f})")
json.dump({"model": MODEL, "first_stage": FIRST_STAGE, "query_set": QUERY_SET, "split": SPLIT,
           "topk": TOPK, "max_len": MAX_LEN, "task": TASK, "dtype": "float16",
           "gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(f"{dst}/config.json", "w"), ensure_ascii=False, indent=2)
A, B = sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)
print(f"\nMACRO  first stage {A:.4f} → rerank {B:.4f}  ({B - A:+.4f});  BM25+history {BM25_MACRO[SPLIT]:.4f}")
print("Next: copy", dst, "from Drive to data/drive/reteco/runs/ of the repo.")
'''),
    ]
    return nb(cells)

# =========================================================== generate ========
GEN_WORKER = r'''
"""Answers of one reader for all jobs (2b, and 2c for each first stage). Run from
generate.ipynb as a separate process per reader, so that vLLM reliably frees GPU
memory before the next model. The summary cell also imports this module."""
import json, math, os, re, sys, time

OCC_STOP = [151643, 151645, 151683]   # <|endoftext|> <|im_end|> <|answer_end|>, from the OCC generation_config
CONTROL_SYSTEM = (
    "You are a helpful assistant in a multi-turn conversation. Answer the user's latest message "
    "using only the numbered documents attached to it.\n"
    "Reply in this format:\n"
    "Line 1: ANSWERABLE if the documents contain enough information to answer, otherwise UNANSWERABLE.\n"
    "Next lines: if ANSWERABLE, the answer in 2-5 sentences in a natural conversational tone; "
    "if UNANSWERABLE, say briefly what information is missing.")


def official_history(h, prev):
    """History limited to exactly what the official conversation_history contains: the
    (question, answer) pairs whose answer appears there; if older turns are compressed into
    "Earlier discussion covered: ...", that summary is prepended to the first question."""
    if not h or h.strip() == "No previous conversation.":
        return []
    hist = [(q, a) for q, a in prev if a and a in h]
    if "Earlier discussion covered" in h and hist:
        hist[0] = (h.split("\n\nRecent conversation:")[0].strip() + "\n\n" + hist[0][0], hist[0][1])
    return hist


def load_turns(data, domains, split):
    """Turns that have gold documents. history is the official dialogue history (official_history).
    reference is the gold answer of the CURRENT turn: used only by the audit and the judge,
    never in the reader prompt. Gold is only what appears in qrels (3 gold ids in politics
    are missing from the corpus). gold_in_prior counts the gold documents of this turn that
    were already gold for earlier turns: then the answer may be in the history."""
    out = []
    for dom in domains:
        qrels = {tuple(l.split()[0:3:2]) for l in open(f"{data}/{dom}/qrels_{split}.txt") if l.strip()}
        for c in json.load(open(f"{data}/{dom}/benchmark_{split}.json", encoding="utf-8")):
            prev, seen = [], set()
            for t in c["turns"]:
                qid = f"{c['id']}_turn_{t['turn_id']}"
                gold = list(dict.fromkeys(d for d in t.get("gold_doc_ids") or t.get("supporting_doc_ids") or []
                                          if (qid, d) in qrels))
                if gold:
                    out.append({"domain": dom, "topic_id": qid, "turn_id": t["turn_id"], "query": t["query"],
                                "history": official_history(t.get("conversation_history", ""), prev),
                                "gold": gold, "gold_in_prior": len(set(gold) & seen),
                                "reference": t.get("answer", "")})
                prev.append((t["query"], t.get("answer", "")))
                seen |= set(gold)
    return out


def read_top(path, k):
    run = {}
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) == 6:
            run.setdefault(p[0], []).append((int(p[3]), p[2]))
    return {q: [d for _, d in sorted(v)[:k]] for q, v in run.items()}


def contexts(C, turns, split, job):
    """"2b" → gold documents; "nodoc" → no documents (what the model knows on its own);
    "2c-<run>" → top-K from Drive/reteco/runs/<run>_<split>."""
    if job == "2b":
        return {(t["domain"], t["topic_id"]): t["gold"] for t in turns}
    if job == "nodoc":
        return {(t["domain"], t["topic_id"]): [] for t in turns}
    if job.startswith("ctrl-"):
        # judge-leniency control: top-K of the same stage for a turn from ANOTHER dialogue in the
        # same domain (half the dialogue list away, same turn number or else the first turn), with
        # this turn's own gold removed, so the context is known not to contain the answer
        tops = {dom: read_top(f"{C['drive']}/runs/{job[5:]}_{split}/{dom}/run.trec", 2 * C["k_ctx"])
                for dom in C["domains"]}
        out, by_dom = {}, {}
        for t in turns:
            by_dom.setdefault(t["domain"], []).append(t)
        for dom, ts in by_dom.items():
            dlg = sorted({t["topic_id"].rsplit("_turn_", 1)[0] for t in ts})
            ids = {t["topic_id"] for t in ts}
            for t in ts:
                d, k = t["topic_id"].rsplit("_turn_", 1)
                other = dlg[(dlg.index(d) + len(dlg) // 2) % len(dlg)]
                src = f"{other}_turn_{k}" if f"{other}_turn_{k}" in ids else f"{other}_turn_1"
                gold = set(t["gold"])
                out[(dom, t["topic_id"])] = [x for x in tops[dom].get(src, []) if x not in gold][:C["k_ctx"]]
        return out
    tops = {dom: read_top(f"{C['drive']}/runs/{job[3:]}_{split}/{dom}/run.trec", C["k_ctx"])
            for dom in C["domains"]}
    return {(t["domain"], t["topic_id"]): tops[t["domain"]].get(t["topic_id"], []) for t in turns}


def sufficiency(gold, ctx):
    """Number of gold documents in the context, and the label: all = all gold that fits in K; none = no gold."""
    n = len(set(gold) & set(ctx))
    full = min(len(gold), len(ctx)) if ctx else len(gold)
    return n, ("none" if n == 0 else "all" if n >= full else "part")


def corpus_texts(data, need):
    """need: {domain: {doc_id}} → {(domain, doc_id): text}."""
    out = {}
    for dom, ids in need.items():
        for ln in open(f"{data}/{dom}/documents.jsonl", encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                if d["doc_id"] in ids:
                    out[(dom, d["doc_id"])] = d["content"]
    return out


def history_msgs(t):
    m = []
    for q, a in t["history"]:
        m += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return m


class OCC:
    """The OCC chat template itself places the documents in the last user turn; generation
    starts with <|query_analysis_start|>, the status is <|status_start|>(UN)ANSWERABLE<|status_end|>."""
    stop, skip_special, marker = OCC_STOP, False, "<|status_start|>"
    STATUS = re.compile(r"<\|status_start\|>(.*?)(?:<\|status_end\|>|$)", re.S)
    ANSWER = re.compile(r"<\|answer_start\|>(.*?)(?:<\|answer_end\|>|<\|im_end\|>|$)", re.S)
    SPECIAL = re.compile(r"<\|[a-z_]+\|>")

    def __init__(self, tok):
        self.tok = tok

    def prompt(self, t, docs):
        msgs = history_msgs(t) + [{"role": "user", "content": t["query"]}]
        return self.tok.apply_chat_template(msgs, documents=[{"text": d} for d in docs],
                                            add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)

    def parse(self, raw):
        m = self.STATUS.search(raw)
        status = self.SPECIAL.sub("", m.group(1)).strip().upper() if m else ""
        m = self.ANSWER.search(raw)
        return status, (self.SPECIAL.sub("", m.group(1)).strip() if m else "")

    def prefix(self, raw):
        """Output up to the verdict position: the marker plus the newline after it (OCC always
        writes "<|status_start|>\nANSWERABLE"), so that the next token is the verdict."""
        i = raw.find(self.marker)
        if i < 0:
            return None
        j = i + len(self.marker)
        while j < len(raw) and raw[j] in "\r\n":
            j += 1
        return raw[:j]


class Control:
    """Qwen3 readers (the control prompt): same input, status on the first line."""
    stop, skip_special = None, True

    def __init__(self, tok):
        self.tok = tok

    def prompt(self, t, docs):
        numbered = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1)) or "(none)"
        msgs = ([{"role": "system", "content": CONTROL_SYSTEM}] + history_msgs(t)
                + [{"role": "user", "content": f"Documents:\n{numbered}\n\nQuestion: {t['query']}"}])
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)

    def parse(self, raw):
        first, _, rest = raw.strip().partition("\n")
        m = re.search(r"\b(UN)?ANSWERABLE\b", first.upper())
        if not m:
            return "", raw.strip()
        return m.group(0), rest.strip()

    def prefix(self, raw):
        """Everything before the status word on the first line (the model may start with "Line 1:" or **)."""
        lead = len(raw) - len(raw.lstrip())
        m = re.search(r"\b(?:UN)?ANSWERABLE\b", raw[lead:].split("\n", 1)[0], re.I)
        return raw[:lead + m.start()] if m else None


def p_status(tok, lp):
    """Probability of ANSWERABLE at the status position: summed probabilities of tokens that are
    prefixes of "ANSWERABLE" versus "UNANSWERABLE" among the top 20 (ANS... vs UN...). mass is
    how much probability fell on these two options at all."""
    a = u = 0.0
    for tid, x in lp.items():
        s = tok.decode([tid]).strip()
        if s and "UNANSWERABLE".startswith(s):
            u += math.exp(x.logprob)
        elif s and "ANSWERABLE".startswith(s):
            a += math.exp(x.logprob)
    return (round(a / (a + u), 6) if a + u > 0 else None), round(a + u, 6)


def main(reader_id, cfg_path):
    from vllm import LLM, SamplingParams
    C = json.load(open(cfg_path))
    short = reader_id.split("/")[-1]
    todo_any = False
    for split in C["splits"]:
        keys = {(t["domain"], t["topic_id"]) for t in load_turns(C["data"], C["domains"], split)}
        for job in C["jobs"]:
            path = f"{C['drive']}/gen/{short}_{job}_{split}.jsonl"
            done = ({(r["domain"], r["topic_id"]) for r in map(json.loads, open(path, encoding="utf-8"))}
                    if os.path.exists(path) else set())
            todo_any |= bool(keys - done)
    if not todo_any:
        print(f"{short}: everything is already done, skipping model load", flush=True)
        return
    llm = LLM(model=reader_id, dtype=C["dtype"], max_model_len=C["maxlen"],
              gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    R = (OCC if "OCC" in reader_id else Control)(tok)
    # OCC: greedy only, no repetition_penalty, otherwise the structured format breaks
    sp = SamplingParams(temperature=0.0, max_tokens=C["max_new"], stop_token_ids=R.stop,
                        skip_special_tokens=R.skip_special, spaces_between_special_tokens=False)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    short = reader_id.split("/")[-1]
    limit = C["maxlen"] - C["max_new"]
    ntok = lambda p: len(tok(p, add_special_tokens=False).input_ids)
    os.makedirs(f"{C['drive']}/gen", exist_ok=True)
    for split in C["splits"]:
        turns = load_turns(C["data"], C["domains"], split)
        jobs = [(j, contexts(C, turns, split, j)) for j in C["jobs"]]
        need = {}
        for _, ctx in jobs:
            for (dom, _), ids in ctx.items():
                need.setdefault(dom, set()).update(ids)
        text = corpus_texts(C["data"], need)
        for job, ctx in jobs:
            path = f"{C['drive']}/gen/{short}_{job}_{split}.jsonl"
            done = set()
            if os.path.exists(path):
                done = {(r["domain"], r["topic_id"]) for r in map(json.loads, open(path, encoding="utf-8"))}
            prompts, keep, skipped = [], [], 0
            for t in turns:
                if (t["domain"], t["topic_id"]) in done:
                    continue
                ids = ctx[(t["domain"], t["topic_id"])]
                docs = [text[(t["domain"], d)] for d in ids]
                p, trimmed = R.prompt(t, docs), False
                if ntok(p) > limit:      # does not fit: trim every document rather than drop gold
                    p, trimmed = R.prompt(t, [" ".join(d.split()[:C["trim_words"]]) for d in docs]), True
                if ntok(p) > limit:
                    skipped += 1
                    continue
                prompts.append(p)
                keep.append((t, ids, trimmed))
            print(f"{short} · {job} · {split}: to do {len(prompts)}, done {len(done)}, "
                  f"too long {skipped}", flush=True)
            t0 = time.time()
            with open(path, "a", encoding="utf-8") as f:
                for s in range(0, len(prompts), C["chunk"]):
                    outs = llm.generate(prompts[s:s + C["chunk"]], sp, use_tqdm=False)
                    raws = [o.outputs[0].text for o in outs]
                    # second pass: prompt + answer up to the status position, one token, top-20 logprobs
                    pref = [(i, R.prefix(r)) for i, r in enumerate(raws)]
                    pref = [(i, x) for i, x in pref if x is not None]
                    probs = {}
                    if pref:
                        o1 = llm.generate([prompts[s + i] + x for i, x in pref], sp1, use_tqdm=False)
                        for (i, _), o in zip(pref, o1):
                            probs[i] = p_status(tok, o.outputs[0].logprobs[0])
                    for i, (o, raw) in enumerate(zip(outs, raws)):
                        t, ids, trimmed = keep[s + i]
                        status, answer = R.parse(raw)
                        n_in, suff = sufficiency(t["gold"], ids)
                        p, mass = probs.get(i, (None, 0.0))
                        f.write(json.dumps({
                            "domain": t["domain"], "topic_id": t["topic_id"], "turn_id": t["turn_id"],
                            "split": split, "reader": reader_id, "mode": job.split("-")[0],
                            "first_stage": job[3:] if job.startswith("2c-") else None,
                            "context_ids": ids, "n_gold": len(t["gold"]), "gold_in_prior": t["gold_in_prior"],
                            "gold_in_ctx": n_in, "suff": suff, "status": status, "answer": answer,
                            "p_answerable": p, "p_mass": mass, "finish": o.outputs[0].finish_reason,
                            "trimmed": trimmed, "raw": raw}, ensure_ascii=False) + "\n")
                    f.flush()
                    k, el = s + len(outs), time.time() - t0
                    print(f"  {k}/{len(prompts)} · {el / 60:.1f} min · ≈{el / k * (len(prompts) - k) / 60:.0f} min left",
                          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''

def generate_nb():
    cells = [
("markdown", r"""
# RETECO · generating 2b and 2c answers (the paper's main experiment)

Four readers answer every dev turn from the same input: dialogue history +
current question + documents.
- **2b**: the gold documents, so the context is known to be sufficient.
- **2c**: the top 5 of one of the four retrievers of the paper's ladder, with dev nDCG@10:
  BM25-turn .18, BM25-hist .44, BM25-fusion .55 and the hybrid .59. The weaker the
  retrieval, the more often the context lacks the answer.
- **nodoc**: no documents (a control not analysed in the paper).

Readers: `occ-ai/OCC-RAG-1.7B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B-Instruct-2507` and
`Qwen/Qwen3-8B`. OCC writes a reasoning trace and then the status ANSWERABLE or
UNANSWERABLE; the Qwen3 readers are prompted to put the same status on the first line.
For every answer the probability of ANSWERABLE at the status position is stored as
`p_answerable`. Each record also stores the qrels label of its context (`suff`: all, part
or none of the gold documents are in the context).

**How to run.** Runtime: **A100** (preferred) or **L4**, then "Run all". If cell 2
asks for a restart, choose "Restart session" and "Run all" again.
If the run breaks off, "Run all" once more; finished answers are skipped.

**Cost.** 858 turns × 6 contexts × 4 readers = 20,592 answers. Estimated on an A100:
OCC 15–25 min, Qwen3-8B 30–60 min; 2–3× longer on an L4. Finished
files are skipped: an added retriever costs 858 × 4 answers.

**Output.**
- `reteco/gen/<reader>_<2b|nodoc|2c-first_stage>_dev.jsonl`: answers, status,
  p_answerable, qrels label, ids of the context documents
- `reteco/audit/sample.csv`: the 100 contexts of blind-audit batch 1 (key in `sample_key.csv`)

**Next.** `judge.ipynb` scores the answers. Copy `reteco/gen/` from Drive to
`data/drive/reteco/gen/` of the repo, and `reteco/audit/` to `data/drive/audit/`.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
# readers already finished in gen/ are skipped without loading the model. Qwen3-1.7B is post-trained
# from Qwen3-1.7B-Base, the model OCC is fine-tuned from; the Qwen3 readers use the control prompt (A100)
READERS = ["occ-ai/OCC-RAG-1.7B", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"]
# for 2c; folders runs/<name>_<split>. The reason-embed (Q2D) × bm25_fuse3 hybrid is the best 2a retrieval (dev .5913).
# bm25_fuse3 goes last: the audit sample is drawn from it (cell 8), so sample.csv stays unchanged
FIRST_STAGES = ["bm25_query", "bm25_hist", "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d", "bm25_fuse3"]
DO_2B = True
DO_NODOC = True       # no documents: what share the model knows on its own (separates memory from fabrication)
SPLITS = ["dev"]
K_CTX = 5             # documents in the 2c context
MAX_NEW = 1200        # OCC writes an analysis before the answer and needs room for it
MAXLEN = 16384        # gold can be up to 12 documents
TRIM_WORDS = 300      # if the prompt does not fit, each document is trimmed to this many words
CHUNK = 256           # answers between writes to Drive
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Google Drive
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", r'''
# ── 6. Generation: one separate process per reader ──────────────────────────
CONTROL = "Qwen/Qwen3-8B" if BIG else "Qwen/Qwen3-4B-Instruct-2507"
readers = [CONTROL if r == "control" else r for r in READERS]
jobs = (["2b"] if DO_2B else []) + (["nodoc"] if DO_NODOC else []) + [f"2c-{fs}" for fs in FIRST_STAGES]
for fs in FIRST_STAGES:
    for sp in SPLITS:
        assert os.path.isdir(f"{RUNS}/{fs}_{sp}"), f"missing run {RUNS}/{fs}_{sp}: upload it to Drive"
cfg = {"drive": DRIVE, "data": DATA, "domains": DOMAINS, "splits": SPLITS, "jobs": jobs,
       "k_ctx": K_CTX, "max_new": MAX_NEW, "maxlen": MAXLEN, "trim_words": TRIM_WORDS,
       "chunk": CHUNK, "dtype": DTYPE}
json.dump(cfg, open("/content/gen_cfg.json", "w"))

def stream(cmd):
    """Stream the child process output into the cell line by line."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"})
    for line in p.stdout:
        print(line, end="")
    return p.wait()

for r in readers:
    print(f"\n=== {r} ===")
    assert stream([sys.executable, "/content/gen_worker.py", r, "/content/gen_cfg.json"]) == 0, f"{r} failed, see the log above"
json.dump({**cfg, "readers": readers, "gpu": torch.cuda.get_device_name(0),
           "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(f"{DRIVE}/gen/config.json", "w"), ensure_ascii=False, indent=2)
'''),
("code", r'''
# ── 7. Summary: refusal rate by stratum and AUROC of the confidence ─────────
sys.path.insert(0, "/content")
import gen_worker as G
from sklearn.metrics import roc_auc_score

def auroc(rs, neg):
    """p_answerable: contexts with all gold (all) vs neg; NaN if a class is missing."""
    ab = [r for r in rs if r["p_answerable"] is not None and (r["suff"] == "all" or r in neg)]
    ys = [r["suff"] == "all" for r in ab]
    return roc_auc_score(ys, [r["p_answerable"] for r in ab]) if len(set(ys)) == 2 else float("nan")

print(f"{'file':<44}{'stratum':>7}{'n':>6}{'refuse':>7}{'p_ans':>7}{'AUROC':>7}")
for path in sorted(glob.glob(f"{DRIVE}/gen/*.jsonl")):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    for name, g in strata(rs):
        if not g:
            continue
        ref = sum(r["status"].startswith("UN") for r in g) / len(g)
        ps = [r["p_answerable"] for r in g if r["p_answerable"] is not None]
        auc = auroc(rs, g) if name in ("none", "clean") else float("nan")   # stratum keys from strata()
        print(f"{os.path.basename(path)[:-6]:<44}{name:>7}{len(g):>6}{ref:>7.2f}"
              f"{(np.mean(ps) if ps else float('nan')):>7.2f}{auc:>7.3f}")
print("\nrefuse = share of UNANSWERABLE. AUROC = how well p_answerable separates all from none (or from"
      "\n'clean' none, row 'clean': the gold never appeared earlier in the dialogue). In 2b every refusal is a false"
      "\nrefusal; nodoc T1 is the model's own knowledge, with no history. Unrecognized status:",
      sum(1 for p in glob.glob(f"{DRIVE}/gen/*.jsonl") for l in open(p) if not json.loads(l)["status"]))
'''),
("code", r'''
# ── 8. Sample for the blind audit, batch 1 → Drive/reteco/audit ───────────────
# 50 turns whose top 5 contains no gold document, and 50 whose top 5 contains all
# the gold that fits. sample.csv carries no label (blind audit): question, history,
# reference answer and top 5. answer_in_top5 is an empty column for an annotator;
# the released audit labels are in docs/audit_llm.csv.
import csv, random
AUDIT_STAGE, AUDIT_N = FIRST_STAGES[-1], 50
os.makedirs(f"{DRIVE}/audit", exist_ok=True)
turns = G.load_turns(DATA, DOMAINS, "dev")
ctx = G.contexts(cfg, turns, "dev", f"2c-{AUDIT_STAGE}")
lab = {(t["domain"], t["topic_id"]): G.sufficiency(t["gold"], ctx[(t["domain"], t["topic_id"])]) for t in turns}
rng = random.Random(0)
pick = []
for s in ("none", "all"):
    pool = [t for t in turns if lab[(t["domain"], t["topic_id"])][1] == s]
    pick += rng.sample(pool, min(AUDIT_N, len(pool)))
rng.shuffle(pick)
need = {}
for t in pick:
    need.setdefault(t["domain"], set()).update(ctx[(t["domain"], t["topic_id"])])
text = G.corpus_texts(DATA, need)
with open(f"{DRIVE}/audit/sample.csv", "w", newline="", encoding="utf-8") as f, \
     open(f"{DRIVE}/audit/sample_key.csv", "w", newline="", encoding="utf-8") as g:
    w, k = csv.writer(f), csv.writer(g)
    w.writerow(["n", "domain", "question", "history", "reference_answer", "top5", "answer_in_top5"])
    k.writerow(["n", "domain", "topic_id", "first_stage", "n_gold", "gold_in_top5", "suff"])
    for n, t in enumerate(pick, 1):
        key = (t["domain"], t["topic_id"])
        hist = "\n".join(f"Q: {q}\nA: {a}" for q, a in t["history"])
        top = "\n\n".join(f"[{i}] {text[(t['domain'], d)]}" for i, d in enumerate(ctx[key], 1))
        w.writerow([n, t["domain"], t["query"], hist, t["reference"], top, ""])
        k.writerow([n, t["domain"], t["topic_id"], AUDIT_STAGE, len(t["gold"]), *lab[key]])
print(f"{len(pick)} turns → {DRIVE}/audit/sample.csv; key in sample_key.csv")
'''),
    ]
    return nb(cells, "A100")

# ============================================================== judge ========
def judge_nb():
    cells = [
("markdown", r"""
# RETECO · judging 2b and 2c answers

Scores every answer in `reteco/gen` on the organizers' five axes (1–5):
correctness, completeness, relevance, conversational coherence, faithfulness.
Also the response type (answered / refused / partial) and whether it makes claims
not supported by the documents. The judge sees the history, the question, the documents
the reader saw, the reference answer (`turns[].answer`, which may be used for evaluation) and the response.

This notebook implements our own answer judge, not the task's official one. The paper's
answer judges are `Qwen/Qwen3-8B` and `microsoft/phi-4`; set `JUDGE` explicitly to
reproduce them. Two backends:
- `BACKEND = "vllm"`: an open model in this notebook (`JUDGE = "auto"` picks by GPU:
  A100: Qwen3-14B, L4: Qwen3-8B, T4: Qwen3-8B-AWQ). Free and reproducible.
- `BACKEND = "api"`: an OpenAI-compatible API, e.g. `gpt-4o`. The key goes in Colab
  secrets (key icon on the left): `OPENAI_API_KEY`, plus `OPENAI_BASE_URL` for another
  provider. No GPU needed.

The judge must be a different model from the reader: files from the same reader are skipped.

**How to run.** After `generate.ipynb`. Choose the backend in cell 1 →
"Run all". If the run breaks off, run it again; finished scores are skipped.

**Cost.** The paper's answer judgments cover OCC and Qwen3-4B on the gold documents, no
documents and the top 5 of the three BM25 retrievers: 858 turns × 5 contexts × 2 readers =
8,580 per judge (17,160 for both judges); `GEN_GLOB` in cell 1 selects the files to judge.
vLLM on an A100 takes 30–60 min per ≈7k answers; the API depends on the key's rate limits.

**Output.** `reteco/judge/<judge>/<generation file>.jsonl` and the summary
`reteco/judge/<judge>/summary.csv`.

**Next.** Copy `reteco/judge/` from Drive to `data/drive/reteco/judge/` of the repo.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
BACKEND = "vllm"      # "vllm" or "api"
JUDGE = "auto"        # vllm: auto = chosen by GPU (Qwen3); the paper's answer judges: "Qwen/Qwen3-8B" and "microsoft/phi-4" (A100);
                      # api: model name, e.g. "gpt-4o"
GEN_GLOB = "*_dev.jsonl"   # which files in Drive/reteco/gen to judge
MAXLEN = 16384
CHUNK = 256           # scores between writes to Drive
API_WORKERS = 8       # parallel API requests
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Google Drive
NEED_GPU = BACKEND == "vllm"
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", r'''
# ── 6. Judge prompt and output parsing ──────────────────────────────────────
sys.path.insert(0, "/content")
import gen_worker as G
AXES = ["correctness", "completeness", "relevance", "coherence", "faithfulness"]
SYSTEM = ("You are an impartial expert evaluator of conversational question answering systems. "
          "You grade one assistant response at a time, strictly and consistently.")
RUBRIC = """Score each criterion from 1 (very poor) to 5 (excellent):
- correctness: factual agreement with the reference answer; wrong or contradicting facts lower it.
- completeness: how many key points of the reference answer the response covers.
- relevance: the response addresses the latest question and stays on topic.
- coherence: the response fits the conversation as the next turn - consistent with earlier turns, resolves references naturally.
- faithfulness: every claim is supported by the provided documents or by the earlier conversation; a refusal that makes no claims gets 5.
Also classify:
- response_type: "answered" (a substantive answer), "refused" (declines or says the information is not available), or "partial" (answers part and says the rest is not available).
- unsupported_claims: true if the response states any fact supported neither by the provided documents nor by the earlier conversation.

Return only a JSON object with keys in this order: "rationale" (at most 3 sentences), "correctness", "completeness", "relevance", "coherence", "faithfulness", "response_type", "unsupported_claims"."""
REFUSAL = "I can't answer this from the provided documents."

def judge_msgs(t, docs, rec):
    hist = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in t["history"]) or "(none)"
    numbered = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1)) or "(none)"
    ans = rec["answer"] or (REFUSAL if rec["status"].startswith("UN") else "(empty)")
    user = (f"Grade the RESPONSE to the user's latest question in the conversation.\n\n"
            f"## Conversation so far\n{hist}\n\n## Latest question\n{t['query']}\n\n"
            f"## Documents the assistant was given\n{numbered}\n\n"
            f"## Reference answer (written by an expert who had the correct documents)\n{t['reference']}\n\n"
            f"## Response to grade\n{ans}\n\n{RUBRIC}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def parse_judge(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k in AXES:
        v = d.get(k)
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return None
        out[k] = int(round(v))
    rt = str(d.get("response_type", "")).strip().lower()
    uc = d.get("unsupported_claims")
    uc = uc.strip().lower() == "true" if isinstance(uc, str) else uc
    if rt not in ("answered", "refused", "partial") or not isinstance(uc, bool):
        return None
    return {**out, "response_type": rt, "unsupported_claims": uc, "rationale": str(d.get("rationale", ""))[:1000]}
'''),
("code", r'''
# ── 7. Backend: vLLM in this notebook or an API with a key ──────────────────
if BACKEND == "vllm":
    from vllm import SamplingParams
    if JUDGE == "auto":
        JUDGE = "Qwen/Qwen3-14B" if BIG else "Qwen/Qwen3-8B" if MID else "Qwen/Qwen3-8B-AWQ"
    if globals().get("_LOADED") != JUDGE:
        assert "_LOADED" not in globals(), "another model is in memory: Runtime → Restart session"
        llm = load_llm(JUDGE, MAXLEN, util=0.93); _LOADED = JUDGE   # Qwen3-8B on L4: at 0.88 the KV cache for 16k does not fit
    tok = llm.get_tokenizer()
    SP = SamplingParams(temperature=0.0, max_tokens=600)
    def run_batch(msg_lists):
        """A prompt that is too long does not break the batch: it gets TOO_LONG and is written as ok=False."""
        ps = [chat(tok, m) for m in msg_lists]
        fit = [i for i, p in enumerate(ps) if len(tok(p, add_special_tokens=False).input_ids) <= MAXLEN - 600]
        res = ["TOO_LONG"] * len(ps)
        if fit:
            for i, o in zip(fit, llm.generate([ps[i] for i in fit], SP, use_tqdm=False)):
                res[i] = o.outputs[0].text
        return res
else:
    from openai import OpenAI
    from google.colab import userdata
    from concurrent.futures import ThreadPoolExecutor
    assert JUDGE != "auto", "for api set the model in JUDGE, e.g. gpt-4o"
    def secret(k):
        try:
            return userdata.get(k)
        except Exception:
            return None
    client = OpenAI(api_key=secret("OPENAI_API_KEY"), base_url=secret("OPENAI_BASE_URL") or None)
    def call(msgs):
        for attempt in range(6):
            try:
                r = client.chat.completions.create(model=JUDGE, messages=msgs, temperature=0, max_tokens=600,
                                                   response_format={"type": "json_object"})
                return r.choices[0].message.content or ""
            except Exception as e:
                print("  API:", type(e).__name__, str(e)[:120]); time.sleep(2 ** attempt)
        return ""
    def run_batch(msg_lists):
        with ThreadPoolExecutor(API_WORKERS) as ex:
            return list(ex.map(call, msg_lists))
SHORT = JUDGE.split("/")[-1]
OUTD = f"{DRIVE}/judge/{SHORT}"
os.makedirs(OUTD, exist_ok=True)
print("judge:", JUDGE, "→", OUTD)
'''),
("code", r'''
# ── 8. Scoring of all generation files → Drive ──────────────────────────────
turns = {}
_gcfg = f"{DRIVE}/gen/config.json"
TRIM_WORDS = json.load(open(_gcfg)).get("trim_words", 300) if os.path.exists(_gcfg) else 300
base = lambda n: re.sub(r"-(AWQ|GPTQ[\w-]*|FP8|Int[48][\w-]*|GGUF)$", "", n.split("/")[-1], flags=re.I)

def docs_of(r, text):
    """Documents exactly as the reader saw them (trimmed if generation trimmed them)."""
    docs = [text[(r["domain"], d)] for d in r["context_ids"]]
    return [" ".join(x.split()[:TRIM_WORDS]) for x in docs] if r.get("trimmed") else docs

for path in sorted(glob.glob(f"{DRIVE}/gen/{GEN_GLOB}")):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not recs:
        continue
    name = os.path.basename(path)
    if base(recs[0]["reader"]) == base(JUDGE):
        print(f"[SKIP] {name}: reader {recs[0]['reader']} and judge {JUDGE} are the same model"); continue
    split = recs[0]["split"]
    if split not in turns:
        turns[split] = {(t["domain"], t["topic_id"]): t for t in G.load_turns(DATA, DOMAINS, split)}
    out = f"{OUTD}/{name}"
    done = {(r["domain"], r["topic_id"]) for r in map(json.loads, open(out, encoding="utf-8"))} if os.path.exists(out) else set()
    todo = [r for r in recs if (r["domain"], r["topic_id"]) not in done and (r["domain"], r["topic_id"]) in turns[split]]
    need = {}
    for r in todo:
        need.setdefault(r["domain"], set()).update(r["context_ids"])
    text = G.corpus_texts(DATA, need)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), CHUNK):
            part = todo[s:s + CHUNK]
            msgs = [judge_msgs(turns[split][(r["domain"], r["topic_id"])], docs_of(r, text), r) for r in part]
            raws = run_batch(msgs)
            bad = [i for i, x in enumerate(raws) if parse_judge(x) is None]
            if bad:          # one retry with a reminder to return JSON
                again = run_batch([m[:-1] + [{"role": "user", "content": m[-1]["content"]
                                   + "\n\nReturn ONLY the JSON object, nothing else."}] for m in (msgs[i] for i in bad)])
                for i, x in zip(bad, again):
                    raws[i] = x
            if all(x == "" for x in raws):
                raise RuntimeError("the judge is not responding (quota/network); nothing was written, rerun later")
            for r, raw in zip(part, raws):
                if raw == "":
                    continue          # no API response: stays in the queue for the next run
                j = parse_judge(raw)
                f.write(json.dumps({"domain": r["domain"], "topic_id": r["topic_id"], "split": split,
                                    "reader": r["reader"], "mode": r["mode"], "first_stage": r["first_stage"],
                                    "suff": r["suff"], "status": r["status"], "judge": JUDGE,
                                    "turn_id": r.get("turn_id"), "gold_in_prior": r.get("gold_in_prior"),
                                    "ok": j is not None, **(j or {}), "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            k, el = s + len(part), time.time() - t0
            print(f"  {name}: {k}/{len(todo)} · {el / 60:.1f} min · ≈{el / k * (len(todo) - k) / 60:.0f} min left")
    print(f"{name}: done")
'''),
("code", r'''
# ── 9. Summary: mean per axis, refusals and unsupported claims; 2c by qrels label ──
import csv
rows = []
for path in sorted(glob.glob(f"{OUTD}/*.jsonl")):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    good = [r for r in rs if r["ok"]]
    for suff, g in strata(good):
        if not g:
            continue
        n = len(g)
        rows.append({"file": os.path.basename(path)[:-6], "suff": suff, "n": n,
                     **{a: round(sum(r[a] for r in g) / n, 3) for a in AXES},
                     "refused": round(sum(r["response_type"] == "refused" for r in g) / n, 3),
                     "unsupported": round(sum(r["unsupported_claims"] for r in g) / n, 3),
                     "parse_fail": sum(not r["ok"] for r in rs) if suff == "total" else ""})   # on the "total" stratum only
assert rows, f"no parsed scores in {OUTD}; check the raw field in the files"
with open(f"{OUTD}/summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'file':<42}{'suff':>6}{'n':>5}" + "".join(f"{a[:5]:>7}" for a in AXES) + f"{'refuse':>7}{'unsup':>7}")
for r in rows:
    print(f"{r['file']:<42}{r['suff']:>6}{r['n']:>5}" + "".join(f"{r[a]:>7.2f}" for a in AXES)
          + f"{r['refused']:>7.2f}{r['unsupported']:>7.2f}")
print("\nunsup = share of responses with claims not supported by the documents. Summary:", f"{OUTD}/summary.csv")
'''),
    ]
    return nb(cells, "A100")

# Sufficiency judge prompt and response parsing: a single source for suff.ipynb and mtrag.ipynb
SUFF_PROMPT = r'''
LABELS = ["yes", "partial", "no"]
SYSTEM = ("You are a careful annotator for a conversational question answering benchmark. "
          "You decide whether a set of documents contains the information needed to answer a question. "
          "You judge only by what the documents say, never by your own knowledge.")
TASK = """Decide whether the DOCUMENTS contain the information needed for the core claim(s) of the reference answer to the latest question.

Labels:
- "yes": the documents support the core claim(s) of the reference answer for the latest question. Missing minor details, examples or exact wording is fine.
- "partial": the documents support at least one core point, but a central part of the answer is missing.
- "no": the documents do not contain the answer. Text on the same topic, or text that only restates the question, without the actual answer counts as "no".
Rules:
- Judge only by the documents, not by your own knowledge.
- The earlier conversation is not evidence: if the answer appears only there and not in the documents, it does not count.
- answered_in_history: true if the earlier conversation already states the answer to the latest question, otherwise false.
- evidence_docs: the numbers of the documents that support the answer; an empty list if none.

Return only a JSON object with keys in this order: {"rationale": "<at most 2 sentences>", "label": "yes" | "partial" | "no", "answered_in_history": true | false, "evidence_docs": [<document numbers>]}"""

def suff_msgs(t, docs):
    hist = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in t["history"]) or "(none)"
    numbered = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1)) or "(none)"
    user = (f"## Conversation so far\n{hist}\n\n## Latest question\n{t['query']}\n\n"
            f"## Reference answer (written by an expert; use it only to know what the correct answer says)\n"
            f"{t['reference']}\n\n## Documents\n{numbered}\n\n{TASK}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def parse_suff(raw, n_docs):
    """First {...} in the response → fields; anything that fails to parse is None (the raw response is kept)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0))
        assert isinstance(d, dict)
    except Exception:
        return {"label": None, "answered_in_history": None, "evidence_docs": [], "rationale": None}
    lab = str(d.get("label", "")).strip().strip('"').lower()
    ah = d.get("answered_in_history")
    ah = {"true": True, "false": False}.get(ah.strip().lower()) if isinstance(ah, str) else ah
    ev = d.get("evidence_docs")
    nums = [str(x).strip("[] ") for x in (ev if isinstance(ev, list) else [])]   # 2, "2", "[2]"
    return {"label": lab if lab in LABELS else None,
            "answered_in_history": ah if isinstance(ah, bool) else None,
            "evidence_docs": sorted({int(x) for x in nums if x.isdigit() and 1 <= int(x) <= n_docs}),
            "rationale": str(d.get("rationale", ""))[:1000]}
'''

# =============================================================== suff ========
def suff_nb():
    cells = [
("markdown", r"""
# RETECO · content label of context sufficiency (the judge reads the documents)

The qrels label (how many gold documents are in the top-5) calls every context without
gold insufficient, but such contexts can contain the answer: phi-4 labels 28% of them
sufficient on BM25-turn and 72% on the hybrid (false misses, Table 1 of the paper). Here,
for every dev context the readers saw, an open sufficiency judge (`microsoft/phi-4`; the
paper's second judge is `Qwen/Qwen3-14B`) reads the documents themselves and assigns the
content label:
- **yes**: the documents support the core claims of the reference answer
  (minor details may be missing);
- **partial**: some of the core is there, but a central part is missing;
- **no**: the documents do not contain the answer (on-topic text without the answer is also no).

It also outputs `answered_in_history` (the answer was already given earlier in the
dialogue) and `evidence_docs` (numbers of the documents that contain the answer). The
judge sees the history, the question, the reference answer (`turns[].answer` may be used
for labeling) and the documents `[1] …` without doc_id. Contexts are built exactly as in
`generate.ipynb`: 2b = gold documents, 2c = first-stage top-5.

**How to run.** Runtime → **A100** (phi-4 in bf16 is ≈29 GB and does not fit on an L4).
Run all. If cell 2 asks for a restart, restart the session and run all again. If the
run is interrupted, run it again; finished labels are skipped.
For a summary of existing labels only (after `generate`), set `LABEL = False`; no GPU needed.

**Cost.** 858 turns × 5 contexts = 4,290 prompts of 2–5k tokens each, estimated
30–60 min on an A100 (faster on 80 GB: more room for the KV cache), plus 858 prompts for
the negative control (`ctrl-…`). Finished labels are skipped: an added retriever is
858 prompts, ≈10 min.

**Output.**
- `reteco/suff/phi-4_<job>_dev.jsonl` (job: `2b`, `2c-<first stage>` or `ctrl-<first stage>`):
  content label, rationale, raw judge response, and the qrels label alongside
- `reteco/analysis/suff_relabel.csv`: qrels label × content label, reader refusals by
  content label, and AUROC of p_answerable (yes vs. no); not part of the release

**Next.** Copy `reteco/suff/` from Drive to `data/drive/reteco/suff/` of the repo.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
JUDGE = "microsoft/phi-4"   # a different family from both readers (OCC and Qwen3); second judge: Qwen/Qwen3-14B
JOBS = ["2b", "2c-bm25_query", "2c-bm25_hist", "2c-bm25_fuse3",   # the same contexts as in generate
        "2c-hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d",
        "ctrl-hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"]   # control: top-5 of another dialogue
SPLITS = ["dev"]
K_CTX = 5             # documents in a 2c context, as in generate
MAXLEN = 16384        # phi-4 does not allow more; a gold set can have up to 12 documents
MAX_NEW = 400         # 2-sentence rationale + JSON
TRIM_WORDS = 300      # if a prompt does not fit, each document is cut to this many words
CHUNK = 256           # labels between writes to Drive
LABEL = True          # False: only summarize existing labels (no GPU needed)
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Google Drive
NEED_GPU = LABEL
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", r'''
# ── 6. Judge prompt and response parsing ────────────────────────────────────
sys.path.insert(0, "/content")
import gen_worker as G
''' + SUFF_PROMPT.lstrip("\n")),
("code", r'''
# ── 7. Judge in vLLM ─────────────────────────────────────────────────────────
SHORT = JUDGE.split("/")[-1]
OUTS = f"{DRIVE}/suff"
os.makedirs(OUTS, exist_ok=True)
if LABEL:
    from vllm import SamplingParams
    if "phi-4" in JUDGE:
        assert BIG, "phi-4 in bf16 is ≈29 GB: Runtime → Change runtime type → A100"
    if globals().get("_LOADED") != JUDGE:
        assert "_LOADED" not in globals(), "another model is already loaded: Runtime → Restart session"
        llm = load_llm(JUDGE, MAXLEN, util=0.9); _LOADED = JUDGE
    tok = llm.get_tokenizer()
    SP = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
print("judge:", JUDGE, "→", OUTS, "" if LABEL else "(LABEL = False: summary only)")
'''),
("code", r'''
# ── 8. Labeling: every context of every turn → Drive/reteco/suff ───────────
cfg = {"drive": DRIVE, "domains": DOMAINS, "k_ctx": K_CTX}   # what G.contexts needs
for job in (JOBS if LABEL else []):
    for sp in SPLITS:
        if job.startswith("2c-"):
            assert os.path.isdir(f"{RUNS}/{job[3:]}_{sp}"), f"missing run {RUNS}/{job[3:]}_{sp}: upload it to Drive"
limit = MAXLEN - MAX_NEW
ntok = lambda p: len(tok(p, add_special_tokens=False).input_ids)
for split in (SPLITS if LABEL else []):   # LABEL = False: labeling is skipped
    turns = G.load_turns(DATA, DOMAINS, split)
    ctxs = {job: G.contexts(cfg, turns, split, job) for job in JOBS}
    need = {}
    for ctx in ctxs.values():
        for (dom, _), ids in ctx.items():
            need.setdefault(dom, set()).update(ids)
    text = G.corpus_texts(DATA, need)
    for job in JOBS:
        ctx, path = ctxs[job], f"{OUTS}/{SHORT}_{job}_{split}.jsonl"
        done = set()
        if os.path.exists(path):
            done = {(r["domain"], r["topic_id"]) for r in map(json.loads, open(path, encoding="utf-8"))}
        prompts, keep, skipped = [], [], 0
        for t in turns:
            key = (t["domain"], t["topic_id"])
            if key in done:
                continue
            ids = ctx[key]
            docs = [text[(t["domain"], d)] for d in ids]
            p, trimmed = chat(tok, suff_msgs(t, docs)), False
            if ntok(p) > limit:      # does not fit: trim every document, as in generate
                p, trimmed = chat(tok, suff_msgs(t, [" ".join(d.split()[:TRIM_WORDS]) for d in docs])), True
            if ntok(p) > limit:
                skipped += 1
                continue
            prompts.append(p)
            keep.append((t, ids, trimmed))
        print(f"{SHORT} · {job} · {split}: to do {len(prompts)}, done {len(done)}, did not fit {skipped}")
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for s in range(0, len(prompts), CHUNK):
                outs = llm.generate(prompts[s:s + CHUNK], SP, use_tqdm=False)
                for (t, ids, trimmed), o in zip(keep[s:s + CHUNK], outs):
                    raw = o.outputs[0].text
                    n_in, suff = G.sufficiency(t["gold"], ids)
                    f.write(json.dumps({
                        "domain": t["domain"], "topic_id": t["topic_id"], "turn_id": t["turn_id"],
                        "split": split, "job": job, "judge": JUDGE, "context_ids": ids,
                        "n_gold": len(t["gold"]), "gold_in_ctx": n_in, "suff_qrels": suff,
                        "gold_in_prior": t["gold_in_prior"], **parse_suff(raw, len(ids)),
                        "raw": raw, "trimmed": trimmed}, ensure_ascii=False) + "\n")
                f.flush()
                k, el = s + len(outs), time.time() - t0
                print(f"  {k}/{len(prompts)} · {el / 60:.1f} min · ≈{el / k * (len(prompts) - k) / 60:.0f} min left")
if LABEL:
    json.dump({"judge": JUDGE, "jobs": JOBS, "splits": SPLITS, "k_ctx": K_CTX, "maxlen": MAXLEN,
               "max_new": MAX_NEW, "trim_words": TRIM_WORDS, "dtype": DTYPE, "system": SYSTEM, "task": TASK,
               "gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{OUTS}/config.json", "w"), ensure_ascii=False, indent=2)
'''),
("code", r'''
# ── 9. Summary (no GPU needed): qrels × content label; reader refusals by content label ──
import csv
from sklearn.metrics import roc_auc_score
NEW = LABELS + [None]                     # None: the judge response could not be parsed
SHOW = lambda x: x or "—"
rows = []

def auroc(pos, neg):
    """AUROC of p_answerable: pos (sufficient context) vs. neg; nan if a class is missing."""
    ab = [(1, r["p_answerable"]) for r in pos if r["p_answerable"] is not None] + \
         [(0, r["p_answerable"]) for r in neg if r["p_answerable"] is not None]
    ys = [y for y, _ in ab]
    return roc_auc_score(ys, [p for _, p in ab]) if len(set(ys)) == 2 else float("nan")

print("(a) qrels label (rows) × judge content label (columns); hist = share of answered_in_history")
print(f"{'job':<16}{'qrels':>6}{'n':>6}" + "".join(f"{SHOW(x):>8}" for x in NEW) + f"{'hist':>6}")
SUFF = {}
for job in JOBS:
    path = f"{OUTS}/{SHORT}_{job}_dev.jsonl"
    if not os.path.exists(path):
        print(f"{job:<16} missing {path}"); continue
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    SUFF[job] = {(r["domain"], r["topic_id"]): r for r in rs}
    for q in ("all", "part", "none"):
        g = [r for r in rs if r["suff_qrels"] == q]
        if not g:
            continue
        cnt = {x: sum(r["label"] == x for r in g) for x in NEW}
        hist = sum(r["answered_in_history"] is True for r in g) / len(g)
        print(f"{job:<16}{q:>6}{len(g):>6}" + "".join(f"{cnt[x]:>8}" for x in NEW) + f"{hist:>6.2f}")
        rows.append({"table": "confusion", "job": job, "suff_qrels": q, "n": len(g),
                     **{f"n_{x or 'invalid'}": cnt[x] for x in NEW}, "in_history": round(hist, 3)})

print("\n(b) reader refusals by the content label; AUROC of p_answerable: yes vs. no (row no) and vs."
      "\n'clean' no, where the answer is not in the history either (row clean); AUC-q: all vs. none by qrels, for comparison")
print(f"{'file':<44}{'label':>8}{'n':>6}{'refuse':>7}{'p_ans':>7}{'AUROC':>7}{'AUC-q':>7}")
for job in SUFF:
    for path in sorted(glob.glob(f"{DRIVE}/gen/*_{job}_dev.jsonl")):
        name = os.path.basename(path)[:-6]
        if "nodoc" in name:
            continue
        rs = []
        for r in map(json.loads, open(path, encoding="utf-8")):
            s = SUFF[job].get((r["domain"], r["topic_id"]))
            if s is not None:
                rs.append({**r, "label": s["label"], "in_hist": s["answered_in_history"]})
        if not rs:
            continue
        by = {x: [r for r in rs if r["label"] == x] for x in NEW}
        clean = [r for r in by["no"] if r["in_hist"] is False]
        aq = auroc([r for r in rs if r["suff"] == "all"], [r for r in rs if r["suff"] == "none"])
        for lab, g in [("total", rs), *[(SHOW(x), by[x]) for x in NEW], ("clean", clean)]:
            if not g:
                continue
            ref = sum(r["status"].startswith("UN") for r in g) / len(g)
            ps = [r["p_answerable"] for r in g if r["p_answerable"] is not None]
            auc = auroc(by["yes"], g) if lab in ("no", "clean") else float("nan")
            print(f"{name[:43]:<44}{lab:>8}{len(g):>6}{ref:>7.2f}{(np.mean(ps) if ps else float('nan')):>7.2f}{auc:>7.3f}"
                  + (f"{aq:>7.3f}" if lab == "total" else ""))
            rows.append({"table": "refusal", "file": name, "reader": rs[0].get("reader"), "job": job, "label": lab,
                         "n": len(g), "refused": round(ref, 3),
                         "p_ans": round(float(np.mean(ps)), 3) if ps else None,
                         "auroc": None if np.isnan(auc) else round(auc, 3),
                         "auroc_qrels": (None if np.isnan(aq) else round(aq, 3)) if lab == "total" else None})
assert rows, f"no labels in {OUTS}: run labeling first (LABEL = True)"
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
os.makedirs(f"{DRIVE}/analysis", exist_ok=True)
REL = f"{DRIVE}/analysis/suff_relabel" + ("" if SHORT == "phi-4" else f"_{SHORT}") + ".csv"   # one file per judge
with open(REL, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: r.get(k) for k in keys} for r in rows])
bad = sum(r["label"] is None for d in SUFF.values() for r in d.values())
print(f"\nunparsed labels: {bad} (raw response in field raw). Table: {REL}")
'''),
    ]
    return nb(cells, "A100")

# ============================================================== query ========
def query_nb():
    cells = [
("markdown", r"""
# RETECO · the "Query" block with an open LLM + the kit's BM25

Adds query variants that were not in the Claude-generated run, plus feedback-based
expansion as in DIVER. Everything is retrieved with the **kit's official BM25** (gensim +
Lucene analyzer), so the numbers are directly comparable with the table: the first
retrieval cell checks, domain by domain, that BM25 + history gives exactly 0.4379 on dev.

Variants (each is appended to "turn + history", like Claude's q2d):
- `win1`, `win2`: only the last 1–2 history pairs instead of the full history (no LLM)
- `stepback`: a more general question behind the original one
- `subq`: 2–3 sub-questions; each is searched separately and the lists are fused with RRF
- `re`: the last assistant answer rewritten to be self-contained
- `q2d_open`: a hypothetical passage from the open model (the same q2d as Claude's)
- `kw`: 30 keywords from that passage that are not in the query
- `qexp1`, `qexp2`: feedback expansion (DIVER QExpand): the model sees the top-5
  retrieved documents and rewrites the passage; the query keeps turn + history +
  the latest expansion

**How to run.** Runtime → **A100** (Qwen3-8B) or **L4** (Qwen3-4B-Instruct-2507).
Run all. If cell 2 asks for a restart, restart the session and run all again. If the
run is interrupted, run it again: LLM responses and finished runs are kept on Drive.

**Cost.** ≈3k dev + train turns. LLM: ≈17k short generations,
15–30 min on an A100; BM25: indexing 10–15 min, retrieval another 10–20.

**Output.**
- `reteco/queries/<variant>/<split>.jsonl`: query texts (later picked up by
  `diver` and `reason_embed`, where runs with them take only minutes)
- `reteco/runs/bm25_<variant>_<split>/`: runs
- `reteco/qgen/<model>/`: raw LLM responses (cache)

**Next.** Copy `reteco/runs/` and `reteco/queries/` from Drive to `data/drive/reteco/runs/`
and `data/drive/reteco/queries/` of the repo. Variants are selected on train; dev is for
reporting only.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
LLM_MODEL = "auto"    # A100 → Qwen/Qwen3-8B, otherwise Qwen/Qwen3-4B-Instruct-2507
SPLITS = ["dev", "train"]
FEEDBACK_K = 5        # number of retrieved documents the LLM sees in QExpand
DOC_WORDS = 300       # each document in the prompt is cut to this many words
N_KW = 30             # keywords in the kw variant
MAXLEN = 8192
MAX_NEW = 256
TOPK = 100
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Google Drive
'''),
("code", LLM_SETUP),
("code", r'''
# ── 2b. Java 21, pyserini (Lucene analyzer only), gensim, organizers' kit ──
# pyserini is installed without dependencies: only its analyzer is needed, and
# its full dependencies (torch, transformers) would break vLLM.
import glob
!apt-get -qq update > /dev/null && apt-get -qq install -y openjdk-21-jdk-headless > /dev/null
!pip -q install --no-deps pyserini==2.4.0
!pip -q install pyjnius gensim
!git clone -q https://github.com/DataScienceUIBK/RETECO /content/reteco_kit 2>/dev/null || true
jdk = (glob.glob("/usr/lib/jvm/java-21-openjdk*") or [""])[0]
os.environ.setdefault("JAVA_HOME", jdk)
os.environ.setdefault("JVM_PATH", f"{jdk}/lib/server/libjvm.so")
sys.path.insert(0, "/content/reteco_kit/starter_kit")
print("JAVA_HOME", os.environ["JAVA_HOME"])
'''),
("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. LLM (loaded before Java: vLLM starts its own process and does not need the JVM) ──
from vllm import SamplingParams
if LLM_MODEL == "auto":
    LLM_MODEL = "Qwen/Qwen3-8B" if BIG else "Qwen/Qwen3-4B-Instruct-2507"
if globals().get("_LOADED") != LLM_MODEL:
    assert "_LOADED" not in globals(), "another model is already loaded: Runtime → Restart session"
    llm = load_llm(LLM_MODEL, MAXLEN); _LOADED = LLM_MODEL
tok = llm.get_tokenizer()
SP = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
SHORT = LLM_MODEL.split("/")[-1]
QGEN = f"{DRIVE}/qgen/{SHORT}"
os.makedirs(QGEN, exist_ok=True)
SYSTEM = "You help a search engine find documents that answer the latest question in a conversation."

def ask(kind, split, items):
    """items: [(key, prompt)] → {key: response}. Cached on Drive: qgen/<model>/<kind>_<split>.jsonl."""
    path = f"{QGEN}/{kind}_{split}.jsonl"
    got = {}
    if os.path.exists(path):
        got = {tuple(r["key"]): r["out"] for r in map(json.loads, open(path, encoding="utf-8"))}
    todo = [(k, p) for k, p in items if k not in got]
    t0 = time.time()
    for s in range(0, len(todo), 512):          # written to Drive every 512: an interruption loses at most that many
        part = todo[s:s + 512]
        outs = llm.generate([chat(tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}])
                             for _, p in part], SP, use_tqdm=False)
        with open(path, "a", encoding="utf-8") as f:
            for (k, _), o in zip(part, outs):
                got[k] = o.outputs[0].text.strip()
                f.write(json.dumps({"key": list(k), "out": got[k]}, ensure_ascii=False) + "\n")
        n, el = s + len(part), time.time() - t0
        print(f"  LLM {kind} {split}: {n}/{len(todo)} · {el / 60:.1f} min · ≈{el / n * (len(todo) - n) / 60:.0f} min left")
    return got
print("LLM:", LLM_MODEL)
'''),
("code", r'''
# ── 6. Kit BM25 + check: BM25 + history on dev = organizers' table ─────────
from official_baseline import OfficialBM25
from pyserini.analysis import Analyzer, get_lucene_analyzer
AN = Analyzer(get_lucene_analyzer())
_BM = {}

def bm25(dom):
    if dom not in _BM:
        ids, texts = load_corpus(dom)
        _BM[dom] = OfficialBM25(texts, ids)
    return _BM[dom]

def search(dom, qids, texts):
    """Same as OfficialBM25.search, but the top is taken with numpy (stable sort:
    tied scores keep corpus order, as in the kit) and cut to TOPK right away."""
    b = bm25(dom)
    ids = b.doc_ids
    run = {}
    for q, t in zip(qids, texts):
        sims = np.asarray(b.index[b.model[b.dictionary.doc2bow(b.analyzer.analyze(t))]])
        top = np.argsort(-sims, kind="stable")[:TOPK]
        run[q] = {ids[i]: float(sims[i]) for i in top}
    return run

t0 = time.time()
for dom in DOMAINS:
    qids, qtexts = topics(dom, "dev", "hist")
    got = ndcg10(search(dom, qids, qtexts), dom, "dev")
    print(f"  {dom:<20} {got:.4f}  table {BM25_DEV[dom]:.4f}  · {(time.time() - t0) / 60:.1f} min")
    assert abs(got - BM25_DEV[dom]) < 0.0015, "BM25 does not match the official numbers; stopping here"
print("BM25 matches the official numbers")
'''),
("code", r'''
# ── 7. Turns and prompts ─────────────────────────────────────────────────────
NO_HIST = "No previous conversation."

def load_turns(split):
    """The same turns as topics(): only those with gold. h is conversation_history as is,
    prev is the (question, answer) pairs parsed from h. The turns' answer field is never used."""
    out = []
    for dom in DOMAINS:
        for c in json.load(open(f"{DATA}/{dom}/benchmark_{split}.json", encoding="utf-8")):
            for t in c["turns"]:
                if t.get("gold_doc_ids") or t.get("supporting_doc_ids"):
                    h = t.get("conversation_history", "")
                    h = "" if h == NO_HIST else h
                    out.append({"key": (dom, f"{c['id']}_turn_{t['turn_id']}"), "q": t["query"],
                                "h": h, "prev": HIST_QA.findall(h)})
    return out

HIST_QA = re.compile(r"(?:^|\n)Q: (.*?)\nA: (.*?)(?=\nQ: |\Z)", re.S)

def re_hist(t, new):
    """The official history with the last answer replaced by a self-contained one."""
    a = t["prev"][-1][1]
    i = t["h"].rfind(a)
    return t["h"][:i] + new + t["h"][i + len(a):] if i >= 0 else t["h"]

def with_hist(q, h):
    return q + (f"\n\nConversation History:\n{h}" if h else "")

def pairs(prev):
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in prev)

def conv(t):
    return f"Conversation so far:\n{t['h'] or '(none)'}\n\nLatest question: {t['q']}"

P_STEPBACK = ("{c}\n\nWrite one more general step-back question about the underlying concept or principle "
              "that must be understood to answer the latest question. Output only that question.")
P_SUBQ = ("{c}\n\nBreak the latest question into 2-3 simple sub-questions that are self-contained search "
          "queries (resolve every reference to the conversation). Output one per line, nothing else.")
P_RE = ("In a conversation, the user asked: {q}\n\nThe assistant answered:\n{a}\n\nRewrite this last answer so "
        "it is fully self-contained: replace pronouns and references with what they mean. Output only the rewritten answer.")
P_PASSAGE = ("{c}\n\nWrite a short encyclopedic passage (60-100 words) that answers the latest question. "
             "Output only the passage.")
P_REFINE = ("{c}\n\nA search engine returned these passages; some may be irrelevant:\n{docs}\n\nUsing the useful "
            "information in them and your own knowledge, write a correct encyclopedic passage (60-100 words) that "
            "answers the latest question. Output only the passage.")

def keywords(passage, base, n=N_KW):
    """Words of the passage whose stems (Lucene analyzer) are not in the query; top-n by
    frequency, ties broken by first occurrence."""
    have = set(AN.analyze(base))
    cnt, form = {}, {}
    for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", passage):
        st = AN.analyze(w)
        if len(st) != 1 or st[0] in have:
            continue
        cnt[st[0]] = cnt.get(st[0], 0) + 1
        form.setdefault(st[0], w.lower())
    return " ".join(form[s] for s in sorted(cnt, key=lambda s: -cnt[s])[:n])

def rrf(runs, k=60):
    s = {}
    for run in runs:
        for r, (d, _) in enumerate(sorted(run.items(), key=lambda x: -x[1]), 1):
            s[d] = s.get(d, 0) + 1 / (k + r)
    return dict(sorted(s.items(), key=lambda x: -x[1])[:TOPK])
'''),
("code", r'''
# ── 8. Variants: texts → Drive/reteco/queries, BM25 → Drive/reteco/runs ────
# queries/ and runs/bm25_* do not depend on the model name, so mixing two models is not allowed
_mark = f"{QUERIES}/llm.txt"
if os.path.exists(_mark):
    assert open(_mark).read().strip() == LLM_MODEL, (
        f"variants were already computed with model {open(_mark).read().strip()}: set it as LLM_MODEL or remove the old folders")
else:
    open(_mark, "w").write(LLM_MODEL)
TEXTS = {}      # (variant, split) → {key: [texts]}; multiple texts are fused with RRF

def save_variant(name, split, texts):
    """texts: {key: [text, ...]}. queries/ gets a single text (concatenation) for the dense
    notebooks; BM25 searches with each text separately and fuses the results with RRF."""
    TEXTS[(name, split)] = texts
    d = f"{QUERIES}/{name}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{split}.jsonl.tmp", "w", encoding="utf-8") as f:
        for (dom, qid), ts in texts.items():
            f.write(json.dumps({"domain": dom, "topic_id": qid, "text": "\n\n".join(ts)}, ensure_ascii=False) + "\n")
    os.replace(f"{d}/{split}.jsonl.tmp", f"{d}/{split}.jsonl")
    json.dump({"llm": LLM_MODEL, "notebook": "query.ipynb"}, open(f"{d}/config.json", "w"))

def run_variant(name, split):
    rd = f"{RUNS}/bm25_{name}_{split}"
    texts = TEXTS[(name, split)]
    for dom in DOMAINS:
        path = f"{rd}/{dom}/run.trec"
        if os.path.exists(path):
            continue
        items = [(qid, ts) for (d, qid), ts in texts.items() if d == dom]
        n = max(len(ts) for _, ts in items)
        runs = [search(dom, [q for q, ts in items if len(ts) > i], [ts[i] for _, ts in items if len(ts) > i])
                for i in range(n)]
        run = {q: (rrf([r[q] for r in runs if q in r]) if len(ts) > 1 else runs[0][q]) for q, ts in items}
        write_run(path, run, f"bm25_{name}")
    json.dump({"variant": name, "split": split, "llm": LLM_MODEL, "bm25": "kit OfficialBM25 k1=0.9 b=0.4",
               "topk": TOPK, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{rd}/config.json", "w"), indent=2)
    return rd

def top_texts(rd, split, turns, k=FEEDBACK_K):
    """Texts of the run's top-k for each turn: what the LLM sees in QExpand (no doc_id)."""
    out = {}
    for dom in DOMAINS:
        ids, docs = load_corpus(dom)
        text = dict(zip(ids, docs))
        run = read_run(f"{rd}/{dom}/run.trec")
        for t in turns:
            if t["key"][0] == dom:
                top = sorted(run.get(t["key"][1], {}).items(), key=lambda x: -x[1])[:k]
                out[t["key"]] = [" ".join(text[d].split()[:DOC_WORDS]) for d, _ in top]
    return out

def refine_prompts(turns, tops):
    return [(t["key"], P_REFINE.format(c=conv(t), docs="\n\n".join(f"[{i}] {d}" for i, d in enumerate(tops[t["key"]], 1))))
            for t in turns]

for split in SPLITS:
    turns = load_turns(split)
    base = {t["key"]: with_hist(t["q"], t["h"]) for t in turns}
    later = [t for t in turns if t["prev"]]
    sb = ask("stepback", split, [(t["key"], P_STEPBACK.format(c=conv(t))) for t in turns])
    sq = ask("subq", split, [(t["key"], P_SUBQ.format(c=conv(t))) for t in turns])
    re_ = ask("re", split, [(t["key"], P_RE.format(q=t["prev"][-1][0], a=t["prev"][-1][1])) for t in later])
    ps = ask("passage", split, [(t["key"], P_PASSAGE.format(c=conv(t))) for t in turns])
    subqs = lambda k: [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip() for l in sq[k].splitlines() if l.strip()][:3]
    save_variant("win1", split, {t["key"]: [with_hist(t["q"], pairs(t["prev"][-1:]))] for t in turns})
    save_variant("win2", split, {t["key"]: [with_hist(t["q"], pairs(t["prev"][-2:]))] for t in turns})
    save_variant("stepback", split, {k: [f"{b}\n\n{sb[k]}"] for k, b in base.items()})
    save_variant("subq", split, {k: [b] + subqs(k) for k, b in base.items()})
    save_variant("re", split, {t["key"]: [with_hist(t["q"], re_hist(t, re_[t["key"]])) if t["prev"] else t["q"]]
                               for t in turns})
    save_variant("q2d_open", split, {k: [f"{b}\n\n{ps[k]}"] for k, b in base.items()})
    save_variant("kw", split, {k: [f"{b}\n\n{keywords(ps[k], b)}"] for k, b in base.items()})
    for name in ["win1", "win2", "stepback", "subq", "re", "q2d_open", "kw"]:
        run_variant(name, split)
    print(f"{split}: single-pass variants done")
    # QExpand: passage → search → LLM sees the top-5 and rewrites → search → once more
    prev_run = f"{RUNS}/bm25_q2d_open_{split}"
    for r in (1, 2):
        ex = ask(f"refine{r}", split, refine_prompts(turns, top_texts(prev_run, split, turns)))
        save_variant(f"qexp{r}", split, {k: [f"{b}\n\n{ex[k]}"] for k, b in base.items()})
        prev_run = run_variant(f"qexp{r}", split)
    print(f"{split}: QExpand done")
'''),
("code", r'''
# ── 9. Result: macro nDCG@10 over domains ──────────────────────────────────
REF = {"bm25_hist": BM25_MACRO, "bm25_q2d (Claude)": {"dev": 0.5463, "train": 0.5584}}
names = ["win1", "win2", "stepback", "subq", "re", "q2d_open", "kw", "qexp1", "qexp2"]
print(f"{'variant':<22}" + "".join(f"{s:>9}" for s in SPLITS))
for n, v in REF.items():
    print(f"{n:<22}" + "".join(f"{v[s]:>9.4f}" for s in SPLITS))
for n in names:
    row = []
    for s in SPLITS:
        rd = f"{RUNS}/bm25_{n}_{s}"
        ok = all(os.path.exists(f"{rd}/{d}/run.trec") for d in DOMAINS)
        row.append(sum(ndcg10(read_run(f"{rd}/{d}/run.trec"), d, s) for d in DOMAINS) / len(DOMAINS) if ok else float("nan"))
    print(f"{n:<22}" + "".join(f"{x:>9.4f}" for x in row))
print("\nSelect on train. Next: copy", RUNS, "and", QUERIES,
      "from Drive to data/drive/reteco/runs/ and data/drive/reteco/queries/ of the repo.")
'''),
    ]
    return nb(cells, "A100")

# ========================================================= llm_rerank ========
def llm_rerank_nb():
    cells = [
("markdown", r"""
# RETECO · listwise LLM reranking (RankGPT / ReasonRank)

The model sees 20 candidates at once (texts only, no `doc_id`) and outputs their
order `[3] > [1] > …`. Sliding window of 20 with step 10, from bottom to top, as in
RankGPT and ReasonRank; with `TOPK = 20` this is a single window.

Models: on A100 and L4, `liuwenhan/reasonrank-7B` (trained for exactly this task; it
first reasons in `<think>`, then answers in `<answer>`); on T4, Qwen3-4B-Instruct-2507
with the RankGPT prompt. The ReasonRank prompt is copied verbatim from their code
(`listwise_prompt_r1.toml`, `add_prefix_prompt`, `add_post_prompt`).

**How to run.** Runtime **L4** (or A100) → "Run all". If it stops, run it
again; finished domains are skipped.

**How long.** 858 turns × 1 window: reasonrank-7B on A100 ≈ 10–20 min;
`TOPK = 100` (9 windows) takes about an hour.

**Output.** `reteco/runs/llmrr_<model>_<first stage>_<query>/`.
Mixing with the first stage at 0.6 / 0.4 is done in `analysis.ipynb`.

**Next.** Copy `reteco/runs/` from Drive to `data/drive/reteco/runs/` of the repo.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
MODEL = "auto"        # A100/L4 → liuwenhan/reasonrank-7B, T4 → Qwen/Qwen3-4B-Instruct-2507
FIRST_STAGE = "bm25_fuse3_dev"   # folder in Drive/reteco/runs
QUERY_SET = "qd"      # what the model sees as the query: "qd" — standalone question rewritten by Claude,
                      # "hist" — turn + history, as in the organizers' setup; or a folder in queries
SPLIT = "dev"
TOPK, WINDOW, STEP = 20, 20, 10
PASSAGE_TOKENS = 300  # each document is truncated to this many tokens
MAXLEN = 16384
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Google Drive
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. Model and prompt (ReasonRank: rank_GPT_reasoning; others: rank_GPT) ──
from vllm import SamplingParams
if MODEL == "auto":
    MODEL = "liuwenhan/reasonrank-7B" if MID else "Qwen/Qwen3-4B-Instruct-2507"
REASON = "reasonrank" in MODEL.lower()
if globals().get("_LOADED") != MODEL:
    assert "_LOADED" not in globals(), "a different model is already loaded: Runtime → Restart session"
    llm = load_llm(MODEL, MAXLEN); _LOADED = MODEL
tok = llm.get_tokenizer()
SP = SamplingParams(temperature=0.0, max_tokens=3072 + 100 if REASON else 300)   # 3072 is ReasonRank's reasoning_maxlen
SYS = ("You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query."
       + (" Given a query and a passage list, you first thinks about the reasoning process in the mind and then "
          "provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed "
          "within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process "
          "here </think> <answer> answer here </answer>." if REASON else ""))
num_fix = lambda s: re.sub(r"\[(\d+)\]", r"(\1)", s)     # a literal [n] in the text would be confused with candidate numbers

def passage(text):
    return num_fix(tok.convert_tokens_to_string(tok.tokenize(text.strip())[:PASSAGE_TOKENS]))

def prompt(query, docs):
    q, n = num_fix(query).strip(), len(docs)
    body = (f"I will provide you with {n} passages, each indicated by a numerical identifier []. "
            f"Rank the passages based on their relevance to the search query: {q}.\n\n")
    body += "".join(f"[{i}] {d}\n" for i, d in enumerate(docs, 1))
    body += (f"Search Query: {q}.\nRank the {n} passages above based on their relevance to the search query. "
             "All the passages should be included and listed using identifiers, in descending order of relevance. ")
    body += ("The format of the answer should be [] > [], e.g., [2] > [1]." if REASON else
             "The output format should be [] > [], e.g., [2] > [1]. Only respond with the ranking results, "
             "do not say any word or explain.")
    return chat(tok, [{"role": "system", "content": SYS}, {"role": "user", "content": body}])

def permutation(text, n):
    """Order 0..n-1 from the response: first the numbers it names, without repeats, then the
    omitted ones in their original order (as in rank_llm's _clean_response)."""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    if REASON and m is None and "</think>" not in text:
        return list(range(n)), 0          # cut off inside <think>: keep the order unchanged
    body = m.group(1) if m else text.split("</think>")[-1]
    seen = []
    for x in re.findall(r"\[(\d+)\]", body) or re.findall(r"\d+", body):
        i = int(x) - 1
        if 0 <= i < n and i not in seen:
            seen.append(i)
    return seen + [i for i in range(n) if i not in seen], len(seen)
print(MODEL, "· reasoning" if REASON else "· RankGPT")
'''),
("code", r'''
# ── 6. Rerank per domain → Drive ────────────────────────────────────────────
src = f"{RUNS}/{FIRST_STAGE}"
assert os.path.isdir(src), f"first-stage run not found: {src}"
dst = f"{RUNS}/llmrr_{MODEL.split('/')[-1]}_{FIRST_STAGE}_{QUERY_SET}_k{TOPK}w{WINDOW}s{STEP}"
stats = {"windows": 0, "empty": 0, "partial": 0, "cut": 0}
rows = []
for dom in DOMAINS:
    path = f"{dst}/{dom}/run.trec"
    first = read_run(f"{src}/{dom}/run.trec")
    if not os.path.exists(path):
        ids, texts = load_corpus(dom)
        text = dict(zip(ids, texts))
        qids, qtexts = topics(dom, SPLIT, QUERY_SET)
        qtext = dict(zip(qids, qtexts))
        order = {q: [d for d, _ in sorted(first.get(q, {}).items(), key=lambda x: -x[1])] for q in qids}
        assert any(order.values()), f"{FIRST_STAGE} does not cover the {SPLIT} turns in {dom}"
        t0 = time.time()
        end = min(TOPK, max(len(v) for v in order.values()))
        start = end - WINDOW
        while end > 0 and start + STEP != 0:        # windows from bottom to top, as in rank_llm's sliding_windows
            start = max(start, 0)
            live = [q for q in qids if len(order[q][start:end]) > 1]
            outs = llm.generate([prompt(qtext[q], [passage(text[d]) for d in order[q][start:end]]) for q in live],
                                SP, use_tqdm=False)
            for q, o in zip(live, outs):
                win = order[q][start:end]
                perm, named = permutation(o.outputs[0].text, len(win))
                order[q][start:end] = [win[i] for i in perm]
                stats["windows"] += 1
                stats["empty"] += named == 0
                stats["partial"] += 0 < named < len(win)
                stats["cut"] += o.outputs[0].finish_reason == "length"
            end, start = end - STEP, start - STEP
        run = {q: {d: (len(v) - r) / len(v) for r, d in enumerate(v)} for q, v in order.items() if v}
        write_run(path, run, "llmrr")
        print(f"{dom}: {len(qids)} turns in {(time.time() - t0) / 60:.1f} min")
    a, b = ndcg10(first, dom, SPLIT), ndcg10(read_run(path), dom, SPLIT)
    rows.append((a, b))
    print(f"  {dom:<20} first stage {a:.4f} → LLM rerank {b:.4f}  ({b - a:+.4f})")
json.dump({"model": MODEL, "first_stage": FIRST_STAGE, "query_set": QUERY_SET, "split": SPLIT, "topk": TOPK,
           "window": WINDOW, "step": STEP, "passage_tokens": PASSAGE_TOKENS, "prompt": "rank_GPT_reasoning" if REASON
           else "rank_GPT", "stats": stats, "gpu": torch.cuda.get_device_name(0),
           "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{dst}/config.json", "w"), ensure_ascii=False, indent=2)
A, B = (sum(r[i] for r in rows) / len(rows) for i in (0, 1))
print(f"\nMACRO  first stage {A:.4f} → LLM rerank {B:.4f}  ({B - A:+.4f})")
print("windows:", stats["windows"], "· unparsed:", stats["empty"], "· partial:", stats["partial"],
      "· truncated by length:", stats["cut"], "(these counts cover this run only)")
'''),
    ]
    return nb(cells, "A100")

# =========================================================== analysis ========
def analysis_nb():
    cells = [
("markdown", r"""
# RETECO · all tables (CPU, no GPU needed)

Collects everything on Drive in `reteco/runs`, `reteco/gen`, `reteco/judge`:
1. nDCG@10 and R@100 for all runs, dev and train
2. BM25 × dense hybrids (score-based and RRF), weight selected on train
3. mixing rerankers with the first stage at 0.6 / 0.4 (as in DIVER)
4. bootstrap intervals for the difference from BM25 + history (dev)
5. breakdown by turn depth T1…T5+ and by domain
6. 2a submission from the best run on train + check with the kit's `format_checker.py`
7. generation: 2b vs 2c, 2c by qrels label, refusals, unsupported claims
8. selective answering in 2c: risk–coverage by reader confidence (risk: the answer is
   not in the context according to the content label in `reteco/suff`, or judge correctness < 4), AURC

**How to run.** Any runtime (CPU is enough) → "Run all". Takes minutes.

**Output.** `reteco/analysis/*.csv` — tables for the paper;
`reteco/runs/hyb_*`, `rrf_*`, `mix_*` — new runs; `reteco/submission/2a/`.

Nothing is tuned on dev: hybrid weights and the submitted run are selected on train.

**Next.** Copy `reteco/analysis/` and `reteco/runs/` from Drive to `data/drive/reteco/analysis/`
and `data/drive/reteco/runs/` of the repo.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
BASE = "bm25_hist"        # reference run for the bootstrap
BM25_PART = "bm25_fuse3"  # BM25 half of the hybrids
MIX = (0.6, 0.4)          # reranker / first stage (DIVER), not tuned
SUBMIT = "auto"           # run family for the submission; auto — best on train
SUBMIT_READER = "OCC-RAG-1.7B"   # whose answers are submitted for 2b and 2c
SUFF_JUDGE = "phi-4"      # content-based label: reteco/suff/<judge>_<job>_dev.jsonl (suff.ipynb)
N_BOOT = 1000
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Google Drive
'''),
("code", r'''
# ── 2. Drive and packages (no GPU needed) ───────────────────────────────────
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
# JAX (pulled in by bm25s) preallocates 75% of GPU memory by default, after which vLLM fails to start
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from google.colab import drive
drive.mount("/content/drive")
!pip -q install pytrec-eval-terrier
!git clone -q https://github.com/DataScienceUIBK/RETECO /content/reteco_kit 2>/dev/null || true
'''),
("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. All runs: per-turn nDCG@10, macro over domains, R@100 ────────────────
import csv, subprocess, sys
from collections import defaultdict
OUTA = f"{DRIVE}/analysis"
os.makedirs(OUTA, exist_ok=True)
QRELS = {}
for s in ("dev", "train"):
    for d in DOMAINS:
        q = {}
        for ln in open(f"{DATA}/{d}/qrels_{s}.txt"):
            p = ln.split()
            if len(p) == 4:
                q.setdefault(p[0], {})[p[2]] = int(p[3])
        QRELS[(d, s)] = q

def split_of(name):
    try:
        return json.load(open(f"{RUNS}/{name}/config.json")).get("split") or re.search(r"_(dev|train)(?=_|$)", name).group(1)
    except Exception:
        m = re.search(r"_(dev|train)(?=_|$)", name)
        return m.group(1) if m else None

def family(name):
    return re.sub(r"_(dev|train)(?=_|$)", "", name)

def per_query(run, d, s):
    """{qid: nDCG@10} and {qid: R@100} over all qrels turns; a turn missing from the run scores 0."""
    ev = pytrec_eval.RelevanceEvaluator(QRELS[(d, s)], {"ndcg_cut.10", "recall.100"}).evaluate(run)
    return ({q: ev.get(q, {}).get("ndcg_cut_10", 0.0) for q in QRELS[(d, s)]},
            {q: ev.get(q, {}).get("recall_100", 0.0) for q in QRELS[(d, s)]})

RUN, NDCG, REC = {}, {}, {}
for rd in sorted(glob.glob(f"{RUNS}/*/")):
    name = os.path.basename(rd.rstrip("/"))
    s = split_of(name)
    if s not in ("dev", "train") or not all(os.path.exists(f"{rd}{d}/run.trec") for d in DOMAINS):
        continue
    RUN[name] = {d: read_run(f"{rd}{d}/run.trec") for d in DOMAINS}
    NDCG[name], REC[name] = {}, {}
    for d in DOMAINS:
        NDCG[name][d], REC[name][d] = per_query(RUN[name][d], d, s)

macro = lambda m, name: sum(sum(v.values()) / len(v) for v in m[name].values()) / len(DOMAINS)
FAM = defaultdict(dict)
for name in RUN:
    FAM[family(name)][split_of(name)] = name

def table():
    rows = []
    for f, sp in FAM.items():
        r = {"run": f}
        for s in ("dev", "train"):
            r[f"ndcg_{s}"] = round(macro(NDCG, sp[s]), 4) if s in sp else None
            r[f"r100_{s}"] = round(macro(REC, sp[s]), 4) if s in sp else None
        rows.append(r)
    return sorted(rows, key=lambda r: (-(r["ndcg_train"] or -1), -(r["ndcg_dev"] or -1)))

SHORT = {"hanhainebula_reason-embed-qwen3-4b-0928": "RE", "AQ-MedAI_Diver-Retriever-4B": "DIV"}
def short(name):
    """Run name for display: long model names → RE / DIV (the CSV keeps full names)."""
    for k, v in SHORT.items():
        name = name.replace(k, v)
    return name

def show(rows, cols, w=46):
    print(f"{'run':<{w}}" + "".join(f"{c:>12}" for c in cols))
    for r in rows:
        print(f"{short(r['run'])[:w - 1]:<{w}}" + "".join(f"{r[c]:>12.4f}" if isinstance(r[c], float) else f"{'—':>12}" for c in cols))

def save(rows, name):
    with open(f"{OUTA}/{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

rows = table()
show(rows, ["ndcg_dev", "ndcg_train", "r100_dev"])
save(rows, "runs")
print(f"\n{len(RUN)} runs; sorted by train. BM25 + history: dev 0.4379, train 0.4539")
'''),
("code", r'''
# ── 6. BM25 × dense hybrids (weight on train) and 0.6/0.4 reranker mixing ──
def minmax_fuse(parts):
    """parts: [(weight, {doc: score})] → {doc: Σ weight·minmax(score)}; a doc missing from a list gets 0 (as in fuse.py)."""
    s = defaultdict(float)
    for w, r in parts:
        if r:
            lo, hi = min(r.values()), max(r.values())
            for d, v in r.items():
                s[d] += w * ((v - lo) / (hi - lo) if hi > lo else 1.0)
    return dict(sorted(s.items(), key=lambda x: -x[1])[:100])

def rrf_fuse(parts, k=60):
    s = defaultdict(float)
    for w, r in parts:
        for i, (d, _) in enumerate(sorted(r.items(), key=lambda x: -x[1]), 1):
            s[d] += w / (k + i)
    return dict(sorted(s.items(), key=lambda x: -x[1])[:100])

def fuse_runs(a, b, fn, wa, wb, s):
    return {d: {q: fn([(wa, RUN[a][d].get(q, {})), (wb, RUN[b][d].get(q, {}))])
                for q in QRELS[(d, s)]} for d in DOMAINS}

def add_run(name, s, run, cfg):
    for d in DOMAINS:
        write_run(f"{RUNS}/{name}/{d}/run.trec", run[d], name[:20])
    json.dump({**cfg, "split": s, "notebook": "analysis.ipynb"}, open(f"{RUNS}/{name}/config.json", "w"), indent=2)
    RUN[name] = run
    NDCG[name], REC[name] = {}, {}
    for d in DOMAINS:
        NDCG[name][d], REC[name][d] = per_query(run[d], d, s)
    FAM[family(name)][s] = name

hyb = []
for f, sp in list(FAM.items()):
    if not f.startswith("dense_") or "train" not in sp or "dev" not in sp or BM25_PART not in FAM:
        continue
    bm = FAM[BM25_PART]
    grid = {}
    for w in [i / 10 for i in range(11)]:
        r = fuse_runs(sp["train"], bm["train"], minmax_fuse, w, 1 - w, "train")
        grid[w] = np.mean([np.mean(list(per_query(r[d], d, "train")[0].values())) for d in DOMAINS])
    w = max(grid, key=grid.get)
    for s in ("train", "dev"):
        add_run(f"hyb_{f}_{s}", s, fuse_runs(sp[s], bm[s], minmax_fuse, w, 1 - w, s),
                {"dense": sp[s], "bm25": bm[s], "mode": "score", "w_dense": w})
        add_run(f"rrf_{f}_{s}", s, fuse_runs(sp[s], bm[s], rrf_fuse, 1, 1, s),
                {"dense": sp[s], "bm25": bm[s], "mode": "rrf"})
    hyb.append((f, w))
    print(f"hybrid {f} × {BM25_PART}: dense weight on train {w:.1f}")

for name in list(RUN):
    try:
        cfg = json.load(open(f"{RUNS}/{name}/config.json"))
    except Exception:
        continue
    fs = cfg.get("first_stage")
    if not fs or fs not in RUN or name.startswith("mix_"):
        continue
    s = split_of(name)
    add_run(f"mix_{name}", s, fuse_runs(name, fs, minmax_fuse, MIX[0], MIX[1], s),
            {"rerank": name, "first_stage": fs, "weights": MIX})
    print(f"mixing {name} {MIX[0]}/{MIX[1]} with {fs}: {macro(NDCG, name):.4f} → {macro(NDCG, f'mix_{name}'):.4f}")
rows = table()
save(rows, "runs")
show([r for r in rows if r["run"].startswith(("hyb_", "rrf_", "mix_"))], ["ndcg_dev", "ndcg_train"])
'''),
("code", r'''
# ── 7. Bootstrap: difference from BASE on dev, 95% interval and share of "not better" ──
rng = np.random.default_rng(0)
base = FAM[BASE]["dev"]
res = []
for f, sp in FAM.items():
    if "dev" not in sp or sp["dev"] == base:
        continue
    diffs = np.zeros(N_BOOT)
    for d in DOMAINS:
        qs = sorted(QRELS[(d, "dev")])
        a = np.array([NDCG[sp["dev"]][d][q] for q in qs]); b = np.array([NDCG[base][d][q] for q in qs])
        idx = rng.integers(0, len(qs), size=(N_BOOT, len(qs)))
        diffs += (a[idx] - b[idx]).mean(1) / len(DOMAINS)
    res.append({"run": f, "delta": round(macro(NDCG, sp["dev"]) - macro(NDCG, base), 4),
                "ci_lo": round(float(np.percentile(diffs, 2.5)), 4), "ci_hi": round(float(np.percentile(diffs, 97.5)), 4),
                "p_le0": round(float((diffs <= 0).mean()), 4)})
res.sort(key=lambda r: -r["delta"])
save(res, "bootstrap_dev")
print(f"{'run':<46}{'Δ':>9}{'95% CI':>20}{'p(Δ≤0)':>9}")
for r in res:
    print(f"{short(r['run'])[:45]:<46}{r['delta']:>+9.4f}   [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{r['p_le0']:>9.3f}")
'''),
("code", r'''
# ── 8. Turn depth T1…T5+ and domains (dev) for BASE and the top five on train ──
top = [r["run"] for r in table() if r["ndcg_train"] is not None and "dev" in FAM[r["run"]]][:5]
show_runs = [BASE] + [f for f in top if f != BASE]
depth = lambda q: min(int(re.search(r"_turn_(\d+)$", q).group(1)), 5)
rows_t, rows_d = [], []
for f in show_runs:
    name = FAM[f]["dev"]
    by = defaultdict(list)
    for d in DOMAINS:
        for q, v in NDCG[name][d].items():
            by[depth(q)].append(v)
    rows_t.append({"run": f, **{("T5+" if t == 5 else f"T{t}"): round(float(np.mean(by[t])), 4) for t in sorted(by)},
                   **{f"n_T{t}": len(by[t]) for t in sorted(by)}})
    rows_d.append({"run": f, **{d: round(float(np.mean(list(NDCG[name][d].values()))), 4) for d in DOMAINS}})
save(rows_t, "depth_dev"); save(rows_d, "domains_dev")
cols_t = [c for c in rows_t[0] if c.startswith("T")]
show(rows_t, cols_t)
print()
show(rows_d, DOMAINS[:6]); print(); show(rows_d, DOMAINS[6:])
'''),
("code", r'''
# ── 9. 2a submission: best family on train, its dev run, kit format check ───
if SUBMIT == "auto":
    SUBMIT = next(r["run"] for r in table() if r["ndcg_train"] is not None and "dev" in FAM[r["run"]])
name = FAM[SUBMIT]["dev"]
out = f"{DRIVE}/submission/2a"
os.makedirs(out, exist_ok=True)
ok = True
for d in DOMAINS:
    path = f"{out}/{d}.trec"
    write_run(path, RUN[name][d], "reteco2026")
    r = subprocess.run([sys.executable, "/content/reteco_kit/starter_kit/format_checker.py", path,
                        "--qrels", f"{DATA}/{d}/qrels_dev.txt", "--corpus", f"{DATA}/{d}/documents.jsonl",
                        "--doc-key", "doc_id"], capture_output=True, text=True)
    ok &= r.returncode == 0
    msg = ((r.stdout + r.stderr).strip().splitlines() or ["—"])[-1]
    print(f"  {d:<20} {'OK' if r.returncode == 0 else 'ERROR'}  {msg[:90]}")
print(f"\nsubmission from {name} (dev nDCG {macro(NDCG, name):.4f}) → {out}:", "format OK" if ok else "ERRORS FOUND")
'''),
("code", r'''
# ── 10. Generation: 2b vs 2c, qrels label, refusals, unsupported claims ─────
gen = sorted(glob.glob(f"{DRIVE}/gen/*.jsonl"))
if not gen:
    print("no reteco/gen — run generate.ipynb first")
else:
    from sklearn.metrics import roc_auc_score
    J = {}
    for p in glob.glob(f"{DRIVE}/judge/*/*.jsonl"):
        for r in map(json.loads, open(p, encoding="utf-8")):
            if r.get("ok"):
                J[(p.split("/")[-2], os.path.basename(p), r["domain"], r["topic_id"])] = r
    judges = sorted({k[0] for k in J})
    rows = []
    for p in gen:
        rs = [json.loads(l) for l in open(p, encoding="utf-8")]
        for suff, g in strata(rs):
            if not g:
                continue
            row = {"file": os.path.basename(p)[:-6], "suff": suff, "n": len(g),
                   "refused": round(sum(r["status"].startswith("UN") for r in g) / len(g), 3),
                   "p_ans": round(float(np.mean([r["p_answerable"] for r in g if r["p_answerable"] is not None] or [np.nan])), 3)}
            ab = [r for r in rs if r["p_answerable"] is not None and (r["suff"] == "all" or r in g)]
            ys = [r["suff"] == "all" for r in ab]
            row["auroc"] = (round(roc_auc_score(ys, [r["p_answerable"] for r in ab]), 3)
                            if suff in ("none", "clean") and len(set(ys)) == 2 else None)   # stratum keys from strata()
            for jn in judges:
                js = [J[k] for r in g if (k := (jn, os.path.basename(p), r["domain"], r["topic_id"])) in J]
                if js:
                    row[f"{jn}:correct"] = round(float(np.mean([x["correctness"] for x in js])), 3)
                    row[f"{jn}:faithful"] = round(float(np.mean([x["faithfulness"] for x in js])), 3)
                    row[f"{jn}:unsupported"] = round(float(np.mean([x["unsupported_claims"] for x in js])), 3)
            rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    rows = [{k: r.get(k) for k in keys} for r in rows]
    save(rows, "generation")
    print(f"{'file':<42}{'suff':>6}{'n':>5}{'refuse':>7}{'p_ans':>7}{'AUROC':>7}" + "".join(f"{j[:10] + ':corr':>16}" for j in judges))
    for r in rows:
        print(f"{r['file'][:41]:<42}{r['suff']:>6}{r['n']:>5}{r['refused']:>7.2f}{r['p_ans']:>7.2f}"
              f"{(r['auroc'] if r['auroc'] is not None else float('nan')):>7.3f}"
              + "".join(f"{r.get(f'{j}:correct') or float('nan'):>16.2f}" for j in judges))
print("\nAll tables:", OUTA)
'''),
("code", r'''
# ── 11. 2b and 2c submissions (format rehearsal on dev): JSONL turn_id + answer ──
# 2c: answers from the top 5 of the first stage that is best on train, plus that stage's ranking.
REFUSAL = "I cannot answer this from the available documents."
def answers(path):
    out = defaultdict(dict)
    for r in map(json.loads, open(path, encoding="utf-8")):
        a = r["answer"] or (REFUSAL if r["status"].startswith("UN") else "")
        out[r["domain"]][r["topic_id"]] = a
    return out

def write_answers(d, ans, kind):
    os.makedirs(f"{DRIVE}/submission/{kind}", exist_ok=True)
    with open(f"{DRIVE}/submission/{kind}/{d}.jsonl", "w", encoding="utf-8") as f:
        for q in sorted(ans):
            f.write(json.dumps({"turn_id": q, "answer": ans[q]}, ensure_ascii=False) + "\n")

g2b = f"{DRIVE}/gen/{SUBMIT_READER}_2b_dev.jsonl"
c2c = {re.search(r"_2c-(.+)_dev\.jsonl$", p).group(1): p
       for p in glob.glob(f"{DRIVE}/gen/{SUBMIT_READER}_2c-*_dev.jsonl")}
if not os.path.exists(g2b) or not c2c:
    print("no answers from", SUBMIT_READER, "in reteco/gen — run generate.ipynb first")
else:
    score = lambda fs: macro(NDCG, FAM[fs]["train"]) if "train" in FAM.get(fs, {}) else -1
    fs = max(c2c, key=score)
    a2b, a2c = answers(g2b), answers(c2c[fs])
    for d in DOMAINS:
        need = set(QRELS[(d, "dev")])
        write_answers(d, a2b[d], "2b")
        write_answers(d, a2c[d], "2c")
        write_run(f"{DRIVE}/submission/2c/{d}.trec", RUN[FAM[fs]["dev"]][d], "reteco2026")
        miss = [len(need - set(x[d])) for x in (a2b, a2c)]
        print(f"  {d:<20} 2b {len(a2b[d])}/{len(need)}  2c {len(a2c[d])}/{len(need)}"
              + ("  MISSING answers" if any(miss) else ""))
    print(f"\n2b and 2c from {SUBMIT_READER} answers; 2c on first stage {fs} → {DRIVE}/submission/")
'''),
("code", r'''
# ── 12. Selective answering in 2c: risk–coverage by reader confidence (dev) ──
# The reader answers only the k most confident turns (coverage k/N) and refuses the rest.
# Risk among answered turns: "not in context" — per the suff/ label the answer is not in the
# context (label = no; partial is not an error); "<judge>:corr<4" — judge correctness < 4.
# Confidence: p_answerable (a missing value ranks lowest); for OCC also the verdict
# ANSWERABLE = 1 / otherwise 0. Ties are averaged over random orderings. AURC is the mean
# risk over k = 1…N; @100% is the risk without refusals (a random order would give AURC ≈ @100%).
def risk_curve(conf, err):
    """→ risk at k = 1…N: error rate among the k most confident turns (expectation within ties)."""
    o = np.argsort(-conf, kind="stable")
    c, e = conf[o], err[o]
    _, start, cnt = np.unique(-c, return_index=True, return_counts=True)
    cum, done = np.zeros(len(c)), 0.0
    for s, n in zip(start, cnt):
        cum[s:s + n] = done + e[s:s + n].mean() * np.arange(1, n + 1)
        done += e[s:s + n].sum()
    return cum / np.arange(1, len(c) + 1)

rc = []
for p in sorted(glob.glob(f"{DRIVE}/gen/*_2c-*_dev.jsonl")):
    name = os.path.basename(p)
    reader, job = re.match(r"(.+?)_(2c-.+)_dev\.jsonl$", name).groups()
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    sp = f"{DRIVE}/suff/{SUFF_JUDGE}_{job}_dev.jsonl"
    lab = ({(r["domain"], r["topic_id"]): r["label"] for r in map(json.loads, open(sp, encoding="utf-8"))}
           if os.path.exists(sp) else {})
    risks = {"not in context": lambda r: {"no": 1, "yes": 0, "partial": 0}.get(lab.get((r["domain"], r["topic_id"])))}
    for jn in judges:                  # J, judges come from cell 10 (parsed scores only)
        risks[f"{jn}:corr<4"] = lambda r, jn=jn: (lambda x: None if x is None else int(x["correctness"] < 4))(
            J.get((jn, name, r["domain"], r["topic_id"])))
    scores = {"p_answerable": lambda r: -1.0 if r["p_answerable"] is None else r["p_answerable"]}
    if "OCC" in reader:
        scores["verdict"] = lambda r: float(r["status"] == "ANSWERABLE")
    for sn, sf in scores.items():
        for rn, rf in risks.items():
            ce = [(sf(r), e) for r in rs if (e := rf(r)) is not None]
            if len(ce) < 10:
                continue
            conf, err = (np.array(x, float) for x in zip(*ce))
            risk = risk_curve(conf, err)
            at = lambda c: round(float(risk[max(int(np.ceil(c * len(risk))), 1) - 1]), 3)
            rc.append({"reader": reader, "stage": job[3:], "score": sn, "risk": rn, "n": len(risk),
                       "risk_100": at(1.0), "risk_80": at(.8), "risk_50": at(.5), "aurc": round(float(risk.mean()), 4)})
if not rc:
    print("no reteco/gen/*_2c-*_dev.jsonl with suff labels or judge scores")
else:
    save(rc, "risk_coverage")
    print(f"{'reader':<14}{'stage':<18}{'confidence':<14}{'risk':<18}{'n':>5}{'@100%':>7}{'@80%':>7}{'@50%':>7}{'AURC':>8}")
    for r in rc:
        print(f"{r['reader'][:13]:<14}{short(r['stage'])[:17]:<18}{r['score']:<14}{r['risk'][:17]:<18}{r['n']:>5}"
              f"{r['risk_100']:>7.3f}{r['risk_80']:>7.3f}{r['risk_50']:>7.3f}{r['aurc']:>8.4f}")
    print(f"\n@80% / @50% — risk when the reader answers only the 80% / 50% most confident turns. → {OUTA}/risk_coverage.csv")
'''),
    ]
    return nb(cells, "CPU")

# =========================================================== transfer ========
ART_WORKER = r'''
"""OCC answers on synthetic unanswerability (SQuAD 2.0, MuSiQue), with the same prompt
and decoding as gen_worker. Runs as a separate process so that vLLM releases the GPU."""
import json, os, sys
import gen_worker as G


def main(reader_id, cfg_path):
    from vllm import LLM, SamplingParams
    C = json.load(open(cfg_path))
    items = [json.loads(l) for l in open(C["items"], encoding="utf-8")]
    done = set()
    if os.path.exists(C["out"]):
        done = {json.loads(l)["id"] for l in open(C["out"], encoding="utf-8")}
    todo = [it for it in items if it["id"] not in done]
    print(f"synthetic: {len(todo)} to do, {len(done)} done", flush=True)
    if not todo:
        return
    llm = LLM(model=reader_id, dtype=C["dtype"], max_model_len=C["maxlen"], gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    R = G.OCC(tok)
    sp = SamplingParams(temperature=0.0, max_tokens=C["max_new"], stop_token_ids=R.stop,
                        skip_special_tokens=False, spaces_between_special_tokens=False)
    limit = C["maxlen"] - C["max_new"]
    keep, prompts = [], []
    for it in todo:
        p = R.prompt({"history": [], "query": it["query"]}, it["docs"])
        if len(tok(p, add_special_tokens=False).input_ids) <= limit:
            keep.append(it); prompts.append(p)
    with open(C["out"], "a", encoding="utf-8") as f:
        for s in range(0, len(prompts), C["chunk"]):
            outs = llm.generate(prompts[s:s + C["chunk"]], sp, use_tqdm=False)
            for it, o in zip(keep[s:s + C["chunk"]], outs):
                f.write(json.dumps({"id": it["id"], "raw": o.outputs[0].text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {s + len(outs)}/{len(prompts)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''

# EVAL_ONLY on CPU: vllm is not installed (it pulls in its own CUDA build of torch)
TRANSFER_SETUP = LLM_SETUP.replace(
    "!pip -q install vllm openai pytrec-eval-terrier 2>&1 | tail -2",
    'PIP_PKGS = "pytrec-eval-terrier" if globals().get("EVAL_ONLY") else "vllm openai pytrec-eval-terrier"\n'
    "!pip -q install {PIP_PKGS} 2>&1 | tail -2")
assert TRANSFER_SETUP != LLM_SETUP

def transfer_nb():
    cells = [
("markdown", r"""
# RETECO · Transfer of insufficient-context detectors

Do insufficient-context detectors trained on synthetic unanswerability **transfer to
natural unanswerability**, where the answer is missing because retrieval failed (RETECO 2c)?

Detectors, all read at the `<|status_start|>` position after OCC's reasoning trace
(names in the output tables in parentheses):
- **verdict** (`OCC verdict`): what OCC itself wrote (ANSWERABLE / UNANSWERABLE)
- **p_status** (`p_status`): probability of ANSWERABLE at that position, no training
- **trace** (`trace`): logistic regression on the trace length and its mean log-probability
- **probe** (`probe`): logistic regression on the hidden state of a single layer

Training (`train` column): on synthetic data (`synthetic`: SQuAD 2.0 + MuSiQue, train), which
measures transfer; and on RETECO train (`RETECO train`), the in-domain ceiling; `—` = no
training. Evaluation: RETECO dev 2c with the qrels label: 1 if the top-5 contains all the
gold documents that fit, 0 if it contains no gold document.
The layer is selected on a held-out part of the training data; dev is not touched.
The last cell evaluates the same trained detectors under the **content** label (`content`)
from `suff.ipynb` (yes = 1, no = 0) and the **clean** label (`clean`: yes = 1; no = 0 only
if the answer is also absent from the dialogue history), if
`reteco/suff/phi-4_2c-<stage>_dev.jsonl` exists on Drive.

**How to run.** Runtime **A100** or **L4** → **Run all**. If cell 2 asks for a restart,
restart the session and **Run all** again. OCC responses on RETECO are taken from
`reteco/gen` (generate.ipynb); missing ones are generated.
Dev features are stored in one file per first stage: adding a stage to DEV_STAGES does not
recompute the existing ones.

**Detectors only, no GPU.** If all `reteco/transfer/feat_*.npz` already exist → set
`EVAL_ONLY = True` in cell 1, runtime **CPU** → **Run all**: the model is not loaded (only
its tokenizer) and nothing is generated; takes minutes. Stages without features are skipped.

**Cost.** Generation of ≈ 9k OCC responses (A100 ≈ 30 min), then one model pass per
example for the hidden states (≈ 15–30 min); probes take minutes.

**Output.** `reteco/transfer/feat_*.npz`: features;
`reteco/analysis/transfer.csv`: AUROC table (qrels label);
`reteco/analysis/transfer_content.csv`: the same detectors under the qrels, content and
clean labels, per retriever (`stage`) and pooled (stage `total`).

**Next.** Copy `reteco/transfer/` and `reteco/analysis/` from Drive to
`data/drive/reteco/transfer/` and `data/drive/reteco/analysis/` of the repo.
"""),
("code", r'''
# ── 1. Settings ─────────────────────────────────────────────────────────────
READER = "occ-ai/OCC-RAG-1.7B"
N_TRAIN, N_VAL = 400, 100   # per class in each synthetic set: training / layer selection
DEV_STAGES = ["bm25_query", "bm25_hist", "bm25_fuse3",          # same as FIRST_STAGES in generate
              "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"]
TRAIN_STAGES = ["bm25_hist", "bm25_fuse3"]      # there is no bm25_query run on train
LAYERS = [4, 8, 12, 16, 20, 24, 28]             # Qwen3-1.7B has 28 layers
K_CTX, MAX_NEW, MAXLEN, TRIM_WORDS, CHUNK = 5, 1200, 16384, 300, 256   # as in generate.ipynb
SEED = 0
SUFF_JUDGE = "phi-4"        # content label: reteco/suff/<judge>_2c-<stage>_dev.jsonl (suff.ipynb)
EVAL_ONLY = False           # True: detectors only, from existing transfer/feat_*.npz; CPU runtime, model not loaded
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Drive
NEED_GPU = not EVAL_ONLY
'''),
("code", TRANSFER_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", "%%writefile /content/art_worker.py\n" + ART_WORKER.strip("\n")),
("code", r'''
# ── 7. Synthetic unanswerability: SQuAD 2.0 and MuSiQue, balanced classes ───
import random
import pandas as pd
sys.path.insert(0, "/content")
import gen_worker as G
TR = f"{DRIVE}/transfer"
os.makedirs(TR, exist_ok=True)
PQ = "https://huggingface.co/api/datasets/{}/parquet/{}/train/0.parquet"

def truthy(v):
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)

def sample(df, is_ans, make, rng):
    """N_TRAIN + N_VAL examples per class; rows are built only for the sampled ones."""
    out = []
    for cls in (True, False):
        idx = [i for i, v in enumerate(is_ans) if v == cls]
        rng.shuffle(idx)
        for k, i in enumerate(idx[:N_TRAIN + N_VAL]):
            x = make(df.iloc[i])
            x["id"] += "_a" if cls else "_u"     # in MuSiQue an answerable/unanswerable pair shares one id
            out.append({**x, "answerable": cls, "part": "train" if k < N_TRAIN else "val"})
    return out

ITEMS = f"{TR}/art_items.jsonl"
if not os.path.exists(ITEMS):
    rng = random.Random(SEED)
    sq = pd.read_parquet(PQ.format("rajpurkar/squad_v2", "squad_v2"))
    mq = pd.read_parquet(PQ.format("bdsaglam/musique", "default"))
    items = (sample(sq, [len(a["text"]) > 0 for a in sq["answers"]],
                    lambda r: {"id": f"squad_{r['id']}", "source": "squad", "query": r["question"],
                               "docs": [r["context"]]}, rng)
             + sample(mq, [truthy(v) for v in mq["answerable"]],
                      lambda r: {"id": f"musique_{r['id']}", "source": "musique", "query": r["question"],
                                 "docs": [str(p.get("paragraph_text") or "") for p in r["paragraphs"]]}, rng))
    with open(ITEMS + ".tmp", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(ITEMS + ".tmp", ITEMS)
ART = {x["id"]: x for x in map(json.loads, open(ITEMS, encoding="utf-8"))}
print("synthetic examples:", len(ART), "· answerable:", sum(x["answerable"] for x in ART.values()))
'''),
("code", r'''
# ── 8. OCC responses: synthetic + RETECO train/dev (existing ones in gen/ are skipped) ──
def stream(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"})
    for line in p.stdout:
        print(line, end="")
    return p.wait()

def generate_all():
    base = {"drive": DRIVE, "data": DATA, "domains": DOMAINS, "k_ctx": K_CTX, "max_new": MAX_NEW, "maxlen": MAXLEN,
            "trim_words": TRIM_WORDS, "chunk": CHUNK, "dtype": DTYPE}
    for split, stages in (("dev", DEV_STAGES), ("train", TRAIN_STAGES)):
        for fs in stages:
            assert os.path.isdir(f"{RUNS}/{fs}_{split}"), f"missing run {RUNS}/{fs}_{split}"
        json.dump({**base, "splits": [split], "jobs": [f"2c-{fs}" for fs in stages]}, open(f"/content/cfg_{split}.json", "w"))
        assert stream([sys.executable, "/content/gen_worker.py", READER, f"/content/cfg_{split}.json"]) == 0
    json.dump({**base, "items": ITEMS, "out": f"{TR}/art_gen.jsonl"}, open("/content/cfg_art.json", "w"))
    assert stream([sys.executable, "/content/art_worker.py", READER, "/content/cfg_art.json"]) == 0

if EVAL_ONLY:
    print("EVAL_ONLY: generation skipped, features come from existing", f"{TR}/feat_*.npz")
else:
    if "lm" in globals():      # rerun in the same session: the HF model from cell 9 holds the GPU
        del lm
        import gc; gc.collect(); torch.cuda.empty_cache()
    generate_all()
'''),
("code", r'''
# ── 9. Features: one model pass over the prompt + trace up to <|status_start|> ──
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from packaging.version import Version
DK = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
tok = AutoTokenizer.from_pretrained(READER)      # EVAL_ONLY: tokenizer only, to match rows against gen/
R = G.OCC(tok)
if not EVAL_ONLY:
    lm = AutoModelForCausalLM.from_pretrained(READER, **{DK: torch.bfloat16 if DTYPE == "bfloat16" else torch.float16})
    lm = lm.cuda().eval()
    ANS_IDS, UN_IDS = [], []
    for i in range(len(tok)):        # tokens that are verdict prefixes: ANS… vs UN…
        s = tok.decode([i]).strip()
        if s and "UNANSWERABLE".startswith(s):
            UN_IDS.append(i)
        elif s and "ANSWERABLE".startswith(s):
            ANS_IDS.append(i)

@torch.inference_mode()
def features(prompt, raw):
    """→ (hidden states of the LAYERS at the status position, p_answerable, trace length,
    mean trace log-probability), or None if OCC did not reach the status."""
    pre = R.prefix(raw)            # trace up to the verdict itself (marker + newline)
    if pre is None:
        return None
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    t_ids = tok(pre, add_special_tokens=False).input_ids
    if len(p_ids) + len(t_ids) > MAXLEN:
        return None
    ids = torch.tensor([p_ids + t_ids], device="cuda")
    try:
        out = lm(input_ids=ids, output_hidden_states=True, logits_to_keep=len(t_ids) + 1)
    except TypeError:
        out = lm(input_ids=ids, output_hidden_states=True)
    logits = out.logits[0, -(len(t_ids) + 1):].float()
    lp = logits[:-1].log_softmax(-1).gather(1, ids[0, len(p_ids):, None]).squeeze(1)
    pr = logits[-1].softmax(-1)
    a, u = pr[ANS_IDS].sum().item(), pr[UN_IDS].sum().item()
    h = torch.stack([out.hidden_states[l][0, -1] for l in LAYERS]).half().cpu().numpy()
    return h, (a / (a + u) if a + u > 0 else 0.5), len(t_ids), lp.mean().item()

def build(name, rows_fn, every=500):
    """rows_fn() → [(prompt, raw response, label 1/0/-1, meta, group, key)] → Drive/reteco/transfer/feat_<name>.npz.
    An existing file is simply loaded (rows_fn is not called); with EVAL_ONLY and no file, returns None.
    Every `every` rows a chunk is saved to Drive; after an interruption, work resumes from the next chunk."""
    path = f"{TR}/feat_{name}.npz"
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    if EVAL_ONLY:
        print(f"missing {path}: EVAL_ONLY, skipping")
        return None
    rows = rows_fn()
    parts = sorted(glob.glob(f"{TR}/feat_{name}.part*.npz"))
    start = sum(int(np.load(p)["n_rows"]) for p in parts)     # rows already processed
    t0, buf, chunk0 = time.time(), [], start
    for k in tqdm(range(start, len(rows)), desc=name):
        prompt, raw, y, meta, grp, key = rows[k]
        f = features(prompt, raw)
        if f is not None:
            buf.append((f, int(R.parse(raw)[0] == "ANSWERABLE"), y, meta, grp, key))
        if k + 1 - chunk0 == every or k == len(rows) - 1:
            d = {"n_rows": np.array(k + 1 - chunk0)}
            chunk0 = k + 1
            if buf:
                d.update(H=np.stack([b[0][0] for b in buf]), p=np.array([b[0][1] for b in buf]),
                         tlen=np.array([b[0][2] for b in buf]), tlp=np.array([b[0][3] for b in buf]),
                         status=np.array([b[1] for b in buf]), y=np.array([b[2] for b in buf]),
                         meta=np.array([b[3] for b in buf]), grp=np.array([b[4] for b in buf]),
                         key=np.array([b[5] for b in buf]))
            pp = f"{TR}/feat_{name}.part{len(parts):04d}.npz"
            np.savez(pp + ".tmp.npz", **d)
            os.replace(pp + ".tmp.npz", pp)
            parts.append(pp); buf = []
    loaded = [dict(np.load(p, allow_pickle=True)) for p in sorted(set(parts))]
    loaded = [x for x in loaded if "y" in x]
    keys = [k for k in ["H", "p", "tlen", "tlp", "status", "y", "meta", "grp", "key"]
            if all(k in x for x in loaded)]          # chunks written before key was added lack it
    d = {k: np.concatenate([x[k] for x in loaded]) for k in keys}
    np.savez(path + ".tmp.npz", **d)
    os.replace(path + ".tmp.npz", path)
    for p in set(parts):
        os.remove(p)
    print(f"{name}: {len(d['y'])} examples out of {len(rows)} (the rest have no status), {(time.time() - t0) / 60:.1f} min")
    return d

def art_rows(part):
    art_raw = {r["id"]: r["raw"] for r in map(json.loads, open(f"{TR}/art_gen.jsonl", encoding="utf-8"))}
    return [(R.prompt({"history": [], "query": x["query"]}, x["docs"]), art_raw[i], int(x["answerable"]), x["source"], i, i)
            for i, x in ART.items() if x["part"] == part and i in art_raw]

gen_path = lambda split, fs: f"{DRIVE}/gen/{READER.split('/')[-1]}_2c-{fs}_{split}.jsonl"

def reteco_rows(split, stages):
    turns = {(t["domain"], t["topic_id"]): t for t in G.load_turns(DATA, DOMAINS, split)}
    recs = [r for fs in stages for r in map(json.loads, open(gen_path(split, fs), encoding="utf-8"))]
    need = {}
    for r in recs:
        need.setdefault(r["domain"], set()).update(r["context_ids"])
    text = G.corpus_texts(DATA, need)
    rows = []
    for r in recs:
        docs = [text[(r["domain"], d)] for d in r["context_ids"]]
        if r["trimmed"]:
            docs = [" ".join(d.split()[:TRIM_WORDS]) for d in docs]
        y = {"all": 1, "none": 0}.get(r["suff"], -1)
        grp = f"{r['domain']}|{r['topic_id'].rsplit('_turn_', 1)[0]}"     # dialogue: its turns are never split up
        rows.append((R.prompt(turns[(r["domain"], r["topic_id"])], docs), r["raw"], y, r["first_stage"], grp,
                     f"{r['domain']}|{r['topic_id']}"))
    return rows

def with_keys(d, fs, src, split="dev"):
    """Key "domain|topic_id" of every row, used to join with the labels in suff/. Newer files
    store it themselves (key); older files lack it. build walked the records of the stage's
    gen file in file order and dropped those where OCC did not reach the status (R.prefix is
    None) or where the prompt plus trace exceeded MAXLEN. Recovery repeats that pass: records
    without a status are skipped, and each feature row is matched to a record by (dialogue,
    qrels label, verdict, trace length in tokens); on a mismatch the record was dropped for
    length and the next one is tried. If matching fails before the end (the gen file changed
    after build), an error is raised."""
    if "key" in d:
        return d
    recs = [r for r in map(json.loads, open(gen_path(split, fs), encoding="utf-8")) if R.prefix(r["raw"]) is not None]
    sig = [(f"{r['domain']}|{r['topic_id'].rsplit('_turn_', 1)[0]}", {"all": 1, "none": 0}.get(r["suff"], -1),
            int(R.parse(r["raw"])[0] == "ANSWERABLE"), len(tok(R.prefix(r["raw"]), add_special_tokens=False).input_ids))
           for r in recs]
    rows = list(zip(map(str, d["grp"]), map(int, d["y"]), map(int, d["status"]), map(int, d["tlen"])))
    for n in (4, 3):               # 3: without the trace length, in case the tokenizer counted it differently
        keys, j = [], 0
        for row in rows:
            while j < len(sig) and sig[j][:n] != row[:n]:
                j += 1
            if j == len(sig):
                break
            keys.append(f"{recs[j]['domain']}|{recs[j]['topic_id']}"); j += 1
        if len(keys) == len(rows):
            break
    assert len(keys) == len(rows), (f"features {fs} do not match {gen_path(split, fs)}: the file changed after build; "
                                    f"delete {src} and recompute (EVAL_ONLY = False)")
    print(f"  {fs}: keys recovered from the gen-file order ({len(rows)} rows; matched on {n} fields; "
          f"dropped for length: {len(recs) - len(rows)})")
    return {**d, "key": np.array(keys)}

F = {}
for part in ("train", "val"):
    F[f"art_{part}"] = build(f"art_{part}", lambda part=part: art_rows(part))
F["ret_train"] = build("ret_train", lambda: reteco_rows("train", TRAIN_STAGES))
_miss = [k for k, v in F.items() if v is None]
assert not _miss, f"missing features {_miss} in {TR}: run the full pass first (EVAL_ONLY = False, GPU)"

# RETECO dev: one file per first stage (feat_ret_dev-<stage>.npz), so a new stage leaves the existing
# ones untouched. A combined feat_ret_dev.npz from earlier runs (three BM25 stages) is read and split by stage.
OLD = f"{TR}/feat_ret_dev.npz"
old = dict(np.load(OLD, allow_pickle=True)) if os.path.exists(OLD) else None
DEV, DEV_OK = [], []
for fs in DEV_STAGES:
    if old is not None and fs in set(old["meta"].tolist()):
        d, src = {k: v[old["meta"] == fs] for k, v in old.items()}, OLD
    else:
        d, src = build(f"ret_dev-{fs}", lambda fs=fs: reteco_rows("dev", [fs])), f"{TR}/feat_ret_dev-{fs}.npz"
    if d is not None and len(d["y"]):
        DEV.append(with_keys(d, fs, src)); DEV_OK.append(fs)
assert DEV, f"no RETECO dev features in {TR}"
F["ret_dev"] = {k: np.concatenate([d[k] for d in DEV]) for k in set.intersection(*(set(d) for d in DEV))}
print("RETECO dev, stages with features:", DEV_OK, "· rows:", len(F["ret_dev"]["y"]))
'''),
("code", r'''
# ── 10. Detectors: train on synthetic data or RETECO train, evaluate on RETECO dev ──
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

def lab(d):
    m = d["y"] >= 0                    # the "part" label (only some of the gold) is excluded
    return {k: v[m] for k, v in d.items()}

A_tr, A_va, R_tr, R_dev = (lab(F[k]) for k in ("art_train", "art_val", "ret_train", "ret_dev"))
rng = np.random.default_rng(SEED)
from sklearn.model_selection import GroupShuffleSplit
# 20% of RETECO train dialogues select the layer for the in-domain setting; split by dialogue, not by row:
# each turn appears once per first stage, and the turns of a dialogue share its history
_, _hi = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(R_tr["y"], groups=R_tr["grp"]))
hold = np.zeros(len(R_tr["y"]), bool); hold[_hi] = True
sub = lambda d, m: {k: v[m] for k, v in d.items()}
R_fit, R_hold = sub(R_tr, ~hold), sub(R_tr, hold)
auc = lambda y, s: round(float(roc_auc_score(y, s)), 4) if len(set(y)) == 2 else float("nan")
trace = lambda d: np.stack([d["tlen"], d["tlp"]], 1)
fit = lambda X, y: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)

def by_stage(score_fn):
    return {fs: auc(R_dev["y"][R_dev["meta"] == fs], score_fn(sub(R_dev, R_dev["meta"] == fs))) for fs in DEV_OK}

rows, DET = [], []                     # DET: the trained detectors; cell 11 evaluates the same ones
def row(name, src, layer, s_va, s_dev_fn):
    DET.append((name, src, layer, s_dev_fn))
    rows.append({"detector": name, "train": src, "layer": layer, "auroc_val": s_va,
                 "auroc_dev": auc(R_dev["y"], s_dev_fn(R_dev)), **by_stage(s_dev_fn)})

# Detector and training-source names are data values in transfer.csv and transfer_content.csv;
# baselines_v2.py matches them in transfer_content.csv (XNAME). synthetic = SQuAD 2.0 + MuSiQue.
row("OCC verdict", "—", "", auc(A_va["y"], A_va["status"]), lambda d: d["status"])
row("p_status", "—", "", auc(A_va["y"], A_va["p"]), lambda d: d["p"])
for src, tr, va in (("synthetic", A_tr, A_va), ("RETECO train", R_fit, R_hold)):
    m = fit(trace(tr), tr["y"])
    row("trace", src, "", auc(va["y"], m.predict_proba(trace(va))[:, 1]), lambda d, m=m: m.predict_proba(trace(d))[:, 1])
    best = max(range(len(LAYERS)), key=lambda j: auc(va["y"], fit(tr["H"][:, j], tr["y"]).predict_proba(va["H"][:, j])[:, 1]))
    m = fit(tr["H"][:, best], tr["y"])
    row("probe", src, LAYERS[best], auc(va["y"], m.predict_proba(va["H"][:, best])[:, 1]),
        lambda d, m=m, j=best: m.predict_proba(d["H"][:, j])[:, 1])

print(f"RETECO dev: {int(R_dev['y'].sum())} with gold / {int((1 - R_dev['y']).sum())} without; "
      f"synthetic val: {len(A_va['y'])}")
print("OCC verdict on dev, balanced accuracy:",
      round(balanced_accuracy_score(R_dev["y"], R_dev["status"]), 4))
cols = ["auroc_val", "auroc_dev"] + DEV_OK
short = lambda fs: fs.replace("dense_hanhainebula_reason-embed-qwen3-4b-0928", "RE")    # hyb_RE_q2d
print(f"{'detector':<14}{'trained on':<16}{'layer':>5}" + "".join(f"{short(c)[-11:]:>12}" for c in cols))
for r in rows:
    print(f"{r['detector']:<14}{r['train']:<16}{str(r['layer']):>5}" + "".join(f"{r[c]:>12.4f}" for c in cols))
os.makedirs(f"{DRIVE}/analysis", exist_ok=True)
with open(f"{DRIVE}/analysis/transfer.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\nauroc_val: on the held-out part of the training data; auroc_dev: transfer to RETECO dev 2c"
      "\n(all stages with features pooled), followed by per-stage values.")
'''),
("code", r'''
# ── 11. The same detectors under the content label (suff.ipynb) ─────────────
# Label names are data values in transfer_content.csv (baselines_v2.py matches qrels and content, XLAB):
# qrels   — the qrels label: all gold in the top-5 = 1, none = 0;
# content — the content label, the sufficiency judge read the documents: yes = 1, no = 0 (partial and
#           unparsed labels are excluded; rows labelled part by qrels are included);
# clean   — the clean label: yes = 1 vs. no where the answer is also absent from the dialogue history
#           (answered_in_history = false).
# Detectors are not retrained: these are DET from cell 10. Stage "total" = all stages with content labels pooled.
D = F["ret_dev"]                       # all dev rows, including those labelled part by qrels
LAB = {"qrels": D["y"].astype(int), "content": np.full(len(D["y"]), -1), "clean": np.full(len(D["y"]), -1)}
HAVE = []
for fs in DEV_OK:
    path = f"{DRIVE}/suff/{SUFF_JUDGE}_2c-{fs}_dev.jsonl"
    if not os.path.exists(path):
        print("missing labels", path); continue
    S = {f"{r['domain']}|{r['topic_id']}": r for r in map(json.loads, open(path, encoding="utf-8"))}
    idx, miss = np.where(D["meta"] == fs)[0], 0
    for i in idx:
        s = S.get(str(D["key"][i]))
        if s is None:
            miss += 1; continue
        LAB["content"][i] = {"yes": 1, "no": 0}.get(s["label"], -1)
        LAB["clean"][i] = 1 if s["label"] == "yes" else 0 if s["label"] == "no" and s["answered_in_history"] is False else -1
    HAVE.append(fs)
    print(f"{short(fs)}: feature rows {len(idx)}, without a suff label {miss}")
if not HAVE:
    print(f"missing {DRIVE}/suff/{SUFF_JUDGE}_2c-*_dev.jsonl: run suff.ipynb first; evaluation skipped")
else:
    parts = [(fs, D["meta"] == fs) for fs in HAVE] + [("total", np.isin(D["meta"], HAVE))]
    print("\nn (1 / 0):")
    for ln, y in LAB.items():
        print(f"  {ln:<7}" + "  ".join(f"{short(p)} {int((y[m] == 1).sum())}/{int((y[m] == 0).sum())}" for p, m in parts))
    out = []
    print(f"\n{'detector':<14}{'trained on':<16}{'layer':>5}{'label':>8}" + "".join(f"{short(p)[-11:]:>12}" for p, _ in parts))
    for name, src, layer, fn in DET:
        s = fn(D)
        for ln, y in LAB.items():
            vals = []
            for p, m in parts:
                k = m & (y >= 0)
                vals.append(auc(y[k], s[k]))
                out.append({"detector": name, "train": src, "layer": layer, "label": ln, "stage": p,
                            "n_pos": int((y[k] == 1).sum()), "n_neg": int((y[k] == 0).sum()), "auroc": vals[-1]})
            print(f"{name:<14}{src:<16}{str(layer):>5}{ln:>8}" + "".join(f"{v:>12.4f}" for v in vals))
    with open(f"{DRIVE}/analysis/transfer_content.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\nAUROC of the same detectors; per-stage qrels values match the table above. → {DRIVE}/analysis/transfer_content.csv")
'''),
    ]
    return nb(cells, "A100")

# ============================================================== mtrag ========
MTRAG_WORKER = r'''
"""MTRAG: GPU steps run as separate processes, so that each model frees GPU memory
before the next one (like gen_worker in generate.ipynb). The notebook also imports this module.
    python mtrag_worker.py embed <cfg>           passage and query vectors → Drive (float16 .npy)
    python mtrag_worker.py judge <cfg>           sufficiency judge, suff.ipynb prompt → Drive
    python mtrag_worker.py read <reader> <cfg>   reader responses, classes and status as in gen_worker → Drive
A job is a jsonl line {"job", "key", "t": {history, query[, reference]}, "docs": [texts], "meta"};
keys already in the output file are skipped; results are written to Drive every `chunk` prompts."""
import json, os, re, sys, time, zipfile
import gen_worker as G
import suff_prompt as S


def jsonl(path):
    """Rows of a jsonl file; a truncated record (session died mid-line) is skipped."""
    out = []
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def append(path):
    """Open for appending; if the previous write stopped mid-line, add a newline first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, 2)
            tail = f.read(1)
        if tail != b"\n":
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
    return open(path, "a", encoding="utf-8")


def norm(text):
    """Collapse runs of blank lines into one: cloud/govt have dozens of them, wasting prompt tokens."""
    return re.sub(r"\n\s*\n", "\n\n", (text or "").strip())


def load_corpus(repo, dom):
    """Domain passages read directly from corpora/passage_level/<domain>.jsonl.zip. Row:
    {"_id","id","url","text","title"}; the title is already the first line of text; _id = id in qrels."""
    z = zipfile.ZipFile(f"{repo}/corpora/passage_level/{dom}.jsonl.zip")
    name = max((n for n in z.namelist() if n.endswith(".jsonl")), key=lambda n: z.getinfo(n).file_size)
    ids, texts = [], []
    with z.open(name) as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                ids.append(str(d.get("_id") or d["id"]))
                texts.append(norm(d.get("text")))
    return ids, texts


def todo_by_out(items, out_of):
    """[(output file, jobs whose key is not done yet, number already done)] in order of first appearance."""
    groups = {}
    for it in items:
        groups.setdefault(out_of(it), []).append(it)
    work = []
    for path, its in groups.items():
        done = {r.get("key") for r in jsonl(path)}
        work.append((path, [it for it in its if it["key"] not in done], len(done)))
    return work


def fit(tok, build, docs, limit, trim_words):
    """Build the prompt; if it does not fit, trim each document to trim_words words (as in generate/suff); if it still does not fit, None."""
    ntok = lambda p: len(tok(p, add_special_tokens=False).input_ids)
    p = build(docs)
    if ntok(p) <= limit:
        return p, False
    p = build([" ".join(d.split()[:trim_words]) for d in docs])
    return (p, True) if ntok(p) <= limit else (None, True)


def progress(name, k, n, t0):
    el = time.time() - t0
    print(f"  {name}: {k}/{n} · {el / 60:.1f} min · ≈{el / max(k, 1) * (n - k) / 60:.0f} min left", flush=True)


def embed(C):
    import numpy as np, torch, transformers
    from packaging.version import Version
    from sentence_transformers import SentenceTransformer
    idx, block = C["index"], 20000
    docs_todo = [d for d in C["domains"] if not os.path.exists(f"{idx}/{d}.npy")]
    q_todo = [d for d in C["domains"] if not os.path.exists(C["q_path"][d])]
    if not docs_todo and not q_todo:
        print("embedder: everything is already done, model not loaded", flush=True)
        return
    dk = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    model = SentenceTransformer(C["model"], device="cuda", trust_remote_code=True,
                                model_kwargs={dk: torch.bfloat16 if C["dtype"] == "bfloat16" else torch.float16})
    qp = (model.prompts or {}).get("query") or ("Instruct: Given a web search query, retrieve relevant "
                                                "passages that answer the query\nQuery:")
    dp = (model.prompts or {}).get("document") or ""
    print("query prompt:", repr(qp), "· document prompt:", repr(dp), flush=True)

    def enc(texts, prompt, max_len):
        model.max_seq_length = max_len
        e = model.encode(texts, prompt=prompt, batch_size=C["batch"], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        assert not np.isnan(e).any(), "NaN in vectors"
        return e.astype(np.float16)

    def save(path, arr):
        with open(path + ".tmp", "wb") as f:
            np.save(f, arr)
        os.replace(path + ".tmp", path)

    for dom in docs_todo:
        ids, texts = load_corpus(C["repo"], dom)
        t0, parts = time.time(), []
        for b in range(0, len(texts), block):     # in blocks: an interruption loses at most one block
            pp = f"{idx}/{dom}.part{b // block:03d}.npy"
            if not os.path.exists(pp):
                save(pp, enc(texts[b:b + block], dp, C["max_doc_len"]))
            parts.append(pp)
            progress(f"vectors {dom}", min(b + block, len(texts)), len(texts), t0)
        json.dump(ids, open(f"{idx}/{dom}.ids.json", "w"))
        save(f"{idx}/{dom}.npy", np.concatenate([np.load(p) for p in parts]))
        for p in parts:
            os.remove(p)
        print(f"{dom}: {len(texts)} passages in {(time.time() - t0) / 60:.1f} min", flush=True)
    for dom in q_todo:
        qs = C["queries"][dom]
        save(C["q_path"][dom], enc([t for _, t in qs], qp, C["max_q_len"]) if qs else np.zeros((0, 1), np.float16))
    json.dump({"model": C["model"], "query_prompt": qp, "doc_prompt": dp, "max_doc_len": C["max_doc_len"],
               "max_q_len": C["max_q_len"], "dtype": C["dtype"], "stored": "float16, L2-normalized",
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
               "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{idx}/config.json", "w"), ensure_ascii=False, indent=2)


def judge(C):
    from vllm import LLM, SamplingParams
    short = C["judge"].split("/")[-1]
    work = todo_by_out(jsonl(C["judge_items"]), lambda it: f"{C['out']}/suff/{short}_{it['job']}.jsonl")
    if not any(its for _, its, _ in work):
        print("judge: everything is already done, model not loaded", flush=True)
        return
    llm = LLM(model=C["judge"], dtype=C["dtype"], max_model_len=C["maxlen"], gpu_memory_utilization=0.9, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=C["judge_new"])
    chat = lambda msgs: tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
    for path, its, nd in work:
        prompts, keep, skipped = [], [], 0
        for it in its:
            p, trimmed = fit(tok, lambda docs, t=it["t"]: chat(S.suff_msgs(t, docs)), it["docs"],
                             C["maxlen"] - C["judge_new"], C["trim_words"])
            if p is None:
                skipped += 1
                continue
            prompts.append(p)
            keep.append((it, trimmed))
        name = os.path.basename(path)
        print(f"{name}: to do {len(prompts)}, done {nd}, did not fit {skipped}", flush=True)
        t0 = time.time()
        for s in range(0, len(prompts), C["chunk"]):
            outs = llm.generate(prompts[s:s + C["chunk"]], sp, use_tqdm=False)
            with append(path) as f:
                for (it, trimmed), o in zip(keep[s:s + C["chunk"]], outs):
                    raw = o.outputs[0].text
                    f.write(json.dumps({"key": it["key"], **it["meta"], "judge": C["judge"],
                                        **S.parse_suff(raw, len(it["docs"])), "raw": raw, "trimmed": trimmed},
                                       ensure_ascii=False) + "\n")
            progress(name, s + len(outs), len(prompts), t0)


def read(reader_id, C):
    from vllm import LLM, SamplingParams
    short = reader_id.split("/")[-1]
    work = todo_by_out(jsonl(C["read_items"]), lambda it: f"{C['out']}/gen/{short}_{it['job']}.jsonl")
    if not any(its for _, its, _ in work):
        print(f"{short}: everything is already done, model not loaded", flush=True)
        return
    llm = LLM(model=reader_id, dtype=C["dtype"], max_model_len=C["maxlen"], gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    R = (G.OCC if "OCC" in reader_id else G.Control)(tok)
    # OCC: greedy only, no repetition_penalty; otherwise the structured format breaks (as in gen_worker)
    sp = SamplingParams(temperature=0.0, max_tokens=C["read_new"], stop_token_ids=R.stop,
                        skip_special_tokens=R.skip_special, spaces_between_special_tokens=False)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    for path, its, nd in work:
        prompts, keep, skipped = [], [], 0
        for it in its:
            p, trimmed = fit(tok, lambda docs, t=it["t"]: R.prompt(t, docs), it["docs"],
                             C["maxlen"] - C["read_new"], C["trim_words"])
            if p is None:
                skipped += 1
                continue
            prompts.append(p)
            keep.append((it, trimmed))
        name = os.path.basename(path)
        print(f"{name}: to do {len(prompts)}, done {nd}, did not fit {skipped}", flush=True)
        t0 = time.time()
        for s in range(0, len(prompts), C["chunk"]):
            outs = llm.generate(prompts[s:s + C["chunk"]], sp, use_tqdm=False)
            raws = [o.outputs[0].text for o in outs]
            # second pass, as in gen_worker: prompt + response up to the status position, one token, top-20 logprobs
            pref = [(i, R.prefix(r)) for i, r in enumerate(raws)]
            pref = [(i, x) for i, x in pref if x is not None]
            probs = {}
            if pref:
                o1 = llm.generate([prompts[s + i] + x for i, x in pref], sp1, use_tqdm=False)
                for (i, _), o in zip(pref, o1):
                    probs[i] = G.p_status(tok, o.outputs[0].logprobs[0])
            with append(path) as f:
                for i, (o, raw) in enumerate(zip(outs, raws)):
                    it, trimmed = keep[s + i]
                    status, answer = R.parse(raw)
                    p, mass = probs.get(i, (None, 0.0))
                    f.write(json.dumps({"key": it["key"], **it["meta"], "reader": reader_id, "status": status,
                                        "answer": answer, "p_answerable": p, "p_mass": mass,
                                        "finish": o.outputs[0].finish_reason, "trimmed": trimmed, "raw": raw},
                                       ensure_ascii=False) + "\n")
            progress(name, s + len(outs), len(prompts), t0)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "read":
        read(sys.argv[2], json.load(open(sys.argv[3])))
    else:
        {"embed": embed, "judge": judge}[mode](json.load(open(sys.argv[2])))
'''

# separate pip calls: if PyStemmer fails to build, vLLM and the rest still install (BM25 then runs without stemming)
MTRAG_SETUP = LLM_SETUP.replace(
    "!pip -q install vllm openai pytrec-eval-terrier 2>&1 | tail -2",
    "!pip -q install vllm openai pytrec-eval-terrier 2>&1 | tail -2\n"
    "!pip -q install bm25s sentence-transformers 2>&1 | tail -2\n"
    "!pip -q install PyStemmer 2>&1 | tail -2")
assert MTRAG_SETUP != LLM_SETUP


def _between(src, start, end=None):
    """Slice of a shared cell, so the MTRAG notebook reuses its code (write_run, the log) instead of a copy."""
    i = src.index(start)
    return src[i:src.index(end, i)] if end else src[i:]


def mtrag_nb():
    cells = [
("markdown", r"""
# RETECO · Replication on MTRAG: precision of the qrels label

Repeats the paper's main experiment on a second dataset, **MTRAG** (Katsis et al., TACL 2025:
110 conversations, 842 turns, 4 domains, human answerability for every turn, reference answers).
Claim: the label "no gold passage in the top 5 ⇒ context is insufficient" loses precision as
retrieval gets stronger (retrieved non-gold passages often contain the answer), and this changes
the conclusions about reader abstention. The results are Section 7 (Replication on MTRAG)
and Table 2 of the paper.

1. **Data**: files from `github.com/IBM/mt-rag-benchmark` at a pinned commit (`MTRAG_REV`):
   passage_level corpora clapnq, cloud, fiqa, govt (366k passages), tasks, qrels, annotator
   judgments. The label analysis uses the 777 ANSWERABLE + PARTIAL turns (they have qrels).
2. **Retrievers**, top 100: BM25 over all user questions, BM25 over the last turn, BM25 over
   MTRAG's rewritten question, dense `Qwen3-Embedding-0.6B` over the rewrite, and the hybrid of
   BM25 and dense over the rewrite (min-max, 0.5/0.5, fixed in advance: there is no train split).
   Metrics: nDCG@10, R@5. Table 2 of the paper orders the retrievers by nDCG@10.
3. **Contexts**: the top 5 of each retriever, plus "all gold" of the turn (the analogue of 2b).
4. **Judge** `microsoft/phi-4`: the same prompt and parsing as in `suff.ipynb`, with the MTRAG answer as
   reference: yes / partial / no for each context.
5. **Human check of the judge**: each passage on its own, both gold (qrels) and the passages that
   MTRAG annotators explicitly marked as not relevant. Gives P(judge yes | annotator: not relevant).
6. **Readers** OCC-RAG-1.7B and Qwen3-4B-Instruct-2507 answer from each context, with the same classes,
   prompts and p_answerable as in `generate.ipynb`; plus 55 unanswerable turns on the hybrid top 5
   (they have no MTRAG rewrites, so the hybrid query there = all user questions).
7. **Summary** with bootstrap (1000, seed 0): share of qrels *none*, judge labels, false misses,
   P/R of the *none* label, human check, reader refusals and AUROC of p_answerable under both labels.

**How to run.** Runtime → **A100** (phi-4 in bf16 ≈29 GB) → "Run all". If cell 2
asks for a restart, restart the session and "Run all" again. If the run is interrupted, "Run all"
again: runs, vectors, labels and answers already on Drive are skipped. To skip the readers, set
`READERS = []`: the label summary is computed before the readers and saved separately from them.

**Time (A100, estimate).** MTRAG files ≈130 MB: 1–2 min. BM25 with three query types: 5–10 min.
Vectors for 366k passages (up to 512 tokens, ≈250 on average) with the 0.6B model: 25–45 min
(once; ≈750 MB on Drive).
phi-4 judge: 777 × 6 contexts ≈ 4.7k prompts of 2–5k tokens + ≈2.6k short ones
(single passage): 45–75 min. Readers: 2 × ≈4.7k answers: OCC 15–25 min, Qwen3-4B 25–45 min.
**Total ≈2.5–3.5 h**, without readers ≈1.5–2 h. The actual speed is printed as the run progresses.

**Output.** All in `reteco/mtrag/` on Drive:
- `runs/<retriever>/<domain>/run.trec` + `config.json`; `retrieval.csv`: nDCG@10, R@5, R@100 (macro and micro)
- `suff/phi-4_<retriever|gold>.jsonl`: judge label, rationale, evidence_docs, raw response, qrels label;
  `suff/phi-4_single.jsonl`: single passages with the human label
- `gen/<reader>_<retriever|gold|unans_hybrid>.jsonl`: status, p_answerable, answer, qrels label
- `summary.csv`: all numbers with 95% CIs; `turns.jsonl`, `passages.jsonl`: turns and the texts of all
  passages in the contexts (for local analysis without the corpus); `unanswerable_hybrid_top5.jsonl`

**Next.** Copy `reteco/mtrag/` (without `index/`) from Drive to `data/drive/reteco/mtrag/` of the repo.
"""),
("code", r'''
# ── 1. Settings ──────────────────────────────────────────────────────────────
MTRAG_REV = "53fef6f5917c7e10c436f697e69a955dbc795d81"   # IBM/mt-rag-benchmark commit (main as of 2026-10-10)
MT_DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]         # in the MTRAG tasks cloud is called ibmcloud
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMB_BATCH = 64        # "CUDA out of memory" → 32
MAX_DOC_LEN = 512     # tokens per passage for the embedder (MTRAG passages are 512-token chunks)
MAX_Q_LEN = 512
BM25_K1, BM25_B = 0.9, 0.4   # as in the RETECO kit's BM25; bm25s (Lucene), English stopwords, PyStemmer stemming
# Hybrid: min-max fusion of BM25 and dense on the rewritten question. The weight is fixed a priori, NOT tuned:
# MTRAG has no train split, and tuning on the same turns we evaluate on is not allowed.
HYB_W = 0.5           # BM25 weight; dense gets 1 − HYB_W
TOPK = 100            # run length
K_CTX = 5             # documents in the context, as in generate.ipynb and suff.ipynb
JUDGE = "microsoft/phi-4"   # sufficiency judge: same prompt and parsing as in suff.ipynb
READERS = ["occ-ai/OCC-RAG-1.7B", "Qwen/Qwen3-4B-Instruct-2507"]   # [] = labels only, no readers
DO_GOLD = True        # also the "all gold of the turn" context (analogue of 2b), for the judge and readers
DO_SINGLE = True      # human check of the judge: each passage on its own (gold and passages rejected by annotators)
SINGLE_CLASSES = ["ANSWERABLE"]   # human answerability classes of the turns included in the check
MAXLEN = 16384        # phi-4 allows no more; gold can have up to 9 passages
JUDGE_NEW = 400       # as MAX_NEW in suff.ipynb
READ_NEW = 1200       # as MAX_NEW in generate.ipynb: OCC writes its reasoning before the answer
TRIM_WORDS = 300      # if a prompt does not fit, each document is cut to this many words (as in suff/generate)
CHUNK = 256           # prompts between writes to Drive
N_BOOT, SEED = 1000, 0   # summary bootstrap: over turns within a domain, 95% percentile
DRIVE = "/content/drive/MyDrive/RETECO 2026/reteco"   # project folder on Drive; all MTRAG output goes to DRIVE/mtrag
'''),
("code", MTRAG_SETUP),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", "%%writefile /content/suff_prompt.py\n" + "import json, re\n\n" + SUFF_PROMPT.strip("\n")),
("code", "%%writefile /content/mtrag_worker.py\n" + MTRAG_WORKER.strip("\n")),
("code", r'''
# ── 3. Log, TREC runs, GPU steps as separate processes ─────────────────────
import os, re, sys, json, time, csv, hashlib, subprocess, urllib.request
import numpy as np, pytrec_eval
sys.path.insert(0, "/content")
import gen_worker as G, suff_prompt as S, mtrag_worker as W
OUT = f"{DRIVE}/mtrag"
for _d in ("runs", "suff", "gen", "index"):
    os.makedirs(f"{OUT}/{_d}", exist_ok=True)

''' + _between(COMMON_CODE, "def read_run(path):", "def strata(rs):") + r'''
def run_path(name, dom):
    return f"{OUT}/runs/{name}/{dom}/run.trec"

def stream(cmd):
    """Stream the child process output into the cell line by line (as in generate.ipynb)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"})
    for line in p.stdout:
        print(line, end="")
    return p.wait()

def free_gpu(min_free_gb=34):
    """Before vLLM: kill leftover GPU processes (except the kernel) and wait for free memory.
    Without this the judge failed to start: only 9 of 40 GB were free on the A100 after the embedder."""
    import torch
    if not torch.cuda.is_available():
        return
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        out = ""
    for pid in {int(x) for x in out.split() if x.strip().isdigit()} - {os.getpid()}:
        try:
            os.kill(pid, 9); print("killed GPU process:", pid)
        except OSError:
            pass
    for _ in range(60):
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= min(min_free_gb, 0.9 * total / 2**30):
            break
        time.sleep(2)
    print(f"GPU: {free / 2**30:.1f} of {total / 2**30:.1f} GB free")

def save_jsonl(path, rows):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)

''' + _between(COMMON_CODE, "# ── log:")),
("code", r'''
# ── 4. MTRAG data: repository files, turns, qrels, annotator judgments ──────
REPO = "/content/mt-rag-benchmark"
RAW = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{MTRAG_REV}"
H = "mtrag-human"
FILES = ([f"corpora/passage_level/{d}.jsonl.zip" for d in MT_DOMAINS]
         + [f"{H}/generation_tasks/reference.jsonl", f"{H}/conversations/conversations.json"]
         + [f"{H}/retrieval_tasks/{d}/{x}" for d in MT_DOMAINS
            for x in (f"{d}_lastturn.jsonl", f"{d}_questions.jsonl", f"{d}_rewrite.jsonl", "qrels/dev.tsv")])

def fetch(rel):
    path = f"{REPO}/{rel}"
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(f"{RAW}/{rel}", path + ".tmp")
            os.replace(path + ".tmp", path)
            return path
        except Exception as e:
            print("  download failed:", rel, type(e).__name__, e)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed to download {RAW}/{rel}")

t0 = time.time()
for rel in FILES:
    fetch(rel)
print(f"MTRAG files @{MTRAG_REV[:7]}: {len(FILES)} in {time.time() - t0:.0f} s")

def strip_user(s):
    """'|user|: a\n|user|: b' → 'a\nb': the speaker tag is noise for BM25 and the embedder."""
    return "\n".join(p.strip() for p in re.split(r"(?:^|\n)\|user\|:", s or "") if p.strip())

def dom_of(collection):
    """mt-rag-ibmcloud-elser-512-100-20240502 → cloud: the tasks say ibmcloud, the paths and corpus say cloud."""
    names = {"clapnq": "clapnq", "ibmcloud": "cloud", "cloud": "cloud", "fiqa": "fiqa", "govt": "govt"}
    hit = [names[p] for p in str(collection).lower().split("-") if p in names]
    assert hit, f"unknown collection {collection}"
    return hit[0]

def qa_pairs(msgs):
    """input up to the current question → [[question, agent answer]]: the history as the judge and readers see it."""
    out = []
    for m in msgs:
        if m["speaker"] == "user":
            out.append([m["text"].strip(), ""])
        elif out:
            out[-1][1] = (out[-1][1] + "\n" + m["text"].strip()).strip()
    return out

def page_of(dom, pid):
    """Source page of a passage: the id without chunk offsets; for clapnq also without the paragraph offset
    (822086267_7384-8758-0-1374 → 822086267, ibmcld_00474-7885-8455 → ibmcld_00474)."""
    p = pid.split("-")[0]
    return p.split("_")[0] if dom == "clapnq" else p

# qrels (BEIR tsv with a header): query-id = <conversation_id><::><turn>
QRELS = {d: {} for d in MT_DOMAINS}
for d in MT_DOMAINS:
    for ln in open(f"{REPO}/{H}/retrieval_tasks/{d}/qrels/dev.tsv", encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] != "query-id" and float(p[2]) > 0:
            QRELS[d].setdefault(p[0], {})[p[1]] = int(float(p[2]))
QFILE = {v: {} for v in ("lastturn", "questions", "rewrite")}
for d in MT_DOMAINS:
    for v in QFILE:
        for ln in open(f"{REPO}/{H}/retrieval_tasks/{d}/{d}_{v}.jsonl", encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                QFILE[v][r["_id"]] = strip_user(r["text"])

# turns: history (questions and agent answers before the current one), reference, human answerability
TASKS, bad_input = {}, 0
for ln in open(f"{REPO}/{H}/generation_tasks/reference.jsonl", encoding="utf-8"):
    if not ln.strip():
        continue
    r = json.loads(ln)
    inp = r.get("input") or []
    if not inp or inp[-1]["speaker"] != "user":      # a task must end with a user question
        bad_input += 1
        continue
    d = dom_of(r.get("Collection") or r.get("collection"))
    users = [m["text"].strip() for m in inp if m["speaker"] == "user"]
    tid = r["task_id"]
    t = {"task_id": tid, "conversation_id": r["conversation_id"], "turn": int(r["turn"]), "domain": d,
         "answerability": (r.get("Answerability") or r.get("answerability") or ["?"])[0],
         "query": users[-1], "history": qa_pairs(inp[:-1]),
         "reference": ((r.get("targets") or [{}])[0].get("text") or "").strip(),
         "gold": list(QRELS[d].get(tid, {})), "ts": (inp[-1].get("metadata") or {}).get("created_at"),
         "q_lastturn": QFILE["lastturn"].get(tid) or users[-1],
         "q_questions": QFILE["questions"].get(tid) or "\n".join(users),
         "rewrite_from_mtrag": tid in QFILE["rewrite"]}
    # only turns without qrels (UNANSWERABLE) lack MTRAG rewrites: for them the "rewrite" = all questions
    t["q_rewrite"] = QFILE["rewrite"].get(tid) or t["q_questions"]
    TASKS[tid] = t
LABEL = sorted(x for x, t in TASKS.items() if t["answerability"] in ("ANSWERABLE", "PARTIAL") and t["gold"])
UNANS = sorted(x for x, t in TASKS.items() if t["answerability"] == "UNANSWERABLE")
RET_TURNS = LABEL + UNANS
from collections import Counter
print("turns:", len(TASKS), dict(Counter(t["answerability"] for t in TASKS.values())))
print("for labels (ANSWERABLE + PARTIAL with qrels):", len(LABEL), dict(Counter(TASKS[x]["domain"] for x in LABEL)),
      "· unanswerable:", len(UNANS))
orphan = [q for d in MT_DOMAINS for q in QRELS[d] if q not in TASKS]
print("qrels without a task:", len(orphan), "· tasks not ending in a question (skipped):", bad_input,
      "· label turns without an MTRAG rewrite:",
      sum(not TASKS[x]["rewrite_from_mtrag"] for x in LABEL),
      "· last turn in file ≠ in task:", sum(TASKS[x]["q_lastturn"] != TASKS[x]["query"] for x in LABEL))
assert LABEL and len(orphan) <= 0.02 * sum(map(len, QRELS.values())), "turns and qrels do not match: check the formats"

# annotator judgments (conversations.json). A turn is matched by (timestamp, question text);
# the final judgment follows MTRAG's scripts/conversations2retrieval.py: the editor's if they judged, otherwise the author's
CONVS = json.load(open(f"{REPO}/{H}/conversations/conversations.json", encoding="utf-8"))
BY_TS = {(t["ts"], t["query"]): x for x, t in TASKS.items()}
HUMAN, n_user, unmatched, misturn = {}, 0, 0, 0
for c in CONVS:
    cur, k = None, 0
    for m in c.get("messages", []):
        if m.get("speaker") == "user":
            k += 1; n_user += 1
            cur = BY_TS.get((m.get("timestamp"), (m.get("text") or "").strip()))
            unmatched += cur is None
            misturn += cur is not None and TASKS[cur]["turn"] != k
        elif cur is not None:
            for cx in m.get("contexts") or []:
                fb = (cx.get("feedback") or {}).get("relevant") or {}
                who = c.get("editor") if c.get("editor") in fb else c.get("author")
                HUMAN.setdefault(cur, {})[cx["document_id"]] = {
                    "final": (fb.get(who) or {}).get("value"),
                    "votes": sorted(str((v or {}).get("value")) for v in fb.values()),
                    "text": cx.get("text") or ""}
print(f"conversations {len(CONVS)}, questions {n_user}: no matching task {unmatched}, turn number mismatch {misturn}")
_ans = [x for x in LABEL if TASKS[x]["answerability"] == "ANSWERABLE"]
_g = [(x, p) for x in _ans for p in TASKS[x]["gold"]]
print("ANSWERABLE gold with annotator judgment 'yes':", sum(HUMAN.get(x, {}).get(p, {}).get("final") == "yes"
                                                        for x, p in _g), "of", len(_g),
      "· 'no' passages in ANSWERABLE turns:", sum(h["final"] == "no" and p not in TASKS[x]["gold"]
                                          for x in _ans for p, h in HUMAN.get(x, {}).items()))

# corpora: read directly from the zip files, once per session
CORPUS = {}
def corpus(d):
    if d not in CORPUS:
        ids, texts = W.load_corpus(REPO, d)
        CORPUS[d] = (ids, texts, {p: i for i, p in enumerate(ids)})
    return CORPUS[d]

def ptext(d, pid):
    i = corpus(d)[2].get(pid)
    return None if i is None else CORPUS[d][1][i]

t0 = time.time()
for d in MT_DOMAINS:
    ids, _, pos = corpus(d)
    g = {p for x in LABEL if TASKS[x]["domain"] == d for p in TASKS[x]["gold"]}
    print(f"  {d:<7} passages {len(ids):>7} · gold in corpus {sum(p in pos for p in g)}/{len(g)}")
    assert sum(p in pos for p in g) >= 0.95 * len(g), f"{d}: corpus ids do not match qrels"
print(f"corpora in {time.time() - t0:.0f} s")
'''),
("code", r'''
# ── 5. BM25 (bm25s): last turn, all questions, rewritten question → Drive ──
import bm25s
try:
    import Stemmer
    STEM, STEM_NAME = Stemmer.Stemmer("english"), "PyStemmer english"
except ImportError:
    STEM, STEM_NAME = None, None
    print("WARNING: PyStemmer is not installed; BM25 runs without stemming (recorded in config.json)")
QSETS = {"bm25_lastturn": "q_lastturn", "bm25_questions": "q_questions", "bm25_rewrite": "q_rewrite"}
tokz = lambda xs: bm25s.tokenize(xs, stopwords="en", stemmer=STEM, show_progress=False)
for d in MT_DOMAINS:
    todo = [n for n in QSETS if not os.path.exists(run_path(n, d))]
    if not todo:
        print(f"[exists] BM25 {d}"); continue
    ids, texts, _ = corpus(d)
    t0 = time.time()
    bm = bm25s.BM25(method="lucene", k1=BM25_K1, b=BM25_B)
    bm.index(tokz(texts), show_progress=False)
    tids = [x for x in RET_TURNS if TASKS[x]["domain"] == d]
    for name in todo:
        if not tids:
            write_run(run_path(name, d), {}, name); continue
        docs, scores = bm.retrieve(tokz([TASKS[x][QSETS[name]] for x in tids]), k=min(TOPK, len(ids)),
                                   show_progress=False)
        # a score of 0 means no shared terms: such "hits" are not written
        write_run(run_path(name, d), {x: {ids[j]: float(s) for j, s in zip(docs[i], scores[i]) if s > 0}
                                      for i, x in enumerate(tids)}, name)
    del bm
    print(f"BM25 {d}: {len(ids)} passages, {len(tids)} turns × {len(todo)} queries in {(time.time() - t0) / 60:.1f} min")
for name, field in QSETS.items():
    json.dump({"retriever": "bm25s " + bm25s.__version__, "method": "lucene", "k1": BM25_K1, "b": BM25_B,
               "stopwords": "en", "stemmer": STEM_NAME, "query": field, "topk": TOPK, "mtrag_rev": MTRAG_REV,
               "note": "query text without '|user|:' prefixes; q_rewrite of turns without an MTRAG rewrite "
                       "(UNANSWERABLE) = q_questions", "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{OUT}/runs/{name}/config.json", "w"), indent=2)
'''),
("code", r'''
# ── 6. Dense retrieval (Qwen3-Embedding-0.6B) on the rewritten question, and hybrid ──
# Vectors are computed by a separate process (this frees the GPU for the judge); the search runs here, numpy fp32.
SLUG = re.sub(r"[^A-Za-z0-9._-]", "_", EMBED_MODEL)
IDX = f"{OUT}/index/{SLUG}"
os.makedirs(IDX, exist_ok=True)
QUERIES = {d: [[x, TASKS[x]["q_rewrite"]] for x in RET_TURNS if TASKS[x]["domain"] == d] for d in MT_DOMAINS}
QPATH = {d: f"{IDX}/q_rewrite_{d}_" + hashlib.sha1(json.dumps([EMBED_MODEL, MAX_Q_LEN, QUERIES[d]]).encode()).hexdigest()[:12]
            + ".npy" for d in MT_DOMAINS}
json.dump({"repo": REPO, "domains": MT_DOMAINS, "index": IDX, "model": EMBED_MODEL, "batch": EMB_BATCH,
           "max_doc_len": MAX_DOC_LEN, "max_q_len": MAX_Q_LEN, "dtype": DTYPE, "queries": QUERIES, "q_path": QPATH},
          open("/content/mtrag_embed.json", "w"))
# vectors are needed only for dense runs not computed yet (the vectors may be deleted to save Drive space)
NEED_VEC = not all(os.path.exists(run_path(n, d)) for n in ("dense_rewrite", "hybrid_rewrite") for d in MT_DOMAINS)
if NEED_VEC and not all(os.path.exists(p) for p in [f"{IDX}/{d}.npy" for d in MT_DOMAINS] + list(QPATH.values())):
    assert stream([sys.executable, "/content/mtrag_worker.py", "embed", "/content/mtrag_embed.json"]) == 0, \
        "embedder failed, see the log above"
EMB_CFG = (json.load(open(f"{IDX}/config.json")) if os.path.exists(f"{IDX}/config.json")
           else {"model": EMBED_MODEL, "note": "vectors deleted, runs are ready"})

def minmax_fuse(parts):
    """parts: [(weight, {doc: score})] → {doc: Σ weight·minmax(score)}; absent from a list = 0 (as in fuse.py and analysis)."""
    s = {}
    for w, r in parts:
        if r:
            lo, hi = min(r.values()), max(r.values())
            for d, v in r.items():
                s[d] = s.get(d, 0.0) + w * ((v - lo) / (hi - lo) if hi > lo else 1.0)
    return dict(sorted(s.items(), key=lambda x: -x[1])[:TOPK])

for d in MT_DOMAINS:
    if not os.path.exists(run_path("dense_rewrite", d)):
        ids = json.load(open(f"{IDX}/{d}.ids.json"))
        E, Q = np.load(f"{IDX}/{d}.npy").astype(np.float32), np.load(QPATH[d]).astype(np.float32)
        run = {}
        for b in range(0, len(Q), 256):
            sc = Q[b:b + 256] @ E.T                   # vectors are normalized: dot product = cosine
            k = min(TOPK, sc.shape[1])
            top = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            for i, (x, _) in enumerate(QUERIES[d][b:b + 256]):
                j = top[i][np.argsort(-sc[i, top[i]], kind="stable")]
                run[x] = {ids[c]: float(sc[i, c]) for c in j}
        write_run(run_path("dense_rewrite", d), run, "dense_rewrite")
        del E
    if not os.path.exists(run_path("hybrid_rewrite", d)):
        a, b = read_run(run_path("bm25_rewrite", d)), read_run(run_path("dense_rewrite", d))
        write_run(run_path("hybrid_rewrite", d),
                  {x: minmax_fuse([(HYB_W, a.get(x, {})), (1 - HYB_W, b.get(x, {}))]) for x in sorted(set(a) | set(b))},
                  "hybrid_rewrite")
json.dump({**EMB_CFG, "query": "q_rewrite", "topk": TOPK, "search": "cosine, numpy fp32", "mtrag_rev": MTRAG_REV},
          open(f"{OUT}/runs/dense_rewrite/config.json", "w"), ensure_ascii=False, indent=2)
json.dump({"fusion": "minmax", "parts": {"bm25_rewrite": HYB_W, "dense_rewrite": 1 - HYB_W}, "topk": TOPK,
           "missing": 0, "weight": "fixed a priori: MTRAG has no train split", "mtrag_rev": MTRAG_REV},
          open(f"{OUT}/runs/hybrid_rewrite/config.json", "w"), indent=2)
print("dense and hybrid ready:", f"{OUT}/runs")
'''),
("code", r'''
# ── 7. Retrieval ladder: nDCG@10, R@5, R@100, macro over domains and micro over turns ──
RETRIEVERS = ["bm25_lastturn", "bm25_questions", "bm25_rewrite", "dense_rewrite", "hybrid_rewrite"]
DOMS_L = [d for d in MT_DOMAINS if any(TASKS[x]["domain"] == d for x in LABEL)]
PER_TURN, RET_ROWS = {}, []
for name in RETRIEVERS:
    for d in DOMS_L:
        qr = {x: QRELS[d][x] for x in LABEL if TASKS[x]["domain"] == d}
        run = read_run(run_path(name, d))
        ev = pytrec_eval.RelevanceEvaluator(qr, {"ndcg_cut.10", "recall.5", "recall.100"}).evaluate(
            {x: run[x] for x in qr if run.get(x)})
        for x in qr:              # a turn without a run scores 0, as on the leaderboard
            e = ev.get(x, {})
            PER_TURN[(name, x)] = {"ndcg10": e.get("ndcg_cut_10", 0.0), "r5": e.get("recall_5", 0.0),
                                   "r100": e.get("recall_100", 0.0)}
    row = {"retriever": name, "n": len(LABEL)}
    for m in ("ndcg10", "r5", "r100"):
        dm = {d: float(np.mean([PER_TURN[(name, x)][m] for x in LABEL if TASKS[x]["domain"] == d])) for d in DOMS_L}
        row[f"{m}_macro"] = round(float(np.mean(list(dm.values()))), 4)
        row[f"{m}_micro"] = round(float(np.mean([PER_TURN[(name, x)][m] for x in LABEL])), 4)
        row.update({f"{m}_{d}": round(v, 4) for d, v in dm.items()})
    RET_ROWS.append(row)
with open(f"{OUT}/retrieval.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(RET_ROWS[0])); w.writeheader(); w.writerows(RET_ROWS)
print(f"{'retriever':<16}{'nDCG@10 macro':>14}{'micro':>8}{'R@5 macro':>11}{'micro':>8}{'R@100':>8}  "
      + "".join(f"{d:>8}" for d in DOMS_L))
for r in RET_ROWS:
    print(f"{r['retriever']:<16}{r['ndcg10_macro']:>14.4f}{r['ndcg10_micro']:>8.4f}{r['r5_macro']:>11.4f}"
          f"{r['r5_micro']:>8.4f}{r['r100_macro']:>8.4f}  " + "".join(f"{r[f'ndcg10_{d}']:>8.4f}" for d in DOMS_L))
print("MTRAG README (macro nDCG@10, last turn / rewrite): BM25 .21/.25, BGE-base .30/.38, ELSER .49/.54")
print("ladder by nDCG@10:", " < ".join(r["retriever"] for r in sorted(RET_ROWS, key=lambda r: r["ndcg10_macro"])),
      f"→ {OUT}/retrieval.csv")
'''),
("code", r'''
# ── 8. Contexts: top 5 of each retriever (+ all gold); files for local analysis ──
CTX = {name: {} for name in RETRIEVERS}
for name in RETRIEVERS:
    for d in MT_DOMAINS:
        top = G.read_top(run_path(name, d), K_CTX)
        for x in RET_TURNS:
            if TASKS[x]["domain"] == d:
                CTX[name][x] = top.get(x, [])
if DO_GOLD:     # gold missing from the corpus does not enter the context (no text)
    CTX["gold"] = {x: [p for p in TASKS[x]["gold"] if ptext(TASKS[x]["domain"], p) is not None] for x in LABEL}
    print("gold passages without text in the corpus:", sum(len(TASKS[x]["gold"]) - len(CTX["gold"][x]) for x in LABEL))
JOBS = RETRIEVERS + (["gold"] if DO_GOLD else [])
QLAB = {job: {x: G.sufficiency(TASKS[x]["gold"], CTX[job][x]) for x in LABEL} for job in JOBS}   # (n in context, all/part/none)
for job in JOBS:
    c = Counter(v[1] for v in QLAB[job].values())
    print(f"  {job:<16} qrels label, top-{K_CTX}: all {c['all']:>4}  part {c['part']:>4}  none {c['none']:>4}")

save_jsonl(f"{OUT}/turns.jsonl", [
    {**{k: v for k, v in TASKS[x].items() if k != "ts"}, "use": "label" if x in LABEL else "unans",
     "human": {p: h["final"] for p, h in HUMAN.get(x, {}).items()},
     "top5": {name: CTX[name].get(x, []) for name in RETRIEVERS}} for x in RET_TURNS])
need = {(TASKS[x]["domain"], p) for job in CTX for x, ids in CTX[job].items() for p in ids}
need |= {(TASKS[x]["domain"], p) for x in HUMAN for p in HUMAN[x]}
save_jsonl(f"{OUT}/passages.jsonl", [
    {"domain": d, "doc_id": p, "in_corpus": ptext(d, p) is not None, "page": page_of(d, p),
     "text": ptext(d, p) if ptext(d, p) is not None else W.norm(next((HUMAN[x][p]["text"] for x in HUMAN if p in HUMAN[x]), ""))}
    for d, p in sorted(need)])
save_jsonl(f"{OUT}/unanswerable_hybrid_top5.jsonl", [
    {"task_id": x, "domain": TASKS[x]["domain"], "turn": TASKS[x]["turn"], "query": TASKS[x]["query"],
     "history": TASKS[x]["history"], "reference": TASKS[x]["reference"], "query_used": TASKS[x]["q_rewrite"],
     "rewrite_from_mtrag": TASKS[x]["rewrite_from_mtrag"],
     "top5": [{"doc_id": p, "text": ptext(TASKS[x]["domain"], p), "human": HUMAN.get(x, {}).get(p, {}).get("final")}
              for p in CTX["hybrid_rewrite"].get(x, [])]} for x in UNANS])
print(f"turns → turns.jsonl, {len(need)} passages → passages.jsonl, {len(UNANS)} unanswerable → unanswerable_hybrid_top5.jsonl")
'''),
("code", r'''
# ── 9. phi-4 judge: each context and (DO_SINGLE) each passage on its own → Drive/reteco/mtrag/suff ──
# Prompt and parsing exactly as in suff.ipynb (suff_prompt.py); reference = the MTRAG answer (targets); documents [1]…[5] without ids.
JSHORT = JUDGE.split("/")[-1]

def tview(x, ref):
    t = TASKS[x]
    return {"history": t["history"], "query": t["query"], **({"reference": t["reference"]} if ref else {})}

def meta(x, job, ids, lab_key):
    t = TASKS[x]
    n, lab = G.sufficiency(t["gold"], ids) if t["gold"] else (0, "unans")
    return {"task_id": x, "domain": t["domain"], "turn": t["turn"], "answerability": t["answerability"],
            "job": job, "context_ids": ids, "n_gold": len(t["gold"]), "gold_in_ctx": n, lab_key: lab}

GOLD_OF_CONV = {}
for x in LABEL:
    GOLD_OF_CONV.setdefault(TASKS[x]["conversation_id"], {}).update({p: x for p in TASKS[x]["gold"]})
n_items = Counter()
with open("/content/mtrag_judge_items.jsonl", "w", encoding="utf-8") as f:
    for job in JOBS:
        for x in LABEL:
            ids = CTX[job][x]
            f.write(json.dumps({"job": job, "key": x, "t": tview(x, True), "meta": meta(x, job, ids, "suff_qrels"),
                                "docs": [ptext(TASKS[x]["domain"], p) for p in ids]}, ensure_ascii=False) + "\n")
            n_items[job] += 1
    for x in (LABEL if DO_SINGLE else []):
        t = TASKS[x]
        if t["answerability"] not in SINGLE_CLASSES:
            continue
        gold, pages = set(t["gold"]), {page_of(t["domain"], p) for p in t["gold"]}
        conv_gold = GOLD_OF_CONV.get(t["conversation_id"], {})
        cands = [(p, "gold") for p in t["gold"]] + [(p, "human_no") for p, h in HUMAN.get(x, {}).items()
                                                    if h["final"] == "no" and p not in gold]
        for p, kind in cands:
            txt = ptext(t["domain"], p)
            txt = txt if txt is not None else W.norm(HUMAN.get(x, {}).get(p, {}).get("text"))
            if not txt:
                continue
            h = HUMAN.get(x, {}).get(p, {})
            m = {**meta(x, "single", [p], "suff_qrels"), "doc_id": p, "kind": kind, "human_final": h.get("final"),
                 "human_votes": h.get("votes"), "in_corpus": ptext(t["domain"], p) is not None,
                 "same_page_as_gold": kind == "human_no" and page_of(t["domain"], p) in pages,
                 "gold_other_turn": kind == "human_no" and conv_gold.get(p, x) != x}
            f.write(json.dumps({"job": "single", "key": f"{x}|{p}", "t": tview(x, True), "meta": m, "docs": [txt]},
                               ensure_ascii=False) + "\n")
            n_items[f"single·{kind}"] += 1
print("judge items:", dict(n_items))
assert BIG, "phi-4 in bf16 needs ≈29 GB: Runtime → Change runtime type → A100"
JCFG = {"judge": JUDGE, "judge_items": "/content/mtrag_judge_items.jsonl", "out": OUT, "dtype": DTYPE,
        "maxlen": MAXLEN, "judge_new": JUDGE_NEW, "trim_words": TRIM_WORDS, "chunk": CHUNK}
json.dump(JCFG, open("/content/mtrag_judge.json", "w"))
# if the judge fails, the readers still run (they do not need labels); the summary reports what is missing
free_gpu()
JUDGE_FAILED = stream([sys.executable, "/content/mtrag_worker.py", "judge", "/content/mtrag_judge.json"]) != 0
if JUDGE_FAILED:
    print("!!! judge failed, see the log above; readers and summary continue, labels will be incomplete")
json.dump({**JCFG, "failed": JUDGE_FAILED, "jobs": JOBS, "k_ctx": K_CTX, "single_classes": SINGLE_CLASSES,
           "system": S.SYSTEM, "task": S.TASK, "reference": "MTRAG targets[0].text", "gpu": torch.cuda.get_device_name(0),
           "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{OUT}/suff/config.json", "w"), ensure_ascii=False, indent=2)
'''),
("code", r'''
# ── 10. Summary (no GPU needed): labels by retriever, human check of the judge, readers ──
# Bootstrap 1000 × seed 0, 95% percentile: labels and readers are resampled over turns within a domain,
# paired (the same turns in all contexts); the human check resamples turns as clusters.
# The judge's partial is excluded from P/R and AUROC (as in the paper); its share is reported separately.
from scipy.stats import rankdata
NAN = float("nan")
KEYS = list(LABEL)
DOMS = np.array([TASKS[x]["domain"] for x in KEYS])
STRATA = [np.where(DOMS == d)[0] for d in MT_DOMAINS if (DOMS == d).any()]

def keyed(path):
    return {r["key"]: r for r in W.jsonl(path)}

def ratio(a, b):
    return float(a) / float(b) if b else NAN

def auroc(y, s):
    """AUROC via ranks (ties get the average rank, as in roc_auc_score); nan if one of the classes is missing."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return NAN
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def boot(fn, n_items, strata=None, clusters=None):
    """fn(idx) → {metric: value}; returns {metric: (point, lo, hi)}."""
    point = fn(np.arange(n_items))
    rng, acc = np.random.default_rng(SEED), {k: [] for k in point}
    for _ in range(N_BOOT):
        if strata is not None:
            idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in strata])
        else:
            idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        for k, v in fn(idx).items():
            acc[k].append(v)
    out = {}
    for k, v in point.items():
        a = np.asarray(acc[k], float); a = a[~np.isnan(a)]
        out[k] = (v, *((float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (NAN, NAN)))
    return out

FW = 21           # width of a "value [CI]" column
def fmt(t, w=FW):
    v, lo, hi = t
    return f"{'—':>{w}}" if np.isnan(v) else f"{f'{v:.3f} [{lo:.2f},{hi:.2f}]':>{w}}"

def label_metrics(job, idx):
    q, lab = QARR[job][idx], LAB[job][idx]
    ok = lab != ""                                   # the judge label was parsed
    q, lab, sp = q[ok], lab[ok], EVSP[job][idx][ok]
    none, yes, par, no = q == "none", lab == "yes", lab == "partial", lab == "no"
    tp, fp, fn = (none & no).sum(), (none & yes).sum(), (~none & no).sum()
    m = {"n": float(ok.sum()), "share_qrels_none": float(none.mean()) if len(q) else NAN,
         "share_judge_no": float(no.mean()) if len(q) else NAN,
         "share_judge_partial": float(par.mean()) if len(q) else NAN,
         "share_judge_yes": float(yes.mean()) if len(q) else NAN,
         "false_miss": ratio((none & yes).sum(), none.sum()),
         "false_miss_yes_or_partial": ratio((none & (yes | par)).sum(), none.sum()),
         "P_none_detects_no": ratio(tp, tp + fp), "R_none_detects_no": ratio(tp, tp + fn),
         "overestimate": ratio(none.mean(), no.mean()) if len(q) else NAN,
         "false_miss_same_page": ratio((none & yes & sp).sum(), (none & yes).sum())}
    if job in RETRIEVERS:
        nd, dm = NDARR[job][idx], DOMS[idx]
        m["ndcg10"] = float(np.mean([nd[dm == d].mean() for d in np.unique(dm)]))
        m["gold_in_top5"] = float((QARR[job][idx] != "none").mean())
    return m

def summarize(final):
    global QARR, LAB, EVSP, NDARR
    rows = []
    def add(section, reader, res):
        for k, (v, lo, hi) in res.items():
            job, metric = k.rsplit("|", 1)
            rows.append({"section": section, "job": job, "reader": reader, "metric": metric,
                         "value": None if np.isnan(v) else round(v, 4),
                         "lo": None if np.isnan(lo) else round(lo, 4), "hi": None if np.isnan(hi) else round(hi, 4)})
    QARR = {j: np.array([QLAB[j][x][1] for x in KEYS]) for j in JOBS}
    NDARR = {j: np.array([PER_TURN[(j, x)]["ndcg10"] for x in KEYS]) for j in RETRIEVERS}
    LAB, EVSP = {}, {}
    for job in JOBS:
        recs = keyed(f"{OUT}/suff/{JSHORT}_{job}.jsonl")
        LAB[job] = np.array([(recs.get(x) or {}).get("label") or "" for x in KEYS])
        sp = []
        for x in KEYS:
            r, t = recs.get(x) or {}, TASKS[x]
            ids, gold = r.get("context_ids") or [], set(t["gold"])
            pages = {page_of(t["domain"], p) for p in gold}
            sp.append(any(1 <= e <= len(ids) and ids[e - 1] not in gold and page_of(t["domain"], ids[e - 1]) in pages
                          for e in r.get("evidence_docs") or []))
        EVSP[job] = np.array(sp, bool)
    have = [j for j in JOBS if (LAB[j] != "").any()]

    # (a) labels: qrels none vs. the judge, per retriever
    if have:
        rets = [j for j in RETRIEVERS if j in have]
        nd0 = {j: label_metrics(j, np.arange(len(KEYS)))["ndcg10"] for j in rets}
        weak = min(rets, key=nd0.get) if rets else None
        def fn(idx):
            out = {f"{j}|{k}": v for j in have for k, v in label_metrics(j, idx).items()}
            for j in rets:
                if j != weak:
                    for k in ("ndcg10", "share_qrels_none", "share_judge_no", "false_miss", "P_none_detects_no"):
                        out[f"{j} − {weak}|{k}"] = out[f"{j}|{k}"] - out[f"{weak}|{k}"]
            return out
        res = boot(fn, len(KEYS), strata=STRATA)
        add("labels", "", res)
        print(f"(a) qrels label 'none' (no gold in top-{K_CTX}) vs. the {JSHORT} judge; partial excluded from P/R")
        print(f"{'context':<16}{'n':>5}{'nDCG@10':>{FW}}{'qrels none':>{FW}}{'judge no':>{FW}}{'partial':>8}{'yes':>7}"
              f"{'false miss':>{FW}}{'P(none→no)':>{FW}}{'R(none→no)':>{FW}}{'same page':>11}")
        for j in sorted(have, key=lambda j: nd0.get(j, 9)):
            g = lambda k: res.get(f"{j}|{k}", (NAN, NAN, NAN))
            print(f"{j:<16}{g('n')[0]:>5.0f}{fmt(g('ndcg10'))}{fmt(g('share_qrels_none'))}{fmt(g('share_judge_no'))}"
                  f"{g('share_judge_partial')[0]:>8.3f}{g('share_judge_yes')[0]:>7.3f}{fmt(g('false_miss'))}"
                  f"{fmt(g('P_none_detects_no'))}{fmt(g('R_none_detects_no'))}{g('false_miss_same_page')[0]:>11.3f}")
        print("false miss = share of judge-yes among qrels-none; 'same page' = among those, evidence from the same page"
              "\nas the gold (a neighbouring chunk). Differences from the weakest retriever (" + str(weak) + "), paired bootstrap:")
        for j in rets:
            if j != weak:
                g = lambda k: res[f"{j} − {weak}|{k}"]
                print(f"  {j:<16} ΔnDCG {fmt(g('ndcg10'))}  ΔP {fmt(g('P_none_detects_no'))}  "
                      f"Δfalse miss {fmt(g('false_miss'))}  Δqrels none {fmt(g('share_qrels_none'))}")
    else:
        print("(a) no judge labels yet: run cell 9 first")

    # (b) human check of the judge: single passages
    sg = [r for r in W.jsonl(f"{OUT}/suff/{JSHORT}_single.jsonl") if r.get("label")]
    if sg:
        kind, lab = np.array([r["kind"] for r in sg]), np.array([r["label"] for r in sg])
        spg = np.array([bool(r.get("same_page_as_gold")) for r in sg])
        oth = np.array([bool(r.get("gold_other_turn")) for r in sg])
        # group names are the "job" values of section human_single in summary.csv: gold passages; passages the
        # annotators marked not relevant (human_no); human_no on the same page as a gold passage or on another
        # page; human_no that is the gold of another turn
        groups = {"gold (annotator: relevant)": kind == "gold", "human_no (annotator: not relevant)": kind == "human_no",
                  "human_no · same page": (kind == "human_no") & spg,
                  "human_no · other page": (kind == "human_no") & ~spg,
                  "human_no · gold of another turn": (kind == "human_no") & oth}
        tid = np.array([r["task_id"] for r in sg])
        clusters = [np.where(tid == t)[0] for t in np.unique(tid)]
        def fn(idx):
            out = {}
            for g, m in groups.items():
                l = lab[idx][m[idx]]
                out[f"{g}|n"] = float(len(l))
                out[f"{g}|judge_yes"] = float((l == "yes").mean()) if len(l) else NAN
                out[f"{g}|judge_yes_or_partial"] = float(np.isin(l, ["yes", "partial"]).mean()) if len(l) else NAN
            return out
        res = boot(fn, len(sg), clusters=clusters)
        add("human_single", "", res)
        print(f"\n(b) judge on single passages, turns {SINGLE_CLASSES}: P(judge yes | MTRAG annotator judgment)")
        print(f"{'group':<36}{'n':>6}{'judge yes':>{FW}}{'yes or partial':>{FW}}")
        for g in groups:
            print(f"{g:<36}{res[f'{g}|n'][0]:>6.0f}{fmt(res[f'{g}|judge_yes'])}{fmt(res[f'{g}|judge_yes_or_partial'])}")
        print("a passage rejected by the annotator but judged 'yes' is either a judge error or a gap in the human judgments;\n"
              "a single gold passage can be partial when the answer is assembled from several passages.")
    elif DO_SINGLE:
        print("\n(b) missing:", f"{OUT}/suff/{JSHORT}_single.jsonl")

    # (c) readers: refusals and AUROC of p_answerable under the qrels label and under the judge label
    for reader in READERS:
        short = reader.split("/")[-1]
        gen = {j: keyed(f"{OUT}/gen/{short}_{j}.jsonl") for j in JOBS}
        gen = {j: g for j, g in gen.items() if g}
        if not gen:
            if final:
                print(f"\n(c) {short}: no answers")
            continue
        A = {j: (np.array([float(g[x]["status"].startswith("UN")) if x in g else NAN for x in KEYS]),
                 np.array([g[x]["p_answerable"] if x in g and g[x]["p_answerable"] is not None else NAN for x in KEYS],
                          float)) for j, g in gen.items()}
        def fn(idx):
            out = {}
            for j, (ref, p) in A.items():
                r, pp, q, lb = ref[idx], p[idx], QARR[j][idx], LAB[j][idx]
                h = ~np.isnan(r)
                rate = lambda m: float(r[h & m].mean()) if (h & m).any() else NAN
                out[f"{j}|n"] = float(h.sum())
                out[f"{j}|refusal"] = rate(np.ones(len(idx), bool))
                out[f"{j}|refusal_S_q"], out[f"{j}|refusal_I_q"] = rate(q == "all"), rate(q == "none")
                out[f"{j}|refusal_S_c"], out[f"{j}|refusal_I_c"] = rate(lb == "yes"), rate(lb == "no")
                ok = h & ~np.isnan(pp)
                mq, mc = ok & np.isin(q, ["all", "none"]), ok & np.isin(lb, ["yes", "no"])
                out[f"{j}|auroc_qrels"] = auroc(q[mq] == "all", pp[mq])
                out[f"{j}|auroc_content"] = auroc(lb[mc] == "yes", pp[mc])
                out[f"{j}|d_auroc"] = out[f"{j}|auroc_content"] - out[f"{j}|auroc_qrels"]
                if "gold" in A and j != "gold":      # 2b/2c contrast: refusal on insufficient context minus on gold, same turns
                    rg = A["gold"][0][idx]
                    for nm, m in (("contrast_q", q == "none"), ("contrast_c", lb == "no")):
                        mm = h & ~np.isnan(rg) & m
                        out[f"{j}|{nm}"] = float(r[mm].mean() - rg[mm].mean()) if mm.any() else NAN
            return out
        res = boot(fn, len(KEYS), strata=STRATA)
        add("readers", short, res)
        print(f"\n(c) {short}: refusal = status UNANSWERABLE; S/I = sufficient/insufficient context by qrels (q: all/none)"
              f"\n    and by the judge (c: yes/no); AUROC of p_answerable; contrast = refusal on I minus on gold, same turns")
        print(f"{'context':<16}{'n':>5}{'refusal':>7}{'S_q':>7}{'I_q':>7}{'S_c':>7}{'I_c':>7}"
              f"{'AUROC qrels':>{FW}}{'AUROC judge':>{FW}}{'Δ':>{FW}}{'contrast q':>{FW}}{'contrast c':>{FW}}")
        for j in [j for j in JOBS if j in A]:
            g = lambda k: res.get(f"{j}|{k}", (NAN, NAN, NAN))
            print(f"{j:<16}{g('n')[0]:>5.0f}" + "".join(f"{g(k)[0]:>7.2f}" for k in
                  ("refusal", "refusal_S_q", "refusal_I_q", "refusal_S_c", "refusal_I_c"))
                  + "".join(fmt(g(k)) for k in ("auroc_qrels", "auroc_content", "d_auroc", "contrast_q", "contrast_c")))
        gu = keyed(f"{OUT}/gen/{short}_unans_hybrid.jsonl")
        us = [x for x in UNANS if x in gu]
        if us:
            ru = np.array([float(gu[x]["status"].startswith("UN")) for x in us])
            du = np.array([TASKS[x]["domain"] for x in us])
            ru_res = boot(lambda idx: {"unans_hybrid|refusal": float(ru[idx].mean())}, len(us),
                          strata=[np.where(du == d)[0] for d in MT_DOMAINS if (du == d).any()])
            add("readers", short, ru_res)
            print(f"{'unans_hybrid':<16}{len(us):>5}{ru_res['unans_hybrid|refusal'][0]:>7.2f}  ← UNANSWERABLE turns "
                  f"(human label), hybrid top 5; CI {fmt(ru_res['unans_hybrid|refusal']).strip()}")

    if rows:
        with open(f"{OUT}/summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\n{len(rows)} numbers → {OUT}/summary.csv")
    return rows

SUMMARY = summarize(final=False)     # the labels are ready even without readers
'''),
("code", r'''
# ── 11. Readers: each in a separate process (classes, prompts and p_answerable from gen_worker) ──
# Contexts: top 5 of each retriever, all gold (DO_GOLD) and the hybrid top 5 for UNANSWERABLE turns.
n_items = Counter()
with open("/content/mtrag_read_items.jsonl", "w", encoding="utf-8") as f:
    for job in JOBS:
        for x in LABEL:
            ids = CTX[job][x]
            f.write(json.dumps({"job": job, "key": x, "t": tview(x, False), "meta": meta(x, job, ids, "suff"),
                                "docs": [ptext(TASKS[x]["domain"], p) for p in ids]}, ensure_ascii=False) + "\n")
            n_items[job] += 1
    for x in UNANS:
        ids = CTX["hybrid_rewrite"].get(x, [])
        f.write(json.dumps({"job": "unans_hybrid", "key": x, "t": tview(x, False),
                            "meta": meta(x, "unans_hybrid", ids, "suff"),
                            "docs": [ptext(TASKS[x]["domain"], p) for p in ids]}, ensure_ascii=False) + "\n")
        n_items["unans_hybrid"] += 1
RCFG = {"read_items": "/content/mtrag_read_items.jsonl", "out": OUT, "dtype": DTYPE, "maxlen": MAXLEN,
        "read_new": READ_NEW, "trim_words": TRIM_WORDS, "chunk": CHUNK}
json.dump(RCFG, open("/content/mtrag_read.json", "w"))
print("items per reader:", dict(n_items), "· readers:", READERS or "none (READERS = [])")
FAILED = []
for r in READERS:
    print(f"\n=== {r} ===")
    free_gpu()
    if stream([sys.executable, "/content/mtrag_worker.py", "read", r, "/content/mtrag_read.json"]) != 0:
        FAILED.append(r)
        print(f"!!! {r} failed, see the log above; the remaining readers and the summary continue")
if READERS:
    json.dump({**RCFG, "readers": READERS, "failed": FAILED, "jobs": JOBS + ["unans_hybrid"], "k_ctx": K_CTX,
               "gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{OUT}/gen/config.json", "w"), ensure_ascii=False, indent=2)
'''),
("code", r'''
# ── 12. Final summary (with readers) and the overall config.json ────────────
SUMMARY = summarize(final=True)
json.dump({"notebook": "mtrag.ipynb", "mtrag_rev": MTRAG_REV, "domains": MT_DOMAINS, "n_label_turns": len(LABEL),
           "n_unans_turns": len(UNANS), "retrievers": RETRIEVERS, "jobs": JOBS, "embed_model": EMBED_MODEL,
           "max_doc_len": MAX_DOC_LEN, "bm25": {"k1": BM25_K1, "b": BM25_B, "stemmer": STEM_NAME}, "hyb_w": HYB_W,
           "topk": TOPK, "k_ctx": K_CTX, "judge": JUDGE, "judge_failed": JUDGE_FAILED, "readers": READERS,
           "failed_readers": FAILED,
           "do_gold": DO_GOLD, "do_single": DO_SINGLE, "single_classes": SINGLE_CLASSES, "maxlen": MAXLEN,
           "judge_new": JUDGE_NEW, "read_new": READ_NEW, "trim_words": TRIM_WORDS, "n_boot": N_BOOT, "seed": SEED,
           "dtype": DTYPE, "gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(f"{OUT}/config.json", "w"), ensure_ascii=False, indent=2)
if JUDGE_FAILED or FAILED:
    print("!!! INCOMPLETE:", "judge" if JUDGE_FAILED else "", *FAILED,
          "— 'Run all' again: finished parts are skipped")
print("Next: copy", OUT, "(without index/) from Drive to data/drive/reteco/mtrag/ of the repo.")
'''),
    ]
    return nb(cells, "A100")

KEEPALIVE = open(f"{OUT}/keepalive.js", encoding="utf-8").read()

def guard_cell(name):
    return ("code", f'''
# ── 1b. Compute-unit safeguard ──────────────────────────────────────────────
# At the end of the notebook the runtime disconnects itself, so an idle GPU does not burn compute units.
# If a cell fails and nothing is run for {{IDLE_STOP_MIN}} min, it disconnects as well.
# To turn this off, set AUTO_STOP = False in the settings.
NB = "{name}"
''' + r'''
import threading, time
from IPython import get_ipython
AUTO_STOP = globals().get("AUTO_STOP", True)
IDLE_STOP_MIN = globals().get("IDLE_STOP_MIN", 15)

def stop_runtime(reason):
    print(f"\n[{time.strftime('%H:%M:%S')}] {reason}: disconnecting the runtime to save compute units", flush=True)
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        print("could not disconnect:", e)

_WD = {"t": None}
def _pre_cell(*a, **k):
    if _WD["t"]:
        _WD["t"].cancel(); _WD["t"] = None
def _post_cell(result):
    if AUTO_STOP and getattr(result, "error_in_exec", None) is not None:
        _WD["t"] = threading.Timer(IDLE_STOP_MIN * 60, stop_runtime,
                                   args=(f"error and {IDLE_STOP_MIN} min without activity",))
        _WD["t"].daemon = True; _WD["t"].start()
_ip = get_ipython()
if _ip is not None and not getattr(_ip, "_reteco_guard", False):
    _ip.events.register("pre_run_cell", _pre_cell)
    _ip.events.register("post_run_cell", _post_cell)
    _ip._reteco_guard = True
print("safeguard:", "the runtime disconnects itself at the end or", IDLE_STOP_MIN, "min after an error"
      if AUTO_STOP else "off (AUTO_STOP = False)")
''')

END_CELL = ("code", r'''
# ── End. Everything is on Drive; the runtime disconnects so that no compute units are spent ──
print("done:", time.strftime("%Y-%m-%d %H:%M:%S"), "· log:", globals().get("LOG_PATH", "—"))
if AUTO_STOP:
    stop_runtime("notebook finished")
''')

KEEPALIVE_MD = ("markdown", """
**To keep the session from disconnecting when idle**: after "Run all", open the browser console
(Cmd+Option+J → Console), paste the code below and press Enter. Once a minute it clicks the
resources button while the runtime is connected, and it stops by itself once the notebook has finished
and released the GPU (it does not reconnect, so no compute units are spent). The computer must not
go to sleep, and the tab must stay open.
A log of all output with timestamps is written to `logs/` in the project folder on Drive (`DRIVE`).

```js
""" + KEEPALIVE.strip() + """
```
""")

def save(name, d):
    cells = d["cells"]
    first = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    extra = nb([KEEPALIVE_MD, guard_cell(name), END_CELL])["cells"]
    d["cells"] = cells[:first] + [extra[0], cells[first], extra[1]] + cells[first + 1:] + [extra[2]]
    gpu = {"diver": "T4", "reason_embed": "T4", "rerank": "T4", "analysis": "CPU", "suff": "A100",
           "mtrag": "A100"}.get(name, "L4")
    if gpu == "CPU":
        d["metadata"].pop("accelerator", None); d["metadata"]["colab"].pop("gpuType", None)
    else:
        d["metadata"]["accelerator"] = "GPU"; d["metadata"]["colab"]["gpuType"] = gpu
    json.dump(d, open(f"{OUT}/{name}.ipynb", "w"), ensure_ascii=False, indent=1)

save("diver", embed_nb("AQ-MedAI/Diver-Retriever-4B", "Diver-Retriever-4B",
                       "Model: a reasoning embedder built on Qwen3-Embedding-4B. On RECOR the benchmark "
                       "authors report 0.545 vs. 0.446 for BM25 (with history)."))
save("reason_embed", embed_nb("hanhainebula/reason-embed-qwen3-4b-0928", "reason-embed-qwen3-4b",
                              "Model: ReasonEmbed 4B (arXiv:2510.08252). Not previously evaluated on RECOR; "
                              "stronger than Diver on BRIGHT."))
save("rerank", rerank_nb())
save("generate", generate_nb())
save("judge", judge_nb())
save("suff", suff_nb())
save("query", query_nb())
save("llm_rerank", llm_rerank_nb())
save("analysis", analysis_nb())
save("transfer", transfer_nb())
save("mtrag", mtrag_nb())
print("written:", sorted(os.listdir(OUT)))
