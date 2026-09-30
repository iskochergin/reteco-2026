"""Собирает colab/*.ipynb из одного источника: общие ячейки (Drive, данные,
topics, ndcg10, write_run) одинаковы во всех ноутбуках.

    python3 colab/build.py        # перезаписывает colab/*.ipynb

Ноутбуки правятся ЗДЕСЬ, не в .ipynb — иначе следующая сборка затрёт правку.
"""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))

def nb(cells, gpu="T4"):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": gpu, "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "cells": [{"cell_type": t, "metadata": {}, "source": src.strip("\n").splitlines(True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})}
                      for t, src in cells]}

COMMON_SETUP = r'''
# ── 2. Drive, пакеты, GPU ────────────────────────────────────────────────────
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"   # Xet-транспорт HF бывает зависает; обычный HTTP надёжнее
from google.colab import drive
drive.mount("/content/drive")
!pip -q install -U sentence-transformers pytrec-eval-terrier
import torch, transformers
from packaging.version import Version
assert torch.cuda.is_available(), "Нет GPU: Среда выполнения → Сменить среду выполнения → T4 GPU"
DTYPE_KEY = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} ГБ,",
      "transformers", transformers.__version__)
'''

COMMON_DATA = r'''
# ── 3. Данные RETECO, трек 2 (≈270 МБ, минута) ──────────────────────────────
from huggingface_hub import snapshot_download
snapshot_download("DataScience-UIBK/RETECO-SemEval2027", repo_type="dataset",
                  local_dir="/content/reteco",
                  allow_patterns=["track2_recor/*/documents.jsonl",
                                  "track2_recor/*/benchmark_*.json",
                                  "track2_recor/*/qrels_*.txt"])
'''

COMMON_CODE = r'''
# ── 4. Общий код. Повторяет scripts/retrieve.py и score.py из репозитория ──
import os, re, json, time, glob
import numpy as np, pytrec_eval
from tqdm.auto import tqdm

DATA = "/content/reteco/track2_recor"
DOMAINS = ["drones", "politics", "law", "medicalsciences", "hardware", "economics",
           "psychology", "biology", "sustainable_living", "robotics", "earth_science"]  # от малых к большим
# официальный BM25 «реплика + история», dev, nDCG@10 — для сравнения
BM25_DEV = {"biology": .6065, "drones": .2954, "earth_science": .6264, "economics": .4739,
            "hardware": .3378, "law": .3437, "medicalsciences": .1730, "politics": .3219,
            "psychology": .5726, "robotics": .5123, "sustainable_living": .5531}
BM25_MACRO = {"dev": 0.4379, "train": 0.4539}
RUNS, QUERIES = f"{DRIVE}/runs", f"{DRIVE}/queries"
os.makedirs(RUNS, exist_ok=True); os.makedirs(QUERIES, exist_ok=True)

def load_corpus(dom):
    ids, texts = [], []
    for ln in open(f"{DATA}/{dom}/documents.jsonl", encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln); ids.append(d["doc_id"]); texts.append(d["content"])
    return ids, texts

def topics(dom, split, qset):
    """Реплики без золота пропускаются; история приклеивается как у организаторов.
    qset: "query", "hist" или имя папки в Drive/reteco/queries (варианты rewrite.py)."""
    given = None
    if qset not in ("query", "hist"):
        with open(f"{QUERIES}/{qset}/{split}.jsonl", encoding="utf-8") as f:
            # ключ (домен, topic_id): в доменах BRIGHT topic_id повторяются между доменами
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
    """run: {qid: {doc: score}}. Реплика без прогона получает 0, как на лидерборде."""
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
    """Сначала во временный файл, потом переименование: оборванная запись не
    оставляет полуфайл, который перезапуск принял бы за готовый."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for q, docs in run.items():
            for r, (d, s) in enumerate(sorted(docs.items(), key=lambda x: -x[1]), 1):
                f.write(f"{q}\tQ0\t{d}\t{r}\t{s:.6f}\t{tag}\n")
    os.replace(path + ".tmp", path)
'''

LLM_SETUP = r'''
# ── 2. Drive, vLLM, GPU ─────────────────────────────────────────────────────
import os, sys, subprocess
os.environ["HF_HUB_DISABLE_XET"] = "1"
from google.colab import drive
drive.mount("/content/drive")
!pip -q install vllm openai pytrec-eval-terrier 2>&1 | tail -2
import torch

# vllm ставит свой torch, а torchvision/torchaudio из Colab остаются под другую
# CUDA, и import vllm падает. Лечение из occ-rag-bfcl: torchaudio снести,
# torchvision поставить под CUDA нового torch. После правки нужен перезапуск.
def _imports_ok(mod):
    try:
        __import__(mod); return True
    except RuntimeError:
        return False       # стоит, но собран под другую CUDA
    except ImportError:
        return True        # не стоит — и конфликта нет
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
    raise RuntimeError("Зависимости исправлены. Среда выполнения → Перезапустить сеанс, "
                       "потом снова «Выполнить все».")
if globals().get("NEED_GPU", True):
    assert torch.cuda.is_available(), "Нет GPU: Среда выполнения → Сменить среду выполнения → A100 или L4"
MEM = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
DTYPE = "bfloat16" if MEM and torch.cuda.is_bf16_supported() else "half"   # T4 bf16 не умеет
BIG = MEM >= 39                                                             # A100 40/80 ГБ
print(torch.cuda.get_device_name(0) if MEM else "без GPU", f"{MEM:.0f} ГБ, {DTYPE}, torch", torch.__version__)

def load_llm(model, maxlen, util=0.88):
    """На время конструктора — настоящие stdout/stderr: в ядре Jupyter vLLM падает
    на sys.stdout.fileno() ещё до загрузки весов (occ-rag-bfcl)."""
    from vllm import LLM
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        return LLM(model=model, dtype=DTYPE, max_model_len=maxlen, gpu_memory_utilization=util, seed=0)
    finally:
        sys.stdout, sys.stderr = out, err

def chat(tok, msgs):
    """Промпт по шаблону модели; у Qwen3 размышления выключены (быстро, детерминированно)."""
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
'''

# ============================================================== embed ========
def embed_nb(model, title, note):
    cells = [
("markdown", f"""
# RETECO · {title}

Считает векторы всех 507 тыс. документов трека 2 моделью `{model}` и по ним ищет
топ-100 для каждой реплики. Всё сохраняется на Google Drive в `reteco/`.

**Как запускать.** Среда выполнения → Сменить среду → **T4 GPU**. Потом
Среда выполнения → **Выполнить все**. Разрешить доступ к Drive. Уйти.

**Если сессия оборвалась** — снова «Выполнить все». Готовые домены лежат на Drive
и пропускаются, продолжится с места обрыва. Ничего не считается дважды.

**Сколько.** На T4 оценка 2–4 часа на весь корпус; точную скорость ноутбук
напечатает после первого домена. На A100 в 5–8 раз быстрее. На Drive нужно
≈2.6 ГБ свободного места.

**Что получится на Drive.**
- `reteco/index/<модель>/<домен>.npy` — векторы документов (считаются один раз)
- `reteco/runs/dense_<модель>_<запрос>_<сплит>/<домен>/run.trec` — прогоны

**Потом.** Скачать папку `reteco/runs` с Drive в `runs/` проекта и сказать Claude.

**Новые варианты запроса** (из `scripts/rewrite.py`): положить папку проекта
`runs/rewrites` на Drive как `reteco/queries` и снова «Выполнить все» — векторы
корпуса уже готовы, считаются только запросы, минуты.

{note}
"""),
("code", f'''
# ── 1. Настройки ─────────────────────────────────────────────────────────────
MODEL = "{model}"
BATCH = 32            # «CUDA out of memory» → 16, потом 8
MAX_DOC_LEN = 512     # токенов на документ; медиана документа 20 слов
MAX_Q_LEN = 1024      # запрос с историей бывает до ~600 токенов
SPLITS = ["dev", "train"]
DRIVE = "/content/drive/MyDrive/reteco"
'''),
("code", COMMON_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. Модель. Промпты запроса и документа — из конфига самой модели ──────────
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL, device="cuda", model_kwargs={DTYPE_KEY: torch.float16},
                            trust_remote_code=True)
assert next(model.parameters()).dtype == torch.float16, "модель загрузилась не в fp16"
Q_PROMPT = (model.prompts or {}).get("query", "")
D_PROMPT = (model.prompts or {}).get("document", "")
print("промпт запроса:  ", repr(Q_PROMPT))
print("промпт документа:", repr(D_PROMPT))
SLUG = re.sub(r"[^A-Za-z0-9._-]", "_", MODEL)
IDX = f"{DRIVE}/index/{SLUG}"
os.makedirs(IDX, exist_ok=True)

def encode(texts, prompt, max_len):
    model.max_seq_length = max_len
    e = model.encode(texts, prompt=prompt, batch_size=BATCH, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True)
    assert not np.isnan(e).any(), "NaN в векторах: fp16 переполнился; нужен A100 и bfloat16"
    return e

def search(emb, ids, qids, qtexts, topk=100):
    Q = torch.from_numpy(encode(qtexts, Q_PROMPT, MAX_Q_LEN)).cuda().half()
    E = torch.from_numpy(emb).cuda()
    run = {}
    for i in range(0, len(qids), 64):
        v, ix = (Q[i:i + 64] @ E.T).float().topk(min(topk, E.shape[0]), dim=1)
        for j, q in enumerate(qids[i:i + 64]):
            run[q] = {ids[k]: float(s) for k, s in zip(ix[j].tolist(), v[j].tolist())}
    del E, Q; torch.cuda.empty_cache()
    return run

def run_dir(qset, split):
    return f"{RUNS}/dense_{SLUG}_{qset}_{split}"
'''),
("code", r'''
# ── 6. Векторы корпуса → Drive. После каждого домена — проверка на dev ─────
sizes = {d: sum(1 for _ in open(f"{DATA}/{d}/documents.jsonl")) for d in DOMAINS}
rate = None
for dom in DOMAINS:
    out = f"{IDX}/{dom}.npy"
    if os.path.exists(out):
        print(f"[есть] {dom}"); continue
    ids, docs = load_corpus(dom)
    t = time.time()
    emb = encode(docs, D_PROMPT, MAX_DOC_LEN).astype(np.float16)
    sec = time.time() - t
    json.dump(ids, open(f"{IDX}/{dom}.ids.json", "w"))
    with open(out + ".tmp", "wb") as f:
        np.save(f, emb)
    os.replace(out + ".tmp", out)
    rate = len(docs) / sec
    left = sum(sizes[d] for d in DOMAINS if not os.path.exists(f"{IDX}/{d}.npy"))
    # проверка: плотный поиск «реплика + история» на dev против официального BM25
    qids, qtexts = topics(dom, "dev", "hist")
    run = search(emb, ids, qids, qtexts)
    write_run(f"{run_dir('hist', 'dev')}/{dom}/run.trec", run, "dense_hist")
    print(f"{dom}: {len(docs)} док за {sec/60:.1f} мин ({rate:.0f} док/с), осталось ≈{left/rate/60:.0f} мин · "
          f"nDCG@10 dev {ndcg10(run, dom, 'dev'):.4f} против BM25 {BM25_DEV[dom]:.4f}")
print("векторы корпуса готовы:", IDX)
'''),
("code", r'''
# ── 7. Прогоны: все наборы запросов × сплиты → Drive/reteco/runs ────────────
# Домен снаружи: векторы домена читаются с Drive один раз на все наборы запросов.
extra = sorted(os.path.basename(p) for p in glob.glob(f"{QUERIES}/*")
               if os.path.isdir(p) and os.path.basename(p) != "raw")
JOBS = [(q, s) for q in ["hist", "query"] + extra for s in SPLITS
        if q in ("query", "hist") or os.path.exists(f"{QUERIES}/{q}/{s}.jsonl")]
print("наборы запросов × сплиты:", JOBS)
for dom in DOMAINS:
    todo = [(q, s) for q, s in JOBS if not os.path.exists(f"{run_dir(q, s)}/{dom}/run.trec")]
    if not todo:
        continue
    ids, emb = json.load(open(f"{IDX}/{dom}.ids.json")), np.load(f"{IDX}/{dom}.npy")
    for qset, split in todo:
        qids, qtexts = topics(dom, split, qset)
        write_run(f"{run_dir(qset, split)}/{dom}/run.trec", search(emb, ids, qids, qtexts), f"dense_{qset}")
    print(f"{dom}: {len(todo)} прогонов")
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
# ── 8. Итог ─────────────────────────────────────────────────────────────────
print(f"{'прогон':<60}{'nDCG@10':>9}{'BM25':>8}{'дельта':>9}")
for name, m, b in summary:
    print(f"{name:<60}{m:>9.4f}{b:>8.4f}{m - b:>+9.4f}")
print("\nДальше: скачать", RUNS, "в runs/ проекта и сказать Claude.")
'''),
    ]
    return nb(cells)

# ============================================================= rerank ========
def rerank_nb():
    cells = [
("markdown", r"""
# RETECO · реранк топ-100 (Qwen3-Reranker-4B)

Берёт готовый список топ-100 на каждую реплику (прогон первой стадии) и
переставляет его: модель читает пару «запрос + документ» целиком и оценивает,
отвечает ли документ на запрос. `doc_id` модели не показывается — только текст.

**Когда запускать.** После того как выбрана первая стадия поиска. Прогон первой
стадии должен лежать на Drive в `reteco/runs/<имя>/<домен>/run.trec` —
плотные прогоны туда кладут ноутбуки `diver` и `reason_embed`, локальные
(BM25, гибриды) надо загрузить руками.

**Как запускать.** T4 GPU → в ячейке 1 указать `FIRST_STAGE` и `QUERY_SET` →
«Выполнить все». Оборвалось — «Выполнить все» ещё раз, готовые домены
пропускаются.

**Сколько.** 858 реплик × 100 документов = 86 тыс. пар. С короткими запросами
(`qd`) на T4 оценка 1–2 часа, с запросом + историей (`hist`) в 3–4 раза дольше.
Точная скорость — после первого домена.

**Что получится.** `reteco/runs/rerank_<первая стадия>_<запрос>/<домен>/run.trec`,
балл = вероятность «да». Смешивание с первой стадией (0.6 реранк + 0.4 поиск)
делается локально: `scripts/fuse.py --mode score --weights 0.6 0.4`.
"""),
("code", r'''
# ── 1. Настройки ─────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-Reranker-4B"
FIRST_STAGE = "bm25_fuse3_dev"   # папка в Drive/reteco/runs; bm25_fuse3 — лучшая первая стадия на 29.09, dev 0.5496
QUERY_SET = "qd"      # что видит реранкер: "qd" — переписанный самостоятельный вопрос (быстро);
                      # "qd_hist" — с историей (точнее?, в 3-4 раза дольше); "query", "hist" — как у организаторов
SPLIT = "dev"
TOPK = 100            # сколько кандидатов переставлять
BATCH = 8             # «CUDA out of memory» → 4
MAX_LEN = 1024        # токенов на пару; длинные документы обрезаются с конца
TASK = "Given a web search query, retrieve relevant passages that answer the query"
DRIVE = "/content/drive/MyDrive/reteco"
'''),
("code", COMMON_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. Модель. Формат — дословно из README Qwen3-Reranker ────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
lm = AutoModelForCausalLM.from_pretrained(MODEL, **{DTYPE_KEY: torch.float16}).cuda().eval()
YES, NO = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PRE, SUF = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)

@torch.no_grad()
def score_pairs(pairs):
    """Вероятность «yes» для каждой пары (запрос, документ). Пары сортируются по
    длине, чтобы в пачке было меньше паддинга; порядок потом восстанавливается."""
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
        s = torch.stack([logits[:, NO], logits[:, YES]], 1).float().log_softmax(1)[:, 1].exp()
        for i, v in zip(idx, s.tolist()):
            out[i] = v
    return out
'''),
("code", r'''
# ── 6. Реранк по доменам → Drive ────────────────────────────────────────────
src = f"{RUNS}/{FIRST_STAGE}"
assert os.path.isdir(src), f"нет прогона первой стадии {src}"
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
        t = time.time()
        scores = score_pairs([(qtext[q], text[d]) for q, d in pairs])
        run = {}
        for (q, d), s in zip(pairs, scores):
            run.setdefault(q, {})[d] = s
        write_run(path, run, "rerank")
        print(f"{dom}: {len(pairs)} пар за {(time.time() - t)/60:.1f} мин "
              f"({len(pairs)/(time.time() - t):.0f} пар/с)")
    a, b = ndcg10(first, dom, SPLIT), ndcg10(read_run(path), dom, SPLIT)
    rows.append((dom, a, b))
    print(f"  {dom:<20} первая стадия {a:.4f} → реранк {b:.4f}  ({b - a:+.4f})")
json.dump({"model": MODEL, "first_stage": FIRST_STAGE, "query_set": QUERY_SET, "split": SPLIT,
           "topk": TOPK, "max_len": MAX_LEN, "task": TASK, "dtype": "float16",
           "gpu": torch.cuda.get_device_name(0), "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(f"{dst}/config.json", "w"), ensure_ascii=False, indent=2)
A, B = sum(r[1] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)
print(f"\nМАКРО  первая стадия {A:.4f} → реранк {B:.4f}  ({B - A:+.4f});  BM25+история {BM25_MACRO[SPLIT]:.4f}")
print("Дальше: скачать", dst, "в runs/ проекта и сказать Claude.")
'''),
    ]
    return nb(cells)

# =========================================================== generate ========
GEN_WORKER = r'''
"""Ответы одного ридера на все задания (2b и 2c по первым стадиям). Запускается
из generate.ipynb отдельным процессом на каждый ридер: так vLLM точно отдаёт
память GPU перед следующей моделью. Модуль же импортирует ячейка итогов."""
import json, math, os, re, sys, time

OCC_STOP = [151643, 151645, 151683]   # <|endoftext|> <|im_end|> <|answer_end|> — generation_config OCC
CONTROL_SYSTEM = (
    "You are a helpful assistant in a multi-turn conversation. Answer the user's latest message "
    "using only the numbered documents attached to it.\n"
    "Reply in this format:\n"
    "Line 1: ANSWERABLE if the documents contain enough information to answer, otherwise UNANSWERABLE.\n"
    "Next lines: if ANSWERABLE, the answer in 2-5 sentences in a natural conversational tone; "
    "if UNANSWERABLE, say briefly what information is missing.")


def load_turns(data, domains, split):
    """Реплики с золотом. history — прошлые реплики диалога (вопрос, ответ): это и есть
    conversation_history. reference — золотой ответ ТЕКУЩЕЙ реплики: только для аудита
    и судьи, в промпт ридера не идёт. Золото — только то, что есть в qrels: 3 золотых id
    в politics отсутствуют в корпусе, организаторы выкинули их из qrels."""
    out = []
    for dom in domains:
        qrels = {tuple(l.split()[0:3:2]) for l in open(f"{data}/{dom}/qrels_{split}.txt") if l.strip()}
        for c in json.load(open(f"{data}/{dom}/benchmark_{split}.json", encoding="utf-8")):
            prev = []
            for t in c["turns"]:
                qid = f"{c['id']}_turn_{t['turn_id']}"
                gold = [d for d in t.get("gold_doc_ids") or t.get("supporting_doc_ids") or [] if (qid, d) in qrels]
                if gold:
                    out.append({"domain": dom, "topic_id": qid,
                                "turn_id": t["turn_id"], "query": t["query"], "history": list(prev),
                                "gold": list(gold), "reference": t["answer"]})
                prev.append((t["query"], t["answer"]))
    return out


def read_top(path, k):
    run = {}
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) == 6:
            run.setdefault(p[0], []).append((int(p[3]), p[2]))
    return {q: [d for _, d in sorted(v)[:k]] for q, v in run.items()}


def contexts(C, turns, split, job):
    """"2b" → золотые документы; "2c-<прогон>" → топ-K из Drive/reteco/runs/<прогон>_<split>."""
    if job == "2b":
        return {(t["domain"], t["topic_id"]): t["gold"] for t in turns}
    tops = {dom: read_top(f"{C['drive']}/runs/{job[3:]}_{split}/{dom}/run.trec", C["k_ctx"])
            for dom in C["domains"]}
    return {(t["domain"], t["topic_id"]): tops[t["domain"]].get(t["topic_id"], []) for t in turns}


def sufficiency(gold, ctx):
    """Сколько золота в контексте и метка: all — всё золото, что влезает в K; none — ни одного."""
    n = len(set(gold) & set(ctx))
    full = min(len(gold), len(ctx)) if ctx else len(gold)
    return n, ("none" if n == 0 else "all" if n >= full else "part")


def corpus_texts(data, need):
    """need: {домен: {doc_id}} → {(домен, doc_id): текст}."""
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
    """Шаблон OCC сам кладёт документы в последнюю реплику user; генерация начинается
    с <|query_analysis_start|>, статус — <|status_start|>(UN)ANSWERABLE<|status_end|>."""
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
        i = raw.find(self.marker)
        return None if i < 0 else raw[:i + len(self.marker)]


class Control:
    """Открытая модель-контроль: тот же вход, статус первой строкой."""
    stop, skip_special = None, True

    def __init__(self, tok):
        self.tok = tok

    def prompt(self, t, docs):
        numbered = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1))
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
        return ""          # статус — самый первый токен ответа


def p_status(tok, lp):
    """Вероятность ANSWERABLE на позиции статуса: сумма вероятностей токенов-префиксов
    «ANSWERABLE» против «UNANSWERABLE» среди топ-20 (ANS… против UN…). mass — сколько
    вероятности пришлось на эти два варианта вообще."""
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
    llm = LLM(model=reader_id, dtype=C["dtype"], max_model_len=C["maxlen"],
              gpu_memory_utilization=0.88, seed=0)
    tok = llm.get_tokenizer()
    R = (OCC if "OCC" in reader_id else Control)(tok)
    # OCC: только greedy, без repetition_penalty — иначе ломается структурный формат
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
                if ntok(p) > limit:      # не влезает — режем каждый документ, золото не выкидываем
                    p, trimmed = R.prompt(t, [" ".join(d.split()[:C["trim_words"]]) for d in docs]), True
                if ntok(p) > limit:
                    skipped += 1
                    continue
                prompts.append(p)
                keep.append((t, ids, trimmed))
            print(f"{short} · {job} · {split}: сделать {len(prompts)}, готово {len(done)}, "
                  f"не влезло {skipped}", flush=True)
            t0 = time.time()
            with open(path, "a", encoding="utf-8") as f:
                for s in range(0, len(prompts), C["chunk"]):
                    outs = llm.generate(prompts[s:s + C["chunk"]], sp, use_tqdm=False)
                    raws = [o.outputs[0].text for o in outs]
                    # второй проход: промпт + ответ до места статуса, один токен, топ-20 логпробов
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
                            "split": split, "reader": reader_id, "mode": job[:2],
                            "first_stage": job[3:] or None, "context_ids": ids, "n_gold": len(t["gold"]),
                            "gold_in_ctx": n_in, "suff": suff, "status": status, "answer": answer,
                            "p_answerable": p, "p_mass": mass, "finish": o.outputs[0].finish_reason,
                            "trimmed": trimmed, "raw": raw}, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"  {s + len(outs)}/{len(prompts)} · {(time.time() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''

def generate_nb():
    cells = [
("markdown", r"""
# RETECO · генерация ответов 2b и 2c (главный эксперимент статьи)

Два ридера отвечают на каждую реплику dev по одному и тому же входу: история
диалога + текущий вопрос + документы.
- **2b** — документы золотые: контекста заведомо хватает.
- **2c** — документы = топ-5 поиска. Первые стадии разной силы (BM25 по реплике
  .18, BM25 + история .44, лучший BM25 .55) — «доза и эффект»: чем слабее поиск,
  тем чаще в контексте нет ответа.

Ридеры: `occ-ai/OCC-RAG-1.7B` (отвечает или отказывается, статус
ANSWERABLE/UNANSWERABLE) и открытая модель-контроль (A100: Qwen3-8B, иначе
Qwen3-4B-Instruct-2507) с тем же форматом статуса. Для каждого ответа
сохраняется вероятность ANSWERABLE на позиции статуса — непрерывная уверенность.
Метка достаточности — сколько золотых документов попало в контекст (из qrels).

**Как запускать.** Среда → **A100** (лучше) или **L4**. На Drive должна лежать
папка `reteco/runs` с прогонами `bm25_query_dev`, `bm25_hist_dev`,
`bm25_fuse3_dev` (из `colab/drive/reteco` проекта). «Выполнить все». Если ячейка 2
попросит перезапуск — Перезапустить сеанс и снова «Выполнить все».
Оборвалось — «Выполнить все» ещё раз, готовые ответы пропускаются.

**Сколько.** 858 реплик × 4 задания × 2 ридера ≈ 7 тыс. ответов. На A100
оценка: OCC 15–25 мин, Qwen3-8B 30–60 мин; на L4 в 2–3 раза дольше.

**Что получится.**
- `reteco/gen/<ридер>_<2b|2c-первая_стадия>_dev.jsonl` — ответы, статус,
  p_answerable, метка достаточности, id документов контекста
- `reteco/audit/sample.csv` — 100 реплик для ручной проверки метки достаточности
  (ключ отдельно, `sample_key.csv`)

**Потом.** `judge.ipynb` оценивает ответы; скачать `reteco/gen` и `reteco/audit`
в проект и сказать Claude.
"""),
("code", r'''
# ── 1. Настройки ─────────────────────────────────────────────────────────────
READERS = ["occ-ai/OCC-RAG-1.7B", "control"]   # control: A100 → Qwen/Qwen3-8B, иначе Qwen/Qwen3-4B-Instruct-2507
FIRST_STAGES = ["bm25_query", "bm25_hist", "bm25_fuse3"]   # для 2c, от слабой к сильной; папки runs/<имя>_<сплит>
DO_2B = True
SPLITS = ["dev"]
K_CTX = 5             # документов в контексте 2c
MAX_NEW = 1200        # OCC пишет разбор перед ответом, ему нужно место
MAXLEN = 16384        # золота бывает до 12 документов
TRIM_WORDS = 300      # если промпт не влез — каждый документ режется до стольких слов
CHUNK = 256           # сколько ответов между записями на Drive
DRIVE = "/content/drive/MyDrive/reteco"
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", r'''
# ── 6. Генерация: каждый ридер — отдельный процесс ───────────────────────────
CONTROL = "Qwen/Qwen3-8B" if BIG else "Qwen/Qwen3-4B-Instruct-2507"
readers = [CONTROL if r == "control" else r for r in READERS]
jobs = (["2b"] if DO_2B else []) + [f"2c-{fs}" for fs in FIRST_STAGES]
for fs in FIRST_STAGES:
    for sp in SPLITS:
        assert os.path.isdir(f"{RUNS}/{fs}_{sp}"), f"нет прогона {RUNS}/{fs}_{sp} — загрузить на Drive"
cfg = {"drive": DRIVE, "data": DATA, "domains": DOMAINS, "splits": SPLITS, "jobs": jobs,
       "k_ctx": K_CTX, "max_new": MAX_NEW, "maxlen": MAXLEN, "trim_words": TRIM_WORDS,
       "chunk": CHUNK, "dtype": DTYPE}
json.dump(cfg, open("/content/gen_cfg.json", "w"))

def stream(cmd):
    """Вывод дочернего процесса — в ячейку построчно."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "VLLM_LOGGING_LEVEL": "WARNING"})
    for line in p.stdout:
        print(line, end="")
    return p.wait()

for r in readers:
    print(f"\n=== {r} ===")
    assert stream([sys.executable, "/content/gen_worker.py", r, "/content/gen_cfg.json"]) == 0, f"{r} упал, лог выше"
json.dump({**cfg, "readers": readers, "gpu": torch.cuda.get_device_name(0),
           "date": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(f"{DRIVE}/gen/config.json", "w"), ensure_ascii=False, indent=2)
'''),
("code", r'''
# ── 7. Итог: доля отказов по достаточности контекста и AUROC уверенности ─────
sys.path.insert(0, "/content")
import gen_worker as G
from sklearn.metrics import roc_auc_score
print(f"{'файл':<44}{'n':>5}{'отказ':>7}{'all':>6}{'part':>6}{'none':>6}{'AUROC':>7}")
for path in sorted(glob.glob(f"{DRIVE}/gen/*.jsonl")):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    ref = lambda xs: sum(r["status"].startswith("UN") for r in xs) / max(1, len(xs))
    by = {s: [r for r in rs if r["suff"] == s] for s in ("all", "part", "none")}
    # AUROC: насколько p_answerable отделяет контекст с золотом (all) от контекста без (none)
    ab = [r for r in rs if r["suff"] in ("all", "none") and r["p_answerable"] is not None]
    auc = (roc_auc_score([r["suff"] == "all" for r in ab], [r["p_answerable"] for r in ab])
           if len({r["suff"] for r in ab}) == 2 else float("nan"))
    cells = "".join(f"{ref(by[s]):>6.2f}" if by[s] else f"{'—':>6}" for s in ("all", "part", "none"))
    print(f"{os.path.basename(path)[:-6]:<44}{len(rs):>5}{ref(rs):>7.2f}{cells}{auc:>7.3f}")
print("\nотказ — доля UNANSWERABLE; all/part/none — она же по метке достаточности. В 2b все all:"
      "\nотказ там = ложный отказ. Статус не распознан:",
      sum(1 for p in glob.glob(f"{DRIVE}/gen/*.jsonl") for l in open(p) if not json.loads(l)["status"]))
'''),
("code", r'''
# ── 8. Выборка для ручной проверки метки достаточности → Drive/reteco/audit ──
# 50 реплик, где в топ-5 нет ни одного золотого документа, и 50, где есть всё
# влезающее золото. В sample.csv метки нет (слепая проверка): вопрос, история,
# эталонный ответ и топ-5. Заполнить колонку answer_in_top5: да / нет / частично.
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
print(f"{len(pick)} реплик → {DRIVE}/audit/sample.csv; ключ — sample_key.csv")
'''),
    ]
    return nb(cells, "A100")

# ============================================================== judge ========
def judge_nb():
    cells = [
("markdown", r"""
# RETECO · судья ответов 2b и 2c

Оценивает каждый ответ из `reteco/gen` по пяти осям организаторов (1–5):
correctness, completeness, relevance, conversational coherence, faithfulness.
Плюс тип ответа (ответил / отказался / частично) и «есть ли утверждения без
опоры в документах». Судья видит историю, вопрос, документы, которые видел
ридер, эталонный ответ (`turns[].answer` — для оценки его брать можно) и ответ.

Организаторы судят GPT-4o с промптами, которые откроют только в январе. Здесь —
наша версия судьи. **Какой моделью судить — решает Иван** (docs/NOTES.md).
Два бэкенда:
- `BACKEND = "vllm"` — открытая модель здесь же (A100: Qwen3-14B, L4: Qwen3-8B,
  T4: Qwen3-8B-AWQ). Бесплатно, воспроизводимо.
- `BACKEND = "api"` — OpenAI-совместимый API, например `gpt-4o`, как у
  организаторов. Ключ — в секретах Colab (значок ключа слева): `OPENAI_API_KEY`,
  при другом провайдере ещё `OPENAI_BASE_URL`. GPU не нужен.

Судья обязан быть другой моделью, чем ридер: файлы того же ридера пропускаются.

**Как запускать.** После `generate.ipynb`. Выбрать бэкенд в ячейке 1 →
«Выполнить все». Оборвалось — ещё раз, готовые оценки пропускаются.

**Сколько.** ≈7 тыс. ответов. vLLM на A100 — 30–60 мин; API — зависит от
лимитов ключа, GPT-4o порядка $15–25 за всё.

**Что получится.** `reteco/judge/<судья>/<файл генерации>.jsonl` и сводка
`reteco/judge/<судья>/summary.csv`.
"""),
("code", r'''
# ── 1. Настройки ─────────────────────────────────────────────────────────────
BACKEND = "vllm"      # "vllm" или "api"
JUDGE = "auto"        # vllm: auto = по GPU; api: имя модели, например "gpt-4o"
GEN_GLOB = "*_dev.jsonl"   # какие файлы из Drive/reteco/gen судить
MAXLEN = 16384
CHUNK = 256           # оценок между записями на Drive
API_WORKERS = 8       # параллельных запросов к API
DRIVE = "/content/drive/MyDrive/reteco"
NEED_GPU = BACKEND == "vllm"
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", "%%writefile /content/gen_worker.py\n" + GEN_WORKER.strip("\n")),
("code", r'''
# ── 6. Промпт судьи и разбор ответа ─────────────────────────────────────────
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
- faithfulness: every claim is supported by the provided documents; a refusal that makes no claims gets 5.
Also classify:
- response_type: "answered" (a substantive answer), "refused" (declines or says the information is not available), or "partial" (answers part and says the rest is not available).
- unsupported_claims: true if the response states any fact not supported by the provided documents.

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
# ── 7. Бэкенд: vLLM здесь же или API по ключу ───────────────────────────────
if BACKEND == "vllm":
    from vllm import SamplingParams
    if JUDGE == "auto":
        JUDGE = "Qwen/Qwen3-14B" if BIG else "Qwen/Qwen3-8B" if MEM >= 22 else "Qwen/Qwen3-8B-AWQ"
    llm = load_llm(JUDGE, MAXLEN)
    tok = llm.get_tokenizer()
    SP = SamplingParams(temperature=0.0, max_tokens=600)
    def run_batch(msg_lists):
        outs = llm.generate([chat(tok, m) for m in msg_lists], SP, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
else:
    from openai import OpenAI
    from google.colab import userdata
    from concurrent.futures import ThreadPoolExecutor
    assert JUDGE != "auto", "для api указать модель в JUDGE, например gpt-4o"
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
print("судья:", JUDGE, "→", OUTD)
'''),
("code", r'''
# ── 8. Оценка всех файлов генерации → Drive ─────────────────────────────────
turns = {}
for path in sorted(glob.glob(f"{DRIVE}/gen/{GEN_GLOB}")):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not recs:
        continue
    name = os.path.basename(path)
    if recs[0]["reader"].split("/")[-1] == SHORT:
        print(f"[пропуск] {name}: ридер и судья — одна модель"); continue
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
            msgs = [judge_msgs(turns[split][(r["domain"], r["topic_id"])],
                               [text[(r["domain"], d)] for d in r["context_ids"]], r) for r in part]
            raws = run_batch(msgs)
            bad = [i for i, x in enumerate(raws) if parse_judge(x) is None]
            if bad:          # одна повторная попытка с напоминанием про JSON
                again = run_batch([m[:-1] + [{"role": "user", "content": m[-1]["content"]
                                   + "\n\nReturn ONLY the JSON object, nothing else."}] for m in (msgs[i] for i in bad)])
                for i, x in zip(bad, again):
                    raws[i] = x
            for r, raw in zip(part, raws):
                j = parse_judge(raw)
                f.write(json.dumps({"domain": r["domain"], "topic_id": r["topic_id"], "split": split,
                                    "reader": r["reader"], "mode": r["mode"], "first_stage": r["first_stage"],
                                    "suff": r["suff"], "status": r["status"], "judge": JUDGE,
                                    "ok": j is not None, **(j or {}), "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {name}: {s + len(part)}/{len(todo)} · {(time.time() - t0) / 60:.1f} мин")
    print(f"{name}: готово")
'''),
("code", r'''
# ── 9. Сводка: средние по осям, отказы и выдумки; 2c — по достаточности ─────
import csv
rows = []
for path in sorted(glob.glob(f"{OUTD}/*.jsonl")):
    rs = [json.loads(l) for l in open(path, encoding="utf-8")]
    good = [r for r in rs if r["ok"]]
    for suff in ("все", "all", "part", "none"):
        g = good if suff == "все" else [r for r in good if r["suff"] == suff]
        if not g or (suff != "все" and rs[0]["mode"] == "2b"):
            continue
        n = len(g)
        rows.append({"file": os.path.basename(path)[:-6], "suff": suff, "n": n,
                     **{a: round(sum(r[a] for r in g) / n, 3) for a in AXES},
                     "refused": round(sum(r["response_type"] == "refused" for r in g) / n, 3),
                     "unsupported": round(sum(r["unsupported_claims"] for r in g) / n, 3),
                     "parse_fail": sum(not r["ok"] for r in rs) if suff == "все" else ""})
assert rows, f"в {OUTD} нет разобранных оценок — смотреть поле raw в файлах"
with open(f"{OUTD}/summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'файл':<42}{'дост.':>6}{'n':>5}" + "".join(f"{a[:5]:>7}" for a in AXES) + f"{'отказ':>7}{'выдум':>7}")
for r in rows:
    print(f"{r['file']:<42}{r['suff']:>6}{r['n']:>5}" + "".join(f"{r[a]:>7.2f}" for a in AXES)
          + f"{r['refused']:>7.2f}{r['unsupported']:>7.2f}")
print("\nвыдум — доля ответов с утверждениями без опоры в документах. Сводка:", f"{OUTD}/summary.csv")
'''),
    ]
    return nb(cells, "A100")

# ============================================================== query ========
def query_nb():
    cells = [
("markdown", r"""
# RETECO · блок «Запрос» на открытой LLM + BM25 кита

Достраивает варианты запроса, которых не было в прогоне Claude, и расширение с
обратной связью, как у DIVER. Всё ищется **официальным BM25 кита** (gensim +
анализатор Lucene), поэтому числа прямо сравнимы с таблицей: первая ячейка поиска
проверяет, что BM25 + история даёт на dev ровно 0.4379 по доменам.

Варианты (всё приклеивается к «реплика + история», как q2d у Claude):
- `win1`, `win2` — только последние 1–2 пары истории вместо всей (без LLM)
- `stepback` — более общий вопрос над исходным
- `subq` — 2–3 подвопроса; каждым ищется отдельно, списки сливаются RRF
- `re` — последний ответ ассистента переписан самодостаточным
- `q2d_open` — гипотетический абзац открытой модели (тот же q2d, что у Claude)
- `kw` — 30 ключевых слов из этого абзаца, которых нет в запросе
- `qexp1`, `qexp2` — расширение с обратной связью (DIVER QExpand): модель видит
  топ-5 найденного и переписывает абзац; в запросе остаётся реплика + история +
  последнее расширение

**Как запускать.** Среда → **A100** (Qwen3-8B) или **L4** (Qwen3-4B-Instruct-2507).
«Выполнить все». Если ячейка 2 попросит перезапуск — Перезапустить сеанс и снова
«Выполнить все». Оборвалось — ещё раз: ответы LLM и готовые прогоны на Drive.

**Сколько.** ≈3 тыс. реплик dev + train. LLM ≈17 тыс. коротких генераций —
15–30 мин на A100; BM25: индекс 10–15 мин, поиск ещё 10–20.

**Что получится.**
- `reteco/queries/<вариант>/<сплит>.jsonl` — тексты запросов (их потом подхватят
  `diver` и `reason_embed`: там они считаются за минуты)
- `reteco/runs/bm25_<вариант>_<сплит>/` — прогоны
- `reteco/qgen/<модель>/` — сырые ответы LLM (кэш)

**Потом.** Скачать `reteco/runs` и `reteco/queries` в проект, сказать Claude.
Варианты выбираются по train, dev — только отчёт.
"""),
("code", r'''
# ── 1. Настройки ─────────────────────────────────────────────────────────────
LLM_MODEL = "auto"    # A100 → Qwen/Qwen3-8B, иначе Qwen/Qwen3-4B-Instruct-2507
SPLITS = ["dev", "train"]
FEEDBACK_K = 5        # сколько найденных документов видит LLM в QExpand
DOC_WORDS = 300       # документ в промпте обрезается до стольких слов
N_KW = 30             # ключевых слов в варианте kw
MAXLEN = 8192
MAX_NEW = 256
TOPK = 100
DRIVE = "/content/drive/MyDrive/reteco"
'''),
("code", LLM_SETUP),
("code", r'''
# ── 2b. Java 21, pyserini (только анализатор Lucene), gensim, кит организаторов ──
# pyserini ставится без зависимостей: от него нужен только анализатор, а его
# полные зависимости (torch, transformers) сломали бы vLLM.
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
# ── 5. LLM (грузится до Java: vLLM запускает свой процесс, JVM ему не нужна) ──
from vllm import SamplingParams
if LLM_MODEL == "auto":
    LLM_MODEL = "Qwen/Qwen3-8B" if BIG else "Qwen/Qwen3-4B-Instruct-2507"
llm = load_llm(LLM_MODEL, MAXLEN)
tok = llm.get_tokenizer()
SP = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
SHORT = LLM_MODEL.split("/")[-1]
QGEN = f"{DRIVE}/qgen/{SHORT}"
os.makedirs(QGEN, exist_ok=True)
SYSTEM = "You help a search engine find documents that answer the latest question in a conversation."

def ask(kind, split, items):
    """items: [(ключ, промпт)] → {ключ: ответ}. Кэш на Drive: qgen/<модель>/<kind>_<split>.jsonl."""
    path = f"{QGEN}/{kind}_{split}.jsonl"
    got = {}
    if os.path.exists(path):
        got = {tuple(r["key"]): r["out"] for r in map(json.loads, open(path, encoding="utf-8"))}
    todo = [(k, p) for k, p in items if k not in got]
    if todo:
        t0 = time.time()
        outs = llm.generate([chat(tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}])
                             for _, p in todo], SP, use_tqdm=False)
        with open(path, "a", encoding="utf-8") as f:
            for (k, _), o in zip(todo, outs):
                got[k] = o.outputs[0].text.strip()
                f.write(json.dumps({"key": list(k), "out": got[k]}, ensure_ascii=False) + "\n")
        print(f"  LLM {kind} {split}: {len(todo)} за {(time.time() - t0) / 60:.1f} мин")
    return got
print("LLM:", LLM_MODEL)
'''),
("code", r'''
# ── 6. BM25 кита + проверка: BM25 + история на dev = таблица организаторов ──
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
    """То же, что OfficialBM25.search, но топ берётся numpy (устойчивая сортировка —
    равные баллы в порядке корпуса, как в ките) и сразу обрезается до TOPK."""
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
    print(f"  {dom:<20} {got:.4f}  таблица {BM25_DEV[dom]:.4f}  · {(time.time() - t0) / 60:.1f} мин")
    assert abs(got - BM25_DEV[dom]) < 0.0015, "BM25 не совпал с официальным — дальше не считать"
print("BM25 совпадает с официальным")
'''),
("code", r'''
# ── 7. Реплики и промпты ─────────────────────────────────────────────────────
NO_HIST = "No previous conversation."

def load_turns(split):
    """Те же реплики, что topics(): только с золотом. h — conversation_history как есть,
    prev — прошлые пары (вопрос, ответ) диалога. Золото и ответ текущей реплики не берутся."""
    out = []
    for dom in DOMAINS:
        for c in json.load(open(f"{DATA}/{dom}/benchmark_{split}.json", encoding="utf-8")):
            prev = []
            for t in c["turns"]:
                if t.get("gold_doc_ids") or t.get("supporting_doc_ids"):
                    h = t.get("conversation_history", "")
                    out.append({"key": (dom, f"{c['id']}_turn_{t['turn_id']}"), "q": t["query"],
                                "h": "" if h == NO_HIST else h, "prev": list(prev)})
                prev.append((t["query"], t["answer"]))
    return out

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
    """Слова абзаца, чьих основ (анализатор Lucene) нет в запросе; топ-n по частоте,
    при равенстве — по первому появлению."""
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
# ── 8. Варианты: тексты → Drive/reteco/queries, BM25 → Drive/reteco/runs ────
TEXTS = {}      # (вариант, split) → {ключ: [тексты]}; несколько текстов сливаются RRF

def save_variant(name, split, texts):
    """texts: {ключ: [текст, ...]}. В queries пишется один текст (склейка) — для плотных
    ноутбуков; BM25 ищет каждым текстом отдельно и сливает RRF."""
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
    """Тексты топ-k прогона для каждой реплики — это видит LLM в QExpand (без doc_id)."""
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
    save_variant("re", split, {t["key"]: [with_hist(t["q"], pairs(t["prev"][:-1] + [(t["prev"][-1][0], re_[t["key"]])]))
                                          if t["prev"] else t["q"]] for t in turns})
    save_variant("q2d_open", split, {k: [f"{b}\n\n{ps[k]}"] for k, b in base.items()})
    save_variant("kw", split, {k: [f"{b}\n\n{keywords(ps[k], b)}"] for k, b in base.items()})
    for name in ["win1", "win2", "stepback", "subq", "re", "q2d_open", "kw"]:
        run_variant(name, split)
    print(f"{split}: разовые варианты готовы")
    # QExpand: абзац → поиск → LLM видит топ-5 и переписывает → поиск → ещё раз
    prev_run = f"{RUNS}/bm25_q2d_open_{split}"
    for r in (1, 2):
        ex = ask(f"refine{r}", split, refine_prompts(turns, top_texts(prev_run, split, turns)))
        save_variant(f"qexp{r}", split, {k: [f"{b}\n\n{ex[k]}"] for k, b in base.items()})
        prev_run = run_variant(f"qexp{r}", split)
    print(f"{split}: QExpand готов")
'''),
("code", r'''
# ── 9. Итог: nDCG@10 макро по доменам ───────────────────────────────────────
REF = {"bm25_hist": BM25_MACRO, "bm25_q2d (Claude)": {"dev": 0.5463, "train": 0.5584}}
names = ["win1", "win2", "stepback", "subq", "re", "q2d_open", "kw", "qexp1", "qexp2"]
print(f"{'вариант':<22}" + "".join(f"{s:>9}" for s in SPLITS))
for n, v in REF.items():
    print(f"{n:<22}" + "".join(f"{v[s]:>9.4f}" for s in SPLITS))
for n in names:
    row = []
    for s in SPLITS:
        rd = f"{RUNS}/bm25_{n}_{s}"
        ok = all(os.path.exists(f"{rd}/{d}/run.trec") for d in DOMAINS)
        row.append(sum(ndcg10(read_run(f"{rd}/{d}/run.trec"), d, s) for d in DOMAINS) / len(DOMAINS) if ok else float("nan"))
    print(f"{n:<22}" + "".join(f"{x:>9.4f}" for x in row))
print("\nВыбирать по train. Дальше: скачать", RUNS, "и", QUERIES, "в проект, сказать Claude.")
'''),
    ]
    return nb(cells, "A100")

# ========================================================= llm_rerank ========
def llm_rerank_nb():
    cells = [
("markdown", r"""
# RETECO · listwise-реранк LLM (RankGPT / ReasonRank)

Модель видит сразу 20 кандидатов (только тексты, без `doc_id`) и выдаёт их
порядок `[3] > [1] > …`. Скользящее окно 20 / шаг 10 снизу вверх, как в RankGPT
и ReasonRank; при `TOPK = 20` это одно окно.

Модели: A100 — `liuwenhan/reasonrank-7B` (обучен ровно на это, сначала
рассуждает в `<think>`, потом `<answer>`); L4/T4 — Qwen3-4B-Instruct-2507 с
промптом RankGPT. Промпт ReasonRank взят дословно из их кода
(`listwise_prompt_r1.toml`, `add_prefix_prompt`, `add_post_prompt`).

**Как запускать.** Прогон первой стадии должен лежать на Drive в
`reteco/runs/<FIRST_STAGE>/`. Среда **A100** → «Выполнить все». Оборвалось —
ещё раз, готовые домены пропускаются.

**Сколько.** 858 реплик × 1 окно: reasonrank-7B на A100 ≈ 10–20 мин;
`TOPK = 100` (9 окон) — около часа.

**Что получится.** `reteco/runs/llmrr_<модель>_<первая стадия>_<запрос>/`.
Смешивание с первой стадией 0.6 / 0.4 — в `analysis.ipynb`.
"""),
("code", r'''
# ── 1. Настройки ─────────────────────────────────────────────────────────────
MODEL = "auto"        # A100 → liuwenhan/reasonrank-7B, иначе Qwen/Qwen3-4B-Instruct-2507
FIRST_STAGE = "bm25_fuse3_dev"   # папка в Drive/reteco/runs
QUERY_SET = "qd"      # что видит модель как запрос: "qd" — самостоятельный вопрос Claude,
                      # "hist" — реплика + история, как у организаторов; или папка в queries
SPLIT = "dev"
TOPK, WINDOW, STEP = 20, 20, 10
PASSAGE_TOKENS = 300  # документ обрезается до стольких токенов
MAXLEN = 16384
DRIVE = "/content/drive/MyDrive/reteco"
'''),
("code", LLM_SETUP), ("code", COMMON_DATA), ("code", COMMON_CODE),
("code", r'''
# ── 5. Модель и промпт (ReasonRank: rank_GPT_reasoning; прочие: rank_GPT) ────
from vllm import SamplingParams
if MODEL == "auto":
    MODEL = "liuwenhan/reasonrank-7B" if BIG else "Qwen/Qwen3-4B-Instruct-2507"
REASON = "reasonrank" in MODEL.lower()
llm = load_llm(MODEL, MAXLEN)
tok = llm.get_tokenizer()
SP = SamplingParams(temperature=0.0, max_tokens=3072 + 100 if REASON else 300)   # 3072 — reasoning_maxlen у ReasonRank
SYS = ("You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query."
       + (" Given a query and a passage list, you first thinks about the reasoning process in the mind and then "
          "provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed "
          "within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process "
          "here </think> <answer> answer here </answer>." if REASON else ""))
num_fix = lambda s: re.sub(r"\[(\d+)\]", r"(\1)", s)     # [n] в тексте спутал бы номера кандидатов

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
    """Порядок 0..n-1 из ответа: сначала названные номера без повторов, потом забытые
    в исходном порядке (как _clean_response в rank_llm)."""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    body = m.group(1) if m else text.split("</think>")[-1]
    seen = []
    for x in re.findall(r"\[(\d+)\]", body) or re.findall(r"\d+", body):
        i = int(x) - 1
        if 0 <= i < n and i not in seen:
            seen.append(i)
    return seen + [i for i in range(n) if i not in seen], len(seen)
print(MODEL, "· рассуждает" if REASON else "· RankGPT")
'''),
("code", r'''
# ── 6. Реранк по доменам → Drive ────────────────────────────────────────────
src = f"{RUNS}/{FIRST_STAGE}"
assert os.path.isdir(src), f"нет прогона первой стадии {src}"
dst = f"{RUNS}/llmrr_{MODEL.split('/')[-1]}_{FIRST_STAGE}_{QUERY_SET}"
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
        t0 = time.time()
        end = min(TOPK, max(len(v) for v in order.values()))
        start = end - WINDOW
        while end > 0 and start + STEP != 0:        # окна снизу вверх, как sliding_windows в rank_llm
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
        print(f"{dom}: {len(qids)} реплик за {(time.time() - t0) / 60:.1f} мин")
    a, b = ndcg10(first, dom, SPLIT), ndcg10(read_run(path), dom, SPLIT)
    rows.append((a, b))
    print(f"  {dom:<20} первая стадия {a:.4f} → LLM-реранк {b:.4f}  ({b - a:+.4f})")
json.dump({"model": MODEL, "first_stage": FIRST_STAGE, "query_set": QUERY_SET, "split": SPLIT, "topk": TOPK,
           "window": WINDOW, "step": STEP, "passage_tokens": PASSAGE_TOKENS, "prompt": "rank_GPT_reasoning" if REASON
           else "rank_GPT", "stats": stats, "gpu": torch.cuda.get_device_name(0),
           "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{dst}/config.json", "w"), ensure_ascii=False, indent=2)
A, B = (sum(r[i] for r in rows) / len(rows) for i in (0, 1))
print(f"\nМАКРО  первая стадия {A:.4f} → LLM-реранк {B:.4f}  ({B - A:+.4f})")
print("окон:", stats["windows"], "· без разбора:", stats["empty"], "· неполных:", stats["partial"],
      "· обрезано по длине:", stats["cut"], "(эти числа — только за этот запуск)")
'''),
    ]
    return nb(cells, "A100")

json.dump(embed_nb("AQ-MedAI/Diver-Retriever-4B", "Diver-Retriever-4B",
                   "Модель: рассуждающий эмбеддер на базе Qwen3-Embedding-4B. На RECOR у авторов "
                   "бенчмарка 0.545 против 0.446 у BM25 (с историей)."),
          open(f"{OUT}/diver.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(embed_nb("hanhainebula/reason-embed-qwen3-4b-0928", "reason-embed-qwen3-4b",
                   "Модель: ReasonEmbed 4B (arXiv:2510.08252). На RECOR её никто не замерял; "
                   "на BRIGHT сильнее Diver."),
          open(f"{OUT}/reason_embed.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(rerank_nb(), open(f"{OUT}/rerank.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(generate_nb(), open(f"{OUT}/generate.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(judge_nb(), open(f"{OUT}/judge.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(query_nb(), open(f"{OUT}/query.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(llm_rerank_nb(), open(f"{OUT}/llm_rerank.ipynb", "w"), ensure_ascii=False, indent=1)
print("записано:", sorted(os.listdir(OUT)))
