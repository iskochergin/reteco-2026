"""Собирает colab/*.ipynb из одного источника: общие ячейки (Drive, данные,
topics, ndcg10, write_run) одинаковы во всех ноутбуках.

    python3 colab/build.py        # перезаписывает colab/*.ipynb

Ноутбуки правятся ЗДЕСЬ, не в .ipynb — иначе следующая сборка затрёт правку.
"""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
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

json.dump(embed_nb("AQ-MedAI/Diver-Retriever-4B", "Diver-Retriever-4B",
                   "Модель: рассуждающий эмбеддер на базе Qwen3-Embedding-4B. На RECOR у авторов "
                   "бенчмарка 0.545 против 0.446 у BM25 (с историей)."),
          open(f"{OUT}/diver.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(embed_nb("hanhainebula/reason-embed-qwen3-4b-0928", "reason-embed-qwen3-4b",
                   "Модель: ReasonEmbed 4B (arXiv:2510.08252). На RECOR её никто не замерял; "
                   "на BRIGHT сильнее Diver."),
          open(f"{OUT}/reason_embed.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump(rerank_nb(), open(f"{OUT}/rerank.ipynb", "w"), ensure_ascii=False, indent=1)
print("записано:", sorted(os.listdir(OUT)))
