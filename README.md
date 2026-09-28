# RETECO_2026

SemEval-2027 Task 1 (RETECO), трек 2 — разговорный поиск. Что за задача и
правила — `CLAUDE.md`. План подходов — `docs/plan.md`.

## Установка (один раз)

```bash
brew install openjdk@21 uv
uv venv --python 3.12 ~/.reteco-venv
uv pip install --python ~/.reteco-venv/bin/python \
    pyarrow huggingface_hub tqdm gensim pytrec-eval-terrier pyserini sentence-transformers
git clone --depth 1 https://github.com/DataScienceUIBK/RETECO.git vendor/RETECO
~/.reteco-venv/bin/hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
    --local-dir data/reteco --include "track2_recor/*" "split_manifest.json"
```

venv лежит вне папки, потому что папка в Yandex.Disk. Дальше везде
`PY=~/.reteco-venv/bin/python`.

## Официальный бейзлайн

```bash
$PY vendor/RETECO/starter_kit/official_baseline.py --data data/reteco --out runs/baseline \
    --track1 --track2 biology drones earth_science economics hardware law \
    medicalsciences politics psychology robotics sustainable_living --splits train dev
$PY scripts/score.py --check-baseline      # сверить с docs/baseline_bm25.csv
```

## Свой прогон

```bash
$PY scripts/retrieve.py --method bm25 --query hist --out runs/2026-09-28_b1_bm25_hist
$PY scripts/score.py --runs runs/2026-09-28_b1_bm25_hist
$PY scripts/fuse.py --runs runs/A runs/B --out runs/AB_rrf     # слияние
```

`--query`: `query` (реплика) · `hist` (реплика + история) · `rewrite` · `rewrite_hist`.
`--method dense --model Qwen/Qwen3-Embedding-0.6B` — плотный; эмбеддинги
корпуса кэшируются в `data/index/`. **Плотный локально не гонять** (18 ГБ),
только в Colab; `.npy` потом положить в `data/index/`.

## Числа (dev, макро по 11 доменам)

| | nDCG@10 |
|---|---|
| BM25, реплика | 0.1827 |
| BM25, реплика + история | **0.4379** |

Recall@100 = 0.822. Разбивка по доменам — `docs/baseline_bm25.csv`.

## Если не работает

`docs/gotchas.md`. Чаще всего: pyserini не видит JDK — путь к brew-JDK задан в
`scripts/retrieve.py`, проверь, что `brew install openjdk@21` прошёл.
