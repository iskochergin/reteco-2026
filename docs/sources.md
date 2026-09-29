# Источники

Всё, на что опирается план. Проверено 28 сентября 2026.

## Задача и данные

- RETECO, сайт задачи: https://datascienceuibk.github.io/RETECO/
- Правила участия (внешние корпуса запрещены, любые модели можно при раскрытии): https://datascienceuibk.github.io/RETECO/participate.html
- Данные: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Стартовый кит: https://github.com/DataScienceUIBK/RETECO
- RECOR, сам бенчмарк, Findings ACL 2026: arXiv:2601.05461, https://aclanthology.org/2026.findings-acl.129/
- Код RECOR (ретриверы, абляции): https://github.com/RECOR-Benchmark/RECOR
- BRIGHT, источник 6 из 11 доменов: arXiv:2407.12883

## Reasoning-поиск (линия BRIGHT)

- Обзор reasoning-intensive retrieval: arXiv:2605.00063
- Воспроизводимые бейзлайны BRIGHT (Anserini/Pyserini/RankLLM), SIGIR 2026 Repro: arXiv:2509.02558
- DIVER, многостадийный пайплайн, 46.8 на BRIGHT: arXiv:2508.07995
- ReasonRank, listwise-реранкер: arXiv:2508.07050, https://github.com/8421BCD/ReasonRank
- Rank1, pointwise-реранкер с рассуждением: arXiv:2502.18418
- ReasonEmbed: arXiv:2510.08252
- ReasonIR: arXiv:2504.20595
- RaDeR: arXiv:2505.18405
- JudgeRank: arXiv:2411.00142
- Reranker-guided search: arXiv:2509.07163
- GroupRank: arXiv:2511.11653

## Разговорный поиск (линия CAsT / QReCC / TopiOCQA / iKAT)

- CHIQ, улучшение истории, Topic Switch +13.2 % MRR, EMNLP 2024: arXiv:2406.05013
- LLM4CS, RAR-промптинг и агрегация: arXiv:2303.06573, https://github.com/kyriemao/LLM4CS
- CFDA & CLIP at TREC iKAT 2025, рабочий рецепт с абляциями: arXiv:2509.15588
- ConvSearch-R1, RL для переписывания: arXiv:2505.15776
- CompCQR, training-free: arXiv:2609.14646
- ICR, итеративное уточнение: arXiv:2509.05100
- Multi-query rewriting, SIGIR 2024: https://dl.acm.org/doi/10.1145/3626772.3657933
- Обзор разговорного поиска, Mo et al., TOIS: https://dl.acm.org/doi/10.1145/3759453
- Multi-aspect query generators: arXiv:2403.19302
- Entailment distillation for conversational retrieval: arXiv:2609.03482

## Расширение запроса

- Query2Doc: arXiv:2303.07678
- Revisiting Feedback Models for HyDE (RM3 вместо склейки): arXiv:2511.19349
- PRF с глубокими моделями, успехи и провалы: arXiv:2108.11044
- Structured query expansion for multi-hop: arXiv:2603.21024
- Retrieval-feedback distillation for LLM QE: arXiv:2603.13776

## Слияние, разреженные и multi-vector модели

- Анализ функций слияния: arXiv:2210.11934
- OpenSearch RRF (RRF на 3.86 % хуже настроенной нормализации): https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
- SPLADE, TOIS: https://dl.acm.org/doi/10.1145/3634912
- История разреженного поиска: https://huggingface.co/blog/yjoonjang/the-past-and-present-of-sparse-retrieval
- Jina-ColBERT-v2: arXiv:2408.16672
- PyLate: arXiv:2508.03555

## Модели на HuggingFace (проверено наличие и размер)

| модель | размер весов | заметка |
|---|---|---|
| AQ-MedAI/Diver-Retriever-4B | 8.0 ГБ | база Qwen3-Embedding-4B; промпты в retrieve.py |
| hanhainebula/reason-embed-qwen3-4b-0928 | 16.1 ГБ fp32, 8 ГБ bf16 | на RECOR не мерили |
| hanhainebula/reason-embed-qwen3-8b-0928 | 30 ГБ | 38.1 на BRIGHT, только A100 |
| BAAI/bge-reasoner-embed-qwen3-8b-0923 | 30 ГБ; AWQ-4bit 8.6 ГБ | 37.1 на BRIGHT; 4-bit влезает в T4 |
| reasonir/ReasonIR-8B | 15 ГБ | только A100 |
| Qwen/Qwen3-Embedding-0.6B / 4B | 1.2 / 8 ГБ | дешёвый плотный бейзлайн |
| Qwen/Qwen3-Reranker-0.6B / 4B / 8B | 1.2 / 8 / 16 ГБ | pointwise yes/no, Apache 2.0 |
| BAAI/bge-reranker-v2-m3 | 2.3 ГБ | запасной cross-encoder |
| liuwenhan/reasonrank-7B | 15 ГБ | listwise, только A100 |

DIVER-Reranker на HF нет: это промптинг Qwen2.5-32B и DeepSeek-R1, заменяем
gpt-oss-120b (AIRI) или Claude.
