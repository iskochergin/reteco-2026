# Data licenses

Data and documentation for the paper "Better Retrievers, Worse Labels: How
Incomplete Relevance Judgments Distort Abstention Evaluation in Conversational
RAG" by Ivan Kochergin and Vasily Konovalov (2026).

The code in this repository (`scripts/`, `docs/paper/scripts/`, `colab/`) is under
the MIT License ([LICENSE](LICENSE)). This file covers everything else: the
release archives, the audit annotations in `docs/`, the aggregate results and
figures under `docs/paper/`, the paper source and the documentation. None of
it contains RECOR or MTRAG corpus passages, questions or reference answers, apart
from the short excerpts and the repeated questions listed at the end. Nothing is
derived from the hidden SemEval-2027 test set.

## License scheme

The release archives are `reteco-runs.zip`, `reteco-runs-extra.zip`,
`reteco-model-outputs.zip`, `reteco-labels-audit.zip` and `mtrag.zip` (release
v1.0), and `reteco-labels-small.zip` and `reteco-expansions.zip` (release v1.1).
All of them use one scheme. Each archive carries a notice
(`data/drive/LICENSE-<archive>.md`) that lists its files and fields under each
license:

- ids, labels, scores, run files, aggregate numbers, settings and prompts:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- machine-generated free text (reader answers and raw outputs, judge rationales
  and raw outputs, audit rationales, synthetic transfer generations, query
  expansions):
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), to the extent
  any copyright subsists;
- files or fields re-emitted from MTRAG (turns and annotator judgments): the
  [Apache License 2.0](LICENSES/Apache-2.0.txt), as MTRAG itself.

Generated text is under CC BY-SA 4.0 because the models read CC BY-SA text and
their outputs paraphrase and sometimes quote it.

| Item | CC BY 4.0 | CC BY-SA 4.0 | Apache-2.0 |
|---|---|---|---|
| `reteco-runs.zip`, `reteco-runs-extra.zip` | all files (run files and their `config.json`) | | |
| `reteco-model-outputs.zip` | all other fields (ids, qrels-derived labels, reader status and scores, judge scores and response types) and `gen/config.json` | `answer` and `raw` in `gen/*.jsonl`; `rationale`, `unsupported_claims` and `raw` in `judge/*/*.jsonl`; `raw` in `transfer/art_gen.jsonl` | |
| `reteco-labels-audit.zip` | all other fields of `suff/*.jsonl`, the judge prompt and settings in `suff/config.json`, `audit/sample_key.csv`, `analysis/*.csv` | `rationale` and `raw` in `suff/*.jsonl` | |
| `reteco-labels-small.zip` | all other fields of `suff/*.jsonl`, the judge prompt and settings in `suff/config_<model>.json` | `rationale` and `raw` in `suff/*.jsonl` | |
| `reteco-expansions.zip` (paths under `runs/rewrites/raw/`) | `domain`, `topic_id`, `topic_switch`, `model` and `fallback` in `dev.jsonl` and `train.jsonl`; the prompts and settings in `config_dev.json` and `config_train.json` | `rewrite` and `passage` in `dev.jsonl` and `train.jsonl` | |
| `mtrag.zip` (paths under `data/drive/reteco/mtrag/`) | everything else: `runs/`, the fields `use` and `top5` of `turns.jsonl`, our labels and scores in `gen/` and `suff/`, the `config.json` files, `retrieval.csv`, `summary.csv` | `answer` and `raw` in `gen/*.jsonl`; `rationale` and `raw` in `suff/*.jsonl` | the MTRAG fields of `turns.jsonl` (`task_id`, `conversation_id`, `turn`, `domain`, `answerability`, `gold`, `human`), the same MTRAG fields where they are repeated in `gen/*.jsonl` and `suff/*.jsonl` (`task_id`, `domain`, `turn`, `answerability`), and `human_final` and `human_votes` in `suff/phi-4_single.jsonl` |
| `docs/audit_llm.csv`, `docs/audit_llm2.csv` | all columns except `rationale` | `rationale` | |
| `docs/paper/out/`, `docs/paper/latex/fig_*.pdf` | all, except the excerpts below | the excerpts below | |
| paper source (`docs/paper/latex/main.tex`, `custom.bib`) and documentation (`README.md`, `docs/DATA.md`, this file) | all | | |

`docs/baseline_bm25.csv` reproduces the official baseline scores published by the
RETECO organizers. A copy of the Apache License 2.0 is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) and, inside `mtrag.zip`, in
`data/drive/reteco/mtrag/LICENSE-APACHE-2.0.txt`.

## Required attributions

**RETECO, RECOR and BRIGHT** (all `reteco-*` archives, the audit files, the
aggregate results). Derived from the RETECO SemEval-2027 Task 1 train and dev
data (https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027,
revision `ea71a271d80711a23a22cb4830ac3186f0df711a`; annotations CC BY 4.0, text
CC BY-SA 4.0), built from RECOR (Ali et al., Findings of ACL 2026,
doi:10.18653/v1/2026.findings-acl.129) and BRIGHT (Su et al., ICLR 2025). The
underlying content belongs to Stack Exchange contributors and to the authors of
the third-party web pages in the corpora. Changes: we generated query
expansions for the train and dev turns, retrieved documents for them, derived
the context-level qrels label (`all`, `part`, `none`) from the qrels, and
generated responses and content labels for the retrieved contexts. Please cite RETECO (Abdallah et al., 2026) and RECOR (Ali et
al., 2026); see the README.

**Model outputs.** Machine-generated text, not human annotation. Readers:
`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B`, and
`occ-ai/OCC-RAG-1.7B`, a reader from an earlier version of the analysis that the
paper does not use. Sufficiency judges: `microsoft/phi-4`, `Qwen/Qwen3-14B`;
small sufficiency judges (`reteco-labels-small.zip`):
`Qwen/Qwen3-4B-Instruct-2507`, `microsoft/Phi-4-mini-instruct`,
`HuggingFaceTB/SmolLM3-3B`, `ibm-granite/granite-3.3-8b-instruct`. Answer
judges: `Qwen/Qwen3-8B`, `microsoft/phi-4`. Models were loaded by Hugging Face id
without a pinned revision; the `config.json` files in `gen/`, `suff/` and
`mtrag/` (and `suff/config_<model>.json` for the small judges) record the run
dates. The query expansions behind BM25-fusion, the
hybrid and the `*_q2d` runs (`reteco-expansions.zip`) were generated by Claude
Sonnet 5 (Anthropic) through Claude Code; `runs/rewrites/raw/config_*.json`
record the prompts and settings. `transfer/art_gen.jsonl` contains OCC-RAG-1.7B
responses to SQuAD 2.0 (Rajpurkar et al., 2018) and MuSiQue (Trivedi et al.,
2022) items and may quote them.

**Blind audit** (`docs/audit_llm.csv`, `docs/audit_llm2.csv`). Generated by
Claude Opus 5.5 (Anthropic). These are LLM annotations, not human ground truth.
The rationales quote short fragments of RECOR documents.

**MTRAG** (`mtrag.zip`). Derived from MTRAG
(https://github.com/IBM/mt-rag-benchmark, commit
`53fef6f5917c7e10c436f697e69a955dbc795d81`), Copyright IBM, licensed under the
Apache License 2.0. Modified: MTRAG's turn ids, answerability labels, gold
passage ids and annotator relevance judgments were converted into per-turn and
per-context records and combined with our runs, labels and model outputs; all
MTRAG text (queries, history, reference answers, passages) was removed. Cite
Katsis et al., "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating
Retrieval-Augmented Generation Systems", TACL 13 (2025) 784-808,
doi:10.1162/tacl.a.19. The generated text fields respond to MTRAG passages that
include ClapNQ (Rosenthal et al., 2024; Wikipedia text via Natural Questions)
and FiQA (via BEIR) text, which is why they are under CC BY-SA 4.0.

## Excerpts of benchmark text

The following files contain short verbatim excerpts of RECOR text (a few
questions and sentences of corpus documents) used as examples in the paper:

- `docs/paper/out/v2/anatomy_examples.csv` (columns `question` and `shared_sentence`)
- `docs/paper/latex/main.tex` (Figure 1 quotes one question and one passage)

In `reteco-expansions.zip`, the `rewrite` field repeats the RECOR question word
for word in 61 of 858 dev lines and 155 of 2,113 train lines: turns that were
already self-contained, and one train turn the model refused (`"fallback": true`),
which keeps the original question. Other rewrites reuse much of the question's
wording.

These excerpts and repeated questions are text from the RETECO SemEval-2027
release (CC BY-SA 4.0); original content by Stack Exchange contributors
(stackexchange.com) and the authors of the third-party web pages in the corpus;
via RECOR (Ali et al., 2026) and BRIGHT (Su et al., 2025). Licensed under
CC BY-SA 4.0:
https://creativecommons.org/licenses/by-sa/4.0/.
