# Released data

This document describes the files in the release archives and the audit files in
`docs/`. Licenses and attributions are in [DATA_LICENSE.md](../DATA_LICENSE.md);
each archive also carries its own notice, `data/drive/LICENSE-<archive>.md`,
which lists the files and fields under each license. No file contains corpus
passages, dialogue history or reference answers of RECOR or MTRAG. The free-text
fields that do occur are model outputs, which may quote short fragments of the
documents they read; a query rewrite can repeat the RECOR question word for word
(216 of the 2,971 rewrites in `reteco-expansions.zip` do, see below).

## Layout after unpacking

All archives unpack at the repository root (`unzip -o <archive>.zip`). Together
with the RETECO dataset (downloaded separately to `data/reteco/`) and the query
variants rebuilt by `scripts/rewrite.py --build-only` they give:

```
data/reteco/track2_recor/<domain>/      RETECO release: documents.jsonl, benchmark_{train,dev}.json, qrels_{train,dev}.txt
data/drive/LICENSE-<archive>.md         license notice of each archive
data/drive/reteco/
  runs/<run>/<domain>/run.trec          retrieval runs, with runs/<run>/config.json
  gen/<reader>_<job>_<split>.jsonl      reader responses, with gen/config.json
  judge/<judge>/<reader>_<job>_dev.jsonl   answer-quality judgments
  suff/<judge>_<job>_dev.jsonl          content labels, with suff/config.json (phi-4) and
                                        suff/config_<model>.json (each small judge)
  transfer/art_gen.jsonl                OCC outputs on SQuAD 2.0 and MuSiQue (not used by the paper)
  analysis/{runs,transfer,transfer_content}.csv
  mtrag/                                Replication on MTRAG (see below)
data/drive/audit/sample_key.csv         key of blind-audit batch 1
runs/rewrites/raw/<split>.jsonl         LLM query expansions, with raw/config_<split>.json
runs/rewrites/<variant>/<split>.jsonl   query variants (rebuilt, not in an archive)
docs/audit_llm.csv, docs/audit_llm2.csv blind-audit annotations (in git)
```

## Identifiers

**Domains.** Eleven RECOR domains: five from StackExchange (`drones`,
`hardware`, `law`, `medicalsciences`, `politics`) and six from BRIGHT
(`biology`, `earth_science`, `economics`, `psychology`, `robotics`,
`sustainable_living`).

**Turns.** A turn is identified by the pair (`domain`, `topic_id`), where
`topic_id` is `<conversation id>_turn_<turn id>` as in the official runs and
qrels. Conversation ids are `ex_<n>` in the StackExchange domains and integers in
the BRIGHT domains, so the same `topic_id` (for example `2_turn_1`) occurs in
several BRIGHT domains; always key by both fields. Only turns with gold documents
in the qrels are included: 858 dev turns from 211 dialogues and 2,113 train turns.

**Documents.** `doc_id` values are the ids of `documents.jsonl` in the RETECO
release. In the StackExchange domains they encode the source thread and the
document type (`..._ex_<n>_doc_<k>`, `..._ex_<n>_neg_<k>`,
`..._ex_<n>_google_neg_<k>`); in the BRIGHT domains they are chunk file names
such as `hot_water_bacteria/Hand_washing_0_13.txt`. Because ids in the
StackExchange domains reveal the dialogue and the document type, no model in
the paper saw document ids; they are used only in the analysis scripts.

**Contexts.** `context_ids` is the ordered list of documents a model saw: the
top-5 of a run in rank order, or the turn's gold documents. `evidence_docs` are
1-based positions into `context_ids`.

**Jobs.** The `<job>` part of file names names the context:

| job | context |
|---|---|
| `2b` | the turn's gold documents (task 2b) |
| `nodoc` | no documents (the no-document control: Section 4.2 and Table 2 of the paper) |
| `2c-<run>` | top-5 of the run `<run>_<split>` (task 2c) |
| `ctrl-<run>` | negative control: the first five documents of `<run>`'s top 10 for a turn of another dialogue in the same domain that are not gold for the current turn. The other dialogue sits half the list length further on in the domain's sorted list of dialogue ids (wrapping around); its turn with the same number is used, or its first turn if there is none. |

**Runs.** The four retrievers of the paper are `bm25_query` (BM25-turn),
`bm25_hist` (BM25-hist, the official baseline), `bm25_fuse3` (BM25-fusion) and
`hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d` (the hybrid). Run
directory names end in `_dev` or `_train`. Other run names combine a method, a
model and a query set: `bm25_` BM25, `dense_<model>_` dense retrieval,
`hyb_dense_<model>_` hybrid of BM25-fusion and that dense run, `rerank_` and
`llmrr_` reranked BM25-fusion lists; query sets `q2d` (turn + history + Query2doc
passage by Claude Sonnet 5), `q2d_open` (the same with the passage by
Qwen3-4B-Instruct-2507) and `qd` (the LLM rewrite of the turn). In file and
column names, a *stage* is a retriever, named by its run name without the split
(in the paper's analyses, one of the four retrievers above).

**Run configs.** Every run directory has a `config.json`, and so do `gen/`,
`suff/`, `mtrag/`, `mtrag/gen/` and `mtrag/suff/`. These files are historical
records, written when each run was produced; the one exception is
`runs/bm25_query_train/config.json`, written for this release. Argument names
and paths in them follow the version of the code that produced the run, not
necessarily the released code: run directories are named as they were at the
time (for example `runs/2026-09-29_bm25_fuse3_dev`), some component runs they
name are not released, and a git hash (`"git"`) refers to the development
repository, not to this repository. In `gen/config.json`,
`mtrag/gen/config.json` and `mtrag/suff/config.json`, Colab paths were replaced
by placeholders: `<DRIVE>` (the notebook's output folder, i.e.
`data/drive/reteco`), `<RETECO_DATA>` (the local copy of the Hugging Face
dataset) and `<TMP>` (temporary files of the MTRAG notebook).

## RECOR files

### Retrieval runs: `runs/<run>/<domain>/run.trec`

Standard TREC run format, one line per retrieved document, tab-separated:

```
topic_id  Q0  doc_id  rank  score  tag
```

Up to 100 documents per turn. `config.json` in each run directory records how
the run was made: the script or notebook and its arguments (`scripts/retrieve.py`,
`scripts/fuse.py`), or the component runs and weights (for example
`"w_dense": 0.7` for the hybrid). `bm25_query_train` is the official BM25
baseline output of the RETECO starter kit (`run_2a_train.trec` of each domain),
copied unchanged.

### Query expansions: `runs/rewrites/raw/<split>.jsonl`

From `reteco-expansions.zip` (release v1.1): the responses of Claude Sonnet 5
that `scripts/rewrite.py` cached when it generated the query expansions behind
Fusion (`bm25_fuse3`), the hybrid and the other `*_q2d` runs. One JSON object
per turn: 858 on dev, 2,113 on train. The model saw the turn and its
conversation history (earlier questions and answers), not the turn's reference
answer or any corpus document.

| field | meaning |
|---|---|
| `domain`, `topic_id` | turn |
| `rewrite` | the turn rewritten as a self-contained question |
| `topic_switch` | `true` if the model finds that the turn moves to another sub-topic |
| `passage` | a 60–100 word passage that could appear in a reference document and answers the turn (the Query2doc and HyDE passage) |
| `model` | the model that answered (`claude-sonnet-5`); `null` on the fallback line |
| `fallback` | only on the one train turn the model refused (`biology`, `50_turn_5`): `true`; its `rewrite` is the original turn and its `passage` is empty |

`config_dev.json` and `config_train.json` hold the system and task prompts
(`system`, `task`; the task prompt has a `%s` where the batch of turns goes), the
arguments of `scripts/rewrite.py` (`model` = `sonnet`, `effort` = `low`, `batch`
= 10 turns per call, `workers`, `timeout`), the version of the Claude Code client
(`cli`) and the date. In 61 dev and 155 train lines (the fallback line included)
`rewrite` is identical to the RECOR question, almost always on the first turn of a
dialogue, which needs no rewriting.

The query variants are rebuilt from these files and the RETECO data without
calling a model:

```bash
python scripts/rewrite.py --split dev --build-only
python scripts/rewrite.py --split train --build-only
```

This writes `runs/rewrites/<variant>/<split>.jsonl` (`domain`, `topic_id`, `text`)
for seven variants: `qd` (rewrite), `qd_hist` (rewrite + history), `ts` (turn,
plus history unless the topic switched), `hyde` (passage), `q2d` (turn + history +
passage), `qd_hyde` (rewrite + passage) and `qd_q2d` (rewrite + history +
passage). Fusion fuses BM25 runs over `q2d`, `qd_q2d` and `qd_hyde`; the dense half
of the hybrid encodes `q2d`. Most variant files contain the turn or its history
(RECOR text), so the variants are rebuilt rather than released.

### Reader responses: `gen/<reader>_<job>_<split>.jsonl`

One JSON object per turn and context. Readers: `Qwen3-1.7B`,
`Qwen3-4B-Instruct-2507`, `Qwen3-8B` (the three readers of the paper) and
`OCC-RAG-1.7B`, a reader from an earlier version of the analysis that the paper
does not use.

| field | meaning |
|---|---|
| `domain`, `topic_id`, `turn_id`, `split` | turn |
| `reader` | Hugging Face id of the reader |
| `mode` | `2b`, `2c` or `nodoc` |
| `first_stage` | run name for `2c`, otherwise `null` |
| `context_ids` | documents shown, in order |
| `n_gold` | number of gold documents of the turn |
| `gold_in_prior` | number of the turn's gold documents that were also gold for an earlier turn |
| `gold_in_ctx` | number of gold documents in the context |
| `suff` | qrels label of the context: `all`, `part` or `none` (see below) |
| `status` | the status the reader wrote: `ANSWERABLE`, `UNANSWERABLE`, or `""` when none was found (see below) |
| `answer` | response text after the status |
| `p_answerable` | p(ANSWERABLE) at the status position (see below); `null` when there is no status position or neither option is among the top-20 tokens |
| `p_mass` | a + u (see below); `0.0` when `p_answerable` is `null` |
| `finish` | vLLM finish reason (`stop` or `length`) |
| `trimmed` | `true` if each document was cut to 300 words to fit the prompt |
| `raw` | full model output; for OCC it includes the reasoning trace |

`gen/config.json` holds the generation settings (readers, jobs, `k_ctx` = 5,
`maxlen` = 16384, `max_new` = 1200, `trim_words` = 300, dtype, GPU, date). The
two `OCC-RAG-1.7B_2c-*_train.jsonl` files (2,113 lines each) are OCC on RECOR
train with the BM25-hist and BM25-fusion top-5, used to train and evaluate the
transfer detectors (not reported in the paper). Some OCC outputs contain
non-English text; they are released as generated.

### Answer-quality judgments: `judge/<judge>/<reader>_<job>_dev.jsonl`

Judges `phi-4` and `Qwen3-8B` score the responses of OCC-RAG-1.7B and
Qwen3-4B-Instruct-2507 on the gold context, without documents, and on the
top-5 of the three BM25 retrievers. The judge sees the dialogue history, the
question, the documents the reader saw, the reference answer and the response.
Fields: the turn and response identifiers of the generation file (`domain`,
`topic_id`, `turn_id`, `split`, `reader`, `mode`, `first_stage`, `suff`,
`status`, `gold_in_prior`); `judge` (Hugging Face id); `ok` (`true` if the judge
output was parsed); `correctness`, `completeness`, `relevance`, `coherence`,
`faithfulness` (integers 1 to 5); `response_type` (`answered`, `refused` or
`partial`); `unsupported_claims` (boolean); `rationale`; `raw` (full judge
output). In the 17 of 17,160 lines where `ok` is `false`, the score fields,
`response_type`, `unsupported_claims` and `rationale` are absent. The paper uses
the judgments of Qwen3-4B-Instruct-2507: a correct answer (Table 2) is an
`ANSWERABLE` status with `response_type` other than `refused` and
`correctness` of at least 4, and `response_type == "refused"` is an alternative
definition of a refusal (Section 4.3).

### Content labels: `suff/<judge>_<job>_dev.jsonl`

Content labels of `phi-4` and `Qwen3-14B` for the jobs `2b`, the four `2c-<run>`
jobs and the negative control `ctrl-hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d`;
858 lines per file.

| field | meaning |
|---|---|
| `domain`, `topic_id`, `turn_id`, `split`, `job` | turn and context |
| `judge` | Hugging Face id of the judge |
| `context_ids`, `n_gold`, `gold_in_ctx`, `gold_in_prior` | as in `gen/` |
| `suff_qrels` | qrels label of the context: `all`, `part` or `none` |
| `label` | content label: `yes`, `partial`, `no`, or `null` if the output could not be parsed |
| `answered_in_history` | judge's flag that the earlier dialogue already states the answer |
| `evidence_docs` | positions (1-based) of the documents the judge names as support |
| `rationale` | judge's rationale (at most two sentences requested) |
| `raw` | full judge output |
| `trimmed` | `true` if documents were cut to 300 words |

`suff/config.json` contains the exact system message and instruction (`system`,
`task`) and the settings (`k_ctx` = 5, `maxlen` = 16384, `max_new` = 400,
`trim_words` = 300, dtype, GPU, date). It was written by the first phi-4 run,
which covered the jobs `2b` and the three BM25 retrievers. The later runs (the
hybrid and negative-control jobs, and all Qwen3-14B jobs) used the same prompt and
settings; only the judge and the list of jobs differ.

The labels of the four small judges (`reteco-labels-small.zip`, release v1.1)
have the same jobs, fields and prompt: `suff/<model>_<job>_dev.jsonl` for
`<model>` = `Qwen3-4B-Instruct-2507`, `Phi-4-mini-instruct`, `SmolLM3-3B` and
`granite-3.3-8b-instruct` (Hugging Face ids `Qwen/Qwen3-4B-Instruct-2507`,
`microsoft/Phi-4-mini-instruct`, `HuggingFaceTB/SmolLM3-3B`,
`ibm-granite/granite-3.3-8b-instruct`), 858 lines per file, written by
`colab/suff_small.ipynb`. Each judge's prompt and settings are in
`suff/config_<model>.json`. `docs/paper/scripts/small_judges.py` reads them.

### Blind audit: `docs/audit_llm.csv`, `docs/audit_llm2.csv`, `data/drive/audit/sample_key.csv`

Batch 1 has 100 BM25-fusion contexts (50 with qrels label `none`, 50 `all`),
batch 2 has 150 contexts with qrels label `none`, 50 each from BM25-turn,
BM25-hist and the hybrid. Labels use the judges' scale (`yes`, `partial`, `no`).

- `sample_key.csv` (batch 1 key): `n` (item number), `domain`, `topic_id`,
  `first_stage` (`bm25_fuse3`), `n_gold`, `gold_in_top5`, `suff` (qrels label).
- `audit_llm.csv` (batch 1): `n` (joins `sample_key.csv`), `domain`,
  `suff_qrels`, `n_gold`, `gold_in_top5`, `A` and `B` (labels of the two
  annotation runs), `adjudicated` (whether a third run resolved a disagreement),
  `final`, `answered_in_history`, `confidence` (`high`, `medium` or `low`),
  `evidence_docs` (space-separated positions), `rationale`.
- `audit_llm2.csv` (batch 2): `n`, `stage` (run name without the split),
  `domain`, `topic_id`, `A`, `B`, `adjudicated`, `final`,
  `answered_in_history`, `evidence_docs`, `rationale`.

The context of an item is the top-5 of its run for that turn, in rank order, so
`evidence_docs` refers to positions in that list.

### Detector transfer: `transfer/art_gen.jsonl`

Not reported in the paper. 2,000 OCC-RAG-1.7B outputs on synthetic answerability items: `id` is
`squad_<SQuAD 2.0 id>_<a|u>` or `musique_<MuSiQue id>_<a|u>`, where `a` marks
the answerable and `u` the unanswerable version (500 of each class from each
source); `raw` is the full OCC output. The source questions and passages are not
included; they come from SQuAD 2.0 and MuSiQue under those datasets' terms.

### Summary tables: `analysis/*.csv`

- `runs.csv`: `run`, `ndcg_dev`, `r100_dev`, `ndcg_train`, `r100_train` for every
  retrieval run evaluated in `analysis.ipynb`, including variants not reported in
  the paper.
- `transfer.csv` (not reported in the paper): AUROC of the transfer detectors under the qrels label:
  `detector`, `train`, `layer`, `auroc_val` (on the held-out part of the
  training data), `auroc_dev` (on RECOR dev 2c contexts), then one column per
  first-stage run.
- `transfer_content.csv` (not reported in the paper): the same detectors under the qrels label, the content
  label (phi-4) and the clean label, per stage and pooled over stages:
  `detector`, `train`, `layer`, `label`, `stage`, `n_pos`, `n_neg`, `auroc`.

Category values (`label` and `stage` occur only in `transfer_content.csv`):

- `detector`: `OCC verdict` (OCC's own status), `p_status` (OCC's
  p(ANSWERABLE)), `trace` (a logistic regression on the length and mean token
  log-probability of OCC's reasoning trace), `probe` (a logistic regression on
  OCC's hidden state at the status position, from the layer in `layer`).
- `train`: `synthetic` (SQuAD 2.0 and MuSiQue items), `RETECO train`, or `—`
  for detectors without training.
- `label`: `qrels` (the qrels label), `content` (the phi-4 content label, `yes`
  against `no`) or `clean` (`yes` against `no` contexts whose answer the judge
  also finds absent from the dialogue history).
- `stage`: `bm25_query`, `bm25_hist`, `bm25_fuse3`, or `total` (the three stages
  pooled).

`docs/paper/scripts/baselines_v2.py` reads `transfer_content.csv` and matches
these values in `XNAME` and `XLAB`.

## MTRAG files (`data/drive/reteco/mtrag/`)

These files are used for the replication on MTRAG (Section 4.4 and Table 3 of
the paper) and for the human check of the judge (Section 4.2).
MTRAG (IBM/mt-rag-benchmark at commit `53fef6f5917c7e10c436f697e69a955dbc795d81`)
has four domains: `clapnq`, `cloud`, `fiqa`, `govt`. A turn is identified by
MTRAG's `task_id` (`<conversation_id><::><turn>`), which is also the topic id of
the runs; passage ids are MTRAG's. Retrievers: `bm25_questions` (BM25 over all
user questions), `bm25_lastturn` (BM25 over the last turn), `bm25_rewrite`
(BM25 over MTRAG's rewritten question), `dense_rewrite` (Qwen3-Embedding-0.6B
over the rewrite), `hybrid_rewrite` (equal-weight min-max fusion of
`bm25_rewrite` and `dense_rewrite`). Further jobs: `gold` (all gold
passages of the turn), `unans_hybrid` (the 55 unanswerable turns answered from
the hybrid top-5), `single` (one passage at a time).

- `turns.jsonl`: one line per turn (832): `task_id`, `conversation_id`, `turn`,
  `domain`, `answerability` (`ANSWERABLE`, `PARTIAL`, `UNANSWERABLE`), `gold`
  (gold passage ids from MTRAG's qrels), `use` (`label` for the 777 analysed
  turns, `unans` for the 55 unanswerable ones), `human` (MTRAG annotators'
  relevance judgments of the passages shown to them on that turn, passage id to
  `yes`, `no` or `null`), `top5` (retriever to the five retrieved passage ids).
  The query, history, rewrite and reference answer texts are removed.
- `runs/<retriever>/<domain>/run.trec` and `runs/<retriever>/config.json`:
  TREC runs, top-100.
- `retrieval.csv`: nDCG@10, recall@5 and recall@100 per retriever, macro and
  micro over domains and per domain.
- `suff/phi-4_<job>.jsonl`: phi-4 content labels, one line per turn and context:
  `key` and `task_id` (both the MTRAG task id), `domain`, `turn`,
  `answerability`, `job`, `context_ids`, `n_gold`, `gold_in_ctx`, `suff_qrels`,
  `judge`, `label`, `answered_in_history`, `evidence_docs`, `rationale`, `raw`,
  `trimmed`, with the same meaning as in the RECOR labels. In
  `phi-4_single.jsonl` each line is one passage judged on its own, with the
  extra fields `doc_id`, `kind` (`gold`, or `human_no` for a passage the
  annotators marked not relevant), `human_final` and `human_votes` (the
  annotators' final and individual judgments), `in_corpus`,
  `same_page_as_gold` and `gold_other_turn`.
- `gen/<reader>_<job>.jsonl`: responses of Qwen3-4B-Instruct-2507 (used by the
  paper) and OCC-RAG-1.7B (from an earlier version of the analysis, not used): `key`, `task_id`, `domain`, `turn`, `answerability`,
  `job`, `context_ids`, `n_gold`, `gold_in_ctx`, `suff` (qrels label, `unans`
  for the unanswerable turns), `reader`, `status`, `answer`, `p_answerable`,
  `p_mass`, `finish`, `trimmed`, `raw`, with the same meaning as in the RECOR
  responses.
- `summary.csv`: every MTRAG number with its 95% bootstrap interval: `section`
  (`labels`, `readers`, `human_single`), `job`, `reader`, `metric`, `value`,
  `lo`, `hi`. In the `labels` section, a `job` of the form `<a> − <b>` is the
  difference between two retrievers. In the `human_single` section, `job` is
  `gold (annotator: relevant)` (gold passages), `human_no (annotator: not
  relevant)` (passages the annotators marked not relevant), `human_no · same
  page` and `human_no · other page` (those on the same page as a gold passage
  and on another page), or `human_no · gold of another turn` (those that are the
  gold passage of another turn). In the `readers` section, refusal rates and the
  2b/2c contrast (`contrast_q`, `contrast_c`) count a response without a status
  as an answer; the paper excludes such responses (Section 3), so its contrast
  for Qwen3-4B on the hybrid (Section 4.4: +.09 content, +.01 qrels) differs
  from `summary.csv` (+.10, +.02).
- `config.json`, `gen/config.json`, `suff/config.json`: settings of the runs
  (retrievers, BM25 parameters, embedding model, judge and its prompt, readers,
  token limits, bootstrap, GPU, date).
- `LICENSE-APACHE-2.0.txt`: the Apache License 2.0, which covers the fields
  re-emitted from MTRAG.

## How the labels were produced

**qrels label.** The gold documents of a turn are its `gold_doc_ids` that appear
in the qrels. For a context of k documents, `all` means it contains
min(n_gold, k) gold documents, `none` that it contains none, and `part` anything
in between. The proxy studied in the paper treats `none` as insufficient.

**Content label.** A sufficiency judge reads the dialogue so far, the latest
question, the reference answer and the numbered documents (without ids) and
answers `yes` (the documents support the core claims of the reference answer),
`partial` (some but not all central points) or `no`, names the supporting
documents and flags whether the answer was already given earlier in the
dialogue. The prompt is not printed in the paper; it is verbatim in
`suff/config.json` (and in `suff/config_<model>.json` for the small judges). Decoding is greedy with at most 400 new tokens; prompts
longer than 16,384 tokens are shortened by cutting each document to 300 words.
Binary metrics compare `yes` with `no` and drop `partial`; `null` labels are
excluded. A false miss is a context with qrels label `none` that the judge
labels `yes`.

**Blind audit.** Two annotation runs of Claude Opus 5.5 labelled each item with
the judges' scale, one reading each item holistically and one checking the
reference answer claim by claim; a third run adjudicated disagreements. Items
were shuffled within a batch, carried no ids and differed only in the documents
shown, so the annotator saw neither the retriever nor the qrels label. These are
LLM annotations, not human ground truth.

**Reader status and p(ANSWERABLE).** Readers put `ANSWERABLE` or
`UNANSWERABLE` on the first line (OCC-RAG-1.7B produces its own status token
after a reasoning trace). For 15 OCC-RAG-1.7B outputs (8 on RECOR, 7 on MTRAG)
the text between OCC's status markers is neither, and `status` keeps it as
parsed. A second pass of the reader reads the 20 most probable tokens at the
status position. Let a and u be the total probabilities of those tokens that are
prefixes of `ANSWERABLE` and of `UNANSWERABLE`. Then `p_answerable` = a / (a + u)
and `p_mass` = a + u. A refusal is a status that starts with `UN`; responses
without a status are excluded from refusal rates and AUROCs.

**MTRAG annotator judgments.** Taken from MTRAG's conversation files: for each
turn, the relevance feedback on the passages shown to the annotator; the final
judgment is the editor's if the editor judged the passage, otherwise the
author's.

## Not released

| Material | Reason | How to obtain it |
|---|---|---|
| RECOR corpus, dialogues, reference answers, qrels | RETECO text is CC BY-SA 4.0 and lacks per-document source attribution; the organizers distribute it | Hugging Face `DataScience-UIBK/RETECO-SemEval2027`, revision `ea71a271d80711a23a22cb4830ac3186f0df711a` |
| `data/drive/audit/sample.csv` and the item files of audit batch 2 | contain questions, history, reference answers and full passages | `generate.ipynb` (cell "8. Sample for the blind audit, batch 1 → Drive/reteco/audit") for batch 1; `docs/paper/scripts/audit_build.py <out_dir>` for batch 2 |
| MTRAG passages and conversation text (`mtrag/passages.jsonl`, text fields of `mtrag/turns.jsonl`, `mtrag/unanswerable_hybrid_top5.jsonl`) | upstream text with its own terms (Wikipedia via ClapNQ, FiQA, crawled government and cloud documentation) | `mtrag.ipynb`, which downloads MTRAG at the pinned commit |
| Query variant files of the Claude Sonnet 5 expansions (`runs/rewrites/<variant>/`) | most hold the turn or its history (RECOR text) next to the generated expansion | `scripts/rewrite.py --build-only` from `reteco-expansions.zip` |
| Open-LLM query variants of `query.ipynb` (`q2d_open` and the variants the paper does not report: `queries/`, raw responses in `qgen/`) | the query files hold the turn and its history (RECOR text) | `query.ipynb` generates them again; the `bm25_q2d_open` runs in `reteco-runs-extra.zip` are the reference |
| Runs of unreported retrieval variants, probe features, Colab logs | not used by any paper script | numbers of the variants are in `analysis/runs.csv` |
| Anything from the hidden SemEval-2027 test set | the organizers do not permit redistribution of hidden evaluation data or inferred gold labels | none |
