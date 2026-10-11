# Better Retrievers, Worse Labels: How Incomplete Relevance Judgments Distort Abstention Evaluation in Conversational RAG

Code, retrieval runs, model outputs, content labels and audit annotations for
the paper of the same title (Ivan Kochergin, 2026).

Evaluating abstention in retrieval-augmented generation requires knowing whether
each retrieved context is sufficient. A common shortcut treats a context that
contains none of the turn's gold documents as unanswerable. On RECOR, the data of
SemEval-2027 RETECO track 2, we compare this qrels-derived label with content
labels from two open LLM judges (phi-4 and Qwen3-14B) and with a blind double
annotation of 250 contexts by a third model family, across four first-stage
retrievers (nDCG@10 .18 to .59) and four readers. As retrieval improves, the
"insufficient" class fills with contexts that do contain the answer: from
turn-only BM25 to a hybrid retriever, the share of no-gold contexts that the
primary judge labels sufficient rises from 28% to 72%, and the precision of "no
gold" as a label of insufficiency falls from .68 to .10. The decline replicates
on MTRAG, whose qrels come from annotators and whose corpus has no mined
negatives (precision .72 to .50 as nDCG@10 rises from .18 to .42). Under the
qrels label, readers' answerability AUROC is understated by up to .19, the
refusal gap between insufficient retrieved contexts and gold contexts vanishes
at a weaker retriever for three of four readers, and a retrieval-score baseline
looks like the best sufficiency detector.

## Main results

**RECOR dev** (858 turns, 11 domains, top-5 contexts). Precision: among no-gold
contexts (qrels label `none`) that the judge labels `yes` or `no`, the share
labelled `no`; `partial` is excluded. False misses: the share of all no-gold
contexts that the judge labels `yes`. The blind audit covers 50 no-gold contexts
per retriever. Confidence intervals are in Table 1 of the paper.

| | BM25-turn | BM25-hist | BM25-fusion | Hybrid |
|---|---:|---:|---:|---:|
| nDCG@10 | .183 | .438 | .550 | .591 |
| Contexts without gold (%) | 71.6 | 35.0 | 22.7 | 18.4 |
| False misses, phi-4 (%) | 28 | 45 | 68 | 72 |
| Precision of "no gold", phi-4 | .68 | .44 | .18 | .10 |
| Precision of "no gold", Qwen3-14B | .84 | .69 | .29 | .21 |
| Precision of "no gold", blind audit | .70 | .62 | .23 | .21 |

**MTRAG** (777 answerable and partially answerable turns, phi-4 judge, five
first-stage retrievers ordered by nDCG@10; definitions as above).

| | BM25, all questions | BM25, last turn | BM25, rewrite | Hybrid | Dense |
|---|---:|---:|---:|---:|---:|
| nDCG@10 | .18 | .22 | .25 | .40 | .42 |
| Contexts without gold (%) | 71 | 64 | 61 | 40 | 37 |
| False misses (%) | 23 | 23 | 27 | 40 | 38 |
| Precision of "no gold" | .72 | .72 | .67 | .50 | .50 |

## Repository layout

```
README.md               this file
LICENSE                 MIT license for the code
DATA_LICENSE.md         licenses and attributions for the data, paper source and docs
LICENSES/Apache-2.0.txt copy of the Apache License 2.0 (applies to MTRAG-derived files)
CITATION.cff            citation metadata
requirements.txt        Python dependencies for local analysis and retrieval
scripts/
  retrieve.py           first-stage retrieval: the starter kit's BM25, Lucene BM25 (pyserini,
                        optional RM3) or dense; writes runs/<name>/<domain>/run.trec
  rewrite.py            LLM query variants (rewrite, HyDE passage, Query2doc passage)
  fuse.py               fusion of runs: RRF or weighted sum of min-max normalised scores
  score.py              nDCG@10 (macro over domains), recall@10/100 and MRR of a run
colab/
  *.ipynb               Colab notebooks for all model inference (see "Model inference")
  build.py              single source of the notebooks; writes colab/*.ipynb
  keepalive.js          browser-console snippet that build.py embeds in each notebook
docs/
  DATA.md               formats and fields of the released files
  baseline_bm25.csv     official BM25 baseline scores per domain, used by score.py
  audit_llm.csv         blind audit, batch 1: 100 BM25-fusion contexts
  audit_llm2.csv        blind audit, batch 2: 150 no-gold contexts (BM25-turn, BM25-hist, hybrid)
  paper/
    latex/              main.tex, custom.bib, tables/tab_*.tex, fig_*.pdf
    scripts/            analysis scripts; each table and figure is written by one of them
    out/                intermediate results written by the scripts (CSV, JSON)
data/                   empty in git; the dataset and the release archives are unpacked here
runs/                   empty in git; output of scripts/retrieve.py and scripts/fuse.py
```

The ACL style files needed to compile `main.tex` (`acl.sty`, `acl_natbib.bst`)
are not included; they are distributed with the ACL paper template.

## Reproducing the paper

### 1. Environment

The official BM25 needs Python 3.12 or later, because the starter kit tokenises
with pyserini 2.4; the analysis scripts alone need only numpy, scipy, matplotlib
and pytrec-eval-terrier.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Retrieval with the official BM25 (`scripts/retrieve.py --method bm25`) imports
the RETECO starter kit. The kit has no software license yet, so it is not
included here; clone it at the commit we used:

```bash
git clone https://github.com/DataScienceUIBK/RETECO.git vendor/RETECO
git -C vendor/RETECO checkout 23093c30e568a337c6293de4c58ffcccddbded5b
```

pyserini needs a Java 21 JDK. Set `JAVA_HOME`, and `JVM_PATH` to the JVM library
(`lib/server/libjvm.so` on Linux, `lib/server/libjvm.dylib` on macOS);
`scripts/retrieve.py` sets both itself when a Homebrew `openjdk@21` is installed.

### 2. Data

The corpora, dialogues and qrels of RECOR come from the RETECO release on
Hugging Face
([DataScience-UIBK/RETECO-SemEval2027](https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027)).
Our analysis used revision `ea71a271d80711a23a22cb4830ac3186f0df711a`:

```bash
hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
    --revision ea71a271d80711a23a22cb4830ac3186f0df711a \
    --local-dir data/reteco --include "track2_recor/*" "split_manifest.json"
```

Then download the release archives listed under [Data release](#data-release)
from the [v1.0 release](https://github.com/iskochergin/reteco-2026/releases/tag/v1.0)
into the repository root, check them and unzip them there:

```bash
BASE=https://github.com/iskochergin/reteco-2026/releases/download/v1.0
for a in reteco-runs reteco-model-outputs reteco-labels-audit mtrag reteco-runs-extra; do
    curl -LO "$BASE/$a.zip"
done
shasum -a 256 *.zip             # or sha256sum; compare with the release page
unzip -o reteco-runs.zip
unzip -o reteco-model-outputs.zip
unzip -o reteco-labels-audit.zip
unzip -o mtrag.zip
unzip -o reteco-runs-extra.zip  # optional: further runs of Table 3
```

The release notes on the release page (`MANIFEST.md`) list the size, file count
and SHA-256 checksum of each archive.

The result is the layout the scripts expect:

```
data/reteco/track2_recor/<domain>/   documents.jsonl, benchmark_{train,dev}.json, qrels_{train,dev}.txt
data/drive/reteco/runs/              retrieval runs
data/drive/reteco/gen/               reader responses
data/drive/reteco/judge/             answer-quality judgments
data/drive/reteco/suff/              content labels
data/drive/reteco/transfer/          OCC outputs on SQuAD 2.0 and MuSiQue
data/drive/reteco/analysis/          summary tables written by the notebooks
data/drive/reteco/mtrag/             MTRAG runs, labels and reader responses
data/drive/audit/sample_key.csv      key of blind-audit batch 1
data/drive/LICENSE-<archive>.md      license notice of each archive
docs/audit_llm.csv, docs/audit_llm2.csv   blind-audit annotations (in git)
```

### 3. Tables and figures

All analysis runs on a CPU from the saved model outputs. Run the scripts from the
repository root in this order: `tables_v2.py` and `paper_figs.py` read the files
that the scripts before them write.

```bash
python docs/paper/scripts/label_v2.py
python docs/paper/scripts/false_miss_anatomy_v2.py
python docs/paper/scripts/consequences_v2.py
python docs/paper/scripts/refusal_flips.py
python docs/paper/scripts/baselines_v2.py
python docs/paper/scripts/control_v2.py
python docs/paper/scripts/mtrag_v2.py
python docs/paper/scripts/reranker_holes.py
python docs/paper/scripts/tables_v2.py
python docs/paper/scripts/paper_figs.py
```

| Paper | Script | Output |
|---|---|---|
| Table 1, Table 5, Figure 2 | `label_v2.py` | `out/v2/label_*.csv`, `audit_stage.csv`, `agreement.csv`, `gold_present_no.csv`, `label_v2.json` |
| Figure 3, Appendix H | `false_miss_anatomy_v2.py` | `out/v2/anatomy*.csv`, `anatomy_summary.json` |
| Figure 4, Tables 7 and 8, common-set AUROC, counts of changed conclusions | `consequences_v2.py` | `out/v2/{refusal,auroc,auroc_common,contrast,decomp,flips}.csv`, `consequences_summary.json` |
| Table 6 | `refusal_flips.py` | `out/refusal_flips.json` |
| Table 9 | `baselines_v2.py` | `out/v2/detectors.csv`, `detectors_pairs.csv`, `selective.csv`, `baselines.json` |
| Negative control (Section 4) | `control_v2.py` | `out/v2/control.csv`, `control.json` |
| Table 2 (label shares, precision), MTRAG evidence classes (Section 7) | `mtrag_v2.py` | `out/v2/mtrag_anatomy.csv`, `mtrag_anatomy.json` |
| Appendix B (rerankers) | `reranker_holes.py` | `out/reranker_holes.json`, `reranker_holes_domains.csv` |
| LaTeX of Tables 1, 2, 5 to 9 | `tables_v2.py` | `latex/tables/tab_*.tex` |
| Figures 2, 3, 4 | `paper_figs.py` | `latex/fig_bias.pdf`, `fig_anatomy.pdf`, `fig_auroc.pdf` |
| Figure 1 | none: a table typeset in `latex/main.tex` | |

Paths in the Output column are relative to `docs/paper/`. The committed files
under `docs/paper/out` and `docs/paper/latex` are the versions used in the paper;
rerunning overwrites them, and `git diff` shows any difference. The scripts also
write files that are not committed and that git ignores: `out/refusal_*.csv` and
`out/refusal_flips.tex` (`refusal_flips.py`), `out/reranker_holes.tex` and
`out/reranker_holes.png` (`reranker_holes.py`), and a PNG next to each figure PDF
(`paper_figs.py`). Table 2 takes nDCG@10 and the reader AUROCs from
`data/drive/reteco/mtrag/summary.csv`, which `mtrag.ipynb` writes, and the label
shares from `mtrag_anatomy.json`. Table 4 counts the model outputs that the
results rest on.

**Figure 3 and the audit sample.** If `data/drive/audit/sample.csv` (the text of
blind-audit batch 1) is present, `false_miss_anatomy_v2.py` checks that the
document order in it matches the released `context_ids`. The file contains
benchmark text and is not released (see [docs/DATA.md](docs/DATA.md)), so on the
released data the script skips this check and writes
`"not checked (sample.csv not released)"` for
`audit1_top5_order_matches_context_ids` in `anatomy_summary.json`, where the
committed file has `"100/100"`. No reported number depends on the check. The
cell "8. Sample for the blind audit, batch 1 → Drive/reteco/audit" of
`generate.ipynb` writes `sample.csv` from the dataset and the BM25-fusion dev run.

**Blind audit.** `audit_build.py <out_dir>` rebuilds the 150 items of batch 2
from the dataset and the dev runs. `audit2_collect.py` collected the two
annotation runs and the adjudication into `docs/audit_llm2.csv`; its input, the
raw output of the annotation runs, is not released, so `docs/audit_llm2.csv` and
`docs/audit_llm.csv` are the record of the audit.

### 4. Retrieval (Table 3)

nDCG@10 and recall@100 of every retrieval run are stored in
`data/drive/reteco/analysis/runs.csv`, written by `analysis.ipynb`. Any released
run can be rescored with the task metric (pytrec_eval nDCG@10, averaged over
turns within a domain, then over the 11 domains):

```bash
python scripts/score.py --runs data/drive/reteco/runs/hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d_dev --split dev
```

The two BM25 baselines are deterministic and can be regenerated with the
starter kit's BM25 (k1 = 0.9, b = 0.4):

```bash
python scripts/retrieve.py --method bm25 --query query --split dev --out runs/bm25_query_dev
python scripts/retrieve.py --method bm25 --query hist  --split dev --out runs/bm25_hist_dev
python scripts/score.py --runs runs/bm25_hist_dev --split dev
```

BM25-fusion sums the min-max normalised scores of three BM25 runs over LLM query
expansions (`scripts/rewrite.py` writes the variants to
`runs/rewrites/<variant>/<split>.jsonl`; the paper's expansions were generated by
Claude Sonnet 5, Appendix A):

```bash
python scripts/retrieve.py --method bm25 --queries runs/rewrites/q2d/dev.jsonl     --split dev --out runs/bm25_q2d_dev
python scripts/retrieve.py --method bm25 --queries runs/rewrites/qd_q2d/dev.jsonl  --split dev --out runs/bm25_qd_q2d_dev
python scripts/retrieve.py --method bm25 --queries runs/rewrites/qd_hyde/dev.jsonl --split dev --out runs/bm25_qd_hyde_dev
python scripts/fuse.py --mode score --runs runs/bm25_q2d_dev runs/bm25_qd_q2d_dev runs/bm25_qd_hyde_dev \
    --tag bm25_fuse3 --out runs/bm25_fuse3_dev
```

The query expansions themselves are not part of this release, so the released
runs are the reference for BM25-fusion and the hybrid. The hybrid mixes the
min-max normalised scores of BM25-fusion and reason-embed-qwen3-4b over the
Query2doc query with dense weight 0.7 (`analysis.ipynb`); fusion and mixing
weights were chosen on train.

To check the environment against the organizers' published baseline, run the
starter kit's baseline on dev and compare (`docs/baseline_bm25.csv` holds the
reference scores):

```bash
python vendor/RETECO/starter_kit/official_baseline.py --data data/reteco --out runs/baseline \
    --splits dev --track1 --track2 biology drones earth_science economics hardware law \
    medicalsciences politics psychology robotics sustainable_living
python scripts/score.py --check-baseline
```

### 5. Model inference (Colab)

All open-model inference ran in Google Colab on a single NVIDIA A100 (40 or
80 GB) or L4 GPU; readers and judges ran with vLLM 0.31 in bfloat16 with greedy
decoding (vLLM seed 0). The notebooks in `colab/` are generated from
`colab/build.py`; edit that file and rebuild rather than editing a notebook:

```bash
python colab/build.py      # rewrites colab/*.ipynb
```

To rerun a step:

1. Copy the unpacked `data/drive/reteco/` folder to Google Drive and set `DRIVE`
   in cell 1 of the notebook to its path (default
   `/content/drive/MyDrive/RETECO 2026/reteco`). The RECOR notebooks read runs
   from `DRIVE/runs/<run>_<split>/` and check that `DRIVE/runs/bm25_hist_dev`
   exists.
2. Open the notebook in Colab, select the runtime given below and choose
   Run all. Each notebook mounts Drive and downloads its data: RETECO track 2
   from Hugging Face, or MTRAG from GitHub in `mtrag.ipynb`. The inference
   notebooks skip outputs that already exist, so an interrupted run resumes.
3. Copy the new output folder from `DRIVE` back into `data/drive/reteco/`
   (`audit/` goes to `data/drive/audit/`). The **Next.** note at the top of each
   notebook names the folders to copy.

The notebooks install the current vLLM release, download the current revision
of the RETECO dataset and clone the current starter kit. For an exact rerun:

- pin `vllm` in the install cell (the paper used vLLM 0.31);
- add `revision="ea71a271d80711a23a22cb4830ac3186f0df711a"` to the
  `snapshot_download` call in the data cell ("3. RETECO data, track 2");
- in `query.ipynb` and `analysis.ipynb`, which clone the starter kit, run
  `!git -C /content/reteco_kit checkout 23093c30e568a337c6293de4c58ffcccddbded5b`
  after the clone.

Models are loaded by Hugging Face id without a pinned revision. The
`config.json` files in `gen/`, `suff/` and `mtrag/` record the settings, GPU and
date of the runs.

| Notebook | Writes (under `DRIVE`) | Used for | Runtime suggested by the notebook |
|---|---|---|---|
| `generate.ipynb` | `gen/<reader>_<job>_dev.jsonl`, `audit/sample.csv`, `audit/sample_key.csv` | reader responses (4 readers × 6 contexts), audit batch 1 | A100 (preferred) or L4 |
| `suff.ipynb` | `suff/<judge>_<job>_dev.jsonl` | content labels (phi-4; Qwen3-14B via `JUDGE`) and the negative control | A100 |
| `judge.ipynb` | `judge/<judge>/<generation file>.jsonl` | answer-quality judgments (set `JUDGE` to `Qwen/Qwen3-8B` or `microsoft/phi-4`) | L4 (Qwen3-8B) or A100 (phi-4) |
| `transfer.ipynb` | `transfer/art_gen.jsonl`, OCC train responses in `gen/`, `analysis/transfer*.csv` | detector transfer (SQuAD 2.0, MuSiQue, RECOR train) | A100 or L4 |
| `mtrag.ipynb` | `mtrag/` | Replication on MTRAG (Section 7): retrieval, phi-4 labels, readers, summary | A100 |
| `reason_embed.ipynb` | `runs/dense_hanhainebula_reason-embed-qwen3-4b-0928_<query>_<split>/` | dense half of the hybrid, Table 3 | T4 (A100 faster) |
| `diver.ipynb` | `runs/dense_AQ-MedAI_Diver-Retriever-4B_<query>_<split>/` | Table 3 | T4 (A100 faster) |
| `query.ipynb` | `queries/<variant>/`, `runs/bm25_<variant>_<split>/` | Query2doc by an open LLM, Table 3 | A100 or L4 |
| `analysis.ipynb` | `runs/hyb_*`, `analysis/runs.csv` | hybrids, Table 3 numbers | CPU |
| `rerank.ipynb` | `runs/rerank_<first stage>_<query>/` | Qwen3-Reranker-4B, Appendix B | T4 |
| `llm_rerank.ipynb` | `runs/llmrr_<model>_<first stage>_<query>/` | ReasonRank-7B, Appendix B | L4 or A100 |

Notes on matching the paper's settings: in `query.ipynb`, set
`LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"` (the default `"auto"` picks
Qwen3-8B on an A100). The dense notebooks pick up query variants found in
`DRIVE/queries/`; the Query2doc expansions used by the hybrid are not released.
`mtrag.ipynb` downloads MTRAG from
[IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark) at commit
`53fef6f5917c7e10c436f697e69a955dbc795d81` and also writes the text files that
this release leaves out (`mtrag/passages.jsonl`, the text fields of
`mtrag/turns.jsonl`).

Models (Hugging Face ids): readers `occ-ai/OCC-RAG-1.7B`, `Qwen/Qwen3-1.7B`,
`Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B`; sufficiency judges
`microsoft/phi-4`, `Qwen/Qwen3-14B`; answer judges `Qwen/Qwen3-8B`,
`microsoft/phi-4`; dense retrieval `hanhainebula/reason-embed-qwen3-4b-0928`,
`AQ-MedAI/Diver-Retriever-4B`, `Qwen/Qwen3-Embedding-0.6B` (MTRAG); rerankers
`Qwen/Qwen3-Reranker-4B`, `liuwenhan/reasonrank-7B`. Qwen3 models run with
thinking disabled. The blind audit used Claude Opus 5.5 and the query expansions
Claude Sonnet 5 (Appendix A).

## Data release

The release archives are assets of the
[v1.0 release](https://github.com/iskochergin/reteco-2026/releases/tag/v1.0)
(links in the table below); the release notes give their SHA-256 checksums.
They unpack at the repository root into `data/drive/` (step 2 above). Formats and fields are described in
[docs/DATA.md](docs/DATA.md), licenses and required attributions in
[DATA_LICENSE.md](DATA_LICENSE.md). No field of the archives holds RECOR or
MTRAG corpus passages, questions or reference answers (model outputs may quote
short fragments of them); obtain the text from the upstream releases.

| Archive | Contents | License |
|---|---|---|
| [`reteco-runs.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-runs.zip) | TREC runs (top-100, 11 domains) of the four ladder retrievers: `bm25_query_{dev,train}`, `bm25_hist_{dev,train}`, `bm25_fuse3_{dev,train}`, `hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d_{dev,train}`; the two reranker runs on BM25-fusion dev (`rerank_bm25_fuse3_dev_qd`, Qwen3-Reranker-4B; `llmrr_reasonrank-7B_bm25_fuse3_dev_qd_k20w20s10`, ReasonRank-7B); a `config.json` per run | CC BY 4.0 |
| [`reteco-runs-extra.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-runs-extra.zip) (optional) | runs of further Table 3 systems, dev and train: BM25 with a Query2doc passage (by Claude and by an open LLM), reason-embed over Query2doc (both), Diver over Query2doc, and the hybrid with Diver | CC BY 4.0 |
| [`reteco-model-outputs.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-model-outputs.zip) | `gen/`: reader responses, 4 readers × {gold, no documents, 4 retrievers} on dev (20,592), and OCC on train with the BM25-hist and BM25-fusion top-5 (4,226); `judge/phi-4/`, `judge/Qwen3-8B/`: answer-quality judgments of OCC and Qwen3-4B (17,160); `transfer/art_gen.jsonl`: OCC on 2,000 SQuAD 2.0 and MuSiQue items | CC BY-SA 4.0 for generated text; CC BY 4.0 for ids, labels, scores and settings |
| [`reteco-labels-audit.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-labels-audit.zip) | `suff/`: content labels of phi-4 and Qwen3-14B for gold, the 4 retrievers and the negative control (10,296), with the judge prompt in `suff/config.json`; `data/drive/audit/sample_key.csv`; `analysis/transfer_content.csv`, `transfer.csv`, `runs.csv` | CC BY 4.0; CC BY-SA 4.0 for judge rationales and raw outputs |
| [`mtrag.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/mtrag.zip) | `mtrag/runs/` (5 retrievers × 4 domains), `retrieval.csv`, `summary.csv`, `config.json`; `mtrag/suff/`: phi-4 labels (4,662 contexts, 2,724 single passages); `mtrag/gen/`: OCC and Qwen3-4B responses (9,434); `mtrag/turns.jsonl` with ids, gold, answerability, annotator judgments and top-5 ids only; a copy of the Apache License 2.0 | Apache-2.0 for MTRAG turns and judgments; CC BY 4.0 for our runs, labels and numbers; CC BY-SA 4.0 for generated text |

`bm25_query_train` (BM25-turn on train) is the starter kit's official BM25
output `run_2a_train.trec`, copied unchanged; its `config.json` says so. The
Table 3 rows "+ LLM rewrite", "+ HyDE passage" and "reason-embed-4B, history"
have no released runs; their scores are in `analysis/runs.csv`.

The blind-audit annotations (`docs/audit_llm.csv`, `docs/audit_llm2.csv`) are
committed to this repository: their `rationale` column is CC BY-SA 4.0, all
other columns CC BY 4.0. The aggregate results under `docs/paper/out`, the
generated tables and the figures are CC BY 4.0, except for the short benchmark
excerpts noted in [DATA_LICENSE.md](DATA_LICENSE.md). Nothing in this
repository is derived from the hidden SemEval-2027 test set.

## Citation

```bibtex
@misc{kochergin2026better,
  title  = {Better Retrievers, Worse Labels: How Incomplete Relevance Judgments
            Distort Abstention Evaluation in Conversational {RAG}},
  author = {Kochergin, Ivan},
  year   = {2026},
  note   = {arXiv preprint}
}
```

If you use the data, please also cite the benchmarks it is derived from:

```bibtex
@misc{abdallah2026reteco,
  title        = {{RETECO}: Reasoning-Oriented Retrieval ({SemEval}-2027 Task 1)},
  author       = {Abdallah, Abdelrahman and Ali, Mohammed and Abdul-Mageed, Muhammad and Duh, Kevin and Jatowt, Adam},
  year         = {2026},
  howpublished = {\url{https://datascienceuibk.github.io/RETECO/}}
}

@inproceedings{ali2026recor,
  title     = {{RECOR}: Reasoning-focused Multi-turn Conversational Retrieval Benchmark},
  author    = {Ali, Mohammed and Abdallah, Abdelrahman and Agarwal, Amit and Patel, Hitesh Laxmichand and Jatowt, Adam},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026},
  pages     = {2688--2723},
  doi       = {10.18653/v1/2026.findings-acl.129}
}

@inproceedings{su2025bright,
  title     = {{BRIGHT}: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  author    = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S. and Tang, Michael and Sun, Ruoxi and Yoon, Jinsung and Arik, Sercan O. and Chen, Danqi and Yu, Tao},
  booktitle = {The Thirteenth International Conference on Learning Representations ({ICLR})},
  year      = {2025}
}

@article{katsis2025mtrag,
  title   = {{MTRAG}: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems},
  author  = {Katsis, Yannis and Rosenthal, Sara and Fadnis, Kshitij and Gunasekara, Chulaka and Lee, Young-Suk and Popa, Lucian and Shah, Vraj and Zhu, Huaiyu and Contractor, Danish and Danilevsky, Marina},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {13},
  pages   = {784--808},
  year    = {2025},
  doi     = {10.1162/tacl.a.19}
}
```

## License

The code (`scripts/`, `docs/paper/scripts/`, `colab/`) is released under the
MIT License ([LICENSE](LICENSE)). The paper source in `docs/paper/latex/` and the
documentation (`README.md`, `docs/DATA.md`, `DATA_LICENSE.md`) are released
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), except the
benchmark excerpts in Figure 1, which are CC BY-SA 4.0. The data (release
archives, audit annotations, aggregate results and figures) is released under
the licenses listed in [DATA_LICENSE.md](DATA_LICENSE.md), which also gives the
attributions required by the upstream licenses.

This repository does not contain the RETECO starter kit
([DataScienceUIBK/RETECO](https://github.com/DataScienceUIBK/RETECO)), which
`scripts/retrieve.py` imports unmodified at runtime, nor any RECOR or MTRAG text
beyond the short excerpts listed in [DATA_LICENSE.md](DATA_LICENSE.md).
Task site: [RETECO](https://datascienceuibk.github.io/RETECO/). MTRAG:
[IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark) (Apache-2.0).
The listwise reranking prompt in `llm_rerank.ipynb` follows the ReasonRank code,
and the Qwen3-Reranker input format follows its model card.
