# Better Retrievers, Worse Labels: How Incomplete Relevance Judgments Distort Abstention Evaluation in Conversational RAG

Code, retrieval runs, model outputs, content labels and audit annotations for
the paper of the same title by Ivan Kochergin (Letovo School) and Vasily
Konovalov (MIRAI), 2026. The arXiv link will be added here.

**Abstract.** Conversational RAG benchmarks such as MTRAG and RETECO evaluate
retrieval and generation end to end: each system answers from its own
retrieval, so abstention must be scored against whether each retrieved context
is sufficient. A convenient shortcut reads sufficiency off the relevance
judgments and treats a context without gold documents as unanswerable. We re-run
the official baselines exactly and test this shortcut on RECOR (RETECO track 2)
with four retrievers (nDCG@10 from .18 to .59) and three Qwen3 readers. Two open
LLM judges and a panel of four small open models label each context by content,
and a model of another family audits 250 contexts blind, with adjudication. The
stronger the retriever, the larger the share of no-gold contexts that still
contain the answer (28% to 72%) and the lower the shortcut's precision (.68 to
.10; .70 to .21 under the audit). A reader agrees: Qwen3-4B answers 62–67% of
these false misses correctly, against 23–30% of no-gold contexts judged
insufficient and 1% with no documents. In most false misses, the answer comes
from a hard negative mined for the same question, another turn's gold or a
neighbouring chunk of the gold page. On MTRAG, whose judgments come from
annotators, precision drops from .72 to .50. Under the shortcut, answerability
AUROC is up to .19 too low, and the best sufficiency detector appears to be a
retrieval score that never reads the documents. We release code, runs and
labels, and recommend scoring refusals against content labels, not the presence
of gold.

## Main results

**RECOR dev** (858 turns, 11 domains, top-5 contexts). Precision: among no-gold
contexts (qrels label `none`) that a labeller labels `yes` or `no`, the share
labelled `no`; `partial` is excluded. False misses: the share of all no-gold
contexts that the judge labels `yes`. Copies of gold: the share of no-gold
contexts that hold a near-verbatim copy of a gold document of the turn, found
without any judge. The blind audit covers 50 no-gold contexts per retriever.
Confidence intervals are in Table 1 and Section 4 of the paper.

| | BM25-turn | BM25-hist | Fusion | Hybrid |
|---|---:|---:|---:|---:|
| nDCG@10 | .183 | .438 | .550 | .591 |
| Contexts without gold (%) | 71.6 | 35.0 | 22.7 | 18.4 |
| Copies of gold among them, no judge (%) | 1.6 | 6.0 | 16.4 | 9.5 |
| False misses, phi-4 (%) | 28 | 45 | 68 | 72 |
| Precision of "no gold", phi-4 | .68 | .44 | .18 | .10 |
| Precision of "no gold", Qwen3-14B | .84 | .69 | .29 | .21 |
| Precision of "no gold", panel of four small judges | .49 | .19 | .04 | .02 |
| Precision of "no gold", blind audit | .70 | .62 | .23 | .21 |

**MTRAG** (777 answerable and partially answerable turns, phi-4 judge, five
first-stage retrievers ordered by nDCG@10; definitions as above; Table 3 of the
paper).

| | BM25, all questions | BM25, last turn | BM25, rewrite | Hybrid | Dense |
|---|---:|---:|---:|---:|---:|
| nDCG@10 | .18 | .22 | .25 | .40 | .42 |
| Contexts without gold (%) | 71 | 64 | 61 | 40 | 37 |
| False misses (%) | 23 | 23 | 27 | 40 | 38 |
| Precision of "no gold" | .72 | .72 | .67 | .50 | .50 |
| Qwen3-4B AUROC, qrels label | .73 | .76 | .70 | .67 | .68 |
| Qwen3-4B AUROC, content label | .74 | .80 | .78 | .77 | .75 |

## What is in the repository

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
  audit_llm.csv         blind audit, batch 1: 100 Fusion contexts
  audit_llm2.csv        blind audit, batch 2: 150 no-gold contexts (BM25-turn, BM25-hist, Hybrid)
  paper/
    latex/              paper source: main.tex, custom.bib, fig_bias.pdf (Figure 2), fig_auroc.pdf (Figure 3)
    scripts/            analysis scripts behind the paper's figures and numbers (step 3 maps each to its source)
    out/v2/             summary files written by the scripts (CSV, JSON)
    out/three_readers/  consequences_v2.py with the paper's three Qwen3 readers (Section 4.3)
    out/*.json, *.csv   outputs of two analyses that the paper does not report
data/                   empty in git; the dataset and the release archives are unpacked here
runs/                   empty in git; output of scripts/retrieve.py and scripts/fuse.py
```

The ACL style files needed to compile `main.tex` (`acl.sty`, `acl_natbib.bst`)
are not included; they are distributed with the ACL paper template.

Names: the paper calls the retrievers BM25-turn, BM25-hist, Fusion and Hybrid;
in file names and code they are `bm25_query`, `bm25_hist`, `bm25_fuse3` (also
called BM25-fusion) and `hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d`.

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
into the repository root, check them and unzip them there. Five archives are
assets of the [v1.0 release](https://github.com/iskochergin/reteco-2026/releases/tag/v1.0)
and two of the [v1.1 release](https://github.com/iskochergin/reteco-2026/releases/tag/v1.1):
`reteco-labels-small.zip`, the labels of the four small judges, and
`reteco-expansions.zip`, the LLM query expansions, which only step 4 needs:

```bash
BASE=https://github.com/iskochergin/reteco-2026/releases/download
for a in reteco-runs reteco-model-outputs reteco-labels-audit mtrag reteco-runs-extra; do
    curl -LO "$BASE/v1.0/$a.zip"
done
for a in reteco-labels-small reteco-expansions; do
    curl -LO "$BASE/v1.1/$a.zip"
done
shasum -a 256 *.zip             # or sha256sum; compare with the release pages
unzip -o reteco-runs.zip
unzip -o reteco-model-outputs.zip
unzip -o reteco-labels-audit.zip
unzip -o mtrag.zip
unzip -o reteco-labels-small.zip
unzip -o reteco-runs-extra.zip  # optional: further retrieval runs
unzip -o reteco-expansions.zip  # optional: query expansions (step 4)
```

The release notes on each release page list the size, file count and SHA-256
checksum of its archives.

The result is the layout the scripts expect:

```
data/reteco/track2_recor/<domain>/   documents.jsonl, benchmark_{train,dev}.json, qrels_{train,dev}.txt
data/drive/reteco/runs/              retrieval runs
data/drive/reteco/gen/               reader responses
data/drive/reteco/judge/             answer-quality judgments
data/drive/reteco/suff/              content labels (phi-4, Qwen3-14B and the four small judges)
data/drive/reteco/transfer/          OCC outputs on SQuAD 2.0 and MuSiQue (not used by the paper)
data/drive/reteco/analysis/          summary tables written by the notebooks
data/drive/reteco/mtrag/             MTRAG runs, labels and reader responses
data/drive/audit/sample_key.csv      key of blind-audit batch 1
data/drive/LICENSE-<archive>.md      license notice of each archive
runs/rewrites/raw/                   LLM query expansions (reteco-expansions.zip)
docs/audit_llm.csv, docs/audit_llm2.csv   blind-audit annotations (in git)
```

### 3. Figures, tables and numbers

All analysis runs on a CPU from the saved model outputs, in seconds to minutes.
Run the scripts from the repository root, in this order (`paper_figs.py` reads
files that the scripts before it write):

```bash
python docs/paper/scripts/label_v2.py
python docs/paper/scripts/small_judges.py
python docs/paper/scripts/judge_free.py
python docs/paper/scripts/control_v2.py
python docs/paper/scripts/leak_check.py
python docs/paper/scripts/false_miss_anatomy_v2.py
RETECO_EXCLUDE=OCC-RAG-1.7B RETECO_OUT=docs/paper/out/three_readers \
    python docs/paper/scripts/consequences_v2.py
python docs/paper/scripts/consequences_v2.py          # all four readers -> out/v2 (auroc.csv for Figure 3)
python docs/paper/scripts/baselines_v2.py
python docs/paper/scripts/mtrag_v2.py
RETECO_FIG_SMALL=1 RETECO_FIG_EXCLUDE=OCC-RAG-1.7B python docs/paper/scripts/paper_figs.py
```

`RETECO_EXCLUDE=OCC-RAG-1.7B` drops the fourth reader, OCC-RAG-1.7B, which comes
from an earlier version of the analysis; the paper uses the three Qwen3 readers.
Dropping a reader does not change the numbers of the other readers: the rows of
the three Qwen3 readers in `out/v2/{auroc,auroc_common,refusal,contrast,decomp}.csv`
equal those in `out/three_readers/`. The counts of changed conclusions do change,
because they include comparisons between readers; the paper's counts are in
`out/three_readers/consequences_summary.json`.

Where each part of the paper comes from. Scripts without a folder are in
`docs/paper/scripts/`; paths that start with `out/` or `latex/` are relative to
`docs/paper/`, all other paths to the repository root.

| Paper | Script | Output |
|---|---|---|
| Figure 1 | none: a table typeset in `latex/main.tex` | |
| Section 3, official BM25 baselines re-run to four decimals | `scripts/retrieve.py`, `scripts/score.py --check-baseline` (step 4) | compared with `docs/baseline_bm25.csv` |
| Section 3, ladder nDCG@10; open-LLM Query2doc passages (.453 instead of .546) | `colab/analysis.ipynb`; `scripts/score.py` on any released run (step 4) | `data/drive/reteco/analysis/runs.csv` |
| Section 3, share of turns with gold in the top-5 (.28 to .82) | `label_v2.py` (1 − `share_none`) | `out/v2/label_stage.csv` |
| Section 3, responses without a status line (at most 75 of 858) | `consequences_v2.py`, three readers | `n_empty_status` in `out/three_readers/consequences_summary.json` |
| Section 3, agreement of the two audit runs (97%) | `label_v2.py` (from columns `A` and `B` of `docs/audit_llm*.csv`) | row `claude_A vs claude_B` of `out/v2/agreement.csv` |
| Section 3, dialogue-level bootstrap (bounds move by at most .03) | `label_v2.py` | `conv_bootstrap` in `out/v2/label_v2.json` |
| Section 4.1 and Figure 2 (numbers) | `label_v2.py` | `out/v2/label_stage.csv`, `label_deltas.csv`, `label_domains.csv`, `audit_stage.csv`, `label_v2.json` |
| Figure 2 (drawing; grey lines: small judges) | `paper_figs.py` with `RETECO_FIG_SMALL=1` | `latex/fig_bias.pdf` |
| Section 4.2, "Without any judge" | `judge_free.py` (reads the RECOR corpus) | `out/v2/judge_free.csv`, `judge_free.json` |
| Table 1 (precision of every labeller, change from BM25-turn to Hybrid, κ with the audit, negative control) | `small_judges.py` (every judge and the panel); the blind-audit row from `label_v2.py` | `out/v2/small_judges.csv`, `small_judges.json`; audit row: `out/v2/audit_stage.csv` |
| Section 4.2, "Other labellers" (Cohen's κ, Fleiss' κ, Gwet's AC1, raw agreement) | `label_v2.py` (phi-4, Qwen3-14B, audit); `small_judges.py` (small judges) | `out/v2/agreement.csv`, `small_judges.json` |
| Section 4.2, "Negative control" (phi-4 and Qwen3-14B, leniency correction) | `control_v2.py` | `out/v2/control.csv`, `control.json` |
| Table 2, Section 4.2 "A reader answers from false-miss contexts" | `leak_check.py` | `out/v2/leak_check.csv`, `leak_check.json` |
| Section 4.2, "Human judgments on MTRAG" | `colab/mtrag.ipynb` (judge on single passages); `mtrag_v2.py` (false misses resting on rejected passages) | `human_single` rows of `data/drive/reteco/mtrag/summary.csv`; `out/v2/mtrag_anatomy.json` |
| Section 4.2, "What a false miss contains" | `false_miss_anatomy_v2.py` | `out/v2/anatomy*.csv`, `anatomy_summary.json` |
| Figure 3, Section 4.3 "Detector quality is understated" (numbers, incl. AUROC on contexts both labels keep) | `consequences_v2.py` with `RETECO_EXCLUDE=OCC-RAG-1.7B RETECO_OUT=docs/paper/out/three_readers` | `out/three_readers/auroc.csv`, `auroc_common.csv` |
| Figure 3 (drawing) | `paper_figs.py` with `RETECO_FIG_EXCLUDE=OCC-RAG-1.7B` (reads `out/v2/auroc.csv` and drops OCC) | `latex/fig_auroc.pdf` |
| Section 4.3, "The qrels label hides the 2b/2c contrast" | `consequences_v2.py`, three readers | `out/three_readers/contrast.csv` (`def` = `v1`: pooled; `same_turn`: within turns) |
| Section 4.3, "The wrong detector wins" | `baselines_v2.py` (rows of the retrieval scores and the three Qwen3 readers) | `out/v2/detectors.csv`, `detectors_pairs.csv`, `baselines.json` |
| Section 4.3, "How many conclusions change" (258 comparisons, 75 changes, 13 by sign; judge swap: 25; answer-judge check: 4 and 6 of 7) | `consequences_v2.py`, three readers | `flips` → `phi-4` → `extended, verdict` and `flips_judge_vs_judge` in `out/three_readers/consequences_summary.json`; `flips.csv`, `decomp.csv`, `refusal.csv` |
| Table 3, Section 4.4 (nDCG@10, label shares, precision, false misses, Qwen3-4B AUROC) | `colab/mtrag.ipynb` | `labels` and `readers` rows of `data/drive/reteco/mtrag/summary.csv` |
| Section 4.4, what the dense retriever's false misses cite | `mtrag_v2.py` | `out/v2/mtrag_anatomy.csv`, `mtrag_anatomy.json` |
| Section 4.4, Qwen3-4B's within-turn 2b/2c contrast on the hybrid (+.22 on 58 turns under the content label, significant; +.01 under qrels, not; no status line on 13–23% of contexts) | `mtrag_contrast.py` | `out/v2/mtrag_contrast.json` (`paper_same_turn`; `no_status`) |

The figures need only the committed files under `docs/paper/out/v2/`:

```bash
RETECO_FIG_SMALL=1 RETECO_FIG_EXCLUDE=OCC-RAG-1.7B python docs/paper/scripts/paper_figs.py
```

writes `fig_bias.pdf` and `fig_auroc.pdf` into `docs/paper/latex/` (with
matplotlib 3.10.8 and Times New Roman installed, they differ from the committed
PDFs only in their creation date), a PNG next to each,
and `fig_anatomy.pdf`, a chart of the false-miss classes that is not in the
paper (git ignores it). `paper_figs.py` describes its other settings in its
docstring.

The committed files under `docs/paper/out` and `docs/paper/latex` are the
versions used in the paper; rerunning overwrites them, and `git diff` shows any
difference.

**MTRAG 2b/2c contrast.** For Qwen3-4B on the MTRAG hybrid, Section 4.4 reports
the within-turn 2b/2c contrast as defined on RECOR (`def` = `same_turn` in
`consequences_v2.py`): turns whose hybrid context is insufficient and whose gold
context is sufficient under the same label, responses without a status line
excluded as stated in Section 3. It is +.22 [.06, .38] on 58 turns under the
content label and +.01 [−.06, .09] under the qrels label. The `contrast_c` and
`contrast_q` rows of `summary.csv` for `hybrid_rewrite` (+.10 and +.02) use the
definition of `mtrag.ipynb` instead: the gold context need not be sufficient,
responses without a status line count as answers, and the bootstrap resamples
turns within a domain. `mtrag_contrast.py` computes the paper's numbers and, as a
check, reproduces the notebook's rows.

**Audit sample check.** If `data/drive/audit/sample.csv` (the text of blind-audit
batch 1) is present, `false_miss_anatomy_v2.py` checks that the document order
in it matches the released `context_ids`. The file contains benchmark text and
is not released (see [docs/DATA.md](docs/DATA.md)), so on the released data the
script skips this check and writes `"not checked (sample.csv not released)"` for
`audit1_top5_order_matches_context_ids` in `anatomy_summary.json`, where the
committed file has `"100/100"`. No reported number depends on the check. The
cell "8. Sample for the blind audit, batch 1 → Drive/reteco/audit" of
`generate.ipynb` writes `sample.csv` from the dataset and the Fusion dev run.

**Blind audit.** `audit_build.py <out_dir>` rebuilds the 150 items of batch 2
from the dataset and the dev runs. `audit2_collect.py` collected the two
annotation runs and the adjudication into `docs/audit_llm2.csv`; its input, the
raw output of the annotation runs, is not released, so `docs/audit_llm2.csv` and
`docs/audit_llm.csv` are the record of the audit.

**Scripts and outputs that the paper does not use.** `refusal_flips.py` (the
first version of the refusal analysis, two readers; `out/refusal_flips.json`),
`reranker_holes.py` (rerankers over Fusion; `out/reranker_holes*`) and
`tables_v2.py` (LaTeX tables of an earlier, longer version of the paper; writes
to `out/tables/`, which git ignores) are kept for completeness. In `out/v2/`,
the four-reader files written by `consequences_v2.py` without variables
(`refusal.csv`, `auroc.csv`, `auroc_common.csv`, `contrast.csv`, `decomp.csv`,
`flips.csv`, `consequences_summary.json`) and `selective.csv` are not reported
in the paper, except that Figure 3 is drawn from the Qwen3 rows of `auroc.csv`.

### 4. Retrieval (Section 3)

nDCG@10 and recall@100 of every retrieval run are stored in
`data/drive/reteco/analysis/runs.csv`, written by `analysis.ipynb`. Any released
run can be rescored with the task metric (pytrec_eval nDCG@10, averaged over
turns within a domain, then over the 11 domains):

```bash
python scripts/score.py --runs data/drive/reteco/runs/hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d_dev --split dev
python scripts/score.py --runs data/drive/reteco/runs/bm25_q2d_open_dev --split dev   # reteco-runs-extra.zip
```

The two BM25 baselines are deterministic and can be regenerated with the
starter kit's BM25 (k1 = 0.9, b = 0.4):

```bash
python scripts/retrieve.py --method bm25 --query query --split dev --out runs/bm25_query_dev
python scripts/retrieve.py --method bm25 --query hist  --split dev --out runs/bm25_hist_dev
python scripts/score.py --runs runs/bm25_hist_dev --split dev
```

Fusion sums the min-max normalised scores of three BM25 runs over LLM query
expansions. The paper's expansions, generated by Claude Sonnet 5 (Section 3),
are in `reteco-expansions.zip`: the raw responses that `scripts/rewrite.py`
cached in `runs/rewrites/raw/`. With `--build-only`, the script rebuilds the
query variants from them and the RETECO data without calling a model and writes
them to `runs/rewrites/<variant>/<split>.jsonl`:

```bash
python scripts/rewrite.py --split dev --build-only
python scripts/retrieve.py --method bm25 --queries runs/rewrites/q2d/dev.jsonl     --split dev --out runs/bm25_q2d_dev
python scripts/retrieve.py --method bm25 --queries runs/rewrites/qd_q2d/dev.jsonl  --split dev --out runs/bm25_qd_q2d_dev
python scripts/retrieve.py --method bm25 --queries runs/rewrites/qd_hyde/dev.jsonl --split dev --out runs/bm25_qd_hyde_dev
python scripts/fuse.py --mode score --runs runs/bm25_q2d_dev runs/bm25_qd_q2d_dev runs/bm25_qd_hyde_dev \
    --tag bm25_fuse3 --out runs/bm25_fuse3_dev
```

Compare the result with the released `bm25_fuse3_dev` run, for example with
`scripts/score.py`. The hybrid mixes the min-max normalised scores of Fusion and
reason-embed-qwen3-4b over the Query2doc query (the `q2d` variant) with dense
weight 0.7 (`analysis.ipynb`); fusion and mixing weights were chosen on train.
To recompute its dense half, copy `runs/rewrites/q2d/` to `DRIVE/queries/q2d/`
and run `reason_embed.ipynb` (step 5).

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

All open-model inference ran in Google Colab on a single GPU. Readers and judges
ran on an NVIDIA A100 (40 or 80 GB) or L4 with vLLM 0.31 in bfloat16 with greedy
decoding (vLLM seed 0); dense corpus embeddings were computed in float16, partly
on a T4 (Section 3). The notebooks in `colab/` are generated from
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
date of the runs (`suff/config_<model>.json` for the small judges).

| Notebook | Writes (under `DRIVE`) | Used for | Runtime suggested by the notebook |
|---|---|---|---|
| `generate.ipynb` | `gen/<reader>_<job>_dev.jsonl`, `audit/sample.csv`, `audit/sample_key.csv` | reader responses for gold, no documents and the four retrievers (Sections 3, 4.2, 4.3); it runs four readers, of which the paper uses the three Qwen3 readers; audit batch 1 | A100 (preferred) or L4 |
| `suff.ipynb` | `suff/<judge>_<job>_dev.jsonl` | content labels of phi-4 and Qwen3-14B (via `JUDGE`) and the negative control (Sections 4.1, 4.2, Table 1) | A100 |
| `suff_small.ipynb` | `suff/<model>_<job>_dev.jsonl`, `suff/config_<model>.json` | content labels of the four small judges, same prompt and contexts (Table 1, Figure 2) | A100 (or L4) |
| `judge.ipynb` | `judge/<judge>/<generation file>.jsonl` | answer-quality judgments (Table 2; refusals read from an answer judge in Section 4.3); set `JUDGE` to `Qwen/Qwen3-8B` or `microsoft/phi-4` | L4 (Qwen3-8B) or A100 (phi-4) |
| `mtrag.ipynb` | `mtrag/` | replication on MTRAG (Section 4.4, Table 3) and the human check of the judge (Section 4.2): retrieval, phi-4 labels, readers, summary | A100 |
| `reason_embed.ipynb` | `runs/dense_hanhainebula_reason-embed-qwen3-4b-0928_<query>_<split>/` | dense half of the hybrid (Section 3) | T4 (A100 faster) |
| `query.ipynb` | `queries/<variant>/`, `runs/bm25_<variant>_<split>/` | Query2doc by an open LLM (Section 3) | A100 or L4 |
| `analysis.ipynb` | `runs/hyb_*`, `analysis/runs.csv` | the hybrid and the nDCG@10 of all runs (Section 3) | CPU |
| `transfer.ipynb` | `transfer/art_gen.jsonl`, OCC train responses in `gen/`, `analysis/transfer*.csv` | detector transfer (SQuAD 2.0, MuSiQue, RECOR train); not reported in the paper | A100 or L4 |
| `diver.ipynb` | `runs/dense_AQ-MedAI_Diver-Retriever-4B_<query>_<split>/` | a further dense retriever; not reported in the paper | T4 (A100 faster) |
| `rerank.ipynb` | `runs/rerank_<first stage>_<query>/` | Qwen3-Reranker-4B; not reported in the paper | T4 |
| `llm_rerank.ipynb` | `runs/llmrr_<model>_<first stage>_<query>/` | ReasonRank-7B; not reported in the paper | L4 or A100 |

Notes on matching the paper's settings: in `query.ipynb`, set
`LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"` (the default `"auto"` picks
Qwen3-8B on an A100). In `generate.ipynb` and `mtrag.ipynb`, the readers the
paper uses are the Qwen3 models in `READERS`. The dense notebooks pick up query
variants found in `DRIVE/queries/`; for the hybrid, rebuild the variants from
`reteco-expansions.zip` and copy `runs/rewrites/q2d/` there (step 4).
`mtrag.ipynb` downloads MTRAG from
[IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark) at commit
`53fef6f5917c7e10c436f697e69a955dbc795d81` and also writes the text files that
this release leaves out (`mtrag/passages.jsonl`, the text fields of
`mtrag/turns.jsonl`).

Models (Hugging Face ids) used in the paper: readers `Qwen/Qwen3-1.7B`,
`Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B`; sufficiency judges
`microsoft/phi-4`, `Qwen/Qwen3-14B`; small sufficiency judges
`Qwen/Qwen3-4B-Instruct-2507`, `microsoft/Phi-4-mini-instruct`,
`HuggingFaceTB/SmolLM3-3B`, `ibm-granite/granite-3.3-8b-instruct`; answer judges
`Qwen/Qwen3-8B`, `microsoft/phi-4`; dense retrieval
`hanhainebula/reason-embed-qwen3-4b-0928`, `Qwen/Qwen3-Embedding-0.6B` (MTRAG).
Qwen3 models run with thinking disabled. The blind audit used Claude Opus 5.5
and the query expansions Claude Sonnet 5 (Section 3; released in
`reteco-expansions.zip`). The notebooks and the
archives also cover models that the paper does not use: the reader
`occ-ai/OCC-RAG-1.7B`, the dense retriever `AQ-MedAI/Diver-Retriever-4B` and
the rerankers `Qwen/Qwen3-Reranker-4B` and `liuwenhan/reasonrank-7B`.

## Data release

The release archives are assets of the
[v1.0](https://github.com/iskochergin/reteco-2026/releases/tag/v1.0) and
[v1.1](https://github.com/iskochergin/reteco-2026/releases/tag/v1.1) releases
(links in the table below); the release notes give their SHA-256 checksums. They
unpack at the repository root (step 2 above), into `data/drive/` and, for
`reteco-expansions.zip`, into `runs/rewrites/raw/`. Formats and fields are
described in [docs/DATA.md](docs/DATA.md), licenses and required attributions in
[DATA_LICENSE.md](DATA_LICENSE.md). No field of the archives holds RECOR or
MTRAG corpus passages, dialogue history or reference answers, and only the query
rewrites hold questions: 216 of the 2,971 rewrites in `reteco-expansions.zip`
repeat the RECOR question word for word, mostly on first turns, which need no
rewriting. Other model outputs may quote short fragments of what they read.
Obtain the text from the upstream releases.

The archives were built for an earlier version of the analysis that also used a
fourth reader, OCC-RAG-1.7B. The paper does not use it, but
`reteco-model-outputs.zip` and `mtrag.zip` still contain its outputs, listed
below. They are kept so that the released files stay unchanged. The paper's
numbers leave them out (step 3 shows how); some scripts still write OCC rows
next to the Qwen3 rows.

| Archive | Release | Contents | License |
|---|---|---|---|
| [`reteco-runs.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-runs.zip) | v1.0 | TREC runs (top-100, 11 domains) of the four retrievers: `bm25_query_{dev,train}`, `bm25_hist_{dev,train}`, `bm25_fuse3_{dev,train}`, `hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d_{dev,train}`; two reranker runs on Fusion dev that the paper does not report (`rerank_bm25_fuse3_dev_qd`, Qwen3-Reranker-4B; `llmrr_reasonrank-7B_bm25_fuse3_dev_qd_k20w20s10`, ReasonRank-7B); a `config.json` per run | CC BY 4.0 |
| [`reteco-runs-extra.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-runs-extra.zip) (optional) | v1.0 | further runs, dev and train: BM25 with a Query2doc passage by Claude (`bm25_q2d`) and by an open LLM (`bm25_q2d_open`; both in Section 3), reason-embed over Query2doc (both), Diver over Query2doc, and the hybrid with Diver | CC BY 4.0 |
| [`reteco-model-outputs.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-model-outputs.zip) | v1.0 | `gen/`: reader responses on dev for {gold, no documents, 4 retrievers}: 15,444 of the three Qwen3 readers, used by the paper, and 5,148 of OCC-RAG-1.7B; `judge/phi-4/`, `judge/Qwen3-8B/`: answer-quality judgments of Qwen3-4B and OCC-RAG-1.7B (17,160). It also contains outputs of a fourth reader, OCC-RAG-1.7B, from an earlier version of the analysis that the paper does not use: the OCC files in `gen/` and `judge/` above, OCC on train with the BM25-hist and Fusion top-5 (4,226) and `transfer/art_gen.jsonl` (OCC on 2,000 SQuAD 2.0 and MuSiQue items) | CC BY-SA 4.0 for generated text; CC BY 4.0 for ids, labels, scores and settings |
| [`reteco-labels-audit.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/reteco-labels-audit.zip) | v1.0 | `suff/`: content labels of phi-4 and Qwen3-14B for gold, the 4 retrievers and the negative control (10,296), with the judge prompt in `suff/config.json`; `data/drive/audit/sample_key.csv`; `analysis/runs.csv`; `analysis/transfer.csv`, `transfer_content.csv` (detector transfer, not reported in the paper) | CC BY 4.0; CC BY-SA 4.0 for judge rationales and raw outputs |
| [`reteco-labels-small.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.1/reteco-labels-small.zip) | v1.1 | `suff/`: content labels of the four small judges (Qwen3-4B-Instruct-2507, Phi-4-mini-instruct, SmolLM3-3B, Granite-3.3-8B-instruct) for gold, the 4 retrievers and the negative control (24 files × 858 = 20,592), with each judge's prompt and settings in `suff/config_<model>.json` | CC BY 4.0; CC BY-SA 4.0 for judge rationales and raw outputs |
| [`reteco-expansions.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.1/reteco-expansions.zip) (optional) | v1.1 | `runs/rewrites/raw/`: the query expansions of Claude Sonnet 5 behind Fusion, the hybrid and the `*_q2d` runs, one line per turn (858 dev, 2,113 train) with a self-contained rewrite, a topic-switch flag and a Query2doc/HyDE passage; the prompts and settings in `config_{dev,train}.json`. `scripts/rewrite.py --build-only` rebuilds the query variants from them (step 4) | CC BY-SA 4.0 for rewrites and passages; CC BY 4.0 for ids, flags, prompts and settings |
| [`mtrag.zip`](https://github.com/iskochergin/reteco-2026/releases/download/v1.0/mtrag.zip) | v1.0 | `mtrag/runs/` (5 retrievers × 4 domains), `retrieval.csv`, `summary.csv`, `config.json`; `mtrag/suff/`: phi-4 labels (4,662 contexts, 2,724 single passages); `mtrag/gen/`: responses of Qwen3-4B, used by the paper, and of OCC-RAG-1.7B, not used (9,434 in all); `mtrag/turns.jsonl` with ids, gold, answerability, annotator judgments and top-5 ids only; a copy of the Apache License 2.0 | Apache-2.0 for MTRAG turns and judgments; CC BY 4.0 for our runs, labels and numbers; CC BY-SA 4.0 for generated text |

`bm25_query_train` (BM25-turn on train) is the starter kit's official BM25
output `run_2a_train.trec`, copied unchanged; its `config.json` says so.

The blind-audit annotations (`docs/audit_llm.csv`, `docs/audit_llm2.csv`) are
committed to this repository: their `rationale` column is CC BY-SA 4.0, all
other columns CC BY 4.0. The aggregate results under `docs/paper/out` and the
figures are CC BY 4.0, except for the short benchmark excerpts noted in
[DATA_LICENSE.md](DATA_LICENSE.md). Nothing in this repository is derived from
the hidden SemEval-2027 test set.

## Citation

The arXiv link will be added.

```bibtex
@misc{kochergin2026better,
  title  = {Better Retrievers, Worse Labels: How Incomplete Relevance Judgments
            Distort Abstention Evaluation in Conversational {RAG}},
  author = {Kochergin, Ivan and Konovalov, Vasily},
  year   = {2026},
  note   = {Preprint; arXiv link will be added}
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
