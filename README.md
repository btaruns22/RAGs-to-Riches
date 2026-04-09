# RAGs-to-Riches

## Overview
RAGs-to-Riches tests whether retrieval improves LLM decision-making on a structured intraday trading task. The system does not ask the model to predict the market from scratch. It asks the model to inspect the first five SPY opening bars, compare that setup to historical examples and rules, and decide whether the setup should be treated as a `TAKE` or `PASS`.

The project now uses a strict train/test separation:
- training corpus for retrieval: February 14, 2023 through March 1, 2025
- testing/evaluation period: March 2, 2025 through present

The ChromaDB collection is built only from the training set so the test period is never inserted into the retrieval database.

## Research Question
**Does retrieval-augmented reasoning improve the reliability and decision accuracy of an LLM when evaluating a 5-minute SPY opening setup compared to a baseline LLM with no retrieval?**

## Project Structure

```text
main.py
project_config.py
services/
  s3_client.py
  openrouter_embeddings.py
pipeline/
  features.py
  dataset.py
trading_strategies/
  breakout_strategy.py
prompts/
  prompt_utils.py
  rag_prompt.py
rag/
  knowledge_base.py
  retriever.py
  vector_store.py
llm/
  baseline.py
  rag.py
  rag_manual.py
  rag_vector.py
evaluation/
  evaluation.py
data/generated/
```

## Data Pipeline

The pipeline reads two Massive flat-file datasets:
- `us_stocks_sip/minute_aggs_v1/` for SPY
- `us_indices/minute_aggs_v1/` for VIX via ticker `I:VIX`

### Feature Window
- `09:30` through `09:34` ET
- used to create the setup representation seen by the model and stored in retrieval memory

### Outcome Window
- `09:35` through `10:30` ET
- used only to create the path-dependent answer key

### Labeling Logic
The entry price is the `09:34` close.

For each trading day, scan the `09:35` through `10:30` SPY bars:
- if price reaches `entry_price * 1.003` before `entry_price * 0.998`, label `TAKE`
- if price reaches the stop first, label `FAIL_FAKEOUT`
- if neither threshold is hit by `10:30`, label `PASS`
- if a single bar touches both target and stop, treat it conservatively as `FAIL_FAKEOUT`

For LLM evaluation, the ternary outcome is also collapsed to a binary decision label:
- `TAKE` -> `TAKE`
- `FAIL_FAKEOUT` -> `PASS`
- `PASS` -> `PASS`

## Generated Datasets

Running `python main.py` writes the following files under `data/generated/`:

### `spy_open_setup_raw.csv`
Five rows per trading day for the setup window only.

Columns:
- `date`
- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`

This file is the direct LLM input. Baseline and RAG both see one day’s 5 raw SPY bars from this file.

### `spy_open_setup_features.csv`
One row per trading day containing engineered setup features plus outcome labels.

Columns:
- `date`
- `spy_open`
- `previous_close`
- `gap_pct`
- `first_1m_return`
- `net_movement`
- `opening_range_high`
- `opening_range_low`
- `opening_range_width`
- `breakout_direction`
- `volatility`
- `volume`
- `rvol_10d`
- `vix_at_open`
- `entry_price`
- `outcome_label`
- `label`
- `max_gain_reached`
- `max_drawdown_reached`

Definitions:
- `rvol_10d`: opening-window volume divided by the trailing 10-day average opening-window volume
- `vix_at_open`: open price from the first available `I:VIX` bar at or after `09:30` ET
- `entry_price`: SPY `09:34` close
- `outcome_label`: path-dependent ternary label from the outcome window
- `label`: binary evaluation target used by the LLM experiment
- `max_gain_reached`: best percent move from entry to any high in the outcome window
- `max_drawdown_reached`: worst percent move from entry to any low in the outcome window

This file serves two roles:
- retrieval memory
- ground-truth answer key for evaluation

## Retrieval Design

### Manual Retrieval
`rag/retriever.py` uses hard-coded similarity across key setup features:
- `gap_pct`
- `first_1m_return`
- `net_movement`
- `volatility`
- `rvol_10d`
- `vix_at_open`

### Vector Retrieval
`rag/vector_store.py` builds a local persistent Chroma collection from the training portion of `spy_open_setup_features.csv`.

Important rules:
- only dates from `2023-02-14` through `2025-03-01` are added to Chroma
- the test period is never inserted into the collection
- the current query date is excluded from retrieved examples

Each embedded historical document combines:
- the 5 raw SPY setup bars from `spy_open_setup_raw.csv`
- the engineered setup summary from `spy_open_setup_features.csv`
- historical outcome metadata and labels

Embeddings use OpenRouter with:
- API base: `https://openrouter.ai/api/v1`
- default model: `openai/text-embedding-3-small`

Chroma stays local on disk under `data/generated/chroma/`.

## Prompt Pipeline

### Baseline
- input: one date’s 5 raw SPY bars from `spy_open_setup_raw.csv`
- no retrieval
- output: `TAKE` or `PASS`

### RAG
- input: the same 5 raw SPY bars
- retrieved context:
  - strategy rules
  - similar historical setups from the training corpus
  - historical labels and outcomes for those retrieved setups
- output: `TAKE` or `PASS`

The current day’s ground-truth label is never shown in the prompt.

## Train/Test Protocol

- train/retrieval corpus:
  - February 14, 2023 through March 1, 2025
- evaluation period:
  - March 2, 2025 through present

`llm/baseline.py` and `llm/rag.py` automatically evaluate only on the test-period rows by default.

## Run Order

Use the project in this order:

```bash
python main.py
python -m rag.vector_store
python -m llm.baseline
python -m llm.rag_manual
python -m llm.rag_vector
python -m evaluation.evaluation
```

What each command does:

1. `python main.py`
- pulls SPY and VIX minute data from Massive
- builds the setup-window raw file
- builds the feature/outcome file
- applies the path-dependent `TAKE` / `FAIL_FAKEOUT` / `PASS` labeler

2. `python -m rag.vector_store`
- builds or refreshes the local Chroma collection
- indexes only the training subset from February 14, 2023 through March 1, 2025
- uses OpenRouter embeddings via `openai/text-embedding-3-small`

3. `python -m llm.baseline`
- evaluates the baseline model on test-period dates only
- writes `data/generated/baseline_results.csv`

4. `python -m llm.rag_manual`
- runs test-period RAG using the manual retriever
- writes `data/generated/rag_results_manual.csv`

5. `python -m llm.rag_vector`
- runs test-period RAG using local Chroma retrieval
- queries the prebuilt training-only vector store
- writes `data/generated/rag_results_vector.csv`

6. `python -m evaluation.evaluation`
- compares baseline and one chosen RAG result file against the ground truth
- writes `data/generated/comparison_results.csv`

Note:
- the evaluator defaults to `rag_results.csv`
- if you use the wrappers, either rename the desired RAG file or call `compare_runs(...)` with the file you want to compare

## Chunked Dataset Builds

For long Massive runs, build the dataset in chunks instead of one multi-hour pass. `main.py` now accepts date ranges and writes chunk-specific files automatically:

```bash
python main.py --start 2023-02-14 --end 2023-12-31
python main.py --start 2024-01-01 --end 2024-12-31
python main.py --start 2025-01-01 --end 2025-03-01
python main.py --start 2025-03-02 --end 2026-04-08
```

Those runs produce files like:
- `data/generated/2023-02-14_2023-12-31_spy_open_setup_features.csv`
- `data/generated/2023-02-14_2023-12-31_spy_open_setup_raw.csv`

Then merge them into the canonical combined files:

```bash
python -m pipeline.merge_chunks
```

Chunked runs automatically seed `previous_close` from the prior trading day so the first row of each chunk stays consistent with the full-range logic.

## Tech Stack
- Python
- pandas / numpy
- Massive flat files
- OpenAI-compatible chat client
- local ChromaDB
- OpenRouter embeddings with `openai/text-embedding-3-small`
- custom retrieval pipeline

Pinecone, LangChain, and LlamaIndex are not required for the current implementation.

## Local Setup

Recommended Python version: `3.11`

```bash
python3.11 -m venv CS_6180_RAG
source CS_6180_RAG/bin/activate
pip install -r requirements.txt
```

Required `.env` values:

```env
MASSIVE_ACCESS_KEY=...
MASSIVE_SECRET_KEY=...
MASSIVE_S3_ENDPOINT=https://files.massive.com
MASSIVE_S3_BUCKET=flatfiles
OPENROUTER_API_KEY=...
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

The `.env` file is gitignored and should never be committed.
