# RAGs-to-Riches

## Overview
RAGs-to-Riches is a research project that explores whether Retrieval-Augmented Generation (RAG) improves the ability of large language models (LLMs) to follow strict, rule-based logic in structured environments. Using financial market data as a test case, the system evaluates whether an intraday trading setup satisfies predefined strategy rules.

Rather than predicting markets, this project focuses on **decision validation** — determining whether a trade setup meets specific criteria during the first five minutes of the NYSE market open.

---

## Research Question
**Does retrieval-augmented reasoning improve the reliability, decision accuracy, and explanation quality of LLMs when evaluating structured financial signals compared to a baseline LLM without retrieval?**

---

## System Design

### Input
- A single trading day's SPY opening sequence from `spy_open_raw_minutes.csv`
- Time window: **9:30–9:34 AM ET** (5 one-minute candles)

### Context (RAG)
- Strategy rules (formalized trading logic)
- Labeled historical examples (past market setups)

### Output
- **Decision:** Take Trade / Pass
- **Confidence Score**
- **Explanation:** grounded in rules and examples

---

## Project Structure

### 1. Data Pipeline
SPY 1-minute bars are pulled directly from [Massive](https://massive.com) via their S3-compatible flat file endpoint (`us_stocks_sip/minute_aggs_v1/`). The current codebase is organized into small modules with a single entry point in `main.py`.

Current package layout:

```text
main.py
services/
  s3_client.py
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
llm/
  baseline.py
  rag.py
```

Pipeline responsibilities:

- `services/s3_client.py`
  - connects to Massive's S3-compatible endpoint
  - lists accessible flat files and streams daily minute aggregate files
- `pipeline/features.py`
  - extracts SPY rows from each file
  - filters to the `09:30` through `09:34` ET opening window
  - computes the agreed feature schema
- `pipeline/dataset.py`
  - loops over trading dates
  - builds `spy_open_raw_minutes.csv` and `spy_open_features.csv`
  - applies deterministic labels via `trading_strategies/breakout_strategy.py`
- `prompts/`
  - contains the baseline prompt formatter and future prompt files from the prompting workstream
- `rag/` and `llm/`
  - contain the scaffold for retrieval-augmented prompting and baseline-vs-RAG evaluation

To generate the dataset, run:

```bash
python main.py
```

Credentials (`MASSIVE_ACCESS_KEY`, `MASSIVE_SECRET_KEY`, `MASSIVE_S3_ENDPOINT`, `MASSIVE_S3_BUCKET`) must be set in a `.env` file. The `.env` file is gitignored and should not be committed.

### 1.1 Run Order

Use the project in this order:

```bash
python main.py
python -m llm.baseline
python -m llm.rag
python -m evaluation.evaluation
```

What each command does:

1. `python main.py`
- pulls historical SPY minute data from Massive flat files
- filters each day to the `09:30` through `09:34` opening window
- builds:
  - `spy_open_raw_minutes.csv`
  - `spy_open_features.csv`
- applies deterministic ground-truth labels using `trading_strategies/breakout_strategy.py`

2. `python -m llm.baseline`
- loads one date at a time from `spy_open_raw_minutes.csv`
- sends only that raw 5-bar opening sequence to the LLM
- saves model predictions to `baseline_results.csv`

3. `python -m llm.rag`
- loads the same one-date raw 5-bar opening sequence from `spy_open_raw_minutes.csv`
- retrieves strategy rules and similar historical examples from `spy_open_features.csv`
- excludes the current date from retrieval
- saves model predictions to `rag_results.csv`

4. `python -m evaluation.evaluation`
- compares `baseline_results.csv` and `rag_results.csv`
- joins both against the ground truth from `spy_open_features.csv`
- prints side-by-side summary metrics
- saves merged comparison output to `comparison_results.csv`

### 2. Datasets

The pipeline produces two output files covering approximately 2 years of history.

---

#### `spy_open_raw_minutes.csv` — Raw minute bars (LLM input)

5 rows per trading day — one per minute bar. This is the primary model input for both the baseline and RAG experiments.

| column | description |
|--------|-------------|
| `date` | trading day in `YYYY-MM-DD` |
| `time` | bar time in `HH:MM` ET (09:30 through 09:34) |
| `open` | bar open price |
| `high` | bar high price |
| `low` | bar low price |
| `close` | bar close price |
| `volume` | bar volume |

---

#### `spy_open_features.csv` — Engineered features + ground truth (RAG knowledge base)

One row per trading day. Serves as the retrieval memory and the ground truth for evaluation.

| column | description |
|--------|-------------|
| `date` | trading day in `YYYY-MM-DD` |
| `spy_open` | open price of the `09:30` candle |
| `previous_close` | previous trading day regular-session close |
| `gap_pct` | % gap from previous close to `09:30` open |
| `first_1m_return` | % return of the first candle, `09:30` open to `09:30` close |
| `net_movement` | % move across the full window, `09:30` open to `09:34` close |
| `opening_range_high` | highest `high` from `09:30` through `09:34` |
| `opening_range_low` | lowest `low` from `09:30` through `09:34` |
| `opening_range_width` | `opening_range_high - opening_range_low` |
| `breakout_direction` | `UP`, `DOWN`, or `NONE` based on the 5-minute move |
| `volatility` | average intrabar range across the five candles |
| `volume` | total volume from `09:30` through `09:34` |
| `volume_ratio` | opening-window volume divided by trailing 20-day average opening-window volume |
| `label` | ground truth: `TAKE` or `PASS` |

**Ground truth logic:** `label` is created deterministically by `trading_strategies/breakout_strategy.py`. The current strategy is a momentum breakout rule set: `TAKE` requires `breakout_direction = UP`, `net_movement >= 0.25`, `volume_ratio >= 1.2`, `opening_range_width >= 0.3`, and `first_1m_return >= 0`. Otherwise the row is labeled `PASS`.

**How the two files work together:**
- `spy_open_raw_minutes.csv` is the **query input** — the LLM sees one date's 5-bar opening sequence and must decide `TAKE TRADE` or `PASS TRADE`
- `spy_open_features.csv` is the **memory and answer key** — the RAG system retrieves similar historical setups from this file, and the `label` column is the ground truth used for evaluation
- Leakage rule: the current test date must never be retrieved as an example for itself
- Baseline and RAG prompts must never include the current row's `label`

### 3. Baseline Model
- Implemented as a separate prompt path in `llm/baseline.py`
- LLM should receive only one date's 5 raw bars from `spy_open_raw_minutes.csv`
- Outputs decision + explanation without retrieved context

### 4. RAG System
- Scaffolded in `rag/`, `prompts/rag_prompt.py`, and `llm/rag.py`
- Retrieve:
  - relevant strategy rules
  - similar historical examples from `spy_open_features.csv`
- Retrieved examples may include their historical labels because this design is testing case-based reasoning
- The current date must be excluded from retrieval to avoid leakage
- Augment the raw-minute baseline prompt with retrieved context before LLM inference

### 5. Prompt Pipeline

Target prompt design:

1. Baseline prompt
- Input: one date from `spy_open_raw_minutes.csv`
- Contains only the 5 one-minute bars for that day
- Asks the LLM to decide `TAKE TRADE` or `PASS TRADE`
- No retrieved rules or examples

2. RAG prompt
- Input: the same one-date 5-bar sequence from `spy_open_raw_minutes.csv`
- Retrieved context:
  - strategy rules
  - similar historical rows from `spy_open_features.csv`
  - historical labels for those retrieved rows may be included
- Exclude the current date from retrieval
- Do not include the current date's ground-truth label in the prompt

### 6. Evaluation
Compare baseline vs RAG system using:
- **Accuracy** (correct decisions vs ground truth)
- **Consistency** (stability across multiple runs)
- **Explanation Quality** (grounded vs hallucinated reasoning)

---

## Tech Stack
- Python (pandas, numpy)
- Massive S3 flat files (market data source)
- LLM APIs (GPT-4o, Claude)
- Vector Database (ChromaDB / Pinecone)
- RAG Framework (LangChain / LlamaIndex)

## Local Setup

Recommended Python version: `3.11`

The official Massive Python client supports Python `3.9+`, but `3.11` is the most practical team default for this project because it is broadly compatible with the data and RAG libraries in this repo.

Create a virtual environment and install dependencies:

```bash
python3.11 -m venv CS_6180_RAG
source CS_6180_RAG/bin/activate
pip install -r requirements.txt
```

If `python3.11` is not available on your machine, use another Python `3.9+` interpreter.

The virtual environment directory is ignored via `.gitignore` and should not be committed.

---

## Project Timeline

### Week 1
- Data collection & feature engineering
- Define strategy rules
- Generate labeled dataset

### Week 2
- Build baseline LLM system
- Run initial experiments

### Week 3
- Implement RAG pipeline
- Integrate retrieval into LLM prompts

### Week 4
- Evaluate performance
- Analyze results and generate insights

---

## Team Responsibilities & Weekly Breakdown

### Week 1 — Data + Strategy
**Goal:** Build dataset, define trading rules, and generate labels

**Data & Feature Engineering Lead (Primary)**
- Pull historical SPY data from Massive S3 flat files
- Filter to 9:30–9:34 window (5 one-minute bars)
- Engineer features:
  - breakout (up/down)
  - volatility
  - volume
  - net movement
- Produce `spy_open_raw_minutes.csv` and `spy_open_features.csv`

**RAG & Evaluation Lead**
- Define trading strategy rules with clear thresholds
- Implement deterministic labeling logic
- Ensure features align with retrieval needs

**LLM & Prompting Lead**
- Design initial prompt format
- Define:
  - input structure (features → text)
  - output format:
    - decision
    - confidence
    - explanation

**Deliverables**
- Clean dataset
- Feature set
- Strategy rules
- Ground truth labels

---

### Week 2 — Baseline LLM
**Goal:** Evaluate LLM without retrieval

**LLM & Prompting Lead (Primary)**
- Build baseline prompt
- Run LLM on dataset
- Ensure consistent output format

**RAG & Evaluation Lead**
- Build evaluation pipeline
- Compare predictions vs labels
- Implement metrics:
  - accuracy
  - consistency (multiple runs)

**Data & Feature Engineering Lead**
- Convert dataset into LLM input format
- Validate feature correctness
- Debug edge cases

**Deliverables**
- Working baseline system
- Initial performance results

---

### Week 3 — RAG System
**Goal:** Integrate retrieval layer

**RAG & Evaluation Lead (Primary)**
- Build vector database
- Store strategy rules and historical examples
- Generate embeddings
- Implement retrieval logic:
  - top-k similar examples
  - relevant rules

**LLM & Prompting Lead**
- Update prompt to include retrieved context
- Maintain consistent output format

**Data & Feature Engineering Lead**
- Prepare dataset for retrieval
- Define similarity features
- Assist with embedding structure

**Deliverables**
- Fully working RAG system

---

### Week 4 — Evaluation & Analysis
**Goal:** Compare baseline vs RAG system

**RAG & Evaluation Lead (Primary)**
- Run experiments on both systems
- Compute:
  - accuracy
  - consistency
  - performance differences

**LLM & Prompting Lead**
- Analyze explanation quality (grounded vs hallucinated)

**Data & Feature Engineering Lead**
- Analyze performance across scenarios:
  - trending vs choppy days
- Identify patterns in results

**Deliverables**
- Final comparison
- Key findings and insights

---

## Key Insight
This project bridges **structured numerical data** and **LLM reasoning**. Unlike traditional RAG systems that operate purely on text, this system evaluates how well LLMs can apply explicit rules to structured signals when supported by retrieved knowledge.

---

## Contributors
- Ricky Lee
- Tarun Badarvada
- Dina Barua
