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
- Structured features derived from SPY 1-minute candles
- Time window: **9:30–9:35 AM (first 5 minutes of market open)**

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

### 2. Datasets

The pipeline produces two output files covering approximately 2 years of history.

---

#### `spy_open_raw_minutes.csv` — Raw minute bars (LLM input)

5 rows per trading day — one per minute bar. This is what the LLM observes as the opening window unfolds.

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

One row per trading day. Serves as both the RAG retrieval knowledge base and the ground truth for evaluation.

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

**Ground truth logic:** `label` is derived from the close of the 5-minute candle (the `09:34` close). A setup is labeled `TAKE` if there is a clear directional breakout (`breakout_direction` is not `NONE`), confirmed by above-average volume (`volume_ratio >= 1.2`), and a meaningful price move (`|net_movement| >= 0.2%`). The exact thresholds are placeholders — the RAG & Evaluation Lead is responsible for finalizing them before the dataset is regenerated.

**How the two files work together:**
- `spy_open_raw_minutes.csv` is the **signal** — the LLM observes these bars minute-by-minute and tries to identify a valid setup as early as possible (ideally before bar 5)
- `spy_open_features.csv` is the **memory** — the RAG system retrieves similar historical setups from this file to inform the LLM's decision, and the `label` column is what the LLM's decision is evaluated against

### 3. Baseline Model
- Implemented as a separate prompt path in `llm/baseline.py`
- LLM receives only the current feature row formatted by `prompts/prompt_utils.py`
- Outputs decision + explanation without retrieved context

### 4. RAG System
- Scaffolded in `rag/`, `prompts/rag_prompt.py`, and `llm/rag.py`
- Retrieve:
  - relevant strategy rules
  - similar historical examples from `spy_open_features.csv`
- Augment the baseline prompt with retrieved context before LLM inference

### 5. Evaluation
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
