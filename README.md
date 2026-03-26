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
- Collect historical SPY 1-minute candle data
- Filter to first 5 minutes of each trading day
- Engineer structured features:
  - breakout direction
  - volatility
  - volume expansion
  - momentum indicators

#### Raw Dataset
The raw dataset consists of minute-by-minute SPY market data and is stored separately to preserve original signals.

Example file: `raw_minutes.csv`

Columns:
- datetime
- open
- high
- low
- close
- volume

### 2. Dataset Creation
- Each row represents one trading day
- Apply deterministic rules to generate:
  - **Ground truth labels (TAKE / PASS)**
- Final dataset used for evaluation

#### Processed Feature Dataset
The processed dataset contains one row per trading day and serves as the model-ready input for evaluation.

Example file: `spy_open_features.csv`

Each row includes:
- date
- SPY open
- previous close
- gap %
- first 1-min return
- first 3-min return
- first 5-min return
- opening range high
- opening range low
- opening range width
- total first-5-min volume
- relative volume
- volatility measure
- breakout flag
- label (TAKE / PASS)

This dataset is generated through feature engineering and rule-based labeling from the raw data.

### 3. Baseline Model
- LLM receives only structured features
- Outputs decision + explanation

### 4. RAG System
- Retrieve:
  - relevant strategy rules
  - similar historical examples
- Augment LLM prompt with retrieved context

### 5. Evaluation
Compare baseline vs RAG system using:
- **Accuracy** (correct decisions vs ground truth)
- **Consistency** (stability across multiple runs)
- **Explanation Quality** (grounded vs hallucinated reasoning)

---

## Tech Stack
- Python (pandas, numpy)
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
- Pull historical SPY data
- Filter to 9:30–9:35 window
- Group data by day
- Engineer features:
  - breakout (up/down)
  - volatility
  - volume
  - net movement
- Produce final dataset (CSV)

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
