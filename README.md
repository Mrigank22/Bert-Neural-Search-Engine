# BERT-Neural-Search-Engine

A dual-stage AI-powered search engine combining fast BM25 retrieval with advanced BERT-based neural reranking for highly relevant, semantic search results. Built on the MS MARCO dataset, this project demonstrates a modern approach to large-scale information retrieval.

![Landing Page](images/Main.png)

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [System Overview](#system-overview)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [Future Work](#future-work)

## Features

- **Hybrid retrieval**: BM25 for fast candidate selection, BERT for neural reranking
- **User-friendly web interface** for querying and result exploration
- **Adjustable top-k results** for flexible search needs
- **Scalable preprocessing and evaluation pipeline**
- **Built on real-world MS MARCO dataset**

## Demo

### Landing Page
![Landing Page](images/Landing.jpg)

- Enter your query in the search bar
- Adjust the number for top results as needed
- Click "Search" to get results

### Searching State
![Searching](images/Searching.png)

- The system searches through over 755,000 indexed passages
- Real-time feedback on search progress

### Results Display
![Results](images/Results.png)

- Top results are shown with both BM25 and Neural (BERT) scores
- Relevant keywords are highlighted
- Query and result count are displayed for transparency

## System Overview

1. **BM25 Indexing**: Fast lexical retrieval to shortlist relevant passages
2. **Pair Generation**: Builds positive and negative query-passage pairs for neural training
3. **BERT Reranker**: Re-ranks BM25 candidates using a fine-tuned BERT model for semantic matching
4. **Evaluation**: Reports metrics like MRR and precision to measure system performance

## Dataset & Preprocessing

- **Source**: MS MARCO dataset
- **Scale**: ~10,000 queries, 800,000 passages (subset for demonstration)
- **Cleaning**: Removes nulls, non-English content, and irrelevant samples
- **Mapping**: Links queries to passages using provided relevance labels

## Installation

```bash
git clone https://github.com/Mrigank22/Bert-Neural-Search-Engine.git
cd Bert-Neural-Search-Engine
pip install -r requirements.txt
```

**Requirements:**  
Python 3.8+, PyTorch, transformers, rank-bm25, polars, scikit-learn, nltk, pandas, numpy

## Usage

### 1. Preprocessing

```bash
python preprocess.py --input data/raw --output data/processed
```

### 2. BM25 Indexing

```bash
python bm25_index.py --data data/processed
```

### 3. Train BERT Reranker

```bash
python train_reranker.py --train data/processed/train.csv --val data/processed/val.csv
```

### 4. Evaluation

```bash
python evaluate.py --model checkpoints/bert_reranker.pt --test data/processed/test.csv
```

### 5. Search Demo

```bash
python search.py --query "Your search query here"
```

## Project Structure

```
Bert-Neural-Search-Engine/
│
├── data/                   # Raw and processed datasets
├── images/                 # Interface and result screenshots
├── models/                 # Saved model checkpoints
├── scripts/                # Pipeline scripts
│   ├── preprocess.py
│   ├── bm25_index.py
│   ├── train_reranker.py
│   ├── evaluate.py
│   └── search.py
├── utils/                  # Helper functions
├── requirements.txt
└── README.md
```

## Contributors

- **Sandhya S** (BM25, dataset, analysis)
- **Mrigank Pendyala** (BERT reranker, training, evaluation)
- **Sharvani Pallempati** (Preprocessing, hyperparameter tuning)

## Future Work

- Integrate hard negatives from BM25 for improved neural training
- Scale up with larger datasets and GPU clusters
- Add multilingual and domain adaptation capabilities

---

**For methodology and detailed results, see the project report.**

*Images used in this README are included in the `images/` directory:*
- `Main.png` – Landing page and query input
- `Landing.jpg` – Edited Landing page and query input
- `Searching.jpg` – Searching state
- `Results.jpg` – Example results with scores and highlights
