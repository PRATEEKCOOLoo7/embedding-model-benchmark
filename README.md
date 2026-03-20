# Embedding Model Benchmark for RAG Pipelines

Benchmarking framework for evaluating embedding models across retrieval quality, latency, cost, and domain-specific accuracy. Helps you pick the right embedding model for your RAG pipeline instead of defaulting to "whatever OpenAI offers."

## Why This Exists

RAG pipeline quality is bottlenecked by retrieval, and retrieval quality depends on embedding model choice. Different models excel at different things:

| Model | Best For | Weakness |
|---|---|---|
| OpenAI text-embedding-3-large | General purpose, high quality | Expensive, API dependency |
| Cohere embed-v3 | Multilingual, compression | Latency |
| BGE-large-en-v1.5 | Self-hosted, low latency | Lower quality on domain-specific |
| E5-mistral-7b | Domain adaptation, long context | GPU required, slow |
| GTE-Qwen2 | Chinese + English, long docs | Large model size |

This benchmark tells you which model gives the best retrieval quality for YOUR data at YOUR latency budget.

## What It Measures

### Retrieval Quality
- **Recall@K**: Does the correct document appear in top K results?
- **MRR** (Mean Reciprocal Rank): How high does the correct document rank?
- **NDCG@K**: Are the top K results in the right order?
- **Domain-Specific Accuracy**: How well does it handle your domain's terminology?

### Operational Metrics
- **Latency**: P50/P95/P99 embedding generation time
- **Throughput**: Embeddings per second (batch and single)
- **Cost**: $ per 1M tokens embedded
- **Index Size**: Storage per 1M documents

### Chunking Strategy Comparison
- **Fixed-size chunks**: 256 / 512 / 1024 tokens
- **Semantic chunking**: Split on topic boundaries
- **Recursive splitting**: Hierarchical with overlap
- **Sliding window**: Overlapping fixed windows

## Project Structure

```
embedding-model-benchmark/
├── README.md
├── requirements.txt
├── benchmark/
│   ├── __init__.py
│   ├── runner.py               # Main benchmark orchestrator
│   ├── retrieval_eval.py       # Recall, MRR, NDCG calculation
│   ├── latency_profiler.py     # P50/P95/P99 measurement
│   ├── cost_calculator.py      # Token counting + pricing
│   └── report_generator.py     # HTML/JSON comparison reports
├── models/
│   ├── __init__.py
│   ├── openai_embedder.py      # OpenAI text-embedding-3
│   ├── cohere_embedder.py      # Cohere embed-v3
│   ├── huggingface_embedder.py # BGE, E5, GTE self-hosted
│   └── base_embedder.py        # Abstract interface
├── chunking/
│   ├── __init__.py
│   ├── fixed_size.py           # Fixed token count chunks
│   ├── semantic.py             # Topic-boundary splitting
│   ├── recursive.py            # LangChain recursive splitter
│   └── sliding_window.py       # Overlapping windows
├── datasets/
│   ├── financial_qa.jsonl      # Financial domain Q&A pairs
│   └── sales_outreach.jsonl    # Sales/revenue domain pairs
├── tests/
│   └── test_benchmark.py
└── examples/
    └── run_benchmark.py
```

## Usage

```python
from benchmark import BenchmarkRunner
from models import OpenAIEmbedder, HuggingFaceEmbedder
from chunking import FixedSize, SemanticChunking

runner = BenchmarkRunner(
    dataset="datasets/financial_qa.jsonl",
    models=[
        OpenAIEmbedder("text-embedding-3-large"),
        OpenAIEmbedder("text-embedding-3-small"),
        HuggingFaceEmbedder("BAAI/bge-large-en-v1.5"),
    ],
    chunking_strategies=[
        FixedSize(512),
        FixedSize(1024),
        SemanticChunking(),
    ],
    top_k=[3, 5, 10],
)

results = runner.run()
runner.generate_report(results, output="reports/embedding_comparison.html")
```

