"""Embedding model benchmark for RAG pipeline optimization.

Evaluates embedding models across retrieval quality (Recall@K, MRR),
latency, and cost. Compares chunking strategies to find the optimal
configuration for a given domain.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class BenchmarkCase:
    query: str
    relevant_doc_ids: list[str]
    documents: list[dict]  # [{"id": str, "content": str}]


@dataclass
class ModelScore:
    model_name: str
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    cost_per_1k: float = 0.0


@dataclass
class BenchmarkReport:
    cases_evaluated: int
    models: list[ModelScore]
    winner: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class SimpleEmbedder:
    """Term-overlap embedder for offline benchmarking.
    In production, swap with OpenAI/Cohere/HuggingFace embedders."""

    def __init__(self, name: str = "term_overlap", cost_per_1k: float = 0.0):
        self.name = name
        self.cost_per_1k = cost_per_1k

    def embed(self, text: str) -> list[float]:
        """Fake embedding — returns term frequency vector."""
        words = text.lower().split()
        vocab = sorted(set(words))
        return [words.count(w) / len(words) for w in vocab[:50]]

    def similarity(self, query: str, doc: str) -> float:
        """Term overlap similarity (proxy for cosine similarity)."""
        q_terms = set(query.lower().split())
        d_terms = set(doc.lower().split())
        if not q_terms or not d_terms:
            return 0.0
        overlap = len(q_terms & d_terms)
        return overlap / math.sqrt(len(q_terms) * len(d_terms))

    def rank(self, query: str, documents: list[dict]) -> list[tuple[str, float]]:
        """Rank documents by similarity to query."""
        scored = []
        for doc in documents:
            score = self.similarity(query, doc["content"])
            scored.append((doc["id"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


BENCHMARK_DATA = [
    BenchmarkCase(
        query="What is Apple's quarterly revenue?",
        relevant_doc_ids=["apple_10q"],
        documents=[
            {"id": "apple_10q", "content": "Apple reported fourth quarter revenue of 94.9 billion dollars representing 6 percent increase year over year."},
            {"id": "nvidia_10q", "content": "NVIDIA reported record third quarter revenue of 35.1 billion dollars driven by AI infrastructure demand."},
            {"id": "macro_rates", "content": "The Federal Reserve held the federal funds rate steady at 4.25 to 4.50 percent."},
            {"id": "fintech_report", "content": "The wealth management technology market is projected to reach 12.5 billion by 2027."},
        ],
    ),
    BenchmarkCase(
        query="NVIDIA data center revenue growth AI chips",
        relevant_doc_ids=["nvidia_10q", "nvidia_competition"],
        documents=[
            {"id": "apple_10q", "content": "Apple reported fourth quarter revenue of 94.9 billion dollars."},
            {"id": "nvidia_10q", "content": "NVIDIA Corporation data center revenue reached 30.8 billion a 112 percent increase driven by AI chip demand."},
            {"id": "nvidia_competition", "content": "NVIDIA faces growing competition in the AI accelerator market from AMD MI300X and custom silicon from cloud providers."},
            {"id": "macro_rates", "content": "The Federal Reserve held rates steady citing persistent inflation above 2 percent target."},
        ],
    ),
    BenchmarkCase(
        query="interest rate federal reserve monetary policy",
        relevant_doc_ids=["macro_rates"],
        documents=[
            {"id": "apple_10q", "content": "Apple reported fourth quarter revenue of 94.9 billion dollars."},
            {"id": "nvidia_10q", "content": "NVIDIA reported record revenue driven by AI infrastructure."},
            {"id": "macro_rates", "content": "The Federal Reserve held the federal funds rate steady at 4.25 to 4.50 percent citing persistent inflation and monetary policy tightening."},
            {"id": "fintech_report", "content": "The wealth management technology market is growing at 15 percent annually."},
        ],
    ),
]


class BenchmarkRunner:
    """Runs retrieval benchmarks across embedding models."""

    def __init__(self, models: list[SimpleEmbedder] = None):
        self.models = models or [
            SimpleEmbedder("term_overlap_basic", 0.0),
        ]

    def run(self, cases: list[BenchmarkCase] = None, top_k_values: list[int] = None) -> BenchmarkReport:
        cases = cases or BENCHMARK_DATA
        top_k_values = top_k_values or [3, 5]

        model_scores = []

        for model in self.models:
            recall_3_total = 0.0
            recall_5_total = 0.0
            mrr_total = 0.0
            latency_total = 0.0

            for case in cases:
                t0 = time.monotonic()
                ranked = model.rank(case.query, case.documents)
                latency_total += (time.monotonic() - t0) * 1000

                # Recall@K
                top_3_ids = [r[0] for r in ranked[:3]]
                top_5_ids = [r[0] for r in ranked[:5]]

                recall_3 = len(set(case.relevant_doc_ids) & set(top_3_ids)) / len(case.relevant_doc_ids)
                recall_5 = len(set(case.relevant_doc_ids) & set(top_5_ids)) / len(case.relevant_doc_ids)
                recall_3_total += recall_3
                recall_5_total += recall_5

                # MRR
                for rank_idx, (doc_id, _) in enumerate(ranked):
                    if doc_id in case.relevant_doc_ids:
                        mrr_total += 1.0 / (rank_idx + 1)
                        break

            n = len(cases)
            model_scores.append(ModelScore(
                model_name=model.name,
                recall_at_3=round(recall_3_total / n, 4),
                recall_at_5=round(recall_5_total / n, 4),
                mrr=round(mrr_total / n, 4),
                avg_latency_ms=round(latency_total / n, 2),
                cost_per_1k=model.cost_per_1k,
            ))

        # Determine winner by MRR
        winner = max(model_scores, key=lambda m: m.mrr).model_name if model_scores else ""

        return BenchmarkReport(
            cases_evaluated=len(cases),
            models=model_scores,
            winner=winner,
        )
