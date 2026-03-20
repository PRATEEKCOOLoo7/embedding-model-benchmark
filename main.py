"""Embedding Model Benchmark — Demo"""
import logging
from core.benchmark import BenchmarkRunner, SimpleEmbedder, BENCHMARK_DATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")

def main():
    print(f"\n{'='*60}")
    print("  Embedding Model Benchmark — Demo")
    print(f"{'='*60}")

    models = [
        SimpleEmbedder("term_overlap_v1", cost_per_1k=0.0),
        SimpleEmbedder("term_overlap_v2", cost_per_1k=0.0),  # same algo, different instance
    ]
    runner = BenchmarkRunner(models)
    report = runner.run()

    print(f"\n  Cases: {report.cases_evaluated}")
    print(f"  Winner: {report.winner}\n")

    for m in report.models:
        print(f"  {m.model_name}:")
        print(f"    Recall@3: {m.recall_at_3:.3f} | Recall@5: {m.recall_at_5:.3f}")
        print(f"    MRR: {m.mrr:.3f} | Latency: {m.avg_latency_ms:.2f}ms")
        print(f"    Cost/1K: ${m.cost_per_1k}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
