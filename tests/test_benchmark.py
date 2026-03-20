import pytest
from core.benchmark import BenchmarkRunner, SimpleEmbedder, BENCHMARK_DATA, BenchmarkCase

class TestSimpleEmbedder:
    def test_similarity_identical(self):
        e = SimpleEmbedder("test")
        assert e.similarity("apple revenue", "apple revenue") > 0.5

    def test_similarity_different(self):
        e = SimpleEmbedder("test")
        assert e.similarity("apple revenue", "weather sunny rain") < 0.2

    def test_rank_returns_sorted(self):
        e = SimpleEmbedder("test")
        docs = [
            {"id": "d1", "content": "apple quarterly revenue growth earnings"},
            {"id": "d2", "content": "weather forecast sunny temperature"},
        ]
        ranked = e.rank("apple revenue earnings", docs)
        assert ranked[0][0] == "d1"

class TestBenchmarkRunner:
    def test_run_produces_report(self):
        runner = BenchmarkRunner([SimpleEmbedder("test")])
        report = runner.run()
        assert report.cases_evaluated == len(BENCHMARK_DATA)
        assert len(report.models) == 1
        assert report.winner != ""

    def test_recall_range(self):
        runner = BenchmarkRunner([SimpleEmbedder("test")])
        report = runner.run()
        for m in report.models:
            assert 0 <= m.recall_at_3 <= 1
            assert 0 <= m.recall_at_5 <= 1
            assert 0 <= m.mrr <= 1

    def test_multiple_models(self):
        models = [SimpleEmbedder("a"), SimpleEmbedder("b")]
        runner = BenchmarkRunner(models)
        report = runner.run()
        assert len(report.models) == 2

    def test_custom_cases(self):
        cases = [BenchmarkCase(
            query="test query",
            relevant_doc_ids=["d1"],
            documents=[{"id": "d1", "content": "test query answer"},
                      {"id": "d2", "content": "unrelated stuff"}],
        )]
        runner = BenchmarkRunner([SimpleEmbedder("test")])
        report = runner.run(cases)
        assert report.cases_evaluated == 1
        assert report.models[0].recall_at_3 > 0

    def test_benchmark_data_valid(self):
        for case in BENCHMARK_DATA:
            assert len(case.query) > 0
            assert len(case.relevant_doc_ids) > 0
            assert len(case.documents) >= 2
            for rel_id in case.relevant_doc_ids:
                assert any(d["id"] == rel_id for d in case.documents)
