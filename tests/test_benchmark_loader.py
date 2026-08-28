"""Unit tests for Benchmark Loader and Ground-Truth Evaluators."""

import pytest
from ares.data.benchmark_loader import (
    BenchmarkSample,
    evaluate_prediction,
    extract_math_answer,
    extract_mcq_answer,
    load_gsm8k_samples,
    load_mbpp_samples,
    load_ai2_arc_samples,
    load_wikitext_samples,
    load_reasoning_samples,
    load_all_benchmark_samples,
)


class TestGroundTruthEvaluators:
    """Test response extractors and accuracy evaluators."""

    def test_extract_math_answer(self):
        assert extract_math_answer("The result is #### 42") == "42"
        assert extract_math_answer("The answer is 60 km/h.") == "60"
        assert extract_math_answer("x = 15") == "15"
        assert extract_math_answer("Therefore, 3.14") == "3.14"

    def test_extract_mcq_answer(self):
        assert extract_mcq_answer("The correct answer is (B)") == "B"
        assert extract_mcq_answer("Choice: C") == "C"
        assert extract_mcq_answer("A. Nitrogen") == "A"
        assert extract_mcq_answer("D") == "D"

    def test_evaluate_math_numeric(self):
        assert evaluate_prediction("The answer is 60", "60", "math_numeric") is True
        assert evaluate_prediction("Speed is 60.0 km/h", "60", "math_numeric") is True
        assert evaluate_prediction("The answer is 55", "60", "math_numeric") is False

    def test_evaluate_multiple_choice(self):
        assert evaluate_prediction("The answer is (B)", "B", "multiple_choice") is True
        assert evaluate_prediction("Choice C is correct", "C", "multiple_choice") is True
        assert evaluate_prediction("The answer is (A)", "B", "multiple_choice") is False

    def test_evaluate_code_snippet(self):
        pred = "def square(n):\n    return n * n"
        target = "def square(n): return n * n"
        assert evaluate_prediction(pred, target, "code_snippet") is True

    def test_evaluate_general_text(self):
        pred = "Artificial intelligence is important because it automates tasks."
        target = "automates tasks"
        assert evaluate_prediction(pred, target, "general_text") is True


class TestBenchmarkLoaders:
    """Test loading benchmark samples across domains."""

    def test_load_gsm8k(self):
        samples = load_gsm8k_samples(n_samples=5)
        assert len(samples) == 5
        assert all(isinstance(s, BenchmarkSample) for s in samples)
        assert all(s.domain == "math" for s in samples)
        assert all(s.eval_type == "math_numeric" for s in samples)

    def test_load_mbpp(self):
        samples = load_mbpp_samples(n_samples=5)
        assert len(samples) == 5
        assert all(s.domain == "code" for s in samples)
        assert all(s.eval_type == "code_snippet" for s in samples)

    def test_load_ai2_arc(self):
        samples = load_ai2_arc_samples(n_samples=5)
        assert len(samples) == 5
        assert all(s.domain == "science" for s in samples)
        assert all(s.eval_type == "multiple_choice" for s in samples)

    def test_load_wikitext(self):
        samples = load_wikitext_samples(n_samples=5)
        assert len(samples) == 5
        assert all(s.domain == "general" for s in samples)

    def test_load_reasoning(self):
        samples = load_reasoning_samples(n_samples=5)
        assert len(samples) == 5
        assert all(s.domain == "reasoning" for s in samples)

    def test_load_all_benchmark_samples(self):
        all_samples = load_all_benchmark_samples(n_samples_per_domain=3)
        assert set(all_samples.keys()) == {"general", "math", "code", "science", "reasoning"}
        for domain, samples in all_samples.items():
            assert len(samples) == 3
