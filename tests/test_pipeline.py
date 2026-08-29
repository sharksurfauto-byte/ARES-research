"""Unit and Integration Tests for ARES Pipeline (PRD §3, §4.1, §5.2)."""

import os
import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

from ares.data.benchmark_loader import BenchmarkSample
from ares.experts.lora_expert import LoRAExpert, LoRAExpertConfig
from ares.experts.manager import ExpertManager
from ares.grm.architecture import GRM
from ares.lrm.architecture import LRM
from ares.pipeline import (
    ARESPipeline,
    BaselineComparator,
    BaselineSampleResult,
    EvaluationReport,
    MetricsCalculator,
    PipelineConfig,
    PipelineResult,
)


class MockModelOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class MockBackboneModel(nn.Module):
    def __init__(self, hidden_dim=64, vocab_size=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Mock transformer layers
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, output_hidden_states=True, **kwargs):
        x = self.embedding(input_ids)
        hiddens = [x]
        for layer in self.model.layers:
            x = layer(x)
            hiddens.append(x)
        return MockModelOutput(hidden_states=hiddens)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        # Generate dummy tokens
        batch_size = input_ids.shape[0]
        dummy_tokens = torch.randint(1, self.vocab_size, (batch_size, max_new_tokens), device=input_ids.device)
        return torch.cat([input_ids, dummy_tokens], dim=1)


class MockBackbone:
    def __init__(self, hidden_dim=64):
        self.hidden_size = hidden_dim
        self._model = MockBackboneModel(hidden_dim=hidden_dim)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        tokens = [2, 3, 4, 5]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join([f"tok_{t}" for t in token_ids])


@pytest.fixture
def mock_pipeline():
    hidden_dim = 64
    config = PipelineConfig(
        model_name="mock-model",
        hidden_dim=hidden_dim,
        device="cpu",
        max_new_tokens=5,
    )
    backbone = MockBackbone(hidden_dim=hidden_dim)
    tokenizer = MockTokenizer()
    grm = GRM(input_dim=hidden_dim, hidden_dim=32, domain_classes=5)
    lrm = LRM(input_dim=hidden_dim, hidden_dim=32)
    expert_manager = ExpertManager(input_dim=hidden_dim, hidden_dim=32, n_experts=5)

    return ARESPipeline(
        config=config,
        backbone=backbone,
        tokenizer=tokenizer,
        grm=grm,
        lrm=lrm,
        expert_manager=expert_manager,
    )


class TestARESPipeline:
    """Test ARESPipeline components and execution flow."""

    def test_pipeline_initialization(self, mock_pipeline):
        assert mock_pipeline.backbone is not None
        assert mock_pipeline.tokenizer is not None
        assert mock_pipeline.grm is not None
        assert mock_pipeline.lrm is not None
        assert mock_pipeline.expert_manager is not None
        assert len(mock_pipeline.expert_names) == 5
        assert len(mock_pipeline.route_names) == 6

    def test_evaluate_reliability(self, mock_pipeline):
        batch_size = 1
        seq_len = 4
        hidden_dim = 64
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        pooled_hidden = torch.randn(batch_size, hidden_dim)

        info = mock_pipeline.evaluate_reliability(hidden_states, pooled_hidden)

        assert "domain_prediction" in info
        assert "domain_confidence" in info
        assert "global_reliability" in info
        assert "feasibility" in info
        assert "token_reliability" in info
        assert "failure_risk" in info
        assert "uncertainty_score" in info

        assert 0.0 <= info["global_reliability"] <= 1.0
        assert 0.0 <= info["failure_risk"] <= 1.0
        assert 0.0 <= info["uncertainty_score"] <= 1.0
        assert info["domain_prediction"] in mock_pipeline.expert_names

    def test_routing_strategies(self, mock_pipeline):
        pooled = torch.randn(1, 64)
        info = {
            "global_reliability": 0.8,
            "domain_idx": 1,  # math
        }

        # 1. Base Strategy
        idx, route, probs = mock_pipeline.route(pooled, info, strategy="base")
        assert idx == 0
        assert route == "BASE"

        # 2. Fixed Strategy
        idx, route, probs = mock_pipeline.route(pooled, info, strategy="fixed_code")
        assert idx == 3
        assert route == "code"

        # 3. Threshold Strategy (High reliability -> Base)
        idx, route, probs = mock_pipeline.route(pooled, info, strategy="threshold")
        assert idx == 0
        assert route == "BASE"

        # Threshold Strategy (Low reliability -> Domain Expert)
        low_info = {"global_reliability": 0.2, "domain_idx": 1}
        idx, route, probs = mock_pipeline.route(pooled, low_info, strategy="threshold")
        assert idx == 2
        assert route == "math"

        # 4. Oracle Strategy
        idx, route, probs = mock_pipeline.route(pooled, info, strategy="oracle", oracle_domain="science")
        assert idx == 4
        assert route == "science"

        # 5. Dynamic ARES Strategy
        idx, route, probs = mock_pipeline.route(pooled, info, strategy="dynamic")
        assert 0 <= idx <= 5
        assert route in mock_pipeline.route_names

    def test_generate_end_to_end(self, mock_pipeline):
        prompt = "Solve 2 + 2"
        res = mock_pipeline.generate(prompt=prompt, strategy="dynamic", max_new_tokens=5)

        assert isinstance(res, PipelineResult)
        assert res.prompt == prompt
        assert len(res.generated_text) > 0
        assert res.selected_route in mock_pipeline.route_names
        assert 0 <= res.route_idx <= 5
        assert res.tokens_generated == 5
        assert "total_ms" in res.latency_ms
        assert "backbone_ms" in res.latency_ms
        assert "reliability_ms" in res.latency_ms
        assert "router_ms" in res.latency_ms
        assert "generation_ms" in res.latency_ms

    def test_generate_with_expert_hook(self, mock_pipeline):
        prompt = "Write code"
        # Force route to an expert
        res = mock_pipeline.generate(prompt=prompt, strategy="fixed_code", max_new_tokens=5)
        assert res.selected_route == "code"
        assert res.route_idx == 3
        assert res.tokens_generated == 5


class TestBaselineComparator:
    """Test Baseline comparison suite."""

    def test_baseline_comparator_single_sample(self, mock_pipeline):
        comparator = BaselineComparator(
            pipeline=mock_pipeline,
            strategies=["BASE", "FIXED_EXPERT", "DYNAMIC_ARES", "ORACLE_ROUTER"],
            fixed_expert="math",
        )

        sample = BenchmarkSample(
            sample_id="test_0",
            domain="math",
            prompt="What is 10 + 10?",
            target_answer="20",
            eval_type="math_numeric",
        )

        result = comparator.evaluate_sample(sample, max_new_tokens=5)

        assert isinstance(result, BaselineSampleResult)
        assert result.sample_id == "test_0"
        assert result.domain == "math"
        assert set(result.results.keys()) == {"BASE", "FIXED_EXPERT", "DYNAMIC_ARES", "ORACLE_ROUTER"}
        assert set(result.correctness.keys()) == {"BASE", "FIXED_EXPERT", "DYNAMIC_ARES", "ORACLE_ROUTER"}
        assert set(result.latencies_ms.keys()) == {"BASE", "FIXED_EXPERT", "DYNAMIC_ARES", "ORACLE_ROUTER"}

    def test_baseline_comparator_batch(self, mock_pipeline):
        comparator = BaselineComparator(
            pipeline=mock_pipeline,
            strategies=["BASE", "DYNAMIC_ARES"],
        )

        samples = [
            BenchmarkSample(
                sample_id=f"sample_{i}",
                domain="math" if i % 2 == 0 else "science",
                prompt=f"Question {i}",
                target_answer=f"Answer {i}",
                eval_type="general_text",
            )
            for i in range(4)
        ]

        batch_results = comparator.evaluate_batch(samples, max_new_tokens=3, verbose=False)
        assert len(batch_results) == 4


class TestMetricsCalculator:
    """Test Metrics calculation and report generation."""

    def test_calculate_metrics_and_reporting(self, mock_pipeline, tmp_path):
        comparator = BaselineComparator(
            pipeline=mock_pipeline,
            strategies=["BASE", "DYNAMIC_ARES", "FIXED_EXPERT", "ORACLE_ROUTER"],
            fixed_expert="math",
        )

        samples = [
            BenchmarkSample(
                sample_id=f"sample_{i}",
                domain=["math", "code", "science", "general"][i % 4],
                prompt=f"Question {i}",
                target_answer=f"tok_{i+1}",
                eval_type="general_text",
            )
            for i in range(4)
        ]

        batch_results = comparator.evaluate_batch(samples, max_new_tokens=3, verbose=False)
        report = MetricsCalculator.calculate_metrics(
            batch_results,
            metadata={"test_run": True},
        )

        assert isinstance(report, EvaluationReport)
        assert report.total_samples == 4
        assert len(report.domains_evaluated) == 4
        assert "DYNAMIC_ARES" in report.strategy_metrics

        # Test Markdown generation
        md = report.to_markdown()
        assert "# ARES End-to-End Evaluation Report" in md
        assert "Strategy Comparison Summary" in md
        assert "DYNAMIC_ARES" in md

        # Test JSON serialization & saving
        json_path = tmp_path / "report.json"
        report.save_json(json_path)
        assert json_path.exists()

        # Test dictionary conversion
        d = report.to_dict()
        assert d["total_samples"] == 4
        assert "strategy_metrics" in d
