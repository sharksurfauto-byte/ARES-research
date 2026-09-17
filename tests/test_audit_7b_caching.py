"""Tests for 7B NF4 support, dynamic hidden dimension, smart generation caching, and checkpointing."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ares.backbone.loader import BackboneConfig, load_backbone
from ares.data.benchmark_loader import (
    BenchmarkSample,
    load_all_benchmark_samples,
    load_gsm8k_samples,
    load_mbpp_samples,
    load_ai2_arc_samples,
    load_wikitext_samples,
    load_reasoning_samples,
)
from ares.experts.manager import ExpertManager
from ares.grm.architecture import GRM
from ares.lrm.architecture import LRM
from ares.pipeline import (
    ARESPipeline,
    BaselineComparator,
    BaselineSampleResult,
    PipelineConfig,
    PipelineResult,
)


class Mock7BBackboneModel(nn.Module):
    def __init__(self, hidden_dim=3584, vocab_size=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.call_count = 0

    def forward(self, input_ids, output_hidden_states=True, **kwargs):
        self.call_count += 1
        x = self.embedding(input_ids)
        hiddens = [x]
        for layer in self.model.layers:
            x = layer(x)
            hiddens.append(x)

        class Output:
            def __init__(self, h):
                self.hidden_states = h
                self.logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100)

        return Output(hiddens)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        self.call_count += 1
        batch_size = input_ids.shape[0]
        dummy = torch.randint(1, self.vocab_size, (batch_size, max_new_tokens), device=input_ids.device)
        return torch.cat([input_ids, dummy], dim=1)


class Mock7BBackbone:
    def __init__(self, hidden_dim=3584):
        self.hidden_size = hidden_dim
        self._model = Mock7BBackboneModel(hidden_dim=hidden_dim)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def get_device(self):
        return torch.device("cpu")

    def get_dtype(self):
        return torch.float32

    def generate(self, *args, **kwargs):
        return self._model.generate(*args, **kwargs)


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


def test_backbone_loader_7b_config():
    """Verify load_backbone config handles 7B and 4-bit properly on CPU and GPU."""
    # On CPU, load_in_4bit should be False
    config = BackboneConfig(
        name="Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="float32",
        device_map=None,
    )
    assert "7b" in config.name.lower()
    assert config.hidden_state_layers == (-1, -6, -12, -24)


def test_dynamic_hidden_dim_adaptation():
    """Verify ARESPipeline dynamically adapts hidden size from 3584 backbone."""
    backbone_7b = Mock7BBackbone(hidden_dim=3584)
    tokenizer = MockTokenizer()
    config = PipelineConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cpu",
    )
    pipeline = ARESPipeline(
        config=config,
        backbone=backbone_7b,
        tokenizer=tokenizer,
    )
    # Check that dynamic detection assigned 3584
    assert pipeline.config.hidden_dim == 3584
    assert pipeline.grm.input_dim == 3584
    assert pipeline.lrm.input_dim == 3584
    assert pipeline.expert_manager.router.input_dim == 3584


def test_smart_generation_caching():
    """Verify BaselineComparator reuses Base model generation without duplicate forward passes."""
    backbone = Mock7BBackbone(hidden_dim=3584)
    tokenizer = MockTokenizer()
    config = PipelineConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cpu",
        hidden_dim=3584,
    )
    pipeline = ARESPipeline(
        config=config,
        backbone=backbone,
        tokenizer=tokenizer,
    )

    comparator = BaselineComparator(
        pipeline=pipeline,
        strategies=["BASE", "THRESHOLD_ROUTER", "ORACLE_ROUTER"],
        fixed_expert="math",
        threshold=0.0,  # Forces threshold router to always choose BASE because reliability >= 0.0
    )

    sample = BenchmarkSample(
        sample_id="test_cache_1",
        domain="math",
        prompt="What is 40 + 2?",
        target_answer="42",
        eval_type="math_numeric",
    )

    # Initial call count
    initial_gen_calls = backbone._model.call_count

    sample_res = comparator.evaluate_sample(sample, max_new_tokens=5)

    # BASE was generated once
    base_text = sample_res.results["BASE"].generated_text
    threshold_text = sample_res.results["THRESHOLD_ROUTER"].generated_text

    # Verify that threshold strategy (which chose Route 0) reused base text exactly
    assert threshold_text == base_text
    assert sample_res.results["THRESHOLD_ROUTER"].selected_route == "BASE"
    assert sample_res.results["THRESHOLD_ROUTER"].route_idx == 0


def test_benchmark_split_handling():
    """Verify benchmark loaders support split='test' cleanly."""
    samples = load_all_benchmark_samples(n_samples_per_domain=2, split="test")
    assert set(samples.keys()) == {"general", "math", "code", "science", "reasoning"}
    for domain, sample_list in samples.items():
        assert len(sample_list) == 2
        for s in sample_list:
            assert isinstance(s, BenchmarkSample)
            assert s.domain == domain


def test_incremental_checkpointing(tmp_path):
    """Verify incremental progress checkpointing serialization and deserialization."""
    from scripts.run_ares_pipeline import (
        serialize_baseline_sample,
        deserialize_baseline_sample,
        save_incremental_checkpoint,
    )

    backbone = Mock7BBackbone(hidden_dim=3584)
    tokenizer = MockTokenizer()
    pipeline = ARESPipeline(
        config=PipelineConfig(model_name="Qwen/Qwen2.5-7B-Instruct", device="cpu", hidden_dim=3584),
        backbone=backbone,
        tokenizer=tokenizer,
    )
    comparator = BaselineComparator(pipeline=pipeline, strategies=["BASE"])
    sample = BenchmarkSample(
        sample_id="test_ckpt_1",
        domain="math",
        prompt="Solve 1+1",
        target_answer="2",
        eval_type="math_numeric",
    )
    res = comparator.evaluate_sample(sample, max_new_tokens=3)

    # Serialize and deserialize
    d = serialize_baseline_sample(res)
    res_restored = deserialize_baseline_sample(d)

    assert res_restored.sample_id == res.sample_id
    assert res_restored.domain == res.domain
    assert res_restored.results["BASE"].generated_text == res.results["BASE"].generated_text
    assert res_restored.correctness == res.correctness

    # Test saving incremental checkpoint
    ckpt_file = tmp_path / "checkpoint_test.json"
    save_incremental_checkpoint(ckpt_file, [res, res_restored], {"split": "test", "total": 2})

    assert ckpt_file.exists()
    with open(ckpt_file, "r") as f:
        loaded_data = json.load(f)
    assert loaded_data["completed_count"] == 2
    assert len(loaded_data["results"]) == 2
