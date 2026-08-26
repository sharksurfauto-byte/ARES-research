"""Tests for Expert System: LoRA experts, Router, ExpertManager (PRD §3.2.5, §3.2.6)."""

import pytest
import torch
import torch.nn as nn

from ares.experts.lora_expert import LoRAExpert, LoRAExpertConfig, LoRALayer
from ares.experts.manager import ExpertManager, Router, RouterConfig


INPUT_DIM = 896
BATCH_SIZE = 4


class TestLoRALayer:
    def test_output_shape(self):
        layer = LoRALayer(in_features=INPUT_DIM, out_features=INPUT_DIM, r=16, lora_alpha=32)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        delta = layer(x)
        assert delta.shape == (BATCH_SIZE, INPUT_DIM)

    def test_initial_output_near_zero(self):
        layer = LoRALayer(in_features=INPUT_DIM, out_features=INPUT_DIM, r=16, lora_alpha=32)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        delta = layer(x)
        assert delta.abs().max().item() < 1e-6, "B is zero-init so output should be ~0"

    def test_different_in_out_features(self):
        layer = LoRALayer(in_features=512, out_features=1024, r=8, lora_alpha=16)
        x = torch.randn(BATCH_SIZE, 512)
        delta = layer(x)
        assert delta.shape == (BATCH_SIZE, 1024)

    def test_3d_input(self):
        layer = LoRALayer(in_features=INPUT_DIM, out_features=INPUT_DIM, r=16, lora_alpha=32)
        x = torch.randn(BATCH_SIZE, 10, INPUT_DIM)
        delta = layer(x)
        assert delta.shape == (BATCH_SIZE, 10, INPUT_DIM)

    def test_scaling_factor(self):
        layer = LoRALayer(in_features=64, out_features=64, r=8, lora_alpha=32)
        assert layer.scaling == 4.0

    def test_gradient_flows(self):
        layer = LoRALayer(in_features=INPUT_DIM, out_features=INPUT_DIM, r=16, lora_alpha=32)
        nn.init.normal_(layer.lora_B.weight, std=0.01)
        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        delta = layer(x)
        loss = delta.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.lora_A.weight.grad is not None
        assert layer.lora_B.weight.grad is not None


class TestLoRAExpert:
    def setup_method(self):
        self.config = LoRAExpertConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            expert_name="math",
            in_features=INPUT_DIM,
            out_features=INPUT_DIM,
        )
        self.expert = LoRAExpert(self.config)
        self.expert.eval()

    def test_output_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = self.expert(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_initial_output_is_identity(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = self.expert(x)
        diff = (out - x).abs().max().item()
        assert diff < 1e-4, f"Initial expert should be near-identity, got diff={diff}"

    def test_specific_module_forward(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = self.expert(x, module_name="q_proj")
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_invalid_module_name_raises(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        with pytest.raises(ValueError, match="Unknown module"):
            self.expert(x, module_name="nonexistent")

    def test_has_correct_number_of_lora_layers(self):
        assert len(self.expert.lora_layers) == 4

    def test_expert_name(self):
        assert self.expert.expert_name == "math"

    def test_config_scaling(self):
        assert self.config.scaling == 2.0

    def test_specialization_score(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        score = self.expert.specialization_score(x)
        assert score.shape == (BATCH_SIZE, 1)
        assert (score >= 0).all()

    def test_trainable_params_positive(self):
        n_params = self.expert.get_num_trainable_params()
        assert n_params > 0

    def test_3d_input(self):
        x = torch.randn(BATCH_SIZE, 10, INPUT_DIM)
        out = self.expert(x)
        assert out.shape == (BATCH_SIZE, 10, INPUT_DIM)

    def test_gradient_flows_through_gate(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        for expert in self.expert.lora_layers.values():
            nn.init.normal_(expert.lora_B.weight, std=0.01)
        out = self.expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestRouter:
    def setup_method(self):
        self.config = RouterConfig(
            input_dim=INPUT_DIM,
            hidden_dim=256,
            n_experts=5,
            dropout=0.0,
            temperature=1.0,
        )
        self.router = Router(self.config)
        self.router.eval()

    def test_output_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs = self.router(x)
        assert probs.shape == (BATCH_SIZE, 6)

    def test_output_is_probability_distribution(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs = self.router(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)
        assert (probs >= 0).all()

    def test_temperature_scaling(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        router_cold = Router(RouterConfig(input_dim=INPUT_DIM, hidden_dim=256, n_experts=5, temperature=0.1))
        router_hot = Router(RouterConfig(input_dim=INPUT_DIM, hidden_dim=256, n_experts=5, temperature=10.0))
        router_cold.mlp = self.router.mlp
        router_hot.mlp = self.router.mlp
        router_cold.eval()
        router_hot.eval()

        probs_cold = router_cold(x)
        probs_hot = router_hot(x)

        entropy_cold = -(probs_cold * probs_cold.log().clamp(min=-100)).sum(dim=-1).mean()
        entropy_hot = -(probs_hot * probs_hot.log().clamp(min=-100)).sum(dim=-1).mean()
        assert entropy_hot > entropy_cold, "Higher temperature should give higher entropy"

    def test_get_logits_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        logits = self.router.get_logits(x)
        assert logits.shape == (BATCH_SIZE, 6)

    def test_gradient_flows(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        probs = self.router(x)
        loss = probs.sum()
        loss.backward()
        assert x.grad is not None


class TestExpertManager:
    def setup_method(self):
        self.manager = ExpertManager(
            input_dim=INPUT_DIM,
            hidden_dim=256,
            n_experts=5,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            router_dropout=0.0,
        )
        self.manager.eval()

    def test_output_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = self.manager(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_output_shape_with_routing_info(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out, info = self.manager(x, return_routing_info=True)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)
        assert "routing_probs" in info
        assert info["routing_probs"].shape == (BATCH_SIZE, 6)
        assert "selected_experts" in info
        assert info["selected_experts"].shape == (BATCH_SIZE,)

    def test_routing_probs_valid_distribution(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        _, info = self.manager(x, return_routing_info=True)
        probs = info["routing_probs"]
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)
        assert (probs >= 0).all()

    def test_has_correct_number_of_experts(self):
        assert len(self.manager.experts) == 5

    def test_expert_names(self):
        names = [e.expert_name for e in self.manager.experts]
        assert names == ["general", "math", "code", "science", "reasoning"]

    def test_load_balancing_loss(self):
        x = torch.randn(32, INPUT_DIM)
        _, info = self.manager(x, return_routing_info=True)
        lb_loss = self.manager.load_balancing_loss(info["routing_probs"])
        assert lb_loss.shape == ()
        assert lb_loss.item() >= 0

    def test_load_balancing_loss_uniform(self):
        uniform_probs = torch.ones(32, 6) / 6.0
        lb_loss = self.manager.load_balancing_loss(uniform_probs)
        assert lb_loss.item() > 0

    def test_usage_stats(self):
        self.manager.reset_usage_stats()
        x = torch.randn(32, INPUT_DIM)
        self.manager(x)
        stats = self.manager.get_usage_stats()
        assert len(stats) > 0
        total = sum(stats.values())
        assert abs(total - 1.0) < 1e-5

    def test_reset_usage_stats(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        self.manager(x)
        self.manager.reset_usage_stats()
        stats = self.manager.get_usage_stats()
        assert len(stats) == 0

    def test_trainable_params(self):
        params = self.manager.get_num_trainable_params()
        assert params["router"] > 0
        assert params["total"] > 0
        assert len(params["experts"]) == 5

    def test_gradient_flows(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        for expert in self.manager.experts:
            for lora_layer in expert.lora_layers.values():
                nn.init.normal_(lora_layer.lora_B.weight, std=0.01)
        out = self.manager(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_custom_expert_names(self):
        mgr = ExpertManager(
            input_dim=64,
            n_experts=3,
            expert_names=["alpha", "beta", "gamma"],
            lora_r=4,
            lora_alpha=8,
        )
        names = [e.expert_name for e in mgr.experts]
        assert names == ["alpha", "beta", "gamma"]

    def test_single_sample(self):
        x = torch.randn(1, INPUT_DIM)
        out = self.manager(x)
        assert out.shape == (1, INPUT_DIM)

    def test_large_batch(self):
        x = torch.randn(128, INPUT_DIM)
        out = self.manager(x)
        assert out.shape == (128, INPUT_DIM)


class TestIntegration:
    def test_expert_manager_initial_near_identity(self):
        """At init, LoRA B=0 so experts are identity; output should approximate input."""
        manager = ExpertManager(
            input_dim=INPUT_DIM,
            n_experts=5,
            lora_dropout=0.0,
            router_dropout=0.0,
        )
        manager.eval()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = manager(x)
        diff = (out - x).abs().max().item()
        assert diff < 1e-4, f"Initial manager should be near-identity, got diff={diff}"

    def test_expert_config_from_schema(self):
        """Verify our config matches schema.py ExpertConfig defaults."""
        from ares.config.schema import ExpertConfig as SchemaExpertConfig

        schema = SchemaExpertConfig()
        config = LoRAExpertConfig(
            r=schema.r,
            lora_alpha=schema.lora_alpha,
            lora_dropout=schema.lora_dropout,
            target_modules=schema.target_modules,
            expert_name="general",
            in_features=INPUT_DIM,
            out_features=INPUT_DIM,
        )
        expert = LoRAExpert(config)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out = expert(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)

    def test_router_config_from_schema(self):
        """Verify our RouterConfig aligns with schema.py."""
        from ares.config.schema import RouterConfig as SchemaRouterConfig

        schema = SchemaRouterConfig()
        router = Router(RouterConfig(
            input_dim=INPUT_DIM,
            hidden_dim=schema.hidden_dim,
            n_experts=schema.num_experts,
            dropout=schema.dropout,
        ))
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs = router(x)
        assert probs.shape == (BATCH_SIZE, schema.num_experts + 1)

    def test_end_to_end_train_step(self):
        """Simulate a single training step."""
        manager = ExpertManager(
            input_dim=INPUT_DIM,
            n_experts=5,
            lora_dropout=0.0,
            router_dropout=0.0,
        )
        manager.train()

        for expert in manager.experts:
            for lora_layer in expert.lora_layers.values():
                nn.init.normal_(lora_layer.lora_B.weight, std=0.01)

        optimizer = torch.optim.Adam(manager.parameters(), lr=1e-4)

        x = torch.randn(16, INPUT_DIM)
        target = torch.randn(16, INPUT_DIM)

        out, info = manager(x, return_routing_info=True)
        mse_loss = nn.functional.mse_loss(out, target)
        lb_loss = manager.load_balancing_loss(info["routing_probs"])
        total_loss = mse_loss + 0.01 * lb_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0
