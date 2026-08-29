"""Tests for Router Network, Switch Transformer LB Loss, and Router Trainer (PRD §3.2.5, §4.4)."""

import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ares.router import (
    EXPERT_NAMES,
    ROUTE_NAMES,
    Router,
    RouterConfig,
    RouterLoss,
    RouterTrainer,
    RoutingOutput,
    SwitchLoadBalancingLoss,
    generate_oracle_targets,
)
from ares.experts.manager import ExpertManager


INPUT_DIM = 896
BATCH_SIZE = 8


class TestRouterArchitecture:
    def setup_method(self):
        self.config = RouterConfig(
            input_dim=INPUT_DIM,
            hidden_dim=256,
            num_layers=2,
            n_experts=5,
            dropout=0.0,
            temperature=1.0,
            top_k=1,
            routing_mode="soft",
        )
        self.router = Router(self.config)
        self.router.eval()

    def test_output_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs = self.router(x)
        assert probs.shape == (BATCH_SIZE, 6)

    def test_3d_input_shape(self):
        x = torch.randn(BATCH_SIZE, 10, INPUT_DIM)
        probs = self.router(x)
        assert probs.shape == (BATCH_SIZE, 10, 6)

    def test_output_is_probability_distribution(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs = self.router(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)
        assert (probs >= 0).all()

    def test_temperature_scaling(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs_cold = self.router(x, temperature=0.1)
        probs_hot = self.router(x, temperature=10.0)

        entropy_cold = -(probs_cold * (probs_cold + 1e-8).log()).sum(dim=-1).mean()
        entropy_hot = -(probs_hot * (probs_hot + 1e-8).log()).sum(dim=-1).mean()
        assert entropy_hot > entropy_cold

    def test_top_k_routing(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        # Top-1
        probs_top1 = self.router(x, mode="top_k")
        assert probs_top1.shape == (BATCH_SIZE, 6)
        # In top-1, exactly 1 non-zero element per sample
        non_zeros = (probs_top1 > 0).sum(dim=-1)
        assert (non_zeros == 1).all()

    def test_top_2_routing(self):
        cfg = RouterConfig(input_dim=INPUT_DIM, hidden_dim=256, n_experts=5, top_k=2, routing_mode="top_k")
        router = Router(cfg)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs = router(x)
        non_zeros = (probs > 0).sum(dim=-1)
        assert (non_zeros == 2).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)

    def test_gumbel_softmax_routing(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        probs_gumbel = self.router(x, mode="gumbel_softmax", hard=False)
        assert probs_gumbel.shape == (BATCH_SIZE, 6)
        assert torch.allclose(probs_gumbel.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)

        # Hard gumbel softmax (one-hot with straight-through gradient)
        probs_hard = self.router(x, mode="gumbel_softmax", hard=True)
        assert ((probs_hard == 0) | (probs_hard == 1)).all()
        assert torch.allclose(probs_hard.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5)

    def test_return_dict(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        output = self.router(x, return_dict=True)
        assert isinstance(output, RoutingOutput)
        assert output.routing_probs.shape == (BATCH_SIZE, 6)
        assert output.logits.shape == (BATCH_SIZE, 6)
        assert output.selected_experts.shape == (BATCH_SIZE,)
        assert output.confidence.shape == (BATCH_SIZE,)
        assert output.base_prob.shape == (BATCH_SIZE, 1)
        assert output.expert_probs.shape == (BATCH_SIZE, 5)

    def test_route_method(self):
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        selected, confidence, probs = self.router.route(x)
        assert selected.shape == (BATCH_SIZE,)
        assert confidence.shape == (BATCH_SIZE,)
        assert probs.shape == (BATCH_SIZE, 6)
        assert (confidence >= 0).all() and (confidence <= 1.0 + 1e-5).all()

    def test_save_and_load_checkpoint(self, tmp_path):
        save_file = tmp_path / "router_test.pt"
        self.router.save_checkpoint(save_file)
        assert save_file.exists()

        loaded_router = Router.load_checkpoint(save_file)
        loaded_router.eval()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        orig_out = self.router(x)
        loaded_out = loaded_router(x)
        assert torch.allclose(orig_out, loaded_out, atol=1e-6)


class TestSwitchLoadBalancingLoss:
    def test_uniform_distribution_loss(self):
        lb_loss_fn = SwitchLoadBalancingLoss(n_classes=6, coeff=1.0)
        # Uniform probs across 6 classes
        uniform_probs = torch.ones(60, 6) / 6.0
        # If equal batch assigned to each class (10 items per class)
        loss = lb_loss_fn(uniform_probs)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_imbalanced_distribution_higher_loss(self):
        lb_loss_fn = SwitchLoadBalancingLoss(n_classes=6, coeff=1.0)
        # Uniform probs
        uniform_probs = torch.ones(60, 6) / 6.0
        # Heavily skewed probs (everything to class 0)
        skewed_probs = torch.zeros(60, 6)
        skewed_probs[:, 0] = 0.95
        skewed_probs[:, 1:] = 0.01

        uniform_loss = lb_loss_fn(uniform_probs)
        skewed_loss = lb_loss_fn(skewed_probs)
        assert skewed_loss.item() > uniform_loss.item()

    def test_gradient_flows_through_loss(self):
        router = Router(RouterConfig(input_dim=INPUT_DIM, hidden_dim=256, n_experts=5))
        lb_loss_fn = SwitchLoadBalancingLoss(n_classes=6, coeff=0.01)
        x = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        probs = router(x)
        loss = lb_loss_fn(probs)
        loss.backward()
        assert x.grad is not None


class TestOracleTargetGeneration:
    def test_oracle_targets(self):
        # 5 samples, one from each domain (0: general, 1: math, 2: code, 3: science, 4: reasoning)
        domain_labels = torch.tensor([0, 1, 2, 3, 4])
        # Sample 0 and 2 are correct (base model succeeded); sample 1, 3, 4 failed
        correctness_labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])

        targets = generate_oracle_targets(domain_labels, correctness_labels, mode="oracle")

        # Correct samples should route to 0 (base)
        assert targets[0].item() == 0  # base
        assert targets[2].item() == 0  # base
        # Failed samples should route to domain_id + 1
        assert targets[1].item() == 2  # math expert (domain 1 -> class 2)
        assert targets[3].item() == 4  # science expert (domain 3 -> class 4)
        assert targets[4].item() == 5  # reasoning expert (domain 4 -> class 5)

    def test_expert_only_mode(self):
        domain_labels = torch.tensor([0, 1, 2, 3, 4])
        correctness_labels = torch.ones(5)
        targets = generate_oracle_targets(domain_labels, correctness_labels, mode="expert_only")
        assert (targets == torch.tensor([1, 2, 3, 4, 5])).all()

    def test_base_only_mode(self):
        domain_labels = torch.tensor([0, 1, 2, 3, 4])
        correctness_labels = torch.zeros(5)
        targets = generate_oracle_targets(domain_labels, correctness_labels, mode="base_only")
        assert (targets == torch.zeros(5)).all()


class TestRouterTrainer:
    def setup_method(self):
        self.config = RouterConfig(
            input_dim=INPUT_DIM,
            hidden_dim=256,
            n_experts=5,
            dropout=0.1,
            temperature=1.0,
        )
        self.router = Router(self.config)
        self.device = torch.device("cpu")
        self.trainer = RouterTrainer(
            router=self.router,
            device=self.device,
            config={"learning_rate": 1e-3, "lambda_lb": 0.01, "epochs": 2},
        )

    def test_trainer_run_and_eval(self):
        n_samples = 32
        train_x = torch.randn(n_samples, INPUT_DIM)
        train_y = torch.randint(0, 6, (n_samples,))
        val_x = torch.randn(16, INPUT_DIM)
        val_y = torch.randint(0, 6, (16,))

        history = self.trainer.train(
            train_representations=train_x,
            train_targets=train_y,
            val_representations=val_x,
            val_targets=val_y,
            epochs=2,
            batch_size=8,
        )

        assert len(history) == 2
        assert "train/total_loss" in history[0]
        assert "val/accuracy" in history[0]
        assert "val/entropy" in history[0]

    def test_checkpoint_compatibility_with_pipeline(self, tmp_path):
        ckpt_path = tmp_path / "router.pt"
        self.trainer.save_checkpoint(ckpt_path, epoch=2, metrics={"val/accuracy": 0.85})

        assert ckpt_path.exists()
        loaded_dict = torch.load(str(ckpt_path), weights_only=False)

        # Must have "router_state_dict" for scripts/run_ares_pipeline.py
        assert "router_state_dict" in loaded_dict
        # Must have "model_state_dict" for ARES utils
        assert "model_state_dict" in loaded_dict
        assert "config" in loaded_dict
        assert "sha256" in loaded_dict

        # Verify expert manager can load this checkpoint directly
        manager = ExpertManager(input_dim=INPUT_DIM)
        manager.router.load_state_dict(loaded_dict["router_state_dict"])

        x = torch.randn(4, INPUT_DIM)
        out, info = manager(x, return_routing_info=True)
        assert out.shape == (4, INPUT_DIM)
        assert "routing_probs" in info
