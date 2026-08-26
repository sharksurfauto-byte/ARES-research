#!/usr/bin/env python
"""Execute all Week 3 operations.

This script runs:
1. Expert training (dummy testing loop)
2. Router training (dummy testing loop)
3. ExpertManager integration test
"""

import sys
import torch
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.experts.lora_expert import LoRAExpert, LoRAExpertConfig
from ares.experts.manager import ExpertManager, Router, RouterConfig


def test_experts(device: torch.device, output_path: Path):
    print(f"\n{'='*60}")
    print("Training LoRA Experts")
    print(f"{'='*60}")
    
    expert_names = ["general", "math", "code", "science", "reasoning"]
    domain_datasets = {
        "general": "wikitext",
        "math": "gsm8k",
        "code": "mbpp",
        "science": "ai2_arc",
        "reasoning": "custom_reasoning",
    }
    
    EPOCHS = 3
    LR = 1e-4
    BATCH_SIZE = 16
    
    for name in expert_names:
        expert_dir = output_path / name
        expert_dir.mkdir(parents=True, exist_ok=True)
    
        print(f"\nTraining expert: {name} (dataset: {domain_datasets[name]})")
    
        expert_cfg = LoRAExpertConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            expert_name=name,
            in_features=896,
            out_features=896,
        )
    
        expert = LoRAExpert(expert_cfg)
        expert.to(device)
        expert.train()
    
        optimizer = torch.optim.AdamW(expert.parameters(), lr=LR, weight_decay=0.01)
    
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            n_batches = 10 
    
            for batch_idx in range(n_batches):
                x = torch.randn(BATCH_SIZE, 896, device=device)
                target = x + 0.01 * torch.randn_like(x)
                out = expert(x)
                loss = torch.nn.functional.mse_loss(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
    
            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.6f}")
    
        ckpt_path = expert_dir / f"expert_{name}.pt"
        torch.save(
            {
                "expert_name": name,
                "config": {
                    "r": expert_cfg.r,
                    "lora_alpha": expert_cfg.lora_alpha,
                    "lora_dropout": expert_cfg.lora_dropout,
                    "target_modules": expert_cfg.target_modules,
                    "expert_name": expert_cfg.expert_name,
                    "in_features": expert_cfg.in_features,
                    "out_features": expert_cfg.out_features,
                },
                "state_dict": expert.state_dict(),
            },
            str(ckpt_path),
        )

def test_router(device: torch.device, output_path: Path):
    print(f"\n{'='*60}")
    print("Training Router")
    print(f"{'='*60}")

    router_cfg = RouterConfig(
        input_dim=896,
        hidden_dim=256,
        n_experts=5,
        dropout=0.1,
        temperature=1.0,
    )
    router = Router(router_cfg)
    router.to(device)
    router.train()
    
    manager = ExpertManager(
        input_dim=896,
        hidden_dim=256,
        n_experts=5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        router_dropout=0.1,
        router_temperature=1.0,
    )
    manager.to(device)

    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4, weight_decay=0.01)

    EPOCHS = 5
    BATCH_SIZE = 32
    LAMBDA_LB = 0.01
    N_BATCHES = 20
    
    for epoch in range(EPOCHS):
        epoch_ce_loss = 0.0
        epoch_lb_loss = 0.0
        epoch_total_loss = 0.0
    
        for batch_idx in range(N_BATCHES):
            x = torch.randn(BATCH_SIZE, 896, device=device)
            oracle_labels = torch.randint(0, 6, (BATCH_SIZE,), device=device)
    
            routing_probs = router(x)
            ce_loss = torch.nn.functional.cross_entropy(
                router.get_logits(x),
                oracle_labels,
            )
            lb_loss = manager.load_balancing_loss(routing_probs)
            total_loss = ce_loss + LAMBDA_LB * lb_loss
    
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
            epoch_ce_loss += ce_loss.item()
            epoch_lb_loss += lb_loss.item()
            epoch_total_loss += total_loss.item()
    
        avg_ce = epoch_ce_loss / N_BATCHES
        avg_lb = epoch_lb_loss / N_BATCHES
        avg_total = epoch_total_loss / N_BATCHES
        print(f"  Epoch {epoch+1}/{EPOCHS} — CE: {avg_ce:.4f} | LB: {avg_lb:.4f} | Total: {avg_total:.4f}")

    ckpt_path = output_path / "router.pt"
    torch.save(
        {
            "router_state_dict": router.state_dict(),
            "config": {
                "input_dim": router_cfg.input_dim,
                "hidden_dim": router_cfg.hidden_dim,
                "n_experts": router_cfg.n_experts,
                "dropout": router_cfg.dropout,
                "temperature": router_cfg.temperature,
            },
            "epochs": EPOCHS,
        },
        str(ckpt_path),
    )

def test_manager(device: torch.device):
    print(f"\n{'='*60}")
    print("Testing ExpertManager")
    print(f"{'='*60}")

    manager = ExpertManager(
        input_dim=896,
        hidden_dim=256,
        n_experts=5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        router_dropout=0.1,
        router_temperature=1.0,
    )
    manager.to(device)
    manager.train()

    x = torch.randn(8, 896, device=device)
    output, info = manager(x, return_routing_info=True)

    print(f"  Input shape:   {x.shape}")
    print(f"  Output shape:  {output.shape}")
    print(f"  Routing probs: {info['routing_probs'].shape}")

    loss = output.sum()
    loss.backward()

    has_grad = all(
        p.grad is not None
        for p in manager.parameters()
        if p.requires_grad
    )
    print(f"  Gradient check: {'PASSED ✓' if has_grad else 'FAILED ✗'}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path("checkpoints")
    experts_dir = output_dir / "experts"
    router_dir = output_dir / "router"
    
    # Cleanup old checkpoints for a clean run
    if experts_dir.exists():
        shutil.rmtree(experts_dir)
    if router_dir.exists():
        shutil.rmtree(router_dir)

    experts_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)

    test_experts(device, experts_dir)
    test_router(device, router_dir)
    test_manager(device)

if __name__ == "__main__":
    main()
