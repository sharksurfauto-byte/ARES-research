#!/usr/bin/env python
"""Train GRM and LRM reliability models (PRD §11 #5).

Trains the Global Reliability Model (GRM) and Local Reliability Model (LRM)
on collected representations.

Usage:
    python scripts/train_reliability_models.py \
        --config configs/reliability/reliability_models.yaml \
        --input_dir representations/ \
        --output_dir checkpoints/reliability \
        --epochs 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from ares import GRM, LRM, GRMTrainer, LRMTrainer
from ares.calibration import (
    TemperatureScaling,
    compute_ece,
    fit_temperature_scaling,
)
from ares.utils.wandb_utils import init_wandb, log_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRM and LRM reliability models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reliability/reliability_models.yaml",
        help="Path to reliability models config",
    )
    parser.add_argument(
        "--data_dir",
        "--input_dir",
        dest="input_dir",
        type=str,
        default="representations/multi_domain",
        help="Directory containing collected representations (e.g. representations/multi_domain)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/reliability",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply temperature scaling + isotonic calibration after training",
    )
    return parser.parse_args()


def load_representations(input_dir: str) -> dict[str, torch.Tensor]:
    """Load representations from dataset directory.

    Args:
        input_dir: Directory with .pt files from collect_representations.py or harvest_real_data.py

    Returns:
        Dictionary with train/val tensors
    """
    input_path = Path(input_dir)
    train_pt = input_path / "train.pt"
    val_pt = input_path / "val.pt"

    # 1. Check for standard train.pt & val.pt partitions from RepresentationDataset
    if train_pt.exists():
        from ares.representations.dataset import RepresentationDataset

        train_ds = RepresentationDataset.load(train_pt)
        train_tensors = train_ds.get_tensors()
        train_reps = train_tensors["representations"]
        train_domain = train_tensors["domain_labels"]
        train_feas = train_tensors["feasibility_labels"]

        val_reps, val_domain, val_feas = None, None, None
        n_val = 0
        if val_pt.exists():
            val_ds = RepresentationDataset.load(val_pt)
            if len(val_ds) > 0:
                val_tensors = val_ds.get_tensors()
                val_reps = val_tensors["representations"]
                val_domain = val_tensors["domain_labels"]
                val_feas = val_tensors["feasibility_labels"]
                n_val = len(val_ds)

        return {
            "train_representations": train_reps,
            "val_representations": val_reps if val_reps is not None else train_reps[:max(1, len(train_reps)//10)],
            "train_domain_labels": train_domain,
            "val_domain_labels": val_domain if val_domain is not None else train_domain[:max(1, len(train_domain)//10)],
            "train_feasibility_labels": train_feas,
            "val_feasibility_labels": val_feas if val_feas is not None else train_feas[:max(1, len(train_feas)//10)],
            "n_train": len(train_ds),
            "n_val": n_val if n_val > 0 else max(1, len(train_ds)//10),
            "input_dim": train_reps.shape[1],
        }

    # 2. Fallback: single .pt file
    pt_files = list(input_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    data = torch.load(pt_files[0], weights_only=False)

    representations = data.get("representations", [])
    samples = data.get("samples", [])

    if not representations and not samples:
        raise ValueError(f"No representations or samples found in {pt_files[0]}")

    if representations:
        reps_tensor = (
            torch.stack(representations)
            if isinstance(representations[0], torch.Tensor)
            else torch.tensor(representations)
        ).detach()
    else:
        reps_tensor = torch.stack([s.representation.detach().cpu() for s in samples])

    num_layers = 4  # default: -1, -6, -12, -24
    n_samples = len(samples) if samples else reps_tensor.shape[0]

    n = n_samples
    perm = torch.randperm(n)
    n_val = max(1, n // 10)
    n_train = n - n_val

    train_sample_idx = perm[:n_train]
    val_sample_idx = perm[n_train:]

    def sample_idx_to_rep_idx(sample_idx, num_layers=4):
        return [sample_idx * num_layers + layer for layer in range(num_layers)]

    if len(reps_tensor) == n_samples * num_layers:
        train_rep_idx = [idx for si in train_sample_idx for idx in sample_idx_to_rep_idx(si)]
        val_rep_idx = [idx for si in val_sample_idx for idx in sample_idx_to_rep_idx(si)]
    else:
        train_rep_idx = train_sample_idx
        val_rep_idx = val_sample_idx

    train_reps = reps_tensor[train_rep_idx]
    val_reps = reps_tensor[val_rep_idx]

    train_labels = None
    val_labels = None

    if samples is not None and len(samples) > 0:
        correct_flags = [s.correctness for s in samples]
        correct_tensor = torch.tensor(correct_flags, dtype=torch.float)
        if len(train_reps) == len(train_sample_idx) * num_layers:
            train_labels = correct_tensor[train_sample_idx].repeat_interleave(num_layers)
            val_labels = correct_tensor[val_sample_idx].repeat_interleave(num_layers)
        else:
            train_labels = correct_tensor[train_sample_idx]
            val_labels = correct_tensor[val_sample_idx]

    domain2idx = {"general": 0, "math": 1, "code": 2, "science": 3, "reasoning": 4}
    if samples:
        if len(train_reps) == len(train_sample_idx) * num_layers:
            train_domain = torch.tensor(
                [domain2idx.get(samples[i].domain, 0) for i in train_sample_idx]
            ).repeat_interleave(num_layers)
            val_domain = torch.tensor(
                [domain2idx.get(samples[i].domain, 0) for i in val_sample_idx]
            ).repeat_interleave(num_layers)
        else:
            train_domain = torch.tensor(
                [domain2idx.get(samples[i].domain, 0) for i in train_sample_idx]
            )
            val_domain = torch.tensor(
                [domain2idx.get(samples[i].domain, 0) for i in val_sample_idx]
            )
    else:
        train_domain = torch.zeros(len(train_rep_idx), dtype=torch.long)
        val_domain = torch.zeros(len(val_rep_idx), dtype=torch.long)

    return {
        "train_representations": train_reps,
        "val_representations": val_reps,
        "train_domain_labels": train_domain,
        "val_domain_labels": val_domain,
        "train_feasibility_labels": train_labels,
        "val_feasibility_labels": val_labels,
        "n_train": n_train,
        "n_val": n_val,
        "input_dim": reps_tensor.shape[1],
    }


def main():
    args = parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # If config already includes "configs/" prefix, use as-is from project root
        if args.config.startswith("configs/"):
            config_path = Path(__file__).parent.parent / args.config
        else:
            config_path = Path(__file__).parent.parent / "configs" / args.config

    # Load config
    config_dict = OmegaConf.load(str(config_path))

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Load representations
        logger.info(f"Loading representations from {args.input_dir}")
        data = load_representations(args.input_dir)

        input_dim = data["input_dim"]
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Train samples: {data['n_train']}, Val samples: {data['n_val']}")

        # Check backbone hidden size
        logger.info(f"Using representation feature dimension: {input_dim}")

        # Initialize W&B
        wandb_logger = init_wandb(
            config={"reliability": dict(config_dict)},
            project="ares-research",
            mode="online" if __import__("sys").platform != "cli" else "online",
        )

        # ========== Train GRM ==========
        logger.info("=" * 50)
        logger.info("Training Global Reliability Model (GRM)")
        logger.info("=" * 50)

        grm = GRM(
            input_dim=input_dim,
            **{
                "hidden_dim": config_dict.get("grm", {}).get("hidden_dim", 512),
                "num_layers": config_dict.get("grm", {}).get("num_layers", 2),
                "num_heads": config_dict.get("grm", {}).get("num_heads", 4),
                "dropout": config_dict.get("grm", {}).get("dropout", 0.1),
            },
        ).to(device)

        grm_trainer = GRMTrainer(
            model=grm,
            device=device,
            config={"learning_rate": args.lr, "batch_size": args.batch_size},
            wandb_logger=wandb_logger,
        )

        # Prepare and sanitize tensors
        train_reps = torch.nan_to_num(data["train_representations"].float().to(device), nan=0.0)
        train_domain = torch.clamp(data["train_domain_labels"].long().to(device), 0, 4)
        train_feas = (
            torch.clamp(data["train_feasibility_labels"].float().to(device), 0.0, 1.0)
            if data["train_feasibility_labels"] is not None
            else torch.ones(train_reps.shape[0], device=device)
        )

        val_reps = (
            torch.nan_to_num(data["val_representations"].float().to(device), nan=0.0)
            if data["val_representations"] is not None
            else None
        )
        val_domain = (
            torch.clamp(data["val_domain_labels"].long().to(device), 0, 4)
            if data["val_domain_labels"] is not None
            else None
        )
        val_feas = (
            torch.clamp(data["val_feasibility_labels"].float().to(device), 0.0, 1.0)
            if data["val_feasibility_labels"] is not None
            else None
        )

        # Train GRM
        grm_history = grm_trainer.train(
            representations=train_reps,
            domain_labels=train_domain,
            feasibility_labels=train_feas,
            epochs=args.epochs,
            val_representations=val_reps,
            val_domain_labels=val_domain,
            val_feasibility_labels=val_feas,
        )

        # Save GRM checkpoint
        grm_path = Path(args.output_dir) / "grm.pt"
        grm_trainer.save(str(grm_path))
        logger.info(f"GRM checkpoint saved to {grm_path}")

        # Calibrate GRM if requested
        if args.calibrate:
            logger.info("Calibrating GRM...")
            # Get validation logits and labels
            grm.eval()
            with torch.no_grad():
                val_repr = data["val_representations"].to(device)
                val_domain = data["val_domain_labels"].to(device)
                val_feas = (
                    data["val_feasibility_labels"].to(device)
                    if data["val_feasibility_labels"] is not None
                    else torch.ones(val_repr.shape[0], device=device)
                )

                # Get domain logits
                domain_logits, feasibility, global_rel = grm(val_repr)

            # Fit temperature scaling on feasibility
            temp_scaler = TemperatureScaling(device=device)
            temp_fit = temp_scaler.fit(
                feasibility,
                val_feas,
                epochs=10,
            )
            logger.info(f"GRM temperature: {temp_fit['temperature']:.4f}")

        # ========== Train LRM ==========
        logger.info("=" * 50)
        logger.info("Training Local Reliability Model (LRM)")
        logger.info("=" * 50)

        # Initialize LRM
        lrm = LRM(
            input_dim=input_dim,
            **{
                "hidden_dim": config_dict.get("lrm", {}).get("hidden_dim", 512),
                "num_layers": config_dict.get("lrm", {}).get("num_layers", 2),
                "num_heads": config_dict.get("lrm", {}).get("num_heads", 4),
                "dropout": config_dict.get("lrm", {}).get("dropout", 0.1),
            },
        ).to(device)

        # For LRM, we need token-level data. Create simple token-level dataset
        # from the collected representations
        train_hidden = data["train_representations"].to(device)
        train_labels = (
            data["train_feasibility_labels"].to(device)
            if data["train_feasibility_labels"] is not None
            else torch.ones(train_hidden.shape[0], device=device)
        )

        # Repeat representations for token-level (simulate per-token hidden states)
        # In practice, these would come from the actual backbone hidden states
        seq_len = 32  # Assume fixed sequence length

        # Tile hidden states to simulate sequence dimension
        train_hidden_tiled = (
            train_hidden.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, input_dim)
        )
        train_labels_tiled = train_labels.unsqueeze(1).expand(-1, seq_len).reshape(-1).float()

        # Create attention mask matching tiled tensor size
        train_mask = torch.ones(train_hidden_tiled.shape[0], device=device).float()

        lrm_trainer = LRMTrainer(
            model=lrm,
            device=device,
            config={"learning_rate": args.lr, "batch_size": args.batch_size, "pos_weight": 1.0},
            wandb_logger=wandb_logger,
        )

        # Train LRM
        lrm_history = lrm_trainer.train(
            token_hidden_states=train_hidden_tiled,
            correctness_labels=train_labels_tiled,
            attention_mask=train_mask,
            epochs=args.epochs,
        )

        # Save LRM checkpoint
        lrm_path = Path(args.output_dir) / "lrm.pt"
        lrm_trainer.save(str(lrm_path))
        logger.info(f"LRM checkpoint saved to {lrm_path}")

        # Calibrate LRM if requested
        if args.calibrate:
            logger.info("Calibrating LRM...")
            lrm.eval()
            with torch.no_grad():
                # Get predictions
                correctness_prob, failure_risk = lrm(train_hidden.to(device))

                # Move to numpy for calibration
                probs = correctness_prob.squeeze().cpu().numpy()
                true_labels = train_labels.cpu().numpy()

                # Filter by valid values
                valid_probs = probs
                valid_labels = true_labels

                if len(valid_probs) > 0 and len(valid_labels) > 0:
                    # Fit temperature scaling
                    temp_fit = fit_temperature_scaling(
                        torch.tensor(valid_probs, device=device),
                        torch.tensor(valid_labels, device=device),
                        epochs=10,
                    )
                    logger.info(f"LRM temperature: {temp_fit['temperature']:.4f}")

                    # Apply isotonic regression
                    from ares.calibration.isotonic import (
                        apply_isotonic_regression,
                        fit_isotonic_regression,
                    )

                    ir_model = fit_isotonic_regression(valid_probs, valid_labels)
                    calibrated_probs = apply_isotonic_regression(valid_probs, ir_model)

                    # Compute ECE before and after
                    ece_before = compute_ece(valid_probs, valid_labels)
                    ece_after = compute_ece(calibrated_probs, valid_labels)
                    logger.info(f"LRM ECE before calibration: {ece_before:.4f}")
                    logger.info(f"LRM ECE after calibration: {ece_after:.4f}")
                    logger.info(f"LRM ECE improvement: {ece_before - ece_after:.4f}")

        # Save final history
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Log final metrics to W&B
        if wandb_logger is not None:
            log_metrics(
                wandb_logger,
                {
                    "grm/final_train_loss": grm_history["train"][-1]["loss"],
                    "grm/final_train_acc": grm_history["train"][-1]["domain_accuracy"],
                    "lrm/final_train_loss": lrm_history["train"][-1]["loss"],
                    "lrm/final_train_acc": lrm_history["train"][-1]["accuracy"],
                },
            )

        logger.info("Reliability models training complete!")
        print(f"Checkpoints saved to {args.output_dir}/")
        print("  - grm.pt (Global Reliability Model)")
        print("  - lrm.pt (Local Reliability Model)")

    except Exception as e:
        logger.error(f"Reliability models training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
