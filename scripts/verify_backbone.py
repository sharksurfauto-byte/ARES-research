#!/usr/bin/env python
"""Verification script for ARES backbone.

Loads Qwen2.5 model, runs forward pass, extracts hidden states,
and verifies checkpoint save/load with SHA256.

Usage:
    python scripts/verify_backbone.py --config configs/backbone/qwen_0_5b.yaml
    python scripts/verify_backbone.py --model Qwen/Qwen2.5-0.5B --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares import (
    BackboneConfig,
    load_backbone,
    verify_backbone,
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint,
    init_ddp,
    cleanup_ddp,
    is_distributed,
    is_main_process,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify ARES backbone")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/backbone/qwen_0_5b.yaml",
        help="Path to Hydra config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, auto)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=32,
        help="Test sequence length"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Test batch size"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/backbone",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--no-ddp",
        action="store_true",
        help="Skip DDP initialization"
    )
    return parser.parse_args()


def load_config(config_path: str) -> DictConfig:
    """Load Hydra config."""
    # Initialize hydra with config path (must be absolute)
    config_dir = Path(config_path).parent.resolve()
    hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None
    )
    cfg = hydra.compose(config_name=Path(config_path).stem)
    return cfg


def create_test_input(batch_size: int, seq_length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Create dummy input tensor."""
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def main():
    args = parse_args()

    # Initialize DDP if not disabled
    ddp_initialized = False
    if not args.no_ddp:
        try:
            ddp_initialized = init_ddp()
            if ddp_initialized:
                logger.info(f"DDP initialized: rank={get_rank()}, world_size={get_world_size()}")
        except Exception as e:
            logger.warning(f"DDP initialization failed: {e}")

    # Only run verification on main process
    if not is_main_process():
        if ddp_initialized:
            cleanup_ddp()
        return 0

    try:
        # Load config
        logger.info(f"Loading config from {args.config}")
        cfg = load_config(args.config)

        # Create backbone config
        backbone_cfg_dict = OmegaConf.to_container(cfg.backbone, resolve=True)
        if args.model:
            backbone_cfg_dict["name"] = args.model
        if args.device != "auto":
            backbone_cfg_dict["device_map"] = args.device

        backbone_config = BackboneConfig(**backbone_cfg_dict)

        # Load backbone
        logger.info(f"Loading backbone: {backbone_config.name}")
        backbone = load_backbone(backbone_config)

        # Verify backbone
        logger.info("Running backbone verification...")
        device = backbone.get_device()
        test_input = create_test_input(
            args.batch_size,
            args.seq_length,
            backbone.vocab_size,
            device
        )

        results = verify_backbone(backbone, test_input)

        # Print results
        logger.info("=" * 50)
        logger.info("VERIFICATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Model loaded: {results['model_loaded']}")
        logger.info(f"Forward pass: {results['forward_pass']}")
        logger.info(f"Logits shape: {results['logits_shape']}")
        logger.info(f"Hidden states extracted: {results['hidden_states_extracted']}")
        if results['hidden_states_shapes']:
            logger.info(f"Hidden states shapes: {results['hidden_states_shapes']}")
        logger.info(f"Errors: {results['errors']}")
        logger.info("=" * 50)

        # Check expected shapes
        expected_hidden_layers = len(backbone_config.hidden_state_layers) + 1  # +1 for embeddings
        if results['hidden_states_shapes']:
            actual_layers = len(results['hidden_states_shapes'])
            if actual_layers >= expected_hidden_layers:
                logger.info(f"✓ Hidden states: {actual_layers} layers (expected >= {expected_hidden_layers})")
            else:
                logger.warning(f"✗ Hidden states: {actual_layers} layers (expected >= {expected_hidden_layers})")

        # Test checkpoint save/load
        logger.info("Testing checkpoint save/load...")
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "verify_backbone_test.pt"

        # Save checkpoint
        save_checkpoint(
            model=backbone._model if hasattr(backbone, '_model') else backbone,
            epoch=0,
            step=1,
            metrics={"verification": "test"},
            path=checkpoint_path,
            config=backbone_config.to_dict(),
            verify_sha256=True,
        )

        # Verify checkpoint
        verify_results = verify_checkpoint(checkpoint_path)
        logger.info(f"Checkpoint verification: {verify_results}")

        # Load checkpoint into new model
        logger.info("Testing checkpoint load into new model...")
        backbone2 = load_backbone(backbone_config)
        load_checkpoint(
            path=checkpoint_path,
            model=backbone2._model if hasattr(backbone2, '_model') else backbone2,
            device=device,
            verify_sha256=True,
        )

        # Verify loaded model produces same output
        with torch.no_grad():
            out1 = backbone.forward(test_input)
            out2 = backbone2.forward(test_input)
            diff = (out1.logits - out2.logits).abs().max().item()
            logger.info(f"Max logit difference after load: {diff}")
            if diff < 1e-5:
                logger.info("✓ Checkpoint load produces identical outputs")
            else:
                logger.warning(f"✗ Checkpoint load difference: {diff}")

        # Summary
        logger.info("=" * 50)
        if results['forward_pass'] and results['hidden_states_extracted'] and verify_results['model_sha256_valid']:
            logger.info("✓ ALL VERIFICATIONS PASSED")
            return 0
        else:
            logger.error("✗ SOME VERIFICATIONS FAILED")
            return 1

    except Exception as e:
        logger.exception(f"Verification failed with error: {e}")
        return 1
    finally:
        if ddp_initialized:
            cleanup_ddp()


if __name__ == "__main__":
    sys.exit(main())