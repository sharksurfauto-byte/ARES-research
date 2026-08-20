# CLAUDE.md

Essential guidance for ARES Research. Detailed specs in ARES_RESEARCH_PRD.md.

## Status

**Week 1: COMPLETED** (2026-08-19) — Backbone infrastructure verified on Kaggle
- `src/ares/backbone/` — Qwen2.5 loader, frozen params, `use_cache=False`, eager attention
- `src/ares/utils/` — DDP, W&B, SHA256 checkpoints
- `configs/backbone/` — qwen_0_5b, qwen_1_5b, qwen_7b_4bit
- All 56 tests pass; `verify_backbone.py` verified on 2x T4

**Week 2: IN PROGRESS** — Representation Collector, GRM/LRM, Calibration

## Commands (use daily)

```bash
# Install
pip install -e ".[dev]"

# Lint & format
black .
ruff check . --fix

# Test (most common)
python -m pytest tests/ -v
python -m pytest tests/test_backbone_loader.py -v
python -m pytest tests/test_checkpoint.py -v
python -m pytest tests/test_ddp.py -v
python -m pytest tests/test_collect_representations.py -v
python -m pytest tests/test_grm.py -v
python -m pytest tests/test_lrm.py -v

# Quick verify
python scripts/generate_sample.py --prompt "Artificial intelligence is important because"

# Week 2 scripts (run on Kaggle)
python scripts/collect_representations.py --config configs/reliability/representation_collection.yaml --model_name Qwen/Qwen2.5-0.5B --max_samples 100 --analyze
python scripts/train_reliability_models.py --config configs/reliability/reliability_models.yaml --input_dir representations/ --output_dir checkpoints/reliability --epochs 10
```

## Architecture (big picture)

**ARES**: Adaptive Reliability with Expert Specialization — learned routing layer on frozen Qwen2.5 backbone.

- **Backbone**: Qwen2.5 (0.5B/1.5B/7B, 4-bit NF4 via bitsandbytes), `use_cache=False`, `attn_implementation="eager"`
- **Layer 1**: Representation Collector (layers `{-1, -6, -12, -24}`, pooled hidden states)
- **Layer 2**: GRM (2-layer transformer encoder) → domain + global reliability
- **Layer 3**: LRM (2-layer transformer) → token-wise correctness prob
- **Layer 4**: Router (MLP, hidden=256) → {base, expert_0..4}
- **Layer 5**: LoRA experts (r=16, alpha=32): E0-general, E1-math, E2-code, E3-science, E4-reasoning

**Critical**: Frozen backbone, `use_cache=False`, config-driven (YAML under `configs/`)

## Development Workflow

```bash
git checkout -b feature/<name>
# make changes
black . && ruff check . --fix
python -m pytest tests/ -v  # before commit
```

## Import Verification (Critical)

**Before committing any file, verify imports work:**

```bash
# Quick syntax + import check
python -c "import sys; sys.path.insert(0, 'src'); import ares; print('OK')"

# Or for specific module
python -c "import sys; sys.path.insert(0, 'src'); from ares.representations import RepresentationCollector; print('OK')"

# Or run tests
python -m pytest tests/ -v --tb=short
```

**Rules:**
1. After writing any `.py` file, run the import check above
2. All imports must resolve without `ModuleNotFoundError` or `ImportError`
3. Relative imports must use proper depth (`from ..module import X` not `from ...module import X`)
4. Add missing `__init__.py` files for new packages
5. Export all public classes/functions in `__init__.py` files

## What's in ARES_RESEARCH_PRD.md (reference)

- PRD §3: Architecture diagram & component details
- PRD §4: Training pipelines (GRM, LRM, router, experts)
- PRD §5: Evaluation framework, baselines B0-B4, metrics
- PRD §6: 4-week timeline (Week 1 ✅, Week 2 🔄)
- PRD §7: Hardware, config structure, abbreviations
- PRD §11: 8-phase quick-start bash commands
- PRD §9: Risk mitigation table
- PRD §10, §12: Paper structure & success criteria

---

*Keep CLAUDE.md lean. All detailed specs, scripts, and configurations are in ARES_RESEARCH_PRD.md. Edit CLAUDE.md only when the essential frequently-needed info changes.*