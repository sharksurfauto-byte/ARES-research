# CLAUDE.md

Essential guidance for ARES Research. Detailed specs in ARES_RESEARCH_PRD.md.

## Commands (use daily)

```bash
# Install
pip install -e ".[dev]"

# Lint & format
black .
ruff check . --fix

# Test (most common)
python -m pytest tests/ -v
python -m pytest tests/test_phase0_utils.py -v
python -m pytest tests/test_collect_representations.py -v
python -m pytest tests/test_train_reliability.py -v
python -m pytest tests/test_train_experts.py -v
python -m pytest tests/test_evaluate_routing.py -v

# Quick verify
python scripts/generate_sample.py --prompt "Artificial intelligence is important because"
```

## Architecture (big picture)

**ARES**: Adaptive Reliability with Expert Specialization — learned routing layer on frozen Qwen2.5 backbone.

- **Backbone**: Qwen2.5 (0.5B/1.5B/7B, 4-bit NF4 via bitsandbytes), `use_cache=False`, `attn_implementation="eager"`
- **Reliability**: GRM (global, pooled) + LRM (local, token-wise)
- **Router**: MLP routes to {base, expert_0..4} (E0-general, E1-math, E2-code, E3-science, E4-reasoning)
- **Experts**: LoRA adapters (r=16, alpha=32), domain-specialized

**Critical**: Frozen backbone, `use_cache=False`, config-driven (YAML under `configs/`)

## Development Workflow

```bash
git checkout -b feature/<name>
# make changes
black . && ruff check . --fix
python -m pytest tests/ -v  # before commit
```

## What's in ARES_RESEARCH_PRD.md (reference)

- PRD §3: Full architecture diagram & component details
- PRD §4: Training pipelines (GRM, LRM, router, experts)
- PRD §5: Evaluation framework, baselines B0-B4, metrics
- PRD §6: 4-week timeline
- PRD §7: Hardware, config structure, abbreviations
- PRD §11: 8-phase quick-start bash commands
- PRD §9: Risk mitigation table
- PRD §10, §12: Paper structure & success criteria

---

*Keep CLAUDE.md lean. All detailed specs, scripts, and configurations are in ARES_RESEARCH_PRD.md. Edit CLAUDE.md only when the essential frequently-needed info changes.*