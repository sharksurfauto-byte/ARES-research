# ARES Research PRD — Adaptive Routing with Expert Specialization

## Project: ARES — Adaptive Reliability with Expert Specialization

## Version: 1.0 — Research Paper Edition

## Date: 2026-08-18

## Primary Contact: Aliasghar Jawadwala

---

## Executive Summary

This paper presents **ARES: Adaptive Expert Routing with Specialization**, a framework that adds a reliability-aware routing layer on top of a frozen pretrained language model backbone. The core contribution is a **learned adaptive routing algorithm** that uses dual reliability signals (Global Reliability Model + Local Reliability Model) to dynamically decide per-token whether to use the base model or invoke a specialized expert.

**Key claims:**
1. A learned router can outperform simple threshold-based routing while maintaining selective computation
2. Domain-specialized LoRA experts trained on targeted failure modes improve accuracy over a single base model
3. The system achieves better accuracy than the base model with 60-70% fewer expert invocations than always-on expert systems
4. Calibrated reliability scores enable trustworthy selective prediction (the system can correctly identify when it should not trust its own output)

**Hardware**: 2x T4 GPUs on Kaggle (DDP), single-GPU local with 4-bit NF4 quantization

**Model**: Qwen2.5-0.5B / 1.5B for development; Qwen2.5-7B with bitsandbytes 4-bit for final experiments

---

## 1. Problem Statement

Modern LLMs produce fluent but incorrect outputs. Standard confidence mechanisms (token probability, entropy) are insufficient — a token can have high probability while the statement is factually wrong.

**The fundamental question this paper answers:**
> Can we build a lightweight adaptive layer that monitors internal representations, estimates when the model is likely to be wrong, and selectively invokes specialized computation only when necessary?

---

## 2. Research Hypotheses

### H1 — Routing Efficacy
>A learned adaptive router using GRM+LRM signals achieves higher accuracy than threshold-based routing at equivalent expert utilization, and achieves higher accuracy with fewer expert invocations than always-on expert systems.

### H2 — Expert Specialization
>Domain-specialized LoRA experts (E0-general, E1-math, E2-code, E3-science, E4-reasoning) trained on targeted failure modes produce measurable specialization effects — each expert outperforms the base model on its designated domain.

### H3 — Calibration
>GRM/LRM reliability scores are well-calibrated (low ECE) enabling trustworthy selective prediction — when the system says R(x) < 0.5, the prediction is indeed wrong more than 50% of the time.

### H4 — Selective Compute
>By routing only low-reliability tokens to experts, the system achieves better accuracy-base tradeoff than both (a) the base model alone and (b) always-on expert systems, measured by the selective risk-coverage curve.

---

## 3. Architecture

### 3.1 High-Level Diagram

```text
                           ┌─────────────────────────┐
                           │  Frozen Qwen Backbone   │
                           │     (parameters fixed)  │
                           └───────┬─────────────────┘
                                   │
                                   ▼
                           ┌─────────────────────────┐
                           │  Hidden Representations │
                           │  (from selected layers) │
                           └───────┬─────────────────┘
                                   │
    ┌────────────────────────────┼─────────────────────────────┐
    │                            │                             │
    ▼                            ▼                             ▼
┌─────────────────┐      ┌─────────────────┐         ┌─────────────────┐
│   Global RM     │      │   Local RM      │         │  Router Network │
│   (Domain +    │      │   (Token-level) │         │   (Learned)     │
│   Global Rel.) │      │                 │         └───────┬─────────┘
└───────┬─────────┘      └─────────────────┘           │
        │                          │                    │
        │                          │                    │
        ▼                          ▼                    ▼
   ┌───────────────┐          ┌──────────────┐   ┌─────────────────────┐
   │  Reliability  │          │  Failure     │   │  Routing Decision   │
   │  Score R(x)   │          │  Risk        │   │  (expert/no-expert) │
   └───────┬─────────┘          └──────────────┘   └───────┬─────────────┘
           │                              │                │
           │                              │                │
           └──────────────────────────────┘                │
                                                             ▼
                                                    ┌─────────────────────┐
                                                    │  Output Generation  │
                                                    │  (Base or Expert)   │
                                                    └─────────────────────┘
```

### 3.2 Component Details

#### 3.2.1 Backbone (Layer 0)
- **Qwen2.5** family: 0.5B, 1.5B, or 7B (4-bit NF4)
- **Frozen** — no gradient updates to backbone weights
- `output_hidden_states=True` enabled for representation extraction
- `attn_implementation="eager"` for output_attentions=True compatibility
- Device: 2x T4 Kaggle DDP or single GPU with `device_map="auto"` + 4-bit

#### 3.2.2 Representation Collector (Layer 1)
- Extracts hidden states from **multiple layers**: {-1, -6, -12, -24} (negative = from end)
- **Pooling methods**: last-token, mean-pooled, max-pooled
- **Output**: representation vectors of dimension `hidden_dim`
- Stores with: sample_id, domain, task, layer, representation, logits, prediction, correctness, confidence, entropy, margin

#### 3.2.3 Global Reliability Model (GRM) — Layer 2
- **Architecture**: 2-layer transformer encoder (hidden dim = 2× input_dim, 4 attention heads)
- **Inputs**: pooled hidden representation
- **Outputs**:
  - `domain_logits`: 4-class (code, general, math, science, reasoning)
  - `domain_probs`: softmax over domains
  - `feasibility`: scalar in [0,1] — "is this representation reliable?"
  - `global_reliability`: scalar in [0,1]
- **Training**: supervised on (representation, correctness_label) pairs
- **Pretraining**: self-supervised on unlabeled data (reconstruction + contrastivity)

#### 3.2.4 Local Reliability Model (LRM) — Layer 3
- **Architecture**: 2-layer transformer over token-wise representations
- **Inputs**: per-token hidden states (sequence length × hidden_dim)
- **Outputs**:
  - `correctness_prob`: per-token P(correct|H_token) in [0,1]
  - `failure_risk`: per-token 1 - correctness_prob
  - Token-level reliability scores
- **Training**: trained on token-level correctness labels

#### 3.2.5 Router Network — Layer 4
- **Architecture**: Small MLP (2-layer, hidden=256, gelu) taking [GRM_output || LRM_output || representation_stats]
- **Outputs**:
  - `route_type`: {base, expert_0, expert_1, ..., expert_4}
  - `confidence`: scalar in [0,1] — router's confidence in its decision
  - `domain_prob`: over expert selection (if routing to expert)
- **Training**: reinforced policy gradient or supervised on oracle routing decisions

#### 3.2.6 Expert System — Layer 5
- **4-5 LoRA adapters** (E0-general, E1-math, E2-code, E3-science, E4-reasoning)
- **r=16, lora_alpha=32, lora_dropout=0.05**
- **Target modules**: q_proj, k_proj, v_proj, o_proj (qwen-specific)
- **Training**: each expert trained on domain-specific data with correctness labels
- **Specialization**: each expert should outperform base model on its designated domain

#### 3.2.7 Output Generation
- If route = base: `forward_base()` — Qwen with no experts active
- If route = expert_i: `forward_expert(expert_name)` — Qwen with expert_i adapter active
- Token-level: next token from selected path

---

## 4. Training Pipeline

### 4.1 Data Preparation

#### 4.1.1 Representation Collection Dataset
- **Domains**: general, math, code, science, reasoning
- **Sources**: wikitext, gsm8k, mbpp, ai2_arc, custom reasoning datasets
- **Labels**: correctness (exact match for math/code, reference-based for general/science)
- **Size**: ~10K samples per domain for training, ~2K for validation

#### 4.1.2 Expert Training Data
- **Per-expert**: filtered samples where that expert should shine
- **Math expert**: gsm8k, math word problems
- **Code expert**: mbpp, HumanEval
- **Science expert**: ai2_arc, domain-specific QA
- **General expert**: wikitext, trivia questions
- **Reasoning expert**: addsub, GSM-hard, other reasoning benchmarks

### 4.2 GRM Training

1. **Supervised phase**: 
   - Input: pooled representation + correctness label
   - Loss: binary cross entropy for feasibility + cross-entropy for domain classification
   
2. **Self-supervised phase** (optional):
   - Contrastive loss on positive/negative pairs
   - Reconstruction loss

### 4.3 LRM Training

1. Per-token correctness prediction
2. Binary classification: correct/incorrect given token hidden state
3. Loss: BCE with class weighting (handle imbalance)

### 4.4 Router Training

**Option A — Supervised (oracle):**
- Generate oracle decisions: route to expert if base would be wrong, else base
- Train MLP to mimic oracle decisions
- Loss: cross-entropy for route_type + MSE for confidence

**Option B — Reinforcement:**
- Policy gradient: reward = accuracy - λ × expert_utilization
- Baseline: moving average of rewards
- KL penalty: keep router close to uniform initially

### 4.5 Expert Training

Each expert trained independently:
1. LoRA adapter on frozen Qwen backbone
2. Data: domain-specific subset
3. Objective: causal LM loss with correctness-supervised weighting
4. Save adapter weights separately

### 4.6 Calibration

After all training:
1. Collect reliability scores R(x) and correctness labels on validation set
2. Fit temperature scaling: optimal T that minimizes NLL
3. Fit isotonic regression on (R(x), correctness) for non-parametric calibration
4. Report: ECE before/after calibration, Brier score, reliability diagrams

---

## 5. Evaluation Framework

### 5.1 Baseline Matrix

| Baseline | Description |
|----------|-------------|
| **B0** | Qwen alone — no routing, no experts |
| **B1** | Qwen + confidence threshold (entropy/probability only) |
| **B2** | Qwen + GRM only (reliability score, always base) |
| **B3** | Qwen + always-on experts (all 4 experts active) |
| **B4 (ARES)** | Full ARES: GRM + LRM + learned router + selective experts |

### 5.2 Metrics

#### 5.2.1 Predictive Performance
- **Accuracy**: percentage of correct predictions (per-domain, overall)
- **Per-domain accuracy**: math, code, science, general, reasoning
- **Average accuracy across domains**

#### 5.2.2 Reliability Calibration
- **ECE (Expected Calibration Error)**: primary calibration metric
- **Brier score**: proper scoring rule for probabilistic predictions
- **Reliability diagram**: plotted bins of confidence vs. observed accuracy
- **NLL (Negative Log-Likelihood)**: proper accuracy-oriented metric

#### 5.2.3 Selective Prediction Metrics
- **Selective accuracy**: accuracy only on inputs where system doesn't abstain
- **Coverage**: fraction of inputs where system produces output (vs. abstains)
- **Risk-coverage curve**: plots accuracy vs. coverage across thresholds
- **Area under risk-coverage curve (AURC)**: single-number summary

#### 5.2.4 Expert Utilization
- **Expert activation rate**: % of tokens/inputs where an expert was invoked
- **Per-expert activation**: % for each expert individually
- **Load balancing**: entropy of expert activation distribution
- **Compute overhead**: relative FLOPs or time vs. base model alone

#### 5.2.5 Routing Analysis
- **Routing accuracy**: % of times router made the "correct" decision (oracle comparison)
- **Abstain rate**: % of inputs where router chose "base" path
- **Router confidence analysis**: correlation between router confidence and actual correctness

### 5.3 Statistical Testing

- **Bootstrap resampling**: 1000 samples for confidence intervals on accuracy differences
- **Paired t-tests**: compare ARES vs. each baseline
- **Significance threshold**: p < 0.05 (two-tailed)
- **Report**: mean ± std across 3 random seeds

### 5.4 Human Evaluation (Qualitative)

- 50 random failure cases from each baseline
- Expert-written analysis of: why did base fail? which expert would help? did router make right call?
- Score: "router correct", "expert would fix", "both wrong", etc.

---

## 6. 4-Week Implementation Timeline

### Week 1: Infrastructure & Backbone ✅ COMPLETED (2026-08-18)
- [x] Set up repo with model-agnostic backbone abstraction
- [x] Qwen2.5-0.5B loading + verification
- [x] DDP setup for 2x T4 Kaggle
- [x] Config system (Hydra/OmegaConf)
- [x] W&B experiment tracking
- [x] Checkpoint system with SHA256 metadata
- **Milestone**: Can load Qwen, run forward, extract hidden states, save checkpoints with verification — **VERIFIED ON KAGGLE**

### Week 2: Representation & Reliability Probes ✅ COMPLETED (2026-08-24)
- [x] Multi-layer representation collector
- [x] GRM transformer architecture + training loop
- [x] LRM transformer architecture + training loop
- [x] Self-supervised pretraining on unlabeled data (contrastive + reconstruction)
- [x] Calibration module (temp scaling + isotonic)
- **Milestone**: GRM and LRM can be trained on representation data; calibration works; self-supervised pretraining functional - **VERIFIED ON KAGGLE**

### Week 3: Experts & Router ✅ COMPLETED (2026-08-26)
- [x] 5 LoRA experts (E0-general, E1-math, E2-code, E3-science, E4-reasoning) with r=16, alpha=32
- [x] Router network architecture (MLP 896→256→6) + Switch Transformer load-balancing loss
- [x] End-to-end ARES pipeline (`scripts/run_ares_pipeline.py`): backbone → representations → GRM/LRM → router → experts → generation
- [x] Adaptive computation policies & early-exit routing logic
- [x] 124 unit tests covering shapes, autograd flow, routing weights, and pipeline integration
- **Milestone**: Full ARES architecture runs end-to-end; verified dynamic domain evaluation and routing on Kaggle — **VERIFIED ON KAGGLE**

### Week 4: Real Data Harvesting & Empirical Paper Evaluation (In Progress)
- [x] Benchmark data loaders (`src/ares/data/benchmark_loader.py`) for GSM8K, MBPP, AI2-ARC, WikiText, and CommonsenseQA
- [x] Ground-truth answer parsing and correctness evaluators (`evaluate_prediction`)
- [x] Real multi-domain representation harvesting pipeline (`scripts/harvest_real_data.py`)
- [ ] Retrain GRM/LRM and LoRA experts on real harvested benchmark representations
- [ ] Train Router MLP on real oracle labels ($y_{\text{oracle}} = \text{Expert if Base wrong, else Base}$)
- [ ] Baseline implementations & comparison suite (B0–B4)
- [ ] Full paper evaluation metrics (Accuracy, ECE, Risk-Coverage AURC, 60–70% compute savings)
- [ ] Statistical significance testing & paper figures/tables generation
- **Milestone**: Complete empirical evaluation results across all 5 benchmark datasets; paper-ready artifacts generated

---

## 7. Technical Specifications

### 7.1 Hardware Requirements
- **Minimum**: Single GPU with 16GB+ VRAM (Qwen2.5-0.5B or 1.5B)
- **Recommended**: 2x T4 GPUs (Kaggle) for DDP training
- **Final experiments**: Qwen2.5-7B with bitsandbytes NF4 4-bit (single GPU with 16GB+ or 2x T4)

### 7.2 Software Stack
- **Python**: ≥3.10
- **PyTorch**: ≥2.1
- **Transformers**: ≥4.41, Qwen2.5 support
- **PEFT**: ≥0.12 for LoRA
- **bitsandbytes**: ≥0.45 for 4-bit quantization
- **Hydra/OmegaConf**: ≥1.3 for config
- **W&B**: ≥0.15 for experiment tracking
- **NumPy/SciPy**: standard
- **Matplotlib/Seaborn**: for visualization

### 7.3 Model Configuration

```yaml
# Example config structure
model:
  name: "Qwen/Qwen2.5-7B"
  revision: "main"
  torch_dtype: "bfloat16"  # or "float16"
  device_map: "auto"  # or custom dict for multi-GPU
  use_cache: false  # CRITICAL: disable for dynamic expert switching
  attn_implementation: "eager"  # required for output_attentions=True
  load_in_4bit: true  # final experiments only; bitsandbytes NF4

routing:
  num_experts: 5  # E0-general through E4-reasoning
  router_hidden_dim: 256
  router_num_layers: 2
  
experts:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

reliability:
  grm:
    hidden_dim: 512
    num_layers: 2
    num_heads: 4
  lrm:
    hidden_dim: 512
    num_layers: 2
    num_heads: 4

calibration:
  temperature_scaling: true
  isotonic_regression: true

evaluation:
  domains: ["general", "math", "code", "science", "reasoning"]
  seeds: [42, 123, 456]
```

### 7.4 Key Design Decisions

1. **Frozen backbone** — no weight updates to Qwen; all adaptation via LoRA/experts/router
2. **4-bit NF4 quantization** — via bitsandbytes; enables 7B on single GPU
3. **Eager attention** — required for `output_attentions=True` and attention analysis
4. **`use_cache=False`** — critical for correctness during dynamic expert switching
5. **DDP-ready** — all training scripts support multi-GPU via `torchrun` or Hugging Face `accelerate`
6. **Configuration-driven** — no hard-coded magic numbers; everything in YAML

### 7.5 Abbreviations
- **GRM**: Global Reliability Model
- **LRM**: Local Reliability Model
- **LoRA**: Low-Rank Adaptation
- **MoE**: Mixture of Experts (not full MoE in this version — discrete specialized experts)
- **ECE**: Expected Calibration Error
- **AURC**: Area Under Risk-Coverage curve
- **DDP**: Distributed Data Parallel

---

## 8. Expected Contributions

### 8.1 Primary Research Contributions

1. **Learned adaptive routing algorithm** — A neural network router that uses GRM+LRM signals to dynamically select between base model and specialized experts, outperforming simple threshold-based approaches.

2. **Domain-specialized expert system** — 4-5 LoRA experts each specialized for different domains (math, code, science, general, reasoning), with empirical evidence of specialization benefits.

3. **Calibrated reliability estimation** — GRM/LRM probes that produce well-calibrated reliability scores (low ECE), enabling trustworthy selective prediction — the system can correctly identify when it should not trust its own output.

4. **Selective computation framework** — Demonstrated tradeoff between accuracy and compute: ARES achieves better accuracy than the base model while invoking experts on only 60-70% of inputs (vs. 100% for always-on experts).

5. **Comprehensive evaluation protocol** — Standardized baseline matrix (B0-B4), metrics (accuracy, ECE, risk-coverage, selective prediction, expert utilization), and statistical testing framework for future research.

### 8.2 Technical Contributions

1. **Model-agnostic backbone abstraction** — Framework designed to work with any pretrained LLM (Qwen, Llama, Phi, Gemma) with minimal configuration changes.

2. **Multi-layer representation fusion** — Strategies for combining representations from multiple layers for reliability estimation.

3. **Self-supervised reliability probe pretraining** — Using unlabeled data to pretrain GRM/LRM before supervised fine-tuning.

4. **Efficient expert training pipeline** — LoRA experts trained on targeted failure modes without retraining the full backbone.

### 8.3 Practical Contributions

1. **Plug-and-play reliability layer** — Can be added to any frozen LLM without modifying the backbone.

2. **Deployment-friendly** — 4-bit quantization enables single-GPU inference of 7B models; selective compute reduces average latency.

3. **Trustworthy AI** — Calibrated reliability scores give users quantifiable confidence in when to trust or question the system's output.

---

## 9. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **4-bit quantization instability** | Medium | High | Test on multiple GPUs; fall back to float16; gradient checkpointing |
| **Router doesn't learn meaningful decisions** | Medium | Medium | Supervised oracle pretraining first; curriculum learning on routing; extensive ablation |
| **Experts specialize poorly** | Medium | Medium | Domain-specific data curation; diversity loss in expert training; verify each expert beats base on its domain |
| **Calibration fails (poor ECE)** | Low | Medium | Temperature scaling + isotonic regression pipeline; report both before/after |
| **DDP training instability** | Medium | Medium | Start single-GPU; gradual DDP warm-up; gradient clipping; verify identical results on 1 GPU vs 2 T4 |
| **Memory overflow with 7B + experts** | High (dev), Low (final) | High | 0.5B/1.5B for dev; 4-bit for final; gradient checkpointing; CPU offloading |
| **Evaluation overfits to specific benchmarks** | Low | Medium | Multiple domains; statistical testing across 3 seeds; report per-domain results |

---

## 10. Paper Structure (Outline)

1. **Introduction** — LLM reliability problem, ARES contribution
2. **Related Work** — Reliability, calibration, Mixture-of-Experts, adaptive computation
3. **Method** — Full architecture description (sections 3-5 of this PRD)
4. **Experimental Setup** — Hardware, data, training details (sections 7, 9)
5. **Results** — Baseline matrix B0-B4, ablations, risk-coverage curves, ECE, expert utilization
6. **Analysis** — Router behavior per domain, expert specialization quality, calibration analysis, failure cases
7. **Conclusion & Future Work** — Summary, limitations, directions (MoE, continuous experts, other backbones)

---

## 11. Success Criteria for Paper Acceptance

### Minimum (must achieve at least one):
- [ ] ARES > B0 (Qwen alone) on overall accuracy
- [ ] ARES achieves lower ECE than B0 (better calibrated)
- [ ] ARES activates experts on ≤70% of inputs while matching B3 (always-on) accuracy

### Strong (should achieve 2+):
- [ ] ARES > B0 on accuracy + lower ECE
- [ ] ARES < B3 accuracy but with ≤50% expert activation (compute savings)
- [ ] ARES > B3 accuracy at equivalent or lower expert utilization
- [ ] Risk-coverage curve of ARES dominates all baselines

### Exceptional (paper standout):
- [ ] ARES achieves both higher accuracy AND lower ECE than all baselines
- [ ] ARES with 4-bit 7B matches 16-bit 0.5B baseline accuracy
- [ ] Calibrated reliability successfully enables human-interpretable confidence assessment
- [ ] Ablation study reveals interesting insights (e.g., "LRM contributes X, GRM contributes Y")

---

## 12. Quick-Start for New Researchers

```bash
# 1. Install
pip install -e .

# 2. Quick verification
python scripts/generate_sample.py --prompt "Artificial intelligence is important because"

# 3. Phase 0: Backbone + verification
python -m pytest tests/test_phase0_utils.py -v

# 4. Phase 1: Representation collection (dev with 0.5B)
python scripts/collect_representations.py --config configs/reliability/representation_collection.yaml \
    --model_name Qwen/Qwen2.5-0.5B --max_samples 100 --analyze

# 5. Phase 2: Train GRM/LRM
python scripts/train_reliability_models.py --config configs/reliability/reliability_models.yaml \
    --input_dir representations/ --output_dir checkpoints/reliability --epochs 10

# 6. Phase 3: Train experts
python scripts/train_experts.py --config configs/experts/expert_mixture.yaml --epochs 3

# 7. Phase 4: Evaluate routing
python scripts/evaluate_phase5_routing.py --num_samples 500 --confidence_threshold 0.7

# 8. Launch visualizer
python scripts/run_visualizer.py --port 8501 --production
```