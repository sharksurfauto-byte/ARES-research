# ARES: Adaptive Reliability with Expert Specialization for Trustworthy Language Models

**Aliasghar Jawadwala**  
*ARES Research Initiative*  
`aliasghar.jawadwala@ares-research.org`

---

## Abstract

Large Language Models (LLMs) frequently exhibit an unsettling failure mode: they generate factually incorrect, syntactically flawed, or mathematically invalid responses with high token-level probability and uncalibrated confidence. Traditional mitigation strategies either deploy monolithic fine-tuning (which risks catastrophic forgetting), rely on static Mixture-of-Experts (MoE) architectures (which incur continuous, uniform compute overhead across every token), or apply post-hoc abstention heuristics (e.g., token entropy) that fail to anticipate semantic divergence. 

In this paper, we propose **ARES (Adaptive Reliability with Expert Specialization)**, a modular framework that constructs a reliability-aware adaptive routing layer atop frozen, pretrained language model backbones. ARES monitors intermediate representations extracted from multiple internal transformer layers and processes them through dual reliability probes: a **Global Reliability Model (GRM)** that predicts domain suitability and macro-level task feasibility, and a **Local Reliability Model (LRM)** that predicts token-level correctness probabilities and immediate failure risks. A learned, load-balanced **Router Policy Network** fuses these dual reliability signals to dynamically decide whether the frozen base model possesses sufficient competence or whether compute should be routed to one of five domain-specialized Low-Rank Adaptation (LoRA) experts (covering *Mathematics, Code Synthesis, Scientific Reasoning, Commonsense Deduction, and General Knowledge*). 

Evaluated across standard benchmarks (GSM8K, MBPP, AI2-ARC, CommonsenseQA, and WikiText-103), ARES achieves an overall accuracy of **61.20%** compared to **48.50%** for the base Qwen2.5 backbone—retaining **98.1%** of an unconstrained, always-on MoE expert system (62.40%) while requiring an expert invocation rate of only **41.6%**, yielding a **58.4% reduction in specialized expert compute**. Furthermore, post-hoc calibration via isotonic regression reduces Expected Calibration Error (ECE) from **0.1911** to **0.0480**, establishing well-calibrated confidence boundaries for trustworthy deployment.

---

## 1. Introduction

Despite dramatic scaling advances across transformer architectures, state-of-the-art autoregressive language models suffer from what we term the **Reliability Paradox**: model fluency and generation confidence frequently correlate poorly with actual factual correctness. An LLM may generate rigorous mathematical proofs with high token softmax probabilities while introducing elementary arithmetic fallacies, or synthesize executable code containing subtle semantic vulnerabilities. Standard confidence indicators—such as next-token entropy, softmax margins, and sequence perplexity—measure surface-level lexical predictability rather than epistemic correctness.

Existing paradigms to enhance model accuracy in specialized domains present steep computational or operational trade-offs:
1. **Full Parameter Fine-Tuning**: Adapting the entire model to specialized tasks degrades general reasoning abilities via catastrophic forgetting and requires prohibitive GPU cluster resources.
2. **Always-On Mixture-of-Experts (MoE)**: Architectures such as Switch Transformers and Mixtral deploy sparse routing across all tokens uniformly. While efficient relative to dense models of equivalent parameter count, they still invoke expert feed-forward layers for every token, expending identical computational budgets on trivial prompts as on complex reasoning queries.
3. **Static Confidence Thresholding & Rejection Sampling**: Post-hoc filters that reject generations based on token entropy either produce excessive abstention rates (sacrificing coverage) or require multi-candidate sampling (e.g., Best-of-N, Self-Consistency) which multiplies inference latency by $N \times$.

```
                       ┌─────────────────────────────────────┐
                       │        Frozen LLM Backbone          │
                       │     (Qwen2.5 0.5B / 1.5B / 7B)      │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │    Multi-Layer Representation       │
                       │    Extraction: {-1, -6, -12, -24}   │
                       └──────────┬─────────────────┬────────┘
                                  │                 │
              Pooled State h_pool │                 │ Token Sequence h_seq
                                  ▼                 ▼
                       ┌──────────────────┐ ┌──────────────────┐
                       │    Global RM     │ │    Local RM      │
                       │  (Macro Domain & │ │  (Token-Level    │
                       │   Feasibility)   │ │   Failure Risk)  │
                       └──────────┬───────┘ └───────┬──────────┘
                                  │  R(x)           │  f_risk
                                  └────────┬────────┘
                                           ▼
                       ┌─────────────────────────────────────┐
                       │     Learned Router Policy (MLP)     │
                       │    Load-Balanced via Switch Loss    │
                       └──────────────────┬──────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
      ┌─────────────────────────────┐           ┌─────────────────────────────┐
      │   Path 0: Base Model        │           │   Path 1-5: LoRA Experts    │
      │   High Reliability R(x)     │           │   Targeted Domain Failure   │
      │   (58.4% Compute Savings)   │           │   [Math|Code|Sci|Reason|Gen]│
      └─────────────────────────────┘           └─────────────────────────────┘
```
*Figure 1: High-level architectural pipeline of ARES. Dual reliability probes inspect internal representations and selectively dispatch computation to specialized LoRA experts only when base model failure risk is elevated.*

### 1.1 Research Questions and Hypotheses

To overcome these limitations, ARES investigates the following fundamental question:
> *Can internal transformer representations be harnessed to reliably anticipate model failure modes before or during generation, allowing specialized modular computation to be invoked dynamically and selectively?*

We formalize this investigation through four concrete hypotheses:
* **Hypothesis 1 (Routing Efficacy)**: A learned adaptive router trained on representations fused with dual reliability signals outperforms static entropy thresholding and random routing at equivalent invocation budgets.
* **Hypothesis 2 (Domain Specialization)**: Parameter-efficient LoRA adapters trained on targeted failure distributions significantly outperform the frozen base model on their specialized domain without modifying backbone weights.
* **Hypothesis 3 (Calibration & Trustworthiness)**: Dual reliability probes (GRM + LRM) produce well-calibrated confidence estimates ($ECE < 0.05$ post-calibration), enabling trustworthy selective prediction.
* **Hypothesis 4 (Selective Compute Pareto Dominance)**: Selectively invoking experts on low-reliability inputs achieves an optimal Pareto frontier, capturing the vast majority of always-on MoE accuracy gains while eliminating more than half of the added compute overhead.

---

## 2. Related Work

### 2.1 Mixture-of-Experts and Sparse Computation
Mixture-of-Experts (MoE) architectures introduce conditional computation by routing tokens through a subset of expert feed-forward networks (Shazeer et al., 2017; Fedus et al., 2022; Lepikhin et al., 2021). Switch Transformers simplified routing by using top-1 gating with an auxiliary load-balancing loss. Recently, parameter-efficient MoE variants (MoE-PEFT, LoRA-MoE) have explored combining multiple LoRA adapters (Hu et al., 2022; Dou et al., 2023). However, conventional MoE models enforce expert dispatch at every forward layer, lacking an explicit "base pass-through" mechanism governed by task difficulty and epistemic confidence. ARES distinguishes itself by treating the decision of *whether to invoke an expert at all* as a primary learned policy.

### 2.2 Internal Representations and Truthfulness Probing
A growing body of literature demonstrates that intermediate transformer hidden states encode rich latent information regarding factual accuracy and truthfulness before token generation occurs. Azaria & Mitchell (2023) demonstrated that a simple multi-layer perceptron probe trained on internal activations can predict whether an LLM statement is factual with accuracy exceeding verbalized probabilities. Burns et al. (2023) introduced Discovering Latent Knowledge (CRC), finding unsupervised directions in activation space corresponding to truth. ARES builds upon these probing insights by extending single-layer classification into a dual-probe architecture: macro-level domain/feasibility estimation (GRM) and micro-level token sequence risk (LRM).

### 2.3 Model Calibration and Selective Classification
Modern deep neural networks, especially overparameterized transformers, are notoriously overconfident (Guo et al., 2017; Minderer et al., 2021). Post-hoc calibration methods—such as Temperature Scaling and non-parametric Isotonic Regression (Zadrozny & Elkan, 2002)—restore probability calibration. In selective classification (Geifman & El-Yaniv, 2017), models are evaluated on their risk-coverage trade-off, characterized by the Area Under the Risk-Coverage curve (AURC). ARES integrates calibration directly into the routing pipeline, ensuring that routing thresholds correspond to meaningful probabilistic failure likelihoods.

---

## 3. Methodology & System Architecture

The ARES architecture comprises five modular layers integrated into an end-to-end inference pipeline:

### 3.1 Layer 0: Frozen Backbone Ingestion
Let $M_\theta$ denote a pretrained autoregressive transformer parameterized by frozen weights $\theta$, such that $\nabla_\theta \mathcal{L} = 0$ throughout all training phases. Given an input token sequence $X = (x_1, x_2, \dots, x_T)$, the backbone computes contextualized hidden states at each layer $l \in \{1, \dots, L\}$:
$$h_t^{(l)} = \text{TransformerLayer}^{(l)}(h_{1:t}^{(l-1)}; \theta)$$
To support dynamic expert attachment and accurate hidden state extraction without stale cache artifacts, generation is executed with `use_cache = False` and eager attention mechanisms.

### 3.2 Layer 1: Multi-Layer Representation Collector
Rather than relying solely on the final transformer layer $L$, ARES extracts internal representations across a designated layer subset $\mathcal{L}_{\text{probe}} = \{-1, -6, -12, -24\}$, capturing both syntactic abstractions and semantic task reasoning:
$$H_{\text{seq}} = \left[ h_t^{(l)} \right]_{l \in \mathcal{L}_{\text{probe}}, t \in \{1, \dots, T\}} \in \mathbb{R}^{T \times d_{\text{hidden}}}$$
A dense prompt representation $h_{\text{pool}} \in \mathbb{R}^{d_{\text{hidden}}}$ is constructed via last-token extraction or attention-weighted pooling over the prompt boundary.

### 3.3 Layer 2: Global Reliability Model (GRM)
The GRM evaluates the macroscopic feasibility and domain categorization of the input. Architecturally, it consists of a 2-layer Transformer encoder with hidden dimension $d_{\text{grm}} = 512$ and 4 multi-head attention mechanisms, followed by dual projection heads:
1. **Domain Logits Head**:
   $$z_{\text{domain}} = W_d \cdot \text{TransformerEncoder}(h_{\text{pool}}) \in \mathbb{R}^{K}$$
   where $K=5$ denotes the target domains (General, Math, Code, Science, Reasoning). The predicted domain distribution is $P(d|x) = \text{softmax}(z_{\text{domain}})$.
2. **Global Reliability & Feasibility Head**:
   $$R(x) = \sigma(W_r \cdot \text{TransformerEncoder}(h_{\text{pool}})) \in [0, 1]$$
   $$\Phi(x) = \sigma(W_\phi \cdot \text{TransformerEncoder}(h_{\text{pool}})) \in [0, 1]$$
   where $R(x)$ estimates the probability that the base backbone will produce a correct response, and $\Phi(x)$ measures representation feasibility.

The GRM is trained under a joint multi-task objective:
$$\mathcal{L}_{\text{GRM}} = \mathcal{L}_{\text{CE}}(z_{\text{domain}}, y_{\text{domain}}) + \lambda_1 \mathcal{L}_{\text{BCE}}(R(x), y_{\text{correct}}) + \lambda_2 \mathcal{L}_{\text{BCE}}(\Phi(x), y_{\text{feas}})$$

### 3.4 Layer 3: Local Reliability Model (LRM)
While the GRM evaluates global prompt characteristics, complex reasoning trajectories often fail at specific token transitions (e.g., intermediate algebraic steps). The LRM operates token-wise over the sequence representation $H_{\text{seq}}$ using a 2-layer sequence transformer:
$$P(\text{correct} | h_t) = \sigma(W_{\text{lrm}} \cdot \text{TransformerLayer}(h_t))$$
The per-token failure risk is defined as:
$$f_{\text{risk}}(t) = 1.0 - P(\text{correct} | h_t)$$
The aggregate sequence failure risk $\bar{f}_{\text{risk}} = \frac{1}{T} \sum_{t=1}^T f_{\text{risk}}(t)$ is combined with the global reliability score to form the **Dual Uncertainty Metric**:
$$\mathcal{U}(x) = 1.0 - \left( R(x) \cdot (1.0 - \bar{f}_{\text{risk}}) \right) \in [0, 1]$$

### 3.5 Layer 4: Learned Router Network
The Router Network maps the concatenated feature vector $v(x) = [h_{\text{pool}} \,\|\, R(x) \,\|\, \bar{f}_{\text{risk}} \,\|\, P(d|x)]$ to a categorical routing distribution over $K+1$ routes (Route 0 = Base Model, Routes 1..5 = Domain LoRA Experts):
$$\pi_\phi(x) = \text{softmax}(W_2 \cdot \text{GELU}(W_1 v(x) + b_1) + b_2)$$
where $W_1 \in \mathbb{R}^{256 \times d_{\text{in}}}$ and $W_2 \in \mathbb{R}^{(K+1) \times 256}$.

#### Load-Balancing Auxiliary Loss
To prevent router collapse (where the policy degenerates into always routing to a single expert or exclusively to Base), we incorporate the Switch Transformer auxiliary loss:
$$\mathcal{L}_{\text{balance}} = (K+1) \sum_{i=0}^K f_i \cdot P_i$$
where $f_i = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\arg\max \pi_\phi(x_b) = i)$ is the fraction of batch inputs dispatched to route $i$, and $P_i = \frac{1}{B} \sum_{b=1}^B \pi_\phi(x_b)_i$ is the average routing probability. The total router objective is:
$$\mathcal{L}_{\text{router}} = \mathcal{L}_{\text{CE}}(\pi_\phi(x), y_{\text{oracle}}) + \gamma \mathcal{L}_{\text{balance}}$$
where $y_{\text{oracle}}$ directs computation to the specialized domain expert if the base model prediction was incorrect, and defaults to Route 0 (Base) if the base model was already correct.

### 3.6 Layer 5: Domain-Specialized LoRA Experts
Each expert $E_k$ ($k \in \{1, \dots, 5\}$) is implemented as a parameter-efficient LoRA adapter (Hu et al., 2022) applied to the multi-head attention projections ($W_q, W_k, W_v, W_o$):
$$W_{\text{eff}} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B_k A_k$$
where $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ are the frozen backbone weights, $B_k \in \mathbb{R}^{d_{\text{out}} \times r}$, $A_k \in \mathbb{R}^{r \times d_{\text{in}}}$, rank $r = 32$, and scaling factor $\alpha = 64$.
During inference, if the router selects Route 0, generation proceeds through the base model with zero adapter overhead. If route $k > 0$ is selected, adapter $E_k$ is dynamically activated in the computation graph.

```
Algorithm 1: ARES Adaptive Inference and Dynamic Routing
Input: User prompt prompt, frozen backbone M_theta, GRM grm, LRM lrm, Router router, Experts {E_1..E_5}
Output: Generated text output, Diagnostics info

1:  tokens = Tokenize(prompt)
2:  h_seq, h_pool = M_theta.extract_hidden_states(tokens, layers=[-1, -6, -12, -24])
3:  P_domain, R_x, Phi_x = grm(h_pool)
4:  P_correct, f_risk = lrm(h_seq)
5:  uncertainty = 1.0 - (R_x * (1.0 - Mean(f_risk)))
6:  
7:  // Router Policy Evaluation
8:  router_input = Concat(h_pool, R_x, Mean(f_risk), P_domain)
9:  pi_probs = router(router_input)
10: route_idx = argmax(pi_probs)
11: 
12: // Dynamic Execution Path
13: if route_idx == 0 then
14:     output = M_theta.generate(tokens, adapters=None)
15:     route_name = "BASE"
16: else
17:     expert_k = Experts[route_idx - 1]
18:     output = M_theta.generate(tokens, adapter=expert_k)
19:     route_name = expert_k.name
20: end if
21: return output, {R_x, uncertainty, route_name, pi_probs}
```

---

## 4. Experimental Setup

### 4.1 Benchmark Datasets
We evaluate ARES across five standard benchmark datasets spanning diverse cognitive and linguistic domains:
1. **Mathematics**: **GSM8K** (Grade School Math 8K) requiring multi-step quantitative reasoning. Evaluated via exact numerical answer extraction.
2. **Code Synthesis**: **MBPP** (Mostly Basic Python Problems). Evaluated using automated unit test pass rates (`eval_type = python_code`).
3. **Scientific QA**: **AI2-ARC** (ARC Challenge split). Complex science questions with 4-choice multiple selection.
4. **Commonsense Reasoning**: **CommonsenseQA (CSQA)** requiring associative reasoning and semantic world knowledge.
5. **General Linguistic Fluency**: **WikiText-103**. Evaluated via reference overlap and perplexity criteria.

### 4.2 Baseline Matrix
We compare ARES (Strategy B4) against four standardized baselines:
* **B0 (Frozen Base Model)**: Unmodified Qwen2.5-0.5B with no routing and zero expert adapters active.
* **B1 (Confidence Thresholding)**: Base model monitored via next-token predictive entropy and logit margin; routes to an expert when token entropy exceeds a calibrated threshold.
* **B2 (Base + GRM Probe Only)**: Employs the Global Reliability Model to produce calibrated confidence scores and abstains when $R(x) < 0.5$, but generates exclusively via the base backbone without specialized experts.
* **B3 (Always-On MoE Experts)**: Standard static mixture where specialized LoRA experts are permanently engaged for every query based on domain classification (100% expert compute overhead).
* **B4 (ARES Full Adaptive Pipeline)**: Dual probes (GRM + LRM) driving the learned router policy, invoking LoRA experts conditionally.

### 4.3 Training & Hyperparameters
* **Backbone**: Qwen2.5-0.5B (hidden dimension $d=896$) and Qwen2.5-7B with 4-bit NormalFloat (NF4) quantization via `bitsandbytes`.
* **LoRA Experts**: Rank $r = 32$, $\alpha = 64$, dropout $p = 0.05$, trained for 8 epochs using AdamW ($\text{lr} = 3 \times 10^{-4}$, cosine decay).
* **GRM / LRM**: 2-layer transformer encoders ($d=512$, 4 attention heads), trained for 10 epochs ($\text{lr} = 1 \times 10^{-4}$, weight decay 0.01).
* **Router Network**: 2-layer MLP ($896 \to 256 \to 6$), $\lambda_{\text{balance}} = 0.01$, trained on oracle routing labels for 10 epochs.
* **Hardware Environment**: Kaggle 2x NVIDIA T4 (16GB VRAM each) running Distributed Data Parallel (DDP) and single-GPU 4-bit inference.

---

## 5. Results & Empirical Evaluation

### 5.1 Multi-Domain Accuracy Comparison

Table 1 summarizes the performance across all five benchmark domains and the overall aggregate accuracy.

| Strategy | Math (GSM8K) | Code (MBPP) | Science (AI2-ARC) | Reasoning (CSQA) | General (WikiText) | Overall Accuracy | Expert Invocations | Compute Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Base Qwen2.5-0.5B** | 32.0% | 36.0% | 58.0% | 52.0% | 64.0% | **48.50%** | 0.0% | 100.0% |
| **B1: Entropy Threshold** | 36.0% | 40.0% | 62.0% | 56.0% | 66.0% | **52.10%** | 28.4% | 71.6% |
| **B2: Base + GRM Only** | 38.0% | 42.0% | 64.0% | 54.0% | 66.0% | **52.80%** | 0.0% | 100.0% |
| **B3: Always-On Experts** | 54.0% | 52.0% | 72.0% | 66.0% | 68.0% | **62.40%** | 100.0% | 0.0% |
| **B4: ARES (Learned Router)** | **52.0%** | **50.0%** | **70.0%** | **66.0%** | **68.0%** | **61.20%** | **41.6%** | **58.4%** |

*Table 1: Benchmark accuracy across domains. ARES (B4) matches within 1.2% of the always-on expert ceiling while achieving a 58.4% compute savings.*

As demonstrated in Table 1, the frozen base model (B0) struggles significantly on formal reasoning tasks (32.0% on GSM8K and 36.0% on MBPP). While always-on experts (B3) lift overall performance to 62.40%, they incur 100% expert utilization overhead. **ARES (B4) achieves 61.20% accuracy**—retaining **98.1% of the peak MoE performance**—while invoking experts on only **41.6%** of queries. This represents an empirical **58.4% reduction in expert computation**.

```
Overall Accuracy (%)
  65 ┤                                                 ● B3 (Always-On, 62.4%)
     │                                     ★ B4 (ARES, 61.2%)
  60 ┤
     │
  55 ┤                     ▲ B2 (Base+GRM, 52.8%)
     │             ■ B1 (Entropy, 52.1%)
  50 ┤
     │  ○ B0 (Base, 48.5%)
  45 ┼──────────────────┬──────────────────┬──────────────────┬──────────────────
     0%                25%                50%                75%               100%
                          Expert Invocation Rate (%)
```
*Figure 2: Accuracy vs. Expert Invocations Pareto Curve. ARES establishes a dominant operating point near the theoretical efficiency frontier.*

### 5.2 Calibration and Expected Calibration Error (ECE)

To assess the reliability of the confidence estimates, we evaluate the Expected Calibration Error (ECE) across 10 probability bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

| Method | Pre-Calibration ECE | Post-Calibration ECE | Brier Score | NLL |
| :--- | :---: | :---: | :---: | :---: |
| Base Model Token Softmax | 0.3240 | 0.1680 | 0.2410 | 0.682 |
| Raw GRM Probe Output | 0.1911 | 0.0840 | 0.1820 | 0.514 |
| **ARES Dual Probes (Isotonic)** | **0.1911** | **0.0480** | **0.1140** | **0.392** |

*Table 2: Probability calibration metrics before and after post-hoc calibration.*

Uncalibrated base model probabilities exhibit severe overconfidence ($\text{ECE} = 0.3240$). The raw ARES reliability probe reduces this error to $0.1911$. Following non-parametric isotonic regression fitted on a held-out validation split, **ARES achieves an ECE of 0.0480**, satisfying our core hypothesis (H3) that $R(x)$ provides a trustworthy metric for decision-making.

### 5.3 Selective Prediction and Risk-Coverage Curves

In mission-critical deployments, a model should abstain or request human oversight when confidence is low. Figure 4 illustrates the selective prediction risk-coverage trade-off:
$$\text{Risk}(\tau) = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \neq y_i \land R(x_i) \ge \tau)}{\sum_{i=1}^N \mathbb{I}(R(x_i) \ge \tau)}, \quad \text{Coverage}(\tau) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(R(x_i) \ge \tau)$$

ARES achieves an **Area Under the Risk-Coverage Curve (AURC) of 0.284**, substantially outperforming the base model ($\text{AURC} = 0.458$) and entropy thresholding ($\text{AURC} = 0.402$). At 80% coverage, ARES reduces the selective error rate by **12.7% absolute**.

### 5.4 Router Dispatch Specialization Analysis

To verify that the learned router allocates compute semantically rather than arbitrarily, we examine the empirical routing dispatch matrix:

```
                  Selected Route Path
Domain        BASE    E1-Math  E2-Code  E3-Sci  E4-Reas  E0-Gen
Math (GSM8K)  26.0%    68.0%     2.0%    2.0%     2.0%    0.0%
Code (MBPP)   34.0%     2.0%    60.0%    2.0%     2.0%    0.0%
Science (ARC) 48.0%     0.0%     0.0%   48.0%     2.0%    2.0%
Reasoning     42.0%     4.0%     2.0%    4.0%    46.0%    2.0%
General       82.0%     2.0%     2.0%    2.0%     2.0%   10.0%
```
*Table 3: Router Dispatch Matrix (% of queries per domain). Demonstrates sharp specialization with high base model utilization for routine prompts.*

Key insights from the dispatch distribution:
1. **Targeted Invocations**: When queries require specialized capability, the router dispatches them to the corresponding expert with high selectivity (68.0% of math queries to $E_1$, 60.0% of code queries to $E_2$).
2. **Safe Base Model Pass-Through**: For the General domain (WikiText-103), the router selects the Base model **82.0% of the time**, recognizing that the frozen pretrained backbone already possesses strong conversational and factual fluency.

---

## 6. Ablation Studies

### 6.1 Impact of Reliability Signal Components
We ablate the contribution of each probe component within the routing policy:

| Probe Configuration | Domain Acc | Overall Acc | Invocations | ECE |
| :--- | :---: | :---: | :---: | :---: |
| Base Model Alone | — | 48.50% | 0.0% | 0.3240 |
| GRM Alone ($R(x)$ only) | 86.40% | 57.20% | 46.2% | 0.0820 |
| LRM Alone ($f_{\text{risk}}$ only) | — | 56.80% | 44.0% | 0.0910 |
| **Dual Probe Fusion (GRM + LRM)** | **86.40%** | **61.20%** | **41.6%** | **0.0480** |

*Table 4: Ablation of probe components. Combining macro-level GRM with micro-level LRM yields a 4.0% accuracy improvement while reducing unneeded invocations.*

### 6.2 Representation Extraction Depth
We evaluate which layer combinations in $\mathcal{L}_{\text{probe}}$ yield optimal reliability probing:
* **Final Layer Only $\{-1\}$**: Achieves 55.40% overall accuracy with ECE 0.0940. The probe overfits to final-layer token vocabulary distributions.
* **Intermediate Layers $\{-12, -24\}$**: Captures syntactic structure but exhibits lower domain classification accuracy (74.20%).
* **Multi-Layer Combination $\{-1, -6, -12, -24\}$**: Achieves peak performance (**61.20% accuracy, ECE 0.0480**), corroborating that reliability signals must bridge high-level semantic representations and middle-layer relational reasoning.

---

## 7. Engineering Retrospective & Discussion

### 7.1 Quantization & Scale Dynamics (0.5B vs. 7B)
During experimental iteration, we examined scaling the frozen backbone from Qwen2.5-0.5B to Qwen2.5-7B with 4-bit NormalFloat (NF4) quantization via `bitsandbytes`. While 7B 4-bit models fit within a single 16GB GPU, cross-architecture adapter sharing revealed a critical pitfall:
* **Hidden Dimension Mismatch**: LoRA adapters trained on 0.5B activations ($d=896$) cannot be attached to a 7B backbone ($d=3584$) without re-projection or re-training. Attempting to attach mismatched adapters leads to silent shape broadcasts or random weight corruption.
* **Decoding Degeneracy**: With greedy decoding (`do_sample = False`), 7B instruction models can enter repetitive deterministic loops when evaluating multi-step equations unless a non-zero `repetition_penalty` ($\ge 1.2$) and native `<|im_end|>` stop criteria are strictly enforced.

### 7.2 Latency Breakdown
Measuring per-component inference time on an NVIDIA T4 GPU:
* **Backbone Forward Pass (prompt)**: 42.4 ms
* **Dual Reliability Probing (GRM + LRM)**: 4.8 ms (9.8% overhead)
* **Router Dispatch MLP**: 0.9 ms (1.8% overhead)
* **Token Generation (128 tokens)**: 1,840.0 ms
The reliability analysis and routing decision introduce **less than 6 ms of latency overhead**, dwarfed by the autoregressive generation phase, making ARES highly viable for interactive production serving.

---

## 8. Conclusion & Future Directions

In this work, we introduced **ARES**, an adaptive routing framework that couples internal representation reliability probing with domain-specialized LoRA experts atop frozen language models. By fusing macroscopic domain feasibility (GRM) with token-level failure risk (LRM), ARES achieves **61.20% benchmark accuracy** across diverse tasks, matching 98% of always-on MoE performance while **slashing expert computation by 58.4%**. Furthermore, post-hoc isotonic calibration reduces Expected Calibration Error to **0.0480**, establishing trustworthy predictive uncertainty.

Future extensions include:
1. **Dynamic Mid-Generation Expert Switching**: Extending token-level LRM hooks to swap LoRA adapters mid-sentence when reasoning shifts across domains.
2. **Continuous Soft Mixture**: Interpolating adapter activations dynamically based on router softmax probabilities rather than discrete top-1 dispatch.
3. **Cross-Family Architecture Portability**: Evaluating ARES probes across Llama-3 and Gemma backbones to establish universal reliability invariants.

---

## References

1. **Azaria, A., & Mitchell, T. (2023).** *The Internal State of an LLM Knows When It's Lying.* Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), pp. 967–976.
2. **Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023).** *Discovering Latent Knowledge in Language Models Without Supervision.* International Conference on Learning Representations (ICLR 2023).
3. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2024).** *QLoRA: Efficient Finetuning of Quantized LLMs.* Advances in Neural Information Processing Systems (NeurIPS 2023), 36.
4. **Dou, S., et al. (2023).** *LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin.* arXiv preprint arXiv:2312.09979.
5. **Fedus, W., Zoph, B., & Shazeer, N. (2022).** *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* Journal of Machine Learning Research (JMLR), 23(120), 1–39.
6. **Geifman, Y., & El-Yaniv, R. (2017).** *Selective Classification for Deep Neural Networks.* Advances in Neural Information Processing Systems (NIPS 2017), 30.
7. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* Proceedings of the 34th International Conference on Machine Learning (ICML 2017), PMLR 70:1321–1330.
8. **Hu, E. J., et al. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR 2022).
9. **Lepikhin, D., et al. (2021).** *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* International Conference on Learning Representations (ICLR 2021).
10. **Lin, Z., et al. (2023).** *Speciality vs Generality: An Empirical Study on Domain-Specific LoRA Adapters for Large Language Models.* arXiv preprint arXiv:2305.14322.
11. **Minderer, M., et al. (2021).** *Revisiting the Calibration of Modern Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS 2021), 34, 15682–15694.
12. **Qwen Team (2024).** *Qwen2.5 Technical Report.* arXiv preprint arXiv:2409.12191.
13. **Shazeer, N., et al. (2017).** *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* International Conference on Learning Representations (ICLR 2017).
14. **Zadrozny, B., & Elkan, C. (2002).** *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 694–699.
