# ARES: Adaptive Reliability with Expert Specialization for Trustworthy Language Models

**Aliasghar Jawadwala**  
*School of Computing Sciences, Hindustan Institute of Technology and Science (HITS), Chennai, Tamil Nadu, India*  
`24cu0330065@student.hindustanuniv.ac.in`, `sharksurfauto@gmail.com`  
*Target: IEEE Transactions on Neural Networks and Learning Systems (TNNLS) / IEEE Transactions on Artificial Intelligence (TAI)*  
*September 2026*

---

### Abstract
Large Language Models (LLMs) frequently exhibit a profound failure mode known as the **Reliability Paradox**: models emit factually corrupt, syntactically defective, or mathematically erroneous completions accompanied by deceptively high token softmax probabilities and uncalibrated confidence. Existing mitigations present severe practical dilemmas: monolithic fine-tuning incites catastrophic forgetting; conventional Mixture-of-Experts (MoE) architectures execute uniform conditional computation across every forward token, incurring excessive floating-point operations (FLOPs) even for routine conversational queries; and heuristic post-hoc filters (e.g., token entropy) fail to foresee structural reasoning divergence. In this paper, we propose **ARES (Adaptive Reliability with Expert Specialization)**, an adaptive computational framework that couples internal representation reliability probing with domain-specialized Low-Rank Adaptation (LoRA) experts atop frozen autoregressive backbones. ARES monitors multi-layer intermediate representations ($\mathcal{L}_{\text{probe}} = \{-1, -6, -12, -24\}$) and routes them through dual lightweight probes: a **Global Reliability Model (GRM)** estimating domain classification and macro-task feasibility $R(x) \in [0, 1]$, and a **Local Reliability Model (LRM)** estimating token-level correctness probabilities and failure risks $f_{\text{risk}}(t)$. A learned, load-balanced **Router Policy Network** regularized via Switch Transformer auxiliary loss dynamically directs execution between an unmodified base pass-through path and five specialized LoRA adapters (covering *Mathematics, Code Synthesis, Scientific Reasoning, Commonsense Deduction, and General Knowledge*). Evaluated across five benchmark datasets (GSM8K, MBPP, AI2-ARC, CommonsenseQA, and WikiText-103) on NVIDIA T4 hardware, ARES achieves **61.20%** overall accuracy compared to **48.50%** for the frozen Qwen2.5 backbone—retaining **98.1%** of an unconstrained always-on MoE baseline (62.40%) while invoking specialized experts on only **41.6%** of queries, yielding a **58.4% reduction in added expert compute**. Furthermore, post-hoc isotonic calibration reduces Expected Calibration Error (ECE) from **0.1911** to **0.0480**, establishing well-calibrated confidence boundaries for trustworthy selective prediction.

**Index Terms**—Adaptive computation, calibration, expected calibration error (ECE), large language models (LLMs), low-rank adaptation (LoRA), mixture of experts (MoE), selective prediction, uncertainty estimation.

---

## I. INTRODUCTION

Autoregressive Large Language Models (LLMs) have demonstrated remarkable generative competence across natural language reasoning, multi-step problem solving, and software synthesis. However, deploying these models in mission-critical environments remains precarious due to what we formulate as the **Reliability Paradox**: model generation fluency and lexical predictability exhibit poor correlation with epistemic correctness. Modern transformers trained via maximum likelihood estimation readily generate mathematically invalid lemmas, unexecutable code, or fabricated scientific citations while assigning maximum softmax likelihood to each constituent sub-word token. Standard post-hoc confidence heuristics—such as sequence perplexity, token-level entropy, and logit margins—measure token familiarity rather than factual truthfulness.

Existing paradigms for adapting language models to complex specialized domains impose steep operational tradeoffs:
1. **Monolithic Parameter Fine-Tuning**: Updating all weights of an LLM for domain adaptation degrades general reasoning capabilities via catastrophic forgetting, alters foundational alignment, and requires prohibitive compute clusters for iterative retraining.
2. **Uniform Mixture-of-Experts (MoE)**: Top-$k$ conditional MoE models activate specialized feed-forward subnetworks uniformly across all tokens. While computationally more efficient than dense models of equal parameter scale, they nonetheless force expensive expert dispatch on trivial, conversational inputs that the underlying base model can solve effortlessly.
3. **Rejection Sampling & Ensembling**: Methods such as Best-of-$N$, Self-Consistency, or tree-search decoding multiply inference latency by $N \times$, rendering them economically unviable for real-time interactive systems.

To circumvent these dilemmas, this paper introduces **ARES (Adaptive Reliability with Expert Specialization)**. ARES operates on the hypothesis that the internal representation space of a frozen language model encodes early predictive indicators of task difficulty and operational failure before complete autoregressive emission occurs. By monitoring multi-layer intermediate activations, ARES determines *whether* specialized computation is required, and if so, *which* domain-specialized adapter should be engaged.

### A. Research Hypotheses
* **Hypothesis 1 (Routing Efficacy)**: A learned neural router trained on fused multi-layer representations and dual reliability probes achieves superior accuracy and lower selective risk than static confidence thresholding or random expert assignment at equivalent invocation budgets.
* **Hypothesis 2 (Domain Specialization)**: Parameter-efficient LoRA adapters trained on targeted domain failure modes measurably outperform the frozen backbone on their designated domains without degrading orthogonal capabilities.
* **Hypothesis 3 (Probe Calibration & Trustworthiness)**: Supervised multi-task reliability probes coupled with post-hoc isotonic regression yield well-calibrated confidence estimates ($\text{ECE} < 0.05$), enabling trustworthy selective prediction.
* **Hypothesis 4 (Pareto Compute Dominance)**: Selectively routing only low-reliability inputs to domain experts achieves a dominant Pareto frontier, retaining $\ge 95\%$ of unconstrained always-on MoE accuracy while eliminating $> 50\%$ of added expert computation.

---

## II. RELATED WORK & PRELIMINARIES

### A. Sparse Routing and Mixture-of-Experts
Conditional computation through Mixture-of-Experts (MoE) introduces sparsity into deep networks by gating inputs across sub-networks (Shazeer et al., 2017; Fedus et al., 2022). Switch Transformers simplified routing using top-1 gating with an auxiliary load-balancing loss. Parameter-efficient MoE frameworks, such as LoRAMoE (Dou et al., 2023), explore gating across multiple Low-Rank Adapters (Hu et al., 2022) to alleviate catastrophic forgetting. However, traditional MoE models enforce expert dispatch across every token unconditionally. ARES introduces a foundational architectural departure: the decision of *whether to invoke an expert at all* is treated as a primary learned policy, allowing an unmodified base model pass-through for the majority of queries.

### B. Internal Representation Probing
Mechanistic interpretability research indicates that transformer hidden activations contain linear and non-linear representations of factual truthfulness (Azaria & Mitchell, 2023; Burns et al., 2023). Azaria and Mitchell demonstrated that an MLP classifier trained on internal representations predicts truthfulness more accurately than surface-level output probabilities. Burns et al. identified unsupervised truth directions using contrast-consistent search. ARES operationalizes these insights into an active inference pipeline, employing dual probes (GRM + LRM) to extract macro-level domain affinity and micro-level token failure risks from multiple internal layers.

### C. Calibration and Selective Classification
Deep neural networks exhibit significant miscalibration, frequently assigning overconfident probabilities to incorrect classifications (Guo et al., 2017; Minderer et al., 2021). Temperature scaling and non-parametric isotonic regression restore probability calibration. In selective classification (Geifman & El-Yaniv, 2017), models are evaluated on their risk-coverage curve, quantified by the Area Under the Risk-Coverage curve (AURC). ARES embeds post-hoc isotonic calibration directly into its dual probes, establishing that a reliability threshold $\tau = 0.5$ rigorously maps to a 50% empirical error probability.

---

## III. SYSTEM ARCHITECTURE & MATHEMATICAL FORMULATIONS

### A. Layer 0: Frozen Backbone Ingestion
Let $M_\theta$ denote a pretrained autoregressive transformer parameterized by frozen weights $\theta$ ($\nabla_\theta \mathcal{L} = 0$). Given an input token sequence $X = (x_1, x_2, \dots, x_T) \in \mathcal{V}^T$, the backbone computes hidden states at each layer $l \in \{1, \dots, L\}$:
$$h_t^{(l)} = \text{TransformerLayer}^{(l)}(h_{1:t}^{(l-1)}; \theta) \in \mathbb{R}^{d_{\text{hidden}}}$$
To ensure dynamic adapter switching without state cross-contamination, generation is executed with $\text{use\_cache} = \text{False}$ and eager attention mechanisms. For 7B models, 4-bit NormalFloat (NF4) quantization compresses weights to 4.3 GB, fitting within a single 16 GB VRAM budget.

### B. Layer 1: Multi-Layer Representation Extraction
Rather than restricting analysis to final-layer embeddings $h_T^{(L)}$, ARES extracts representations across a layer subset $\mathcal{L}_{\text{probe}} = \{-1, -6, -12, -24\}$:
$$H_{\text{seq}} = \left[ h_t^{(l)} \right]_{l \in \mathcal{L}_{\text{probe}}, t \in \{1, \dots, T\}} \in \mathbb{R}^{T \times d_{\text{hidden}}}$$
A dense pooled prompt representation $h_{\text{pool}} \in \mathbb{R}^{d_{\text{hidden}}}$ is extracted via mean-pooling over prompt token boundaries:
$$h_{\text{pool}} = \frac{1}{T} \sum_{t=1}^T h_t^{(-1)}$$

### C. Layer 2: Global Reliability Model (GRM)
The GRM evaluates macroscopic domain affinity and prompt feasibility. It comprises a 2-layer Transformer encoder ($d_{\text{grm}} = 512, n_{\text{heads}} = 4$) with dual projection heads:
$$z_{\text{domain}} = W_d \cdot \text{TransformerEncoder}(h_{\text{pool}}) \in \mathbb{R}^{K}$$
$$R(x) = \sigma(W_r \cdot \text{TransformerEncoder}(h_{\text{pool}})) \in [0, 1]$$
$$\Phi(x) = \sigma(W_\phi \cdot \text{TransformerEncoder}(h_{\text{pool}})) \in [0, 1]$$
where $K=5$ denotes the target domains, $P(d|x) = \text{softmax}(z_{\text{domain}})$, $R(x)$ estimates base model success likelihood, and $\Phi(x)$ measures representation feasibility. The GRM objective is:
$$\mathcal{L}_{\text{GRM}} = \mathcal{L}_{\text{CE}}(z_{\text{domain}}, y_{\text{domain}}) + \lambda_1 \mathcal{L}_{\text{BCE}}(R(x), y_{\text{acc}}) + \lambda_2 \mathcal{L}_{\text{BCE}}(\Phi(x), y_{\text{feas}})$$
with $\lambda_1 = 1.0, \lambda_2 = 0.5$.

### D. Layer 3: Local Reliability Model (LRM)
To capture token-level reasoning errors before they cascade, the LRM processes $H_{\text{seq}}$ using a 2-layer sequence transformer:
$$P(\text{correct} | h_t) = \sigma(W_{\text{lrm}} \cdot \text{TransformerLayer}(h_t))$$
Per-token failure risk is defined as $f_{\text{risk}}(t) = 1.0 - P(\text{correct} | h_t)$. The aggregate sequence risk $\bar{f}_{\text{risk}} = \frac{1}{T} \sum_{t=1}^T f_{\text{risk}}(t)$ is combined with global reliability into the **Dual Uncertainty Metric**:
$$\mathcal{U}(x) = 1.0 - \left( R(x) \cdot (1.0 - \bar{f}_{\text{risk}}) \right) \in [0, 1]$$

### E. Layer 4: Learned Router Policy Network
The Router maps the concatenated feature vector:
$$v(x) = \left[ h_{\text{pool}} \,\|\, R(x) \,\|\, \bar{f}_{\text{risk}} \,\|\, P(d|x) \right] \in \mathbb{R}^{d_{\text{hidden}} + K + 2}$$
to a routing policy distribution $\pi_\phi(x)$ over $K+1$ routes:
$$\pi_\phi(x) = \text{softmax}\left( W_2 \cdot \text{GELU}(W_1 v(x) + b_1) + b_2 \right)$$
where $W_1 \in \mathbb{R}^{256 \times d_{\text{in}}}$ and $W_2 \in \mathbb{R}^{(K+1) \times 256}$.

**Load-Balancing Regularization**:
$$\mathcal{L}_{\text{balance}} = (K+1) \sum_{i=0}^K f_i \cdot P_i$$
$$\mathcal{L}_{\text{router}} = \mathcal{L}_{\text{CE}}(\pi_\phi(x), y_{\text{oracle}}) + \gamma \mathcal{L}_{\text{balance}}$$
with $\gamma = 0.01$.

### F. Layer 5: Domain-Specialized LoRA Experts
Five domain experts $E_1 \dots E_5$ are parameterized via LoRA applied to attention projections ($W_q, W_k, W_v, W_o$):
$$W_{\text{eff}} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B_k A_k$$
with rank $r=32$ and scaling $\alpha=64$.

### G. Repetition Penalty and Dynamic Stopping
Greedy decoding collapse is prevented via an unconditional repetition penalty $\rho = 1.2$:
$$\tilde{z}_i = \begin{cases} z_i / \rho & \text{if } z_i > 0 \\ z_i \cdot \rho & \text{if } z_i \le 0 \end{cases} \quad \forall i \in \mathcal{T}_{\text{generated}}$$
with dual stop tokens $\mathcal{T}_{\text{eos}} = \{151643, 151645\}$.

---

## IV. ARCHITECTURAL EXPLORATION & ABLATION EXPERIMENTS

### A. Representation Extraction Layer Depth
* **Shallow Layers ($\{-24\}$)**: 68.2% domain classification accuracy.
* **Deep Layer ($\{-1\}$)**: 81.4% domain accuracy, but poor sensitivity to multi-step reasoning failures.
* **Multi-Layer Fusion ($\{-1, -6, -12, -24\}$)**: Peak domain classification of **86.40%** and lowest calibration error.

### B. Probe Architecture Comparison
* **Linear Probe**: 72.40% domain accuracy, 0.4 ms latency, 4.5K parameters.
* **2-Layer MLP (ReLU)**: 79.80% domain accuracy, 1.2 ms latency, 235K parameters.
* **2-Layer Transformer (4 Heads)**: **86.40% domain accuracy**, 4.8 ms latency, 1.05M parameters.

### C. LoRA Rank & CoT Targets
Naive ground-truth supervision yielded 12% accuracy on GSM8K. Transitioning to step-by-step Chain-of-Thought (CoT) solutions lifted accuracy to 52.0%. LoRA rank ablation demonstrated that $r=32, \alpha=64$ provided optimal capacity without parameter bloat.

---

## V. EXPERIMENTAL RESULTS

### A. Multi-Domain Benchmark Evaluation (250 Test Queries on NVIDIA T4)

| Strategy | GSM8K (Math) | MBPP (Code) | AI2-ARC (Sci) | CSQA (Reas) | WikiText (Gen) | Overall Acc | Invocations | Compute Savings | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Base Qwen2.5-0.5B** | 32.0% | 36.0% | 58.0% | 52.0% | 64.0% | **48.50%** | 0.0% | 100.0% | 2622.7 ms |
| **B1: Entropy Threshold** | 36.0% | 40.0% | 62.0% | 56.0% | 66.0% | **52.10%** | 28.4% | 71.6% | 2621.4 ms |
| **B2: Base + GRM Only** | 38.0% | 42.0% | 64.0% | 54.0% | 66.0% | **52.80%** | 0.0% | 100.0% | 2623.1 ms |
| **Fixed Single Expert ($E_{\text{math}}$)** | 52.0% | 36.0% | 56.0% | 50.0% | 62.0% | **51.20%** | 100.0% | 0.0% | 2619.9 ms |
| **Random Router** | 42.0% | 44.0% | 64.0% | 58.0% | 66.0% | **54.80%** | 84.8% | 15.2% | 2619.4 ms |
| **B3: Always-On MoE** | 54.0% | 52.0% | 72.0% | 66.0% | 68.0% | **62.40%** | 100.0% | 0.0% | 2620.5 ms |
| **B4: ARES (Learned Router)** | **52.0%** | **50.0%** | **70.0%** | **66.0%** | **68.0%** | **61.20%** | **41.6%** | **58.4%** | **2620.9 ms** |
| **Oracle Router (Upper Bound)** | 56.0% | 54.0% | 74.0% | 68.0% | 70.0\% | **64.40%** | 100.0% | 0.0% | 2620.5 ms |

### B. Expected Calibration Error (ECE)
* **Base Softmax Confidence**: Pre-ECE 0.3240, Post-ECE 0.1680, Brier Score 0.2410
* **Raw GRM Probe Output**: Pre-ECE 0.1911, Post-ECE 0.0840, Brier Score 0.1820
* **ARES Dual Probes (Isotonic)**: Pre-ECE 0.1911, **Post-ECE 0.0480**, **Brier Score 0.1140**

### C. Selective Prediction (AURC)
ARES achieves an **AURC of 0.284** vs. Base Model 0.458 and Token Entropy 0.402, delivering a **12.7% absolute error reduction** at 80% answer coverage.

### D. Router Dispatch Specialization Matrix
* Math (GSM8K): 68.0% dispatched to $E_{\text{math}}$, 26.0% to Base.
* Code (MBPP): 60.0% dispatched to $E_{\text{code}}$, 34.0% to Base.
* General (WikiText): **82.0% dispatched to Base Model**, zero unnecessary expert overhead.

---

## VI. CONCLUSION

ARES successfully demonstrates that internal representation probing enables highly efficient, reliable, and trustworthy adaptive computation. By coupling macroscopic domain feasibility with token-level failure risk, ARES achieves **61.20% benchmark accuracy**—capturing **98.1% of always-on MoE performance** while reducing expert computation by **58.4%** and establishing calibrated confidence ($ECE = 0.0480$).
