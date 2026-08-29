"""End-to-End ARES Pipeline Engine (PRD §3, §7.4).

Coordinates:
Frozen Backbone → Representation Extraction → GRM & LRM Reliability Analysis →
Router Decision → Dynamic LoRA Expert Hook Generation → Output & Diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ares.backbone.loader import Backbone, BackboneConfig, load_backbone
from ares.experts.lora_expert import LoRAExpert
from ares.experts.manager import ExpertManager, Router
from ares.grm.architecture import GRM
from ares.lrm.architecture import LRM


DEFAULT_EXPERT_NAMES = ["general", "math", "code", "science", "reasoning"]


@dataclass
class PipelineConfig:
    """Configuration for end-to-end ARES pipeline execution."""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    checkpoints_dir: str = "checkpoints"
    grm_checkpoint: Optional[str] = None
    lrm_checkpoint: Optional[str] = None
    router_checkpoint: Optional[str] = None
    expert_checkpoints: Optional[Dict[str, str]] = None
    
    device: str = "auto"
    reliability_threshold: float = 0.5
    routing_strategy: str = "dynamic"  # "dynamic", "base", "fixed", "threshold", "oracle", "random"
    fixed_expert_name: str = "math"
    
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    
    expert_names: List[str] = field(default_factory=lambda: list(DEFAULT_EXPERT_NAMES))
    hidden_dim: Optional[int] = None


@dataclass
class PipelineResult:
    """Detailed result of an ARES pipeline execution."""

    prompt: str
    generated_text: str
    selected_route: str  # "BASE", "general", "math", "code", "science", "reasoning"
    route_idx: int  # 0 for BASE, 1..5 for experts
    routing_probs: Dict[str, float]
    domain_prediction: str
    domain_confidence: float
    global_reliability: float
    feasibility: float
    token_reliability: float
    failure_risk: float
    uncertainty_score: float
    latency_ms: Dict[str, float]
    tokens_generated: int
    route_confidence: float
    full_output_text: str = ""


class ARESPipeline:
    """End-to-End ARES Inference and Dynamic Routing Pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        backbone: Optional[Backbone] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        grm: Optional[GRM] = None,
        lrm: Optional[LRM] = None,
        expert_manager: Optional[ExpertManager] = None,
    ):
        self.config = config or PipelineConfig()
        
        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.expert_names = self.config.expert_names
        self.route_names = ["BASE"] + self.expert_names
        
        # Components
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.grm = grm
        self.lrm = lrm
        self.expert_manager = expert_manager
        
        # Load components if not supplied
        if self.backbone is None or self.tokenizer is None or self.grm is None or self.expert_manager is None:
            self._load_components()

    def _load_components(self):
        """Load or initialize all pipeline components and weights."""
        ckpt_dir = Path(self.config.checkpoints_dir)

        # 1. Backbone & Tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.backbone is None:
            backbone_cfg = BackboneConfig(
                name=self.config.model_name,
                device_map=str(self.device),
                use_cache=False,
                attn_implementation="eager",
            )
            self.backbone = load_backbone(backbone_cfg)

        hidden_dim = self.config.hidden_dim or getattr(self.backbone, "hidden_size", 896)

        # 2. GRM
        if self.grm is None:
            self.grm = GRM(
                input_dim=hidden_dim,
                hidden_dim=512,
                domain_classes=len(self.expert_names),
            ).to(self.device)
            self.grm.eval()

            grm_path = Path(self.config.grm_checkpoint) if self.config.grm_checkpoint else ckpt_dir / "reliability" / "grm.pt"
            if grm_path.exists():
                try:
                    ckpt = torch.load(grm_path, map_location=self.device, weights_only=False)
                    self.grm.load_state_dict(ckpt.get("model_state_dict", ckpt))
                except Exception as e:
                    print(f"Warning: Failed to load GRM checkpoint from {grm_path}: {e}")

        # 3. LRM
        if self.lrm is None:
            self.lrm = LRM(
                input_dim=hidden_dim,
                hidden_dim=512,
            ).to(self.device)
            self.lrm.eval()

            lrm_path = Path(self.config.lrm_checkpoint) if self.config.lrm_checkpoint else ckpt_dir / "reliability" / "lrm.pt"
            if lrm_path.exists():
                try:
                    ckpt = torch.load(lrm_path, map_location=self.device, weights_only=False)
                    self.lrm.load_state_dict(ckpt.get("model_state_dict", ckpt))
                except Exception as e:
                    print(f"Warning: Failed to load LRM checkpoint from {lrm_path}: {e}")

        # 4. ExpertManager & Router
        if self.expert_manager is None:
            self.expert_manager = ExpertManager(
                input_dim=hidden_dim,
                n_experts=len(self.expert_names),
                expert_names=self.expert_names,
            ).to(self.device)
            self.expert_manager.eval()

            router_path = Path(self.config.router_checkpoint) if self.config.router_checkpoint else ckpt_dir / "router" / "router.pt"
            if router_path.exists():
                try:
                    ckpt = torch.load(router_path, map_location=self.device, weights_only=False)
                    if "router_state_dict" in ckpt:
                        self.expert_manager.router.load_state_dict(ckpt["router_state_dict"])
                    elif "model_state_dict" in ckpt:
                        self.expert_manager.router.load_state_dict(ckpt["model_state_dict"])
                except Exception as e:
                    print(f"Warning: Failed to load Router checkpoint from {router_path}: {e}")

            # Load individual experts
            for i, name in enumerate(self.expert_names):
                exp_path = None
                if self.config.expert_checkpoints and name in self.config.expert_checkpoints:
                    exp_path = Path(self.config.expert_checkpoints[name])
                else:
                    candidates = [
                        ckpt_dir / "experts" / name / f"expert_{name}.pt",
                        ckpt_dir / name / f"expert_{name}.pt",
                        ckpt_dir / f"expert_{name}.pt",
                    ]
                    for cand in candidates:
                        if cand.exists():
                            exp_path = cand
                            break

                if exp_path and exp_path.exists():
                    try:
                        self.expert_manager.load_expert(i, exp_path, strict=False)
                    except Exception as e:
                        print(f"Warning: Failed to load Expert {name} checkpoint: {e}")

    def evaluate_reliability(
        self,
        hidden_states: torch.Tensor,
        pooled_hidden: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute dual reliability signals (GRM + LRM) and uncertainty."""
        with torch.no_grad():
            pooled_f32 = pooled_hidden.to(dtype=torch.float32)
            seq_f32 = hidden_states.to(dtype=torch.float32)

            # GRM Forward
            domain_logits, feasibility, global_rel = self.grm(pooled_f32)
            domain_probs = torch.softmax(domain_logits, dim=-1)
            pred_domain_idx = domain_probs.argmax(dim=-1).item()
            pred_domain = self.expert_names[pred_domain_idx]
            domain_conf = domain_probs[0, pred_domain_idx].item()
            grm_rel = global_rel.squeeze().item()
            feas = feasibility.squeeze().item()

            # LRM Forward
            correctness_prob, failure_risk = self.lrm(seq_f32)
            token_rel_mean = correctness_prob.mean().item()
            failure_risk_mean = failure_risk.mean().item()

            # Dual Uncertainty Estimate: High when global reliability is low or token risk is high
            uncertainty = 1.0 - (grm_rel * (1.0 - failure_risk_mean))
            uncertainty = max(0.0, min(1.0, uncertainty))

            return {
                "domain_logits": domain_logits,
                "domain_probs": domain_probs,
                "domain_prediction": pred_domain,
                "domain_idx": pred_domain_idx,
                "domain_confidence": domain_conf,
                "global_reliability": grm_rel,
                "feasibility": feas,
                "token_reliability": token_rel_mean,
                "failure_risk": failure_risk_mean,
                "uncertainty_score": uncertainty,
            }

    def route(
        self,
        pooled_hidden: torch.Tensor,
        reliability_info: Dict[str, Any],
        strategy: str = "dynamic",
        oracle_domain: Optional[str] = None,
    ) -> Tuple[int, str, Dict[str, float]]:
        """Determine routing destination based on representation and reliability."""
        strategy = strategy.lower()
        pooled_f32 = pooled_hidden.to(dtype=torch.float32)

        # Get learned routing probabilities
        with torch.no_grad():
            routing_probs_tensor = self.expert_manager.router(pooled_f32)
            probs_dict = {
                name: routing_probs_tensor[0, i].item()
                for i, name in enumerate(self.route_names)
            }

        if strategy == "base":
            return 0, "BASE", probs_dict

        elif strategy.startswith("fixed"):
            target = self.config.fixed_expert_name
            if "_" in strategy:
                target = strategy.split("_", 1)[1]
            if target in self.expert_names:
                idx = self.expert_names.index(target) + 1
                return idx, target, probs_dict
            return 1, self.expert_names[0], probs_dict

        elif strategy == "threshold":
            # If reliability is high, use Base; if low, route to GRM predicted domain expert
            if reliability_info["global_reliability"] >= self.config.reliability_threshold:
                return 0, "BASE", probs_dict
            else:
                domain_idx = reliability_info["domain_idx"]
                return domain_idx + 1, self.expert_names[domain_idx], probs_dict

        elif strategy == "oracle":
            if oracle_domain and oracle_domain in self.expert_names:
                idx = self.expert_names.index(oracle_domain) + 1
                return idx, oracle_domain, probs_dict
            elif oracle_domain == "base":
                return 0, "BASE", probs_dict
            # Fallback to dynamic if oracle domain unknown
            selected_idx = routing_probs_tensor.argmax(dim=-1).item()
            return selected_idx, self.route_names[selected_idx], probs_dict

        elif strategy == "random":
            import random
            rand_idx = random.randint(0, len(self.route_names) - 1)
            return rand_idx, self.route_names[rand_idx], probs_dict

        else:
            # Dynamic ARES routing (learned router decision)
            selected_idx = routing_probs_tensor.argmax(dim=-1).item()
            return selected_idx, self.route_names[selected_idx], probs_dict

    def _get_generation_target_layer(self) -> Optional[nn.Module]:
        """Find the top-level transformer layer of the backbone to attach LoRA hook."""
        model = getattr(self.backbone, "_model", self.backbone)
        # Check standard architectures (Qwen, LLaMA, Mistral, GPT-2)
        if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
            return model.model.layers[-1]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h") and len(model.transformer.h) > 0:
            return model.transformer.h[-1]
        return None

    def _make_expert_hook(self, expert: LoRAExpert):
        """Create forward hook that applies LoRA expert adaptation to hidden states."""
        def hook_fn(module, input_args, output):
            if isinstance(output, tuple):
                h = output[0]
                adapted = expert(h)
                return (adapted,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return expert(output)
            return output
        return hook_fn

    def generate(
        self,
        prompt: str,
        strategy: Optional[str] = None,
        oracle_domain: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> PipelineResult:
        """Execute full end-to-end ARES pipeline for a single prompt."""
        strategy = strategy or self.config.routing_strategy
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        timing: Dict[str, float] = {}
        total_start = time.perf_counter()

        # ─── 1. Tokenize Prompt ──────────────────────────────────────────────
        raw_inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in raw_inputs.items()
        }

        # ─── 2. Backbone Forward Pass ────────────────────────────────────────
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.backbone(**inputs, output_hidden_states=True)
            # Last layer hidden state
            last_hidden = outputs.hidden_states[-1]
            # Pooled representation (last token)
            pooled_hidden = last_hidden[:, -1, :]
        timing["backbone_ms"] = (time.perf_counter() - t0) * 1000.0

        # ─── 3. Reliability Analysis (GRM + LRM) ─────────────────────────────
        t0 = time.perf_counter()
        reliability_info = self.evaluate_reliability(last_hidden, pooled_hidden)
        timing["reliability_ms"] = (time.perf_counter() - t0) * 1000.0

        # ─── 4. Routing Decision ─────────────────────────────────────────────
        t0 = time.perf_counter()
        route_idx, selected_route, routing_probs = self.route(
            pooled_hidden=pooled_hidden,
            reliability_info=reliability_info,
            strategy=strategy,
            oracle_domain=oracle_domain,
        )
        timing["router_ms"] = (time.perf_counter() - t0) * 1000.0

        # ─── 5. Text Generation (Base vs LoRA Expert Hook) ───────────────────
        t0 = time.perf_counter()
        model = getattr(self.backbone, "_model", self.backbone)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        hook_handle = None
        if route_idx > 0:
            expert = self.expert_manager.experts[route_idx - 1]
            target_layer = self._get_generation_target_layer()
            if target_layer is not None:
                hook_handle = target_layer.register_forward_hook(self._make_expert_hook(expert))

        try:
            with torch.no_grad():
                gen_output = model.generate(**inputs, **gen_kwargs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        timing["generation_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0

        # ─── 6. Decode Generated Text ────────────────────────────────────────
        full_text = self.tokenizer.decode(gen_output[0], skip_special_tokens=True)
        # Extract only new tokens
        input_len = inputs["input_ids"].shape[1]
        new_token_ids = gen_output[0][input_len:]
        new_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        tokens_generated = len(new_token_ids)

        return PipelineResult(
            prompt=prompt,
            generated_text=new_text.strip(),
            full_output_text=full_text,
            selected_route=selected_route,
            route_idx=route_idx,
            routing_probs=routing_probs,
            domain_prediction=reliability_info["domain_prediction"],
            domain_confidence=reliability_info["domain_confidence"],
            global_reliability=reliability_info["global_reliability"],
            feasibility=reliability_info["feasibility"],
            token_reliability=reliability_info["token_reliability"],
            failure_risk=reliability_info["failure_risk"],
            uncertainty_score=reliability_info["uncertainty_score"],
            latency_ms=timing,
            tokens_generated=tokens_generated,
            route_confidence=routing_probs.get(selected_route, 0.0),
        )
