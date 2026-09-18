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
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ares.backbone.loader import Backbone, BackboneConfig, load_backbone
from ares.experts.lora_expert import LoRAExpert
from ares.experts.manager import ExpertManager, Router
from ares.grm.architecture import GRM
from ares.lrm.architecture import LRM


DEFAULT_EXPERT_NAMES = ["general", "math", "code", "science", "reasoning"]


class StopOnTokens(StoppingCriteria):
    """Dynamically halts generation when any designated stop token is reached."""

    def __init__(self, stop_token_ids: List[int]):
        super().__init__()
        self.stop_token_ids = set(stop_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in input_ids:
            if len(seq) > 0 and seq[-1].item() in self.stop_token_ids:
                return True
        return False


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
    
    max_new_tokens: int = 1024
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
        self.peft_model = None
        
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
            is_7b = any(tag in self.config.model_name.lower() for tag in ["7b", "8b", "4bit"])
            dev_str = str(self.device)
            use_4bit = is_7b and dev_str != "cpu"

            if use_4bit:
                d_map = {"": str(self.device)} if dev_str.startswith("cuda") else "auto"
            else:
                d_map = None if dev_str != "cpu" else "cpu"

            self.backbone = load_backbone(
                self.config.model_name,
                device=self.device,
                load_in_4bit=use_4bit,
                device_map=d_map,
            )

        # Dynamically detect hidden dimension from loaded backbone
        detected_dim = getattr(self.backbone, "hidden_size", None)
        if detected_dim is None and hasattr(self.backbone, "model") and hasattr(self.backbone.model, "config"):
            detected_dim = getattr(self.backbone.model.config, "hidden_size", None)
        if detected_dim is None and hasattr(self.backbone, "_model") and hasattr(self.backbone._model, "config"):
            detected_dim = getattr(self.backbone._model.config, "hidden_size", None)

        hidden_dim = self.config.hidden_dim or detected_dim or (3584 if "7b" in self.config.model_name.lower() else 896)
        self.config.hidden_dim = hidden_dim

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

        # 5. Load Native HuggingFace PEFT Multi-Adapters (only if true PEFT weight files exist)
        self.peft_model = None
        raw_model = getattr(self.backbone, "_model", getattr(self.backbone, "model", self.backbone))
        has_peft_weights = False
        for name in self.expert_names:
            exp_dir = ckpt_dir / "experts" / name
            if not exp_dir.exists():
                exp_dir = ckpt_dir / name
            if exp_dir.exists() and (
                (exp_dir / "adapter_model.safetensors").exists()
                or (exp_dir / "adapter_model.bin").exists()
            ):
                has_peft_weights = True
                break

        if has_peft_weights:
            try:
                from peft import PeftModel
                first_loaded = False
                for name in self.expert_names:
                    exp_dir = ckpt_dir / "experts" / name
                    if not exp_dir.exists():
                        exp_dir = ckpt_dir / name

                    if not (
                        exp_dir.exists()
                        and (
                            (exp_dir / "adapter_model.safetensors").exists()
                            or (exp_dir / "adapter_model.bin").exists()
                        )
                    ):
                        continue

                    try:
                        if not first_loaded:
                            self.peft_model = PeftModel.from_pretrained(
                                raw_model,
                                str(exp_dir.resolve()),
                                adapter_name=name,
                                local_files_only=True,
                            )
                            first_loaded = True
                        else:
                            self.peft_model.load_adapter(
                                str(exp_dir.resolve()),
                                adapter_name=name,
                                local_files_only=True,
                            )
                    except Exception as err:
                        print(f"[ARES Pipeline] Note: Could not attach PEFT adapter '{name}': {err}")

                if self.peft_model is not None:
                    print(f"[ARES Pipeline] PEFT multi-adapters active: {list(self.peft_model.peft_config.keys())}")
            except Exception as e:
                print(f"[ARES Pipeline] Note: PEFT multi-adapter initialization bypassed: {e}")
        else:
            print(f"[ARES Pipeline] Native ARES LoRAExperts active in ExpertManager (fused generation hook).")

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
        pooled_hidden: Optional[torch.Tensor] = None,
        reliability_info: Optional[Dict[str, Any]] = None,
        strategy: str = "dynamic",
        oracle_domain: Optional[str] = None,
    ) -> Tuple[int, str, Dict[str, float]]:
        """Determine routing destination based on representation and reliability."""
        strategy = strategy.lower()

        if strategy == "base":
            probs_dict = {name: (1.0 if i == 0 else 0.0) for i, name in enumerate(self.route_names)}
            return 0, "BASE", probs_dict

        elif strategy.startswith("fixed"):
            target = self.config.fixed_expert_name
            if "_" in strategy:
                target = strategy.split("_", 1)[1]
            if target in self.expert_names:
                idx = self.expert_names.index(target) + 1
                probs_dict = {name: (1.0 if name == target else 0.0) for name in self.route_names}
                return idx, target, probs_dict
            probs_dict = {name: (1.0 if i == 1 else 0.0) for i, name in enumerate(self.route_names)}
            return 1, self.expert_names[0], probs_dict

        if pooled_hidden is None:
            raise ValueError(f"pooled_hidden cannot be None for routing strategy '{strategy}'")

        pooled_f32 = pooled_hidden.to(dtype=torch.float32)

        # Get learned routing probabilities
        with torch.no_grad():
            routing_probs_tensor = self.expert_manager.router(pooled_f32)
            probs_dict = {
                name: routing_probs_tensor[0, i].item()
                for i, name in enumerate(self.route_names)
            }

        if strategy == "threshold":
            assert reliability_info is not None, "reliability_info required for threshold strategy"
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

    def _get_generation_target_module(self) -> Tuple[Optional[nn.Module], str]:
        """Find the optimal hook attachment point in the backbone.

        Returns:
            Tuple of (module, hook_type):
            - If final norm layer found: (norm_module, "pre_hook")
            - Fallback to top-level transformer layer: (layer_module, "forward_hook")
        """
        model = getattr(self.backbone, "_model", getattr(self.backbone, "model", self.backbone))
        # 1. Prefer pre-hook on final normalization layer (Qwen2, LLaMA, Mistral, Gemma, GPT-2)
        if hasattr(model, "model") and hasattr(model.model, "norm") and model.model.norm is not None:
            return model.model.norm, "pre_hook"
        elif hasattr(model, "norm") and model.norm is not None:
            return model.norm, "pre_hook"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f") and model.transformer.ln_f is not None:
            return model.transformer.ln_f, "pre_hook"

        # 2. Fallback to forward hook on last transformer layer
        if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
            return model.model.layers[-1], "forward_hook"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h") and len(model.transformer.h) > 0:
            return model.transformer.h[-1], "forward_hook"
        elif hasattr(model, "layers") and len(model.layers) > 0:
            return model.layers[-1], "forward_hook"
        return None, "none"

    def _get_generation_target_layer(self) -> Optional[nn.Module]:
        """Backward-compatible alias for target layer lookup."""
        mod, _ = self._get_generation_target_module()
        return mod

    def _make_expert_hook(self, expert: LoRAExpert):
        """Create forward hook that applies LoRA expert adaptation to hidden states."""
        target_device = self.device
        target_dtype = None
        raw_model = getattr(self.backbone, "_model", getattr(self.backbone, "model", self.backbone))
        for p in raw_model.parameters():
            target_device = p.device
            target_dtype = p.dtype
            break

        fast_expert = expert.to_fast(target_device, target_dtype) if hasattr(expert, "to_fast") else expert

        def hook_fn(module, input_args, output):
            if isinstance(output, tuple):
                h = output[0]
                adapted = fast_expert(h)
                return (adapted,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return fast_expert(output)
            return output
        return hook_fn

    def _attach_expert_for_generation(self, expert: LoRAExpert):
        """Zero-overhead context manager attaching expert adapter for generation."""
        from contextlib import contextmanager

        raw_model = getattr(self.backbone, "_model", getattr(self.backbone, "model", self.backbone))
        target_device = self.device
        target_dtype = None
        for p in raw_model.parameters():
            target_device = p.device
            target_dtype = p.dtype
            break

        fast_expert = expert.to_fast(target_device, target_dtype) if hasattr(expert, "to_fast") else expert
        target_module, hook_type = self._get_generation_target_module()

        if target_module is None:
            raise RuntimeError("[ARES Pipeline] Could not locate target transformer layer or norm for expert attachment.")

        @contextmanager
        def hook_scope():
            if hook_type == "pre_hook":
                def pre_fn(module, args):
                    return (fast_expert(args[0]),)
                handle = target_module.register_forward_pre_hook(pre_fn)
            else:
                def post_fn(module, input_args, output):
                    if isinstance(output, tuple):
                        return (fast_expert(output[0]),) + output[1:]
                    return fast_expert(output)
                handle = target_module.register_forward_hook(post_fn)
            try:
                yield
            finally:
                handle.remove()

        return hook_scope()

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

        # ─── 1. Tokenize Prompt (with Chat Template if Instruct model) ───────
        formatted_prompt = prompt
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and "<|im_start|>" not in prompt
            and (
                "Instruct" in getattr(self.config, "model_name", "")
                or getattr(self.tokenizer, "chat_template", None) is not None
            )
        ):
            try:
                clean_user_content = prompt.rstrip("\nAnswer:").rstrip("Answer:").strip()
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful and mathematically precise AI assistant. Answer the user's question directly, clearly, and step by step.",
                    },
                    {"role": "user", "content": clean_user_content},
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted_prompt = prompt

        raw_inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        target_device = self.device
        if hasattr(self.backbone, "get_device"):
            try:
                target_device = self.backbone.get_device()
            except Exception:
                target_device = self.device
        inputs = {
            k: v.to(target_device) if hasattr(v, "to") else v
            for k, v in raw_inputs.items()
        }

        # Check if routing strategy is predetermined (fixed or base)
        is_fixed = (strategy in ("base", "fixed_expert")) or (isinstance(strategy, str) and strategy.startswith("fixed_"))

        if is_fixed:
            # Fast path: O(1) route lookup, skipping redundant 7B forward pass + GRM + LRM + Router
            route_idx, selected_route, routing_probs = self.route(
                pooled_hidden=None,
                reliability_info=None,
                strategy=strategy,
                oracle_domain=oracle_domain,
            )
            reliability_info = {
                "domain_prediction": selected_route,
                "domain_confidence": 1.0,
                "global_reliability": 1.0,
                "feasibility": 1.0,
                "token_reliability": torch.empty(0),
                "failure_risk": 0.0,
                "uncertainty_score": 0.0,
            }
            timing["backbone_ms"] = 0.0
            timing["reliability_ms"] = 0.0
            timing["router_ms"] = 0.0
        else:
            # ─── 2. Backbone Forward Pass ────────────────────────────────────
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.backbone(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                pooled_hidden = last_hidden[:, -1, :]
            timing["backbone_ms"] = (time.perf_counter() - t0) * 1000.0

            # ─── 3. Reliability Analysis (GRM + LRM) ─────────────────────────
            t0 = time.perf_counter()
            reliability_info = self.evaluate_reliability(last_hidden, pooled_hidden)
            timing["reliability_ms"] = (time.perf_counter() - t0) * 1000.0

            # ─── 4. Routing Decision ─────────────────────────────────────────
            t0 = time.perf_counter()
            route_idx, selected_route, routing_probs = self.route(
                pooled_hidden=pooled_hidden,
                reliability_info=reliability_info,
                strategy=strategy,
                oracle_domain=oracle_domain,
            )
            timing["router_ms"] = (time.perf_counter() - t0) * 1000.0

        # ─── 5. Text Generation (Native PEFT Adapter vs Fused FastLoRA) ─────
        t0 = time.perf_counter()
        raw_model = getattr(self.backbone, "_model", getattr(self.backbone, "model", self.backbone))

        eos_ids = [self.tokenizer.eos_token_id]
        try:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id not in eos_ids and im_end_id > 0:
                eos_ids.append(im_end_id)
        except Exception:
            pass

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": eos_ids if len(eos_ids) > 1 else self.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "repetition_penalty": 1.2,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            if (
                self.peft_model is not None
                and route_idx > 0
                and selected_route in getattr(self.peft_model, "peft_config", {})
            ):
                self.peft_model.set_adapter(selected_route)
                gen_output = self.peft_model.generate(**inputs, **gen_kwargs)
            elif route_idx > 0:
                expert = None
                if (
                    self.expert_manager is not None
                    and hasattr(self.expert_manager, "experts")
                    and (route_idx - 1) < len(self.expert_manager.experts)
                ):
                    expert = self.expert_manager.experts[route_idx - 1]

                if expert is not None:
                    with self._attach_expert_for_generation(expert):
                        gen_output = raw_model.generate(**inputs, **gen_kwargs)
                else:
                    raise RuntimeError(
                        f"[ARES Pipeline FATAL] Expert route '{selected_route}' (idx {route_idx}) requested, "
                        f"but neither PEFT adapter nor target transformer layer/expert hook could be attached! "
                        f"peft_model={self.peft_model is not None}, expert={expert is not None}. Silent fallback to base model is disabled."
                    )
            elif self.peft_model is not None:
                # Base model route: disable adapter
                with self.peft_model.disable_adapter():
                    gen_output = self.peft_model.generate(**inputs, **gen_kwargs)
            else:
                gen_output = raw_model.generate(**inputs, **gen_kwargs)

        timing["generation_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0

        # ─── 6. Decode Generated Text ────────────────────────────────────────
        full_text = self.tokenizer.decode(gen_output[0], skip_special_tokens=True)
        # Extract only new tokens
        input_len = inputs["input_ids"].shape[1]
        new_token_ids = gen_output[0][input_len:]
        new_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        # Clean up any trailing chat markers or artificial stop sequences
        for stop_marker in ["<|im_end|>", "<|endoftext|>", "\n\nUser:", "\n\nQuestion:"]:
            if stop_marker in new_text:
                new_text = new_text.split(stop_marker)[0]
        new_text = new_text.strip()
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
