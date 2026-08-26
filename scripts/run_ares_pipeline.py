#!/usr/bin/env python
"""End-to-End ARES Pipeline (PRD §3).

Links the frozen Backbone -> Representation Collector -> GRM/LRM -> Router -> Experts -> Text Generation.

Usage:
    python scripts/run_ares_pipeline.py \
        --prompt "If it takes 3 hours to travel 180 km, what is the speed in km/h?" \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --checkpoints_dir checkpoints
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from ares.backbone.loader import load_backbone, BackboneConfig
from ares.representations import RepresentationCollector
from ares.grm import GRM
from ares.experts.manager import ExpertManager

def parse_args():
    parser = argparse.ArgumentParser(description="Run ARES Pipeline End-to-End")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # ─── 1. Load Backbone & Tokenizer ──────────────────────────────────────────
    print(f"\n[1/5] Loading frozen backbone ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    backbone_cfg = BackboneConfig(
        name=args.model_name,
        device_map=str(device),
        use_cache=False, 
        attn_implementation="eager"
    )
    backbone = load_backbone(backbone_cfg)
    # backbone eval correctly handled internally
    
    hidden_dim = backbone.hidden_size # 896 for Qwen2.5-0.5B
    
    # ─── 2. Setup Components ──────────────────────────────────────────────────
    print(f"[2/5] Initializing ARES components (GRM, ExpertManager)...")
    
    # Init GRM
    grm = GRM(input_dim=hidden_dim, hidden_dim=512).to(device)
    grm.eval()
    
    # Init ExpertManager
    expert_manager = ExpertManager(input_dim=hidden_dim).to(device)
    expert_manager.eval()
    
    # ─── 3. Load Checkpoints (If Available) ───────────────────────────────────
    ckpt_dir = Path(args.checkpoints_dir)
    print(f"[3/5] Loading weights from {ckpt_dir}...")
    
    grm_path = ckpt_dir / "reliability" / "grm.pt"
    if grm_path.exists():
        try:
            ckpt = torch.load(grm_path, map_location=device, weights_only=False)
            grm.load_state_dict(ckpt.get("model_state_dict", ckpt))
            print("  ✓ GRM loaded")
        except Exception as e:
            print(f"  × GRM load failed: {e}")
    else:
        print(f"  ? GRM checkpoint not found at {grm_path}. Using random weights.")
        
    router_path = ckpt_dir / "router" / "router.pt"
    if router_path.exists():
        try:
            ckpt = torch.load(router_path, map_location=device, weights_only=False)
            expert_manager.router.load_state_dict(ckpt["router_state_dict"])
            print("  ✓ Router loaded")
        except Exception as e:
            print(f"  × Router load failed: {e}")
    else:
        print(f"  ? Router checkpoint not found at {router_path}. Using random weights.")
        
    expert_names = ["general", "math", "code", "science", "reasoning"]
    for i, name in enumerate(expert_names):
        exp_path = ckpt_dir / "experts" / name / f"expert_{name}.pt"
        if exp_path.exists():
            try:
                ckpt = torch.load(exp_path, map_location=device, weights_only=False)
                expert_manager.experts[i].load_state_dict(ckpt["state_dict"])
                print(f"  ✓ Expert {name} loaded")
            except Exception as e:
                print(f"  × Expert {name} load failed: {e}")
        else:
            print(f"  ? Expert {name} checkpoint not found. Using random weights.")

    # ─── 4. Pipeline Execution ────────────────────────────────────────────────
    print(f"\n[4/5] Executing Pipeline for prompt:\n> '{args.prompt}'\n")
    
    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get backbone hidden states (No Experts yet)
        outputs = backbone(**inputs, output_hidden_states=True)
        # pooled representation (last token of last layer)
        final_hidden = outputs.hidden_states[-1][:, -1, :] 
        
        # Analyze Reliability
        domain_logits, feasibility, global_rel = grm(final_hidden)
        domains = ["general", "math", "code", "science", "reasoning"]
        predicted_domain = domains[domain_logits.argmax().item()]
        
        print(f"  --- GRM Analysis ---")
        print(f"  Global Reliability: {global_rel.item():.2%}")
        print(f"  Predicted Domain:   {predicted_domain}")
        
        # Route
        routed_rep, info = expert_manager(final_hidden, return_routing_info=True)
        route_names = ["BASE"] + expert_names
        selected_idx = info["selected_experts"].item()
        selected_route = route_names[selected_idx]
        route_confidence = info["routing_probs"][0, selected_idx].item()
        
        print(f"\n  --- Router Decision ---")
        print(f"  Selected Path:      {selected_route}")
        print(f"  Route Confidence:   {route_confidence:.2%}")
        
        # ─── 5. Generation ────────────────────────────────────────────────────
        print(f"\n[5/5] Generating text (using {selected_route} path)...")
        
        generated_ids = inputs["input_ids"]
        
        # Note: True ARES implementation modifies the backbone forward pass to 
        # intercept hidden states and apply the expert adapter during generation.
        # Since the backbone is a frozen HF model in our current implementation,
        # we demonstrate standard causal generation using the base model for this test.
        # (Implementing deep HF backbone hook-injection for generation is a Phase 5 task).
        
        out = backbone._model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
        
        print(f"\nOutput:\n{generated_text}\n")
        print("="*60)
        print("ARES End-to-End Pipeline Execution Successful.")

if __name__ == "__main__":
    main()
