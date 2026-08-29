"""Router Network package for ARES (PRD §3.2.5, §4.4).

Provides:
- Router: MLP router for base vs expert routing with soft/top-k/Gumbel-Softmax support.
- RouterConfig: Configuration dataclass for Router architecture.
- RoutingOutput: Container for routing probabilities, logits, confidence, and expert paths.
- SwitchLoadBalancingLoss: Switch Transformer auxiliary load-balancing loss.
- RouterLoss: Combined classification and load-balancing loss.
- RouterTrainer: Supervised training loop for router on oracle decisions.
- generate_oracle_targets: Generates oracle routing labels from domain and correctness data.
"""

from .architecture import EXPERT_NAMES, ROUTE_NAMES, Router, RouterConfig, RoutingOutput
from .loss import RouterLoss, SwitchLoadBalancingLoss, generate_oracle_targets
from .trainer import RouterTrainer

__all__ = [
    "Router",
    "RouterConfig",
    "RoutingOutput",
    "RouterLoss",
    "SwitchLoadBalancingLoss",
    "RouterTrainer",
    "generate_oracle_targets",
    "EXPERT_NAMES",
    "ROUTE_NAMES",
]
