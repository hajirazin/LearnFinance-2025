"""Time the actual full policy backward pass before the larger scratch run."""

import json
import time
from pathlib import Path

import torch

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.policy import PPODiscoveryActorCritic
from brain_api.core.ppo_discovery.synthetic import make_synthetic_state

ROOT = Path(__file__).resolve().parent
if not torch.backends.mps.is_available():
    raise RuntimeError("The market training benchmark requires the available Mac GPU")
torch.set_num_threads(4)
state = make_synthetic_state()
policy = PPODiscoveryActorCritic(PPODiscoveryConfig()).to("mps").eval()
with torch.no_grad():
    action, _, _ = policy.sample_action_value_log_prob(state)
elapsed = []
for _ in range(3):
    torch.mps.synchronize()
    start = time.perf_counter()
    logp, value, _, _ = policy.evaluate_actions([state] * 4, [action] * 4)
    loss = -logp.mean() + value.square().mean()
    policy.zero_grad()
    loss.backward()
    torch.mps.synchronize()
    elapsed.append(time.perf_counter() - start)
result = {
    "microbatch": 4,
    "full_policy_forward_backward_seconds": elapsed,
    "median_seconds_per_state": sorted(elapsed)[1] / 4,
    "mps_driver_allocated_gb": torch.mps.driver_allocated_memory() / 1e9,
}
(ROOT / "market_benchmark.json").write_text(json.dumps(result, indent=2))
print(result, flush=True)
