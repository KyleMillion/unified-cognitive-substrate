#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  OPTION B: Separate Context Energy Field                                   ║
║                                                                            ║
║  Problem: Injection schedule creates permanent hotspots on the energy      ║
║  field that drown out context-specific injections.                          ║
║                                                                            ║
║  Fix: Add a "context_energy" field that is:                                ║
║    - Zeroed before each consult                                            ║
║    - Populated ONLY with context-relevant capability injections            ║
║    - Read by the router via delta * context_energy_at_dst                  ║
║    - NOT contaminated by the periodic injection schedule                   ║
║                                                                            ║
║  This tests whether the torus CAN produce context-sensitive routing        ║
║  when the energy signal isn't buried under schedule noise.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfield_kernel import (
    TorusfieldEngine, TorusState, EdgeGraph, PolicyKernel,
    AEGIS_CAPABILITY_MANIFEST, AEGIS_SEMANTIC_MAP,
)
from patched_engine import build_node_coordinate_map, EnergyCoupledRouter


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    all_keys = set(vec_a) | set(vec_b)
    if not all_keys:
        return 0.0
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class ContextFieldRouter:
    """Router that reads a SEPARATE context_energy field.
    
    Key difference from the delta-fix: this reads a field that is
    zeroed and freshly populated for each context query, not the
    main energy field contaminated by the injection schedule.
    """
    def __init__(self, alpha=1.25, beta=1.10, gamma=0.35,
                 delta=0.50, temperature=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.temperature = temperature

    def score_with_context(self, graph: EdgeGraph, policy: PolicyKernel,
                           src: str, context_energy: Dict[str, float]
                           ) -> List[Dict]:
        """Score outgoing edges using a clean context energy map.
        
        context_energy: {node_name: energy_value} — built fresh per context,
        not from the torus field.
        """
        candidates = graph.adj.get(src, [])
        scored = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)

            # Original terms
            s = (self.alpha * e.u - self.beta * e.c
                 + self.gamma * e.n + e.w + bias)

            # Context energy — clean, per-query, no schedule contamination
            ctx_e = context_energy.get(e.dst, 0.0)
            s += self.delta * ctx_e

            scored.append({
                "to": e.dst,
                "edge_index": ei,
                "score": round(s, 6),
                "base_score": round(s - self.delta * ctx_e, 6),
                "context_boost": round(self.delta * ctx_e, 6),
            })
        scored.sort(key=lambda x: -x["score"])
        return scored


def build_context_energy(capabilities: List[str],
                         node_coords: Dict[str, Tuple[int, int]],
                         state: TorusState,
                         injection_magnitude: float = 2.0,
                         diffusion_steps: int = 5,
                         theta_mix: float = 0.08,
                         phi_mix: float = 0.06) -> Dict[str, float]:
    """Build a CLEAN context energy map by injecting into a temporary
    field, diffusing briefly, then reading energy at all node positions.
    
    This simulates what the torus SHOULD do: inject at relevant capabilities,
    let energy diffuse to topological neighbors, then read the resulting
    energy landscape. But on a SEPARATE scratch field.
    """
    size = state.theta_bins * state.phi_bins

    # Zero scratch field
    if "context_energy" not in state.fields:
        state.fields["context_energy"] = [0.0] * size
    else:
        state.fields["context_energy"] = [0.0] * size

    # Inject at relevant capabilities
    for cap in capabilities:
        if cap in node_coords:
            th, ph = node_coords[cap]
            state.inject(f"ctx_{cap}", injection_magnitude, th, ph,
                        "context_energy")

    # Limited diffusion — just enough to spread to immediate neighbors
    for _ in range(diffusion_steps):
        state.diffuse("context_energy", theta_mix, phi_mix)

    # Read energy at all node positions
    energy_map = {}
    all_caps = [c["name"] for c in AEGIS_CAPABILITY_MANIFEST]
    for cap_name in all_caps:
        if cap_name in node_coords:
            th, ph = node_coords[cap_name]
            energy_map[cap_name] = state.get("context_energy", th, ph)

    return energy_map


def run_option_b_test():
    print("=" * 70)
    print("OPTION B: Separate Context Energy Field")
    print("=" * 70)
    print()

    # Boot and train engine
    engine = TorusfieldEngine.boot_aegis(seed=42)
    engine.run(300)
    engine.cursor_node = "read"

    node_coords = build_node_coordinate_map(AEGIS_CAPABILITY_MANIFEST, 24, 32)

    contexts = {
        "research": ["web_search", "web_fetch", "browser", "foundry_research"],
        "build": ["write", "exec", "foundry_implement", "foundry_write_skill"],
        "monitor": ["session_status", "foundry_metrics", "foundry_overseer"],
        "communicate": ["message", "tts", "sessions_send", "nodes"],
        "automate": ["foundry_write_hook", "foundry_crystallize", "process"],
    }

    router = ContextFieldRouter(delta=0.50)
    advisories = {}

    for ctx_name, caps in contexts.items():
        # Build CLEAN context energy — no schedule contamination
        ctx_energy = build_context_energy(
            caps, node_coords, engine.state,
            injection_magnitude=2.0,
            diffusion_steps=3,  # Less diffusion = more localized signal
        )

        # Score with clean context
        scored = router.score_with_context(
            engine.graph, engine.policy, "read", ctx_energy)

        advisories[ctx_name] = {s["to"]: s["score"] for s in scored}

        # Show top-5 with context breakdown
        print(f"  [{ctx_name}] Top-5 advisory:")
        for item in scored[:5]:
            print(f"    {item['to']:30s}  base={item['base_score']:+.4f}  "
                  f"ctx_boost={item['context_boost']:+.4f}  "
                  f"total={item['score']:+.4f}")

        # Show which nodes got the most context boost
        boosts = [(s["to"], s["context_boost"]) for s in scored
                  if s["context_boost"] > 0.01]
        if boosts:
            boosts.sort(key=lambda x: -x[1])
            boosted_names = [f"{n}({b:+.2f})" for n, b in boosts[:4]]
            print(f"    Context-boosted: {', '.join(boosted_names)}")
        print()

    # Measure context sensitivity
    ctx_names = list(contexts.keys())
    pairs = []
    for i in range(len(ctx_names)):
        for j in range(i + 1, len(ctx_names)):
            sim = cosine_similarity(
                advisories[ctx_names[i]], advisories[ctx_names[j]])
            pairs.append(1.0 - sim)
    variance = statistics.mean(pairs)

    # Check unique top-1 picks
    top1 = {ctx: max(scores, key=scores.get) for ctx, scores in advisories.items()}
    unique_top1 = len(set(top1.values()))

    # Check unique top-3 rankings
    top3 = {}
    for ctx, scores in advisories.items():
        sorted_nodes = sorted(scores, key=scores.get, reverse=True)[:3]
        top3[ctx] = sorted_nodes

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Advisory variance: {variance:.6f}")
    print(f"  (Original torus: 0.000000)")
    print(f"  (Delta-fix torus: 0.000020)")
    print()
    print(f"Unique #1 recommendations: {unique_top1} out of 5 contexts")
    print()
    print("Top-1 by context:")
    for ctx, node in top1.items():
        print(f"  {ctx:15s} → {node}")
    print()
    print("Top-3 by context:")
    for ctx, nodes in top3.items():
        print(f"  {ctx:15s} → {' → '.join(nodes)}")

    # Pairwise ranking comparison
    print()
    print("Pairwise advisory distances (0=identical, 1=orthogonal):")
    for i in range(len(ctx_names)):
        for j in range(i + 1, len(ctx_names)):
            sim = cosine_similarity(
                advisories[ctx_names[i]], advisories[ctx_names[j]])
            dist = 1.0 - sim
            print(f"  {ctx_names[i]:15s} vs {ctx_names[j]:15s}: {dist:.6f}")

    print()
    print("=" * 70)
    if unique_top1 >= 3:
        print(f"VERDICT: OPTION B WORKS — {unique_top1}/5 contexts get distinct")
        print(f"  top recommendations. Torus CAN do context-sensitive routing")
        print(f"  when context energy is isolated from schedule energy.")
        print(f"  Variance: {variance:.6f} (vs 0.000000 original, 0.000020 delta-fix)")
    elif unique_top1 >= 2:
        print(f"VERDICT: PARTIAL — {unique_top1}/5 distinct top picks. Better than")
        print(f"  the delta fix but not yet reliably context-sensitive.")
    else:
        print(f"VERDICT: OPTION B FAILS — Still only {unique_top1} distinct top pick.")
        print(f"  The problem is not just schedule contamination.")
    print("=" * 70)

    return variance, unique_top1, top1


# ============================================================================
# DIFFUSION STEP SWEEP: How much diffusion helps vs hurts?
# ============================================================================

def test_diffusion_sweep():
    print()
    print("=" * 70)
    print("DIFFUSION SWEEP: Optimal context diffusion steps")
    print("=" * 70)
    print()

    engine = TorusfieldEngine.boot_aegis(seed=42)
    engine.run(300)
    engine.cursor_node = "read"

    node_coords = build_node_coordinate_map(AEGIS_CAPABILITY_MANIFEST, 24, 32)

    contexts = {
        "research": ["web_search", "web_fetch", "browser", "foundry_research"],
        "build": ["write", "exec", "foundry_implement", "foundry_write_skill"],
        "monitor": ["session_status", "foundry_metrics", "foundry_overseer"],
    }

    diffusion_steps_range = [0, 1, 2, 3, 5, 8, 15, 25]

    print(f"{'steps':>6s}  {'variance':>10s}  {'unique#1':>8s}  "
          f"{'research#1':>20s}  {'build#1':>20s}  {'monitor#1':>20s}")
    print("-" * 90)

    for d_steps in diffusion_steps_range:
        router = ContextFieldRouter(delta=0.50)
        advisories = {}

        for ctx_name, caps in contexts.items():
            ctx_energy = build_context_energy(
                caps, node_coords, engine.state,
                injection_magnitude=2.0,
                diffusion_steps=d_steps,
            )
            scored = router.score_with_context(
                engine.graph, engine.policy, "read", ctx_energy)
            advisories[ctx_name] = {s["to"]: s["score"] for s in scored}

        ctx_names = list(contexts.keys())
        pairs = []
        for i in range(len(ctx_names)):
            for j in range(i + 1, len(ctx_names)):
                sim = cosine_similarity(
                    advisories[ctx_names[i]], advisories[ctx_names[j]])
                pairs.append(1.0 - sim)
        variance = statistics.mean(pairs)

        top1 = {ctx: max(scores, key=scores.get)
                for ctx, scores in advisories.items()}
        unique = len(set(top1.values()))

        print(f"{d_steps:6d}  {variance:10.6f}  {unique:8d}  "
              f"{top1['research']:>20s}  {top1['build']:>20s}  "
              f"{top1['monitor']:>20s}")

    print()
    print("0 steps = pure injection, no diffusion (maximum locality)")
    print("More steps = more spread (approaches uniform distribution)")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  OPTION B: Does a Separate Context Field Fix Routing?       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    result = run_option_b_test()
    test_diffusion_sweep()

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    if result[1] >= 3:
        print("The torus CAN provide context-sensitive routing when:")
        print("  1. Context energy is on a separate field from schedule energy")
        print("  2. Diffusion is limited (few steps) to preserve signal locality")
        print("  3. The router reads context_energy at destination nodes")
        print()
        print("Implementation path for UCS v1.2:")
        print("  - Add 'context_energy' field to TorusState")
        print("  - bridge.consult() zeros + populates context_energy")
        print("  - Router.pick() reads context_energy via delta term")
        print("  - Injection schedule continues using 'energy' field (unchanged)")
    else:
        print("Even with a clean context field, the torus does not produce")
        print("reliably different routing for different contexts.")
        print("Consider Option C: simplify away the energy surface.")
