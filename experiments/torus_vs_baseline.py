#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               TORUSFIELD vs BASELINE — EMPIRICAL COMPARISON                ║
║                                                                            ║
║  Experiment Log Entry per Emergent Judgment Section 4:                      ║
║  Hypothesis, Change, Measurement, Before/After, Verdict.                   ║
║                                                                            ║
║  Question: Does the toroidal energy surface add measurable value over      ║
║  plain edge reinforcement + trace-log pattern mining?                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author: Claude (Anthropic), at Kyle's direction                           ║
║  Date: 2026-02-26                                                          ║
║  Framework: William Kyle Million (~K¹) / IntuiTek¹                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import math
import json
import random
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfield_kernel import (
    TorusfieldEngine, AEGIS_CAPABILITY_MANIFEST, AEGIS_SEMANTIC_MAP,
    _clamp,
)
from baseline_engine import BaselineEngine


# ============================================================================
# EXPERIMENT INFRASTRUCTURE
# ============================================================================

RESULTS = []

def experiment(name: str, hypothesis: str, measurement: str):
    def decorator(fn):
        def wrapper():
            print(f"\n{'='*70}")
            print(f"EXPERIMENT: {name}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Measurement: {measurement}")
            print(f"{'='*70}")
            try:
                verdict, data = fn()
                RESULTS.append({
                    "name": name,
                    "hypothesis": hypothesis,
                    "measurement": measurement,
                    "verdict": verdict,
                    "data": data,
                })
                print(f"\nVERDICT: {verdict}")
                print(f"DATA: {json.dumps(data, indent=2)}")
            except Exception as e:
                RESULTS.append({
                    "name": name,
                    "hypothesis": hypothesis,
                    "verdict": "ERROR",
                    "data": {"error": f"{type(e).__name__}: {e}"},
                })
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


def rank_correlation(list_a: List[str], list_b: List[str]) -> float:
    """Spearman rank correlation between two ordered lists of node names."""
    if not list_a or not list_b:
        return 0.0
    all_items = sorted(set(list_a) | set(list_b))
    n = len(all_items)
    if n < 2:
        return 1.0
    rank_a = {item: i for i, item in enumerate(list_a)}
    rank_b = {item: i for i, item in enumerate(list_b)}
    max_rank = max(len(list_a), len(list_b))
    d_sq_sum = 0
    for item in all_items:
        ra = rank_a.get(item, max_rank)
        rb = rank_b.get(item, max_rank)
        d_sq_sum += (ra - rb) ** 2
    return 1 - (6 * d_sq_sum) / (n * (n**2 - 1))


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two score vectors (keyed by node name)."""
    all_keys = set(vec_a) | set(vec_b)
    if not all_keys:
        return 0.0
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ============================================================================
# EXPERIMENT 1: Do both engines learn the same edge weights?
# ============================================================================

@experiment(
    name="E1 — Edge Weight Convergence",
    hypothesis=(
        "After identical action sequences, both engines should learn similar "
        "edge weights, since reinforcement uses the same formula. Any divergence "
        "is caused by the torus energy field influencing the executor's reward "
        "calculation (coherence term)."
    ),
    measurement=(
        "Run 500 steps on both engines with same seed. Compare edge weights "
        "via mean absolute difference and correlation."
    ),
)
def test_edge_weight_convergence():
    torus = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)

    # Run both for 500 steps with same RNG seed
    torus.run(500)
    baseline.run(500)

    # Compare edge weights
    assert len(torus.graph.edges) == len(baseline.graph.edges)

    torus_weights = [e.w for e in torus.graph.edges]
    baseline_weights = [e.w for e in baseline.graph.edges]

    diffs = [abs(a - b) for a, b in zip(torus_weights, baseline_weights)]
    mean_diff = statistics.mean(diffs)
    max_diff = max(diffs)
    median_diff = statistics.median(diffs)

    # Count edges where sign differs (qualitative disagreement)
    sign_disagreements = sum(
        1 for a, b in zip(torus_weights, baseline_weights)
        if (a > 0.01 and b < -0.01) or (a < -0.01 and b > 0.01)
    )

    # Correlation
    n = len(torus_weights)
    mean_t = statistics.mean(torus_weights)
    mean_b = statistics.mean(baseline_weights)
    cov = sum((t - mean_t) * (b - mean_b) for t, b in
              zip(torus_weights, baseline_weights)) / n
    std_t = statistics.stdev(torus_weights) if n > 1 else 1
    std_b = statistics.stdev(baseline_weights) if n > 1 else 1
    correlation = cov / (std_t * std_b) if std_t > 0 and std_b > 0 else 0

    data = {
        "total_edges": n,
        "mean_abs_diff": round(mean_diff, 6),
        "median_abs_diff": round(median_diff, 6),
        "max_abs_diff": round(max_diff, 6),
        "sign_disagreements": sign_disagreements,
        "correlation": round(correlation, 4),
        "torus_nonzero": sum(1 for w in torus_weights if abs(w) > 0.001),
        "baseline_nonzero": sum(1 for w in baseline_weights if abs(w) > 0.001),
    }

    if mean_diff < 0.01 and correlation > 0.95:
        verdict = "CONVERGED — Torus energy field has negligible effect on edge learning"
    elif mean_diff < 0.05 and correlation > 0.80:
        verdict = "MINOR DIVERGENCE — Torus coherence term creates small reward differences"
    else:
        verdict = "SIGNIFICANT DIVERGENCE — Torus energy field materially changes learning"

    return verdict, data


# ============================================================================
# EXPERIMENT 2: Artifact Detection Parity
# ============================================================================

@experiment(
    name="E2 — Artifact Detection Parity",
    hypothesis=(
        "Artifact detection (wormholes, attractors, resonances) is based on "
        "trace log statistics, not energy fields. Both engines should detect "
        "similar artifacts from similar traces."
    ),
    measurement=(
        "Run 1000 steps on both, compare artifact counts, kinds, and IDs."
    ),
)
def test_artifact_detection_parity():
    torus = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)

    torus.run(1000)
    baseline.run(1000)

    t_arts = {a.kind: [] for a in torus.store.artifacts.values()}
    b_arts = {a.kind: [] for a in baseline.store.artifacts.values()}
    for a in torus.store.artifacts.values():
        t_arts.setdefault(a.kind, []).append(a)
    for a in baseline.store.artifacts.values():
        b_arts.setdefault(a.kind, []).append(a)

    # Compare by kind
    kinds = sorted(set(list(t_arts.keys()) + list(b_arts.keys())))
    comparison = {}
    for kind in kinds:
        t_list = t_arts.get(kind, [])
        b_list = b_arts.get(kind, [])
        comparison[kind] = {
            "torus_count": len(t_list),
            "baseline_count": len(b_list),
        }

    # Check if the same edge indices appear as wormholes
    t_worm_edges = {a.payload.get("edge_index") for a in
                    torus.store.get_by_kind("wormhole")}
    b_worm_edges = {a.payload.get("edge_index") for a in
                    baseline.store.get_by_kind("wormhole")}
    worm_overlap = len(t_worm_edges & b_worm_edges)
    worm_union = len(t_worm_edges | b_worm_edges)
    worm_jaccard = worm_overlap / worm_union if worm_union > 0 else 0

    # Same for attractors
    t_attr_nodes = {a.payload.get("node") for a in
                    torus.store.get_by_kind("attractor")}
    b_attr_nodes = {a.payload.get("node") for a in
                    baseline.store.get_by_kind("attractor")}
    attr_overlap = len(t_attr_nodes & b_attr_nodes)
    attr_union = len(t_attr_nodes | b_attr_nodes)
    attr_jaccard = attr_overlap / attr_union if attr_union > 0 else 0

    data = {
        "artifact_counts": comparison,
        "torus_total": len(torus.store.artifacts),
        "baseline_total": len(baseline.store.artifacts),
        "wormhole_edge_jaccard": round(worm_jaccard, 4),
        "wormhole_torus_edges": sorted(t_worm_edges - {None}),
        "wormhole_baseline_edges": sorted(b_worm_edges - {None}),
        "attractor_node_jaccard": round(attr_jaccard, 4),
        "attractor_torus_nodes": sorted(t_attr_nodes - {None}),
        "attractor_baseline_nodes": sorted(b_attr_nodes - {None}),
    }

    if worm_jaccard > 0.7 and attr_jaccard > 0.7:
        verdict = "CONFIRMED — Artifact detection is trace-based, not energy-dependent"
    elif worm_jaccard > 0.4:
        verdict = "PARTIAL — Some artifact divergence due to different routing paths"
    else:
        verdict = "DIVERGENT — Torus routing produces materially different artifacts"

    return verdict, data


# ============================================================================
# EXPERIMENT 3: Context Sensitivity (THE KEY TEST)
# ============================================================================

@experiment(
    name="E3 — Context Sensitivity of Routing Advisory",
    hypothesis=(
        "The torus's primary unique value proposition is context-dependent routing: "
        "energy injection at capability positions + diffusion should make the advisory "
        "responsive to WHAT the agent is about to do. The baseline has no mechanism "
        "for this — its advisory is identical regardless of context. If the torus "
        "produces meaningfully different advisories for different contexts while the "
        "baseline doesn't, the torus adds value."
    ),
    measurement=(
        "Generate advisories for 5 different contexts on both engines. Measure "
        "within-engine advisory variance (do different contexts produce different "
        "rankings?) and between-engine comparison (does the torus differentiate "
        "more than the baseline?)."
    ),
)
def test_context_sensitivity():
    # Start both from identical learned state
    torus = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)
    torus.run(300)
    baseline.run(300)

    # Define 5 distinct task contexts with target capabilities
    contexts = {
        "research": ["web_search", "web_fetch", "browser", "foundry_research"],
        "build": ["write", "exec", "foundry_implement", "foundry_write_skill"],
        "monitor": ["session_status", "foundry_metrics", "foundry_overseer"],
        "communicate": ["message", "tts", "sessions_send", "nodes"],
        "automate": ["foundry_write_hook", "foundry_crystallize", "process"],
    }

    # Force both to same cursor for fair comparison
    common_cursor = "read"
    torus.cursor_node = common_cursor
    baseline.cursor_node = common_cursor

    torus_advisories = {}
    baseline_advisories = {}

    for ctx_name, caps in contexts.items():
        # TORUS: inject energy, diffuse, then score
        # Make a fresh copy of energy state for each context test
        saved_energy = torus.state.fields["energy"][:]
        for cap in caps:
            cap_names = [c["name"] for c in AEGIS_CAPABILITY_MANIFEST]
            if cap in cap_names:
                idx = cap_names.index(cap)
                th = idx % torus.state.theta_bins
                ph = (idx * 3) % torus.state.phi_bins
                torus.inject_event(f"ctx_{cap}", 1.5, th, ph, "energy")
        # Diffuse 15 steps (same as consult warmup)
        for _ in range(15):
            torus.state.diffuse("energy", torus.theta_mix, torus.phi_mix)
            torus.state.decay(torus.leak_decay)

        # Score from cursor
        t_scores = {}
        for eidx in torus.graph.adj.get(common_cursor, []):
            e = torus.graph.edges[eidx]
            bias = torus.policy.edge_bias.get(eidx, 0.0)
            # Torus score includes energy landscape influence via executor
            # But routing score itself is edge-based. The energy influences
            # the REWARD on execution, not the pre-selection score.
            # Let me check: does the router use energy? No — the router
            # uses alpha*u - beta*c + gamma*n + w + bias. Energy only
            # affects the executor's coherence-based reward.
            # So for pre-selection advisory, torus and baseline are IDENTICAL.
            # The energy only matters for reinforcement quality.
            s = (torus.router.alpha * e.u - torus.router.beta * e.c
                 + torus.router.gamma * e.n + e.w + bias)
            t_scores[e.dst] = round(s, 6)
        torus_advisories[ctx_name] = t_scores

        # Restore energy state for next context
        torus.state.fields["energy"] = saved_energy

        # BASELINE: no context mechanism
        b_scores = {}
        for eidx in baseline.graph.adj.get(common_cursor, []):
            e = baseline.graph.edges[eidx]
            bias = baseline.policy.edge_bias.get(eidx, 0.0)
            s = (baseline.router.alpha * e.u - baseline.router.beta * e.c
                 + baseline.router.gamma * e.n + e.w + bias)
            b_scores[e.dst] = round(s, 6)
        baseline_advisories[ctx_name] = b_scores

    # Measure within-engine variance
    ctx_names = list(contexts.keys())

    def advisory_variance(advisories: Dict[str, Dict[str, float]]) -> float:
        """Mean pairwise distance between advisories for different contexts."""
        pairs = []
        for i in range(len(ctx_names)):
            for j in range(i+1, len(ctx_names)):
                sim = cosine_similarity(
                    advisories[ctx_names[i]], advisories[ctx_names[j]])
                pairs.append(1.0 - sim)  # distance
        return statistics.mean(pairs) if pairs else 0.0

    torus_variance = advisory_variance(torus_advisories)
    baseline_variance = advisory_variance(baseline_advisories)

    # Check if baseline produces identical advisories (it should)
    baseline_all_same = all(
        baseline_advisories[ctx_names[0]] == baseline_advisories[cn]
        for cn in ctx_names[1:]
    )

    # Critical insight discovery
    # Let me check: does the torus router actually use energy in scoring?
    # Looking at Router.pick(): s = alpha*u - beta*c + gamma*n + w + bias
    # Energy is NOT in the routing score. It only enters through the executor's
    # coherence calculation which affects REWARD, not PRE-SELECTION.
    torus_all_same = all(
        torus_advisories[ctx_names[0]] == torus_advisories[cn]
        for cn in ctx_names[1:]
    )

    data = {
        "torus_variance": round(torus_variance, 6),
        "baseline_variance": round(baseline_variance, 6),
        "baseline_identical_across_contexts": baseline_all_same,
        "torus_identical_across_contexts": torus_all_same,
        "critical_finding": (
            "The torus Router.pick() formula is: alpha*u - beta*c + gamma*n + w + bias. "
            "Energy fields are NOT in this formula. Energy only enters through the "
            "Executor's coherence-based reward, which affects future edge weight "
            "learning (reinforcement), NOT current advisory scoring. "
            "This means for pre-selection advisory purposes, the torus and baseline "
            "produce IDENTICAL context-independent rankings from the same cursor."
        ),
        "sample_torus_research": dict(sorted(
            torus_advisories["research"].items(), key=lambda x: -x[1])[:5]),
        "sample_torus_build": dict(sorted(
            torus_advisories["build"].items(), key=lambda x: -x[1])[:5]),
        "sample_baseline_research": dict(sorted(
            baseline_advisories["research"].items(), key=lambda x: -x[1])[:5]),
    }

    if torus_all_same and baseline_all_same:
        verdict = (
            "CONFIRMED CONCERN — Neither engine differentiates advisory by context. "
            "The torus energy field does NOT influence routing scores. "
            "Context sensitivity in UCS bridge.consult() comes from the bridge's "
            "keyword-to-capability mapping, NOT from the torus."
        )
    elif not torus_all_same and baseline_all_same:
        verdict = "TORUS ADDS VALUE — Energy field creates context-differentiated routing"
    else:
        verdict = "UNEXPECTED — Both engines show context variance"

    return verdict, data


# ============================================================================
# EXPERIMENT 4: Energy Field Persistence Across Boots
# ============================================================================

@experiment(
    name="E4 — Energy Field Persistence Value",
    hypothesis=(
        "Energy fields are saved in torusfield_state.json and restored on boot. "
        "If the accumulated energy landscape meaningfully differs from a fresh "
        "engine's landscape, then energy persistence adds information that edge "
        "weights alone don't capture."
    ),
    measurement=(
        "Run 500 steps, export state, reimport into a new engine. Compare "
        "energy field against a fresh engine. Measure information content "
        "(non-zero cells, entropy, total energy)."
    ),
)
def test_energy_persistence():
    # Train an engine
    trained = TorusfieldEngine.boot_aegis(seed=42)
    trained.run(500)

    # Export and reimport
    state = trained.export_state()
    restored = TorusfieldEngine.boot_aegis(seed=99)  # different seed
    restored.import_state(state)

    # Fresh engine for comparison
    fresh = TorusfieldEngine.boot_aegis(seed=42)

    # Compare energy fields
    trained_energy = trained.state.fields["energy"]
    restored_energy = restored.state.fields["energy"]
    fresh_energy = fresh.state.fields["energy"]

    # Are trained and restored identical? (tests persistence fidelity)
    persistence_diffs = [abs(a - b) for a, b in
                         zip(trained_energy, restored_energy)]
    persistence_max_diff = max(persistence_diffs)

    # How different is trained from fresh?
    divergence = [abs(a - b) for a, b in zip(trained_energy, fresh_energy)]
    divergence_mean = statistics.mean(divergence)
    divergence_max = max(divergence)

    # Information content
    def field_info(field):
        nonzero = sum(1 for v in field if abs(v) > 1e-6)
        total = sum(abs(v) for v in field)
        max_v = max(abs(v) for v in field)
        return {"nonzero_cells": nonzero, "total_energy": round(total, 4),
                "max_amplitude": round(max_v, 6)}

    trained_info = field_info(trained_energy)
    fresh_info = field_info(fresh_energy)

    # After 500 steps of decay at 0.995 per step, initial energy decays by
    # 0.995^500 ≈ 0.082. So 92% of injected energy decays away.
    decay_factor = trained.leak_decay ** 500

    data = {
        "persistence_fidelity_max_diff": round(persistence_max_diff, 10),
        "persistence_faithful": persistence_max_diff < 1e-8,
        "trained_field_info": trained_info,
        "fresh_field_info": fresh_info,
        "divergence_mean": round(divergence_mean, 6),
        "divergence_max": round(divergence_max, 6),
        "theoretical_decay_500_steps": round(decay_factor, 6),
        "note": (
            f"After 500 steps at decay={trained.leak_decay}, "
            f"initial energy retains {decay_factor*100:.1f}% of original amplitude. "
            f"The energy field is mostly injection residue from the last few steps, "
            f"not accumulated history."
        ),
    }

    if trained_info["total_energy"] < 0.1:
        verdict = (
            "NEGLIGIBLE — Energy field is near-zero after decay. "
            "Persistence adds no meaningful information."
        )
    elif divergence_mean < 0.001:
        verdict = "MINIMAL — Trained energy field barely differs from fresh"
    else:
        verdict = "MEANINGFUL — Energy field carries persistent information beyond edge weights"

    return verdict, data


# ============================================================================
# EXPERIMENT 5: Reinforcement Quality — Coherence Reward vs Fixed Reward
# ============================================================================

@experiment(
    name="E5 — Coherence-Based vs Fixed Reward Signal Quality",
    hypothesis=(
        "The torus executor computes reward using energy gradient coherence: "
        "reward = utility_weight * edge.u + coherence_weight * coherence. "
        "The baseline uses fixed: reward = utility_weight * edge.u. "
        "If coherence provides a better reward signal, the torus should learn "
        "more discriminating edge weights (higher variance, better separation "
        "between good and bad edges)."
    ),
    measurement=(
        "Run 1000 steps on both. Compare edge weight distributions: "
        "variance, range, and separation between top/bottom quartiles."
    ),
)
def test_reward_quality():
    torus = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)

    torus.run(1000)
    baseline.run(1000)

    t_weights = sorted([e.w for e in torus.graph.edges])
    b_weights = sorted([e.w for e in baseline.graph.edges])

    def weight_stats(weights):
        nonzero = [w for w in weights if abs(w) > 0.001]
        if len(nonzero) < 4:
            return {"nonzero": len(nonzero), "insufficient_data": True}
        q1 = nonzero[len(nonzero)//4]
        q3 = nonzero[3*len(nonzero)//4]
        return {
            "nonzero_count": len(nonzero),
            "mean": round(statistics.mean(nonzero), 6),
            "stdev": round(statistics.stdev(nonzero), 6),
            "min": round(min(nonzero), 6),
            "max": round(max(nonzero), 6),
            "q1": round(q1, 6),
            "q3": round(q3, 6),
            "iqr": round(q3 - q1, 6),
            "range": round(max(nonzero) - min(nonzero), 6),
        }

    t_stats = weight_stats(t_weights)
    b_stats = weight_stats(b_weights)

    data = {
        "torus_weights": t_stats,
        "baseline_weights": b_stats,
    }

    t_iqr = t_stats.get("iqr", 0)
    b_iqr = b_stats.get("iqr", 0)

    if abs(t_iqr - b_iqr) < 0.005:
        verdict = (
            "NO DIFFERENCE — Coherence reward produces same weight distribution "
            "as fixed reward. Energy field does not improve learning signal quality."
        )
    elif t_iqr > b_iqr * 1.2:
        verdict = (
            f"TORUS BETTER — Coherence reward produces {((t_iqr/b_iqr)-1)*100:.1f}% "
            f"wider weight separation (IQR: {t_iqr:.4f} vs {b_iqr:.4f})"
        )
    elif b_iqr > t_iqr * 1.2:
        verdict = (
            f"BASELINE BETTER — Fixed reward produces {((b_iqr/t_iqr)-1)*100:.1f}% "
            f"wider weight separation (IQR: {b_iqr:.4f} vs {t_iqr:.4f})"
        )
    else:
        verdict = "MARGINAL — Small difference in weight distributions"

    return verdict, data


# ============================================================================
# EXPERIMENT 6: Routing Trajectory Divergence Over Time
# ============================================================================

@experiment(
    name="E6 — Trajectory Divergence Over Time",
    hypothesis=(
        "Even if individual routing scores are similar, the torus's coherence "
        "reward may cause the engines to explore different trajectories over "
        "time, leading to different learned preferences. Measure how quickly "
        "the two engines' routing trajectories diverge."
    ),
    measurement=(
        "Run both engines step-by-step for 500 steps with same seed. "
        "At each step, record which node was chosen. Measure: "
        "what % of steps chose the same node? When does divergence start?"
    ),
)
def test_trajectory_divergence():
    torus = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)

    agreements = []
    torus_path = []
    baseline_path = []
    first_divergence = None

    for i in range(500):
        t_dst = torus.step_theta()
        b_dst = baseline.step()

        torus_path.append(t_dst)
        baseline_path.append(b_dst)

        agree = (t_dst == b_dst)
        agreements.append(agree)

        if not agree and first_divergence is None:
            first_divergence = i

    total = len(agreements)
    agreement_count = sum(agreements)
    agreement_pct = agreement_count / total * 100

    # Agreement by window
    windows = {}
    for start in [0, 100, 200, 300, 400]:
        end = min(start + 100, total)
        window_agree = sum(agreements[start:end])
        windows[f"steps_{start}-{end}"] = f"{window_agree}/{end-start}"

    # Node frequency comparison
    t_freq = {}
    b_freq = {}
    for n in torus_path:
        t_freq[n] = t_freq.get(n, 0) + 1
    for n in baseline_path:
        b_freq[n] = b_freq.get(n, 0) + 1

    data = {
        "total_steps": total,
        "agreement_count": agreement_count,
        "agreement_pct": round(agreement_pct, 2),
        "first_divergence_step": first_divergence,
        "agreement_by_window": windows,
        "torus_top5_destinations": dict(sorted(
            t_freq.items(), key=lambda x: -x[1])[:5]),
        "baseline_top5_destinations": dict(sorted(
            b_freq.items(), key=lambda x: -x[1])[:5]),
    }

    if agreement_pct > 90:
        verdict = (
            f"NEAR-IDENTICAL — {agreement_pct:.1f}% same routing decisions. "
            "Torus energy field has negligible effect on actual trajectory."
        )
    elif agreement_pct > 70:
        verdict = (
            f"MODERATE DIVERGENCE — {agreement_pct:.1f}% agreement. "
            f"Torus coherence reward creates some routing differences."
        )
    else:
        verdict = (
            f"SIGNIFICANT DIVERGENCE — Only {agreement_pct:.1f}% agreement. "
            "Torus energy dynamics substantially alter routing behavior."
        )

    return verdict, data


# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

ALL_EXPERIMENTS = [
    test_edge_weight_convergence,
    test_artifact_detection_parity,
    test_context_sensitivity,
    test_energy_persistence,
    test_reward_quality,
    test_trajectory_divergence,
]

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TORUSFIELD vs BASELINE — EMPIRICAL COMPARISON             ║")
    print("║  Does the torus add value over plain edge RL?              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    for exp_fn in ALL_EXPERIMENTS:
        exp_fn()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for r in RESULTS:
        icon = "◆" if "CONFIRMED" in r["verdict"] or "TORUS" in r["verdict"] else "◇"
        print(f"\n{icon} {r['name']}")
        print(f"  {r['verdict']}")

    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)
