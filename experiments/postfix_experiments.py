#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  POST-FIX EXPERIMENT: Does Energy-Coupled Routing Fix E3?                  ║
║                                                                            ║
║  Before: Router.pick() = alpha*u - beta*c + gamma*n + w + bias            ║
║  After:  Router.pick() = alpha*u - beta*c + gamma*n + w + bias            ║
║                          + delta * energy_at_destination                    ║
║                                                                            ║
║  Re-running E3 (context sensitivity), E5 (reward quality),                 ║
║  and E6 (trajectory divergence) to measure the effect.                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Date: 2026-02-26                                                          ║
║  Predecessor: torus_vs_baseline.py (E1-E6, pre-fix)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import math
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torusfield_kernel import (
    TorusfieldEngine, AEGIS_CAPABILITY_MANIFEST, AEGIS_SEMANTIC_MAP,
)
from baseline_engine import BaselineEngine
from patched_engine import (
    PatchedTorusfieldEngine, build_node_coordinate_map, EnergyCoupledRouter,
)


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


# ============================================================================
# E3-FIXED: Context Sensitivity with Energy-Coupled Router
# ============================================================================

def test_e3_fixed():
    print("=" * 70)
    print("E3-FIXED: Context Sensitivity — Energy-Coupled Router")
    print("=" * 70)
    print()
    print("The original E3 showed that injecting energy at relevant")
    print("capabilities had ZERO effect on advisory scores because")
    print("Router.pick() didn't read energy. The patched router does.")
    print()

    # Boot patched engine and train it
    patched = PatchedTorusfieldEngine.boot_aegis(seed=42, delta=0.40)
    patched.run(300)

    # Also boot original torus and baseline for three-way comparison
    original = TorusfieldEngine.boot_aegis(seed=42)
    original.run(300)
    baseline = BaselineEngine.boot_aegis(seed=42)
    baseline.run(300)

    node_coords = build_node_coordinate_map(
        AEGIS_CAPABILITY_MANIFEST, 24, 32)

    contexts = {
        "research": ["web_search", "web_fetch", "browser", "foundry_research"],
        "build": ["write", "exec", "foundry_implement", "foundry_write_skill"],
        "monitor": ["session_status", "foundry_metrics", "foundry_overseer"],
        "communicate": ["message", "tts", "sessions_send", "nodes"],
        "automate": ["foundry_write_hook", "foundry_crystallize", "process"],
    }

    common_cursor = "read"
    patched.cursor_node = common_cursor
    original.cursor_node = common_cursor
    baseline.cursor_node = common_cursor

    patched_advisories = {}
    original_advisories = {}
    baseline_advisories = {}

    for ctx_name, caps in contexts.items():
        # === PATCHED ENGINE: inject energy, diffuse, score with energy coupling ===
        saved_energy = patched.state.fields["energy"][:]

        for cap in caps:
            if cap in node_coords:
                th, ph = node_coords[cap]
                patched.inject_event(f"ctx_{cap}", 2.0, th, ph, "energy")

        for _ in range(15):
            patched.state.diffuse("energy", patched.theta_mix, patched.phi_mix)
            patched.state.decay(patched.leak_decay)

        # Score using energy-coupled router
        p_scores = {}
        energy_contributions = {}
        scored = patched.get_context_advisory(common_cursor)
        for item in scored:
            p_scores[item["to"]] = item["score"]
            energy_contributions[item["to"]] = item["energy_contribution"]

        patched_advisories[ctx_name] = p_scores

        # Show energy influence for this context
        top_energy = sorted(energy_contributions.items(),
                            key=lambda x: -abs(x[1]))[:5]
        print(f"  [{ctx_name}] Top energy influences:")
        for node, energy in top_energy:
            print(f"    {node}: {energy:+.4f}")
        print()

        patched.state.fields["energy"] = saved_energy

        # === ORIGINAL ENGINE: inject energy, diffuse, score WITHOUT coupling ===
        saved_orig = original.state.fields["energy"][:]

        for cap in caps:
            if cap in node_coords:
                th, ph = node_coords[cap]
                original.inject_event(f"ctx_{cap}", 2.0, th, ph, "energy")

        for _ in range(15):
            original.state.diffuse("energy", original.theta_mix, original.phi_mix)
            original.state.decay(original.leak_decay)

        o_scores = {}
        for eidx in original.graph.adj.get(common_cursor, []):
            e = original.graph.edges[eidx]
            bias = original.policy.edge_bias.get(eidx, 0.0)
            s = (1.25 * e.u - 1.10 * e.c + 0.35 * e.n + e.w + bias)
            o_scores[e.dst] = round(s, 6)
        original_advisories[ctx_name] = o_scores

        original.state.fields["energy"] = saved_orig

        # === BASELINE: no energy at all ===
        b_scores = {}
        for eidx in baseline.graph.adj.get(common_cursor, []):
            e = baseline.graph.edges[eidx]
            bias = baseline.policy.edge_bias.get(eidx, 0.0)
            s = (1.25 * e.u - 1.10 * e.c + 0.35 * e.n + e.w + bias)
            b_scores[e.dst] = round(s, 6)
        baseline_advisories[ctx_name] = b_scores

    # Measure advisory variance across contexts
    ctx_names = list(contexts.keys())

    def advisory_variance(advisories):
        pairs = []
        for i in range(len(ctx_names)):
            for j in range(i + 1, len(ctx_names)):
                sim = cosine_similarity(
                    advisories[ctx_names[i]], advisories[ctx_names[j]])
                pairs.append(1.0 - sim)
        return statistics.mean(pairs) if pairs else 0.0

    p_var = advisory_variance(patched_advisories)
    o_var = advisory_variance(original_advisories)
    b_var = advisory_variance(baseline_advisories)

    # Check if context actually changes the ranking
    def top_n_ranking(advisories, n=5):
        return {
            ctx: [k for k, _ in sorted(
                scores.items(), key=lambda x: -x[1])[:n]]
            for ctx, scores in advisories.items()
        }

    p_rankings = top_n_ranking(patched_advisories)
    o_rankings = top_n_ranking(original_advisories)

    # Check: does patched produce different #1 picks for different contexts?
    p_top1 = {ctx: ranks[0] for ctx, ranks in p_rankings.items()}
    o_top1 = {ctx: ranks[0] for ctx, ranks in o_rankings.items()}

    p_unique_top1 = len(set(p_top1.values()))
    o_unique_top1 = len(set(o_top1.values()))

    print("=" * 70)
    print("CONTEXT SENSITIVITY RESULTS")
    print("=" * 70)
    print()
    print(f"Advisory variance (higher = more context-sensitive):")
    print(f"  Patched (energy-coupled):  {p_var:.6f}")
    print(f"  Original torus (no fix):   {o_var:.6f}")
    print(f"  Baseline (no energy):      {b_var:.6f}")
    print()
    print(f"Unique #1 recommendations across 5 contexts:")
    print(f"  Patched:  {p_unique_top1} distinct top picks")
    print(f"  Original: {o_unique_top1} distinct top picks")
    print()
    print("Top-5 rankings by context (PATCHED):")
    for ctx, ranks in p_rankings.items():
        print(f"  {ctx:15s}: {' → '.join(ranks)}")
    print()
    print("Top-5 rankings by context (ORIGINAL — should be identical):")
    for ctx, ranks in o_rankings.items():
        print(f"  {ctx:15s}: {' → '.join(ranks)}")
    print()

    # Detailed score comparison for research vs build contexts
    print("=" * 70)
    print("SCORE COMPARISON: 'research' vs 'build' contexts")
    print("=" * 70)
    research_scores = patched_advisories["research"]
    build_scores = patched_advisories["build"]

    # Show nodes where context makes the biggest difference
    all_nodes = set(research_scores) & set(build_scores)
    diffs = {}
    for node in all_nodes:
        diff = research_scores.get(node, 0) - build_scores.get(node, 0)
        diffs[node] = diff

    print("\nNodes MOST favored by 'research' context vs 'build':")
    for node, diff in sorted(diffs.items(), key=lambda x: -x[1])[:5]:
        print(f"  {node:30s}: research={research_scores[node]:+.4f}  "
              f"build={build_scores[node]:+.4f}  diff={diff:+.4f}")

    print("\nNodes MOST favored by 'build' context vs 'research':")
    for node, diff in sorted(diffs.items(), key=lambda x: x[1])[:5]:
        print(f"  {node:30s}: research={research_scores[node]:+.4f}  "
              f"build={build_scores[node]:+.4f}  diff={diff:+.4f}")

    # VERDICT
    print()
    print("=" * 70)
    if p_var > 0.001 and o_var < 0.001:
        print("VERDICT: FIX CONFIRMED — Energy coupling creates context-sensitive routing")
        print(f"  Patched variance: {p_var:.6f} (context-differentiated)")
        print(f"  Original variance: {o_var:.6f} (context-blind)")
        print(f"  Improvement: {p_var / max(o_var, 1e-9):.1f}x more context-sensitive")
    elif p_var > o_var * 2:
        print("VERDICT: FIX WORKS — Patched engine is significantly more context-sensitive")
        print(f"  Patched variance: {p_var:.6f}")
        print(f"  Original variance: {o_var:.6f}")
    else:
        print("VERDICT: FIX INSUFFICIENT — Energy coupling did not create meaningful differentiation")
    print("=" * 70)

    return p_var, o_var, b_var


# ============================================================================
# E5-FIXED: Weight Distribution with Energy-Coupled Routing
# ============================================================================

def test_e5_fixed():
    print()
    print("=" * 70)
    print("E5-FIXED: Weight Distribution — Energy-Coupled Learning")
    print("=" * 70)
    print()

    patched = PatchedTorusfieldEngine.boot_aegis(seed=42, delta=0.40)
    original = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)

    patched.run(1000)
    original.run(1000)
    baseline.run(1000)

    def weight_stats(engine_name, edges):
        weights = [e.w for e in edges]
        nonzero = [w for w in weights if abs(w) > 0.001]
        if len(nonzero) < 4:
            print(f"  {engine_name}: insufficient non-zero edges ({len(nonzero)})")
            return {}
        q1 = sorted(nonzero)[len(nonzero) // 4]
        q3 = sorted(nonzero)[3 * len(nonzero) // 4]
        stats = {
            "nonzero": len(nonzero),
            "mean": statistics.mean(nonzero),
            "stdev": statistics.stdev(nonzero),
            "iqr": q3 - q1,
            "range": max(nonzero) - min(nonzero),
        }
        print(f"  {engine_name:20s}: nonzero={stats['nonzero']:3d}  "
              f"mean={stats['mean']:+.4f}  stdev={stats['stdev']:.4f}  "
              f"iqr={stats['iqr']:.4f}  range={stats['range']:.4f}")
        return stats

    p_stats = weight_stats("Patched", patched.graph.edges)
    o_stats = weight_stats("Original Torus", original.graph.edges)
    b_stats = weight_stats("Baseline", baseline.graph.edges)

    print()
    if p_stats and o_stats and b_stats:
        p_iqr = p_stats["iqr"]
        o_iqr = o_stats["iqr"]
        b_iqr = b_stats["iqr"]
        if p_iqr > o_iqr * 1.15:
            print(f"VERDICT: Energy-coupled routing improves weight separation "
                  f"({((p_iqr / o_iqr) - 1) * 100:.1f}% wider IQR)")
        elif abs(p_iqr - o_iqr) < 0.005:
            print("VERDICT: Energy coupling does not change weight distribution")
        else:
            print(f"VERDICT: Marginal difference (IQR: patched={p_iqr:.4f} "
                  f"vs original={o_iqr:.4f})")

    return p_stats, o_stats, b_stats


# ============================================================================
# E6-FIXED: Trajectory Divergence — Patched vs Original vs Baseline
# ============================================================================

def test_e6_fixed():
    print()
    print("=" * 70)
    print("E6-FIXED: Trajectory Divergence — Three-Way Comparison")
    print("=" * 70)
    print()

    patched = PatchedTorusfieldEngine.boot_aegis(seed=42, delta=0.40)
    original = TorusfieldEngine.boot_aegis(seed=42)
    baseline = BaselineEngine.boot_aegis(seed=42)

    p_path = []
    o_path = []
    b_path = []

    po_agree = []  # patched vs original
    pb_agree = []  # patched vs baseline
    ob_agree = []  # original vs baseline

    for i in range(500):
        p = patched.step_theta()
        o = original.step_theta()
        b = baseline.step()

        p_path.append(p)
        o_path.append(o)
        b_path.append(b)

        po_agree.append(p == o)
        pb_agree.append(p == b)
        ob_agree.append(o == b)

    def window_stats(agreements, label):
        total = sum(agreements)
        pct = total / len(agreements) * 100
        windows = {}
        for start in [0, 100, 200, 300, 400]:
            end = min(start + 100, len(agreements))
            w = sum(agreements[start:end])
            windows[f"{start}-{end}"] = w
        print(f"  {label:25s}: {total}/{len(agreements)} ({pct:.1f}%)")
        for k, v in windows.items():
            print(f"    steps {k}: {v}/100")
        return pct

    print("Trajectory agreement over 500 steps:")
    po_pct = window_stats(po_agree, "Patched vs Original")
    print()
    pb_pct = window_stats(pb_agree, "Patched vs Baseline")
    print()
    ob_pct = window_stats(ob_agree, "Original vs Baseline")

    # Node frequency comparison
    print()
    print("Top-5 most visited nodes:")

    def top_nodes(path, label):
        freq = {}
        for n in path:
            freq[n] = freq.get(n, 0) + 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:5]
        print(f"  {label}: {', '.join(f'{n}({c})' for n, c in top)}")

    top_nodes(p_path, "Patched ")
    top_nodes(o_path, "Original")
    top_nodes(b_path, "Baseline")

    print()
    print(f"VERDICT: Patched-vs-Original agreement: {po_pct:.1f}%")
    if po_pct < 80:
        print("  → Energy coupling SIGNIFICANTLY changes routing behavior")
    elif po_pct < 95:
        print("  → Energy coupling creates MODERATE routing differences")
    else:
        print("  → Energy coupling has MINIMAL effect on routing")

    return po_pct, pb_pct, ob_pct


# ============================================================================
# DELTA SENSITIVITY: What's the right coupling strength?
# ============================================================================

def test_delta_sweep():
    print()
    print("=" * 70)
    print("DELTA SWEEP: Finding optimal energy coupling strength")
    print("=" * 70)
    print()

    deltas = [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    contexts = {
        "research": ["web_search", "web_fetch", "browser", "foundry_research"],
        "build": ["write", "exec", "foundry_implement", "foundry_write_skill"],
        "monitor": ["session_status", "foundry_metrics", "foundry_overseer"],
    }
    node_coords = build_node_coordinate_map(AEGIS_CAPABILITY_MANIFEST, 24, 32)

    print(f"{'delta':>6s}  {'variance':>10s}  {'unique_top1':>11s}  "
          f"{'research_#1':>15s}  {'build_#1':>15s}  {'monitor_#1':>15s}")
    print("-" * 80)

    for delta in deltas:
        eng = PatchedTorusfieldEngine.boot_aegis(seed=42, delta=delta)
        eng.run(300)
        eng.cursor_node = "read"

        advisories = {}
        for ctx_name, caps in contexts.items():
            saved = eng.state.fields["energy"][:]
            for cap in caps:
                if cap in node_coords:
                    th, ph = node_coords[cap]
                    eng.inject_event(f"ctx_{cap}", 2.0, th, ph, "energy")
            for _ in range(15):
                eng.state.diffuse("energy", eng.theta_mix, eng.phi_mix)
                eng.state.decay(eng.leak_decay)

            router = EnergyCoupledRouter(delta=delta)
            scored = router.score_outgoing(
                eng.graph, eng.policy, "read",
                state=eng.state, node_coords=node_coords)
            advisories[ctx_name] = {s["to"]: s["score"] for s in scored}
            eng.state.fields["energy"] = saved

        # Measure variance
        ctx_names = list(contexts.keys())
        pairs = []
        for i in range(len(ctx_names)):
            for j in range(i + 1, len(ctx_names)):
                sim = cosine_similarity(
                    advisories[ctx_names[i]], advisories[ctx_names[j]])
                pairs.append(1.0 - sim)
        variance = statistics.mean(pairs)

        # Top-1 per context
        top1 = {}
        for ctx, scores in advisories.items():
            top1[ctx] = max(scores, key=scores.get)
        unique = len(set(top1.values()))

        print(f"{delta:6.2f}  {variance:10.6f}  {unique:11d}  "
              f"{top1['research']:>15s}  {top1['build']:>15s}  "
              f"{top1['monitor']:>15s}")

    print()
    print("delta=0.00 is the original (broken) behavior.")
    print("We want: high variance + different top-1 picks per context.")


# ============================================================================
# RUN ALL
# ============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  POST-FIX EXPERIMENTS: Energy-Coupled Router                ║")
    print("║  Does connecting energy to routing fix E3?                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    e3_result = test_e3_fixed()
    e5_result = test_e5_fixed()
    e6_result = test_e6_fixed()
    test_delta_sweep()

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print("E3 (Context Sensitivity):")
    print(f"  Patched variance:  {e3_result[0]:.6f}")
    print(f"  Original variance: {e3_result[1]:.6f}")
    print(f"  Baseline variance: {e3_result[2]:.6f}")
    if e3_result[0] > 0.001:
        print("  → ENERGY COUPLING WORKS: Routing is now context-sensitive")
    print()
    print("E5 (Weight Distribution): See above")
    print()
    print("E6 (Trajectory Divergence):")
    print(f"  Patched vs Original: {e6_result[0]:.1f}% agreement")
    print(f"  Patched vs Baseline: {e6_result[1]:.1f}% agreement")
    print(f"  Original vs Baseline: {e6_result[2]:.1f}% agreement")
