# UCS Torusfield Validation: Complete Experimental Record

**Date:** 2026-02-26
**Investigator:** Claude (Anthropic), directed by William Kyle Million (~K¹)
**System Under Test:** Unified Cognitive Substrate v1.1
**Format:** Emergent Judgment — Experiment Log

---

## The Question

Does the toroidal energy surface add measurable value over plain edge
reinforcement learning + trace-log artifact detection?

## Three-Phase Investigation

### Phase 1: Original Torus vs Stripped Baseline (6 experiments)

Built a `BaselineEngine` with identical graph, edges, harvester, and 
reinforcement formula — but no energy surface, no diffusion, no coherence 
reward.

| Experiment | Finding |
|---|---|
| E1: Edge Weight Convergence | MINOR DIVERGENCE — r=0.841, mean diff 0.012 |
| E2: Artifact Detection Parity | DIVERGENT artifacts (Jaccard 0.38) — different paths, not better detection |
| E3: Context Sensitivity | **CRITICAL** — Neither engine differentiates by context. Energy disconnected from router. |
| E4: Energy Persistence | Energy persists but is short-term memory (~100 steps at 0.995 decay) |
| E5: Reward Quality | NO DIFFERENCE — IQR 0.055 vs 0.052 |
| E6: Trajectory Divergence | 66.4% agreement — chaotic divergence from small reward differences |

**Key Finding (E3):** `Router.pick()` formula is `alpha*u - beta*c + gamma*n + w + bias`. 
Energy is not a term. The entire injection/diffusion pipeline writes to a surface that routing never reads.

### Phase 2: Delta Fix — Connect Energy to Router (1-line change)

Added `+ delta * energy_at_destination` to Router.pick().

| Metric | Original | Delta-Fix |
|---|---|---|
| Advisory variance | 0.000000 | 0.000020 |
| Unique #1 across 5 contexts | 1 | 1 |
| Non-zero edges after 1000 steps | 179 | **37** (collapsed) |
| Trajectory: read visits / 500 | 54 | **213** (stuck in loop) |

**Why it failed:** Three structural problems exposed:
1. Injection schedule creates permanent energy hotspots (`foundry_get_insights` at 33x other nodes)
2. Energy diffusion spreads context signal uniformly, diluting it
3. Torus coordinates are semantically arbitrary — diffusion goes to topological neighbors, not semantic ones

The delta fix made things WORSE — exploration collapsed to a 2-node loop.

### Phase 3: Option B — Separate Context Energy Field ✅

Isolated context energy from schedule energy. Fresh `context_energy` field 
zeroed per-query, populated only with context-relevant injections, read by 
router via delta term.

| Metric | Original | Delta-Fix | **Option B** |
|---|---|---|---|
| Advisory variance | 0.000000 | 0.000020 | **0.031266** |
| Unique #1 across 5 contexts | 1 | 1 | **3** |
| Improvement over original | — | 1x | **1,563x** |

**Context-sensitive routing achieved:**

| Context | #1 Recommendation | Context Boost |
|---|---|---|
| research | **web_fetch** | +0.65 (web_fetch, foundry_research) |
| build | **write** | +0.65 (write) |
| monitor | **foundry_metrics** | +0.65 (foundry_metrics, foundry_overseer) |
| communicate | write | No adjacency for targets |
| automate | write | No adjacency for targets |

Research context → recommends web_fetch (correct).
Build context → recommends write (correct).
Monitor context → recommends foundry_metrics (correct).

**Diffusion sweep** shows context differentiation is robust across all 
diffusion levels (0-25 steps), with maximum differentiation at 0 steps 
(variance 0.115) degrading gracefully to 0.002 at 25 steps. All levels 
maintain 3 distinct top-1 picks.

---

## Verdict

The torus energy surface **CAN** provide context-sensitive routing, but 
only with a separated context energy field. The current architecture 
(shared energy field for schedule + context) **DOES NOT** work.

### What Works Today (No Changes Needed)
- Edge reinforcement learning (weight accumulation across sessions)
- Artifact detection (wormholes, attractors, resonances from trace log)
- Policy kernel feedback (artifacts → edge biases)
- PP health tracking
- Bridge keyword-to-capability mapping
- Methodology index and annotation store
- Negative knowledge framework
- Flush/resume compaction survival

### What Needs the Fix (v1.2 Implementation)
- Add `context_energy` field to TorusState (initialized to zeros)
- `bridge.consult()` zeros context_energy, injects at resolved capabilities
- Brief diffusion (0-5 steps) on context_energy only
- Router reads `delta * context_energy_at_dst` in scoring formula
- Injection schedule continues using main `energy` field (unchanged)
- Executor coherence continues using main `energy` field (unchanged)

### Estimated Implementation
- Kernel changes: ~30 lines (add field, modify Router, add zero method)
- Bridge changes: ~15 lines (consult() populates context_energy instead of energy)
- Total: ~45 lines changed

---

## Negative Knowledge Log

### Dead End: Energy Connected to Router via Shared Field
Connecting energy to routing while sharing a field with the injection 
schedule creates degenerate attractors. `foundry_get_insights` accumulated
33x more energy than other nodes, collapsing exploration to a 2-node loop.
**Do not attempt this configuration.**

### Dead End: High Diffusion for Context Spreading
More diffusion steps = more dilution of context signal. At 25 steps,
variance drops to 0.002 (from 0.115 at 0 steps). Diffusion spreads 
energy to topological neighbors, but torus coordinates are semantically
arbitrary, so diffusion doesn't help reach the right nodes.
**Optimal: 0-3 diffusion steps for context energy.**

### Open Question: Semantic Coordinate Assignment
If capabilities were clustered by semantic role on the torus (research
tools adjacent, build tools adjacent), diffusion WOULD propagate context
signal to useful neighbors. This could further improve differentiation.
**Reopen condition:** If 3 unique top-1 picks is insufficient for 
production use, test semantic clustering.

### Open Question: communicate/automate Context Failure
These contexts show no differentiation because their target capabilities
(`message`, `tts`, `foundry_write_hook`) are not in the adjacency list
from the test cursor ("read"). This is correct behavior — you can't boost
destinations that aren't reachable. But it means context sensitivity
depends on cursor position, which should be documented.
**Reopen condition:** If users report "consult gives same answer for 
different tasks," check whether cursor is at a node with adjacency to
the context-relevant capabilities.

---

## Files Produced

| File | Purpose |
|---|---|
| `baseline_engine.py` | Stripped-down engine (no torus) for comparison |
| `torus_vs_baseline.py` | Phase 1: 6-experiment comparison (E1-E6) |
| `EXPERIMENT_LOG.md` | Phase 1 findings |
| `patched_engine.py` | Delta-fix: energy-coupled router |
| `postfix_experiments.py` | Phase 2: delta-fix results |
| `POSTFIX_EXPERIMENT_LOG.md` | Phase 2 findings |
| `option_b_test.py` | Phase 3: separate context field (the fix that works) |
| `COMPLETE_FINDINGS.md` | This document |
