# Post-Fix Experiment Log: Energy-Coupled Router Results

**Date:** 2026-02-26
**Predecessor:** EXPERIMENT_LOG.md (E1-E6 pre-fix results)
**Fix Applied:** Added `delta * energy_at_destination` to Router.pick() formula

---

## What We Expected

Energy coupling would make consult()'s energy injection at relevant 
capabilities actually influence the routing advisory, creating different
top recommendations for different task contexts.

## What Actually Happened

### E3-FIXED: Context Sensitivity
| Metric | Original | Patched | Change |
|---|---|---|---|
| Advisory variance | 0.000000 | 0.000020 | Technically non-zero |
| Unique #1 picks across 5 contexts | 1 | 1 | **No change** |

The variance went from zero to 0.00002. That's technically non-zero but 
effectively meaningless. More importantly: **the #1 recommendation never
changes across contexts at ANY delta value from 0.0 to 1.0.**

### Why: Three Deeper Problems Exposed

**Problem 1: Energy Diffusion Dilutes Context Signal**

After 15 diffusion steps on a 24×32 torus, energy injected at 4 specific
capabilities has spread across the surface. The differential between a
"research-relevant node" and an "irrelevant node" is ~0.2 on scores
ranging from 8 to 33. The context signal is drowned by baseline scores.

**Problem 2: Injection Schedule Creates Permanent Hotspots**

The engine's injection_schedule periodically injects energy at ALL
capabilities. `foundry_get_insights` accumulates a score of 33.05 —
3x higher than any other node. Context injection of 2.0 at 4 nodes
can't compete with persistent schedule-driven energy. The patched engine
visits `foundry_get_insights` 201 out of 500 steps — it's locked in a
degenerate attractor.

**Problem 3: Torus Coordinates Are Semantically Arbitrary**

Capabilities are assigned coordinates by: `theta = index % 24`,
`phi = (index * 3) % 32`. This means `web_search` (a research tool) and
`foundry_write_hook` (an automation tool) might be adjacent on the torus
while `web_search` and `web_fetch` (both research tools) are far apart.
Energy diffusion propagates to *topological* neighbors, not *semantic*
neighbors. Diffusion doesn't help because the topology doesn't encode 
semantic relationships.

### E5-FIXED: Weight Distribution (WORSE)
| Metric | Original | Patched |
|---|---|---|
| Non-zero edges | 179 | **37** |
| IQR | 0.055 | **0.030** |

The patched engine learned FEWER edges because it explored fewer paths.
The energy hotspot at `foundry_get_insights` collapsed exploration.
Energy coupling in its current form doesn't improve learning — it
degrades it.

### E6-FIXED: Trajectory Divergence
| Comparison | Agreement |
|---|---|
| Patched vs Original | **7.8%** |
| Patched vs Baseline | **8.8%** |
| Original vs Baseline | 66.4% |

The patched engine takes radically different paths — but WORSE ones.
`read(213), foundry_get_insights(201)` = stuck in a 2-node loop.

### Delta Sweep: No Value at Any Coupling Strength
| Delta | Unique #1 | Research #1 | Build #1 | Monitor #1 |
|---|---|---|---|---|
| 0.00 | 1 | write | write | write |
| 0.10 | 1 | foundry_get_insights | foundry_get_insights | foundry_get_insights |
| 0.20 | 1 | foundry_metrics | foundry_metrics | foundry_metrics |
| 0.40 | 1 | foundry_get_insights | foundry_get_insights | foundry_get_insights |
| 0.60 | 1 | foundry_get_insights | foundry_get_insights | foundry_get_insights |
| 0.80 | 1 | foundry_get_insights | foundry_get_insights | foundry_get_insights |
| 1.00 | 1 | foundry_get_insights | foundry_get_insights | foundry_get_insights |

At delta=0: all contexts recommend "write" (edge-weight driven).
At delta>0: all contexts recommend "foundry_get_insights" (energy-hotspot driven).
**No delta value produces context-differentiated routing.**

---

## Diagnosis

The one-line fix was necessary but not sufficient. It exposed that the
energy architecture has three coupled failures:

1. **Diffusion without semantic topology** = noise, not signal
2. **Shared energy field for schedule + context** = permanent hotspots
   dominate transient context injections  
3. **Uniform injection magnitude** = context injection can't overcome
   schedule-accumulated energy

## What Would Actually Fix It (Three Options)

### Option A: Semantic Coordinate Assignment (Medium Effort)
Cluster semantically related capabilities together on the torus so that
diffusion propagates to useful neighbors:
- Research cluster: web_search, web_fetch, browser, foundry_research
- Build cluster: write, exec, edit, foundry_implement
- Monitor cluster: session_status, foundry_metrics, foundry_overseer

This makes diffusion meaningful — energy injected at web_search would
spread to web_fetch and browser, not to random unrelated capabilities.

### Option B: Separate Context Field (Low Effort)
Add a `context_energy` field separate from the injection schedule's
`energy` field. Router reads context_energy (which is fresh per-consult)
while ignoring schedule energy. This eliminates hotspot contamination.

### Option C: Simplify — Remove Energy, Keep What Works (Lowest Effort)
Accept that the torus energy surface doesn't earn its complexity.
Remove it. Keep: edge RL, artifact detection, policy kernel, PP tracking,
and the bridge's keyword-to-capability mapping + methodology index.
These are the components that produce measurable value.

The bridge's consult() already provides context sensitivity through its
resolve_capabilities_from_context() function and methodology index lookup.
The energy surface was supposed to add a second layer of context awareness,
but it doesn't. The bridge layer is doing the real work.

---

## Negative Knowledge Update

**Dead End (CONFIRMED):** Simply connecting energy to routing (delta fix)
does not produce context-sensitive advisory. The problems are structural
(arbitrary topology, shared field, diffusion dilution), not parametric.

**Dead End (NEW):** Energy coupling with the current injection schedule
creates degenerate attractors. `foundry_get_insights` at score 33 vs
everything else at 4-10 collapses exploration to a 2-node loop.

**Reopen Conditions:**
- Option A (semantic coordinates): Test whether clustering capabilities
  by semantic role on the torus produces context-differentiated routing
- Option B (separate field): Test whether a dedicated context_energy
  field, populated only by consult(), produces different #1 picks for
  different contexts
- If neither A nor B produces unique #1 picks: the torus energy surface
  should be removed per Option C

---

## What This Means for UCS

The Unified Cognitive Substrate's value proposition has two layers:

**Layer 1 (PROVEN):** Edge reinforcement learning + artifact detection +
methodology preservation + negative knowledge + structured reflection.
All of this works. The bridge's consult(), report(), reflect(), flush(),
resume() operations provide genuine value through the keyword-capability
mapping, methodology index, and annotation store.

**Layer 2 (UNPROVEN):** Toroidal energy dynamics providing context-aware
routing modulation. This layer does not currently work as designed. The
fix attempted here confirmed that the problem is architectural, not a
missing connection.

**Recommendation:** Ship with Layer 1. Pursue Option B (separate context
field) as the next experiment. If that doesn't produce context
differentiation, accept Option C and simplify.

The system's novelty — the fusion of quantitative routing with qualitative
metacognitive preservation — does not depend on the torus being elegant.
It depends on the bridge being effective. The bridge is effective.
