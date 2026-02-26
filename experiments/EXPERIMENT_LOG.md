# Experiment Log: Torusfield vs Baseline — Does the Torus Add Value?

**Date:** 2026-02-26
**Author:** Claude (Anthropic), at Kyle Million's direction
**Framework:** UCS v1.1 / IntuiTek¹
**Format:** Emergent Judgment Section 4 — Experiment Logging

---

## Hypothesis

The toroidal energy surface (energy injection, diffusion, coherence-based
reward, θ/φ cycling) adds measurable routing quality over a plain directed
graph that uses the same edge reinforcement formula and the same trace-log
artifact detection — but without any energy field.

## Change

Built `BaselineEngine`: identical graph topology (45 nodes, 249 edges),
identical `Harvester`, `ArtifactStore`, `PolicyKernel`, `PPEngine`,
identical reinforcement formula (`edge.w += lr * (net - 0.5)`). Removed:
energy surface, diffusion, injection, coherence-based executor reward.
Replaced executor with fixed `reward = utility_weight * edge.u`.

## Measurements (6 Experiments)

### E1 — Edge Weight Convergence
| Metric | Value |
|---|---|
| Mean absolute weight difference | 0.0124 |
| Correlation | 0.841 |
| Sign disagreements | 0 |
| Max single-edge difference | 0.118 |

**Reading:** The coherence term creates small but real reward differences.
No edge learned the *wrong sign* (all agree on positive/negative), but
magnitudes drift apart over 500 steps. The torus isn't teaching
fundamentally different lessons — it's adding noise to the same signal.

### E2 — Artifact Detection Parity
| Artifact Kind | Torus | Baseline |
|---|---|---|
| Wormholes | 8 | 10 |
| Attractors | 2 | 2 |
| Resonances | 1 | 0 |
| **Total** | **11** | **12** |

| Overlap Metric | Jaccard |
|---|---|
| Wormhole edge overlap | 0.385 |
| Attractor node overlap | 0.000 |

**Reading:** The artifact *counts* are similar, but the *specific* artifacts
diverge substantially. This is NOT because the detection algorithm behaves
differently — it's because the engines walk different paths (see E6), so
the trace logs contain different edge/node frequency patterns. The
harvester finds different patterns because it's fed different data.

**Critical nuance:** This doesn't mean the torus *improves* artifact
detection. It means the torus *changes which artifacts get detected* by
changing the exploration trajectory. Whether these are better or worse
artifacts is not measurable without ground truth.

### E3 — Context Sensitivity of Routing Advisory ⚠️ KEY FINDING
| Metric | Torus | Baseline |
|---|---|---|
| Advisory variance across 5 contexts | 0.0 | 0.0 |
| Context-differentiated? | **NO** | **NO** |

**Reading:** This is the most important finding. The `Router.pick()` scoring
formula is:

```
score = alpha * edge.u - beta * edge.c + gamma * edge.n + edge.w + policy_bias
```

**Energy is not a term in this formula.** The energy field enters the system
*only* through the `Executor.run()` coherence calculation, which affects
the *reward signal* used for *future* edge weight updates. It does NOT
affect the *current* routing advisory.

This means:
- `bridge.consult()` injects energy at relevant capabilities and diffuses
  for 15 steps — but this energy injection **has zero effect on the advisory
  scores returned to the agent**
- The context sensitivity that `consult()` provides comes entirely from
  the bridge's `resolve_capabilities_from_context()` keyword mapping and
  the methodology index lookup — **NOT from the torus**
- The torus energy injection is writing to a surface that nothing reads
  during advisory generation

**This is an architectural gap, not a design flaw that can be fixed by
tuning parameters. The router would need to be modified to include an
energy-at-destination term in its scoring formula for context injection
to matter.**

### E4 — Energy Field Persistence Value
| Metric | Trained (500 steps) | Fresh |
|---|---|---|
| Non-zero cells | 768 | 0 |
| Total energy | 5,436.7 | 0.0 |
| Max amplitude | 27.76 | 0.0 |

| Decay Analysis | |
|---|---|
| Per-step decay rate | 0.995 |
| Retention after 500 steps | 8.2% |

**Reading:** The energy field *does* persist and *does* carry information.
But at 0.995 decay per step, it's dominated by the most recent ~100 steps
of injection. Historical energy (from early usage sessions) decays to
negligibility. In boot-per-call mode where the engine runs 15-step warmups,
the persisted energy from the *previous session's* warmup will have decayed
by `0.995^15 ≈ 0.927` — retaining 92.7% — which is meaningful. But energy
from two sessions ago, after two warmup cycles, retains `0.995^30 ≈ 0.860`.
After 10 sessions: `0.995^150 ≈ 0.472`. The energy field is a
**short-term memory**, not a long-term one. Edge weights are the actual
long-term memory.

### E5 — Coherence-Based vs Fixed Reward Signal Quality
| Metric | Torus | Baseline |
|---|---|---|
| Non-zero edges | 179 | 169 |
| Weight mean | -0.049 | -0.042 |
| Weight stdev | 0.125 | 0.091 |
| IQR | 0.055 | 0.052 |
| Range | 1.356 | 1.036 |

**Reading:** The distributions are nearly identical. The torus has slightly
higher variance (stdev 0.125 vs 0.091), but the IQR — which measures the
bulk of the distribution — differs by only 0.003. The coherence reward
signal does not produce meaningfully more discriminating edge weights
than the fixed reward signal.

The wider *range* in the torus (1.356 vs 1.036) comes from a single
outlier edge driven to -1.156. This is the coherence term amplifying
a penalty on one edge — not systematic improvement.

### E6 — Trajectory Divergence Over Time
| Step Window | Agreement |
|---|---|
| 0–100 | 100/100 (100%) |
| 100–200 | 78/100 (78%) |
| 200–300 | 100/100 (100%) |
| 300–400 | 37/100 (37%) |
| 400–500 | 17/100 (17%) |
| **Overall** | **332/500 (66.4%)** |

First divergence at step 132. Agreement collapses to 17% by the last window.

**Reading:** This is the chaotic divergence pattern expected from any
system where small reward differences compound through reinforcement.
The coherence term creates slightly different rewards → slightly different
weights → different routing choices → different traces → different rewards
→ accelerating divergence. This is *sensitivity to initial conditions*,
not evidence that the torus trajectories are *better*.

The pattern of 100% → 78% → 100% → 37% → 17% (the 200-300 re-convergence)
is characteristic of softmax routing over a small graph: when two engines
are near the same high-reward attractor basin, they converge; when small
weight differences push them into different basins, they diverge rapidly.

---

## Verdict

### What the torus DOES:
1. **Creates trajectory divergence** via coherence-modulated rewards (E6)
2. **Persists short-term energy patterns** across boot cycles (E4)
3. **Slightly increases weight variance** through coherence amplification (E5)

### What the torus does NOT do:
1. **Does NOT provide context-sensitive routing advisory** — energy is
   architecturally disconnected from the router scoring formula (E3)
2. **Does NOT improve edge weight discrimination** — IQR nearly identical (E5)
3. **Does NOT improve artifact detection** — different artifacts found, but
   only because different trajectories explored, not because detection is
   better (E2)

### The core question: "Does the torus add measurable value?"

**For the current architecture: No, not in a way that justifies the
complexity cost.**

The energy surface is ~400 lines of code (injection, diffusion, decay,
coherence calculation, field persistence) producing effects that are either:
- Unmeasurable (E5: same weight distributions)
- Disconnected from the mechanism they should serve (E3: energy doesn't
  enter routing)
- Indistinguishable from noise (E6: trajectory divergence could be from
  any perturbation)

### But — the torus COULD add value with one architectural change:

If the `Router.pick()` formula were modified to include an
energy-at-destination term:

```python
# Current (energy disconnected):
score = alpha * u - beta * c + gamma * n + w + bias

# Proposed (energy connected):
energy_at_dst = self.state.read("energy", dst_theta, dst_phi)
score = alpha * u - beta * c + gamma * n + w + bias + delta * energy_at_dst
```

This would make `bridge.consult()`'s energy injection actually influence
the advisory — which is what the architecture *appears to intend* but
does not currently implement. With this change, injecting energy at
"web_search" capabilities before routing would actually boost scores
for edges leading toward web_search, creating genuine context sensitivity.

**This is the experiment that should be run next.**

---

## Negative Knowledge (EJ Section 3)

**Dead End:** Assuming the torus energy field influences routing because
it *exists in the same engine*. It doesn't. The coupling is only through
rewards, which is a one-step-delayed, heavily-damped channel.

**Reopen Condition:** If Router.pick() is modified to include energy
readings, re-run E3 and E5 to test whether context sensitivity and
weight discrimination improve.

---

## Recommendation

**Short term:** The UCS bridge's `consult()` operation should be honest
about where its advisory comes from. Currently, it injects energy (which
does nothing for advisory) and then returns edge scores (which are
context-independent from the engine's perspective). The context sensitivity
users experience comes from the bridge's keyword-to-capability mapping
and methodology index — which are valuable and should be preserved.

**Medium term:** Implement the energy-coupled router and re-run this
experiment suite. If E3 then shows context differentiation and E5 shows
improved discrimination, the torus has earned its complexity. If not,
the energy surface should be simplified out.

**What to keep regardless:** Edge reinforcement, artifact detection
(harvester + store), policy kernel feedback, PP health tracking. These
all work identically with or without the energy surface and constitute
the proven value of the system.
