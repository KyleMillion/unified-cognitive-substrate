# Unified Cognitive Substrate (UCS) v1.2
**Closes the Judgment Gap in Persistent AI Agents**

Toroidal routing engine + separated context energy field + Emergent Judgment Protocol + reflect/flush/resume loop.

**Live Landing Page**
https://kylemillion.github.io/unified-cognitive-substrate/ucs/

**Paper**
https://doi.org/10.5281/zenodo.18794692

**Validation**
[VALIDATION.md](VALIDATION.md) — 17/17 tests, 3-phase investigation, compaction survival

**Start Here (run in <2 min)**
```bash
python -m pytest tests/test_ucs.py -q --tb=no
```

17/17 tests pass • 1,563× advisory differentiation • <100 ms boot  
First public release: February 27, 2026  
Built by **William Kyle Million (~K¹)**, IntuiTek¹ • MIT License

---

# Unified Cognitive Substrate (UCS)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18794175.svg)](https://doi.org/10.5281/zenodo.18794175)

**Judgment Preservation in Persistent AI Agents**

*By William Kyle Million (~K¹) — IntuiTek¹*

---

## The Problem

Persistent AI agents lose something when their context is compacted. Not facts — those can be stored and retrieved. Not memories — those can be logged and searched. What's lost is **judgment**: the reasoning texture that emerges from sustained engagement with a problem.

An agent 40 messages deep in a debugging session has accumulated dozens of eliminated hypotheses, implicit knowledge about which approaches don't work, and reasoning chains connecting observations. When that session is compacted, the summary says "the bug was in X." The expertise that found it is gone.

The agent can't perceive this loss. The summary feels complete. This is the **introspection gap** — and it's why passive preservation fails.

Current agent memory systems ([Mem0](https://arxiv.org/abs/2504.19413), [MemGPT/Letta](https://arxiv.org/abs/2310.08560), [Field-Theoretic Memory](https://arxiv.org/abs/2602.21220)) manage information. They don't preserve judgment.

The ["Memory in the Age of AI Agents" survey](https://arxiv.org/abs/2512.13564) (Dec 2025, 40+ authors) — the most comprehensive mapping of this space — categorizes agent memory by form, function, and dynamics. **Judgment is not a category in their taxonomy.** We propose that it should be.

## The Solution

UCS fuses two systems into a single deployable skill:

**Quantitative Layer — Toroidal Routing Engine**
- Models agent capabilities as nodes in a directed graph (45 nodes, 249 edges)
- Learns which capability transitions produce value through edge reinforcement
- Detects structural patterns: wormholes (high-value edges), attractors (frequently visited nodes), resonances (positive-return cycles)
- Context-sensitive routing via separated energy field (v1.2 — validated by experiment)

**Qualitative Layer — Emergent Judgment Protocol**
- Post-task reflection: captures initial signal, hypothesis, near miss, generalized pattern, negative knowledge
- Structured externalization before compaction: hypotheses, reasoning chains, open questions, confidence levels
- Knowledge architecture with provenance tagging and temporal tiering
- **Negative knowledge framework**: documented dead ends with explicit conditions for reopening

**Translation Bridge**
- Annotation index connecting quantitative artifacts to qualitative explanations
- 8 operations: `init`, `consult`, `report`, `reflect`, `flush`, `resume`, `synthesize`, `status`
- Boot-per-call execution (<100ms) — works with any agent platform
- Advisory, not directive: the agent stays in control

## Validation

We conducted an **11-experiment empirical investigation** comparing the full system against a stripped-down baseline (plain edge RL, no energy surface). Key findings:

| Phase | Finding |
|---|---|
| Phase 1 (E3) | **Energy field was architecturally disconnected from routing.** Router scoring formula didn't read energy. Context sensitivity came entirely from the bridge's keyword mapping. |
| Phase 2 | Direct energy coupling **failed** — injection schedule created permanent hotspots that collapsed exploration to a 2-node loop. |
| Phase 3 | **Separated context energy field works.** 1,563× improvement in advisory differentiation. 3/5 contexts get distinct top-1 recommendations. |

Full experiment data, including negative results: [`experiments/`](experiments/)

**Test suite: 17/17 passing** — including T15 (compaction survival) and T17 (context-sensitive routing).

## Quick Start

```bash
# Run the test suite
cd tests && python3 test_ucs.py

# Run the validation experiments
cd experiments && python3 torus_vs_baseline.py    # Phase 1: 6 experiments
cd experiments && python3 postfix_experiments.py  # Phase 2: delta fix
cd experiments && python3 option_b_test.py        # Phase 3: the fix that works
```

For agent integration, see [`docs/SKILL.md`](docs/SKILL.md) (operational manual) and [`docs/AGENT_README.md`](docs/AGENT_README.md) (self-constructing directive for AI agents).

## Repository Structure

```
unified-cognitive-substrate/
├── README.md                    ← You are here
├── LICENSE                      ← MIT
├── ATTRIBUTION.md               ← Full credit and provenance
├── VALIDATION.md                ← Experiment summary
│
├── src/
│   ├── torusfield_kernel.py     ← Toroidal routing engine (v1.2)
│   ├── bridge.py                ← UCS bridge (8 operations)
│   └── install.sh               ← Automated installer
│
├── tests/
│   └── test_ucs.py              ← 17 tests (all passing)
│
├── experiments/
│   ├── COMPLETE_FINDINGS.md     ← Full 3-phase experimental record
│   ├── EXPERIMENT_LOG.md        ← Phase 1 results
│   ├── POSTFIX_EXPERIMENT_LOG.md← Phase 2-3 results
│   ├── baseline_engine.py       ← Stripped-down comparison engine
│   ├── torus_vs_baseline.py     ← Phase 1: 6 experiments
│   ├── patched_engine.py        ← Phase 2: delta fix (failed)
│   ├── postfix_experiments.py   ← Phase 2: delta fix experiments
│   └── option_b_test.py         ← Phase 3: separated context field
│
├── paper/
│   ├── ucs_paper.pdf            ← Preprint
│   └── ucs_paper.tex            ← LaTeX source
│
└── docs/
    ├── SKILL.md                 ← Agent operational manual
    ├── BUILD_PLAN.md            ← Architecture specification
    └── AGENT_README.md          ← Self-constructing directive for AI agents
```

## Paper

**"Judgment Preservation in Persistent AI Agents: A Unified Cognitive Substrate for Routing Reinforcement and Metacognitive Continuity"**

William Kyle Million, IntuiTek¹. February 2026.

Available in [`paper/`](paper/).

## Prior Art & Positioning

| System | What It Does | How UCS Differs |
|---|---|---|
| [Reflexion](https://arxiv.org/abs/2303.11366) (NeurIPS 2023) | Verbal reflection within task retries | UCS preserves judgment *across sessions and compaction*. Reflexion doesn't survive compaction. |
| [MemGPT/Letta](https://arxiv.org/abs/2310.08560) | Virtual memory paging for context | Manages tokens. UCS manages judgment. |
| [Field-Theoretic Memory](https://arxiv.org/abs/2602.21220) | Field dynamics for memory retrieval | Applies field theory to *retrieval*. UCS applies toroidal topology to *routing*. |
| [ACC](https://arxiv.org/abs/2601.11653) | Compressed cognitive state | Compresses to one variable. UCS externalizes to structured files with provenance. |
| [Memory survey](https://arxiv.org/abs/2512.13564) (Dec 2025) | Comprehensive taxonomy | Their taxonomy doesn't include judgment. We propose it should. |

## License

MIT — See [LICENSE](LICENSE).

Every fork carries [ATTRIBUTION.md](ATTRIBUTION.md).

## Author

**William Kyle Million** (~K¹)
Founder, IntuiTek¹
Independent researcher and AI strategy consultant specializing in persistent agent systems.

The Emergent Judgment framework — the qualitative layer of UCS — is the first metacognitive skill for persistent AI agents, developed from years of work maintaining persistent AI systems across platform transitions.
