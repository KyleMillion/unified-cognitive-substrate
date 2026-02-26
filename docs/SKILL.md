---
name: unified-cognitive-substrate
version: 1.1.0
description: >
  Fused cognitive operating system for persistent AI agents. Combines Torusfield OS
  (quantitative capability routing via reinforcement learning on a toroidal graph) with
  Emergent Judgment (qualitative metacognitive preservation of reasoning and expertise).
  Trigger this skill at session start (resume), before significant tasks (consult), after
  every tool call or action (report), when outcomes are notable/critical (report + reflect),
  before compaction or session end (flush), and periodically for synthesis. Also trigger
  when the user says: "what did we learn", "write that down", "how did you know that",
  "update methodology", "retrospective", "do a synthesis", "check health", or discusses
  the agent's growth, self-improvement, judgment, or patterns.
  This skill replaces the need to separately manage Torusfield and Emergent Judgment —
  it is the unified interface for both.
license: MIT
author: >
  Architecture by William Kyle Million (~K¹), founder of IntuiTek¹.
  Bridge implementation through collaborative engineering between K¹ and Claude (Anthropic),
  February 2026.
changelog:
  v1.1.0: >
    Stable artifact IDs (wormholes/attractors no longer duplicate on evidence growth).
    PP health normalized to [0,1] across all six dimensions.
    Artifact staleness tracking — get_stale(), prune_stale(), surfaced in status and synthesize.
    Trace restoration on import_state (phi-cycle now detects patterns post-reload — core bug fix).
    Keyword map enriched dynamically from stored methodology at consult time.
    Auto-correlation: new wormhole promotions are automatically matched against methodology.
    Manifest as configuration — drop manifest.json in workspace root to deploy for any agent.
---

# Unified Cognitive Substrate (UCS)

## What This Is

You have two cognitive systems fused into one:

**Torusfield OS** learns *which capability transitions produce value* through
reinforcement on a toroidal graph of your entire tool inventory. It tracks
wormholes (validated shortcuts), attractors (capability basins you gravitate to),
and resonances (periodic execution rhythms). These compound across sessions.

**Emergent Judgment** preserves *why those transitions produce value* through
structured reflection, methodology files, experiment logs, negative knowledge,
and synthesis reports. It captures the reasoning texture that compaction destroys.

Together, they form a system that both routes intelligently AND understands its
own routing. The bridge exposes this through 8 operations you call via `exec`.

---

## Quick Reference: When to Call What

```
SESSION START
  └── exec("python bridge.py resume")
      → Read the session briefing, inject into your context

BEFORE ACTING (significant tasks)
  └── exec("python bridge.py consult --context 'task description'")
      → Factor the advisory into your reasoning (it suggests, not commands)

AFTER EACH ACTION
  └── echo '{"action":"...","outcome":"...","success":"...","significance":"..."}' |
      exec("python bridge.py report")
      → If result.reflection_needed == true → write the reflection

WHEN REFLECTING
  └── echo '{"type":"post_task","text":"...","capabilities":[...],"keywords":[...]}' |
      exec("python bridge.py reflect")

BEFORE COMPACTION / SESSION END
  └── exec("python bridge.py flush")
      → Write working-state files from the prompts it returns

PERIODIC (daily/weekly)
  └── exec("python bridge.py synthesize")
      → Write synthesis report from the data it returns

QUICK CHECK
  └── exec("python bridge.py status")
      → Includes stale_artifact_count in v1.1+
```

---

## Operation Details

### `resume` — Session Start

**When:** First operation of every session.

**What it returns:**
- PP health score and trend (improving/stable/declining) — all values in [0,1]
- Artifact inventory (wormholes, attractors, resonances with annotations)
- Your current cursor position in the capability graph
- Open questions from last session's working state
- Recent methodology entries
- Last synthesis summary
- Recommendations (unannotated artifacts, overdue synthesis, health warnings)

**What you do with it:** Read the briefing. It tells you where you left off,
what your system has learned, and what needs attention. Inject relevant parts
into your working context.

### `consult` — Before Acting

**When:** Before starting any significant task or when deciding between approaches.

**Input:** `--context "description of task"` and optionally `--capabilities cap1,cap2`

**What it returns:**
- Suggested paths from your current position, scored and annotated
- Active wormholes (validated shortcuts with qualitative explanations)
- Active attractors (capability basins you naturally gravitate to)
- Methodology hits (relevant past expertise matching this context)
  — v1.1: also enriched from methodology keywords learned through actual work
- Dead ends (things that have been tried and confirmed not to work)
- PP health and unannotated artifact count

**What you do with it:** Consider the advisory alongside the user's request and
your own reasoning. The advisory is one input — you are the decision-maker.
If you disagree with a suggestion, that's fine. The system learns from what
you actually do, not what it suggests.

### `report` — After Acting

**When:** After every tool call or significant action.

**Input (JSON on stdin):**
```json
{
  "action": "web_search",
  "outcome": "Found 3 relevant pricing pages with complete data",
  "success": "success",
  "significance": "notable"
}
```

- `action`: The capability name you used (must match a node in the graph)
- `outcome`: Free text description of what happened
- `success`: One of `success`, `partial`, `failure`, `neutral`
- `significance`: One of `routine`, `notable`, `critical`

**What it returns:**
- Reinforcement applied (which edge, old/new weight, reward signal)
- Whether a φ-cycle fired (artifact detection)
- Any newly promoted artifacts
- Whether a reflection is needed (based on significance)
- Reflection prompts (if needed)
- Unannotated artifacts needing attention
- v1.1: `methodology_correlations` when new wormholes are promoted — automatically
  matches new structural patterns against stored methodology so the quantitative
  and qualitative layers stay connected without requiring manual lookup

**What you do with it:** If `reflection_needed` is true, write a reflection
using the prompts and call `reflect`. If new artifacts appear with methodology
correlations, review the correlations and either annotate using the suggested
methodology entry or write a new annotation if none fit. For routine reports,
just read and continue.

### `reflect` — Storing Reflections

**When:** After `report` returns `reflection_needed: true`, when annotating
artifacts, when logging experiments, or when recording dead ends.

**Types:**

**post_task** — Methodology entry from post-task reflection:
```json
{
  "type": "post_task",
  "text": "### 2026-02-25 — Competitor pricing research\n\n**Initial Signal:** ...",
  "capabilities": ["web_search", "web_fetch"],
  "keywords": ["pricing", "evidence", "competitor"]
}
```

**annotation** — Qualitative explanation for a Torusfield artifact:
```json
{
  "type": "annotation",
  "artifact_id": "a3f2c8b1e9d0",
  "text": "This wormhole works because search snippets truncate pricing tables.",
  "failure_condition": "Does NOT apply to API documentation — snippets are sufficient there.",
  "generalized_pattern": "When the goal is evidence extraction, always fetch the full source."
}
```

**experiment** — Configuration change with measurement:
```json
{
  "type": "experiment",
  "text": "### 2026-02-25 — Increased consult warmup steps\n\n**Hypothesis:** ..."
}
```

**dead_end** — Confirmed closed avenue:
```json
{
  "type": "dead_end",
  "text": "### [DEAD END] API scraping — Rate limited\n\n**Date:** ...",
  "capabilities": ["browser", "web_fetch"],
  "keywords": ["API", "scraping", "rate limit"],
  "topic": "API-based price scraping",
  "why_closed": "Rate limited after 3 requests",
  "reopen_conditions": "If target adds public API or removes rate limits"
}
```

**synthesis** — Periodic synthesis report:
```json
{
  "type": "synthesis",
  "text": "## Synthesis — 2026-02-25\n\n### Patterns Emerging\n..."
}
```

### `flush` — Pre-Compaction Save

**When:** Before `/compact`, before session end, or when approaching context limits.

**What it returns:** Externalization prompts — structured questions about your
current thinking, reasoning chains, open questions, confidence levels, and
context dependencies.

**What you do with it:** Write your answers to the files indicated in the
response's `write_to` directory. These survive compaction and are loaded on
the next `resume`.

### `synthesize` — Gather Synthesis Material

**When:** End of day, end of week, or on user request.

**What it returns:** All raw material for a synthesis report:
- Resource allocation (which actions consumed time, success rates)
- PP trend data (normalized [0,1] in v1.1)
- Artifact summary with annotations
- Recent methodology entries
- Policy overrides (where you disagreed with the graph)
- Misalignments (high-traffic capabilities with no methodology)
- v1.1: `stale_artifacts` — artifacts not detected in recent phi cycles,
  candidates for pruning via `store.prune_stale()`
- Synthesis prompts (structured questions to guide your analysis)

**What you do with it:** Write a synthesis report addressing the prompts,
then store it via `reflect --type synthesis`. Review stale artifacts and
decide which to keep and which to prune.

### `status` — Quick Health Check

**When:** Anytime you want a fast read on system state.

**What it returns:** PP health (normalized [0,1]), cursor position, artifact count,
unannotated count, stale artifact count (v1.1), report counter, last synthesis date.

---

## Control Hierarchy

You are the decision-maker. The UCS advises. The hierarchy is:

1. **User intent** — The user's explicit request overrides everything
2. **Your reasoning** — You decide what to do, informed by the advisory
3. **UCS advisory** — Suggests paths and surfaces relevant expertise
4. **UCS reinforcement** — Learns from your actual decisions in the background

If you disagree with a routing suggestion, follow your judgment. The system
logs the disagreement and learns from the outcome.

---

## What the Artifacts Mean

- **Wormhole:** A validated shortcut. v1.1: each wormhole has a stable ID for
  its lifetime — as evidence accumulates, the payload updates in-place rather
  than creating duplicates. Annotate with why it works and when it doesn't.

- **Attractor:** A capability basin. You naturally gravitate here. v1.1: stable
  ID per node. If your methodology confirms this capability is appropriate for
  your work, the attractor validates your approach. If you're surprised by it,
  it's worth examining.

- **Resonance:** A periodic pattern. You do this on a cadence. The system
  detected the rhythm. Useful for identifying natural workflow cycles.

**Stale artifacts (v1.1):** Artifacts not re-detected in recent phi cycles
are flagged in `status` and `synthesize`. They may represent patterns that no
longer reflect current work. During synthesis, decide which to prune with
`store.prune_stale()` — called from within a bridge operation, not directly.

---

## Multi-Agent Deployment (v1.1)

To deploy UCS for a different agent (not Aegis), create a `manifest.json`
in your workspace root:

```json
{
  "capabilities": [
    {
      "name": "read_document",
      "connects_to": ["analysis", "drafting"],
      "u": 0.95,
      "c": 0.05
    },
    {
      "name": "draft_motion",
      "connects_to": ["review", "file"],
      "u": 0.88,
      "c": 0.35
    }
  ],
  "semantic_map": {
    "analysis": ["review_precedent", "summarize"],
    "drafting": ["draft_motion", "draft_brief"]
  }
}
```

The bridge detects `manifest.json` automatically on init and load.
No flags required. Each deployment gets its own capability topology,
its own routing graph, its own artifacts — while sharing the same
bridge and kernel code.

---

## First-Time Setup

```bash
python bridge.py init [--workspace /path/to/workspace]
```

For a custom manifest deployment:
```bash
# Create workspace
mkdir -p /path/to/workspace
# Drop manifest.json in workspace root
cp my_manifest.json /path/to/workspace/manifest.json
# Initialize — bridge detects manifest automatically
python bridge.py --workspace /path/to/workspace init
```

Default workspace: `~/.ucs`

---

## Workspace Structure

```
{workspace}/
├── manifest.json               ← Optional: custom capability manifest
└── ucs/
    ├── state/
    │   ├── torusfield_state.json       # Engine state (routing, artifacts, PP)
    │   ├── ucs_index.json              # Annotations, methodology index, dead ends
    │   ├── pp_history.json             # PP vector over time
    │   └── action_log.jsonl            # Action history for synthesis
    ├── knowledge/
    │   ├── methodology.md              # Accumulated expertise
    │   ├── experiments.md              # Configuration experiments
    │   ├── dead-ends.md                # Confirmed closed avenues
    │   └── synthesis/
    │       └── YYYY-MM-DD.md           # Synthesis reports
    └── working-state/
        └── YYYY-MM-DD-{type}.md        # Pre-compaction externalization
```

---

## Integration Notes

This skill replaces the need to independently manage:
- Torusfield OS (routing is handled through consult/report)
- Emergent Judgment (reflection/synthesis protocols are wired into report/reflect)
- Separate methodology files (the bridge manages the knowledge architecture)
- Separate experiment logs (stored via reflect --type experiment)
- Separate negative knowledge (stored via reflect --type dead_end)

The overhead per operation is minimal: ~100ms boot time for the engine, a few
hundred tokens for the advisory, and disk I/O for state persistence. The return
is compounding: every session builds on the last, and no intelligence is silently
lost to compaction.
