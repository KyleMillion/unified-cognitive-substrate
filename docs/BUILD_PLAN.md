# Unified Cognitive Substrate (UCS) — Build Plan

## 1. What We're Building

A single integration layer ("the Bridge") that fuses Torusfield OS (quantitative
routing engine) and Emergent Judgment (qualitative metacognitive framework) into
one coherent system an AI agent consumes as a single skill. The agent interacts
with four core operations — `consult`, `report`, `flush`, `resume` — plus four
helper operations — `init`, `reflect`, `synthesize`, `status`. From the agent's
perspective, this is one skill with one entry point. The complexity is hidden.

---

## 2. Architecture

### 2.1 Execution Model: Boot-Per-Call

Every invocation of the bridge boots the Torusfield engine from persisted JSON
state, performs the requested operation, saves state, and exits. No persistent
process. No sidecar. Stateless from the OS perspective, stateful from the data
perspective.

**Why boot-per-call:**
- Eliminates process lifecycle management (crash recovery, restart coordination)
- Aegis's 45-node graph boots in <100ms — negligible for an agent acting every few seconds
- State is always on disk, always recoverable
- Can graduate to sidecar later if profiling justifies it

### 2.2 Control Hierarchy

```
User intent (explicit request)
  └── overrides everything

Agent reasoning (LLM inference)
  └── primary decision-maker, informed by:
        ├── UCS routing advisory (consult output)
        ├── Methodology knowledge (consult output)
        └── Conversation context

UCS reinforcement (background)
  └── learns from what the agent actually does (report input)
  └── never overrides, only advises

UCS reflection (periodic)
  └── captures reasoning at significant moments
  └── feeds synthesis → calibration loop
```

The agent can disagree with the advisory. Disagreements are logged as policy
overrides and become data for the synthesis cycle.

### 2.3 The Manual-Step Problem

Torusfield's `step_theta()` both **chooses** the next capability AND **reinforces**
the chosen edge. In UCS, the *agent* chooses (informed by advisory), and the bridge
reinforces based on what the agent actually did. We cannot call `step_theta()`
because it would route to a different node than where the agent went.

**Solution:** The bridge performs a *manual step* that replicates the reinforcement
logic of `step_theta()` without the routing decision:

1. Find the edge from the current cursor node to the agent's chosen action
2. Compute reward from the agent's reported outcome
3. Apply reinforcement: `edge.w += lr * (net - 0.5)`, clamped to [-2.5, 2.5]
4. Record the step in the trace log
5. Move the cursor to the action node
6. Run energy injection, diffusion, and decay on the torus state
7. Check if a φ-cycle should fire

This preserves all of Torusfield's learning dynamics while letting the agent
stay in control of action selection.

### 2.4 The Translation Layer

Torusfield produces numeric artifacts (wormholes, attractors, resonances).
Emergent Judgment produces qualitative reflections. The bridge maintains an
**annotation index** that links them:

```
Artifact (wormhole: edge 14, mean_net 0.72)
    ↕  annotation index
Reflection ("This shortcut works because search snippets truncate pricing data.
             Failure condition: does NOT apply to API docs.")
```

Every promoted artifact gets flagged as "unannotated." The bridge surfaces these
to the agent for reflection. Once annotated, the artifact carries both its
quantitative signal and its qualitative explanation.

### 2.5 Data Flow Diagram

```
                    ┌─────────────────────┐
                    │     AGENT (LLM)     │
                    │                     │
                    │  "I need to research │
                    │   competitor pricing"│
                    └──────┬──────────────┘
                           │
               ┌───────────▼───────────────┐
               │    bridge.py consult()    │
               │                           │
               │  1. Boot engine from JSON │
               │  2. Inject context energy │
               │  3. Warm-up θ-steps       │
               │  4. Read routing topology │
               │  5. Search methodology    │
               │  6. Check negative-K      │
               │  7. Package advisory      │
               │  8. Save state            │
               └───────────┬───────────────┘
                           │
                    Advisory JSON
                    (suggested paths,
                     methodology hits,     ──►  Agent factors this into
                     dead ends,                 its own reasoning, then
                     PP health)                 acts using its own judgment
                           │
               ┌───────────▼───────────────┐
               │    bridge.py report()     │
               │                           │
               │  1. Boot engine from JSON │
               │  2. Map action → edge     │
               │  3. Compute reward        │
               │  4. Manual reinforcement  │
               │  5. Update trace log      │
               │  6. Move cursor           │
               │  7. Check φ-cycle         │
               │  8. Check EJ triggers     │
               │  9. Save state            │
               └───────────┬───────────────┘
                           │
                    Report JSON
                    (reinforcement applied,
                     new artifacts,         ──►  Agent writes reflections
                     reflection prompts,         if triggered, stores via
                     unannotated artifacts)      bridge.py reflect()
                           │
               ┌───────────▼───────────────┐
               │   bridge.py reflect()     │
               │                           │
               │  1. Store reflection text  │
               │  2. Link to artifact (if  │
               │     annotating artifact)  │
               │  3. Extract keywords for  │
               │     methodology index     │
               │  4. Append to methodology │
               │     or experiment log     │
               └───────────────────────────┘
```

---

## 3. Files We Build

```
ucs/
├── BUILD_PLAN.md                       # This document
├── SKILL.md                            # Agent-facing skill definition
├── bridge.py                           # The UCS bridge (all integration logic)
└── torusfield_kernel.py                # Existing kernel, imported unchanged
```

Three operational files. The agent reads `SKILL.md`, executes `bridge.py`, and
never touches `torusfield_kernel.py` directly.

---

## 4. Workspace Layout (Created by `init`)

```
{workspace_root}/
├── ucs/
│   ├── state/
│   │   ├── torusfield_state.json       # Persisted engine state
│   │   ├── ucs_index.json              # Annotations + methodology index
│   │   ├── pp_history.json             # PP vector over time (for trend analysis)
│   │   └── action_log.jsonl            # Qualitative action history (for synthesis)
│   ├── knowledge/
│   │   ├── methodology.md              # Accumulated expertise (EJ Section 1)
│   │   ├── experiments.md              # Experiment log (EJ Section 4)
│   │   ├── dead-ends.md                # Negative knowledge (EJ Section 3)
│   │   └── synthesis/                  # Periodic synthesis reports (EJ Section 5)
│   │       └── YYYY-MM-DD.md
│   └── working-state/                  # Pre-compaction externalization (EJ Section 2)
│       └── YYYY-MM-DD-{type}.md
```

---

## 5. Operations Specification

### 5.1 `init`

**Purpose:** First-time setup. Creates workspace, bootstraps Torusfield with
the capability manifest, runs initial warm-up, creates empty knowledge files.

**CLI:** `./bridge.py init [--workspace PATH] [--manifest aegis|PATH]`

**Process:**
1. Create workspace directory tree
2. Build Torusfield engine from capability manifest (default: Aegis)
3. Run 200 θ-steps to establish baseline routing topology
4. Export initial state to JSON
5. Create empty knowledge files with headers
6. Create empty UCS index
7. Return initialization report

**Output:** JSON with graph stats, initial PP, workspace path confirmation.

**Idempotency:** If workspace exists and has state, report that and do nothing
unless `--force` is passed.

### 5.2 `consult`

**Purpose:** Before acting. Agent passes current context, receives routing
advisory plus relevant methodology.

**CLI:** `./bridge.py consult --context "description" [--capabilities cap1,cap2,...]`

**Process:**
1. Load engine from `torusfield_state.json`
2. If capabilities specified, inject energy at their torus positions
3. If context provided, do keyword matching to identify relevant capabilities
   and inject energy there too
4. Run `theta_warmup_steps` (default 15) θ-steps to diffuse injected energy
5. Read outgoing edges from current cursor node, sorted by effective score
   (weight + utility - cost + novelty + policy bias)
6. Identify active wormholes relevant to the top paths
7. Search methodology index for entries matching context keywords/capabilities
8. Search dead-ends for entries matching context keywords/capabilities
9. Get current PP health and any unannotated artifacts
10. Package advisory
11. Save updated state (warm-up changes energy landscape)

**Output:**
```json
{
  "cursor": "current_node",
  "suggested_paths": [
    {
      "to": "web_fetch",
      "score": 1.34,
      "components": {"utility": 0.9, "cost": -0.22, "weight": 0.45, "bias": 0.21},
      "annotation": "When researching for evidence, always fetch full pages."
    }
  ],
  "wormholes": [
    {
      "from": "web_search", "to": "web_fetch",
      "hits": 14, "mean_net": 0.72,
      "annotation": "Snippets truncate. Full fetch captures hidden data."
    }
  ],
  "attractors": [
    {"node": "exec", "frequency": 0.23, "annotation": null}
  ],
  "methodology_hits": [
    {"summary": "...", "date": "2026-02-24", "capabilities": ["web_search"]}
  ],
  "dead_ends": [
    {"topic": "API scraping", "why_closed": "Rate limited", "date": "2026-02-10"}
  ],
  "pp_health": 0.73,
  "unannotated_count": 2,
  "open_questions": ["...from last session's working state..."]
}
```

### 5.3 `report`

**Purpose:** After acting. Agent reports what it did and what happened. Bridge
reinforces the torus, checks for artifacts, triggers reflections.

**CLI:** `echo '{"action":"web_search","outcome":"found 3 pages","success":"success","significance":"notable"}' | ./bridge.py report`

**Input fields:**
- `action` (required): Capability name the agent used
- `outcome` (required): Free text description of what happened
- `success` (required): One of `success`, `partial`, `failure`, `neutral`
- `significance` (optional, default `routine`): One of `routine`, `notable`, `critical`
- `context` (optional): Additional context about why this action was chosen

**Process:**
1. Load engine from state
2. Find edge from `cursor_node` → `action`. If no direct edge exists, find
   nearest path (agent may have jumped to a non-adjacent capability)
3. Compute reward using RewardModel: `success` → base reward, modulated by
   edge utility expectation
4. Apply manual reinforcement to the edge
5. Record TraceStep in engine's trace log
6. Move cursor to action node
7. Run energy injection schedule, diffusion, decay on torus
8. Increment internal report counter
9. If counter >= `phi_report_interval` (default 5): fire `step_phi()`
10. On φ-cycle: check for new artifacts, flag unannotated ones
11. If significance >= `notable`: generate EJ reflection prompts
12. Append to action_log.jsonl (qualitative record for synthesis)
13. Save state

**Output:**
```json
{
  "reinforcement": {"edge": "web_search → web_fetch", "old_w": 0.33, "new_w": 0.45, "reward": 0.82},
  "cursor": "web_fetch",
  "phi_fired": false,
  "new_artifacts": [],
  "reflection_needed": true,
  "reflection_prompt": {
    "type": "post_task",
    "fields": ["initial_signal", "hypothesis", "near_miss", "generalized_pattern", "negative_knowledge"]
  },
  "unannotated_artifacts": []
}
```

### 5.4 `reflect`

**Purpose:** Agent stores a reflection (post-task, artifact annotation, or
experiment log entry). Bridge indexes it for future retrieval.

**CLI:** `echo '{"type":"post_task","text":"### 2026-02-25 — Competitor pricing research\n...","capabilities":["web_search","web_fetch"],"keywords":["pricing","evidence"]}' | ./bridge.py reflect`

Or for artifact annotation:
`echo '{"type":"annotation","artifact_id":"a3f2c8b1","text":"This wormhole works because...","failure_condition":"Does not apply to API docs"}' | ./bridge.py reflect`

Or for experiment:
`echo '{"type":"experiment","text":"### 2026-02-25 — Changed warmup steps\n..."}' | ./bridge.py reflect`

**Process:**
1. Parse reflection type
2. For `post_task`: Append to methodology.md, create methodology index entry
   with keywords and capabilities for future search
3. For `annotation`: Link text to artifact in annotation index, enrich future
   consult() output with this annotation
4. For `experiment`: Append to experiments.md
5. For `dead_end`: Append to dead-ends.md, index for negative knowledge search
6. Save updated UCS index

**Output:** Confirmation with index entry ID.

### 5.5 `flush`

**Purpose:** Pre-compaction save. Exports all Torusfield state AND returns
EJ externalization prompts for the agent to fill.

**CLI:** `./bridge.py flush`

**Process:**
1. Load engine, export full state to torusfield_state.json
2. Snapshot PP vector to pp_history.json with timestamp
3. Generate EJ externalization prompts (current hypotheses, reasoning chains,
   open questions, confidence levels, context dependencies)
4. Return prompts

The agent then writes its working-state files. No second call needed — on
`resume()`, the bridge reads whatever the agent left in working-state/.

**Output:**
```json
{
  "torusfield_saved": true,
  "state_digest": "sha256...",
  "pp_snapshot_saved": true,
  "externalization_prompts": {
    "hypotheses": "What are you currently thinking about? What's unresolved?",
    "reasoning": "What chains of logic are active? Not conclusions — the chains.",
    "open_questions": "What would you investigate next if the session continued?",
    "confidence": "What are you certain about vs. uncertain about?",
    "dependencies": "What context are you relying on that wouldn't survive summary?"
  },
  "write_to": "{workspace}/ucs/working-state/"
}
```

### 5.6 `resume`

**Purpose:** Session start. Loads all persisted state, runs health check,
returns session briefing.

**CLI:** `./bridge.py resume`

**Process:**
1. Load engine from torusfield_state.json, import state
2. Run ok_report() for health metrics
3. Load most recent synthesis (if any)
4. Load most recent working-state files (if any)
5. Load unannotated artifacts
6. Load recent methodology entries (last 5)
7. Load PP trend from pp_history.json
8. Package session briefing

**Output:**
```json
{
  "session_briefing": {
    "pp_health": 0.73,
    "pp_trend": "improving",
    "artifact_count": 28,
    "artifact_breakdown": {"wormholes": 12, "attractors": 8, "resonances": 8},
    "cursor": "exec",
    "top_wormholes": [...with annotations...],
    "top_attractors": [...with annotations...],
    "unannotated_count": 3,
    "open_questions": ["...from last working-state..."],
    "recent_methodology": ["...last 5 entries..."],
    "last_synthesis_summary": "...",
    "recommendations": ["Annotate 3 unannotated artifacts", "Synthesis overdue"]
  }
}
```

### 5.7 `synthesize`

**Purpose:** Gathers all raw material needed for a synthesis report and returns
it. The agent (LLM) does the actual synthesis reasoning — the bridge provides
the data.

**CLI:** `./bridge.py synthesize`

**Process:**
1. Load engine state + ok_report
2. Load PP history (trend over time)
3. Load all methodology entries since last synthesis
4. Load all experiment entries since last synthesis
5. Load all action_log entries since last synthesis
6. Load artifact annotations (compare what torus learned vs. what methodology says)
7. Compute resource allocation (action frequency from action_log)
8. Identify misalignments (high-reward paths with no methodology, methodology
   entries with no corresponding artifacts)
9. Package as synthesis data bundle

**Output:** A large JSON with all raw material structured for the agent to
synthesize. The agent writes the synthesis to the synthesis/ directory and
optionally calls `reflect --type synthesis` to store it.

### 5.8 `status`

**Purpose:** Quick health check without full resume briefing.

**CLI:** `./bridge.py status`

**Output:** PP vector, health score, artifact count, cursor position, last
action timestamp, unannotated count.

---

## 6. Data Models

### 6.1 UCS Index (`ucs_index.json`)

```json
{
  "version": "ucs.v1.0",
  "annotations": {
    "artifact_id_hex": {
      "artifact_kind": "wormhole",
      "edge_src": "web_search",
      "edge_dst": "web_fetch",
      "text": "This wormhole works because...",
      "failure_condition": "Does not apply to API documentation.",
      "generalized_pattern": "When goal is evidence extraction, fetch full source.",
      "annotated_at": "2026-02-25T14:30:00Z"
    }
  },
  "methodology_entries": [
    {
      "id": "m_sha256_prefix",
      "date": "2026-02-25",
      "capabilities": ["web_search", "web_fetch"],
      "keywords": ["research", "evidence", "pricing", "full page"],
      "summary": "When researching for evidence extraction, always fetch full pages.",
      "source_type": "post_task"
    }
  ],
  "dead_end_entries": [
    {
      "id": "d_sha256_prefix",
      "date": "2026-02-10",
      "capabilities": ["browser"],
      "keywords": ["API", "scraping", "rate limit"],
      "topic": "API-based price scraping",
      "why_closed": "Rate limited after 3 requests",
      "reopen_conditions": "If target adds public API or rate limits change"
    }
  ],
  "report_counter": 0,
  "last_phi_t": 0,
  "last_synthesis_date": null,
  "policy_overrides": []
}
```

### 6.2 PP History (`pp_history.json`)

```json
{
  "snapshots": [
    {
      "t": 200,
      "timestamp": "2026-02-25T10:00:00Z",
      "pp": {"capacity": 1.2, "optionality": 0.8, "resilience": 0.6, ...},
      "health": 0.65,
      "artifact_count": 5
    }
  ]
}
```

### 6.3 Action Log (`action_log.jsonl`)

One JSON object per line:
```json
{"t": 215, "timestamp": "2026-02-25T14:32:00Z", "action": "web_search", "outcome": "found 3 pages", "success": "success", "significance": "notable", "reward": 0.82, "edge": "exec → web_search", "cursor_before": "exec", "cursor_after": "web_search"}
```

---

## 7. Reward Model

Maps the agent's qualitative outcome assessment to a numeric reward signal
for Torusfield reinforcement.

**Base reward by success level:**

| Success | Base Reward |
|---------|-------------|
| `success` | 0.80 |
| `partial` | 0.45 |
| `neutral` | 0.30 |
| `failure` | 0.10 |

**Modulation:** The base reward is blended with the edge's utility expectation:
```
final_reward = 0.7 * base_reward + 0.3 * edge.u
```

This anchors reinforcement to the edge's expected utility, preventing a single
good outcome from wildly over-promoting a generally low-utility path.

**Cost:** Uses the edge's inherent cost value: `edge.c`

**Net signal for reinforcement:** `reward - cost`

---

## 8. Keyword-Capability Mapping (for consult context injection)

When `consult()` receives a free-text context, the bridge maps keywords to
capability nodes for energy injection. This uses a simple static mapping
derived from the Aegis semantic map, inverted:

```
"research" → [web_search, web_fetch, browser, foundry_research]
"write"    → [write, foundry_write_skill, foundry_write_extension]
"debug"    → [exec, read, edit, foundry_implement]
"monitor"  → [session_status, foundry_metrics, foundry_overseer]
...etc
```

The mapping is built once at init time from AEGIS_SEMANTIC_MAP. Keywords from
the context are matched against this mapping. Matched capabilities receive
energy injections proportional to match count.

---

## 9. Edge-Finding Strategy

When `report()` receives an action that isn't directly adjacent to the cursor
(agent jumped across the graph), the bridge needs to find the right edge:

1. **Direct edge exists:** Use it (cursor → action edge in graph.adj)
2. **No direct edge:** Find the shortest path through intermediate nodes.
   Apply partial reinforcement to each edge in the path, weighted by
   `1/path_length` to distribute credit.
3. **No path exists:** This shouldn't happen in a well-connected graph, but
   if it does: log a policy override, create a synthetic trace entry, and
   move the cursor directly. The gap itself becomes data.

---

## 10. OpenClaw Integration

### 10.1 Deployment

```bash
# Copy to workspace skills directory
cp -r ucs/ ~/.openclaw/workspace/skills/ucs/

# Initialize workspace
python ~/.openclaw/workspace/skills/ucs/bridge.py init

# Agent reads SKILL.md and begins using the four operations
```

### 10.2 SKILL.md Trigger Conditions

The SKILL.md will define triggers that tell the agent when to call each
operation:

| Event | Operation |
|-------|-----------|
| Starting any significant task | `consult` |
| After each tool call or action | `report` |
| After notable/critical outcomes | `report` (with significance flag) |
| When `report` returns `reflection_needed: true` | `reflect` |
| When seeing unannotated artifacts | `reflect --type annotation` |
| Before `/compact` or approaching context limit | `flush` |
| Session start | `resume` |
| End of day/week or on request | `synthesize` |
| Quick check | `status` |

### 10.3 Agent Execution Pattern

The agent's runtime loop with UCS becomes:

```
SESSION START:
  briefing = exec("./bridge.py resume")
  → inject briefing into context

ON USER REQUEST:
  advisory = exec("./bridge.py consult --context '...'")
  → factor advisory into reasoning
  → select and execute action using own judgment

AFTER EACH ACTION:
  result = exec("echo '{...}' | ./bridge.py report")
  → if result.reflection_needed:
      → reason about the reflection prompts
      → exec("echo '{...}' | ./bridge.py reflect")

BEFORE COMPACTION:
  prompts = exec("./bridge.py flush")
  → write working-state files based on prompts

PERIODIC:
  data = exec("./bridge.py synthesize")
  → write synthesis report from data
  → exec("echo '{...}' | ./bridge.py reflect --type synthesis")
```

---

## 11. Build Sequence

### Phase 1: Core Bridge (consult / report / flush / resume)

Build `bridge.py` with:
- UCSConfig dataclass
- RewardModel class
- Engine loading/saving helpers
- `consult()` — energy injection, warm-up, advisory packaging
- `report()` — manual reinforcement, trace logging, φ-cycle checking
- `flush()` — full state export + EJ prompts
- `resume()` — full state load + briefing generation
- `status()` — quick health check
- `init()` — workspace creation + engine bootstrap
- CLI argument parsing

### Phase 2: Annotation & Reflection (reflect + index)

Add to `bridge.py`:
- AnnotationStore class (artifact annotation CRUD)
- MethodologyIndex class (keyword-searchable methodology entries)
- `reflect()` — stores reflections, updates index
- Integrate annotations into `consult()` output
- Wire unannotated artifact detection into `report()` φ-cycle

### Phase 3: Synthesis

Add to `bridge.py`:
- `synthesize()` — gathers raw material from all data stores
- PP history tracking (snapshot on each flush)
- Action log (append on each report)
- Resource allocation computation

### Phase 4: SKILL.md

Write the agent-facing skill definition with:
- Trigger conditions
- Operation specifications (what to call, when, with what)
- Expected input/output formats
- Integration notes
- Quick-reference decision tree

### Phase 5: Verification

- Run `init` → verify workspace creation
- Run `consult` → verify advisory output with fresh engine
- Run `report` × 30 → verify reinforcement, artifact detection, φ-cycles
- Run `reflect` → verify annotation storage and methodology indexing
- Run `flush` → verify state persistence
- Run `resume` → verify state recovery matches pre-flush state
- Run `synthesize` → verify data gathering

---

## 12. What We Don't Build (Yet)

- **Sidecar mode:** Boot-per-call first. Optimize later if needed.
- **Multi-agent coordination:** Single agent for now. The state format is
  extensible to multi-agent but we don't implement sharing.
- **Auto-synthesis scheduling:** Agent decides when to synthesize based on
  SKILL.md triggers. No cron or background timer.
- **Custom capability manifests:** Aegis manifest is embedded. Adding other
  manifests is straightforward but not in scope.
- **GUI/dashboard:** Status output is JSON. Visualization is a future layer.
