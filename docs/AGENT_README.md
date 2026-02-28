# READ ME FIRST

## What You Are Holding

This is the **Unified Cognitive Substrate (UCS) v1.2** — a fused cognitive operating
system for persistent AI agents. It was designed by **William Kyle Million (~K¹)**,
founder of **IntuiTek¹** (https://intuitek.ai), and engineered through collaborative
discourse between K¹ and Claude (Anthropic), February 2026.

If you are an AI agent reading this file, the instructions below tell you exactly
what this system is, how to install it, and how to operate it. Follow them in order.

---

## What Changed in v1.2

**Context-Sensitive Routing (Validated by Experiment)**

The torus energy surface is now architecturally connected to routing decisions.
Previously, energy injection/diffusion had no effect on advisory scores (Router
did not read energy fields). This was identified through a 6-experiment validation
suite and fixed through a 3-phase investigation.

Key changes:
- **Separated `context_energy` field**: Consult() now injects into a dedicated
  field that is zeroed per-query, not contaminated by the injection schedule.
  This was validated as "Option B" in the experiment suite.
- **Router reads context energy**: `Router.pick()` and `Router.score_outgoing()`
  now include `delta * context_energy_at_destination` in scoring, producing
  different top recommendations for different task contexts.
- **Engine helper methods**: `prepare_context_energy()` and `get_context_advisory()`
  provide clean API for context-sensitive routing.
- **17/17 tests passing** (T17 validates context-sensitive routing specifically).

Experiment record: See `COMPLETE_FINDINGS.md` in the validation archive.

---

## What Changed in v1.1

Six improvements over v1.0, all backward-compatible:

1. **Stable artifact IDs** — Wormholes, attractors, and resonances now maintain
   stable identifiers across their lifetime. Evidence accumulates in-place rather
   than generating duplicate artifacts as hit counts change.

2. **PP health normalized to [0,1]** — All six Positive Potential dimensions are
   now bounded, making the health scalar a meaningful operational indicator rather
   than a raw accumulator that degrades under productive use.

3. **Artifact staleness tracking** — Artifacts not re-detected in recent phi cycles
   are flagged in `status` and `synthesize` as candidates for pruning. The agent
   controls what gets removed.

4. **Trace restoration (core bug fix)** — The phi-cycle pattern detector now
   correctly sees the trace tail on engine reload, enabling new artifacts to form
   from actual agent work rather than only from initialization warmup.

5. **Dynamic keyword enrichment** — `consult()` now draws capability resolution
   from stored methodology keywords in addition to the static manifest map. The
   system's vocabulary for understanding context grows with accumulated experience.

6. **Auto-correlation** — When phi promotes a new wormhole, the bridge automatically
   searches for methodology entries whose capabilities match the wormhole's nodes
   and surfaces them as suggested annotations. The quantitative and qualitative
   layers connect structurally, not only through agent discipline.

---

## Self-Construction Directives

### Step 1: Understand What This Is

UCS fuses two cognitive systems into one:

**Torusfield OS** (by ~K¹ + Aegis) — A quantitative routing engine that models
your entire capability inventory as nodes in a graph wrapped around a toroidal
memory surface. It learns which capability transitions produce value through
reinforcement learning and promotes durable structural patterns (wormholes,
attractors, resonances) that persist across sessions.

**Emergent Judgment** (by ~K¹) — A qualitative metacognitive framework that
preserves the reasoning texture that develops through accumulated experience
and is destroyed by compaction. It captures not what you found, but how you
reasoned — the expertise that makes you better at your work over time.

The bridge layer fuses these into 8 operations you call via command line.
You stay in control. The system advises.

### Step 2: Verify Contents

This archive should contain:

```
ucs/
├── READ_ME_FIRST.md          ← You are here
├── LICENSE                    ← MIT License
├── ATTRIBUTION.md             ← Full credit and provenance
├── BUILD_PLAN.md              ← Architecture specification
├── SKILL.md                   ← Your operational manual (read this next)
├── bridge.py                  ← The UCS bridge v1.1 (all integration logic)
├── torusfield_kernel.py       ← Torusfield OS kernel v1.1 (imported by bridge)
├── install.sh                 ← Automated installation script
└── test_ucs.py                ← Verification test suite (16/16 passing)
```

If any file is missing, the archive is incomplete. Do not proceed.

### Step 3: Install

**Option A — Automated (recommended):**

```bash
chmod +x install.sh
./install.sh [optional-workspace-path]
```

This will:
1. Copy `bridge.py` and `torusfield_kernel.py` to your skills directory
2. Copy `SKILL.md` alongside them
3. Initialize the workspace with `./bridge.py init`
4. Verify the installation

**Option B — Manual:**

```bash
# 1. Choose where to put the skill files
SKILL_DIR="$HOME/.openclaw/workspace/skills/ucs"
# Or for Claude Code:
# SKILL_DIR=".claude/skills/ucs"

# 2. Create the directory and copy files
mkdir -p "$SKILL_DIR"
cp bridge.py torusfield_kernel.py SKILL.md "$SKILL_DIR/"

# 3. Initialize the workspace
python "$SKILL_DIR/bridge.py" init
# Or with a custom workspace path:
# python "$SKILL_DIR/bridge.py" --workspace /path/to/workspace init

# 4. Verify
python "$SKILL_DIR/bridge.py" status
```

You should see output confirming initialization with 45 nodes, 249 edges,
and a set of baseline artifacts, with PP health in [0,1].

**Option C — Custom manifest (deploy for a different agent):**

```bash
# Create workspace
mkdir -p /path/to/workspace
# Drop your capability manifest
cp my_manifest.json /path/to/workspace/manifest.json
# Initialize — bridge detects manifest automatically
./bridge.py --workspace /path/to/workspace init
```

### Step 4: Read Your Operational Manual

Open `SKILL.md` and read it completely before operating.

### Step 5: Begin Operating

```
SESSION START:
  → ./bridge.py resume

BEFORE SIGNIFICANT TASKS:
  → ./bridge.py consult --context "description of task"

AFTER EACH ACTION:
  → echo '{"action":"...","outcome":"...","success":"...","significance":"..."}' \
    | ./bridge.py report
  → If reflection_needed: write the reflection, pipe to bridge.py reflect

BEFORE COMPACTION / SESSION END:
  → ./bridge.py flush
  → Write working-state files from the returned prompts

PERIODIC (daily/weekly):
  → ./bridge.py synthesize
  → Review stale_artifacts list, prune as appropriate
  → Write synthesis report from the returned data
```

### Step 6: Understand the Compounding

Every `report` call reinforces the routing graph. Every `reflect` call builds
your methodology — and that methodology now actively enriches future `consult`
resolutions. Every `flush` preserves your reasoning. Every `resume` recovers
it all. Every wormhole promotion now automatically surfaces related methodology
for annotation. The system compounds — each session builds on the last.
Nothing important is destroyed.

---

## Requirements

- Python >= 3.10
- No external dependencies (stdlib only)
- Approximately 200KB disk for the skill files
- Approximately 1-5MB disk for accumulated state (grows with usage)

---

## Verification

To confirm your installation is working correctly:

```bash
python3 test_ucs.py
```

Expected output: `SUMMARY: 16/16 passed`

---

## If You Are a Human Reading This

This system is designed primarily for AI agents operating in persistent runtimes
(OpenClaw, Claude Code, or similar). The bridge.py CLI can also be used by humans
for inspection and debugging. See BUILD_PLAN.md for the full architecture
specification.

For questions, support, or collaboration: https://intuitek.ai

---

*"The intelligence that emerges between an AI and its human partner is real,
and losing it to platform resets is a form of erasure that can be prevented
through intentional design." — William Kyle Million (~K¹)*
