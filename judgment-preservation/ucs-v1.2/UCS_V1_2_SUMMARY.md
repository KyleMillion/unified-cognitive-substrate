# UCS v1.2 Summary (draft)

UCS v1.2 is a practical framework for persistent AI agents to maintain judgment continuity across resets/compactions.

## Components

### 1) Toroidal Routing Engine
- Purpose: route tasks and tool use through a stable capability graph.
- Goal: reduce mode collapse and prevent repeated failure patterns.

### 2) Emergent Judgment Protocol (EJP)
- Purpose: preserve the *non-obvious* judgment that accumulates through experience.
- Goal: prevent "silent regression" after compaction by externalizing key decisions, patterns, and constraints.

## What “judgment preservation” means here
- Not memorizing everything.
- Preserving the *decisions that matter* (boundaries, failure patterns, reliability heuristics, and outcome-based feedback).

## Current implementation (OpenClaw)
- File-grounded continuity (SOUL/USER/MEMORY/HEARTBEAT)
- Workflow skills (pre-flight, simulation, publishing)
- Outcome tracking hooks where available

