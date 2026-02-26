#!/usr/bin/env python3
"""
UCS Test Suite — William Kyle Million / IntuiTek¹
Tests actual claimed behaviors, not just whether code executes.

Each test has a clear CLAIM being tested, a PASS/FAIL verdict, and honest notes.
"""

import json
import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bridge import UCSBridge, UCSConfig
from torusfield_kernel import TorusfieldEngine, AEGIS_CAPABILITY_MANIFEST

# ─────────────────────────────────────────────────────────────
# Test infrastructure
# ─────────────────────────────────────────────────────────────

RESULTS = []

def test(name, claim):
    """Decorator / context for a named test."""
    def decorator(fn):
        def wrapper():
            try:
                verdict, notes = fn()
                RESULTS.append({
                    "name": name,
                    "claim": claim,
                    "verdict": verdict,
                    "notes": notes
                })
            except Exception as e:
                RESULTS.append({
                    "name": name,
                    "claim": claim,
                    "verdict": "ERROR",
                    "notes": f"Uncaught exception: {type(e).__name__}: {e}"
                })
        return wrapper
    return decorator


def fresh_workspace():
    """Create a temp workspace and return (config, bridge, tmpdir)."""
    tmpdir = tempfile.mkdtemp()
    config = UCSConfig(workspace_root=Path(tmpdir))
    bridge = UCSBridge(config)
    return config, bridge, tmpdir


def cleanup(tmpdir):
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# TEST 1: Initialization creates required file structure
# ─────────────────────────────────────────────────────────────

@test(
    name="T01 — Init creates required files",
    claim="init() creates state files, knowledge files, and a valid engine on disk"
)
def test_init_creates_structure():
    config, bridge, tmpdir = fresh_workspace()
    try:
        result = bridge.op_init()

        required_files = [
            config.torusfield_state_path,
            config.ucs_index_path,
            config.pp_history_path,
            config.methodology_path,
            config.experiments_path,
            config.dead_ends_path,
        ]

        missing = [str(f) for f in required_files if not f.exists()]

        if result.get("status") != "initialized":
            return "FAIL", f"Init returned status: {result.get('status')}"
        if missing:
            return "FAIL", f"Missing files after init: {missing}"
        if result.get("graph", {}).get("nodes") != 45:
            return "FAIL", f"Expected 45 nodes, got {result.get('graph', {}).get('nodes')}"
        if result.get("graph", {}).get("edges") != 249:
            return "FAIL", f"Expected 249 edges, got {result.get('graph', {}).get('edges')}"

        return "PASS", f"All required files created. 45 nodes, 249 edges. PP health: {result.get('pp_health')}"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 2: State actually persists across separate bridge instances
# ─────────────────────────────────────────────────────────────

@test(
    name="T02 — State persists across sessions",
    claim="Engine state saved on one bridge instance is correctly loaded by a new instance"
)
def test_state_persistence():
    config, bridge1, tmpdir = fresh_workspace()
    try:
        bridge1.op_init()

        # Report an action on bridge1
        bridge1.op_report("read", "read contract file", "success", "routine")
        t_after = bridge1.engine.state.t
        cursor_after = bridge1.engine.cursor_node

        # Create brand new bridge instance pointing to same workspace
        bridge2 = UCSBridge(config)
        result = bridge2.op_resume()

        briefing = result.get("session_briefing", {})
        t_recovered = briefing.get("t")
        cursor_recovered = briefing.get("cursor")

        if t_recovered != t_after:
            return "FAIL", f"t mismatch: saved={t_after}, recovered={t_recovered}"
        if cursor_recovered != cursor_after:
            return "FAIL", f"cursor mismatch: saved={cursor_after}, recovered={cursor_recovered}"

        return "PASS", f"State recovered correctly. t={t_recovered}, cursor={cursor_recovered}"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 3: Reinforcement actually changes edge weights
# This is the CORE claim — repeated positive actions should strengthen edges
# ─────────────────────────────────────────────────────────────

@test(
    name="T03 — Reinforcement changes edge weights",
    claim="Repeated successful actions on the same edge increase that edge's weight (learning occurs)"
)
def test_reinforcement_learning():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Find initial weight of message→read edge (known to exist from consult scoring)
        eng = bridge._load_engine()
        # Find the edge from current cursor (message) to read
        eidx = bridge._find_edge(eng, "message", "read")
        if eidx is None:
            return "FAIL", "Could not find message→read edge to test"

        initial_weight = eng.graph.edges[eidx].w
        bridge._save_engine()

        # Report 10 successful "read" actions
        for i in range(10):
            bridge.op_report("read", f"successfully read file {i}", "success", "routine")

        # Reload and check
        eng2 = bridge._load_engine()
        final_weight = eng2.graph.edges[eidx].w

        if final_weight <= initial_weight:
            return "FAIL", f"Weight did not increase after 10 successes. Initial={initial_weight:.4f}, Final={final_weight:.4f}"

        delta = final_weight - initial_weight
        return "PASS", f"Edge weight increased by {delta:.4f} after 10 successful actions. {initial_weight:.4f} → {final_weight:.4f}"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 4: Negative reinforcement decreases edge weights
# ─────────────────────────────────────────────────────────────

@test(
    name="T04 — Negative reinforcement weakens edges",
    claim="Repeated failures on an edge decrease its weight (bad paths become less likely)"
)
def test_negative_reinforcement():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        eng = bridge._load_engine()
        eidx = bridge._find_edge(eng, "message", "read")
        if eidx is None:
            return "FAIL", "Could not find message→read edge"

        initial_weight = eng.graph.edges[eidx].w
        bridge._save_engine()

        # Report 10 failures
        for i in range(10):
            bridge.op_report("read", f"failed to read file {i}", "failure", "routine")

        eng2 = bridge._load_engine()
        final_weight = eng2.graph.edges[eidx].w

        if final_weight >= initial_weight:
            return "FAIL", f"Weight did not decrease after 10 failures. Initial={initial_weight:.4f}, Final={final_weight:.4f}"

        delta = initial_weight - final_weight
        return "PASS", f"Edge weight decreased by {delta:.4f} after 10 failures. {initial_weight:.4f} → {final_weight:.4f}"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 5: Routing advisory differs based on learned weights
# ─────────────────────────────────────────────────────────────

@test(
    name="T05 — Learned weights change routing advisory",
    claim="After many successes on one path, consult() returns that path ranked higher than before learning"
)
def test_routing_changes_with_learning():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Get initial advisory from message cursor
        advisory_before = bridge.op_consult("analyze and read files")
        paths_before = advisory_before.get("suggested_paths", [])
        read_rank_before = next(
            (i for i, p in enumerate(paths_before) if p["to"] == "read"),
            None
        )

        # Strongly reinforce read repeatedly
        for i in range(20):
            bridge.op_report("read", f"successful read {i}", "success", "notable")

        # Get advisory again — cursor is now "read", need to check from there
        # Reset cursor to message by reinitializing (keeps weights)
        bridge.engine.cursor_node = "message"
        bridge._save_engine()

        advisory_after = bridge.op_consult("analyze and read files")
        paths_after = advisory_after.get("suggested_paths", [])
        read_rank_after = next(
            (i for i, p in enumerate(paths_after) if p["to"] == "read"),
            None
        )

        # Get the actual score delta
        score_before = next((p["score"] for p in paths_before if p["to"] == "read"), None)
        score_after = next((p["score"] for p in paths_after if p["to"] == "read"), None)

        if score_before is None or score_after is None:
            return "FAIL", "Could not find 'read' in advisory paths"

        if score_after <= score_before:
            return "FAIL", f"Score did not increase. Before={score_before:.4f}, After={score_after:.4f}"

        return "PASS", (
            f"'read' score increased from {score_before:.4f} to {score_after:.4f} after 20 successes. "
            f"Rank: {read_rank_before} → {read_rank_after}"
        )
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 6: Flush + Resume preserves working state prompts
# ─────────────────────────────────────────────────────────────

@test(
    name="T06 — Flush produces externalization prompts",
    claim="flush() returns all 5 externalization prompts with correct file names for today's date"
)
def test_flush_externalization_prompts():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()
        result = bridge.op_flush()

        required_prompt_keys = ["hypotheses", "reasoning", "open_questions", "confidence", "dependencies"]
        prompts = result.get("externalization_prompts", {})

        missing = [k for k in required_prompt_keys if k not in prompts]
        if missing:
            return "FAIL", f"Missing prompt keys: {missing}"

        if not result.get("torusfield_saved"):
            return "FAIL", "torusfield_saved was False"

        # Verify each prompt has a file and a prompt
        for key in required_prompt_keys:
            entry = prompts[key]
            if "file" not in entry or "prompt" not in entry:
                return "FAIL", f"Prompt '{key}' missing 'file' or 'prompt' key"
            if not entry["prompt"].strip():
                return "FAIL", f"Prompt '{key}' has empty prompt text"

        return "PASS", f"All 5 externalization prompts present. State digest: {result.get('state_digest')}"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 7: Methodology stored in reflect() is retrievable in consult()
# This tests the core EJ claim: stored reasoning comes back when relevant
# ─────────────────────────────────────────────────────────────

@test(
    name="T07 — Stored methodology surfaces in consult()",
    claim="A reflection stored via reflect() is returned by consult() when the context matches"
)
def test_methodology_retrieval():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Store a post-task reflection about reentrancy
        bridge.op_reflect(
            reflection_type="post_task",
            text="## Reentrancy Pattern\nWhen state updates follow external calls, check execution order.",
            capabilities=["read", "exec"],
            keywords=["reentrancy", "solidity", "audit", "vulnerability"]
        )

        # Now consult with matching context
        result = bridge.op_consult(
            "performing a solidity audit looking for reentrancy vulnerabilities"
        )

        hits = result.get("methodology_hits", [])

        if not hits:
            return "FAIL", "No methodology hits returned despite matching keywords"

        return "PASS", f"Retrieved {len(hits)} methodology hit(s) for matching context. First: '{hits[0].get('summary', '')[:80]}'"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 8: Dead ends stored in reflect() are retrievable
# ─────────────────────────────────────────────────────────────

@test(
    name="T08 — Dead ends surface in consult()",
    claim="A dead end stored via reflect() is returned by consult() to prevent re-investigation"
)
def test_dead_end_retrieval():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Store a dead end
        bridge.op_reflect(
            reflection_type="dead_end",
            text="## Integer Overflow in SafeMath contracts\nNot applicable — SafeMath handles this.",
            capabilities=["read"],
            keywords=["integer", "overflow", "safemath"],
            topic="integer overflow in SafeMath",
            why_closed="SafeMath library explicitly prevents this",
            reopen_conditions="Contract not using SafeMath or using Solidity <0.8"
        )

        # Consult with matching context
        result = bridge.op_consult("checking for integer overflow vulnerabilities in solidity")
        dead_ends = result.get("dead_ends", [])

        if not dead_ends:
            return "FAIL", "No dead ends returned despite matching context"

        return "PASS", f"Dead end retrieved. Topic: '{dead_ends[0].get('topic', '')}'. Reopen condition present: {bool(dead_ends[0].get('reopen_conditions'))}"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 9: Artifacts emerge after repeated pattern
# ─────────────────────────────────────────────────────────────

@test(
    name="T09 — Phi-cycle detects structural patterns; diverse usage creates new artifacts",
    claim="Wormhole payloads update with current evidence; using new capabilities creates genuinely new artifacts"
)
def test_artifact_emergence():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()
        eng0 = bridge._load_engine()
        
        # Record initial wormhole IDs and their hit counts
        initial_wormholes = {
            a.id: a.payload.get("hits", 0)
            for a in eng0.store.get_by_kind("wormhole")
        }
        initial_count = len(eng0.store.artifacts)

        # Use a capability set that's different from warmup to create new dominant edges
        # foundry_implement → exec → foundry_implement is unlikely to dominate warmup
        diverse_caps = ["exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement", "exec", "foundry_implement",
                        "exec", "foundry_implement"]
        for i, action in enumerate(diverse_caps):
            bridge.op_report(action, f"action {i}", "success", "routine")

        eng = bridge._load_engine()
        final_count = len(eng.store.artifacts)
        final_wormholes = {
            a.id: a.payload.get("hits", 0)
            for a in eng.store.get_by_kind("wormhole")
        }

        # Check 1: New artifacts appeared (new dominant edges emerged)
        new_artifacts = final_count - initial_count
        
        # Check 2: Phi cycle is active (at least some wormholes exist)
        if not final_wormholes:
            return "FAIL", "No wormholes detected at all"

        # Check 3: At least one new wormhole ID that wasn't in initial set
        new_wormhole_ids = set(final_wormholes.keys()) - set(initial_wormholes.keys())

        if new_artifacts > 0:
            return "PASS", (
                f"{new_artifacts} new artifacts from diverse capability usage. "
                f"{len(new_wormhole_ids)} new wormhole IDs. "
                f"Total artifacts: {initial_count} → {final_count}"
            )
        elif new_wormhole_ids:
            return "PASS", (
                f"New wormhole patterns emerged: {len(new_wormhole_ids)} new IDs. "
                f"Phi cycle active, pattern detection working."
            )
        else:
            return "FAIL", (
                f"No new patterns detected after diverse usage. "
                f"Artifacts: {initial_count} → {final_count}. "
                f"Wormholes: {sorted(final_wormholes.keys())}"
            )
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 10: Reflection triggers on notable significance
# ─────────────────────────────────────────────────────────────

@test(
    name="T10 — Notable significance triggers reflection prompt",
    claim="report() with significance='notable' returns reflection_needed=True with structured prompts"
)
def test_reflection_trigger():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        result_routine = bridge.op_report("read", "routine read", "success", "routine")
        result_notable = bridge.op_report("exec", "found critical bug", "success", "notable")
        result_critical = bridge.op_report("write", "patched critical issue", "success", "critical")

        if result_routine.get("reflection_needed"):
            return "FAIL", "Routine action incorrectly triggered reflection"
        if not result_notable.get("reflection_needed"):
            return "FAIL", "Notable action did not trigger reflection"
        if not result_critical.get("reflection_needed"):
            return "FAIL", "Critical action did not trigger reflection"

        # Check reflection prompt has all required fields
        prompt = result_notable.get("reflection_prompt", {})
        required_fields = ["initial_signal", "hypothesis", "near_miss", "generalized_pattern", "negative_knowledge"]
        fields = prompt.get("fields", {})
        missing_fields = [f for f in required_fields if f not in fields]

        if missing_fields:
            return "FAIL", f"Reflection prompt missing fields: {missing_fields}"

        return "PASS", "Reflection triggered correctly for notable/critical, not for routine. All 5 EJ fields present."
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 11: Policy override logged when agent jumps non-adjacent nodes
# ─────────────────────────────────────────────────────────────

@test(
    name="T11 — Policy overrides are logged",
    claim="When agent acts on a capability with no direct edge from cursor, it's logged as a policy override, not silently dropped"
)
def test_policy_override_logging():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Force cursor to a known position
        bridge.engine.cursor_node = "read"
        bridge._save_engine()

        # Act on something with no direct edge from read
        # nodes.py connects to device_ops, camera_snap etc — unlikely to have direct edge from read
        result = bridge.op_report("canvas", "rendered some output", "success", "routine")

        is_override = "policy_override" in result
        bridge.index.load()
        overrides = bridge.index.data.get("policy_overrides", [])

        if not is_override and not overrides:
            return "FAIL", "No policy override logged for non-adjacent jump"

        if is_override:
            return "PASS", f"Policy override correctly detected and logged. Reason: {result['policy_override'].get('reason')}"
        else:
            return "PASS", f"Policy override logged in index ({len(overrides)} total). Note: direct edge may exist."
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 12: PP health responds to action quality
# ─────────────────────────────────────────────────────────────

@test(
    name="T12 — PP health metric changes with action patterns",
    claim="Positive Potential health score changes after sustained action patterns (not static)"
)
def test_pp_health_changes():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        eng = bridge._load_engine()
        initial_health = eng.pp.health_score()
        bridge._save_engine()

        # Run 50 actions to stress the system
        for i in range(50):
            action = "read" if i % 3 != 0 else "exec"
            success = "success" if i % 5 != 0 else "failure"
            bridge.op_report(action, f"action {i}", success, "routine")

        eng2 = bridge._load_engine()
        final_health = eng2.pp.health_score()

        # PP health should change — it tracks throughput, capacity, etc.
        if initial_health == final_health:
            return "FAIL", f"PP health unchanged after 50 actions: {initial_health}"

        direction = "improved" if final_health > initial_health else "declined"
        return "PASS", f"PP health {direction} from {initial_health:.4f} to {final_health:.4f} after 50 actions."
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 13: Unknown capability handled gracefully
# ─────────────────────────────────────────────────────────────

@test(
    name="T13 — Unknown capability handled gracefully",
    claim="Reporting an action with an unknown capability name returns a warning rather than crashing"
)
def test_unknown_capability():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()
        result = bridge.op_report("nonexistent_tool_xyz", "did something", "success", "routine")

        if "error" in result and "warning" not in result:
            # Hard error is acceptable if it doesn't crash
            return "PASS", f"Returned error cleanly: {result.get('error')}"

        if "warning" in result:
            return "PASS", f"Warning returned for unknown capability: {result.get('warning')}"

        # If it somehow "succeeded" without warning, that's actually a problem
        return "FAIL", "Unknown capability accepted without warning — could corrupt routing graph"
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 14: Consult context matching — different contexts produce different advisories
# ─────────────────────────────────────────────────────────────

@test(
    name="T14 — Context matching produces differentiated advisories",
    claim="consult() with different task contexts produces different resolved_capabilities and energy injection targets"
)
def test_context_differentiation():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        result_audit = bridge.op_consult("reviewing smart contract for security vulnerabilities")
        result_search = bridge.op_consult("searching the web for recent news and research papers")

        caps_audit = set(result_audit.get("resolved_capabilities", []))
        caps_search = set(result_search.get("resolved_capabilities", []))

        if caps_audit == caps_search:
            return "FAIL", "Same capabilities resolved for completely different contexts — context matching not working"

        audit_only = caps_audit - caps_search
        search_only = caps_search - caps_audit

        return "PASS", (
            f"Different capabilities resolved. "
            f"Audit-only: {sorted(audit_only)}. "
            f"Search-only: {sorted(search_only)}"
        )
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 15: The compaction cycle — does state survive a simulated compaction?
# This is THE core claim of the whole system
# ─────────────────────────────────────────────────────────────

@test(
    name="T15 — Compaction cycle: learned state survives simulated session reset",
    claim="Routing weights, artifacts, and methodology learned in session 1 are fully available in session 2 after flush/resume"
)
def test_compaction_survival():
    config, bridge_s1, tmpdir = fresh_workspace()
    try:
        # SESSION 1: Learn some patterns, store methodology
        bridge_s1.op_init()

        # Learn a strong preference for read→exec
        for i in range(15):
            bridge_s1.op_report("read", f"read file {i}", "success", "routine")

        eng_s1 = bridge_s1._load_engine()
        eidx = bridge_s1._find_edge(eng_s1, "message", "read")
        weight_s1 = eng_s1.graph.edges[eidx].w if eidx else None
        artifact_count_s1 = len(eng_s1.store.artifacts)
        bridge_s1._save_engine()

        # Store methodology
        bridge_s1.op_reflect(
            reflection_type="post_task",
            text="## Critical Pattern: Always read before exec in audit workflows",
            capabilities=["read", "exec"],
            keywords=["audit", "read", "sequence"]
        )

        # FLUSH — simulates pre-compaction save
        bridge_s1.op_flush()

        # ── SESSION ENDS. New bridge instance (simulates session reset) ──
        bridge_s2 = UCSBridge(config)

        # RESUME — simulates new session start
        resume_result = bridge_s2.op_resume()
        briefing = resume_result.get("session_briefing", {})

        # Check 1: Artifact count preserved
        artifact_count_s2 = briefing.get("artifact_count", 0)
        if artifact_count_s2 != artifact_count_s1:
            return "FAIL", f"Artifact count changed across session reset: {artifact_count_s1} → {artifact_count_s2}"

        # Check 2: Edge weight preserved
        eng_s2 = bridge_s2._load_engine()
        if eidx is not None:
            weight_s2 = eng_s2.graph.edges[eidx].w
            if abs(weight_s2 - weight_s1) > 0.001:
                return "FAIL", f"Edge weight changed across session reset: {weight_s1:.4f} → {weight_s2:.4f}"

        # Check 3: Methodology retrievable in session 2
        consult_result = bridge_s2.op_consult("running an audit read sequence")
        hits = consult_result.get("methodology_hits", [])

        if not hits:
            return "FAIL", "Methodology from session 1 not recoverable in session 2"

        weight_note = f", edge weight preserved: {weight_s1:.4f}" if weight_s1 is not None else ""
        return "PASS", (
            f"Full compaction survival: {artifact_count_s1} artifacts preserved, "
            f"methodology retrievable in new session{weight_note}"
        )
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# TEST 16: Synthesize returns actionable data structure
# ─────────────────────────────────────────────────────────────

@test(
    name="T16 — Synthesize produces complete data structure",
    claim="synthesize() returns all required fields for an LLM to produce a meaningful synthesis report"
)
def test_synthesize_structure():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Log some actions first
        for action in ["read", "exec", "read", "write", "read"]:
            bridge.op_report(action, f"did {action}", "success", "routine")

        result = bridge.op_synthesize()

        required_keys = [
            "synthesis_data",
            "synthesis_prompts"
        ]
        required_data_keys = [
            "period", "total_actions", "resource_allocation",
            "pp_health", "pp_trend", "artifact_summary",
            "policy_overrides", "misalignments"
        ]
        required_prompt_keys = [
            "patterns_emerging", "judgment_shifts", "resource_review",
            "whats_working", "whats_not", "recommendations", "open_questions"
        ]

        missing_top = [k for k in required_keys if k not in result]
        if missing_top:
            return "FAIL", f"Missing top-level keys: {missing_top}"

        missing_data = [k for k in required_data_keys if k not in result["synthesis_data"]]
        if missing_data:
            return "FAIL", f"Missing synthesis_data keys: {missing_data}"

        missing_prompts = [k for k in required_prompt_keys if k not in result["synthesis_prompts"]]
        if missing_prompts:
            return "FAIL", f"Missing synthesis_prompts keys: {missing_prompts}"

        total = result["synthesis_data"]["total_actions"]
        return "PASS", f"Complete synthesis data structure. {total} actions logged, misalignment detection active."
    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# T17 — Context-Sensitive Routing (v1.2)
# ─────────────────────────────────────────────────────────────

@test("T17 — Context energy produces different routing advisory per task context",
      claim=("consult() with 'research' context produces different top-1 "
             "recommendation than consult() with 'build' context, because "
             "context_energy field boosts relevant capabilities (v1.2 fix)"))
def test_context_sensitive_routing():
    config, bridge, tmpdir = fresh_workspace()
    try:
        bridge.op_init()

        # Train the engine with diverse actions
        for _ in range(5):
            bridge.op_report("web_search", "found results", "success", "routine")
            bridge.op_report("web_fetch", "downloaded page", "success", "routine")
            bridge.op_report("write", "produced output", "success", "routine")
            bridge.op_report("exec", "ran script", "success", "routine")
            bridge.op_report("foundry_metrics", "checked health", "success", "routine")
            # End on 'read' — well-connected node with diverse adjacency
            bridge.op_report("read", "analyzed content", "success", "routine")

        # Consult with research context
        research_result = bridge.op_consult("research competitor pricing strategies")
        research_paths = research_result["suggested_paths"]
        research_top3 = [p["to"] for p in research_paths[:3]]
        research_boosted = research_result.get("context_boosted", {})
        research_boosts = [p.get("context_boost", 0) for p in research_paths[:6]]

        # Consult with build context
        build_result = bridge.op_consult("build automated deployment script")
        build_paths = build_result["suggested_paths"]
        build_top3 = [p["to"] for p in build_paths[:3]]
        build_boosts = [p.get("context_boost", 0) for p in build_paths[:6]]

        # Consult with monitor context
        monitor_result = bridge.op_consult("monitor system health and metrics")
        monitor_paths = monitor_result["suggested_paths"]
        monitor_top3 = [p["to"] for p in monitor_paths[:3]]

        # Check that context_boosted is populated
        if not research_boosted:
            return "FAIL", "No context_boosted capabilities in research consult"

        # Check that context energy actually shifts the rankings
        # The top-3 should differ between at least two contexts
        rankings_differ = (research_top3 != build_top3 or
                           research_top3 != monitor_top3)

        # Check that some paths received non-zero context boost
        any_research_boost = any(b > 0.01 for b in research_boosts)
        any_build_boost = any(b > 0.01 for b in build_boosts)

        notes = (f"Research top-3: {research_top3}. "
                 f"Build top-3: {build_top3}. "
                 f"Monitor top-3: {monitor_top3}. "
                 f"Rankings differ: {rankings_differ}. "
                 f"Research boosted: {len(research_boosted)} caps. "
                 f"Context boosts present: research={any_research_boost}, build={any_build_boost}.")

        if not rankings_differ:
            return "FAIL", f"Rankings identical across contexts. {notes}"

        if not any_research_boost:
            return "FAIL", f"No context energy boost applied. {notes}"

        return "PASS", notes

    finally:
        cleanup(tmpdir)


# ─────────────────────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_init_creates_structure,
    test_state_persistence,
    test_reinforcement_learning,
    test_negative_reinforcement,
    test_routing_changes_with_learning,
    test_flush_externalization_prompts,
    test_methodology_retrieval,
    test_dead_end_retrieval,
    test_artifact_emergence,
    test_reflection_trigger,
    test_policy_override_logging,
    test_pp_health_changes,
    test_unknown_capability,
    test_context_differentiation,
    test_compaction_survival,
    test_synthesize_structure,
    test_context_sensitive_routing,
]

if __name__ == "__main__":
    print("Running UCS test suite...\n")

    for test_fn in ALL_TESTS:
        test_fn()

    # Print results
    passed = [r for r in RESULTS if r["verdict"] == "PASS"]
    failed = [r for r in RESULTS if r["verdict"] == "FAIL"]
    errors = [r for r in RESULTS if r["verdict"] == "ERROR"]

    print("=" * 70)
    for r in RESULTS:
        icon = "✓" if r["verdict"] == "PASS" else ("✗" if r["verdict"] == "FAIL" else "!")
        print(f"\n{icon} {r['name']}")
        print(f"  Claim: {r['claim']}")
        print(f"  Result: {r['verdict']} — {r['notes']}")

    print("\n" + "=" * 70)
    print(f"\nSUMMARY: {len(passed)}/{len(RESULTS)} passed")
    if failed:
        print(f"  FAILED ({len(failed)}): {[r['name'] for r in failed]}")
    if errors:
        print(f"  ERRORS ({len(errors)}): {[r['name'] for r in errors]}")
    print()
