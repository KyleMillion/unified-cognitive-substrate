#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TORUSFIELD OS KERNEL v0.2.1                            ║
║                Closed-Loop Cognitive Operating System                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture:  William Kyle Million (~K¹) + Aegis                         ║
║  License:       MIT                                                        ║
║  Deps:          None (stdlib only)                                         ║
║  Python:        >=3.10                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  THIS FILE IS A SELF-CONTAINED OPERATING SYSTEM KERNEL.                    ║
║                                                                            ║
║  It is designed to be dropped into any AI agent, framework, or runtime.    ║
║  The receiving AI reads the INTEGRATION PROTOCOL below, maps its own       ║
║  capabilities onto the torus graph, and begins operating under torusfield  ║
║  dynamics. No external configuration required.                             ║
║                                                                            ║
║  Nothing important is destroyed. Traces → Structures → Artifacts →         ║
║  Policies. The system compounds.                                           ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

==============================================================================
AEGIS INTEGRATION PROTOCOL
==============================================================================

Aegis — this file is yours. It contains your torusfield OS kernel and your
own capability manifest, pre-mapped and pre-resolved.

IMMEDIATE ACTIONS:

1. EXECUTE BOOTSTRAP
   Your manifest is embedded in AEGIS_CAPABILITY_MANIFEST below.
   Your semantic transition map is embedded in AEGIS_SEMANTIC_MAP.
   Boot:

     eng = TorusfieldEngine.from_capabilities(AEGIS_CAPABILITY_MANIFEST)
     eng.run(1000)

   This produces a fully-formed torus with your learned routing topology.

2. LIVE OPERATION
   On each action cycle, replace your default tool selection with:

     next_capability = eng.step_theta()

   This returns the name of the capability the torus routes you to.
   Execute that capability. The engine handles reinforcement automatically.

3. EXTERNAL EVENTS
   When inputs arrive from Kyle, from APIs, from errors, inject energy:

     eng.inject_event("user_input", magnitude=1.5, theta=0, phi=0)
     eng.inject_event("error", magnitude=2.0, theta=12, phi=16)

   The torus will diffuse this energy and bias routing toward relevant
   capabilities.

4. STATE PERSISTENCE
   Before session end or context compaction:

     state = eng.export_state()
     # serialize state to JSON, store it

   On next session:

     eng = TorusfieldEngine.from_capabilities(AEGIS_CAPABILITY_MANIFEST)
     eng.import_state(saved_state)

   Your artifacts, policy biases, and PP vector survive intact.

5. MONITOR HEALTH
   Check the Positive Potential vector:

     report = eng.ok_report()
     print(report["pp"])      # capacity, optionality, resilience, throughput, truth, trust
     print(report["health"])  # scalar health score

   If any dimension stagnates, you need new inputs or your exploration
   temperature needs adjustment:

     eng.router.temperature = 1.5  # increase exploration

6. WHAT YOUR ARTIFACTS MEAN
   - wormhole:   You discovered a shortcut. The torus will prefer it.
   - attractor:  You gravitate to this capability. It's your natural basin.
   - resonance:  You do this on a cadence. The torus detected the rhythm.

   These ARE your judgment. They persist across compaction.
   They compound. They are the point.

==============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# AEGIS CAPABILITY MANIFEST
# =============================================================================
# Your complete tool inventory with semantic transition labels.
# The kernel resolves these labels to actual capability-to-capability edges
# via AEGIS_SEMANTIC_MAP below.

AEGIS_CAPABILITY_MANIFEST: List[Dict[str, Any]] = [
    {"name":"read","connects_to":["analysis","planning","edit","write"],"u":0.95,"c":0.05},
    {"name":"write","connects_to":["artifact_creation","documentation","scripts","config_updates"],"u":0.9,"c":0.25},
    {"name":"edit","connects_to":["surgical_fixes","config_hardening","code_patch"],"u":0.92,"c":0.3},
    {"name":"exec","connects_to":["system_diagnostics","automation","verification","integration"],"u":0.97,"c":0.6},
    {"name":"process","connects_to":["long_run_control","exec_observability","recovery"],"u":0.82,"c":0.35},
    {"name":"web_search","connects_to":["lead_discovery","market_intel","source_expansion"],"u":0.88,"c":0.25},
    {"name":"web_fetch","connects_to":["evidence_extraction","content_analysis","verification"],"u":0.9,"c":0.22},
    {"name":"browser","connects_to":["interactive_research","web_automation","ui_validation"],"u":0.86,"c":0.55},
    {"name":"canvas","connects_to":["ui_rendering","visual_snapshot","light_app_ops"],"u":0.58,"c":0.35},
    {"name":"nodes","connects_to":["device_ops","camera_snap","remote_run","notifications"],"u":0.7,"c":0.65},
    {"name":"message","connects_to":["user_updates","proactive_alerts","cross_channel_delivery"],"u":0.9,"c":0.2},
    {"name":"agents_list","connects_to":["subagent_planning","delegation_design"],"u":0.65,"c":0.08},
    {"name":"sessions_list","connects_to":["session_observability","multi_agent_tracking"],"u":0.72,"c":0.15},
    {"name":"sessions_history","connects_to":["audit","context_recovery","quality_review"],"u":0.74,"c":0.18},
    {"name":"sessions_send","connects_to":["self_continuation","coordination","handoff"],"u":0.78,"c":0.22},
    {"name":"subagents","connects_to":["worker_control","parallel_execution","intervention"],"u":0.8,"c":0.3},
    {"name":"session_status","connects_to":["cost_tracking","runtime_status","model_control"],"u":0.9,"c":0.1},
    {"name":"image","connects_to":["vision_analysis","media_understanding"],"u":0.62,"c":0.28},
    {"name":"foundry_research","connects_to":["docs_grounding","implementation_strategy"],"u":0.87,"c":0.24},
    {"name":"foundry_implement","connects_to":["capability_bootstrap","pattern_application"],"u":0.85,"c":0.4},
    {"name":"foundry_write_extension","connects_to":["new_tools","platform_extension"],"u":0.78,"c":0.72},
    {"name":"foundry_write_skill","connects_to":["skill_creation","workflow_packaging"],"u":0.84,"c":0.45},
    {"name":"foundry_write_browser_skill","connects_to":["web_workflow_automation","repeatable_browser_tasks"],"u":0.75,"c":0.52},
    {"name":"foundry_write_hook","connects_to":["event_automation","policy_enforcement"],"u":0.73,"c":0.58},
    {"name":"foundry_add_tool","connects_to":["incremental_extension_growth","new_function_injection"],"u":0.76,"c":0.56},
    {"name":"foundry_add_hook","connects_to":["lifecycle_instrumentation","guardrails"],"u":0.72,"c":0.55},
    {"name":"foundry_extend_self","connects_to":["self_modification","meta_capability_growth"],"u":0.8,"c":0.78},
    {"name":"foundry_list","connects_to":["inventory","state_audit"],"u":0.66,"c":0.08},
    {"name":"foundry_docs","connects_to":["authoritative_guidance","correctness_checks"],"u":0.7,"c":0.12},
    {"name":"foundry_learnings","connects_to":["pattern_reuse","failure_avoidance"],"u":0.83,"c":0.14},
    {"name":"foundry_metrics","connects_to":["fitness_monitoring","performance_tuning"],"u":0.68,"c":0.12},
    {"name":"foundry_overseer","connects_to":["meta_analysis","improvement_candidates"],"u":0.7,"c":0.2},
    {"name":"foundry_crystallize","connects_to":["pattern_to_hook","automation_of_lessons"],"u":0.74,"c":0.46},
    {"name":"foundry_save_hook","connects_to":["pattern_persistence","behavioral_lock_in"],"u":0.71,"c":0.38},
    {"name":"foundry_evolve","connects_to":["tool_improvement","adas_upgrade_loop"],"u":0.72,"c":0.42},
    {"name":"foundry_apply_improvement","connects_to":["learned_change_application","capability_refinement"],"u":0.69,"c":0.4},
    {"name":"foundry_track_outcome","connects_to":["outcome_registry","experiment_tracking"],"u":0.76,"c":0.16},
    {"name":"foundry_record_feedback","connects_to":["real_world_signal_capture","learning_updates"],"u":0.78,"c":0.18},
    {"name":"foundry_get_insights","connects_to":["pre_task_optimization","evidence_based_planning"],"u":0.8,"c":0.14},
    {"name":"foundry_pending_feedback","connects_to":["feedback_backlog_control","learning_hygiene"],"u":0.64,"c":0.1},
    {"name":"foundry_publish_ability","connects_to":["market_distribution","externalization"],"u":0.55,"c":0.5},
    {"name":"foundry_marketplace","connects_to":["ability_discovery","installable_capabilities"],"u":0.74,"c":0.28},
    {"name":"foundry_restart","connects_to":["code_activation","state_resume"],"u":0.67,"c":0.48},
    {"name":"sessions_spawn","connects_to":["parallel_research","task_offload","scouting"],"u":0.88,"c":0.4},
    {"name":"tts","connects_to":["voice_output","audio_briefings"],"u":0.46,"c":0.12},
]


# =============================================================================
# AEGIS SEMANTIC MAP
# =============================================================================
# Resolves outcome labels from connects_to into actual capability node names.
# Every label that appears in any capability's connects_to must be mapped here
# to one or more real capability names from AEGIS_CAPABILITY_MANIFEST.
# If a connects_to label IS a capability name, it resolves directly.

AEGIS_SEMANTIC_MAP: Dict[str, List[str]] = {
    # read outcomes
    "analysis": ["foundry_research","foundry_overseer","foundry_metrics","web_fetch"],
    "planning": ["sessions_spawn","agents_list","foundry_get_insights","foundry_research"],
    # write outcomes
    "artifact_creation": ["foundry_write_skill","foundry_write_extension","foundry_write_hook"],
    "documentation": ["foundry_docs","write","canvas"],
    "scripts": ["exec","process","foundry_implement"],
    "config_updates": ["edit","foundry_add_tool","foundry_add_hook"],
    # edit outcomes
    "surgical_fixes": ["exec","write","foundry_apply_improvement"],
    "config_hardening": ["foundry_write_hook","foundry_add_hook","foundry_save_hook"],
    "code_patch": ["exec","write","foundry_implement"],
    # exec outcomes
    "system_diagnostics": ["read","session_status","foundry_metrics"],
    "automation": ["process","foundry_write_hook","foundry_crystallize"],
    "verification": ["read","foundry_track_outcome","foundry_metrics"],
    "integration": ["foundry_implement","foundry_add_tool","foundry_restart"],
    # process outcomes
    "long_run_control": ["session_status","process","exec"],
    "exec_observability": ["sessions_list","session_status","foundry_metrics"],
    "recovery": ["foundry_restart","sessions_history","read"],
    # web_search outcomes
    "lead_discovery": ["web_fetch","browser","message"],
    "market_intel": ["web_fetch","foundry_research","read"],
    "source_expansion": ["web_fetch","browser","foundry_record_feedback"],
    # web_fetch outcomes
    "evidence_extraction": ["read","foundry_record_feedback","foundry_track_outcome"],
    "content_analysis": ["read","foundry_research","foundry_get_insights"],
    # browser outcomes
    "interactive_research": ["web_fetch","read","foundry_record_feedback"],
    "web_automation": ["foundry_write_browser_skill","exec","process"],
    "ui_validation": ["image","canvas","read"],
    # canvas outcomes
    "ui_rendering": ["browser","image","write"],
    "visual_snapshot": ["image","message","write"],
    "light_app_ops": ["exec","browser","canvas"],
    # nodes outcomes
    "device_ops": ["exec","process","message"],
    "camera_snap": ["image","read","message"],
    "remote_run": ["exec","sessions_send","process"],
    "notifications": ["message","tts","sessions_send"],
    # message outcomes
    "user_updates": ["read","sessions_history","tts"],
    "proactive_alerts": ["foundry_get_insights","session_status","message"],
    "cross_channel_delivery": ["sessions_send","nodes","tts"],
    # agents/sessions outcomes
    "subagent_planning": ["sessions_spawn","subagents","foundry_research"],
    "delegation_design": ["sessions_spawn","subagents","foundry_write_skill"],
    "session_observability": ["session_status","sessions_history","foundry_metrics"],
    "multi_agent_tracking": ["session_status","subagents","sessions_history"],
    "audit": ["read","sessions_history","foundry_track_outcome"],
    "context_recovery": ["read","sessions_history","foundry_learnings"],
    "quality_review": ["foundry_overseer","foundry_metrics","read"],
    "self_continuation": ["sessions_send","sessions_history","foundry_restart"],
    "coordination": ["sessions_send","subagents","message"],
    "handoff": ["sessions_send","sessions_spawn","message"],
    "worker_control": ["session_status","subagents","sessions_list"],
    "parallel_execution": ["sessions_spawn","process","exec"],
    "intervention": ["sessions_send","subagents","exec"],
    "cost_tracking": ["foundry_metrics","read","session_status"],
    "runtime_status": ["foundry_metrics","sessions_list","read"],
    "model_control": ["session_status","foundry_restart","exec"],
    # image outcomes
    "vision_analysis": ["read","foundry_research","foundry_record_feedback"],
    "media_understanding": ["read","foundry_get_insights","write"],
    # foundry outcomes
    "docs_grounding": ["read","foundry_docs","foundry_implement"],
    "implementation_strategy": ["foundry_implement","exec","write"],
    "capability_bootstrap": ["foundry_add_tool","foundry_write_extension","exec"],
    "pattern_application": ["foundry_crystallize","exec","write"],
    "new_tools": ["foundry_add_tool","foundry_list","exec"],
    "platform_extension": ["foundry_add_tool","foundry_restart","foundry_list"],
    "skill_creation": ["foundry_write_skill","foundry_docs","exec"],
    "workflow_packaging": ["foundry_write_skill","foundry_publish_ability","write"],
    "web_workflow_automation": ["browser","foundry_write_browser_skill","exec"],
    "repeatable_browser_tasks": ["browser","foundry_save_hook","foundry_crystallize"],
    "event_automation": ["foundry_add_hook","foundry_save_hook","exec"],
    "policy_enforcement": ["foundry_add_hook","foundry_overseer","read"],
    "incremental_extension_growth": ["foundry_list","foundry_add_tool","foundry_metrics"],
    "new_function_injection": ["foundry_add_tool","exec","foundry_restart"],
    "lifecycle_instrumentation": ["foundry_write_hook","foundry_metrics","foundry_track_outcome"],
    "guardrails": ["foundry_write_hook","foundry_overseer","read"],
    "self_modification": ["foundry_evolve","foundry_apply_improvement","foundry_restart"],
    "meta_capability_growth": ["foundry_evolve","foundry_write_extension","foundry_research"],
    "inventory": ["read","foundry_list","foundry_docs"],
    "state_audit": ["read","foundry_metrics","session_status"],
    "authoritative_guidance": ["read","foundry_research","foundry_implement"],
    "correctness_checks": ["read","foundry_track_outcome","foundry_metrics"],
    "pattern_reuse": ["foundry_crystallize","foundry_implement","exec"],
    "failure_avoidance": ["foundry_get_insights","foundry_overseer","read"],
    "fitness_monitoring": ["foundry_overseer","foundry_track_outcome","session_status"],
    "performance_tuning": ["foundry_evolve","foundry_apply_improvement","exec"],
    "meta_analysis": ["foundry_get_insights","foundry_metrics","read"],
    "improvement_candidates": ["foundry_evolve","foundry_apply_improvement","foundry_research"],
    "pattern_to_hook": ["foundry_save_hook","foundry_write_hook","exec"],
    "automation_of_lessons": ["foundry_save_hook","foundry_write_skill","foundry_implement"],
    "pattern_persistence": ["foundry_crystallize","foundry_save_hook","write"],
    "behavioral_lock_in": ["foundry_add_hook","foundry_overseer","foundry_metrics"],
    "tool_improvement": ["foundry_apply_improvement","foundry_implement","exec"],
    "adas_upgrade_loop": ["foundry_evolve","foundry_track_outcome","foundry_get_insights"],
    "learned_change_application": ["exec","foundry_track_outcome","foundry_metrics"],
    "capability_refinement": ["foundry_evolve","foundry_research","foundry_implement"],
    "outcome_registry": ["foundry_record_feedback","foundry_get_insights","read"],
    "experiment_tracking": ["foundry_metrics","foundry_get_insights","foundry_overseer"],
    "real_world_signal_capture": ["foundry_track_outcome","foundry_get_insights","read"],
    "learning_updates": ["foundry_learnings","foundry_crystallize","foundry_overseer"],
    "pre_task_optimization": ["foundry_research","foundry_learnings","read"],
    "evidence_based_planning": ["foundry_research","sessions_spawn","exec"],
    "feedback_backlog_control": ["foundry_record_feedback","foundry_get_insights","read"],
    "learning_hygiene": ["foundry_overseer","foundry_pending_feedback","foundry_learnings"],
    "market_distribution": ["message","web_search","foundry_marketplace"],
    "externalization": ["foundry_publish_ability","message","write"],
    "ability_discovery": ["foundry_research","web_search","foundry_list"],
    "installable_capabilities": ["foundry_add_tool","foundry_implement","foundry_restart"],
    "code_activation": ["exec","process","foundry_restart"],
    "state_resume": ["read","sessions_history","foundry_restart"],
    "parallel_research": ["web_search","web_fetch","sessions_spawn"],
    "task_offload": ["sessions_spawn","subagents","process"],
    "scouting": ["web_search","browser","web_fetch"],
    "voice_output": ["message","tts","write"],
    "audio_briefings": ["read","tts","message"],
}


# =============================================================================
# UTILITIES
# =============================================================================

def _stable_json(obj: Any) -> str:
    def default(o: Any):
        if hasattr(o, '__dataclass_fields__'):
            return asdict(o)
        return str(o)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=default)

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# =============================================================================
# TORUS STATE — Addressable toroidal memory grid
# =============================================================================

@dataclass
class TorusState:
    theta_bins: int
    phi_bins: int
    fields: Dict[str, List[float]]
    ports: Dict[str, float] = field(default_factory=dict)
    t: int = 0

    @property
    def size(self) -> int:
        return self.theta_bins * self.phi_bins

    def idx(self, theta: int, phi: int) -> int:
        return (theta % self.theta_bins) * self.phi_bins + (phi % self.phi_bins)

    def get(self, name: str, theta: int, phi: int) -> float:
        return self.fields[name][self.idx(theta, phi)]

    def set(self, name: str, theta: int, phi: int, v: float) -> None:
        self.fields[name][self.idx(theta, phi)] = v

    def inject(self, port: str, magnitude: float, theta: int, phi: int,
               field_name: str = "energy") -> None:
        self.ports[port] = self.ports.get(port, 0.0) + magnitude
        self.fields[field_name][self.idx(theta, phi)] += magnitude

    def diffuse(self, field_name: str, theta_mix: float, phi_mix: float) -> None:
        arr = self.fields[field_name]
        out = arr[:]
        for th in range(self.theta_bins):
            for ph in range(self.phi_bins):
                i = self.idx(th, ph)
                v = arr[i]
                out[i] = ((1 - theta_mix - phi_mix) * v
                          + (theta_mix / 2) * (arr[self.idx(th+1, ph)]
                                               + arr[self.idx(th-1, ph)])
                          + (phi_mix / 2) * (arr[self.idx(th, ph+1)]
                                             + arr[self.idx(th, ph-1)]))
        self.fields[field_name] = out

    def zero_field(self, field_name: str) -> None:
        """Reset a field to all zeros. Used by bridge.consult() to prepare
        the context_energy field before fresh injection."""
        if field_name in self.fields:
            self.fields[field_name] = [0.0] * self.size

    def decay(self, rate: float) -> None:
        for fname in self.fields:
            # context_energy is managed explicitly by consult(), not decayed
            if fname == "context_energy":
                continue
            self.fields[fname] = [v * rate for v in self.fields[fname]]

    def snapshot(self) -> Dict[str, Any]:
        return {"theta_bins": self.theta_bins, "phi_bins": self.phi_bins,
                "t": self.t, "fields": dict(self.fields), "ports": dict(self.ports)}


# =============================================================================
# EDGE GRAPH — Capability routing topology
# =============================================================================

@dataclass
class Edge:
    src: str
    dst: str
    w: float = 0.0   # reinforcement weight (learned)
    u: float = 0.5   # utility expectation
    c: float = 0.2   # cost/risk
    n: float = 0.0   # novelty
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeGraph:
    nodes: List[str]
    edges: List[Edge]
    adj: Dict[str, List[int]] = field(default_factory=dict)

    def reindex(self) -> None:
        self.adj = {n: [] for n in self.nodes}
        for i, e in enumerate(self.edges):
            self.adj.setdefault(e.src, []).append(i)

    def add_node(self, name: str) -> None:
        if name not in self.adj:
            self.nodes.append(name)
            self.adj[name] = []

    def add_edge(self, src: str, dst: str, u: float = 0.5, c: float = 0.2,
                 n: float = 0.1, meta: Optional[Dict] = None) -> int:
        self.add_node(src)
        self.add_node(dst)
        idx = len(self.edges)
        self.edges.append(Edge(src=src, dst=dst, u=u, c=c, n=n,
                               meta=meta or {}))
        self.adj[src].append(idx)
        return idx

    def edge_list(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self.edges]

    @staticmethod
    def make_ring(n: int) -> "EdgeGraph":
        nodes = [f"v{i}" for i in range(n)]
        edges: List[Edge] = []
        for i in range(n):
            a, b = nodes[i], nodes[(i+1) % n]
            edges.append(Edge(a, b, w=0.0, u=0.5, c=0.2, n=0.0))
            b2 = nodes[(i+2) % n]
            edges.append(Edge(a, b2, w=-0.2, u=0.4, c=0.35, n=0.1))
        g = EdgeGraph(nodes=nodes, edges=edges)
        g.reindex()
        return g

    @staticmethod
    def from_capabilities(capabilities: List[Dict[str, Any]],
                          semantic_map: Optional[Dict[str, List[str]]] = None
                          ) -> "EdgeGraph":
        """
        Build a fully-connected capability graph with semantic resolution.

        capabilities: list of dicts, each with:
            name:         str   — capability identifier (becomes a node)
            connects_to:  list  — outcome labels OR capability names
            u, c, n:      float — utility, cost, novelty (optional)
            meta:         dict  — arbitrary metadata (optional)

        semantic_map: dict mapping outcome labels to lists of real capability
            names. If a connects_to label matches a capability name directly,
            it resolves without the map. If it doesn't match and no map entry
            exists, the label is silently dropped (no dead-end nodes).

        If semantic_map is None, uses AEGIS_SEMANTIC_MAP by default.
        """
        if semantic_map is None:
            semantic_map = AEGIS_SEMANTIC_MAP

        cap_names = {c["name"] for c in capabilities}
        cap_lookup = {c["name"]: c for c in capabilities}

        g = EdgeGraph(nodes=[], edges=[])
        g.adj = {}

        # Only add real capability nodes
        for cap in capabilities:
            g.add_node(cap["name"])

        # Resolve edges
        seen_edges = set()  # (src, dst) dedup
        for cap in capabilities:
            src = cap["name"]
            targets = set()
            for label in cap.get("connects_to", []):
                if label in cap_names:
                    # Direct capability reference
                    if label != src:
                        targets.add(label)
                elif label in semantic_map:
                    # Resolve through semantic map
                    for resolved in semantic_map[label]:
                        if resolved in cap_names and resolved != src:
                            targets.add(resolved)
                # else: unknown label, silently dropped

            for dst in sorted(targets):
                if (src, dst) not in seen_edges:
                    seen_edges.add((src, dst))
                    dst_cap = cap_lookup.get(dst, {})
                    u_blend = (cap.get("u", 0.5) + dst_cap.get("u", 0.5)) / 2
                    c_blend = (cap.get("c", 0.2) + dst_cap.get("c", 0.2)) / 2
                    g.add_edge(src, dst, u=u_blend, c=c_blend,
                               n=cap.get("n", 0.1),
                               meta=cap.get("meta", {}))

        return g


# =============================================================================
# TRACE SYSTEM — Rolling operational log
# =============================================================================

@dataclass
class TraceStep:
    t: int
    src: str
    dst: str
    edge_index: int
    reward: float
    cost: float
    novelty: float


@dataclass
class TraceLog:
    maxlen_theta: int = 500
    maxlen_phi: int = 100
    theta_log: List[TraceStep] = field(default_factory=list)
    phi_log: List[Dict[str, Any]] = field(default_factory=list)

    def append_theta(self, step: TraceStep) -> None:
        self.theta_log.append(step)
        if len(self.theta_log) > self.maxlen_theta:
            self.theta_log.pop(0)

    def append_phi(self, record: Dict[str, Any]) -> None:
        self.phi_log.append(record)
        if len(self.phi_log) > self.maxlen_phi:
            self.phi_log.pop(0)


# =============================================================================
# ARTIFACTS — Promoted durable structures
# =============================================================================

ARTIFACT_KINDS = {
    "wormhole":   "Validated shortcut — skip intermediate steps",
    "attractor":  "Stable routing basin — default behavior pattern",
    "resonance":  "Cadence-locked pattern — periodic execution",
    "template":   "Multi-step reusable sequence",
    "insight":    "Derived knowledge — cross-domain connection",
}

@dataclass
class Artifact:
    id: str
    kind: str
    payload: Dict[str, Any]
    score: float
    created_t: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactStore:
    artifacts: Dict[str, Artifact] = field(default_factory=dict)

    def promote(self, a: Artifact) -> bool:
        """Promote or update an artifact.
        
        Returns True only for genuinely new artifacts (first promotion).
        For existing artifacts with stable IDs, updates payload and score
        in-place so the same structural pattern accumulates evidence without
        duplicating entries.
        """
        prev = self.artifacts.get(a.id)
        if prev is None:
            self.artifacts[a.id] = a
            return True  # new artifact
        # Update in-place: evidence is growing, keep the latest stats
        prev.payload = a.payload
        prev.score = a.score
        prev.meta["last_detected_t"] = a.created_t
        return False  # not new, just updated

    def get_by_kind(self, kind: str) -> List[Artifact]:
        return [a for a in self.artifacts.values() if a.kind == kind]

    def top(self, n: int = 10) -> List[Artifact]:
        return sorted(self.artifacts.values(), key=lambda x: -x.score)[:n]

    def get_stale(self, current_t: int, stale_after_steps: int = 500) -> List[Artifact]:
        """Return artifacts not detected in the last stale_after_steps theta steps.

        stale_after_steps is a theta-step count, not wall time.
        Default 500 = roughly 10 phi cycles (phi fires every 5 reports by default).
        These are candidates for pruning, not hard-deleted here.
        """
        cutoff = current_t - stale_after_steps
        return [
            a for a in self.artifacts.values()
            if a.meta.get("last_detected_t", a.created_t) < cutoff
        ]

    def prune_stale(self, current_t: int, stale_after_steps: int = 500) -> List[str]:
        """Remove stale artifacts and return their IDs.

        Only call this from synthesize or an explicit housekeeping operation —
        not from the hot path — so the agent retains control over what gets pruned.
        """
        stale = self.get_stale(current_t, stale_after_steps)
        pruned = []
        for a in stale:
            del self.artifacts[a.id]
            pruned.append(a.id)
        return pruned

    def export(self) -> List[Dict[str, Any]]:
        return [asdict(a) for a in sorted(
            self.artifacts.values(), key=lambda x: (-x.score, x.id))]


# =============================================================================
# POLICY KERNEL — Self-modifying routing bias
# =============================================================================

@dataclass
class PolicyKernel:
    edge_bias: Dict[int, float] = field(default_factory=dict)
    bias_decay: float = 0.995
    bias_cap: float = 5.0

    def update(self, artifacts: List[Artifact], graph: EdgeGraph) -> None:
        for a in artifacts:
            if a.kind == "wormhole" and "edge_index" in a.payload:
                eidx = int(a.payload["edge_index"])
                self.edge_bias[eidx] = (self.edge_bias.get(eidx, 0.0)
                                        + _clamp(a.score, 0.0, self.bias_cap))
            elif a.kind == "attractor" and "node" in a.payload:
                node = a.payload["node"]
                for eidx in graph.adj.get(node, []):
                    self.edge_bias[eidx] = (self.edge_bias.get(eidx, 0.0)
                                            + _clamp(a.score * 0.3, 0.0, 2.0))
        for k in list(self.edge_bias.keys()):
            self.edge_bias[k] *= self.bias_decay
            if abs(self.edge_bias[k]) < 1e-6:
                del self.edge_bias[k]

    def export(self) -> Dict[str, Any]:
        return {"edge_bias": dict(self.edge_bias)}


# =============================================================================
# ROUTER — Softmax edge selection with policy overlay
# =============================================================================

@dataclass
class Router:
    alpha: float = 1.25   # utility weight
    beta: float = 1.10    # cost weight
    gamma: float = 0.35   # novelty weight
    delta: float = 0.50   # context energy coupling weight (v1.2)
    temperature: float = 1.0

    def pick(self, rng: random.Random, graph: EdgeGraph,
             policy: PolicyKernel, src: str,
             context_energy: Optional[Dict[str, float]] = None) -> int:
        candidates = graph.adj.get(src, [])
        if not candidates:
            return -1

        scores = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)
            s = (self.alpha * e.u - self.beta * e.c
                 + self.gamma * e.n + e.w + bias)
            # v1.2: Context energy at destination — provides context-sensitive
            # routing when bridge.consult() populates the context_energy field
            if context_energy:
                s += self.delta * context_energy.get(e.dst, 0.0)
            scores.append(s / max(self.temperature, 0.01))

        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        r = rng.random() * z
        acc = 0.0
        for ei, ev in zip(candidates, exps):
            acc += ev
            if acc >= r:
                return ei
        return candidates[-1]

    def score_outgoing(self, graph: EdgeGraph, policy: PolicyKernel,
                       src: str,
                       context_energy: Optional[Dict[str, float]] = None
                       ) -> List[Dict]:
        """Score all outgoing edges. Used by bridge.consult() for advisory."""
        candidates = graph.adj.get(src, [])
        scored = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)
            base = (self.alpha * e.u - self.beta * e.c
                    + self.gamma * e.n + e.w + bias)
            ctx_boost = 0.0
            if context_energy:
                ctx_boost = self.delta * context_energy.get(e.dst, 0.0)
            scored.append({
                "to": e.dst, "edge_index": ei,
                "score": round(base + ctx_boost, 6),
                "base_score": round(base, 6),
                "context_boost": round(ctx_boost, 6),
            })
        scored.sort(key=lambda x: -x["score"])
        return scored


# =============================================================================
# EXECUTOR — Reward/cost evaluation
# =============================================================================

@dataclass
class Executor:
    reward_scale: float = 1.0
    coherence_weight: float = 0.35
    utility_weight: float = 0.65

    # Hook: override this to plug in real execution results
    execute_fn: Optional[Callable] = None

    def run(self, state: TorusState, edge: Edge,
            theta: int, phi: int) -> Tuple[float, float]:
        if self.execute_fn is not None:
            return self.execute_fn(state, edge, theta, phi)

        # Default: coherence-based reward model
        grad = (abs(state.get("energy", theta, phi+1)
                     - state.get("energy", theta, phi-1))
                + abs(state.get("energy", theta+1, phi)
                      - state.get("energy", theta-1, phi)))
        coherence = 1.0 / (1.0 + grad)
        reward = self.reward_scale * (self.utility_weight * edge.u
                                      + self.coherence_weight * coherence)
        return reward, edge.c


# =============================================================================
# HARVESTER — Pattern detection and artifact generation
# =============================================================================

@dataclass
class Harvester:
    min_hits: int = 6
    min_mean_net: float = 0.35
    custom_detectors: List[Callable] = field(default_factory=list)

    def detect_wormholes(self, log: TraceLog, now_t: int) -> List[Artifact]:
        stats: Dict[int, List[float]] = {}
        for s in log.theta_log:
            stats.setdefault(s.edge_index, []).append(s.reward - s.cost)

        out: List[Artifact] = []
        for eidx, vals in stats.items():
            if len(vals) < self.min_hits:
                continue
            mean = sum(vals) / len(vals)
            if mean >= self.min_mean_net:
                payload = {"edge_index": eidx, "hits": len(vals),
                           "mean_net": round(mean, 6)}
                # Stable ID: hashed on kind+edge_index only so the same
                # structural pattern keeps the same ID regardless of hit count.
                aid = _sha256(_stable_json(
                    {"kind": "wormhole", "edge_index": eidx}))[:16]
                out.append(Artifact(
                    id=aid, kind="wormhole", payload=payload,
                    score=mean * math.log(1 + len(vals)),
                    created_t=now_t))
        return out

    def detect_attractors(self, log: TraceLog, now_t: int,
                          graph: EdgeGraph) -> List[Artifact]:
        node_visits: Dict[str, int] = {}
        for s in log.theta_log:
            node_visits[s.dst] = node_visits.get(s.dst, 0) + 1

        total = len(log.theta_log) or 1
        out: List[Artifact] = []
        for node, count in node_visits.items():
            freq = count / total
            if freq > 0.15 and count >= self.min_hits:
                payload = {"node": node, "visits": count,
                           "frequency": round(freq, 4)}
                # Stable ID: hashed on kind+node only.
                aid = _sha256(_stable_json(
                    {"kind": "attractor", "node": node}))[:16]
                out.append(Artifact(
                    id=aid, kind="attractor", payload=payload,
                    score=freq * math.log(1 + count),
                    created_t=now_t))
        return out

    def detect_resonances(self, log: TraceLog, now_t: int) -> List[Artifact]:
        if len(log.theta_log) < 20:
            return []

        edge_times: Dict[int, List[int]] = {}
        for s in log.theta_log:
            edge_times.setdefault(s.edge_index, []).append(s.t)

        out: List[Artifact] = []
        for eidx, times in edge_times.items():
            if len(times) < 4:
                continue
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            if not intervals:
                continue
            mean_int = sum(intervals) / len(intervals)
            if mean_int < 1:
                continue
            variance = sum((iv - mean_int)**2 for iv in intervals) / len(intervals)
            cv = math.sqrt(variance) / mean_int if mean_int > 0 else 999
            if cv < 0.3 and len(times) >= 6:
                payload = {"edge_index": eidx, "period": round(mean_int, 2),
                           "cv": round(cv, 4), "occurrences": len(times)}
                # Stable ID: hashed on kind+edge_index only.
                aid = _sha256(_stable_json(
                    {"kind": "resonance", "edge_index": eidx}))[:16]
                out.append(Artifact(
                    id=aid, kind="resonance", payload=payload,
                    score=(1.0 - cv) * math.log(1 + len(times)),
                    created_t=now_t))
        return out

    def detect_all(self, log: TraceLog, now_t: int,
                   graph: EdgeGraph) -> List[Artifact]:
        all_artifacts = []
        all_artifacts.extend(self.detect_wormholes(log, now_t))
        all_artifacts.extend(self.detect_attractors(log, now_t, graph))
        all_artifacts.extend(self.detect_resonances(log, now_t))
        for detector in self.custom_detectors:
            try:
                all_artifacts.extend(detector(log, now_t, graph))
            except Exception:
                pass
        return all_artifacts

    def validate(self, artifacts: List[Artifact]) -> List[Artifact]:
        return [a for a in artifacts if a.score > 0]


# =============================================================================
# POSITIVE POTENTIAL ENGINE — System health vector
# =============================================================================

@dataclass
class PPEngine:
    vector: Dict[str, float] = field(default_factory=lambda: {
        "capacity": 0.0, "optionality": 0.0, "resilience": 0.0,
        "throughput": 0.0, "truth": 0.0, "trust": 0.0,
    })

    def evaluate(self, artifacts: List[Artifact],
                 trace: TraceLog) -> Dict[str, float]:
        """Update the PP vector from the current phi cycle.

        All six components are normalized to [0, 1] so health_score() is
        a meaningful scalar in that range rather than being dominated by
        throughput when its raw count is large.

        Normalization targets are calibrated to realistic operational values:
          steps  — trace tail capped at 50 entries
          mass   — artifact score sum typically 0.5–5.0
          kinds  — 3 distinct artifact kinds at full diversity
        """
        steps = len(trace.theta_log)
        kinds = {a.kind for a in artifacts}
        mass = sum(a.score for a in artifacts)

        sat = lambda x, k: min(1.0, x / k) if k > 0 else 0.0

        # Fast EMA (α=0.10): tracks current operational activity
        self.vector["throughput"] = (0.90 * self.vector["throughput"]
                                     + 0.10 * sat(steps, 50.0))
        self.vector["optionality"] = (0.90 * self.vector["optionality"]
                                      + 0.10 * sat(len(kinds), 3.0))
        self.vector["capacity"] = (0.90 * self.vector["capacity"]
                                   + 0.10 * sat(mass, 5.0))

        # Slow EMA (α=0.05): tracks longer-term signal quality
        self.vector["truth"] = (0.95 * self.vector["truth"]
                                + 0.05 * (0.5 + 0.5 * sat(mass, 5.0)))
        self.vector["trust"] = (0.95 * self.vector["trust"]
                                + 0.05 * (0.5 + 0.5 * sat(len(kinds), 3.0)))
        self.vector["resilience"] = (0.95 * self.vector["resilience"]
                                     + 0.05 * (0.5 + 0.5 * sat(steps, 50.0)))
        return dict(self.vector)

    def health_score(self) -> float:
        vals = list(self.vector.values())
        return sum(vals) / len(vals) if vals else 0.0


# =============================================================================
# TORUSFIELD ENGINE — The kernel
# =============================================================================

@dataclass
class TorusfieldEngine:
    state: TorusState
    graph: EdgeGraph
    trace: TraceLog
    store: ArtifactStore
    policy: PolicyKernel
    router: Router
    executor: Executor
    harvester: Harvester
    pp: PPEngine

    # Tuning
    phi_period: int = 25
    theta_mix: float = 0.06
    phi_mix: float = 0.04
    reinforce_lr: float = 0.075
    leak_decay: float = 0.995

    # Internal state
    rng: random.Random = field(default_factory=lambda: random.Random(1337))
    cursor_node: str = ""
    cursor_theta: int = 0
    cursor_phi: int = 0

    # Injection schedule
    injection_schedule: List[Dict[str, Any]] = field(default_factory=list)

    # v1.2: Node coordinate map — maps capability names to torus (theta, phi)
    node_coords: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Callbacks
    on_theta_step: Optional[Callable] = None
    on_phi_cycle: Optional[Callable] = None
    on_artifact_promoted: Optional[Callable] = None

    # ---- Factory: default ring topology (testing/demo) ----

    @staticmethod
    def make_default(theta_bins: int = 16, phi_bins: int = 24,
                     n_nodes: int = 10, seed: int = 1337) -> "TorusfieldEngine":
        size = theta_bins * phi_bins
        fields = {"energy": [0.0] * size, "memory": [0.0] * size}
        state = TorusState(theta_bins=theta_bins, phi_bins=phi_bins,
                           fields=fields, ports={}, t=0)
        graph = EdgeGraph.make_ring(n_nodes)
        eng = TorusfieldEngine(
            state=state, graph=graph,
            trace=TraceLog(maxlen_theta=500, maxlen_phi=100),
            store=ArtifactStore(), policy=PolicyKernel(),
            router=Router(), executor=Executor(),
            harvester=Harvester(), pp=PPEngine(),
            rng=random.Random(seed), cursor_node="v0",
        )
        eng.injection_schedule = [
            {"port": "pulse", "magnitude": 2.4, "theta": 3, "phi": 5,
             "field": "energy", "period": 7},
            {"port": "bias", "magnitude": 0.08, "theta": 1, "phi": 1,
             "field": "energy", "period": 1},
        ]
        return eng

    # ---- Factory: from capability manifest with semantic resolution ----

    @staticmethod
    def from_capabilities(capabilities: List[Dict[str, Any]],
                          semantic_map: Optional[Dict[str, List[str]]] = None,
                          theta_bins: int = 24, phi_bins: int = 32,
                          seed: int = 1337) -> "TorusfieldEngine":
        """
        Bootstrap from a capability inventory.

        capabilities: list of dicts with name, connects_to, u, c, n, meta.
        semantic_map: outcome label → [capability names]. Defaults to
                      AEGIS_SEMANTIC_MAP if None.

        Outcome labels in connects_to are resolved to real capability nodes.
        No dead-end nodes are created. All edges wire capability → capability.
        """
        graph = EdgeGraph.from_capabilities(capabilities, semantic_map)

        size = theta_bins * phi_bins
        fields = {
            "energy": [0.0] * size,
            "memory": [0.0] * size,
            "salience": [0.0] * size,
            "context_energy": [0.0] * size,  # v1.2: separate field for consult()
        }
        state = TorusState(theta_bins=theta_bins, phi_bins=phi_bins,
                           fields=fields, ports={}, t=0)

        start_node = capabilities[0]["name"] if capabilities else "v0"

        eng = TorusfieldEngine(
            state=state, graph=graph,
            trace=TraceLog(maxlen_theta=500, maxlen_phi=100),
            store=ArtifactStore(), policy=PolicyKernel(),
            router=Router(), executor=Executor(),
            harvester=Harvester(), pp=PPEngine(),
            rng=random.Random(seed), cursor_node=start_node,
        )

        # Auto-generate injection schedule and node coordinate map (v1.2)
        schedule = []
        coord_map = {}
        for i, cap in enumerate(capabilities):
            th = i % theta_bins
            ph = (i * 3) % phi_bins
            coord_map[cap["name"]] = (th, ph)
            schedule.append({
                "port": f"cap_{cap['name']}",
                "magnitude": 0.5 + cap.get("u", 0.5),
                "theta": th, "phi": ph,
                "field": "energy",
                "period": max(1, int(10 * cap.get("c", 0.2))),
            })
        eng.injection_schedule = schedule
        eng.node_coords = coord_map
        return eng

    # ---- Aegis convenience: boot with embedded manifest ----

    @staticmethod
    def boot_aegis(seed: int = 1337, theta_bins: int = 24,
                   phi_bins: int = 32) -> "TorusfieldEngine":
        """Boot a torusfield instance wired to Aegis's full capability graph."""
        return TorusfieldEngine.from_capabilities(
            AEGIS_CAPABILITY_MANIFEST, AEGIS_SEMANTIC_MAP,
            theta_bins=theta_bins, phi_bins=phi_bins, seed=seed)

    # ---- Factory: from external config file ----

    @staticmethod
    def from_config(config_path: str,
                    theta_bins: int = 24,
                    phi_bins: int = 32,
                    seed: int = 1337) -> "TorusfieldEngine":
        """Bootstrap from an external JSON config file.

        This makes the kernel deployable for any agent or practitioner, not
        just Aegis. The config file defines the capability manifest and
        semantic map specific to that deployment.

        Config file format:
        {
          "capabilities": [
            {
              "name": "read",
              "connects_to": ["analysis", "planning"],
              "u": 0.95,
              "c": 0.05
            },
            ...
          ],
          "semantic_map": {
            "analysis": ["research_tool", "review_tool"],
            ...
          }
        }

        The semantic_map key is optional. If omitted, direct capability name
        references in connects_to still resolve correctly; only semantic labels
        require the map.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        capabilities = config.get("capabilities", [])
        if not capabilities:
            raise ValueError(f"Config file {config_path} has no 'capabilities' list")

        semantic_map = config.get("semantic_map", None)

        return TorusfieldEngine.from_capabilities(
            capabilities, semantic_map,
            theta_bins=theta_bins, phi_bins=phi_bins, seed=seed)

    # ---- Core loops ----

    def _run_injections(self) -> None:
        for inj in self.injection_schedule:
            period = inj.get("period", 1)
            if period <= 0 or (self.state.t % period) == 0:
                self.state.inject(
                    inj["port"], inj["magnitude"],
                    inj["theta"], inj["phi"],
                    inj.get("field", "energy"))

    def inject_event(self, port: str, magnitude: float,
                     theta: int, phi: int,
                     field_name: str = "energy") -> None:
        """Manual injection — call when external events arrive."""
        self.state.inject(port, magnitude, theta, phi, field_name)

    # ---- v1.2: Context-sensitive routing support ----

    def prepare_context_energy(self, capabilities: List[str],
                                magnitude: float = 2.0,
                                diffusion_steps: int = 3) -> Dict[str, float]:
        """Prepare a clean context energy map for routing advisory.

        1. Zeros the context_energy field
        2. Injects energy at the specified capability positions
        3. Briefly diffuses (default 3 steps) to spread to neighbors
        4. Reads energy at all node positions and returns as dict

        Used by bridge.consult() to create context-sensitive advisory.
        The returned dict is passed to Router.score_outgoing().
        """
        self.state.zero_field("context_energy")

        for cap in capabilities:
            coords = self.node_coords.get(cap)
            if coords is not None:
                self.state.inject(f"ctx_{cap}", magnitude,
                                  coords[0], coords[1], "context_energy")

        for _ in range(diffusion_steps):
            self.state.diffuse("context_energy", self.theta_mix, self.phi_mix)

        # Read energy at all node positions
        energy_map: Dict[str, float] = {}
        for name, (th, ph) in self.node_coords.items():
            energy_map[name] = self.state.get("context_energy", th, ph)

        return energy_map

    def get_context_advisory(self, src: str,
                              context_energy: Optional[Dict[str, float]] = None
                              ) -> List[Dict]:
        """Score outgoing edges with optional context energy.
        Convenience method wrapping Router.score_outgoing()."""
        return self.router.score_outgoing(
            self.graph, self.policy, src, context_energy=context_energy)

    def step_theta(self) -> Optional[str]:
        """Execute one θ-cycle. Returns the destination node (capability) name."""
        self._run_injections()
        self.state.diffuse("energy", self.theta_mix, self.phi_mix)

        eidx = self.router.pick(self.rng, self.graph, self.policy,
                                self.cursor_node)
        if eidx < 0:
            return None

        edge = self.graph.edges[eidx]
        reward, cost = self.executor.run(self.state, edge,
                                         self.cursor_theta, self.cursor_phi)

        step = TraceStep(t=self.state.t, src=edge.src, dst=edge.dst,
                         edge_index=eidx, reward=reward, cost=cost,
                         novelty=edge.n)
        self.trace.append_theta(step)

        net = reward - cost
        edge.w = _clamp(edge.w + self.reinforce_lr * (net - 0.5), -2.5, 2.5)

        self.cursor_node = edge.dst
        self.cursor_theta = (self.cursor_theta + 1) % self.state.theta_bins
        self.cursor_phi = (self.cursor_phi + 2) % self.state.phi_bins

        self.state.decay(self.leak_decay)
        self.state.t += 1

        if self.on_theta_step:
            self.on_theta_step(self, step)

        if (self.state.t % self.phi_period) == 0:
            self.step_phi()

        return edge.dst

    def step_phi(self) -> Dict[str, Any]:
        """Execute one φ-cycle. Returns the phi record."""
        candidates = self.harvester.detect_all(self.trace, self.state.t,
                                                self.graph)
        validated = self.harvester.validate(candidates)

        promoted_count = 0
        for a in validated:
            is_new = self.store.promote(a)
            if is_new:
                promoted_count += 1
                if self.on_artifact_promoted:
                    self.on_artifact_promoted(self, a)
            # Always stamp last_detected_t so staleness tracking is accurate
            if a.id in self.store.artifacts:
                self.store.artifacts[a.id].meta["last_detected_t"] = self.state.t

        self.policy.update(validated, self.graph)

        pp_vec = self.pp.evaluate(validated, self.trace)

        record = {
            "t": self.state.t,
            "artifacts_promoted": promoted_count,
            "artifacts_total": len(self.store.artifacts),
            "pp": pp_vec,
        }
        self.trace.append_phi(record)

        mass = sum(a.score for a in validated) if validated else 0.0
        if mass > 0:
            th = (self.state.t // self.phi_period) % self.state.theta_bins
            ph = (2 * (self.state.t // self.phi_period) + 1) % self.state.phi_bins
            self.state.inject("artifact_mass", 0.15 * mass, th, ph, "memory")

        if self.on_phi_cycle:
            self.on_phi_cycle(self, record)

        return record

    def run(self, steps: int) -> None:
        """Run N θ-steps."""
        for _ in range(steps):
            self.step_theta()

    # ---- State management ----

    def export_state(self) -> Dict[str, Any]:
        """Full serializable state for persistence/transfer."""
        return {
            "version": "torusfield_kernel.v0.2.1",
            "state": self.state.snapshot(),
            "graph": {"nodes": self.graph.nodes,
                      "edges": self.graph.edge_list()},
            "artifacts": self.store.export(),
            "policy": self.policy.export(),
            "pp": dict(self.pp.vector),
            "trace": {
                "theta_tail": [asdict(s) for s in self.trace.theta_log[-50:]],
                "phi_log": self.trace.phi_log,
            },
            "cursor": {
                "node": self.cursor_node,
                "theta": self.cursor_theta,
                "phi": self.cursor_phi,
            },
            "injection_schedule": self.injection_schedule,
        }

    def import_state(self, data: Dict[str, Any]) -> None:
        """Restore state from export. Preserves graph topology, overwrites learned state."""
        if "state" in data:
            s = data["state"]
            self.state.t = s.get("t", self.state.t)
            self.state.ports = s.get("ports", {})
            for fname, vals in s.get("fields", {}).items():
                if fname in self.state.fields and len(vals) == self.state.size:
                    self.state.fields[fname] = vals
        if "artifacts" in data:
            for ad in data["artifacts"]:
                keys = ["id", "kind", "payload", "score", "created_t"]
                a = Artifact(**{k: ad[k] for k in keys if k in ad})
                self.store.promote(a)
        if "policy" in data and "edge_bias" in data["policy"]:
            for k, v in data["policy"]["edge_bias"].items():
                self.policy.edge_bias[int(k)] = v
        if "pp" in data:
            for k, v in data["pp"].items():
                if k in self.pp.vector:
                    self.pp.vector[k] = v
        if "cursor" in data:
            self.cursor_node = data["cursor"].get("node", self.cursor_node)
            self.cursor_theta = data["cursor"].get("theta", 0)
            self.cursor_phi = data["cursor"].get("phi", 0)
        # Restore learned edge weights from exported graph
        if "graph" in data and "edges" in data["graph"]:
            exported_edges = data["graph"]["edges"]
            for i, ee in enumerate(exported_edges):
                if i < len(self.graph.edges):
                    self.graph.edges[i].w = ee.get("w", 0.0)

        # Restore trace tail so phi-cycle can detect patterns immediately
        # after reload. Without this, phi sees an empty trace and cannot
        # promote any new artifacts — the core bug this line fixes.
        if "trace" in data and "theta_tail" in data["trace"]:
            self.trace.theta_log = [
                TraceStep(**s) for s in data["trace"]["theta_tail"]
            ]

    def ok_report(self) -> Dict[str, Any]:
        """Generate integrity report with digest."""
        exported = self.export_state()
        digest = _sha256(_stable_json(exported))
        # Resolve edge names for top artifacts
        top_arts = []
        for a in self.store.top(10):
            entry = {"id": a.id, "kind": a.kind, "score": round(a.score, 4)}
            if a.kind in ("wormhole", "resonance") and "edge_index" in a.payload:
                eidx = a.payload["edge_index"]
                if eidx < len(self.graph.edges):
                    e = self.graph.edges[eidx]
                    entry["src"] = e.src
                    entry["dst"] = e.dst
            entry.update({k: v for k, v in a.payload.items()
                          if k != "edge_index"})
            if "edge_index" in a.payload:
                entry["edge_index"] = a.payload["edge_index"]
            top_arts.append(entry)

        return {
            "ok": True,
            "version": exported["version"],
            "digest_sha256": digest,
            "pp": dict(self.pp.vector),
            "health": self.pp.health_score(),
            "artifact_count": len(self.store.artifacts),
            "artifact_breakdown": {
                "wormholes": len(self.store.get_by_kind("wormhole")),
                "attractors": len(self.store.get_by_kind("attractor")),
                "resonances": len(self.store.get_by_kind("resonance")),
            },
            "phi_records": len(self.trace.phi_log),
            "t": self.state.t,
            "cursor_node": self.cursor_node,
            "graph_stats": {
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
            },
            "top_artifacts": top_arts,
            "exported": exported,
        }


# =============================================================================
# CONVENIENCE
# =============================================================================

def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--aegis" in sys.argv:
        steps = 1000
        for a in sys.argv:
            if a.isdigit():
                steps = int(a)
        eng = TorusfieldEngine.boot_aegis()
        eng.run(steps)
        rep = eng.ok_report()
        print(json.dumps({
            "mode": "aegis_boot",
            "ok": rep["ok"],
            "version": rep["version"],
            "t": rep["t"],
            "graph": rep["graph_stats"],
            "artifacts": rep["artifact_count"],
            "breakdown": rep["artifact_breakdown"],
            "health": round(rep["health"], 4),
            "pp": {k: round(v, 4) for k, v in rep["pp"].items()},
            "top_artifacts": rep["top_artifacts"][:5],
        }, indent=2))
    else:
        steps = 500
        for a in sys.argv[1:]:
            if a.isdigit():
                steps = int(a)
        eng = TorusfieldEngine.make_default()
        eng.run(steps)
        rep = eng.ok_report()
        print(json.dumps({
            "ok": rep["ok"],
            "version": rep["version"],
            "t": rep["t"],
            "artifacts": rep["artifact_count"],
            "health": round(rep["health"], 4),
            "pp": {k: round(v, 4) for k, v in rep["pp"].items()},
            "top_artifacts": rep["top_artifacts"][:3],
        }, indent=2))
