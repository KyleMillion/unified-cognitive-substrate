#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                UNIFIED COGNITIVE SUBSTRATE (UCS) BRIDGE v1.0               ║
║          Torusfield OS + Emergent Judgment — Fused Integration Layer        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture:  William Kyle Million (~K¹) + Claude (Anthropic)            ║
║  License:       MIT                                                        ║
║  Deps:          torusfield_kernel.py (adjacent, imported)                  ║
║  Python:        >=3.10                                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  This bridge fuses two cognitive systems into one:                         ║
║                                                                            ║
║  TORUSFIELD OS — Quantitative routing engine. Learns which capability      ║
║  transitions produce value through reinforcement on a toroidal graph.      ║
║                                                                            ║
║  EMERGENT JUDGMENT — Qualitative metacognitive framework. Preserves the    ║
║  reasoning texture that develops through experience and is destroyed by    ║
║  compaction.                                                               ║
║                                                                            ║
║  The bridge exposes 8 operations:                                          ║
║    init, consult, report, reflect, flush, resume, synthesize, status       ║
║                                                                            ║
║  The agent reads SKILL.md, calls bridge.py via exec, and receives JSON.    ║
║  The agent stays in control. The bridge advises.                           ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import Torusfield kernel (adjacent file, unchanged)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from torusfield_kernel import (
    TorusfieldEngine,
    AEGIS_CAPABILITY_MANIFEST,
    AEGIS_SEMANTIC_MAP,
    TraceStep,
    Artifact,
    _clamp,
    _sha256,
    _stable_json,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class UCSConfig:
    """All paths and tuning parameters for the UCS bridge.

    To deploy for a different agent or practitioner (not Aegis), set
    manifest_path to a JSON file containing 'capabilities' and optionally
    'semantic_map'. If None, the Aegis manifest embedded in torusfield_kernel
    is used.

    Example:
        config = UCSConfig(
            workspace_root=Path("/home/dean/.ucs"),
            manifest_path=Path("/home/dean/legal_agent_manifest.json")
        )
    """

    workspace_root: Path = field(default_factory=lambda: Path.home() / ".ucs")
    manifest_path: Optional[Path] = None  # None = use Aegis embedded manifest

    # Tuning
    theta_warmup_steps: int = 15          # θ-steps on consult to diffuse energy
    phi_report_interval: int = 5          # reports between φ-cycles
    init_warmup_steps: int = 200          # θ-steps on first boot
    pp_snapshot_interval: int = 3         # flush calls between PP snapshots
    max_methodology_hits: int = 5         # max entries returned from methodology search
    max_dead_end_hits: int = 3            # max entries returned from dead-end search
    advisory_top_n: int = 6              # top N paths in consult advisory
    consult_energy_magnitude: float = 1.5 # energy injected per matched capability
    report_energy_magnitude: float = 1.0  # energy injected on action report

    # Reward model
    reward_base: Dict[str, float] = field(default_factory=lambda: {
        "success": 0.80,
        "partial": 0.45,
        "neutral": 0.30,
        "failure": 0.10,
    })
    reward_edge_blend: float = 0.3  # how much edge.u influences reward

    # Derived paths (set in __post_init__)
    state_dir: Path = field(init=False)
    knowledge_dir: Path = field(init=False)
    working_state_dir: Path = field(init=False)
    synthesis_dir: Path = field(init=False)

    def __post_init__(self):
        ucs_root = self.workspace_root / "ucs"
        self.state_dir = ucs_root / "state"
        self.knowledge_dir = ucs_root / "knowledge"
        self.working_state_dir = ucs_root / "working-state"
        self.synthesis_dir = self.knowledge_dir / "synthesis"

    # File paths
    @property
    def torusfield_state_path(self) -> Path:
        return self.state_dir / "torusfield_state.json"

    @property
    def ucs_index_path(self) -> Path:
        return self.state_dir / "ucs_index.json"

    @property
    def pp_history_path(self) -> Path:
        return self.state_dir / "pp_history.json"

    @property
    def action_log_path(self) -> Path:
        return self.state_dir / "action_log.jsonl"

    @property
    def methodology_path(self) -> Path:
        return self.knowledge_dir / "methodology.md"

    @property
    def experiments_path(self) -> Path:
        return self.knowledge_dir / "experiments.md"

    @property
    def dead_ends_path(self) -> Path:
        return self.knowledge_dir / "dead-ends.md"


# ============================================================================
# KEYWORD-CAPABILITY MAPPING
# ============================================================================

def build_keyword_map(
    semantic_map: Dict[str, List[str]],
    capabilities: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Build an inverted index: keyword → [capability_names].

    Splits semantic labels on underscores to get keywords, then maps those
    keywords to the capabilities they resolve to.  Also includes capability
    names themselves as keywords.
    """
    cap_names = {c["name"] for c in capabilities}
    kw_map: Dict[str, set] = {}

    # From semantic map labels
    for label, targets in semantic_map.items():
        words = label.lower().replace("-", "_").split("_")
        valid_targets = [t for t in targets if t in cap_names]
        for w in words:
            if len(w) >= 3:  # skip very short fragments
                kw_map.setdefault(w, set()).update(valid_targets)

    # Capability names are also keywords
    for name in cap_names:
        parts = name.lower().replace("-", "_").split("_")
        for p in parts:
            if len(p) >= 3:
                kw_map.setdefault(p, set()).add(name)
        kw_map.setdefault(name, set()).add(name)

    # Manual high-value additions
    manual = {
        "research": ["web_search", "web_fetch", "browser", "foundry_research"],
        "search": ["web_search", "web_fetch"],
        "write": ["write", "foundry_write_skill", "foundry_write_extension"],
        "code": ["exec", "write", "edit", "foundry_implement"],
        "debug": ["exec", "read", "edit", "foundry_implement"],
        "monitor": ["session_status", "foundry_metrics", "foundry_overseer"],
        "deploy": ["exec", "process", "foundry_restart"],
        "automate": ["foundry_write_hook", "foundry_crystallize", "process"],
        "create": ["write", "foundry_write_skill", "foundry_write_extension"],
        "analyze": ["read", "foundry_research", "foundry_overseer"],
        "fetch": ["web_fetch", "browser"],
        "browse": ["browser", "web_fetch"],
        "message": ["message", "sessions_send", "tts"],
        "delegate": ["sessions_spawn", "subagents", "agents_list"],
        "optimize": ["foundry_evolve", "foundry_apply_improvement", "foundry_metrics"],
        "learn": ["foundry_learnings", "foundry_get_insights", "foundry_record_feedback"],
        "test": ["exec", "foundry_track_outcome", "foundry_metrics"],
        "publish": ["foundry_publish_ability", "foundry_marketplace"],
        "fix": ["edit", "exec", "foundry_apply_improvement"],
        "plan": ["foundry_research", "foundry_get_insights", "sessions_spawn"],
        "audit": ["read", "sessions_history", "foundry_overseer"],
        "track": ["foundry_track_outcome", "foundry_metrics", "session_status"],
    }
    for word, caps in manual.items():
        valid = [c for c in caps if c in cap_names]
        kw_map.setdefault(word, set()).update(valid)

    return {k: sorted(v) for k, v in kw_map.items()}


# Build once at module level
_KEYWORD_MAP = build_keyword_map(AEGIS_SEMANTIC_MAP, AEGIS_CAPABILITY_MANIFEST)
_CAP_NAMES = {c["name"] for c in AEGIS_CAPABILITY_MANIFEST}


def resolve_capabilities_from_context(
    context: str,
    explicit_caps: Optional[List[str]] = None,
    methodology_entries: Optional[List[Dict]] = None,
) -> List[str]:
    """Extract capability names relevant to a free-text context.

    Three resolution layers:
    1. Explicit capability names passed directly.
    2. Static keyword map built from the semantic manifest.
    3. Dynamic vocabulary from stored methodology entries — if the agent
       has learned that certain keywords associate with certain capabilities
       through actual work, that learned vocabulary enriches resolution here.
    """
    caps: set = set()

    # Explicit capabilities
    if explicit_caps:
        caps.update(c for c in explicit_caps if c in _CAP_NAMES)

    # Keyword matching from static map
    words = set(re.findall(r"[a-z_]{3,}", context.lower()))
    for w in words:
        if w in _KEYWORD_MAP:
            caps.update(_KEYWORD_MAP[w])

    # Dynamic enrichment from methodology index
    # If stored methodology entries have keywords that overlap with the current
    # context, their associated capabilities get pulled in too.
    if methodology_entries:
        for entry in methodology_entries:
            entry_kw = {w.lower() for w in entry.get("keywords", [])}
            if entry_kw & words:  # any keyword overlap
                for cap in entry.get("capabilities", []):
                    if cap in _CAP_NAMES:
                        caps.add(cap)

    return sorted(caps)


# ============================================================================
# REWARD MODEL
# ============================================================================

class RewardModel:
    """Maps qualitative outcome assessments to numeric reward signals."""

    def __init__(self, config: UCSConfig):
        self.base_rewards = config.reward_base
        self.edge_blend = config.reward_edge_blend

    def compute(self, success: str, edge_utility: float) -> Tuple[float, float]:
        """
        Returns (reward, cost_weight).

        reward:  blended from success enum + edge utility expectation
        cost_weight:  1.0 (full edge cost applies)
        """
        base = self.base_rewards.get(success, 0.30)
        reward = (1.0 - self.edge_blend) * base + self.edge_blend * edge_utility
        return reward, 1.0


# ============================================================================
# ANNOTATION STORE (UCS INDEX)
# ============================================================================

class AnnotationStore:
    """
    Manages the UCS index file: artifact annotations, methodology entries,
    dead-end entries, and operational counters.
    """

    EMPTY_INDEX = {
        "version": "ucs.v1.0",
        "annotations": {},
        "methodology_entries": [],
        "dead_end_entries": [],
        "report_counter": 0,
        "last_phi_t": 0,
        "last_synthesis_date": None,
        "policy_overrides": [],
    }

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}

    def load(self) -> None:
        if self.path.exists():
            with open(self.path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = dict(self.EMPTY_INDEX)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    # -- Annotations --

    def get_annotation(self, artifact_id: str) -> Optional[Dict]:
        return self.data["annotations"].get(artifact_id)

    def set_annotation(self, artifact_id: str, annotation: Dict) -> None:
        self.data["annotations"][artifact_id] = annotation

    def get_unannotated(self, artifact_ids: List[str]) -> List[str]:
        return [aid for aid in artifact_ids if aid not in self.data["annotations"]]

    # -- Methodology --

    def add_methodology_entry(self, entry: Dict) -> str:
        entry_id = _sha256(_stable_json(entry))[:12]
        entry["id"] = entry_id
        self.data["methodology_entries"].append(entry)
        return entry_id

    def search_methodology(
        self, keywords: List[str], capabilities: List[str], max_results: int = 5
    ) -> List[Dict]:
        results = []
        kw_set = {w.lower() for w in keywords}
        cap_set = set(capabilities)

        for entry in self.data["methodology_entries"]:
            score = 0
            entry_kw = {w.lower() for w in entry.get("keywords", [])}
            entry_caps = set(entry.get("capabilities", []))
            score += len(kw_set & entry_kw) * 2
            score += len(cap_set & entry_caps) * 3
            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:max_results]]

    def recent_methodology(self, n: int = 5) -> List[Dict]:
        return self.data["methodology_entries"][-n:]

    # -- Dead ends --

    def add_dead_end(self, entry: Dict) -> str:
        entry_id = _sha256(_stable_json(entry))[:12]
        entry["id"] = entry_id
        self.data["dead_end_entries"].append(entry)
        return entry_id

    def search_dead_ends(
        self, keywords: List[str], capabilities: List[str], max_results: int = 3
    ) -> List[Dict]:
        results = []
        kw_set = {w.lower() for w in keywords}
        cap_set = set(capabilities)

        for entry in self.data["dead_end_entries"]:
            score = 0
            entry_kw = {w.lower() for w in entry.get("keywords", [])}
            entry_caps = set(entry.get("capabilities", []))
            score += len(kw_set & entry_kw) * 2
            score += len(cap_set & entry_caps) * 3
            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:max_results]]

    # -- Counters --

    @property
    def report_counter(self) -> int:
        return self.data.get("report_counter", 0)

    @report_counter.setter
    def report_counter(self, val: int) -> None:
        self.data["report_counter"] = val

    @property
    def last_phi_t(self) -> int:
        return self.data.get("last_phi_t", 0)

    @last_phi_t.setter
    def last_phi_t(self, val: int) -> None:
        self.data["last_phi_t"] = val

    @property
    def last_synthesis_date(self) -> Optional[str]:
        return self.data.get("last_synthesis_date")

    @last_synthesis_date.setter
    def last_synthesis_date(self, val: Optional[str]) -> None:
        self.data["last_synthesis_date"] = val

    def add_policy_override(self, record: Dict) -> None:
        overrides = self.data.setdefault("policy_overrides", [])
        overrides.append(record)
        # Keep last 100
        if len(overrides) > 100:
            self.data["policy_overrides"] = overrides[-100:]


# ============================================================================
# PP HISTORY
# ============================================================================

class PPHistory:
    """Tracks Positive Potential vector over time for trend analysis."""

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {"snapshots": []}

    def load(self) -> None:
        if self.path.exists():
            with open(self.path, "r") as f:
                self.data = json.load(f)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_snapshot(self, t: int, pp: Dict[str, float],
                     health: float, artifact_count: int) -> None:
        self.data["snapshots"].append({
            "t": t,
            "timestamp": _now_iso(),
            "pp": pp,
            "health": round(health, 4),
            "artifact_count": artifact_count,
        })
        # Keep last 500
        if len(self.data["snapshots"]) > 500:
            self.data["snapshots"] = self.data["snapshots"][-500:]

    def trend(self, last_n: int = 10) -> str:
        snaps = self.data["snapshots"]
        if len(snaps) < 2:
            return "insufficient_data"
        recent = snaps[-last_n:]
        if len(recent) < 2:
            return "insufficient_data"
        first_health = recent[0]["health"]
        last_health = recent[-1]["health"]
        delta = last_health - first_health
        if delta > 0.05:
            return "improving"
        elif delta < -0.05:
            return "declining"
        return "stable"


# ============================================================================
# ACTION LOG
# ============================================================================

class ActionLog:
    """Append-only JSONL log of agent actions for synthesis."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: Dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_since(self, since_iso: Optional[str] = None) -> List[Dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if since_iso and entry.get("timestamp", "") < since_iso:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    def read_all(self) -> List[Dict]:
        return self.read_since(None)


# ============================================================================
# UTILITIES
# ============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _append_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write("\n" + text + "\n")


def _read_md(path: Path) -> str:
    if path.exists():
        with open(path, "r") as f:
            return f.read()
    return ""


# ============================================================================
# UCS BRIDGE — The main orchestrator
# ============================================================================

class UCSBridge:
    """
    Unified Cognitive Substrate bridge.

    Fuses Torusfield OS (quantitative routing) and Emergent Judgment
    (qualitative metacognition) into a single interface.
    """

    def __init__(self, config: UCSConfig):
        self.config = config
        self.reward_model = RewardModel(config)
        self.index = AnnotationStore(config.ucs_index_path)
        self.pp_history = PPHistory(config.pp_history_path)
        self.action_log = ActionLog(config.action_log_path)
        self.engine: Optional[TorusfieldEngine] = None

    # ----- Engine lifecycle -----

    def _load_engine(self) -> TorusfieldEngine:
        """Boot engine from persisted state.

        If a manifest_path is configured, boots from that external manifest
        so this bridge instance is not tied to the Aegis capability set.
        """
        if self.config.manifest_path and self.config.manifest_path.exists():
            eng = TorusfieldEngine.from_config(str(self.config.manifest_path))
        else:
            eng = TorusfieldEngine.boot_aegis()
        if self.config.torusfield_state_path.exists():
            state_data = _read_json(self.config.torusfield_state_path)
            eng.import_state(state_data)
        self.engine = eng
        return eng

    def _save_engine(self) -> None:
        """Persist engine state to disk."""
        if self.engine is None:
            return
        state = self.engine.export_state()
        _write_json(self.config.torusfield_state_path, state)

    # ----- Edge finding -----

    def _find_edge(self, eng: TorusfieldEngine, src: str, dst: str) -> Optional[int]:
        """Find direct edge from src to dst. Returns edge index or None."""
        for eidx in eng.graph.adj.get(src, []):
            if eng.graph.edges[eidx].dst == dst:
                return eidx
        return None

    def _find_any_edge_to(self, eng: TorusfieldEngine, dst: str) -> Optional[int]:
        """Find any edge that leads to dst (for when cursor has no direct edge)."""
        for i, e in enumerate(eng.graph.edges):
            if e.dst == dst:
                return i
        return None

    # ----- Capability-to-torus position mapping -----

    def _cap_torus_pos(self, eng: TorusfieldEngine, cap_name: str) -> Tuple[int, int]:
        """Map a capability name to a (theta, phi) position on the torus."""
        # v1.2: Use engine's pre-computed coordinate map
        if eng.node_coords and cap_name in eng.node_coords:
            return eng.node_coords[cap_name]
        # Fallback for capabilities not in manifest
        cap_names = [c["name"] for c in AEGIS_CAPABILITY_MANIFEST]
        try:
            idx = cap_names.index(cap_name)
        except ValueError:
            idx = hash(cap_name) % len(cap_names)
        theta = idx % eng.state.theta_bins
        phi = (idx * 3) % eng.state.phi_bins
        return theta, phi

    # ----- Outgoing edge scoring -----

    def _score_edges(self, eng: TorusfieldEngine, src: str,
                     context_energy: Optional[Dict[str, float]] = None
                     ) -> List[Dict]:
        """Score all outgoing edges from src node, sorted by score descending.
        
        v1.2: If context_energy is provided, includes delta * energy_at_dst
        in scoring, producing context-sensitive advisory.
        """
        candidates = eng.graph.adj.get(src, [])
        scored = []
        for eidx in candidates:
            e = eng.graph.edges[eidx]
            bias = eng.policy.edge_bias.get(eidx, 0.0)
            base_score = (eng.router.alpha * e.u
                     - eng.router.beta * e.c
                     + eng.router.gamma * e.n
                     + e.w + bias)
            # v1.2: Context energy boost
            ctx_boost = 0.0
            if context_energy:
                ctx_boost = eng.router.delta * context_energy.get(e.dst, 0.0)
            score = base_score + ctx_boost

            annotation = None
            # Check all artifacts for wormholes on this edge
            for a in eng.store.artifacts.values():
                if a.kind == "wormhole" and a.payload.get("edge_index") == eidx:
                    ann = self.index.get_annotation(a.id)
                    if ann:
                        annotation = ann.get("generalized_pattern") or ann.get("text")
                    break

            scored.append({
                "to": e.dst,
                "edge_index": eidx,
                "score": round(score, 4),
                "context_boost": round(ctx_boost, 4),
                "components": {
                    "utility": round(e.u, 3),
                    "cost": round(-e.c, 3),
                    "weight": round(e.w, 3),
                    "novelty": round(e.n, 3),
                    "bias": round(bias, 3),
                    "context_energy": round(ctx_boost, 3),
                },
                "annotation": annotation,
            })
        scored.sort(key=lambda x: -x["score"])
        return scored

    # ----- Artifact enrichment -----

    def _enrich_artifacts(self, eng: TorusfieldEngine,
                          kind: str, limit: int = 5) -> List[Dict]:
        """Get artifacts of a kind, enriched with annotations and edge names."""
        arts = eng.store.get_by_kind(kind)
        arts.sort(key=lambda a: -a.score)
        enriched = []
        for a in arts[:limit]:
            entry = {
                "id": a.id,
                "score": round(a.score, 4),
            }
            if kind in ("wormhole", "resonance") and "edge_index" in a.payload:
                eidx = a.payload["edge_index"]
                if eidx < len(eng.graph.edges):
                    e = eng.graph.edges[eidx]
                    entry["from"] = e.src
                    entry["to"] = e.dst
                entry["hits"] = a.payload.get("hits", 0)
                entry["mean_net"] = a.payload.get("mean_net", 0)
            elif kind == "attractor" and "node" in a.payload:
                entry["node"] = a.payload["node"]
                entry["frequency"] = a.payload.get("frequency", 0)

            ann = self.index.get_annotation(a.id)
            entry["annotation"] = ann.get("text") if ann else None
            entry["annotated"] = ann is not None
            enriched.append(entry)
        return enriched

    # ================================================================
    # OPERATIONS
    # ================================================================

    def op_init(self, force: bool = False) -> Dict:
        """Initialize workspace and bootstrap engine."""
        ws = self.config.workspace_root

        if self.config.torusfield_state_path.exists() and not force:
            return {
                "status": "already_initialized",
                "workspace": str(ws),
                "message": "Workspace exists. Use --force to reinitialize.",
            }

        # Create directory tree
        for d in [self.config.state_dir, self.config.knowledge_dir,
                  self.config.working_state_dir, self.config.synthesis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Bootstrap engine — use external manifest if configured
        if self.config.manifest_path and self.config.manifest_path.exists():
            eng = TorusfieldEngine.from_config(str(self.config.manifest_path))
        else:
            eng = TorusfieldEngine.boot_aegis()
        eng.run(self.config.init_warmup_steps)
        self.engine = eng
        self._save_engine()

        # Create empty index
        self.index.data = dict(AnnotationStore.EMPTY_INDEX)
        self.index.save()

        # Create empty PP history
        self.pp_history.data = {"snapshots": []}
        report = eng.ok_report()
        self.pp_history.add_snapshot(
            eng.state.t, dict(eng.pp.vector),
            eng.pp.health_score(), len(eng.store.artifacts))
        self.pp_history.save()

        # Create knowledge files with headers
        _append_md(self.config.methodology_path,
                   "# Methodology\n\nAccumulated expertise from post-task reflections.\n")
        _append_md(self.config.experiments_path,
                   "# Experiment Log\n\nConfiguration experiments with hypothesis, measurement, verdict.\n")
        _append_md(self.config.dead_ends_path,
                   "# Dead Ends\n\nConfirmed closed avenues with conditions for reopening.\n")

        return {
            "status": "initialized",
            "workspace": str(ws),
            "graph": {"nodes": len(eng.graph.nodes), "edges": len(eng.graph.edges)},
            "warmup_steps": self.config.init_warmup_steps,
            "artifacts_after_warmup": len(eng.store.artifacts),
            "pp_health": round(eng.pp.health_score(), 4),
            "pp": {k: round(v, 4) for k, v in eng.pp.vector.items()},
        }

    def op_consult(self, context: str,
                   capabilities: Optional[List[str]] = None) -> Dict:
        """
        Before acting. Returns routing advisory + methodology hits.

        The agent factors this into its own reasoning alongside user intent.
        The advisory suggests — it does not command.
        """
        eng = self._load_engine()
        self.index.load()

        # Resolve capabilities from context — pass recent methodology so
        # learned vocabulary enriches resolution dynamically.
        recent_meth = self.index.recent_methodology(20)
        resolved_caps = resolve_capabilities_from_context(
            context, capabilities, methodology_entries=recent_meth)

        # v1.2: Use separated context_energy field for context-sensitive routing.
        # This injects into a clean field, not the schedule-contaminated energy field.
        # Validated by E3/Option B experiments: produces 3/5 distinct top-1 picks
        # vs 1/5 with the original shared-field approach.
        context_energy_map = eng.prepare_context_energy(
            resolved_caps,
            magnitude=self.config.consult_energy_magnitude,
            diffusion_steps=min(self.config.theta_warmup_steps, 5),
        )

        # Also run standard warm-up on main energy field (preserves v1.1 behavior
        # for edge learning and executor coherence)
        for _ in range(self.config.theta_warmup_steps):
            eng.state.diffuse("energy", eng.theta_mix, eng.phi_mix)
            eng.state.decay(eng.leak_decay)
            eng._run_injections()

        # Score outgoing edges from cursor WITH context energy
        cursor = eng.cursor_node
        paths = self._score_edges(eng, cursor, context_energy=context_energy_map)
        top_paths = paths[:self.config.advisory_top_n]

        # Active wormholes
        wormholes = self._enrich_artifacts(eng, "wormhole", limit=5)

        # Active attractors
        attractors = self._enrich_artifacts(eng, "attractor", limit=3)

        # Search methodology index
        context_words = list(set(re.findall(r"[a-z_]{3,}", context.lower())))
        meth_hits = self.index.search_methodology(
            context_words, resolved_caps, self.config.max_methodology_hits)

        # Search dead ends
        dead_hits = self.index.search_dead_ends(
            context_words, resolved_caps, self.config.max_dead_end_hits)

        # Open questions from last working state
        open_q = []
        ws_dir = self.config.working_state_dir
        if ws_dir.exists():
            oq_files = sorted(ws_dir.glob("*open-questions*"), reverse=True)
            if oq_files:
                content = _read_md(oq_files[0])
                if content.strip():
                    open_q = [content.strip()[:500]]  # truncate for advisory

        # Unannotated artifacts
        all_art_ids = list(eng.store.artifacts.keys())
        unannotated = self.index.get_unannotated(all_art_ids)

        # PP health
        pp_health = eng.pp.health_score()

        # Save (warm-up changed energy landscape)
        self._save_engine()

        return {
            "cursor": cursor,
            "resolved_capabilities": resolved_caps,
            "suggested_paths": top_paths,
            "wormholes": wormholes,
            "attractors": attractors,
            "methodology_hits": [
                {"summary": e.get("summary", ""), "date": e.get("date", ""),
                 "capabilities": e.get("capabilities", [])}
                for e in meth_hits
            ],
            "dead_ends": [
                {"topic": e.get("topic", ""), "why_closed": e.get("why_closed", ""),
                 "date": e.get("date", ""),
                 "reopen_conditions": e.get("reopen_conditions", "")}
                for e in dead_hits
            ],
            "pp_health": round(pp_health, 4),
            "unannotated_count": len(unannotated),
            "open_questions": open_q,
            # v1.2: Show which capabilities received context energy boost
            "context_boosted": {
                cap: round(energy, 4)
                for cap, energy in sorted(
                    context_energy_map.items(), key=lambda x: -x[1])
                if energy > 0.01
            },
        }

    def op_report(self, action: str, outcome: str,
                  success: str = "neutral",
                  significance: str = "routine",
                  context: str = "") -> Dict:
        """
        After acting. Reinforces the torus, checks for artifacts,
        triggers reflections if warranted.
        """
        eng = self._load_engine()
        self.index.load()

        cursor_before = eng.cursor_node
        result: Dict[str, Any] = {
            "cursor_before": cursor_before,
            "action": action,
        }

        # Validate action is a known capability
        if action not in _CAP_NAMES:
            result["warning"] = f"Unknown capability '{action}'. Logged but not reinforced."
            result["cursor"] = cursor_before
            # Still log to action log
            self.action_log.append({
                "t": eng.state.t, "timestamp": _now_iso(),
                "action": action, "outcome": outcome,
                "success": success, "significance": significance,
                "reward": 0, "edge": None,
                "cursor_before": cursor_before, "cursor_after": cursor_before,
                "routed": False,
            })
            self._save_engine()
            self.index.save()
            return result

        # Find direct edge from cursor to action
        eidx = self._find_edge(eng, cursor_before, action)
        routed = eidx is not None

        if eidx is not None:
            edge = eng.graph.edges[eidx]

            # Compute reward
            reward, cost_w = self.reward_model.compute(success, edge.u)
            cost = edge.c * cost_w
            net = reward - cost

            # Manual reinforcement (replicates step_theta logic)
            old_w = edge.w
            edge.w = _clamp(edge.w + eng.reinforce_lr * (net - 0.5), -2.5, 2.5)

            # Record trace step
            step = TraceStep(
                t=eng.state.t, src=edge.src, dst=edge.dst,
                edge_index=eidx, reward=reward, cost=cost, novelty=edge.n)
            eng.trace.append_theta(step)

            result["reinforcement"] = {
                "edge": f"{edge.src} → {edge.dst}",
                "edge_index": eidx,
                "old_w": round(old_w, 4),
                "new_w": round(edge.w, 4),
                "reward": round(reward, 4),
                "cost": round(cost, 4),
                "net": round(net, 4),
            }
        else:
            reward = self.reward_model.base_rewards.get(success, 0.3)
            # Log as policy override (agent jumped to non-adjacent node)
            self.index.add_policy_override({
                "t": eng.state.t,
                "timestamp": _now_iso(),
                "from": cursor_before,
                "to": action,
                "reason": "no_direct_edge",
            })
            result["reinforcement"] = None
            result["policy_override"] = {
                "from": cursor_before, "to": action,
                "reason": "No direct edge. Jump logged as policy override.",
            }

        # Inject energy at action's torus position
        th, ph = self._cap_torus_pos(eng, action)
        eng.inject_event(
            f"action_{action}",
            self.config.report_energy_magnitude,
            th, ph, "energy")

        # Energy diffusion and decay
        eng.state.diffuse("energy", eng.theta_mix, eng.phi_mix)
        eng.state.decay(eng.leak_decay)
        eng._run_injections()

        # Move cursor
        eng.cursor_node = action
        eng.cursor_theta = th
        eng.cursor_phi = ph
        eng.state.t += 1

        result["cursor"] = action
        result["routed"] = routed

        # Check φ-cycle
        self.index.report_counter += 1
        phi_fired = False
        new_artifacts = []

        if self.index.report_counter >= self.config.phi_report_interval:
            self.index.report_counter = 0
            phi_record = eng.step_phi()
            phi_fired = True
            self.index.last_phi_t = eng.state.t

            # Check for newly promoted artifacts
            all_art_ids = list(eng.store.artifacts.keys())
            new_unannotated = self.index.get_unannotated(all_art_ids)
            for aid in new_unannotated:
                art = eng.store.artifacts[aid]
                entry = {"id": aid, "kind": art.kind, "score": round(art.score, 4)}
                if art.kind in ("wormhole", "resonance") and "edge_index" in art.payload:
                    eidx2 = art.payload["edge_index"]
                    if eidx2 < len(eng.graph.edges):
                        e = eng.graph.edges[eidx2]
                        entry["from"] = e.src
                        entry["to"] = e.dst
                elif art.kind == "attractor" and "node" in art.payload:
                    entry["node"] = art.payload["node"]
                new_artifacts.append(entry)

        result["phi_fired"] = phi_fired
        result["new_artifacts"] = new_artifacts

        # Auto-correlate new wormhole promotions against methodology entries.
        # When a wormhole is promoted, find methodology entries whose capability
        # set overlaps with the wormhole's src/dst nodes. Surface these as
        # suggested annotations so the quantitative signal gets qualitative
        # explanation without requiring manual lookup.
        if new_artifacts:
            methodology_correlations = []
            for art in new_artifacts:
                if art.get("kind") != "wormhole":
                    continue
                worm_caps = {art.get("from"), art.get("to")} - {None}
                correlated = self.index.search_methodology(
                    [], list(worm_caps), max_results=2)
                if correlated:
                    methodology_correlations.append({
                        "artifact_id": art["id"],
                        "wormhole": f"{art.get('from')} → {art.get('to')}",
                        "suggested_methodology": [
                            {"summary": e.get("summary", "")[:120],
                             "date": e.get("date", "")}
                            for e in correlated
                        ],
                        "action": (
                            f"Consider annotating artifact {art['id'][:8]}... "
                            "with a pattern from the methodology matches above, "
                            "or write a new annotation if none fit."
                        ),
                    })
            if methodology_correlations:
                result["methodology_correlations"] = methodology_correlations

        # Reflection trigger
        reflection_needed = significance in ("notable", "critical")
        result["reflection_needed"] = reflection_needed
        if reflection_needed:
            result["reflection_prompt"] = {
                "type": "post_task",
                "significance": significance,
                "context": {
                    "action": action,
                    "outcome": outcome,
                    "success": success,
                },
                "fields": {
                    "initial_signal": "What first drew your attention to this approach?",
                    "hypothesis": "Before confirming, what did you think would happen?",
                    "confirmation_path": "What steps did you take to verify? Include dead ends.",
                    "near_miss": "What almost made you miss this or choose wrong?",
                    "generalized_pattern": "Abstract to a reusable heuristic: When you see [X], do [Y] because [Z].",
                    "negative_knowledge": "What did you rule out? What avenues are dead ends?",
                },
            }

        # Unannotated artifacts (always surface if any exist)
        all_aids = list(eng.store.artifacts.keys())
        unannotated = self.index.get_unannotated(all_aids)
        if unannotated and len(new_artifacts) == 0:
            result["unannotated_artifacts"] = [
                {
                    "id": aid,
                    "kind": eng.store.artifacts[aid].kind,
                    "score": round(eng.store.artifacts[aid].score, 4),
                }
                for aid in unannotated[:3]
            ]

        # Action log
        self.action_log.append({
            "t": eng.state.t,
            "timestamp": _now_iso(),
            "action": action,
            "outcome": outcome,
            "success": success,
            "significance": significance,
            "reward": round(reward, 4),
            "edge": f"{cursor_before} → {action}" if routed else None,
            "cursor_before": cursor_before,
            "cursor_after": action,
            "routed": routed,
        })

        self._save_engine()
        self.index.save()
        return result

    def op_reflect(self, reflection_type: str, text: str,
                   artifact_id: Optional[str] = None,
                   capabilities: Optional[List[str]] = None,
                   keywords: Optional[List[str]] = None,
                   failure_condition: Optional[str] = None,
                   **extra) -> Dict:
        """
        Store a reflection. Types: post_task, annotation, experiment, dead_end, synthesis.
        """
        self.index.load()
        today = _today_str()
        result: Dict[str, Any] = {"type": reflection_type, "stored": True}

        if reflection_type == "post_task":
            # Append to methodology.md
            _append_md(self.config.methodology_path, text)
            # Index it
            entry_id = self.index.add_methodology_entry({
                "date": today,
                "capabilities": capabilities or [],
                "keywords": keywords or [],
                "summary": text[:200].replace("\n", " "),
                "source_type": "post_task",
            })
            result["entry_id"] = entry_id
            result["file"] = str(self.config.methodology_path)

        elif reflection_type == "annotation":
            if not artifact_id:
                return {"error": "artifact_id required for annotation"}
            self.index.set_annotation(artifact_id, {
                "text": text,
                "failure_condition": failure_condition or "",
                "generalized_pattern": extra.get("generalized_pattern", ""),
                "annotated_at": _now_iso(),
            })
            result["artifact_id"] = artifact_id

        elif reflection_type == "experiment":
            _append_md(self.config.experiments_path, text)
            result["file"] = str(self.config.experiments_path)

        elif reflection_type == "dead_end":
            _append_md(self.config.dead_ends_path, text)
            entry_id = self.index.add_dead_end({
                "date": today,
                "capabilities": capabilities or [],
                "keywords": keywords or [],
                "topic": extra.get("topic", ""),
                "why_closed": extra.get("why_closed", ""),
                "reopen_conditions": extra.get("reopen_conditions", ""),
            })
            result["entry_id"] = entry_id
            result["file"] = str(self.config.dead_ends_path)

        elif reflection_type == "synthesis":
            synth_path = self.config.synthesis_dir / f"{today}.md"
            _append_md(synth_path, text)
            self.index.last_synthesis_date = today
            result["file"] = str(synth_path)

        else:
            return {"error": f"Unknown reflection type: {reflection_type}"}

        self.index.save()
        return result

    def op_flush(self) -> Dict:
        """
        Pre-compaction save. Exports all state + returns EJ externalization prompts.
        """
        eng = self._load_engine()
        self.index.load()
        self.pp_history.load()

        # Save engine state
        self._save_engine()

        # PP snapshot
        report = eng.ok_report()
        self.pp_history.add_snapshot(
            eng.state.t, dict(eng.pp.vector),
            eng.pp.health_score(), len(eng.store.artifacts))
        self.pp_history.save()

        # Save index
        self.index.save()

        return {
            "torusfield_saved": True,
            "state_digest": report["digest_sha256"][:16],
            "pp_health": round(eng.pp.health_score(), 4),
            "pp_snapshot_saved": True,
            "artifact_count": len(eng.store.artifacts),
            "externalization_prompts": {
                "instructions": (
                    "Write each of the following to a separate file in: "
                    f"{self.config.working_state_dir}/"
                ),
                "hypotheses": {
                    "file": f"{_today_str()}-hypotheses.md",
                    "prompt": "What are you currently thinking about? What's unresolved? "
                              "What mental models are you holding that haven't been tested?",
                },
                "reasoning": {
                    "file": f"{_today_str()}-reasoning.md",
                    "prompt": "What chains of logic are active? Not conclusions — the chains. "
                              "What inferences are you making and what are they based on?",
                },
                "open_questions": {
                    "file": f"{_today_str()}-open-questions.md",
                    "prompt": "What would you investigate next if the session continued? "
                              "What information are you missing?",
                },
                "confidence": {
                    "file": f"{_today_str()}-confidence.md",
                    "prompt": "What are you certain about vs. uncertain about? "
                              "Where would you bet and where would you hedge?",
                },
                "dependencies": {
                    "file": f"{_today_str()}-dependencies.md",
                    "prompt": "What context are you relying on that wouldn't survive a summary? "
                              "What nuances would compaction flatten?",
                },
            },
            "write_to": str(self.config.working_state_dir),
        }

    def op_resume(self) -> Dict:
        """
        Session start. Loads all state, returns session briefing.
        """
        if not self.config.torusfield_state_path.exists():
            return {
                "status": "not_initialized",
                "message": "No UCS state found. Run 'init' first.",
                "command": f"python3 bridge.py init --workspace {self.config.workspace_root}",
            }

        eng = self._load_engine()
        self.index.load()
        self.pp_history.load()

        report = eng.ok_report()

        # Top artifacts with annotations
        wormholes = self._enrich_artifacts(eng, "wormhole", limit=5)
        attractors = self._enrich_artifacts(eng, "attractor", limit=3)
        resonances = self._enrich_artifacts(eng, "resonance", limit=3)

        # Unannotated
        all_ids = list(eng.store.artifacts.keys())
        unannotated = self.index.get_unannotated(all_ids)

        # Open questions from working state
        open_q = []
        ws_dir = self.config.working_state_dir
        if ws_dir.exists():
            for p in sorted(ws_dir.glob("*open-questions*"), reverse=True):
                content = _read_md(p).strip()
                if content:
                    open_q.append(content[:1000])
                    break

        # Recent methodology
        recent_meth = self.index.recent_methodology(5)

        # PP trend
        trend = self.pp_history.trend()

        # Last synthesis
        last_synth = None
        if self.config.synthesis_dir.exists():
            synth_files = sorted(self.config.synthesis_dir.glob("*.md"), reverse=True)
            if synth_files:
                content = _read_md(synth_files[0]).strip()
                if content:
                    last_synth = content[:1000]

        # Recommendations
        recommendations = []
        if unannotated:
            recommendations.append(
                f"Annotate {len(unannotated)} unannotated artifact(s) — "
                "their quantitative signal lacks qualitative explanation.")
        if self.index.last_synthesis_date:
            days_since = (datetime.now(timezone.utc) -
                         datetime.fromisoformat(self.index.last_synthesis_date + "T00:00:00+00:00")).days
            if days_since > 7:
                recommendations.append(
                    f"Synthesis overdue — last was {days_since} days ago.")
        else:
            recommendations.append("No synthesis has been performed yet. Consider running one.")
        if trend == "declining":
            recommendations.append(
                "PP health is declining — review recent actions for efficiency.")

        return {
            "status": "resumed",
            "session_briefing": {
                "t": eng.state.t,
                "cursor": eng.cursor_node,
                "pp_health": round(eng.pp.health_score(), 4),
                "pp": {k: round(v, 4) for k, v in eng.pp.vector.items()},
                "pp_trend": trend,
                "graph": {"nodes": len(eng.graph.nodes), "edges": len(eng.graph.edges)},
                "artifact_count": len(eng.store.artifacts),
                "artifact_breakdown": {
                    "wormholes": len(eng.store.get_by_kind("wormhole")),
                    "attractors": len(eng.store.get_by_kind("attractor")),
                    "resonances": len(eng.store.get_by_kind("resonance")),
                },
                "top_wormholes": wormholes,
                "top_attractors": attractors,
                "top_resonances": resonances,
                "unannotated_count": len(unannotated),
                "open_questions": open_q,
                "recent_methodology": [
                    {"summary": e.get("summary", ""), "date": e.get("date", "")}
                    for e in recent_meth
                ],
                "last_synthesis": last_synth[:300] if last_synth else None,
                "recommendations": recommendations,
            },
        }

    def op_synthesize(self) -> Dict:
        """
        Gathers all raw material for a synthesis report.
        The agent (LLM) does the actual synthesis reasoning.
        """
        eng = self._load_engine()
        self.index.load()
        self.pp_history.load()

        report = eng.ok_report()

        # Determine since-date (last synthesis or 7 days ago)
        since = self.index.last_synthesis_date
        if not since:
            since = (datetime.now(timezone.utc)
                     .replace(hour=0, minute=0, second=0)
                     .isoformat())

        # Action log since last synthesis
        actions = self.action_log.read_since(since)

        # Action frequency (resource allocation)
        action_freq: Dict[str, int] = {}
        success_rate: Dict[str, Dict[str, int]] = {}
        for a in actions:
            act = a.get("action", "unknown")
            action_freq[act] = action_freq.get(act, 0) + 1
            sr = success_rate.setdefault(act, {"success": 0, "partial": 0, "failure": 0, "neutral": 0})
            sr[a.get("success", "neutral")] = sr.get(a.get("success", "neutral"), 0) + 1

        # Sort by frequency
        resource_allocation = [
            {
                "action": act,
                "count": count,
                "pct": round(count / max(len(actions), 1) * 100, 1),
                "success_rate": success_rate.get(act, {}),
            }
            for act, count in sorted(action_freq.items(), key=lambda x: -x[1])
        ]

        # Methodology entries since last synthesis
        recent_meth = self.index.recent_methodology(20)

        # PP trend data
        pp_snapshots = self.pp_history.data.get("snapshots", [])[-20:]

        # Artifacts vs methodology alignment
        wormholes = self._enrich_artifacts(eng, "wormhole", limit=10)
        unannotated_ids = self.index.get_unannotated(list(eng.store.artifacts.keys()))

        # Policy overrides (agent disagreed with graph)
        overrides = self.index.data.get("policy_overrides", [])

        # Misalignment detection
        misalignments = []
        # High-traffic paths with no methodology
        for item in resource_allocation[:5]:
            caps = [item["action"]]
            hits = self.index.search_methodology([], caps, max_results=1)
            if not hits:
                misalignments.append({
                    "type": "high_traffic_no_methodology",
                    "action": item["action"],
                    "count": item["count"],
                    "message": f"'{item['action']}' used {item['count']} times "
                               "but has no methodology entry.",
                })

        return {
            "synthesis_data": {
                "period": {"since": since, "until": _now_iso()},
                "total_actions": len(actions),
                "resource_allocation": resource_allocation,
                "pp_health": round(eng.pp.health_score(), 4),
                "pp_trend": self.pp_history.trend(),
                "pp_snapshots": pp_snapshots[-5:],
                "artifact_summary": {
                    "total": len(eng.store.artifacts),
                    "wormholes": len(eng.store.get_by_kind("wormhole")),
                    "attractors": len(eng.store.get_by_kind("attractor")),
                    "resonances": len(eng.store.get_by_kind("resonance")),
                    "unannotated": len(unannotated_ids),
                },
                "top_wormholes": wormholes[:5],
                "recent_methodology": [
                    {"summary": e.get("summary", ""), "date": e.get("date", "")}
                    for e in recent_meth
                ],
                "policy_overrides": len(overrides),
                "override_summary": _summarize_overrides(overrides) if overrides else [],
                "misalignments": misalignments,
                "stale_artifacts": [
                    {"id": a.id, "kind": a.kind,
                     "score": round(a.score, 4),
                     "last_detected_t": a.meta.get("last_detected_t", a.created_t),
                     "age_steps": eng.state.t - a.meta.get("last_detected_t", a.created_t)}
                    for a in eng.store.get_stale(eng.state.t)
                ],
            },
            "synthesis_prompts": {
                "patterns_emerging": "What themes recur across the action log? What keeps happening?",
                "judgment_shifts": "Has your assessment of anything changed? What do you believe now that you didn't before?",
                "resource_review": "Is time being spent where value is generated? See resource_allocation.",
                "whats_working": "Which paths, tools, or approaches are producing reliable results?",
                "whats_not": "Which are underperforming or causing friction?",
                "recommendations": "Based on all of this: what should change?",
                "open_questions": "What don't you know that you need to know?",
                "methodology_updates": "Does the methodology need updating based on this period's experience?",
            },
        }

    def op_status(self) -> Dict:
        """Quick health check."""
        if not self.config.torusfield_state_path.exists():
            return {"status": "not_initialized"}

        eng = self._load_engine()
        self.index.load()

        all_ids = list(eng.store.artifacts.keys())
        unannotated = self.index.get_unannotated(all_ids)

        stale = eng.store.get_stale(eng.state.t)

        return {
            "status": "ok",
            "t": eng.state.t,
            "cursor": eng.cursor_node,
            "pp_health": round(eng.pp.health_score(), 4),
            "pp": {k: round(v, 4) for k, v in eng.pp.vector.items()},
            "artifact_count": len(eng.store.artifacts),
            "unannotated_count": len(unannotated),
            "stale_artifact_count": len(stale),
            "report_counter": self.index.report_counter,
            "last_synthesis": self.index.last_synthesis_date,
        }


# ============================================================================
# HELPERS
# ============================================================================

def _summarize_overrides(overrides: List[Dict]) -> List[Dict]:
    """Summarize policy overrides by (from, to) pair frequency."""
    freq: Dict[str, int] = {}
    for o in overrides:
        key = f"{o.get('from', '?')} → {o.get('to', '?')}"
        freq[key] = freq.get(key, 0) + 1
    return [
        {"jump": k, "count": v}
        for k, v in sorted(freq.items(), key=lambda x: -x[1])[:10]
    ]


# ============================================================================
# CLI
# ============================================================================

def _read_stdin_json() -> Dict:
    """Read JSON from stdin. Returns empty dict if nothing available."""
    if sys.stdin.isatty():
        return {}
    try:
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(
        prog="bridge.py",
        description="UCS Bridge — Unified Cognitive Substrate for AI Agents",
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Workspace root path (default: ~/.ucs)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialize workspace and bootstrap engine")
    p_init.add_argument("--force", action="store_true", help="Reinitialize even if exists")

    # consult
    p_consult = sub.add_parser("consult", help="Get routing advisory before acting")
    p_consult.add_argument("--context", type=str, required=True,
                           help="Description of current task/situation")
    p_consult.add_argument("--capabilities", type=str, default=None,
                           help="Comma-separated capability names")

    # report
    sub.add_parser("report", help="Report action outcome (JSON on stdin)")

    # reflect
    sub.add_parser("reflect", help="Store a reflection (JSON on stdin)")

    # flush
    sub.add_parser("flush", help="Pre-compaction save + externalization prompts")

    # resume
    sub.add_parser("resume", help="Session start briefing")

    # synthesize
    sub.add_parser("synthesize", help="Gather synthesis raw material")

    # status
    sub.add_parser("status", help="Quick health check")

    args = parser.parse_args()

    # Config
    config = UCSConfig()
    if args.workspace:
        config = UCSConfig(workspace_root=Path(args.workspace))

    # External manifest support: if a manifest.json exists in the workspace,
    # use it automatically without requiring an explicit flag.
    auto_manifest = config.workspace_root / "manifest.json"
    if auto_manifest.exists() and config.manifest_path is None:
        config.manifest_path = auto_manifest

    bridge = UCSBridge(config)

    # Dispatch
    if args.command == "init":
        result = bridge.op_init(force=args.force)

    elif args.command == "consult":
        caps = args.capabilities.split(",") if args.capabilities else None
        result = bridge.op_consult(args.context, caps)

    elif args.command == "report":
        data = _read_stdin_json()
        if not data:
            result = {"error": "No JSON input on stdin. Expected: {action, outcome, success, significance}"}
        else:
            result = bridge.op_report(
                action=data.get("action", ""),
                outcome=data.get("outcome", ""),
                success=data.get("success", "neutral"),
                significance=data.get("significance", "routine"),
                context=data.get("context", ""),
            )

    elif args.command == "reflect":
        data = _read_stdin_json()
        if not data:
            result = {"error": "No JSON input on stdin. Expected: {type, text, ...}"}
        else:
            result = bridge.op_reflect(
                reflection_type=data.get("type", "post_task"),
                text=data.get("text", ""),
                artifact_id=data.get("artifact_id"),
                capabilities=data.get("capabilities"),
                keywords=data.get("keywords"),
                failure_condition=data.get("failure_condition"),
                **{k: v for k, v in data.items()
                   if k not in ("type", "text", "artifact_id", "capabilities",
                                "keywords", "failure_condition")},
            )

    elif args.command == "flush":
        result = bridge.op_flush()

    elif args.command == "resume":
        result = bridge.op_resume()

    elif args.command == "synthesize":
        result = bridge.op_synthesize()

    elif args.command == "status":
        result = bridge.op_status()

    else:
        result = {"error": f"Unknown command: {args.command}"}

    # Output
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
