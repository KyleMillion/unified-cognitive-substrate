#!/usr/bin/env python3
"""
BASELINE ENGINE — Plain directed graph with edge RL and artifact detection.
NO torus surface, NO energy fields, NO diffusion, NO θ/φ cycles.

This exists solely to answer: does the toroidal topology add measurable value
over the simpler mechanism of edge reinforcement + trace-log pattern mining?

Same API surface as TorusfieldEngine where it matters for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from torusfield_kernel import (
    EdgeGraph,
    Edge,
    TraceStep,
    TraceLog,
    Artifact,
    ArtifactStore,
    PolicyKernel,
    Harvester,
    PPEngine,
    AEGIS_CAPABILITY_MANIFEST,
    AEGIS_SEMANTIC_MAP,
    _clamp,
    _sha256,
    _stable_json,
)


@dataclass
class BaselineRouter:
    """Softmax edge selection using ONLY edge properties + policy bias.
    No energy field influence whatsoever."""
    alpha: float = 1.25
    beta: float = 1.10
    gamma: float = 0.35
    temperature: float = 1.0

    def pick(self, rng: random.Random, graph: EdgeGraph,
             policy: PolicyKernel, src: str) -> int:
        candidates = graph.adj.get(src, [])
        if not candidates:
            return -1
        scores = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)
            s = (self.alpha * e.u - self.beta * e.c
                 + self.gamma * e.n + e.w + bias)
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
                       src: str) -> List[Dict]:
        """Score all outgoing edges — pure edge properties, no energy."""
        candidates = graph.adj.get(src, [])
        scored = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)
            score = (self.alpha * e.u - self.beta * e.c
                     + self.gamma * e.n + e.w + bias)
            scored.append({
                "to": e.dst,
                "edge_index": ei,
                "score": round(score, 6),
                "components": {
                    "utility": round(e.u, 4),
                    "cost": round(-e.c, 4),
                    "weight": round(e.w, 4),
                    "novelty": round(e.n, 4),
                    "bias": round(bias, 4),
                },
            })
        scored.sort(key=lambda x: -x["score"])
        return scored


@dataclass
class BaselineExecutor:
    """Fixed reward model — no energy field coherence calculation."""
    reward_scale: float = 1.0
    utility_weight: float = 0.80

    def run(self, edge: Edge) -> Tuple[float, float]:
        reward = self.reward_scale * self.utility_weight * edge.u
        return reward, edge.c


@dataclass
class BaselineEngine:
    """Plain directed graph engine. Same learning, no torus."""
    graph: EdgeGraph
    trace: TraceLog
    store: ArtifactStore
    policy: PolicyKernel
    router: BaselineRouter
    executor: BaselineExecutor
    harvester: Harvester
    pp: PPEngine

    reinforce_lr: float = 0.075
    phi_period: int = 25

    rng: random.Random = field(default_factory=lambda: random.Random(1337))
    cursor_node: str = ""
    t: int = 0

    @staticmethod
    def boot_aegis(seed: int = 1337) -> "BaselineEngine":
        graph = EdgeGraph.from_capabilities(
            AEGIS_CAPABILITY_MANIFEST, AEGIS_SEMANTIC_MAP)
        start_node = AEGIS_CAPABILITY_MANIFEST[0]["name"]
        return BaselineEngine(
            graph=graph,
            trace=TraceLog(maxlen_theta=500, maxlen_phi=100),
            store=ArtifactStore(),
            policy=PolicyKernel(),
            router=BaselineRouter(),
            executor=BaselineExecutor(),
            harvester=Harvester(),
            pp=PPEngine(),
            rng=random.Random(seed),
            cursor_node=start_node,
        )

    def step(self) -> Optional[str]:
        """One step: route, execute, reinforce. No energy dynamics."""
        eidx = self.router.pick(self.rng, self.graph, self.policy,
                                self.cursor_node)
        if eidx < 0:
            return None
        edge = self.graph.edges[eidx]
        reward, cost = self.executor.run(edge)

        step = TraceStep(t=self.t, src=edge.src, dst=edge.dst,
                         edge_index=eidx, reward=reward, cost=cost,
                         novelty=edge.n)
        self.trace.append_theta(step)

        net = reward - cost
        edge.w = _clamp(edge.w + self.reinforce_lr * (net - 0.5), -2.5, 2.5)

        self.cursor_node = edge.dst
        self.t += 1

        if (self.t % self.phi_period) == 0:
            self.step_phi()

        return edge.dst

    def step_phi(self) -> Dict[str, Any]:
        candidates = self.harvester.detect_all(self.trace, self.t, self.graph)
        validated = self.harvester.validate(candidates)
        promoted = 0
        for a in validated:
            if self.store.promote(a):
                promoted += 1
        self.policy.update(validated, self.graph)
        pp_vec = self.pp.evaluate(validated, self.trace)
        return {"t": self.t, "promoted": promoted,
                "total": len(self.store.artifacts), "pp": pp_vec}

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def manual_reinforce(self, src: str, dst: str, reward: float,
                         cost: float) -> Optional[int]:
        """Manual reinforcement for bridge-style report(). Returns edge index."""
        for eidx in self.graph.adj.get(src, []):
            if self.graph.edges[eidx].dst == dst:
                edge = self.graph.edges[eidx]
                net = reward - cost
                old_w = edge.w
                edge.w = _clamp(edge.w + self.reinforce_lr * (net - 0.5),
                                -2.5, 2.5)
                step = TraceStep(t=self.t, src=src, dst=dst,
                                 edge_index=eidx, reward=reward,
                                 cost=cost, novelty=edge.n)
                self.trace.append_theta(step)
                self.cursor_node = dst
                self.t += 1
                if (self.t % self.phi_period) == 0:
                    self.step_phi()
                return eidx
        return None

    def inject_context(self, capabilities: List[str]) -> None:
        """Baseline context injection: NO-OP. Nothing to inject into."""
        pass  # This is the point — baseline has no energy surface

    def get_advisory(self, src: str) -> List[Dict]:
        return self.router.score_outgoing(self.graph, self.policy, src)
