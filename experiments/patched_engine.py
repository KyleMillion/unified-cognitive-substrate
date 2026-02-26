#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ENERGY-COUPLED ROUTER — The architectural fix identified by E3            ║
║                                                                            ║
║  Original Router.pick() score:                                             ║
║    s = alpha*u - beta*c + gamma*n + w + bias                              ║
║                                                                            ║
║  Patched Router.pick() score:                                              ║
║    s = alpha*u - beta*c + gamma*n + w + bias + delta*energy_at_dst         ║
║                                                                            ║
║  This connects the energy injection/diffusion pipeline to the routing      ║
║  decision, so that consult()'s energy injection at relevant capabilities   ║
║  actually influences the advisory returned to the agent.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
    TorusfieldEngine,
    TorusState,
    EdgeGraph,
    Edge,
    Router,
    PolicyKernel,
    AEGIS_CAPABILITY_MANIFEST,
    AEGIS_SEMANTIC_MAP,
)


def build_node_coordinate_map(capabilities: List[Dict],
                               theta_bins: int = 24,
                               phi_bins: int = 32) -> Dict[str, Tuple[int, int]]:
    """Map each capability name to its (theta, phi) torus coordinates.
    
    Uses the same formula as TorusfieldEngine.from_capabilities():
        th = i % theta_bins
        ph = (i * 3) % phi_bins
    """
    coord_map = {}
    for i, cap in enumerate(capabilities):
        th = i % theta_bins
        ph = (i * 3) % phi_bins
        coord_map[cap["name"]] = (th, ph)
    return coord_map


@dataclass
class EnergyCoupledRouter:
    """Router that reads the torus energy field at destination nodes.
    
    The delta parameter controls how strongly energy influences routing.
    When delta=0, this behaves identically to the original Router.
    """
    alpha: float = 1.25   # utility weight
    beta: float = 1.10    # cost weight
    gamma: float = 0.35   # novelty weight
    delta: float = 0.40   # NEW: energy coupling weight
    temperature: float = 1.0

    def pick(self, rng: random.Random, graph: EdgeGraph,
             policy: PolicyKernel, src: str,
             state: Optional[TorusState] = None,
             node_coords: Optional[Dict[str, Tuple[int, int]]] = None) -> int:
        """Route with energy-at-destination influence.
        
        If state and node_coords are provided, energy at each destination
        node's torus position is included in the score. Otherwise falls
        back to the original formula.
        """
        candidates = graph.adj.get(src, [])
        if not candidates:
            return -1

        scores = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)
            
            # Original terms
            s = (self.alpha * e.u - self.beta * e.c
                 + self.gamma * e.n + e.w + bias)
            
            # NEW: Energy at destination
            if state is not None and node_coords is not None:
                coords = node_coords.get(e.dst)
                if coords is not None:
                    energy_at_dst = state.get("energy", coords[0], coords[1])
                    s += self.delta * energy_at_dst
            
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
                       state: Optional[TorusState] = None,
                       node_coords: Optional[Dict[str, Tuple[int, int]]] = None
                       ) -> List[Dict]:
        """Score all outgoing edges with energy coupling."""
        candidates = graph.adj.get(src, [])
        scored = []
        for ei in candidates:
            e = graph.edges[ei]
            bias = policy.edge_bias.get(ei, 0.0)
            
            s = (self.alpha * e.u - self.beta * e.c
                 + self.gamma * e.n + e.w + bias)
            
            energy_component = 0.0
            if state is not None and node_coords is not None:
                coords = node_coords.get(e.dst)
                if coords is not None:
                    energy_at_dst = state.get("energy", coords[0], coords[1])
                    energy_component = self.delta * energy_at_dst
                    s += energy_component
            
            scored.append({
                "to": e.dst,
                "edge_index": ei,
                "score": round(s, 6),
                "energy_contribution": round(energy_component, 6),
                "components": {
                    "utility": round(self.alpha * e.u, 4),
                    "cost": round(-self.beta * e.c, 4),
                    "weight": round(e.w, 4),
                    "novelty": round(self.gamma * e.n, 4),
                    "bias": round(bias, 4),
                    "energy": round(energy_component, 4),
                },
            })
        scored.sort(key=lambda x: -x["score"])
        return scored


class PatchedTorusfieldEngine(TorusfieldEngine):
    """TorusfieldEngine with energy-coupled routing.
    
    Drop-in replacement — same API, same state format.
    Only difference: Router reads energy at destination nodes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._energy_router = EnergyCoupledRouter()
        self._node_coords = {}
    
    @staticmethod
    def boot_aegis(seed: int = 1337, theta_bins: int = 24,
                   phi_bins: int = 32,
                   delta: float = 0.40) -> "PatchedTorusfieldEngine":
        """Boot patched engine with energy-coupled routing."""
        # Build via parent factory, then upgrade
        base = TorusfieldEngine.boot_aegis(seed=seed,
                                            theta_bins=theta_bins,
                                            phi_bins=phi_bins)
        
        # Build coordinate map
        node_coords = build_node_coordinate_map(
            AEGIS_CAPABILITY_MANIFEST, theta_bins, phi_bins)
        
        # Create patched engine by copying all state
        patched = PatchedTorusfieldEngine(
            state=base.state,
            graph=base.graph,
            trace=base.trace,
            store=base.store,
            policy=base.policy,
            router=base.router,  # Keep original for fallback
            executor=base.executor,
            harvester=base.harvester,
            pp=base.pp,
            rng=base.rng,
            cursor_node=base.cursor_node,
        )
        patched.injection_schedule = base.injection_schedule
        patched.cursor_theta = base.cursor_theta
        patched.cursor_phi = base.cursor_phi
        patched._energy_router = EnergyCoupledRouter(delta=delta)
        patched._node_coords = node_coords
        return patched
    
    def step_theta(self) -> Optional[str]:
        """Override: use energy-coupled router instead of vanilla Router."""
        self._run_injections()
        self.state.diffuse("energy", self.theta_mix, self.phi_mix)

        # THIS IS THE FIX: router now reads energy at destinations
        eidx = self._energy_router.pick(
            self.rng, self.graph, self.policy, self.cursor_node,
            state=self.state, node_coords=self._node_coords)
        
        if eidx < 0:
            return None

        edge = self.graph.edges[eidx]
        reward, cost = self.executor.run(self.state, edge,
                                         self.cursor_theta, self.cursor_phi)

        from torusfield_kernel import TraceStep, _clamp
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

    def get_context_advisory(self, src: str) -> List[Dict]:
        """Get advisory with energy coupling visible."""
        return self._energy_router.score_outgoing(
            self.graph, self.policy, src,
            state=self.state, node_coords=self._node_coords)
