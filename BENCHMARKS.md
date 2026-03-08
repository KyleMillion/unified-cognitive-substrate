# UCS Retrieval Index Benchmarks (v2.1)

Date: 2026-03-08

## Summary
We added an inverted index over routing snapshots to pre-filter judgment-node retrieval.
A debug counter (guarded by env var) prints the reduction ratio: candidates scored / total nodes.

This file records the first measured ratios from a live UCS state file.

## How to reproduce

### Debug counter
Enable debug output:

```bash
export UCS_DEBUG_INDEX=1
```

Control strictness of the structural pre-filter:

```bash
# default
export UCS_INDEX_MIN_SIGNALS=1
# tighter
export UCS_INDEX_MIN_SIGNALS=2
```

### Consult measurement (baseline)

```bash
UCS_DEBUG_INDEX=1 UCS_INDEX_MIN_SIGNALS=1 \
python3 /home/k/.openclaw/workspace/skills/ucs/bridge.py \
  --workspace /home/k/.ucs consult \
  --context "researching retrieval architecture and web fetch patterns" \
  2>&1 | rg "\[UCS index\]"
```

### Seed methodology
We tested two datasets:

1) **Random baseline (100 nodes):** varied cursors/top_paths/energy.
2) **Clustered dataset (99 nodes):** 3 clusters of 33 nodes each:

- **Cluster A**
  - cursor = `browser`
  - top_paths = [`web_fetch`, `analysis`]
  - context_energy = {`web_fetch`, `analysis`}

- **Cluster B**
  - cursor = `exec`
  - top_paths = [`foundry_research`, `nodes`]
  - context_energy = {`foundry_research`}

- **Cluster C**
  - cursor = `draft`
  - top_paths = [`message`, `read`]
  - context_energy = {`message`, `read`}

### Targeted query structure (clustered test)
For each cluster, query routing_state matched the cluster signature.
Example (Cluster A):

```json
{
  "cursor": "browser",
  "top_paths": [{"to":"web_fetch","score":1.2},{"to":"analysis","score":1.0}],
  "context_energy": {"web_fetch": 1.2, "analysis": 1.0},
  "active_artifacts": [],
  "t": 2000
}
```

## Results

### Random data baseline
- **min_signals=1:** **79/100** candidates scored (**79%**)

### Clustered data, targeted query (min_signals=2)
Each targeted query isolated its corresponding cluster:

- **Query A (matches Cluster A):** **33/99** (**33%**) candidates scored
- **Query B (matches Cluster B):** **33/99** (**33%**) candidates scored
- **Query C (matches Cluster C):** **33/99** (**33%**) candidates scored

## Notes
- `min_signals=1` is intentionally high-recall and will admit many candidates in mixed/random datasets.
- `min_signals=2` (or stronger structural requirements) is where the index begins to sharply isolate clusters.
