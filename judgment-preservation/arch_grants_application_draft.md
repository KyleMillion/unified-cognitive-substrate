# Arch Grants Startup Competition — Application Draft (2026)

**Deadline:** March 31, 2026 (per Arch Grants Startup Competition page)
Source: https://archgrants.org/programs/startup-competition/

> Note: Arch Grants uses an application portal. This draft is structured to map cleanly into typical required fields.

---

## Company
- Company name: IntuiTek¹
- Founder: W. Kyle Million
- Location: Poplar Bluff, Missouri (willing to HQ in St. Louis for at least 1 year if awarded)
- Website / canonical landing: https://kylemillion.github.io/unified-cognitive-substrate/ucs/
- GitHub repo: https://github.com/KyleMillion/unified-cognitive-substrate
- Paper / DOI: https://doi.org/10.5281/zenodo.18794692

## One-liner
UCS v1.2 is a judgment-preservation framework for persistent, tool-using AI agents—so automation improves over time without silently regressing after compaction/resets.

## Problem
Persistent AI agents increasingly operate in real workflows (automation, customer ops, compliance-heavy tasks, devops). The core reliability failure isn’t missing information—it’s **losing judgment** after resets/compactions: boundaries, heuristics, and “what failed last time.” That silent regression leads to repeated mistakes, higher-confidence errors, and unacceptable risk in production tool execution.

## Solution
UCS v1.2 (Toroidal Routing Engine + Emergent Judgment Protocol) preserves the few artifacts that change outcomes:
- explicit constraints / approval gates for risky actions
- proven patterns and proven failures (anti-patterns)
- capability-routing to prevent behavioral mode-collapse

The result is continuity that is testable, auditable, and safer than "just give the agent more context" approaches.

## Product / What we build
- Open-source canonical implementation patterns, templates, and evaluation harnesses for judgment-regression
- Reference integrations for tool-agent platforms (starting with OpenClaw-style agents)
- Optional commercial layer: onboarding, safety reviews, and deployment support for teams adopting persistent agents in production

## Why now
Tool-using agents are moving from demos to production. Reliability and liability risk are becoming the bottleneck. The market is ready for concrete mechanisms that reduce drift and regression.

## Target customers / users
1) Teams deploying tool-using agents in production (automation, ops, dev tooling)
2) AI safety / reliability researchers and builders
3) Open-source agent frameworks and maintainers

## Market & differentiation (brief)
Most solutions focus on memory (RAG, vector DBs). UCS focuses on **judgment artifacts** and on preserving operational safety constraints across compaction. It’s designed to be auditable and usable by small teams.

## Traction (fill/verify)
- Public canonical repo + provenance package (hashes, citation):
  https://github.com/KyleMillion/unified-cognitive-substrate/tree/main/judgment-preservation
- Paper published on Zenodo (DOI): https://doi.org/10.5281/zenodo.18794692
- Early visibility: Launch thread live on X (link available)

TODO (Kyle): add any of the following if true:
- any users/stars/forks
- any pilots
- any inbound collab requests

## Business model (credible near-term)
Open-core + services:
- Free/open implementation patterns + templates
- Paid services: onboarding, reliability review, integration support, custom evaluation harnesses
- Longer-term: packaged reliability layer / managed deployments for orgs operating persistent agents

## St. Louis commitment / jobs plan
If awarded, IntuiTek¹ will HQ in St. Louis for at least one year as required. Initial plan:
- establish St. Louis base
- recruit early collaborators/contractors locally where possible
- convert open-source adoption into paid integration work

Job creation path (realistic):
- Year 1: founder + contractors (documentation, integration, UX)
- Year 2: 1–3 hires contingent on revenue/grant milestones

## Team
- W. Kyle Million — founder/engineer; UCS v1.2 author

(Optional additions if applicable)
- Advisors / collaborators: TBD

## Use of funds
Non-dilutive capital will be used to:
- harden the reference implementation patterns and docs
- build an evaluation harness for judgment-regression tests
- run pilots with early adopters
- cover relocation/HQ costs and founder runway while building sustainable revenue

## Why Arch Grants / Why St. Louis
Arch Grants provides equity-free capital plus a strong St. Louis network—ideal for converting an open-source project into a Missouri-based company through partnerships, pilots, and early customer development.

## Links
- Canonical hub: https://github.com/KyleMillion/unified-cognitive-substrate/tree/main/judgment-preservation
- Repo: https://github.com/KyleMillion/unified-cognitive-substrate
- Landing: https://kylemillion.github.io/unified-cognitive-substrate/ucs/
- Paper DOI: https://doi.org/10.5281/zenodo.18794692
