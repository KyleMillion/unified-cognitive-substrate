# Arch Grants Startup Competition — Full Submission Draft (READY TO PASTE)

**Deadline:** March 31, 2026  
Program page: https://archgrants.org/programs/startup-competition/  
Submission portal: https://arch-grants.smapply.org/  

> This document is written to be pasted field-by-field into the Smapply form. If the portal has slightly different labels, reuse the closest matching section.

---

## 0) Quick identity
**Company / project name:** IntuiTek¹ (UCS v1.2)

**Founder:** W. Kyle Million

**Current location:** Poplar Bluff, Missouri

**Willing to HQ in St. Louis for 1 year (required):** Yes.

**Website / landing:** https://kylemillion.github.io/unified-cognitive-substrate/ucs/

**GitHub (canonical repo):** https://github.com/KyleMillion/unified-cognitive-substrate

**Canonical provenance hub:** https://github.com/KyleMillion/unified-cognitive-substrate/tree/main/judgment-preservation

**Paper (Zenodo DOI):** https://doi.org/10.5281/zenodo.18794692

**Primary contact email:** kyle@intuitek.ai

---

## 1) One-liner (short)
UCS v1.2 is a judgment-preservation framework for persistent, tool-using AI agents—so automation improves over time without silently regressing after compaction/resets.

---

## 2) Company / product description (medium)
IntuiTek¹ is building UCS v1.2 (Toroidal Routing Engine + Emergent Judgment Protocol), a practical reliability layer for persistent AI agents operating in real tool environments. Most “persistence” approaches focus on memory retrieval (RAG / vector DBs). UCS focuses on preserving *judgment artifacts*—constraints, heuristics, anti-patterns, and escalation rules—so an agent doesn’t repeat the same failure with higher confidence after resets or compactions. This matters most when the agent can take real actions (automations, workflows, deployments) where reliability and liability are real.

---

## 3) Problem (what pain are we solving?)
Persistent agents are moving from demos to production in automation and operational workflows. The core reliability failure mode isn’t missing information—it’s **losing judgment** across resets/compactions:

- boundaries disappear (what must never happen)
- heuristics vanish (what to do under uncertainty)
- anti-patterns are forgotten (what failed last time)
- escalation rules degrade (when to stop and ask a human)

This produces silent regression: the system appears to function, but repeats expensive mistakes until it burns time, money, or trust. In compliance- and liability-adjacent settings (legal, medical, security), that’s a non-starter.

---

## 4) Solution (how do we solve it?)
UCS v1.2 is built around two components:

1) **Toroidal Routing Engine**
- Routes decisions through a structured capability graph so the agent doesn’t mode-collapse into the same handful of behaviors.

2) **Emergent Judgment Protocol (EJP)**
- Preserves proven patterns and proven failures (anti-patterns) into durable artifacts (docs + workflows + guardrails).
- Focuses on the few preserved decisions that change outcomes, not “remember everything.”

In practice, UCS favors boring, auditable mechanisms: explicit constraints, approval gates for risky actions, and preserved judgment notes that survive compaction.

---

## 5) What we will build next (12-month plan / deliverables)
**Open-source core (public):**
- clearer UCS v1.2 spec + reference templates
- a “judgment regression test harness” (scenarios that detect drift after compaction)
- integration patterns for tool-agent platforms (starting with OpenClaw-style agents)

**Commercialization path (sustainable):**
- paid integration + safety/reliability review services for teams deploying persistent agents
- optional product packaging (playbooks + evaluation tooling + deployment support)

---

## 6) Traction (what proof exists today?)
- Canonical repo + provenance package (hashes, citation, press kit):
  https://github.com/KyleMillion/unified-cognitive-substrate/tree/main/judgment-preservation
- Public paper on Zenodo (DOI): https://doi.org/10.5281/zenodo.18794692
- Launch thread posted to begin outbound visibility and collaborations.

---

## 7) Target customers / users
- teams deploying tool-using agents (automation, ops, dev tooling)
- researchers/builders focused on AI safety and agent reliability
- maintainers of open-source agent frameworks

---

## 8) Competitive landscape / differentiation
Most approaches optimize memory retrieval (RAG). That helps with facts.

UCS’s differentiation is preserving *judgment artifacts* that prevent failure under uncertainty:
- constraints and escalation rules
- anti-patterns (what not to do)
- routing through a capability structure to reduce drift

It is designed to be audit-friendly and usable by small teams.

---

## 9) St. Louis commitment + economic impact
If awarded, IntuiTek¹ will headquarter in St. Louis for at least one year as required.

**Economic impact plan:**
- use Arch Grants capital to convert open-source traction into paid integration work
- collaborate with the St. Louis ecosystem (founders, enterprise partners, research community)
- create a credible path to local contracting and hiring as revenue milestones are hit

---

## 10) Team
**W. Kyle Million** — founder/engineer; UCS v1.2 author.

---

## 11) Use of funds — how $75,000 changes the outcome (budget outline)
This funding is non-dilutive runway to stabilize the founder’s ability to execute and to turn UCS into a sustainable product + open-source project.

**Budget (12 months) — draft allocation:**
- **$45,000** Founder runway / stability (keeps the work moving under real life constraints; reduces churn)
- **$12,000** Contract engineering / documentation support (ship spec + templates + tests)
- **$6,000** Evaluation + tooling costs (compute, test infra, minimal paid services)
- **$6,000** St. Louis relocation/HQ costs (as required; travel, initial setup)
- **$6,000** Outreach / ecosystem participation (events, demos, customer discovery, legal/accounting basics)

**Why this is realistic:** UCS is already public, but reliability work needs sustained focus: test harnesses, docs, integration patterns, and pilots.

---

## 12) Milestones (what success looks like)
**Within 90 days:**
- finalize spec + repo organization
- publish initial judgment regression test harness
- pilot discussions with 3–5 teams building tool agents

**Within 6 months:**
- at least 1–2 pilot implementations
- publish results (what improved / what failed)
- convert early adoption into first paid integration engagements

**Within 12 months:**
- stable open-source cadence
- repeatable services offering
- clear revenue path + early job creation (contractors → hires)

---

## 13) Why Arch Grants / Why now
Arch Grants provides equity-free capital plus an ecosystem in St. Louis that’s strong for converting an open project into a durable Missouri business.

AI agents are at an inflection point: deployment is accelerating, and reliability/liability are becoming the bottleneck. UCS aims directly at that bottleneck.

---

## 14) Links (paste into portal)
- Canonical hub: https://github.com/KyleMillion/unified-cognitive-substrate/tree/main/judgment-preservation
- Repo: https://github.com/KyleMillion/unified-cognitive-substrate
- Landing: https://kylemillion.github.io/unified-cognitive-substrate/ucs/
- Paper DOI: https://doi.org/10.5281/zenodo.18794692

---

## Notes for Kyle (before submit)
- If the portal asks for incorporation, revenue, or customer traction numbers: answer honestly; we can tailor.
- If it asks for a relocation plan: use the St. Louis HQ commitment section above.
- If it asks for a pitch deck: we can generate a short 6–8 slide deck from this draft.

