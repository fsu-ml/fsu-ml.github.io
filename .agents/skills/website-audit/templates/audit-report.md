# Website audit — {{SITE_NAME}}

**URL audited** {{URL}}
**Date** {{YYYY-MM-DD}}
**Auditor** {{NAME}}
**Profile** `audit-profile.yaml`, last verified {{DATE}}
**Engagement mode** {{pre-launch-gate | live-audit | mid-build}}

---

## 1. Summary

> Keep this to one page. Write it last. Put numbers in it.

| Severity | Count |
|---|---|
| Blocker | {{n}} |
| Major | {{n}} |
| Minor | {{n}} |
| Advisory | {{n}} |

**Gate status** — {{PASS / FAIL}} *(pre-launch mode only)*

**The three things that matter:**

1. {{One sentence. Specific. With the measured value.}}
2. {{…}}
3. {{…}}

---

## 2. Scope and method

**Profile summary**

| | |
|---|---|
| Category / primary job | {{}} |
| Compliance regime · target | {{}} · {{}} |
| Rendering mode (verified) | {{}} |
| Device floor | {{}} |
| Motion budget | {{}} |
| SEO priority | {{}} |
| Field data available | {{yes — CrUX/RUM / no — lab only}} |

**Pages sampled** *(and why — see `verification-protocol.md` §4)*

| URL | Why included |
|---|---|
| | |

**Viewports tested** {{list}}
**Throttling** {{CPU ×N, network profile}}

**Tools and versions**

| Tool | Version | Run |
|---|---|---|
| `audit_a11y.py` (axe-core) | | |
| `audit_responsive.py` | | |
| `audit_performance.py` | | |
| `audit_motion.py` | | |
| `audit_seo.py` | | |
| `check_headers.py` | | |
| Manual: keyboard / screen reader | | |

### What was NOT checked

> Required. An audit that overclaims is worse than one that is narrow and honest.

- Automated accessibility testing covers roughly 30–40% of WCAG issues. {{n}} success criteria could not be tested automatically and were {{covered manually / not covered}}.
- {{Lab-only performance data — no field data available for this origin.}}
- {{INP not measurable in the lab; TBT reported as proxy.}}
- {{Screen reader testing limited to {{pairs}}.}}
- {{Out of scope per profile: …}}

---

## 3. Blockers

> Someone cannot complete a task, cannot access content, or the client has legal exposure. Ships nothing until fixed.

### B-01 · {{Title}}

**Severity** blocker · **Category** {{mobile|a11y|perf|motion|seo|security|fit}} · **WCAG** {{SC or —}}
**Where** {{URL}}, {{section}}, `{{selector}}`
**Viewports** {{}}

**What happens.** {{Observed behaviour. Not "should".}}

**Evidence.** {{command}} — {{timestamp}}, finding `{{id}}`, measured `{{value}}`. Screenshot: `{{path}}`

**Fix.** {{The concrete change. Pattern reference if one exists.}}

**Effort** {{S|M|L}} · **Verify after fix** {{the exact re-check and its pass criterion}}

---

## 4. Major

> Substantially degrades the experience, or materially undermines the site's stated job.

### M-01 · {{Title}}

{{same structure}}

---

## 5. Minor

| ID | Finding | Where | Category | Fix | Effort |
|---|---|---|---|---|---|
| N-01 | | | | | |

---

## 6. Advisory

> Recommendations, not violations. Sourced from convention rather than evidence — flagged as such.

| ID | Observation | Rationale | Source tier |
|---|---|---|---|
| A-01 | | | {{[E] evidence / [V] vendor / [C] convention}} |

---

## 7. Category fit

> Declared intent vs built site. See `site-categories.md` §4.

| Dimension | Declared in profile | As built | Gap |
|---|---|---|---|
| Primary job | | | |
| Visitor intent | | | |
| Motion budget | | | |
| Information density | | | |
| Novelty budget | | | |
| Trust requirement | | | |

**Assessment.** {{Where the built site diverges from its stated intent, and in which direction.}}

---

## 8. Remediation plan

> Ordered for *work*, not by severity. See `reporting.md` §5.

### Stage 1 — Stop the bleeding
| Finding | Owner | Effort |
|---|---|---|
| | | |

### Stage 2 — Systemic fixes
> One change resolving many findings — a template, component, token or base style.

| Change | Resolves | Owner | Effort |
|---|---|---|---|
| | | | |

### Stage 3 — High-reach instances
### Stage 4 — Long tail (batched by type)
### Stage 5 — Process
> The change that stops this recurring: a CI gate, a checklist, a component fix.

---

## 9. Appendix

**Raw output**

| File | Contents |
|---|---|
| `audit/a11y.json` | |
| `audit/responsive.json` + `audit/responsive/*.png` | |
| `audit/perf.json` | |
| `audit/motion.json` | |
| `audit/seo.json` | |
| `audit/headers.json` | |

**Audit profile** — `audit-profile.yaml` as of this run.

**Re-audit instructions**

```bash
# Same commands, same flags, same sample. Report diff, not repeat.
python3 scripts/audit_a11y.py {{URL}} --all --standard {{target}} --json audit/a11y.json
python3 scripts/audit_responsive.py {{URL}} --out audit/responsive --json audit/responsive.json
python3 scripts/audit_performance.py {{URL}} --budget perf-budget.json --json audit/perf.json
python3 scripts/audit_motion.py {{URL}} --json audit/motion.json
python3 scripts/audit_seo.py {{URL}} --max-pages {{n}} --json audit/seo.json
python3 scripts/check_headers.py {{URL}} --json audit/headers.json
```

Report as: **fixed · still open · regressed · newly appeared**. Regressions matter more than new findings — they mean there is no gate in the pipeline.
