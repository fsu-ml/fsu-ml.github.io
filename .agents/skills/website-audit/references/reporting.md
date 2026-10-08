# Reporting — severity, triage, and the deliverable

**Covers:** how observations become a severity-triaged report the client can act on.
**Load when:** stage 11 of `verification-protocol.md`, or whenever writing up findings.
**Produces:** `audit-report.md` at `engagement.report_path`, plus the raw `--json` outputs beside it.

---

## 1. What the report is for

Not to demonstrate thoroughness. **To get the right things fixed, in the right order, with enough detail that someone can fix them without asking you a question.**

That means three properties:

1. **Ordered by consequence, not by category.** A report grouped as "Accessibility / Performance / SEO" forces the reader to interleave three lists themselves. Group by severity; tag by category.
2. **Every finding actionable.** A finding without a concrete fix is a complaint.
3. **Honest about what was not checked.** See `verification-protocol.md` §6.

---

## 2. Severity

Four levels. Assign by **consequence to a user or to the client**, never by how hard it is to fix.

| Severity | Definition | Examples |
|---|---|---|
| **Blocker** | Someone cannot complete a task, cannot access content, or the client has legal exposure. Ships nothing until fixed. | Keyboard trap in checkout · form field with no label · content hidden by a scroll reveal that never fires · site unusable at 360px · contrast failure on body text · `noindex` on production · WCAG A/AA failure where `compliance.regime ≠ none` |
| **Major** | Substantially degrades the experience for many users, or materially undermines the site's stated job. | LCP > 4s on the device floor · sustained jank during the primary scroll · missing reduced-motion handling · no error states on a form · broken internal links in navigation · missing canonical on a duplicated template |
| **Minor** | Real but narrow. Affects some users, some of the time, or is a quality gap with a bounded cost. | Heading level skipped · non-optimal image format · missing `Permissions-Policy` · inconsistent spacing scale · third-party script that could be deferred |
| **Advisory** | A recommendation, a preference, or a claim that is not independently defensible. Includes everything tagged `[C]` in `site-categories.md`, explicitly labelled as convention. **`[F]` folklore is not an advisory — it does not ship at any severity.** | "Consider container queries here" · category-fit observations · design opinions · anything sourced from practitioner convention rather than evidence |

### Rules

- **Compliance regime escalates.** Any WCAG A or AA failure is a **blocker** when `compliance.regime` is anything other than `none`. When it is `none`, it is major or minor by user impact — but say plainly in the report that this classification would change if the site's legal status changed.
- **Blockers on a conversion path escalate.** A minor issue in a checkout, application or signup flow is at least major. Flow completion is binary.
- **Never let effort influence severity.** "Hard to fix" belongs in the effort column, not the severity column. Conflating them is how blockers get quietly reclassified.
- **Advisory is not a dumping ground.** If it cannot be tied to evidence or a named rule, ask whether it belongs in the report at all.
- **`[F]` folklore never appears in the report at all.** Not as a finding, not at advisory. `site-categories.md` §1 is binding here.

### Crosswalk — the local P-scales

Three reference files run their own priority scale inside their checklists. Those scales are **local working notation only**. Nothing reaches the report except the four canonical severities above; translate before writing a finding.

| Canonical | `animation-and-motion.md` §checklists · `dynamic-loading.md` §3.3 (P0–P2) | `seo/triage.md` §1 (P0–P4) |
|---|---|---|
| **Blocker** | **P0** — blocking defect | **P0** (emergency, halts the audit) · **P1** (critical) |
| **Major** | **P1** — should fix | **P2** (high) |
| **Minor** | — (no direct equivalent; grade by consequence) | **P3** (medium) |
| **Advisory** | **P2** — advisory | **P4** (ongoing) |

Two cautions. The two P-scales are **not** the same scale: `P2` means *advisory* in the motion and dynamic-loading checklists and *High* — i.e. major — in `seo/triage.md`. And the mapping is a default, not an override: the §2 Rules still apply on top of it, so a `seo/triage.md` P2 sitting on a conversion path is a blocker, and a motion-checklist P1 under `compliance.regime ≠ none` may be too.

---

## 3. Finding format

```markdown
### B-04 · Pricing table overflows viewport below 400px

**Severity** blocker · **Category** mobile · **WCAG** 1.4.10 Reflow (AA)
**Where** `/pricing`, `<section class="pricing">`, selector `.pricing-table`
**Viewports** 320×568, 360×640, 390×844

**What happens.** The table has `min-width: 720px`, forcing 84px of horizontal
page scroll at 360px. Content in the third column is unreachable without
two-dimensional scrolling.

**Evidence.** `audit_responsive.py https://example.com/pricing` — 2026-08-19,
finding `overflow-x`, measured `scrollWidth 444 / clientWidth 360`.
Screenshot: `audit/responsive/360-pricing.png`

**Fix.** Wrap the table in a scroll container with `overflow-x: auto` and an
accessible name plus `tabindex="0"` so it is keyboard-scrollable, or switch to a
stacked card layout below 640px. Do not set `min-width` on the page-level
container. Pattern: `examples/responsive-foundations.md` §4.

**Effort** S · **Verify after fix** re-run `audit_responsive.py`; `overflow-x` finding absent at all viewports
```

The **Verify after fix** line is not optional. The whole skill is built on verification; a fix nobody re-tested is an assumption again.

---

## 4. Report structure

```
1. Summary            ≤1 page. Counts by severity, the three things that matter,
                      and whether this passes the gate (if pre-launch).
2. Scope & method     Profile summary, sample, tools + versions, dates.
                      What was NOT checked (verification-protocol.md §6).
3. Blockers           Full findings, ordered by user impact.
4. Major              Full findings.
5. Minor              Table form is fine — one line each with a pointer.
6. Advisory           Table form. Clearly marked as recommendations.
7. Category fit       site-categories.md §4 output: declared intent vs built site.
8. Remediation plan   Sequenced. See §5.
9. Appendix           Raw JSON paths, screenshots, the audit profile, coverage
                      statement, re-audit instructions.
```

Template: `../templates/audit-report.md`.

**The summary is the only part most readers will read.** Write it last, make it specific, and put a number in it. "3 blockers, 11 major, 24 minor. The checkout is unusable by keyboard, the pricing table breaks below 400px, and LCP on inner pages is 5.1s on the target device" beats any amount of prose.

---

## 5. Remediation sequencing

Findings ordered by severity is a report. Findings ordered for *work* is a plan, and they are different orders.

| Sequence | Contains | Rationale |
|---|---|---|
| **1. Stop the bleeding** | Anything actively harming users or creating exposure right now | Blockers, plus anything on a conversion path |
| **2. Systemic fixes** | One change that resolves many findings — a template, a component, a design token, a base style | Highest findings-fixed-per-hour by a wide margin. Do these before individual instances. |
| **3. High-reach instances** | Findings on the most-trafficked pages | Impact per fix |
| **4. The long tail** | Everything else, batched by type | Batching by type is faster than batching by page |
| **5. Process** | The change that stops it recurring — a CI gate, a checklist, a component fix | Without this you re-audit into the same findings next year |

Group by **owner** where possible. A report that hands the developer, the content editor and the designer each their own list gets acted on; a single undifferentiated list gets deferred.

For SEO-specific prioritisation, `seo/triage.md`. For accessibility remediation strategy and the harm-then-reach model, `ada/program.md`.

---

## 6. Sourcing and defensibility

Every finding must survive the client asking *"says who?"*

| Finding type | Cite | Defensible? |
|---|---|---|
| WCAG failure | SC number, level, version | Yes — normative |
| Core Web Vitals | Metric, threshold, p75 field or lab, tool + version | Yes, if you state field vs lab |
| Security header | The header, the recommended policy, the check command | Yes |
| Evidence-backed UX (`[E]`) | The study, its date, its scope | Yes |
| Vendor data (`[V]`) | Source, and note who benefits from the finding | Qualified yes |
| Convention (`[C]`) | Say it is convention | Advisory only |
| Folklore (`[F]`) | — | **No — at any severity, including advisory.** Not as a finding, not as supporting colour, not as a parenthetical. Delete it. (`site-categories.md` §1) |

Date your citations. Regulatory deadlines, browser support and tool behaviour all move; a report that does not say when it was written cannot be trusted a year later.

**Never state a threshold you cannot source.** Where a script used a value not backed by a reference document, it labels it a script decision — carry that label into the report.

---

## 7. Re-audit

The second audit should be a diff, not a repeat.

1. Keep the raw `--json` from every run alongside the report.
2. Re-run the same commands with the same flags against the same sample.
3. Report: **fixed · still open · regressed · newly appeared**.
4. Regressions matter more than new findings — a regression means the fix did not hold, and usually means there is no gate in the pipeline.
5. Update `meta.last_verified` in the profile.

If more than a handful of findings regressed, the finding to report is the absence of a CI gate, not the individual regressions.

---

## Related

- `verification-protocol.md` — produces the findings this file formats.
- `scoping.md` — the profile that determines severity escalation.
- `seo/triage.md`, `ada/program.md` — domain-specific prioritisation.
- `site-categories.md` §1 — the evidence-tagging convention severity depends on.
- `../templates/audit-report.md` — the file to copy.
