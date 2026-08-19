# ADA / WCAG reference — routing map

**Covers:** how the ADA/WCAG reference set is partitioned, and which file to load for a given site.
**Load when:** always. This is the only `references/ada/` file loaded unconditionally.
**Do not** load the whole directory. Each file below is self-contained; load the slice the audit profile calls for.

---

## Orientation: three layers, do not confuse them

"ADA compliant" is not a technical spec. **Layer 1 is law** (ADA Title II/III, Section 508, Section 504, EAA, AODA, state statutes) — written in terms of outcomes and non-discrimination, and it almost always *points at* a technical standard rather than defining one. **Layer 2 is the standard** (WCAG 2.0/2.1/2.2, EN 301 549, PDF/UA) — technology-neutral, testable success criteria that say WHAT must be true, not HOW. **Layer 3 is techniques** (HTML/ARIA patterns, PDF tag structures, LaTeX tagging, Word styles) — non-normative implementations, many routes to the same criterion. You conform to a *standard*, which is how you demonstrate compliance with a *law*. Nobody certifies you as ADA compliant; there is no such certification. What you produce is evidence: audit reports, a VPAT/ACR, an accessibility statement, and a remediation record. A second consequence: WCAG was written for web pages, so applying it to a PDF or Word file needs a translation layer (WCAG2ICT) plus a format-specific companion standard (PDF/UA) — see `documents-pdf.md`.

---

## Load-decision table

Conditions key off the site's audit profile (`audit-profile.yaml` — the file `../scoping.md` writes from `../../templates/audit-profile.yaml`). Where a key is absent, treat it as `false` and skip the file.

| Read this file | Load it when | Skip it when |
|---|---|---|
| `html-core.md` | Always, for any HTML audit. Document skeleton, landmarks, headings, images/alt, links vs buttons, tables, contrast, focus, zoom/reflow, ARIA. | Never skip for a web audit. Skip only for a documents-only engagement (`site.type: documents`). |
| `html-forms.md` | `content.has_forms: true` OR `content.has_auth: true` OR `content.has_search: true` (a search box is a form) OR `content.has_multistep_process: true` | Site is purely static reading content with no input of any kind — rare; verify before skipping. |
| `html-core.md` §7 Data tables | `content.has_data_tables: true` — read the section in full rather than skimming it as part of the always-on load. Header association (`<th scope>` / `headers`+`id`), captions, and the layout-vs-data distinction are where these fail. | `content.has_data_tables: false`. A table used purely for layout is still a finding, but it is a §7 finding, not a data-table one. |
| `html-core.md` §11 ARIA + §9 Focus management, with `testing.md` Layer 2 | `content.has_custom_widgets: true` — hand-rolled ARIA. **No scanner can evaluate whether a custom widget's keyboard model is coherent**; this combination is the manual pass that can. Cross-check every widget against `failure-patterns.md`. | `content.has_custom_widgets: false` — the site uses native controls only. Verify by reading the markup, not by asking. |
| `wcag22-new.md` §3.3.7 Redundant Entry, §3.3.8 Accessible Authentication, §3.2.6 Consistent Help | `content.has_multistep_process: true` (checkout, application, wizard) — these three criteria only bite where a process spans steps. Run task 5 in that file's manual task list. | `content.has_multistep_process: false` **and** `compliance.target` is below `wcag22-aa`. |
| `media-and-motion.md` | `content.has_video: true` OR `content.has_audio: true` OR `content.has_animation: true` OR `content.has_carousel: true` OR `content.has_images_of_text: true` | No time-based media, no motion, no auto-updating regions. Note 1.4.2 / 2.2.2 / 2.3.1 are **non-interference** criteria — if there is *any* moving or sounding thing on the page, load this even if it is decorative. |
| `wcag22-new.md` | `compliance.target: wcag22-aa` OR the site was built to a 2.1 mental model and you need to know what it now misses | `compliance.target` is `wcag20-aa` or `wcag21-aa` **and** the client has explicitly declined to exceed it. Building to 2.2 satisfies both, so this is usually a false economy. |
| `failure-patterns.md` | Always, at the start of the manual pass. Symptom-indexed triage list. | Never — it is the cheapest file per finding in the set. |
| `testing.md` | Always, before you record a single finding. Four-layer model, keyboard procedure, screen reader pairs, sampling. | Never for an audit. Skip only if you are answering a narrow implementation question, not auditing. |
| `targets.md` | `compliance.regime` is anything other than `none`, OR the client asks "what do we actually have to do", OR you must state a conformance target in the report. | You are fixing a known bug against an already-agreed target and no legal question is in play. |
| `criteria-index.md` | You need to look up a specific SC number, check level/version of a criterion, or produce a conformance claim / VPAT row list. | Routine auditing. This is a lookup table, not reading material — do not load it "for background". |
| `documents-pdf.md` | `content.has_pdfs: true` | **`content.has_pdfs: false` — do not load.** Most sites serve no PDF. This file also carries the shared WCAG2ICT translation layer used by the other two document files. |
| `documents-latex.md` | `content.has_latex_pdfs: true` (academic, scientific, journal, or mathematics-bearing PDFs, or the client compiles from `.tex`) | **Anything else — do not load.** The overwhelming majority of sites have no LaTeX anywhere. |
| `documents-office.md` | `content.has_office_docs: true` OR `content.has_epub: true` OR `content.has_html_email: true` (covers Word/PowerPoint/Excel, InDesign, Google Docs, Markdown, EPUB, email) | **No downloadable Office/EPUB files and no email deliverable — do not load.** |
| `program.md` | `compliance.deliverable` is `vpat`, `statement`, `exception`, or `roadmap` — i.e. the engagement produces something beyond the audit findings themselves. | The deliverable is the audit report only. |

**Explicitly:** `documents-pdf.md`, `documents-latex.md` and `documents-office.md` are **not** loaded for a site that serves none of those formats. Confirm by crawl, not by assumption — run `../../scripts/audit_a11y.py --inventory-documents <url>` (or `wget --spider -r` piped through an extension filter) and set the profile keys from the result before choosing files.

---

## If you only do one thing

For a site with no formal compliance requirement (`compliance.regime: none`), this is the minimum pass. It is roughly two hours on a mid-size site and it catches the failures that actually block people.

| # | Check | Where the detail lives |
|---|---|---|
| 1 | **Tab through the home page and one form end to end.** Everything interactive reachable, focus always visible, nothing trapped, nothing fully covered by a sticky header or cookie bar. | `testing.md` (Layer 2), `html-core.md` (focus) |
| 2 | **Every form field has a real `<label>`.** A `placeholder` is not a label. | `html-forms.md` |
| 3 | **Every image has a correct `alt`.** Informative → describes information; decorative → `alt=""`; functional → describes the action. Missing `alt` ≠ `alt=""`. | `html-core.md` (alt decision tree) |
| 4 | **Contrast: 4.5:1 body text, 3:1 large text and UI boundaries.** Check the greys first — secondary text and placeholder text fail most often. | `html-core.md` (contrast) |
| 5 | **`<html lang>` set, one `<h1>`, no skipped heading levels.** | `html-core.md` |
| 6 | **No `div onclick`.** Anything clickable is an `<a href>` or a `<button>`. | `html-core.md` |
| 7 | **Zoom the browser to 400% at 1280×1024.** No two-dimensional scrolling, nothing clipped. | `html-core.md` (reflow) |

Anything that fails 1–3 is Tier 1 harm and goes at the top of the report regardless of what the automated scan said.

---

## Audit profile keys referenced above

```yaml
site:
  type: web | documents | mixed
compliance:
  regime: title-ii | title-iii | section-508 | section-504 | acaa | eaa |
          contractual | other | none
  target: wcag20-aa | wcag21-aa | wcag22-aa
  deliverable: none | vpat | statement | exception | roadmap
content:
  has_forms: bool          has_auth: bool           has_search: bool
  has_video: bool          has_audio: bool          has_animation: bool
  has_carousel: bool       has_images_of_text: bool
  has_pdfs: bool           has_office_docs: bool    has_latex_pdfs: bool
  has_epub: bool           has_html_email: bool
  has_data_tables: bool    has_custom_widgets: bool has_multistep_process: bool
```

**Not legal advice.** These files describe technical standards and summarise regulations as publicly published. U.S. rules have moved twice in the last two years; confirm current deadlines with counsel. Source guide prepared 28 July 2026.
