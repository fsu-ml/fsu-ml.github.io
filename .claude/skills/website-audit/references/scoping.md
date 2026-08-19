# Scoping — the intake gate

**Covers:** the questions that must be answered before any audit work begins, and the `audit-profile.yaml` they produce.
**Load when:** always, first, on every invocation of this skill — unless a valid profile already exists for this site.
**Produces:** `audit-profile.yaml` in the site's own repository.

---

## Why this is a gate, not a preamble

Every routing decision in this skill keys off the profile. Without it:

- You load PDF accessibility guidance for a site that serves no PDFs, and skip it for one that serves four hundred.
- You audit against WCAG 2.1 AA when the site is a state government service that must meet 2.2 AA by a statutory deadline.
- You flag a portfolio's expressive motion as excessive, or wave through an e-commerce checkout's.
- You report findings the client cannot act on and miss the one that creates legal exposure.

**Do not begin an audit without a profile.** If asked to "check this site" with no profile present, run the interview first. If the user is unavailable, write the profile with `unknown` values and a `# UNVERIFIED` marker on each, and open the report with an explicit statement of which findings are unreliable as a result.

---

## Procedure

```
1. Look for audit-profile.yaml    → repo root, then ./docs/, then ./.audit/
2. Found?  → §3 Revalidate.  Not found? → §2 Interview.
3. Detect what can be detected    → do not ask what a crawl can answer
4. Write / update the profile     → commit it to the site's repo
5. Only then: choose reference files
     ada/ and seo/ each have a 00-map.md load table keyed to this profile;
     the top-level references/ files are routed from SKILL.md
```

### Detect before you ask

Several profile fields are facts about the built site, not preferences. Determine them, then confirm — do not make the user guess.

| Field | Determine by |
|---|---|
| `content.has_pdfs`, `has_office_docs`, `has_epub` | `python3 scripts/audit_a11y.py <url> --inventory-documents` |
| `content.has_forms`, `has_search`, `has_auth` | crawl for `<form>`, `<input>`, login routes |
| `content.has_video`, `has_audio` | crawl for `<video>`, `<audio>`, embedded player origins |
| `content.has_animation`, `has_carousel` | `python3 scripts/audit_motion.py <url>` |
| `content.has_visualizations` | crawl for `<canvas>`, `<svg>` carrying data, known viz library globals |
| `stack.rendering` | **raw HTML vs rendered DOM diff** — `scripts/audit_seo.py`. Never infer from the framework name. |
| `site.url_count` | sitemap count, or crawl |
| `site.multilingual` | `hreflang`, `lang` attributes, language switcher |

`stack.rendering` recorded from a client's answer is a *claim*. Verify it. This is the single most common source of wrong SEO conclusions.

---

## The interview

Ask in groups. Stop early if the answers make later groups irrelevant. Every question exists because an answer changes what the audit does — none are for the file.

### Group 1 — What is this, and what is it for?

1. **What is the site's URL, and is there a staging/preview URL I should audit instead?**
2. **What is the site's primary job?** — transact · persuade · inform · demonstrate craft · archive. *(Sets `site.primary_job`; drives `site-categories.md`.)*
3. **What category is it closest to?** — e-commerce · portfolio · lab/academic · SaaS/B2B · government · healthcare · news/editorial · nonprofit · local business · documentation · other. *(If "other", `site-categories.md` §3 has a procedure for placing a novel type.)*
4. **Who is the visitor, and what state are they in when they arrive?** Goal-directed and impatient, or exploratory and receptive?
5. **What does success look like?** One sentence. A number if there is one.
6. **How long is this site expected to live, and who maintains it?** *(A site with no maintainer should not ship a dependency-heavy architecture. Sets `site.expected_life`, `site.maintainer`.)*

### Group 2 — Compliance ★ *ask this early; it changes everything downstream*

7. **Does this site have to meet an accessibility standard?**
   Ask it plainly, then probe — clients frequently do not know they are covered:
   - Is the organisation a **state or local government entity**, or acting on one's behalf? → ADA Title II, DOJ web rule, hard deadline.
   - Is it a **federal agency, or a vendor/grantee delivering to one**? → Section 508, and note that 508 obligations **flow down through grants and contracts** — a university lab publishing a federally funded deliverable inherits them.
   - Does it receive **federal financial assistance**? → Section 504.
   - Is it a **place of public accommodation** — retail, healthcare, education, hospitality? → Title III exposure regardless of any rule.
   - Does it serve the **EU/UK**, or sell to European consumers? → EAA / EN 301 549.
   - Is there a **contractual** accessibility requirement, a VPAT request, or a procurement questionnaire in play?
   - Has anyone **received a demand letter or complaint**?

   → sets `compliance.regime`. If the answer is genuinely "no requirement", set `none` — but still run the minimum pass in `ada/00-map.md`. It is two hours and it catches the failures that actually block people.

8. **What conformance target?** WCAG 2.0 AA / 2.1 AA / **2.2 AA (default; building to 2.2 satisfies the others)**. → `compliance.target`
9. **Is there a deliverable beyond the audit?** VPAT/ACR · accessibility statement · documented exception · remediation roadmap · none. → `compliance.deliverable`
10. **Does the site publish PDFs, Word/PowerPoint/Excel files, or EPUBs?** Confirm the crawl result with the client — documents are often served from a CMS the crawl cannot reach. If yes: **are any of the PDFs produced from LaTeX?** → `content.has_pdfs`, `has_office_docs`, `has_latex_pdfs`, `has_epub`

### Group 3 — Mobile

11. **Is this site intended to be mobile friendly?** Effectively always yes; ask anyway, because "yes" commits you to the section-by-section verification in `mobile.md` §5, not a glance at one page.
12. **What is the realistic device and network floor?** Mid-tier Android on 4G, or desktop-on-fibre only (rare, and usually wrong)? → `perf.device_floor`
13. **Is there a native app, or any mobile-specific route** the web version defers to?
14. **Any known sections that misbehave on mobile?** Tables, maps, embedded tools, iframes, data grids and third-party widgets are the usual offenders. Record them; they get audited individually.

### Group 4 — Motion, animation, visualization

15. **How much motion does this site want?** none · restrained · expressive. Be concrete: *what should happen when a visitor scrolls?* → `motion.budget`
16. **Is content revealed as the user scrolls?** If yes, `dynamic-loading.md` fail-open checks become **blocker-severity** — content hidden by JS that never reveals is the most damaging failure in this whole skill.
17. **Are there charts, diagrams, 3D scenes, or mathematical visualizations?** → `content.has_visualizations`, routes to `viz-libraries.md`
18. **Is there a licence constraint on dependencies?** Some institutions require OSI-approved licences — which rules out GSAP despite it being free. → `stack.licence_policy`
19. **Any hard performance requirement?** A stated budget, a contractual SLA, a competitor to beat.

### Group 5 — SEO and discoverability

20. **How much does search visibility matter?** none · basic · full. "Basic" means: don't ship something broken. "Full" means: compete. → `seo.priority`
21. **Physical location or defined service area?** → `seo.local_business` *(the only trigger for the local SEO layer)*
22. **Does it sell online?** → `seo.ecommerce`
23. **Does the client care about appearing in AI answers** (ChatGPT, Perplexity, AI Overviews)? Or are they being sold "GEO services"? → `seo.ai_visibility` *(the second phrasing matters — `seo/L9-ai-search.md` has the hype-vs-evidence section they need)*
24. **Which of these do I have for this domain — Search Console, Bing Webmaster Tools, analytics, CrUX, raw server logs, a backlink tool?** → `seo.data_access` *(record each one held, not a yes/no. Without field data all performance findings are lab-only; and every 🔒 item in `seo/L1-foundations.md` resolves to `CAN'T VERIFY` without its matching instrument — say so in the report rather than passing the check silently)*

### Group 6 — Constraints and delivery

25. **What is out of scope?** Third-party embeds, a legacy section, a CMS nobody controls, content the client cannot change.
26. **What is the fix budget?** Determines whether `reporting.md` §5 and `seo/triage.md` produce a top-5 or a full backlog.
27. **Is this a pre-launch gate or an audit of a live site?** Pre-launch means findings block a deploy; live means they compete with other work.
28. **Where should the report and profile live?** Default: the site's own repo.

---

## Mid-build use

This skill is a post-completion checker, but the rules do not change mid-build — so it is designed to be referenced during development too.

| Mode | What runs |
|---|---|
| **Scoping** (before any code) | Groups 1–6. Write the profile. The profile is now the build's spec for these concerns. |
| **Mid-build consultation** | Profile must already exist. Load only the reference relevant to the question. Do not run the full audit. |
| **Section gate** | When a section is "done": run its slice of `mobile.md` §5 and `animation-and-motion.md`. This is where the user's stated problem — *individual sections failing while the page broadly passes* — actually gets caught. |
| **Pre-launch audit** | The full pass. `verification-protocol.md`. |
| **Post-launch re-audit** | Diff against the previous report. Field data is now available; re-check every lab-only finding against it. |

**When consulted mid-build, answer from the reference and say which check will later verify it.** "Use `<button>`, not `div onclick` — `audit_a11y.py --all` will flag this at the gate" is more useful than the rule alone, because it tells the developer the claim is falsifiable.

---

## Profile schema

Template: `../templates/audit-profile.yaml`. Full key list:

```yaml
site:
  url:              https://example.com
  staging_url:      null
  type:             web | documents | mixed
  primary_job:      transact | persuade | inform | demonstrate-craft | archive
  category:         ecommerce | portfolio | lab | saas | government | healthcare |
                    news | nonprofit | local-business | documentation | other
  url_count:        int
  multilingual:     bool
  expected_life:    months | years | indefinite
  maintainer:       string | none

audience:
  intent:           goal-directed | exploratory | mixed
  success_metric:   string

compliance:
  regime:           title-ii | title-iii | section-508 | section-504 | acaa |
                    eaa | contractual | other | none
  target:           wcag20-aa | wcag21-aa | wcag22-aa
  deliverable:      none | vpat | statement | exception | roadmap
  known_complaint:  bool

content:
  has_forms: bool            has_auth: bool             has_search: bool
  has_video: bool            has_audio: bool            has_animation: bool
  has_carousel: bool         has_images_of_text: bool   has_data_tables: bool
  has_custom_widgets: bool   has_multistep_process: bool
  has_visualizations: bool   has_animation_layer: bool
  has_pdfs: bool             has_office_docs: bool      has_latex_pdfs: bool
  has_epub: bool             has_html_email: bool
  volume:           sparse | moderate | dense

stack:
  rendering:        ssg | ssr | csr | hybrid | unknown    # VERIFY — do not trust
  cms:              string | none
  hosting:          string
  licence_policy:   any | osi-only
  third_parties:    [list of origins]

mobile:
  required:         bool
  known_problem_sections: [list]

motion:
  budget:           none | restrained | expressive
  scroll_reveals:   bool
  reduced_motion_handled: bool     # VERIFY, do not trust

perf:
  device_floor:     mid-android-4g | modern-mobile | desktop-only
  budget_file:      path | none
  hard_requirement: string | none

seo:
  priority:         none | basic | full
  local_business:   bool
  ecommerce:        bool
  ai_visibility:    bool
  data_access:      [search-console, bing-webmaster, analytics, crux,
                     server-logs, backlink-tool, none]

engagement:
  mode:             pre-launch-gate | live-audit | mid-build
  out_of_scope:     [list]
  fix_budget:       top-5 | prioritised | full-backlog
  report_path:      ./audit-report.md

meta:
  created:          YYYY-MM-DD
  last_verified:    YYYY-MM-DD
  verified_by:      string
```

---

## Revalidating an existing profile

A profile older than the site is worse than no profile — it routes confidently to the wrong references.

At the start of every audit run:

| Check | Action if it fails |
|---|---|
| `meta.last_verified` within 90 days, or before the last significant deploy | Re-run detection (§ Detect before you ask) |
| Detected `content.*` flags match recorded ones | Update, and flag any newly-appeared document format — a site that just started publishing PDFs has an unaudited compliance surface |
| `stack.rendering` still matches the raw-vs-rendered diff | Update. A migration from SSG to CSR invalidates every SEO finding in the previous report. |
| `compliance.regime` unchanged | Re-ask. Regulatory deadlines move and the organisation's status can change. |
| Fields marked `unknown` or `# UNVERIFIED` | Ask again before writing a report that depends on them |

Record the diff. A changed profile is itself a finding worth reporting.

---

## Related

- `verification-protocol.md` — what to run once the profile exists.
- `reporting.md` — how findings become a report.
- `ada/00-map.md`, `seo/00-map.md` — the load tables this profile feeds.
- `site-categories.md` — turns `site.primary_job`, `audience.intent` and `motion.budget` into checkable build decisions.
- `../templates/audit-profile.yaml` — the file to copy.
