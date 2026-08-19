# SEO — Router

Covers: the mental model, which SEO layer files to load, and the order they must run in.
Load: always, and only this file, until the load-decision table says otherwise.
Depends on: the site's audit profile — the table below is keyed to profile facts. **`../scoping.md` produces the profile, and that is a hard gate for the whole audit.** `business-context.md` does not create it; it *augments* the existing profile with the SEO-specific fields (`seo.priority`, `seo.local_business`, `seo.ecommerce`, `seo.ai_visibility`, `seo.data_access`, `site.url_count`, `site.multilingual`, the claimed `stack.rendering`, and the `audience.success_metric` / `engagement.out_of_scope` entries its questions sharpen). If no profile exists, stop and run `../scoping.md`.
Source currency: July 2026. Re-verify anything version-dependent (schema support, crawler user-agent names, AI-surface behaviour) before acting on it.

---

## The mental model

Search visibility is a funnel of gates. Each is binary-ish, and **failing an early gate makes every later optimisation worthless.**

```
1. DISCOVER   → Can a crawler find the URL at all?
2. CRAWL      → Is it allowed to fetch it? (robots.txt, WAF, rate limits)
3. RENDER     → Does the content exist after JS runs?
4. INDEX      → Is it allowed and worth storing? (noindex, canonicals, duplication, quality)
5. UNDERSTAND → Does the machine know what this page is and who made it?
6. RANK       → Is it the best answer relative to competitors?
7. APPEAR     → Does the listing / snippet / citation get chosen and clicked?
8. CONVERT    → Does the visit produce something the business cares about?
```

Audit in that order. Always. The most common reason a site sees no result from months of SEO work is polishing gate 6 while gate 2 is broken.

**Google's official position as of May 2026: optimising for AI search *is* SEO.** Google's first formal guide to generative AI features (published 2026-05-15) states AI Overviews and AI Mode are rooted in the core Search ranking and quality systems — retrieval-augmented generation and query fan-out over the same index. There is no separate AI algorithm to game. `L9-ai-search.md` covers what genuinely differs; everything else is fundamentals.

---

## Load-decision table

| File | Load when | Skip when |
|---|---|---|
| `business-context.md` | Always, first within this track. Augments the profile written by `../scoping.md` with the SEO facts every other row is keyed to. | Never skip — if unanswered, mark the profile fields `unknown` and say so in the report. |
| `L1-foundations.md` | Always. `seo.priority: none` included. | Never. A site that can't be crawled is broken regardless of SEO intent. |
| `L2-technical-performance.md` | `seo.priority: basic\|full`, or any CWV/mobile complaint | `seo.priority: none` **and** `../performance.md` + `../mobile.md` already run — L2 is only the SEO-facing readout of those |
| `L3-architecture.md` | `seo.priority: full`, or `site.url_count > ~30`, or `seo.ecommerce: true`, or `site.multilingual: true` | Single-page sites, or `site.url_count < ~10` with flat navigation |
| `L4-onpage.md` | `seo.priority: basic\|full` | `seo.priority: none` — the MVP pass below covers the non-negotiable subset |
| `L5-content.md` | `seo.priority: full`, or the site publishes content (blog, docs, resources, guides) | Brochure sites with no editorial output and no content ambition |
| `L6-structured-data.md` | `seo.priority: basic\|full`, or `seo.ecommerce: true`, or `seo.local_business: true`, or the site already ships JSON-LD | `seo.priority: none` and no existing markup |
| `L7-authority.md` | `seo.priority: full`, or the site competes commercially for search traffic | `seo.priority: none\|basic`; internal tools; sites with no competitive search ambition |
| `L8-local.md` | `seo.local_business: true` — physical location or defined service area | `seo.local_business: false`. Portfolios, SaaS, lab sites, publishers. **Default skip.** |
| `L9-ai-search.md` | `seo.ai_visibility: true`, or `seo.priority: full`, or the client asks about AEO/GEO/LLM visibility, or is being sold "GEO services" | `seo.priority: none` — except the crawler-governance check, which is folded into the MVP pass |
| `L10-measurement.md` | `seo.priority: basic\|full`, or any claim of results will be made | One-off audits with no ongoing owner. Note the absence as a finding. |
| `L1-foundations.md` §"Rendering" deep dive | `stack.rendering: csr\|hybrid\|unknown`, **or** any framework-driven site where the rendering mode has not been verified against actual output | `stack.rendering: ssg` **and** verified by the view-source test in that section. Verify before skipping — never skip on framework name alone. |
| `triage.md` | Always, at the end. Read it when you have 40 findings and limited time. | Never. |
| `question-list.md` | Interview mode; when the client is available to answer; when the audit must cover strategy not just crawl output | Pure unattended technical crawls |
| `tools.md` | When you need the verification instrument for a check, or want the ready-made AI audit prompts | Never worth skipping; it is lookup, not reading |

**Rendering is the trap.** `stack.rendering` recorded in the profile is a *claim*. `L1-foundations.md` tells you how to test it. Do not skip the rendering section because the profile says `ssg`.

---

## Order of operations

The layers are **dependency-ordered, not a menu.** You cannot optimise on-page content for a page that cannot be crawled. You cannot fix cannibalisation on pages that are not indexed. You cannot earn a citation for a passage a retrieval bot is blocked from fetching.

```
business-context.md   (Layer 0 — why, for whom, where. No technical work before this.)
        ↓
L1-foundations        (crawl · render · index) ─── gate. If this fails, STOP and report.
        ↓
L2-technical-perf     (CWV · mobile · hygiene) ─── affects ranking and crawl efficiency
        ↓
L3-architecture       (IA · internal links · URLs) ── determines which pages get equity
        ↓
L4-onpage             (per-page elements) ──── only meaningful on indexed, linked pages
        ↓
L5-content            (research · E-E-A-T · quality bar) ── where SEO is won or lost
        ↓
L6-structured-data    (machine comprehension) ── describes content that must already exist
        ↓
L7-authority ─┬─ L8-local (conditional) ─┬─ L9-ai-search
              └───────────────────────────┘
        (off-site + surface-specific; all three assume L1–L6 pass)
        ↓
L10-measurement       (baseline · monitoring · governance) ── set up BEFORE changes ship
```

Two order exceptions worth knowing:

- **L10's baseline snapshot is taken first, chronologically.** You cannot claim a result without a dated before. Read L10 last; execute its "baseline" block on day 1.
- **A P0 finding in L1 halts the audit.** Report it, get it fixed, re-crawl, then continue. Do not deliver a 40-item content critique for a site serving `Disallow: /`.

---

## Minimum viable SEO pass

For `seo.priority: none` — lab sites, personal portfolios, internal tools, one-off project pages. The goal is not ranking; it is **not shipping something broken.** Nine checks, ~20 minutes, no Search Console required.

- [ ] **robots.txt isn't blocking the site** — `curl -s https://DOMAIN/robots.txt`. Pass: no `Disallow: /` under `User-agent: *`. This is the single most common catastrophe, usually a staging config that shipped.
- [ ] **No sitewide `noindex`** — `curl -sI https://DOMAIN/ | grep -i x-robots-tag` and `curl -s https://DOMAIN/ | grep -i 'name="robots"'`. Pass: neither contains `noindex`.
- [ ] **HTTPS works and HTTP redirects to it** — `curl -sI http://DOMAIN/ | head -1`. Pass: `301` (or `308`) to the `https://` URL, valid unexpired certificate.
- [ ] **One canonical hostname** — `curl -sI https://www.DOMAIN/` and `curl -sI https://DOMAIN/`. Pass: one 200, the other 301s to it. Both serving 200 is duplicate-content sloppiness.
- [ ] **Content exists without JavaScript** — `curl -s https://DOMAIN/ | wc -c` and read the body, or load with JS disabled in DevTools. Pass: the main copy is in the initial HTML. Fail → read the rendering section of `L1-foundations.md` before anything else.
- [ ] **Title and meta description exist and are unique per page** — view source, or crawl with Screaming Frog (free ≤500 URLs). Pass: every page has a distinct `<title>`; no `Untitled`, no template default, no duplicates.
- [ ] **One `<h1>` per page, matching the subject** — `curl -s URL | grep -o '<h1[^>]*>' | wc -l`. Pass: exactly 1.
- [ ] **404s return 404** — `curl -sI https://DOMAIN/definitely-not-a-real-page-xyz | head -1`. Pass: `HTTP/2 404`. A `200` here is a soft 404 and pollutes the index.
- [ ] **AI crawler policy is deliberate, not accidental** — read robots.txt line by line. Pass: every `Disallow` aimed at `GPTBot`, `OAI-SearchBot`, `ClaudeBot`, `Claude-SearchBot`, `PerplexityBot`, or `Google-Extended` traces to a decision someone actually made. Blocking retrieval bots by CDN default while wanting AI visibility is a common self-own — see `L9-ai-search.md` §crawler governance.

If all nine pass, the site is not broken. Anything beyond this is optimisation, and for `seo.priority: none` it is out of scope — say so rather than padding the report.

---

## Honesty rules for the report

- Items marked 🔒 throughout these files **require first-party data** — Search Console, Bing Webmaster Tools, CrUX field data, server logs, backlink tools, analytics. An audit cannot verify them from the outside. Mark them `CAN'T VERIFY` and name the data source needed. Do not infer.
- Status vocabulary for every check: `PASS` / `FAIL` / `CAN'T VERIFY` / `N/A`, plus the evidence used and a severity from `triage.md`.
- Keep the source's epistemic layering visible. These files distinguish **Google says** (documented, citable), **practitioners believe** (correlational, contested), and **hype** (marketed, unevidenced). Do not flatten the three into confident advice.

## Related references outside this directory

| Path | Covers |
|---|---|
| `../performance.md` | Full performance treatment. L2 only reads the SEO-facing consequences. |
| `../mobile.md` | Full mobile/responsive treatment. L2 only covers mobile-first indexing parity. |
| `../ada/` | Accessibility. Overlaps with L2 hygiene and L9 agentic access — semantic HTML does double duty. |
| `../../scripts/audit_seo.py` | Automated collection of the mechanical checks (robots.txt, headers, canonicals, titles, status codes). |
