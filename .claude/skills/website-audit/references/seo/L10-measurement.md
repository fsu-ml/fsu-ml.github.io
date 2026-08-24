# Layer 10 — Measurement, monitoring & governance

Covers: instrumentation, baselines, the maintenance cadence, and pre-launch safety. Merges the source's Layers 10 and 11.
Load: `seo.priority: basic|full`, or whenever a claim of results will be made.
Depends on: nothing — but **execute the baseline block first, chronologically**, before any change ships. Read this file last; run its §10.2 on day 1.

You cannot audit what you don't measure, and you cannot claim results without a baseline. SEO decays; sites rot. **Cadence beats intensity.**

🔒 = requires first-party access.

---

## 10.1 Instrumentation

- [ ] **Google Search Console verified and reviewed at least monthly** 🔒 — the only ground truth for Google. Pass: verified property, named owner, monthly review happening. Panels that must be covered:

  | Panel | What it tells you | Check for |
  |---|---|---|
  | Performance | Impressions, clicks, CTR, average position — by query and by page | Trend vs. previous period; the zero-click pattern (`L9-ai-search.md` §9.11) |
  | Pages | Indexing status and exclusion reasons | Gaps vs. sitemap; "Crawled/Discovered – currently not indexed" |
  | Core Web Vitals | Field data by URL group | Any group in the "poor" band (`L2-technical-performance.md`) |
  | Enhancements | Structured data validity | Errors at zero (`L6-structured-data.md`) |
  | Manual Actions & Security Issues | Penalties and compromises | **Check these first if traffic collapses** |
  | Generative AI performance | AI Overviews / AI Mode impressions and clicks | The only first-party AI data that exists |

- [ ] **Bing Webmaster Tools verified** 🔒 — matters for Bing directly and for ChatGPT search indirectly. Pass: verified, index coverage roughly matches real page count.
- [ ] **Analytics installed, with organic traffic segmented** 🔒 — GA4 or a privacy-friendly alternative. Pass: an organic-only view exists, and AI referral domains are segmented (`L9-ai-search.md` §9.11).
- [ ] **Conversions defined and tracked** 🔒 — form submits, calls, purchases, signups. Pass: each maps to the success metric recorded in `business-context.md`. **Traffic without conversion data is vanity.**
- [ ] **Rank tracking for a defined keyword set** — Pass: a fixed set, tracked, and understood as **directional rather than absolute**. Results are personalised, localised, and increasingly displaced by AI surfaces; treat movement as a signal, not a score.
- [ ] **Uptime and performance monitoring with alerts** — Pass: a monitor exists and someone receives its alerts. Regression thresholds at ~80% of CWV limits (`L2-technical-performance.md`).

---

## 10.2 Baseline — do this before any change ships

- [ ] **Baseline snapshot taken and dated** 🔒 — capture, on one dated page: GSC impressions/clicks/CTR/average position for the trailing 3 months; indexed page count; mobile CWV field numbers; organic sessions and conversions; rank positions for the tracked set; the AI visibility prompt log from `L9-ai-search.md` §9.11. Pass: the snapshot exists with a date on it. Without it, nothing you do afterwards is attributable.
- [ ] **Changes logged with dates** — Pass: a change log (a sheet is fine) recording what shipped and when, so effects can be attributed later. Deploys, content publishes, redirect batches, robots.txt edits.
- [ ] **Reporting cadence agreed, with metrics tied back to the Layer 0 business goal** — Pass: a named interval, a named audience, and metrics that answer the success question from `business-context.md` rather than whichever numbers happen to be up.

---

## 10.3 The maintenance cadence

**Weekly**
- [ ] Check GSC for new errors, manual actions, security issues
- [ ] Monitor rankings and traffic for anomalies
- [ ] Respond to new reviews

**Monthly**
- [ ] Full GSC performance review vs. previous period
- [ ] Core Web Vitals field data check
- [ ] Broken link scan
- [ ] Publish or update content per plan
- [ ] AI visibility prompt check (`L9-ai-search.md` §9.11)
- [ ] GSC Generative AI performance report review 🔒

**Quarterly**
- [ ] Full technical crawl (Screaming Frog / Sitebulb)
- [ ] Content audit: prune, merge, refresh (`L5-content.md`)
- [ ] Backlink profile review (`L7-authority.md`)
- [ ] Competitor SERP re-analysis
- [ ] Cannibalisation check (`L3-architecture.md`)
- [ ] Structured data validation sweep (`L6-structured-data.md`)

**Annually, or on any major change**
- [ ] Full audit against this reference set, starting at `00-map.md`
- [ ] Information architecture review
- [ ] Any migration, redesign, or replatform gets an SEO plan **before** launch, including complete URL mapping and 301s

---

## 10.4 Pre-launch gate — always, before pushing anything live

Four checks. They take five minutes and prevent the most expensive failures in `triage.md`.

- [ ] **Staging `robots.txt` not deployed to production** — `curl -s https://DOMAIN/robots.txt`. Pass: no `Disallow: /`.
- [ ] **`noindex` tags removed from production** — `curl -sI https://DOMAIN/ | grep -i x-robots-tag` and grep the HTML for `name="robots"`. Pass: no `noindex` in either.
- [ ] **Redirects mapped for every changed URL** — diff the old sitemap against the new one; every removed URL needs a 301 to its nearest equivalent. Pass: zero unmapped removals.
- [ ] **Analytics and GSC still firing after launch** — load a page and confirm the tag fires; check GSC for data within 48 hours. Pass: both confirmed.

Post-launch, re-run the first 48 hours' worth: status codes on the top 50 URLs, index coverage trend, and CWV lab numbers on the new templates.

---

## Questions to ask

1. What were my organic sessions and conversions this month last year? 🔒
2. Which 10 pages drive the most organic traffic, and which drive the most revenue? *(Rarely the same list.)* 🔒
3. Which queries earn impressions but almost no clicks? *(Positions 5–15 with weak CTR are usually the fastest available wins.)* 🔒
4. If traffic dropped 40% tomorrow, what's the first thing I'd check?
5. Am I measuring what matters, or what's easy to measure?
6. What's my maintenance cadence, and who owns it?

---

Related: `triage.md` for what to do with the findings; `tools.md` for the instruments; `L9-ai-search.md` §9.11 for AI-specific measurement.

**SEO is not a project with an end date; it's a maintenance discipline with occasional bursts of construction.**
