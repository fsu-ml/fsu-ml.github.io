# Layer 2 — Technical health & performance (SEO-facing)

Covers: the SEO consequences of Core Web Vitals, mobile-first indexing, and technical hygiene.
Load: `seo.priority: basic|full`, or any CWV/mobile complaint. Depends on `L1-foundations.md` passing.
**The deep treatment lives elsewhere.** `../performance.md` owns performance diagnosis and optimisation; `../mobile.md` owns responsive design and mobile UX. This file covers only what search engines do with those signals, and does not duplicate the fix guidance.

🔒 = requires first-party data.

---

## 2.1 Core Web Vitals

Google grades these on **field data** from real Chrome users (the Chrome UX Report), at the **75th percentile**, over a **28-day rolling window**. A perfect Lighthouse score on your laptop proves nothing.

| Metric | Measures | "Good" threshold | Alert at (~80%) |
|---|---|---|---|
| **LCP** (Largest Contentful Paint) | Loading | ≤ 2.5 s | 2.0 s |
| **INP** (Interaction to Next Paint) | Responsiveness | ≤ 200 ms | 160 ms |
| **CLS** (Cumulative Layout Shift) | Visual stability | ≤ 0.1 | 0.08 |

**INP replaced First Input Delay in March 2024.** Any guide still listing FID is out of date. INP is the metric most sites fail, and the hardest to fix, because it is a JavaScript architecture problem, not a plugin problem.

- [ ] **Field data pulled, not lab scores** 🔒 — PageSpeed Insights (top panel = CrUX field data, bottom = Lighthouse lab), GSC → Core Web Vitals report, or the CrUX API. Pass: you can state the three 75th-percentile numbers. Fail: quoting a Lighthouse score as the CWV result.
- [ ] **Mobile checked separately from desktop** 🔒 — PSI has a Mobile/Desktop toggle; GSC splits the report. Pass: both readings recorded. Mobile is usually much worse and mobile is what's indexed.
- [ ] **Any metric in the "poor" band fixed before optimising an already-green metric** — order the three by band (poor → needs improvement → good). Pass: the remediation plan starts with the poor one. Ranking effect is threshold-shaped, not linear; moving LCP from 1.8 s to 1.4 s buys nothing.
- [ ] **LCP element identified and it is the right element** — PSI "Largest Contentful Paint element" diagnostic, or DevTools → Performance → Timings. Pass: the LCP element is the hero image or headline, not a cookie banner or a late-loading font. Fixes → `../performance.md`.
- [ ] **CLS: explicit `width`/`height` on every image, video, iframe, and ad slot** — `curl -s URL | grep -oE '<img [^>]*>' | grep -v 'width='`. Pass: no output. Also reserve space for dynamically injected content (banners, embeds).
- [ ] **Regression alerting set at ~80% of thresholds** — a CrUX/RUM monitor firing at LCP 2.0 s, INP 160 ms, CLS 0.08. Pass: an alert exists and has a named recipient.
- [ ] **Team understands the 28-day lag** — Pass: nobody expects a field-data change the day after deploy. Verify with lab data immediately; verify with field data 28 days later. Manage this expectation in writing or the fix gets judged a failure.

**What SEO actually gets from CWV:** it is a real but modest ranking input, and a large user-experience and conversion input. Do not sell CWV work as a ranking silver bullet; do not let a "poor" band sit unfixed either. Google's position is that page experience matters, not that it outweighs relevance.

**Deep dive → `../performance.md`** for LCP/INP/CLS remediation: preloading, critical CSS, `font-display: swap`, long-task breaking, third-party script budgets, main-thread yielding, image formats and CDN strategy.

---

## 2.2 Mobile — the SEO-facing subset

**Mobile-first indexing means the mobile version *is* the indexed version.** Everything Google knows about the site, it learned from the mobile rendering. That is the whole SEO stake here.

- [ ] **Mobile and desktop serve the same content, links, and structured data** — fetch both and diff: `curl -s -A "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/120 Mobile" URL > m.html` and the desktop equivalent, then `diff <(sed 's/<[^>]*>//g' m.html) <(sed 's/<[^>]*>//g' d.html)`. Pass: no content, link, or JSON-LD present on desktop but missing on mobile. **This is the highest-severity check in this section** — content hidden behind a mobile "read more" that never renders, or a desktop-only nav, means Google never sees it.
- [ ] **No intrusive interstitials on entry** — load a deep page fresh in a mobile viewport. Pass: no full-screen overlay before the content on first view. Google explicitly demotes these. Legally required notices (cookie/age consent) sized reasonably are exempt.
- [ ] **Cookie banners and consent modals don't block content or tank CLS** — measure CLS with the banner in the flow. Pass: banner is position-fixed and reserves its own space, doesn't shift the body.
- [ ] **Responsive across real device widths** — real devices or DevTools device emulation, not a resized desktop window. Pass: no horizontal scroll, no cut-off content, at 360px, 390px, and 768px. Details → `../mobile.md`.
- [ ] **Text legible without zoom; tap targets meet the AA size/spacing minimum** — Pass: base body text ≥ 16 px, and interactive targets are **≥ 24 × 24 CSS px** or pass the 24 px-circle spacing test. That is **WCAG 2.2 SC 2.5.8, Level AA** — normative and legally relevant. **44 × 44 is SC 2.5.5, Level AAA**; 44 pt is Apple HIG and 48 dp is Material, both platform advisories. Do not report 44 as a requirement — `../mobile.md` §1.4 calls conflating these "the single most common way an audit report gets discredited". Report the two separately: *fails AA at 24* (violation) and *below the 44/48 platform recommendation* (advisory). Details → `../mobile.md` §1.4 and `../ada/wcag22-new.md`.

**Deep dive → `../mobile.md`** for responsive strategy, breakpoints, touch UX, and viewport handling.

---

## 2.3 Technical hygiene

- [ ] **No broken internal links (4xx)** 🔒 *crawl required* — Screaming Frog → Response Codes → Client Error (4xx) → Inlinks tab shows the sources. Pass: zero internal 4xx. External 4xx are lower priority but worth fixing.
- [ ] **No internal links pointing through redirect chains** — Screaming Frog → Reports → Redirects → Redirect Chains. Pass: internal links point at the final URL. Every hop is wasted crawl and diluted signal, and it's free to fix.
- [ ] **Redirect chains ≤ 1 hop; no redirect loops** — `curl -sIL URL | grep -iE '^HTTP|^location'`. Pass: at most one `301` before the `200`.
- [ ] **Correct status codes throughout** — Pass: `200` for live pages; `301` for permanent moves; `302` only for genuinely temporary ones; `404` for missing; `410` for intentionally and permanently gone. A `302` used for a permanent move delays consolidation.
- [ ] **Custom 404 page returns a real 404 and helps recovery** — `curl -sI https://DOMAIN/xyz-not-real | head -1` returns `404`, and the page offers search plus links to main sections.
- [ ] **Images compressed and served in modern formats** — check `content-type` on the LCP image: `curl -sI IMAGE_URL | grep -i content-type`. Pass: `image/webp` or `image/avif` where support allows. Details → `../performance.md`.
- [ ] **Fonts subset, self-hosted or preconnected, limited in number** — count distinct font files in DevTools → Network → Font. Pass: a small, justified set. Details → `../performance.md`.
- [ ] **Third-party script inventory taken; every tag manager entry justified** 🔒 — DevTools → Network → JS, sorted by third-party origin; plus a GTM container audit. Pass: a written list where each entry has a named owner and a reason. Fail signature: tags nobody can explain, usually from campaigns that ended.
- [ ] **Favicon present, ≥ 48×48, square, crawlable** — `curl -sI https://DOMAIN/favicon.ico | head -1` plus the `<link rel="icon">` target. Pass: `200`, square, at least 48×48, not robots-blocked. Google uses it in results — it is free SERP real estate.
- [ ] **Accessibility basics in place** — semantic HTML, correct heading order, alt text, ARIA where needed. Pass: an axe/Lighthouse a11y pass with no critical violations. Full treatment → `../ada/`. *Google's own 2026 guidance notes semantic HTML helps both screen readers and AI browser agents parse pages* — this work counts twice, see `L9-ai-search.md` §agentic access.

---

## Questions to ask

1. What is my 75th-percentile mobile INP right now — the real number, from field data? 🔒
2. Which single third-party script costs me the most main-thread time, and what would break if I removed it?
3. When did I last crawl my own site with Screaming Frog or Sitebulb?
4. Are my mobile and desktop pages truly identical in content and links?
5. What's my slowest *template*, not my slowest page? Templates scale; individual pages don't.

---

Next: `L3-architecture.md`. Deep dives: `../performance.md`, `../mobile.md`, `../ada/`.
