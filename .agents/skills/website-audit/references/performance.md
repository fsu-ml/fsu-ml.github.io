# Performance — Core Web Vitals, loading, and main-thread discipline

Framework-agnostic. Owns metric definitions, per-metric diagnosis, delivery, JavaScript budgets, and verification method.
Load: any performance complaint, any CWV finding, or `performance: full`.
Companions: `mobile.md` (responsive/touch), `animation-and-motion.md` (compositing, `prefers-reduced-motion`), `dynamic-loading.md` (route-level and interaction-level loading strategy), `seo/L2-technical-performance.md` (what search engines do with these signals — do not duplicate fixes there), `code-quality.md` (engineering hygiene), `security-and-hygiene.md` (headers).
Scripts: `../scripts/audit_performance.py`.

**Reliability key.** `[P]` = primary source (Chrome/WebKit/MDN/W3C/HTTP Archive). `[S]` = secondary/aggregator, directionally right. `[?]` = contested or unverified — **do not assert as fact in an audit**. Every claim below carries the tag it was researched under. Do not upgrade a tag when writing a report.

---

## 0. Three framings that change how you audit

1. **The thresholds have not moved.** LCP 2.5 s / INP 200 ms / CLS 0.1 at p75 are unchanged since INP replaced FID in **March 2024**. Any 2026 post claiming "Google tightened the thresholds" is wrong. `[P]`
2. **The measurable surface expanded.** Chrome shipped the **Soft Navigations API** (`soft-navigation` + `interaction-contentful-paint` entries) in **Chrome 151, ~July 2026**. SPAs are, for the first time, natively measurable for LCP/CLS/INP/FCP across client-side route changes. `[P]`
3. **The bottleneck moved from bytes to the main thread.** Field INP *improved* (mobile 74% → 77% good) while **lab TBT got dramatically worse** — median mobile TBT rose **58% YoY, 1,209 ms → 1,916 ms**. `[P]` This divergence is the single most important number in the 2025 Web Almanac and it reframes the whole audit: **weight-loss advice ("compress the images") is no longer where the wins are. Execution-time advice is.** An audit that reports only transfer sizes is auditing the 2018 web.

**Two methodological errors to refuse.**
- **Homepage-only auditing.** Secondary pages beat home pages on overall CWV (61%/56% vs 47%/45%) but **lose badly on INP** (mobile 69% vs 80%). Filters, carousels, form validation and accumulated third-party JS live on inner pages. Always audit ≥ 1 inner/secondary template. `[P]`
- **Pre-October-2025 checklists.** Lighthouse 13 removed or renamed most performance audit IDs. Hard-code the `*-insight` IDs (§5.1).

---

## 1. Core Web Vitals 2026

### 1.1 Thresholds (unchanged, authoritative) `[P]`

| Metric | Good | Needs improvement | Poor | Type |
|---|---|---|---|---|
| **LCP** (loading) | ≤ 2.5 s | 2.5 – 4.0 s | > 4.0 s | Field + lab |
| **INP** (responsiveness) | ≤ 200 ms | 200 – 500 ms | > 500 ms | **Field only** |
| **CLS** (visual stability) | ≤ 0.1 | 0.1 – 0.25 | > 0.25 | Field + lab |
| FCP (diagnostic) | < 1.8 s | 1.8 – 3.0 s | > 3.0 s | Field + lab |
| TTFB (diagnostic) | < 0.8 s | 0.8 – 1.8 s | > 1.8 s | Field + lab |
| TBT (lab proxy for INP) | < 200 ms | — | — | **Lab only** |

A site passes CWV only if it passes **all three**. **TTFB and FCP are not Core Web Vitals and are not ranking inputs** — they are diagnostics. Do not report them as failures; report them as causes.

### 1.2 The p75 field model — what "passing" actually means `[P]`

- A page/origin passes only if the **75th percentile** of real page loads meets the target, **segmented separately for mobile and desktop**. One reading is not a result; two are.
- CrUX aggregates a **28-day rolling window**.
- The **CrUX History API** exposes **40 weekly collection periods**, each itself a 28-day window — so consecutive points **overlap by 3 weeks**. Never treat week-over-week deltas as independent observations, and never draw a trend line through them as if they were.
- **BigQuery CrUX is monthly**, published the **second Tuesday after the collection period**.
- Consequence for reporting: after a deploy, verify in lab immediately, and **do not expect field movement for up to 28 days**. Say this in writing before the fix ships or the fix gets judged a failure.

### 1.3 Metric lifecycle `[P]`

Chrome runs metrics through **Experimental → Pending (min. 6 months) → Stable**. Stable metrics change at most **once per year**, with notice. LCP, CLS, INP are all Stable. **FID was deprecated 9 Sep 2024.**

### 1.4 Soft Navigations API — SPA measurement `[P]`

| Item | Detail |
|---|---|
| Definition of a soft nav | user interaction → visible content paint → URL update |
| Detection | **browser-side heuristic**, not framework-emitted — works on existing SPAs with **zero code changes** |
| Final origin trial | **Chrome 147 → 149**, announced **20 Apr 2026** `[P]` |
| Shipped unflagged | **Chrome 151**, stable **~28 July 2026** `[S]` for the exact date |
| New entries | `SoftNavigationEntry` (carries `interactionId`, `navigationId`, `name` = new URL, paint timings, `largestInteractionContentfulPaint`) and `InteractionContentfulPaint` |
| Key change vs prior trial | `interaction-contentful-paint` now fires for **all** interactions, not only those causing a soft nav — usable as a general "how long until the UI updated after this click" metric |
| Attribution fix | `navigationId` added to `largest-contentful-paint`, `interaction-contentful-paint`, `event`, and `layout-shift` entries, so late-observed entries attach to the right navigation |
| `replaceState` | now counts as a soft navigation (feedback-driven change) |
| DevTools | Performance panel shows soft-nav markers from **Chrome 145**, even without the feature enabled |
| **Open question** | **How CrUX will report soft navigations is explicitly undecided.** Do not assume soft-nav data flows into ranking yet. |

An experimental **soft-navigation build of `web-vitals`** exists on GitHub/npm for SPA measurement. `[P]`

**Audit consequence:** "SPAs can't be measured for CWV" is obsolete as of Chrome 151. If the site is an SPA and has no soft-nav measurement, that is now a finding, not an excuse.

### 1.5 Claims to refuse `[?]`

| Claim in circulation | Status |
|---|---|
| "Google tightened INP measurement methodology in 2026, causing an 18–25% p75 regression" | **Not supported by the Chromium INP changelog.** Actual 2026 INP changes: Chrome 147 (early interaction-ID assignment + context-menu fallback), Chrome 148 (report only web-exposed event targets), Chrome 150 (nested `<label>` clicks assigned `interactionId = 0`). Bug fixes, not a tightening. Chrome 144 launched `performance.interactionCount` to stable. `[P]` Treat the narrative as content-farm noise. |
| **"Engagement Reliability (ER)" is a new Core Web Vital** | **No primary Chrome/W3C source exists.** Surfaces only in low-quality SEO content. Do not put it in a report. `[?]` |
| "LoAF is a new Core Web Vital" | **Long Animation Frames API** (`PerformanceObserver` type `long-animation-frame`, **Chrome 123+**) is real and is the correct INP attribution tool, but is **not** a CWV candidate. `[P]` |

### 1.6 Where the web actually stands — the comparison baseline

HTTP Archive Web Almanac 2025, July 2025 crawl. `[P]` Use these to tell a client whether they are bad, average, or good — an unanchored "your LCP is 3.1 s" means nothing to them.

Overall CWV pass: **48% mobile / 56% desktop** (2024: 44/55; 2023: 36/48). Desktop improvement has flattened.

| Metric | Desktop good | Mobile good |
|---|---|---|
| LCP | 74% | 62% (13% poor) |
| INP | 97% | 77% (3% poor) |
| CLS | 72% | 81% |
| FCP | 70% | 55% |
| TTFB | 55% | 44% |

- Mobile CWV by rank is **U-shaped**: top 1,000 (51%) and the long tail (49%) beat the middle (37–42%).
- Top-1,000 sites are the **worst on mobile INP** (63% good vs 77% average), though they improved 10 points YoY.
- **13% desktop / 15% mobile** of pages pass the Lighthouse render-blocking audit. **~85% of the web is render-blocked** — this is the most common finding you will make, and it is not a differentiator on its own.
- TBT percentiles (lab, mobile): **p10 127 ms, p25 679 ms, p50 1,916 ms, p75 4,193 ms, p90 7,555 ms.** Desktop p50 **92 ms**.

---

## 2. Per-metric diagnosis

Structure every metric finding as: **observed value → which sub-part dominates → which cause matches this site → the fix → how you verified the cause, not just the symptom.**

### 2.1 LCP diagnosis

**Decomposition (do this first, always):** LCP = **TTFB → resource load delay → resource load duration → element render delay**.

Chrome field data: most origins with poor LCP spend **< 10% of p75 LCP actually downloading** the LCP image. The dominant cost is **resource load delay of ~1,290 ms at p75**. `[P]`
**Therefore: image compression is usually the wrong lever.** If your report's LCP section leads with "compress the hero image," you skipped the decomposition.

Field context: the LCP element is an **image on 85.3% desktop / 76.0% mobile** of pages. LCP image formats: **JPG 57%, PNG 26%, WebP 11%, AVIF 0.7%** (JPG −4 pp YoY, WebP +4 pp). `[P]`

| Sub-part dominating | Common cause | How to confirm it is *this* site's cause | Fix |
|---|---|---|---|
| **TTFB** (> 800 ms) | No CDN in front of HTML; slow origin render; uncached DB query | `curl -w '%{time_starttransfer}\n' -o /dev/null -s URL`; check `cf-cache-status` / `x-cache` / `age` response headers; WebPageTest TTFB by geography | Edge/CDN HTML caching (**only 33% of HTML document requests are CDN-served** `[P]`); streaming SSR; `stale-while-revalidate` |
| **Resource load delay** (the usual winner) | LCP URL not in the initial HTML — injected by JS, `data-src`, or referenced from an inline/external style | Fetch HTML with JS disabled and grep for the LCP URL: `curl -s URL \| grep -o 'hero[^"]*'`. Absent = confirmed. DevTools → Performance → LCP phases | Put the URL in the initial HTML as `<img src>`/`srcset`; prefer SSR over CSR; **images referenced from inline styles are not preload-scanner-discoverable** |
| **Resource load delay** | LCP image is lazy-loaded | `curl -s URL \| grep -B2 -A2 'loading="lazy"'` and compare with the LCP element ID from Lighthouse `lcp-phases-insight` | Remove `loading="lazy"`. **~16–17% of pages lazy-load their own LCP image** (native `loading="lazy"` 10.4% mobile / 11.5% desktop; JS `data-src` 5.9% mobile), flat since 2024. `[P]` **This is the highest-yield single-line fix on the web.** |
| **Resource load delay** | Wrong priority | Check for `fetchpriority="low"` on the LCP image (**0.3% of pages do this** `[P]`), or its absence (`fetchpriority="high"` present on only **16.3% desktop / 17.3% mobile**) | `fetchpriority="high"` on the LCP `<img>`. **Preload alone does not guarantee high priority** — pair them |
| **Resource load delay** | Cross-host LCP image, no `preconnect` → connection-setup tax | **16–18% of pages** have a cross-host LCP image `[P]`. Compare the image origin to the document origin; check for a matching `<link rel=preconnect>` | `preconnect` that one origin |
| **Resource load duration** (rare) | Genuinely oversized asset | Network panel: transfer size vs rendered CSS pixel size. Flag only if > 2× rendered size | Responsive `srcset`+`sizes`, modern format (§3.6) |
| **Element render delay** | Render-blocking CSS/JS ahead of the paint; font blocking text LCP | Lighthouse `render-blocking-insight`; Performance panel main-thread flame chart before the LCP marker | §3.1–3.2; `font-display` (§3.5) |

**Fix order (highest expected value first):**
1. LCP resource URL in the **initial HTML response**, preload-scanner-discoverable. Never `data-src`.
2. `fetchpriority="high"` on the LCP `<img>` (or on its preload, paired).
3. Remove `loading="lazy"` from the LCP image; remove `decoding="async"` misuse on it.
4. `preconnect` the LCP image's origin if cross-host.
5. Cut render-blocking CSS/JS ahead of it.
6. **Aim for instant navigation:** bfcache eligibility + Speculation Rules prerender. **A prerendered page has near-zero LCP.**
7. CDN + edge caching for TTFB.

### 2.2 INP diagnosis

**Definition:** worst (approximately; 98th-percentile-trimmed for high-interaction pages) interaction latency across the **whole page lifetime**, measured as **input delay + processing time + presentation delay**. Unlike FID it **cannot be gamed by a cheap first click**.

**Attribution first.** Do not guess. Capture **LoAF** (`long-animation-frame`, Chrome 123+) for the worst interaction — it names the **script, source location, and the blocking / style-and-layout split** for the offending frame. This is the 2026 gold standard for INP debugging (`[S]` on "gold standard", `[P]` on the API). Then read the table.

| Sub-part dominating | Common cause | How to confirm it is *this* site's cause | Fix |
|---|---|---|---|
| **Input delay** | The main thread was already busy — usually third-party or hydration work | LoAF `blockingDuration`; Performance panel long tasks overlapping the input event; block third-party origins in DevTools and re-measure | §4.4 third parties; defer/split hydration; ship less JS |
| **Processing time** | One handler doing > 50 ms of synchronous work | LoAF `scripts[].sourceLocation` names the file+line | **Yield often.** `scheduler.yield()` — **Chrome/Edge 129+, Firefox 142+, Safari not supported** — breaks up work while preserving queue position. Fallback: `await new Promise(r => setTimeout(r, 0))`. Reported field INP improvements of **40–60%** on sites that split their main long tasks `[S]` |
| **Processing time** | Stale advice in the codebase | grep the bundle for `isInputPending` | **Chrome reversed its `isInputPending()` recommendation in Oct 2024.** `[P]` Any advice or code using it is stale — replace with `scheduler.yield()` |
| **Presentation delay** | Large rendering update; layout thrashing; huge DOM | Performance panel "forced reflow" warnings; Lighthouse `dom-size-insight` | Batch DOM reads/writes; keep the DOM small; `content-visibility` / CSS containment to skip offscreen layout (§3.9) |
| **All three** | Framework concurrency features present but unused | Check whether the app actually calls the concurrent scheduling APIs, not just which version it is on | Framework levers require **deliberate adoption** — upgrading a major version without adopting the concurrent/transition/signal APIs does **nothing** for INP `[S]` |
| **All three** | JS reimplementing platform features | grep for carousel/tooltip/dialog/date-picker libraries | Drop JS implementations of things now in Baseline (dialogs, popovers, date inputs, auto-sizing fields) — see `code-quality.md` §2 |

Additional levers: **DevTools Coverage → dead code** (§4.5); route- and interaction-level code splitting (`dynamic-loading.md`); audit third-party scripts for synchronous event listeners.

### 2.3 CLS diagnosis

**Definition:** sum of the **largest burst** of unexpected layout shifts (session window, max 5 s gap / 5 s cap). Shifts within **500 ms of a user interaction** are excluded — **but `hover` does not count as an interaction.**

Field reality: **62% of mobile / 65% of desktop pages have at least one unsized image** (improved from 66%/69%). Median unsized images per page: **1–2**; p90: **22–25**. Unsized image height p90: **413 px desktop / 300 px mobile** — **height is far more damaging than width.** `[P]`

| Cause | How to confirm it is *this* site's cause | Fix |
|---|---|---|
| Unsized media | `curl -s URL \| grep -oE '<img [^>]*>' \| grep -v 'width='` → any output is a hit. Also check `<video>`, `<iframe>` | `width`/`height` attributes on every `<img>`, or CSS `aspect-ratio` (Baseline **Widely available**; Chrome/Edge 88, Firefox 89, Safari 15) |
| Dynamically injected content (ads, embeds, banners) with no reserved box | Record a Performance trace; each `layout-shift` entry names its sources | `min-height` on the container — beats the default 0 px |
| Cookie/consent banner pushing content | Measure CLS **with the banner enabled**; most audits accidentally measure a dismissed state | Fixed positioning + reserved space; self-host the CMP (§4.4) |
| Font swap shift | Compare rendered text block height before/after webfont load in a filmstrip | Metric overrides or `font-display: optional` — §3.5. Correctly tuned overrides produce **zero** CLS |
| Layout-inducing animation | Lighthouse `non-composited-animations`; grep CSS for `transition:` / `@keyframes` touching `width`, `height`, `top`, `left`, `margin`, `padding`, `border-width`, `font-size` | `transform`/`opacity` only. Pages that animate any layout-affecting property are **15% less likely** to have good CLS; animating `margin`/`border-width` **roughly doubles** the poor-CLS rate. **Absolutely-positioned elements animating `top`/`left` still shift.** **40% mobile / 44% desktop of pages have non-composited animations** (p90: 11–13 per page). `[P]` See `animation-and-motion.md` |
| `content-visibility` without a size hint | grep CSS for `content-visibility` and check each has `contain-intrinsic-size` | §3.9 — without it you *create* CLS and scrollbar jitter |
| **No bfcache** (back-navigation shift) | DevTools → Application → Back/forward cache → **Test**; field: `NotRestoredReasons` API | §3.4 |

**bfcache is the highest-leverage CLS lever nobody checks.** Its 2022 introduction was the **single biggest CLS improvement Chrome has measured.** `[P]` Blockers: `unload` listeners (**28% desktop / 20% mobile of top-1,000 sites still use them; 11%/10% overall**) and `Cache-Control: no-store` (**23% of sites, up from 21%**). Use `pagehide`/`visibilitychange` instead of `unload`. Chrome may allow bfcache for some `no-store` pages; **Firefox and Safari generally still treat `no-store` as a hard blocker.**

### 2.4 Lab vs field — and why lab misleads

| Source | What it is | What it **cannot** tell you |
|---|---|---|
| **Lighthouse** | Single synthetic run, simulated throttling, no user | **Cannot measure INP** (no input) — uses **TBT** as proxy. Cannot measure real device/network/geography mix. Run-to-run score variance is large |
| **CrUX / PSI field data** | Real Chrome users, 28-day p75, **Chrome only**, opted-in users, origin- or URL-level | No per-pageview telemetry; no attribution; lags reality by up to 28 days; sparse URLs get no data (silent origin fallback); **excludes Safari and Firefox entirely** |
| **RUM (`web-vitals`)** | Your own users, per-pageview, attributable, segmentable | Requires instrumentation; will **not** exactly match CrUX (different population, sampling, and bfcache/prerender handling) — do not present a mismatch as a bug |
| **WebPageTest** | Real devices at real network locations, filmstrip + waterfall | Still synthetic; slow; **not a substitute for p75** |

**Why a 95 Lighthouse score means little:** Lighthouse simulates a fixed device/network profile and never interacts with the page, so a 95 on a dev laptop is routine on a site that fails CWV in the field. The Almanac 2025 divergence — **field INP improving while lab TBT got 58% worse** — is direct evidence that lab and field can move in **opposite directions**. `[P]` Never report a Lighthouse score as the CWV result.

**Canonical RUM collection pattern:**
```js
import {onCLS, onINP, onLCP, onFCP, onTTFB} from 'web-vitals';
// use 'web-vitals/attribution' to also get LoAF/element attribution
function send(metric) {
  const body = JSON.stringify(metric);
  (navigator.sendBeacon && navigator.sendBeacon('/analytics', body)) ||
    fetch('/analytics', {body, method: 'POST', keepalive: true});
}
onCLS(send); onINP(send); onLCP(send); onFCP(send); onTTFB(send);
```
Use the **attribution build** (`web-vitals/attribution`) — plain values without attribution are near-useless for fixing anything. Segment by **device class, connection, country, and page template**: a handful of high-traffic templates commonly drags an entire origin's p75 under, and origin-level data hides it. `[P]`

---

## 3. Loading and delivery

### 3.1 Critical rendering path `[P]`

- **CSS is render-blocking by default.** `<script>` in `<head>` without `defer`/`async`/`type=module` is **parser-blocking**.
- `defer` — download in parallel, execute after parsing, **in document order**, before `DOMContentLoaded`. **Correct default for first-party app code.**
- `async` — download in parallel, execute **as soon as ready, out of order**, can interrupt parsing. **Correct only for independent third-party scripts** (analytics).
- `type="module"` — deferred by default. `type="module" async` executes ASAP.
- **Inline critical CSS** for above-the-fold; load the rest with `media="print" onload="this.media='all'"` or `rel=preload as=style` + swap. Keep inline CSS small — **it isn't cacheable**.
- Only **13–15% of pages** pass Lighthouse's render-blocking audit.

**Check:** `curl -s URL | grep -oE '<script[^>]*>' | grep -v -E 'defer|async|type="module"|type=.module.'` — any parser-blocking `<script>` in `<head>` is a finding. Lighthouse `render-blocking-insight` for the ranked list.

### 3.2 Resource hints — and the over-preloading trap

| Hint | Use for | Priority |
|---|---|---|
| `preconnect` | Cross-origin you will **definitely** hit early (LCP image CDN, font host). **Cap at 2–4** | — |
| `dns-prefetch` | Cheap fallback for lower-confidence cross-origins | — |
| `preload` | The **2–4** truly critical resources the preload scanner cannot find (font files, CSS-referenced hero image, JS-injected LCP image) | Inherits from `as` + `fetchpriority` |
| `modulepreload` | ES module graph warm-up — preloads **and** parses/compiles, unlike `preload as=script` | High |
| `prefetch` | The **next** navigation's resources | **Always Lowest** |

**The anti-pattern to flag: "preload everything."** Every preload competes for bandwidth with resources the browser already discovered and correctly prioritised. Pages with **10+ preloads consistently load slower** than pages with a few well-targeted ones. Every unused `preconnect` burns a TCP+TLS handshake and holds a connection **~10 s**. `[S]` — consistent across multiple 2026 sources and matches Chrome's own reasoning.

**Corroborating primary signal:** Lighthouse 13 **removed both `uses-rel-preload` and `preload-fonts`**, explicitly "due to risks of over recommending." `[P]` **Any audit that blanket-recommends preload is repeating advice Google itself retired.**

Adoption reality: font `preconnect` 22%, font `preload` 15–16%, font `dns-prefetch` 24%; **LCP-image `preload` only 2.1–2.2%.** The problem is **targeting, not volume**. `[P]`

**Check:** `curl -s URL | grep -oE '<link[^>]*rel="?(pre(load|connect|fetch)|modulepreload)"?[^>]*>' | wc -l`, then per-hint counts. DevTools console logs a warning for any preload unused within 3 s — that warning *is* the evidence.

**`fetchpriority`** — `high` | `low` | `auto` on `<img>`, `<link>`, `<script>`, and `fetch()`. **Chrome/Edge 102+, Firefox 132+, Safari 17.2+** — safe to use, degrades harmlessly. Primary uses: `high` on the LCP image; `low` on below-fold images and non-critical third-party fetches.

**`103 Early Hints`** — send `preconnect`/`preload` before the main response. Adoption **~4%, up from ~1% in 2023**, peaking at ~6%. `preconnect` in Early Hints works everywhere; **`preload` works everywhere except Safari.** Supported by Fastly (2020), Cloudflare (2021), Akamai (2023), nginx, H2O, Node, Apache w/ `mod_http2`. **Use only over HTTP/2+.** `[P]`

### 3.3 Speculation Rules API

**Support (Aug 2026):** Chrome/Edge **109+** (prerender 105+, prefetch 110+; `eagerness` + `where` require **121/122**). **Firefox: no support. Safari 26.2: present but disabled by default.** `[P]`/`[S]`
A `<script type="speculationrules">` block is **silently ignored** elsewhere — safe progressive enhancement, **never a load-bearing dependency**.

```html
<script type="speculationrules">
{ "prerender": [{
    "where": { "and": [
      { "href_matches": "/*" },
      { "not": { "href_matches": "/logout/*" } },
      { "not": { "selector_matches": ".no-prerender" } }
    ]},
    "eagerness": "moderate"
}]}
</script>
```

**Eagerness:**

| Value | Trigger |
|---|---|
| `immediate` | as soon as rules are parsed |
| `eager` | currently identical to `immediate`; Chrome intends to move it between `immediate` and `moderate` |
| `moderate` | hover ≥ **200 ms**, or `pointerdown` if sooner; on mobile (no hover) `pointerdown`. **Best default** |
| `conservative` | pointer/touch down only |

Defaults: **list rules → `immediate`; document rules → `conservative`.**

**Chrome concurrency limits:** `immediate`/`eager` → **50 prefetch / 10 prerender**. `moderate`/`conservative` → **2 / 2, FIFO** (a new speculation evicts the oldest). Evicted speculations are cheap to redo because subresources stay in the HTTP cache. `[P]`

**Chrome suppresses speculation entirely** under Save-Data, Energy Saver, memory pressure, when "Preload pages" is off (uBlock Origin turns it off), and in background tabs. Never build a UX that depends on it.

**Pitfalls to audit for:**
- Prerendering **logout, delete, add-to-cart, or any GET that mutates state**. Exclude with `not: href_matches`.
- **Analytics double-counting** — prerendered pages run JS. Gate on `document.prerendering` and the `prerenderingchange` event.
- Server load and user bandwidth from over-eager rules.
- HTTP-header rollout: `Speculation-Rules: "/speculationrules.json"` (**quotes are mandatory** — structured header), `Content-Type: application/speculationrules+json`, CORS if cross-origin, `"relative_to": "document"` if using relative URLs.
- `No-Vary-Search` / `expects_no_vary_search` treats `?utm_*` and client-only params as cache-equivalent. **Chromium-only, prefetch only.**

**Adoption:** ~15% of top-1M; **24% desktop / 25% mobile in the long tail**, only **3–5% in the top 1,000** — driven by a CMS plugin, not hand-rolled implementations. `[P]`
**Debugger:** `specrules.com`; DevTools → Application → Speculative loads.

### 3.4 bfcache

Free, browser-level instant back/forward. **Blockers to audit:** `unload` listeners, `Cache-Control: no-store`, some `Cache-Control: no-cache` patterns, open IndexedDB transactions, in-flight fetch/XHR, WebSocket/WebRTC in some cases.
**Verify:** DevTools → **Application → Back/forward cache → Test**; in the field via the **`NotRestoredReasons` API**.
**Check for `unload`:** `curl -s URL | grep -c "onunload\|addEventListener('unload'\|addEventListener(\"unload\""` plus a grep of the built JS bundles.

### 3.5 Fonts

**87% of mobile pages use at least one web font.** Median font payload: **139 KB home / 122 KB inner**. `[P]`

Current best practice `[S]`, consistent across sources:

| Rule | Detail | Check |
|---|---|---|
| **WOFF2 only** | Nothing else is worth shipping in 2026 | DevTools → Network → Font; any `.ttf`/`.otf`/`.woff` is a finding |
| **Self-host** | A third-party font origin costs a connection, gives up cache control, and carries **GDPR exposure (German court rulings)**. The "shared cache across sites" argument **died with cache partitioning in 2020** | Compare font request origins to the document origin |
| **Variable fonts** where you need **3+ weights** | One file replaces the family and unlocks intermediate weights. For **one** weight, a static subset is smaller | Count distinct `@font-face` files vs weights actually used in CSS |
| **Subset** to the character set actually used | `unicode-range` split, or glyphhanger/fonttools. Latin subset of a typical face: **~15–25 KB** | Any single font file > ~40 KB warrants a subsetting question |
| **`font-display`** | **`swap`** for body text (~100 ms block, infinite swap). **`optional`** is the CLS-safest (≤ 100 ms block, **no** swap period — visitors who miss the window get the fallback for that navigation; the font is cached for next time). `fallback` is the compromise. **Never `auto`/`block`** for body copy | grep CSS for `font-display` — absent means `auto` |
| **Metric overrides** — highest-value, least-known technique | Eliminates the swap shift entirely | see below |
| **Preload exactly the above-the-fold font files** — usually one, at most two | `<link rel=preload as=font type="font/woff2" crossorigin>` — **the `crossorigin` attribute is mandatory even same-origin**, or you double-fetch. Preloading all six weights is a classic own-goal. Note again that Lighthouse **removed** its font-preload audit for over-recommendation risk | `curl -s URL \| grep 'rel="preload"' \| grep 'as="font"'` — count, and check every one has `crossorigin` |

```css
@font-face {
  font-family: "Inter Fallback";
  src: local("Arial");
  size-adjust: 107%;
  ascent-override: 90%;
  descent-override: 22%;
  line-gap-override: 0%;
}
body { font-family: "Inter", "Inter Fallback", sans-serif; }
```
Tools: `fontaine`, framework font primitives that do this automatically, Capsize. **Correctly tuned, `swap` produces zero CLS.**

Fonts referenced only from CSS are **not** preload-scanner-discoverable — that is the **legitimate** case for preload.

### 3.6 Images

| Rule | Detail |
|---|---|
| **Formats** | WebP ≈ **97%** global support; AVIF ≈ **94%** and rising (Chrome, Firefox, **Safari 16.4+**, Edge). AVIF is **~20–30% smaller than WebP** but far slower to encode. Practical 2026 rule: **AVIF → WebP → JPEG via `<picture>`** for hero/marketing imagery; **WebP-only is fine for everything else**. JPEG XL remains not viable cross-browser. `[S]` |
| **Responsive** | `srcset` + `sizes` for resolution switching; `<picture>`/`<source>` for format and art-direction switching. **Wrong or missing `sizes` is a very common real-world bug — it silently causes 2–3× overfetch** |
| **Dimensions** | Explicit `width`/`height` on every `<img>` (intrinsic ratio; CSS can still override with `width:100%; height:auto`), or `aspect-ratio`. See §2.3 for the 62% failure rate |
| **Lazy loading** | `loading="lazy"` on **below-fold images only**. **Never on the LCP image or anything in the initial viewport** — ~16–17% of pages get this wrong. Lighthouse 13 **removed `offscreen-images`** because "offscreen images are already deprioritized by the browser": lazy loading saves bandwidth but **rarely moves lab metrics**. Don't oversell it `[P]` |
| **Decoding** | `decoding="async"` for non-critical images; leave the LCP image at `auto`/`sync`. A **minor lever, not a headline fix** |
| **CDN transforms** | Content-negotiated format via `Accept`, DPR-aware, on-the-fly resize. Standard 2026 baseline; **hand-exported fixed-size JPEGs are a smell** |
| **Never JS lazy-load (`data-src`)** | Hides the URL from the preload scanner. **5.9% of mobile pages still do this for their LCP image** |

**Checks:** modern format actually served — `curl -sI -H 'Accept: image/avif,image/webp,*/*' IMAGE_URL | grep -i content-type`. Overfetch — DevTools → Network → Img, compare intrinsic size to rendered size; flag anything **> 2× its rendered CSS pixel size**.

### 3.7 Video

- Always set a **`poster`**, and give the poster the same CLS treatment as an image (explicit dimensions or `aspect-ratio`).
- `preload="none"` for below-fold or click-to-play; `preload="metadata"` when you need duration; **`preload="auto"` almost never**.
- Autoplay requires **`muted` and `playsinline`** on iOS — `autoplay muted loop playsinline` together, or it silently fails. Respect `prefers-reduced-motion` for decorative background video (`animation-and-motion.md`).
- **Facade pattern** for third-party video embeds — a raw embed can be **500 KB – 1 MB of third-party JS**. Caveat: Lighthouse 13 **removed the `third-party-facades` audit**, but the technique remains sound; the removal was about tooling scope, not efficacy. `[P]`
- Serve **MP4/H.264 as baseline, AV1/WebM as the modern `<source>`**.

### 3.8 Compression, protocol, caching

**Compression** (Web Almanac 2025 CDN chapter): CDNs served **46% Brotli / 42% gzip / 12% Zstandard**. Zstd grew **3% → 12% YoY**; Cloudflare ~15%, Google CDN ~10%. `[P]`

| Codec | Support | Use |
|---|---|---|
| **Brotli** | universal (~96%+) | Best ratio for static text at **level 11**. Precompress static assets at build time |
| **Zstd** (`Content-Encoding: zstd`) | Chrome 123+, Edge 123+, Firefox 126+, **Safari 26+ macOS / 26.3+ iOS** | Much faster compress/decompress; wins for **dynamic** responses where Brotli 11 is too slow. **Not yet universal — always keep Brotli/gzip in the negotiation chain** `[S]` |
| Shared/Dictionary Compression (Compression Dictionary Transport) | emerging | 20–40% claimed gains on repeated content — **still early** `[S]` |

**Practical rule: static → Brotli 11 precompressed; dynamic → zstd or Brotli 4–5; never gzip-only.**
**Check:** `curl -sI -H 'Accept-Encoding: br,zstd,gzip' URL | grep -i content-encoding`.

**Protocol:** HTTP/3 sits around **21–39% depending on measurement** (W3Techs 39.2% of sites; Cloudflare ~35% of its edge traffic; TechnologyChecker 21% of page loads); **> 50% among top-1,000 sites**. Default-on at Cloudflare, Fastly, Akamai, Caddy, recent nginx. Benefits concentrate on **lossy mobile networks** (no head-of-line blocking; 0-RTT resumption): **~20–60% faster loads and 100–300 ms LCP improvement** in those conditions. `[S]`
**HTTP/1.1 in 2026 is a defect** — it forces domain sharding and concat hacks that are now anti-patterns. Lighthouse's `modern-http-insight` replaced the old `uses-http2` audit.
**Check:** `curl -sI --http3 URL | head -1`, or `performance.getEntriesByType('navigation')[0].nextHopProtocol` in the console.

**Caching:**

| Resource | Header |
|---|---|
| Hashed/fingerprinted assets | `Cache-Control: public, max-age=31536000, immutable` |
| HTML | `Cache-Control: no-cache` (revalidate) or a short `max-age` + `stale-while-revalidate`. **Avoid `no-store` — it kills bfcache in Firefox/Safari** |
| API / personalised | `private, no-cache` + `ETag` |
| Edge | `stale-while-revalidate` and `stale-if-error` |

**CDN HTML caching is the biggest untapped win: only 33% of HTML documents are CDN-served.** `[P]`
**Check:** `curl -sI URL | grep -iE 'cache-control|age|cf-cache-status|x-cache|etag'`.

### 3.9 Containment and `content-visibility`

```css
.long-section {
  content-visibility: auto;
  contain-intrinsic-size: auto 600px; /* placeholder height — required */
}
```
- Skips layout/paint for offscreen subtrees. Best gains on **long pages with many heavy sections**; measurable INP/rendering wins.
- **Without `contain-intrinsic-size` you will create CLS and scrollbar jitter.** Prefer the `auto <length>` form so the browser remembers the real size after first render.
- Known caveat: **Safari's Cmd+F does not reliably find text inside `content-visibility: auto` subtrees** `[S]`. (In-page find is otherwise handled by `hidden=until-found` / `beforematch` in Chromium.)
- `contain: layout paint` / `contain: strict` for isolated widgets.
- Support: **Chrome 85, Firefox 125, Safari 18**.

---

## 4. JavaScript discipline

### 4.1 What the web actually ships vs what to target

Almanac 2025, **median home page**: JS **697 KB**, images **1,059 KB**, fonts **139 KB**, CSS **82 KB**, HTML **22 KB**. Median total page: **2.86 MB desktop / 2.56 MB mobile**. `[P]`

**Defensible 2026 targets** (compressed, first-load, per page):

| Site type | JS budget (compressed) | Notes |
|---|---|---|
| Content / marketing / docs | **≤ 50 KB** | Should be near-zero; islands only |
| Commerce / typical product | **≤ 150 KB** | Achievable with server components / islands + route splitting |
| App-like SPA | **≤ 250 KB** initial route | Everything else lazy |
| **Hard ceiling anyone should defend** | **300 KB** | Above this, justify explicitly |

Widely-cited general rule: **< 200 KB compressed initial JS** `[S]`.

**Pair every byte budget with a time budget** — this is the §0 finding in operational form:
- **TBT < 200 ms** on a mid-tier Android (Moto G-class / Snapdragon 695-class).
- **Main-thread work < 2 s** in Lighthouse mobile.

**Bytes lie.** A 500 KB bundle that parses in 200 ms on a MacBook takes **2–3 s on a mid-range phone**. Enforce in CI: `bundlesize`, `size-limit`, `lighthouse-ci` assertions, Lighthouse `budget.json`.

### 4.2 Techniques

| Technique | Detail | Check |
|---|---|---|
| **Code splitting** | By route, then by interaction (`import()` on click / intersection / idle). See `dynamic-loading.md` | Coverage + Network waterfall on route change |
| **Tree shaking** | Requires ESM + `"sideEffects": false` in `package.json` + **no barrel-file re-export sprawl** — barrel files are a top cause of tree-shaking failure in 2026 | `rollup-plugin-visualizer` / `webpack-bundle-analyzer` / `source-map-explorer` / `esbuild --analyze` |
| **No duplicate libraries** | Same library bundled twice at different versions | Lighthouse **`duplicated-javascript-insight`** |
| **No legacy transpilation** | Shipping Babel polyfills / transpiled ES5 to evergreen browsers is **pure dead weight**. Set a modern `browserslist` / `target: es2022` | Lighthouse **`legacy-javascript-insight`** |

### 4.3 Hydration cost and architecture — when a site should not be an SPA

**The cost of hydration is the core problem:** the server renders HTML, the client downloads the same component tree again, re-executes it, and reattaches listeners. **You pay for the page twice, and the page looks ready while being unresponsive** — exactly the INP failure mode, and exactly the TBT-divergence story from §0.

| Approach | Best when | Cost |
|---|---|---|
| **Static / MPA + progressive enhancement** | Content, marketing, docs, blogs, most brochure sites | **None. Should be the default** |
| **Islands** | Mostly-static pages with a few interactive widgets | Islands hydrate independently and progressively |
| **Server components** | Repeated navigation within one app; amortises runtime across a session | Still ships a client runtime; payload streaming |
| **Resumability** | Very large pages where TTI is the bottleneck | Serialization complexity; small ecosystem |
| **Classic SPA** | Genuinely app-like: editors, dashboards, canvases, real-time | **Highest INP/LCP risk, worst measurability** |

**The audit heuristic to state plainly in the report:** if the page is primarily content that a user **reads**, and the SPA exists only to avoid full page reloads, **it should not be an SPA**. Cross-document View Transitions + Speculation Rules now deliver the SPA *feel* on an MPA, with better LCP and better measurability. See `code-quality.md` §3 for the View Transitions mechanics.

**How to detect client-side rendering from outside:** `curl -s URL | sed 's/<[^>]*>//g' | wc -w` — a near-empty body with a large JS payload is CSR. Compare with the rendered word count in DevTools.

### 4.4 Third parties — the single biggest killer

**~90–92% of pages load third-party content.** The median page hits **≥ 9 unique third-party domains**, accounting for **35% of total network activity**; the top 10% of pages issue **175+ third-party requests**. Roughly **67.7% of websites load at least one render-blocking third party.** `[P]`/`[S]`
HTTP Archive analysis identifies **third-party code as the leading cause of poor INP**.

**Audit process (run this in order — it is the highest-yield sequence in this file):**
1. **Enumerate every third-party origin.** DevTools → Network → group by domain; Lighthouse `third-parties-insight`; WebPageTest domain breakdown; Request Map Generator.
2. **For each, record four facts:** who owns it, what business value it delivers, **what it costs in main-thread ms and bytes**, and when it was last reviewed.
3. **Delete the ones nobody can name an owner for.** In practice this is **20–40% of tags on a mature site.**
4. **Defer the rest** past load / behind interaction / behind consent.
5. **SPOF test:** does the page still render and does the core task still complete if `thirdparty.example` never responds? Use DevTools → Network → **request blocking** per origin. Third-party scripts must be `async`, never sync `<script>`, and never sync-injected CSS.

**Tag managers:** frequently the single largest offender. Server-side tag management collapses 10–20 browser requests to endpoints into **one request to your own server**. Audit annually; prune dead tags; use trigger conditions rather than "All Pages." `[S]`

**Web-worker relocation (Partytown-class):** moves third-party scripts to a web worker with synchronous DOM proxying via `Atomics`/service worker. Reported cases: a tag-manager container moved to a worker cut **TBT by 92%**, Lighthouse **70 → 99**. `[S]`
**Caveats to state honestly in the report:** it adds complexity, breaks scripts that need synchronous DOM/timing, and has had maintenance gaps. **Treat it as a targeted intervention for a known-bad heavy tag, not a default.**

**Consent managers (CMPs) are themselves a top INP/LCP offender** — they run early, block render, inject a full-screen banner (CLS), and gate everything downstream. Audit the CMP as a **first-class performance risk**: self-host it, reserve its space, and confirm the banner does not push content. Measure CLS with the banner **present**.

### 4.5 Dead code detection

- **DevTools → Coverage** — `Ctrl+Shift+P` → "Show Coverage" → record + reload, then **click through key interactions**. Red = unused bytes. **Any file > 60% unused is a pruning candidate.**
  **Caveat to keep in the report:** coverage is per-session — code used only on other routes shows as unused. It **identifies candidates; it does not prove deletability.**
- **Unused CSS:** PurgeCSS/`@fullhuman`, framework purge, `stylelint` for dead selectors. **Beware dynamic class names.**
- **Unused deps:** `depcheck`, `knip`, `ts-prune`.
- **Bundle composition:** `rollup-plugin-visualizer`, `webpack-bundle-analyzer`, `source-map-explorer`, `esbuild --analyze`, `bundlephobia`.

---

## 5. Verification — tools, limits, and the correct order

### 5.1 Capability matrix

| Tool | Gives you | **Cannot** give you |
|---|---|---|
| **Lighthouse 13** (Chrome 143+, Node ≥ 22.19) | Lab CWV proxies, Insights audits, opportunities, diagnostics; scriptable via CLI/Node | **INP** (no user input); real device/network mix; p75; anything about real users |
| **Lighthouse CI** | Per-PR regression gating, `budget.json` assertions, historical server, GitHub status checks | Field truth |
| **PageSpeed Insights** | Lighthouse lab **plus** CrUX field data side by side, at URL **and** origin level | Only Chrome field data; only if the URL has enough traffic |
| **CrUX API / History API / BigQuery** | The exact dataset Google ranks on; p75, histograms, **40 weeks** of history, device/connection/country dimensions | Per-pageview attribution; non-Chrome browsers; low-traffic URLs; anything fresher than the 28-day window |
| **RUM (`web-vitals` + attribution)** | Per-pageview, attributable, segmentable, **all** your users, custom dimensions | Competitor comparison; requires deployment |
| **WebPageTest** | Real devices at real network locations, **filmstrip + waterfall + connection view**, scripted multi-step flows, repeat-view, TTFB by geography. 2026 device lab includes ARM agents comparable to **Snapdragon 695 / Dimensity 900** | p75 field truth; user-population reality |
| **Chrome DevTools** | Performance panel + Insights, LoAF traces, Coverage, Network, Application (bfcache, speculative loads), Lighthouse panel, CPU/network throttling | Real-user data |
| **Playwright / Puppeteer** | Programmatic auditing: console errors, network 404s, response headers, screenshots, `PerformanceObserver` extraction, JS-disabled runs, per-viewport checks, `playwright-lighthouse` integration | Realistic device performance — **headless CI machines are unrealistically fast; always throttle explicitly**. Lighthouse-in-Playwright adds **~10–15 s per audit** and **breaks if multiple instances share a debug port** |

**Lighthouse 13 breaking changes (Oct 2025, Chrome 143)** `[P]`:
- Old performance audits were **removed** and replaced by consolidated **Insights** shared with the DevTools Performance panel. Any tooling or checklist referencing `render-blocking-resources`, `uses-webp-images`, `uses-responsive-images`, `uses-long-cache-ttl`, `uses-http2`, `critical-request-chains`, `uses-rel-preconnect`, `third-party-summary`, `dom-size`, `layout-shifts`, `server-response-time`, `redirects`, `uses-text-compression`, `largest-contentful-paint-element`, `prioritize-lcp-image`, `lcp-lazy-loaded`, `work-during-interaction`, `font-display`, `legacy-javascript`, `duplicated-javascript`, or `viewport` **by ID will break** — use the `*-insight` IDs.
- **Removed with no replacement:** `first-meaningful-paint`, `font-size`, `no-document-write`, `offscreen-images`, `preload-fonts`, `third-party-facades`, `uses-passive-event-listeners`, `uses-rel-preload`.
- **Performance scoring weights unchanged:** FCP 10%, Speed Index 10%, **LCP 25%, TBT 30%, CLS 25%**. **TBT + CLS = 55% — Lighthouse is not a CWV score.** Never present it as one.

**Also worth knowing:** `securityheaders.com`, Mozilla Observatory, `testssl.sh` (see `security-and-hygiene.md`), DebugBear / SpeedCurve / Calibre (commercial RUM + synthetic), `unlighthouse` (site-wide Lighthouse), `size-limit`, `knip`, Request Map Generator, `specrules.com`.

### 5.2 The correct order

Running these in the wrong order produces a report full of lab noise. Field first, always.

| Step | Tool | Question it answers | Do not skip to the next until |
|---|---|---|---|
| 1 | **CrUX API / PSI field panel** — origin **and** 3–5 representative templates, **mobile and desktop separately** | Is there a real problem, and which metric? | You have p75 numbers, or an explicit "no field data" note |
| 2 | **CrUX History API** | Is it a regression or a chronic state? (remember the 3-week overlap) | You know whether something changed |
| 3 | **RUM, if present** | Which template / device class / country carries the failure? | You have a segment, not an origin average |
| 4 | **Lighthouse mobile**, 4× CPU throttle, Slow 4G, ≥ 3 page types **including an inner page** | What are the candidate causes? | You have Insights output, not a score |
| 5 | **DevTools Performance + LoAF + Coverage**, manual interaction | Which cause is *actually* this site's? Which script? | You can name a file and a line |
| 6 | **WebPageTest** | Does it hold on a real device / real network / other geography? | Confirmed or refuted |
| 7 | **Playwright / `../scripts/audit_performance.py`** | Automate the deterministic checks (headers, unsized images, script attributes, console, 404s) and regression-gate them | Checks are reproducible |
| 8 | **Lighthouse CI + `budget.json`** | Will it stay fixed? | A budget is enforced in CI |
| 9 | Wait **28 days**, re-read CrUX | Did the field actually move? | — |

**Never:** report a Lighthouse score as the result (step 4 output ≠ step 1 truth); measure CLS with the consent banner dismissed; audit only the homepage; measure INP in Lighthouse (it cannot).

---

## 6. Performance budget template

Fill in, agree with the client **before** remediation, and enforce in CI. Anything without a number is not a budget.

```
SITE: ______________________          DATE: __________
TEMPLATES IN SCOPE (≥1 must be an inner page):
  T1 home:      ______________________
  T2 inner:     ______________________
  T3 inner:     ______________________
DEVICE PROFILE: mid-tier Android (Moto G-class / Snapdragon 695-class), Slow 4G, 4× CPU throttle
```

| # | Budget item | Target | T1 actual | T2 actual | T3 actual | Pass? | Source of truth |
|---|---|---|---|---|---|---|---|
| 1 | LCP p75 mobile | ≤ 2.5 s | | | | | CrUX |
| 2 | LCP p75 desktop | ≤ 2.5 s | | | | | CrUX |
| 3 | INP p75 mobile | ≤ 200 ms | | | | | CrUX / RUM |
| 4 | INP p75 desktop | ≤ 200 ms | | | | | CrUX / RUM |
| 5 | CLS p75 mobile | ≤ 0.1 | | | | | CrUX |
| 6 | CLS p75 desktop | ≤ 0.1 | | | | | CrUX |
| 7 | TTFB p75 | < 800 ms (target < 200 ms on edge) | | | | | CrUX / `curl -w` |
| 8 | TBT (lab, throttled) | < 200 ms | | | | | Lighthouse |
| 9 | Main-thread work (lab) | < 2 s | | | | | Lighthouse |
| 10 | First-load JS, compressed | ≤ 50 / 150 / 250 KB per §4.1; **hard ceiling 300 KB** | | | | | `size-limit` / Network |
| 11 | CSS, compressed | ≤ 150 KB general budget (median web: 82 KB); **≤ ~50 KB** on the mobile device floor per `mobile.md` §3.4 | | | | | Network |
| 12 | Fonts, total | ≤ 2 files preloaded; payload budget (median web: 139 KB) | | | | | Network → Font |
| 13 | Images, total transfer | set explicitly (median web: 1,059 KB) | | | | | Network → Img |
| 14 | Total page weight | set explicitly (median web: 2.56 MB mobile) | | | | | Network |
| 15 | DOM elements | < 1,500 | | | | | `dom-size-insight` |
| 16 | Render-blocking resources | 0 blocking JS in `<head>`; ≤ 1 blocking CSS | | | | | `render-blocking-insight` |
| 17 | `preload` count | ≤ 4, all used within 3 s | | | | | HTML + DevTools warning |
| 18 | `preconnect` count | ≤ 4 | | | | | HTML |
| 19 | Third-party origins | ≤ 9 (median), each with a named owner | | | | | Network by domain |
| 20 | Render-blocking third parties | **0** | | | | | `third-parties-insight` |
| 21 | Unsized media elements | **0** | | | | | `../scripts/audit_performance.py` |
| 22 | Non-composited animations | **0** | | | | | `non-composited-animations` |
| 23 | Console errors | **0** | | | | | Playwright |
| 24 | 404/5xx subresources | **0** | | | | | Playwright |
| 25 | Any single JS file > 60% unused | **0** | | | | | Coverage |
| 26 | Protocol | HTTP/2 or HTTP/3 | | | | | `nextHopProtocol` |
| 27 | Text compression | Brotli or Zstd on all text | | | | | `Content-Encoding` |
| 28 | bfcache eligible | yes | | | | | DevTools bfcache test |
| 29 | Budget enforced in CI | yes | | | | | repo config |

**Alert thresholds:** set regression alerts at ~80% of each CWV target (LCP 2.0 s, INP 160 ms, CLS 0.08) with a named recipient — see `seo/L2-technical-performance.md`.

---

## 7. Stale advice — flag it if you see it in the codebase or in a prior audit

| Stale advice | 2026 reality |
|---|---|
| "Optimise for FID" | FID deprecated **9 Sep 2024**; **INP** since March 2024 |
| "Get a 100 Lighthouse score" | Lab-only; **55% of the score is TBT+CLS**; field p75 is what ranks and what users feel |
| "Preload your fonts / preload everything" | Lighthouse **removed both preload audits** for over-recommendation risk. Cap at 2–4 targeted preloads |
| "Lazy-load all images" | **Never the LCP image**; Lighthouse removed `offscreen-images` because the browser already deprioritises offscreen images |
| "Use `isInputPending()` to break up tasks" | Chrome **reversed** this recommendation (Oct 2024). Use `scheduler.yield()` |
| "Concatenate everything into one bundle / domain-shard" | HTTP/1.1-era. **Harmful under HTTP/2/3** |
| "Use a hosted font CDN, it's cached across sites" | **Cache partitioning (2020) killed the shared-cache benefit**; self-host |
| "Use `<div>` + JS for modals/tooltips/dropdowns" | `<dialog>`, `popover`, anchor positioning, `field-sizing` are all shipping — `code-quality.md` §2 |
| "Serve WebP only, AVIF isn't ready" | AVIF **~94%** support; use `<picture>` with AVIF → WebP → JPEG |
| "gzip is fine" | Brotli is universal; **zstd is 12% of CDN traffic** and rising |
| "Firefox doesn't do view transitions / anchor positioning" | Firefox **132** ships anchor positioning; cross-doc VT is partial in **FF 146–151** |
| "CSS anchor positioning landed in Safari 18.2" | **Wrong — Safari 26.0**, refined in **26.2** |
| "Google tightened INP methodology in 2026" | Not supported by the Chromium INP changelog `[?]` |
| "Engagement Reliability is a new Core Web Vital" | No primary source found. **Do not assert** `[?]` |
| "SPAs can't be measured for CWV" | **Obsolete as of Chrome 151** (Soft Navigations API) |
| "Ship source maps, they're harmless" | See `code-quality.md` §7 — the March 2026 npm incident |
| "`Cache-Control: no-store` is a safe default for HTML" | **Kills bfcache in Firefox/Safari**; 23% of sites do it |

---

## 8. Testable checklist

Every item resolves to **pass / fail / N-A with evidence**. `[AUTO]` = fully automatable via Playwright / curl / Lighthouse JSON — see `../scripts/audit_performance.py`.

### A. Field & lab metrics
- [ ] `[AUTO]` CrUX p75 fetched for the origin **and** 3–5 representative page templates, **mobile and desktop separately** — LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1
- [ ] `[AUTO]` If CrUX has no URL-level data, **note it explicitly** rather than silently falling back to origin
- [ ] `[AUTO]` Lighthouse mobile (4× CPU throttle, Slow 4G) on ≥ 3 page types **including at least one inner/secondary page**
- [ ] `[AUTO]` TBT < 200 ms; main-thread work < 2 s; `dom-size-insight` under threshold
- [ ] RUM present? `web-vitals` (attribution build) or commercial RUM shipping LCP/INP/CLS/TTFB with device + template segmentation
- [ ] If SPA: soft-navigation measurement present, or at minimum **acknowledged as a gap**
- [ ] `[AUTO]` No CWV metric is in "poor" on any tested template

### B. LCP
- [ ] `[AUTO]` LCP element identified on each template (`lcp-phases-insight` / DevTools)
- [ ] `[AUTO]` LCP image URL present in the **initial HTML response** (fetch with JS disabled and grep) — no `data-src`
- [ ] `[AUTO]` LCP image has **no** `loading="lazy"` and **no** `fetchpriority="low"`
- [ ] `[AUTO]` LCP image has `fetchpriority="high"` (or a preload paired with it)
- [ ] `[AUTO]` If the LCP image is cross-origin, a `preconnect` to that origin exists
- [ ] `[AUTO]` TTFB < 800 ms (target < 200 ms with edge/CDN)
- [ ] `[AUTO]` HTML served from a CDN edge (`cf-cache-status` / `x-cache` / `age`)
- [ ] `[AUTO]` LCP **resource-load-delay is not the dominant sub-part**

### C. INP / main thread
- [ ] `[AUTO]` No long task > 200 ms during page load
- [ ] Click/type through the **5 most important interactions** with 4× CPU throttling; record INP per interaction; all ≤ 200 ms
- [ ] `[AUTO]` LoAF traces captured for the worst interaction; the responsible script is **named**
- [ ] `[AUTO]` `scheduler.yield()` or an equivalent yielding strategy in any handler doing > 50 ms of work, **with a Safari fallback**
- [ ] `[AUTO]` No forced-synchronous-layout warnings in the Performance trace
- [ ] `[AUTO]` No `unload` event listeners anywhere on the page

### D. CLS
- [ ] `[AUTO]` Every `<img>`/`<video>`/`<iframe>` has `width`+`height` or `aspect-ratio` — **report the count** of unsized elements
- [ ] `[AUTO]` No `layout-shift` entry with `hadRecentInput: false` and `value > 0.01` after load
- [ ] `[AUTO]` No animation/transition on `width`, `height`, `top`, `left`, `margin`, `padding`, `border-width`, `font-size` (`non-composited-animations`)
- [ ] `[AUTO]` bfcache eligible: DevTools bfcache test passes; no `unload` handler; no `Cache-Control: no-store` on HTML
- [ ] `[AUTO]` Cookie/consent banner and any injected notification reserve their space — **measure CLS with the banner enabled**
- [ ] `[AUTO]` Font fallback metric overrides present (`size-adjust`/`ascent-override`) **OR** `font-display: optional`
- [ ] `[AUTO]` `content-visibility: auto` always paired with `contain-intrinsic-size`

### E. Loading & delivery
- [ ] `[AUTO]` Render-blocking: **0** blocking JS in `<head>`; blocking CSS ≤ 1 file (or inlined critical CSS)
- [ ] `[AUTO]` Every `<script>` has `defer`, `async`, or `type=module` (or is intentionally inline and tiny)
- [ ] `[AUTO]` `preload` count ≤ 4; `preconnect` count ≤ 4; **every preload is actually used within 3 s** (DevTools warns if not)
- [ ] `[AUTO]` No `prefetch` of anything needed for the **current** page
- [ ] `[AUTO]` Fonts: WOFF2 only; ≤ 2 preloaded; every font preload has `crossorigin`; `font-display` is `swap`/`optional`/`fallback` (never `auto`/`block`); self-hosted or justified
- [ ] `[AUTO]` Images: modern format actually served (check `Content-Type` under an AVIF/WebP `Accept`); `srcset`+`sizes` on responsive images; **no image served > 2× its rendered CSS pixel size**
- [ ] `[AUTO]` Below-fold images have `loading="lazy"`; iframes too
- [ ] `[AUTO]` Video: `poster` set; `preload` is `none` or `metadata`; autoplay video has `muted` + `playsinline`; heavy embeds use a facade
- [ ] `[AUTO]` Protocol is HTTP/2 or HTTP/3 (`curl -sI --http3` / `nextHopProtocol`)
- [ ] `[AUTO]` All text responses compressed with Brotli or Zstd (`Content-Encoding`)
- [ ] `[AUTO]` Hashed assets `Cache-Control: public, max-age=31536000, immutable`; HTML **not** `no-store`
- [ ] Speculation Rules present (optional enhancement); if present: `eagerness` is `moderate`/`conservative`, destructive/auth URLs excluded, analytics gated on `document.prerendering`

### F. JavaScript
- [ ] `[AUTO]` Total compressed first-load JS within the agreed budget (default ≤ 200 KB; ≤ 50 KB for content sites)
- [ ] `[AUTO]` No single JS file > 60% unused in DevTools Coverage on first load
- [ ] `[AUTO]` No duplicated library versions (`duplicated-javascript-insight`)
- [ ] `[AUTO]` No legacy transpilation/polyfills for evergreen targets (`legacy-javascript-insight`)
- [ ] `[AUTO]` Third-party origin count and total third-party transfer size recorded; **every origin has a named owner and a business justification**
- [ ] `[AUTO]` No render-blocking third-party script
- [ ] **SPOF test:** page still renders and the core task completes with each third-party origin blocked (DevTools request blocking)
- [ ] `[AUTO]` A performance budget is **enforced in CI** (`budget.json` / `size-limit` / Lighthouse CI assertions)
- [ ] **Architecture check:** if the page is primarily read-only content, it is **not** a client-rendered SPA
