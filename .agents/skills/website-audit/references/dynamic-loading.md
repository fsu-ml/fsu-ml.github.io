# Dynamic loading and scroll-triggered content

**Covers:** scroll-triggered reveals, progressive/deferred content, `content-visibility`, infinite scroll, scroll-jacking and smooth-scroll libraries, and whether a JS animation library is justified.
**Load when:** the page has fade-in-on-scroll, staggered card reveals, a "load more" pattern, infinite scroll, `content-visibility`, a sticky/pinned scroll section, parallax, or a custom scroll feel.
**Why this is its own file, separate from `animation-and-motion.md`:** scroll-triggered reveals have a failure mode nothing else on the page has — **content that never appears at all**. That is a content-availability bug wearing an animation costume, and it outranks every performance concern in this skill.

**Companion files:** `animation-and-motion.md` (pipeline, compositing, measurement), `../examples/scroll-reveal.md` (copy-pasteable implementations), `../examples/motion-system.md`, `performance.md`, `mobile.md`, `viz-libraries.md`, `ada/media-and-motion.md`, `seo/L1-foundations.md`, `../scripts/audit_motion.py`.

**Source material verified 19 August 2026** against MDN BCD JSON, caniuse (StatCounter July 2026), Chromium `main`, W3C/WAI, WebKit release blogs, the Firefox release train, Google Search documentation, and measured npm/CDN artifacts.

---

## 1. IntersectionObserver, done right

### 1.1 Why it beats scroll listeners

Animating from a `scroll` event handler is wrong for three independent reasons:

1. "Modern browsers perform scrolling on a separate process and therefore deliver scroll events **asynchronously**." By the time your handler reads `scrollY`, the compositor **has already painted** a frame you have not caught up to. This is the source of parallax rubber-banding.
2. Scroll events fire **off-cadence** — potentially many per frame. All DOM writes but the last are discarded, and each event can force a layout.
3. The listener runs on the main thread: "Scripts can observe it through the `scroll` event and **cannot interrupt**."

IntersectionObserver runs **once per frame in the "Update the rendering" step** and delivers geometry **without forcing layout**, whereas a scroll handler calling `getBoundingClientRect()` forces synchronous layout on every event.

⚠️ **But IntersectionObserver does NOT run off the main thread.** Flag any documentation, blog post, or PR description claiming otherwise. It is cheap, not free.

If a scroll listener genuinely cannot be avoided, the read/write split is the minimum bar:

```js
let ticking = false, y = 0;
addEventListener('scroll', () => {
  y = window.scrollY;                       // read only
  if (!ticking) {
    ticking = true;
    requestAnimationFrame(() => {
      el.style.transform = `translate3d(0, ${y * 0.5}px, 0)`;   // write only
      ticking = false;
    });
  }
}, { passive: true });
```

### 1.2 Options that actually matter

**`rootMargin`** — 1–4 values, `top right bottom left`. **Units `px` and `%` only**; `em`, `rem`, `vh`, bare numbers, and `calc()` throw `SyntaxError`.

⚠️ **Percentages resolve against the root rect's WIDTH — including the top and bottom values.** `"50% 0px"` on a 1440×900 viewport is **720 px vertically, not 450 px**. This catches almost everyone. `rootMargin` applies only to the intersection root. An element root **with a content clip** uses its **padding box**; without one, its border box.

`root: null` means the **top-level browsing context's** document — inside an iframe that is the top page's viewport, not the iframe's. **For cross-origin-domain targets, `rootMargin`/`scrollMargin` are ignored and `entry.rootBounds` is `null`** — reading `rootBounds.height` throws.

**`scrollMargin`** (margins applied at every scroll container on the path) — Chrome/Edge 120 (Dec 2023), Firefox 141 (2025-07-22), Safari/iOS 26.0 (2025-09-15). **Baseline Newly available Sept 2025, caniuse 86.09%.** Guard with `'scrollMargin' in IntersectionObserver.prototype`.

**`threshold`** outside `[0,1]` throws `RangeError`. `intersectionRatio` is intersected area over the **target's own** area; a zero-area target reports `1` when intersecting.

⚠️ **`isIntersecting` is spec'd threshold-independent, but Blink and WebKit implement it as `thresholdIndex > 0`** ([w3c/IntersectionObserver#432](https://github.com/w3c/IntersectionObserver/issues/432), [Chromium 40693914](https://issues.chromium.org/issues/40693914), [Bugzilla 1611204](https://bugzilla.mozilla.org/show_bug.cgi?id=1611204)). **Never combine a non-zero `threshold` with a bare `isIntersecting` check** — compare `entry.intersectionRatio` explicitly instead.

`trackVisibility` + `delay` are **Chrome 74+ only**; `delay` clamps to ≥100 ms.

### 1.3 Tuning table

| Goal | Setting | Note |
|---|---|---|
| Reveal slightly before the element reaches the viewport edge | `rootMargin: '0px 0px -10% 0px'` with `threshold: 0` | Negative bottom margin shrinks the root, delaying the fire until the element is further in |
| Reveal as soon as one pixel enters | `threshold: 0` | The safe default |
| Reveal when the element is "mostly" visible | `threshold: 0.5` — **only if the element is shorter than the viewport** | See §1.4 bug 2 |
| Fully visible | `threshold: 0.99`, **not `1.0`** | Sub-pixel rounding means `1.0` can never fire |
| Prefetch / lazy work ahead of the fold | `rootMargin: '200px 0px'` | px, not vh |
| Sentinel for a very tall section | 1 px `<div>` at the section top, `threshold: 0` | Immune to the tall-element bug |

### 1.4 Five classic bugs

1. **Flash of hidden content on load.** Spec §3.4.2 **guarantees an initial notification — asynchronously, on the next frame.** So a flash is a **CSS ordering bug, not an observer bug** (see §3).
2. **Elements taller than the viewport never fire.** A 3000 px section in a 900 px viewport maxes out at `intersectionRatio ≈ 0.3`, so `threshold: 0.5` or `1.0` never fires. Fix with `threshold: 0` + `rootMargin`, or a 1 px sentinel. Use `0.99` rather than `1.0`.
3. **bfcache.** `pageshow` fires with `event.persisted === true`; IO registrations survive but produce **no new entries**.
   ```js
   window.addEventListener('pageshow', e => { if (e.persisted) sweepAndReveal(); });
   ```
   Never use `unload` — use `pagehide`. `history.scrollRestoration` defaults to `"auto"`.
4. **Zero-area targets.** `display:none` gives a rect of `0,0,0,0`; the spec allows `isIntersecting: true, ratio: 1` but Blink/WebKit report `false` — **not interoperable**. Same class of bug: `height: 0`, an unsized `<img>`, and **children of a skipped `content-visibility: auto` subtree**. Re-`observe()` after toggling.
5. **Cross-origin:** `rootBounds: null`, margins silently ignored.

### 1.5 Unobserve after reveal

Spec §3.3: observers stay alive per registered target, so a 500-card feed means **500 geometry computations per frame, forever**. Call `unobserve(entry.target)` inside the callback once the element has revealed; call `disconnect()` only on teardown.

**Verify:** in DevTools, break on the observer callback and confirm the observed-target count decreases as you scroll. Or count: `grep -c 'unobserve' src/` should be non-zero wherever a one-shot reveal exists. Confirm the cost with a Performance trace — a growing per-frame cost as you scroll a long list is the tell.

---

## 2. CSS-only scroll-driven animations

Zero JavaScript, and — crucially — **off the main thread entirely** where the engine supports threaded scroll animations.

### 2.1 API surface

Two timeline types:

| Type | Driven by | Anonymous | Named |
|---|---|---|---|
| **Scroll progress** | scroll position of a container (0% = start, 100% = end) | `scroll(...)` | `scroll-timeline-name` / `scroll-timeline` |
| **View progress** | *subject* position within its scrollport | `view(...)` | `view-timeline-name` / `view-timeline` |

```
<scroll()> = scroll( [ <scroller> || <axis> ]? )
<scroller> = root | nearest | self     /* default: nearest */
<axis>     = block | inline | x | y    /* default: block  */

<view()> = view( [ <axis> || <'view-timeline-inset'> ]? )
<view-timeline-inset> = [ [ auto | <length-percentage> ]{1,2} ]#
```

```css
animation-timeline: scroll();             /* = scroll(nearest block) */
animation-timeline: scroll(root);         /* page scroll — reading progress bar */
animation-timeline: scroll(x self);       /* carousel scrolling itself */
animation-timeline: view();               /* = view(block auto) */
animation-timeline: view(block 25% 0%);   /* start firing 25% down the viewport */
```

`nearest` = the nearest ancestor **that is a scroll container**; `self` = the element itself. `x`/`y` are physical, `block`/`inline` are writing-mode-relative. **An axis with no scrollbar makes the timeline inactive (zero progress)** — a silent no-op. Scroller and axis are order-free. `view()`'s subject is always the declaring element and its scroller is not configurable — that is the key difference from `scroll()`. A positive inset insets, **negative outsets**, and `auto` uses `scroll-padding`.

**Named timelines** resolve only to the declaring element's **descendants and following siblings' descendants** — down and forward, never up. `timeline-scope` hoists a name:

```css
body { timeline-scope: --hero; }
.hero { view-timeline-name: --hero; }
.sticky-nav { animation-timeline: --hero; }
```

⚠️ **Zero, or more than one, timeline of that name yields an inactive timeline with no error** — the #1 debugging trap.
⚠️ **`timeline-scope: all` was implemented in Chrome 116 and REMOVED in Chrome 138** (BCD: `"version_added": "116"`, `"version_removed": "138"`). Safari 26 supports it. Flag any use.

**`animation-range`** — subject height *H*, scrollport *V*:

| Keyword | 0% | 100% |
|---|---|---|
| `cover` | first pixel enters the scrollport | last pixel leaves |
| `contain` | subject fully visible (if H > V: fills the scrollport) | subject starts to leave |
| `entry` | first pixel enters | fully entered — **if H > V, clamps to "fills the scrollport"** |
| `entry-crossing` | first pixel enters | **last** pixel crossed the start edge |
| `exit` | begins to exit — **if H > V, when the last pixel has entered** | last pixel leaves |
| `exit-crossing` | **first** pixel crosses the end edge | last pixel leaves |

`entry`/`exit` are equivalent to `entry-crossing`/`exit-crossing` when the subject is shorter than the scrollport. Default on `view()` with no range: **`cover 0%` → `cover 100%`**. ([WebKit cheatsheet, Jul 2025](https://webkit.org/blog/17184/so-many-ranges-so-little-time-a-cheatsheet-of-animation-ranges-for-your-next-scroll-driven-animation/))

⚠️ **The `animation` shorthand trap.** `animation-timeline` is **reset-only in the `animation` shorthand** — it must come *after* it. Same for `animation-range-start`/`-end`. `animation-duration` is ignored.

```css
/* BROKEN — the shorthand resets animation-timeline to auto */
.reveal { animation-timeline: view(); animation: fade 1s linear both; }
/* CORRECT */
.reveal { animation: fade linear both; animation-timeline: view(); animation-range: entry; }
```

**Prefer this over a scroll listener for progress bars.** Animate `transform: scaleX()`, never `width`:

```css
@keyframes grow-progress { from { transform: scaleX(0); } to { transform: scaleX(1); } }
#progress {
  position: fixed; inset: 0 auto auto 0; width: 100%; height: 1em;
  transform-origin: 0 50%;
  animation: grow-progress auto linear;
  animation-timeline: scroll();
}
```

### 2.2 Definitive browser support (19 August 2026)

| Engine | First support | Date |
|---|---|---|
| Chrome / Edge | **115** | 2023-07-18 |
| Opera | 101 | |
| Samsung Internet | 23 | |
| Chrome Android / WebView | 115 | |
| Safari (macOS) / iOS Safari | **26.0** | 2025-09-15 |
| Safari — threaded / compositor-driven | **26.4** | 2026-03-24 |
| Firefox / Firefox Android | **none shipped** (Nightly only) | — |

**Firefox: "Firefox 132+ supports it" is FALSE. "Firefox 152 still behind a flag" is TRUE — and it is still flagged in Firefox 154 today.**

MDN BCD (`animation-timeline.json`): `"firefox": { "version_added": "preview" }` — Nightly only, never shipped to stable. Same for `scroll()`, `view()`, `view-timeline`, `timeline-scope`.

| Firefox | Release day | Status |
|---|---|---|
| 152.0 | 2026-06-16 | flagged off |
| 153.0 | 2026-07-21 | flagged off |
| **154.0** | **2026-08-18 (current stable)** | **flagged off** |
| 155.0 | 2026-09-01 (beta) | flagged off |

Flag: `layout.css.scroll-driven-animations.enabled`. Meta-bugs [1676779](https://bugzilla.mozilla.org/show_bug.cgi?id=1676779), [1676780](https://bugzilla.mozilla.org/show_bug.cgi?id=1676780), [1817303](https://bugzilla.mozilla.org/show_bug.cgi?id=1817303). It is an **Interop 2026 focus area** but is not scheduled. "Firefox 157: Supported" on caniuse is BCD `"preview"` rendered in its Nightly column. The "Firefox 132+" claim likely confuses this with `content-visibility` (Firefox 125, Apr 2024).

**Safari: 26.0 (15 Sept 2025) is the first version. Safari 18 does NOT have it.** BCD `"safari": { "version_added": "26" }`. **Threaded (compositor-driven) scroll animations landed in Safari 26.4 (24 March 2026)** — eligible properties are `opacity`, `transform`, `translate`, `scale`, `rotate`, `filter`, `backdrop-filter`, and Motion Path.

**Global support: 85.43% full, 0% partial** (caniuse, StatCounter July 2026). **Baseline status: "Limited availability"** — blocked by Firefox since September 2025. **Do not call it Baseline.** Flag any documentation or PR description that does.

⚠️ **The polyfill is stale.** [flackr/scroll-timeline](https://github.com/flackr/scroll-timeline) is not archived, but npm `scroll-timeline-polyfill`'s latest release is **1.1.0, published 2024-05-15**. It is not a safety net; do not let a team treat it as one.

### 2.3 Progressive enhancement via `@supports`

Feature detection, all equivalent-ish:

| Test | Notes |
|---|---|
| `@supports (animation-timeline: view())` | **False in Firefox stable** — the pref gates parsing, which is exactly the behaviour you want |
| `@supports (animation-range: entry)` | Narrower; use when you rely on ranges |
| `CSS.supports('animation-timeline','view()')` | JS equivalent |
| `'ViewTimeline' in window` | Tests the WAAPI constructor, not the CSS property |

Pair the feature gate with a motion gate so that failing *either* leaves the content visible. Full implementations in `../examples/scroll-reveal.md`.

---

## 3. 🔴 The fail-open requirement — blocker-severity

> **Never put the hidden state (`opacity: 0`) in a static base rule. Put it inside a `@keyframes` `from` block and let `animation-fill-mode` apply it only when the animation actually exists.**

This is the single highest-value check in this entire skill. A static `opacity: 0` undone only by JS is a **permanent-blank-page hazard**: JS 404s, CSP blocks, ad-blocker interference, a 3G timeout, a non-rendering crawler, an observer that misfires, an element that starts in-viewport, or a user who lands mid-page via `#anchor` or back-navigation — any one of those leaves the content invisible forever.

### 3.1 The anti-pattern to grep for

```css
/* FAIL-CLOSED. This is the bug. */
.reveal { opacity: 0; transform: translateY(2rem); transition: all .6s; }
.reveal.is-visible { opacity: 1; transform: none; }
```

Also fragile — **prefer positive gating over negative un-gating**:

```css
/* Ships opacity:0 by default; the escape hatch loses to specificity/order
   and does nothing at all for JS failure. */
.reveal { opacity: 0; }
@supports not (animation-timeline: view()) { .reveal { opacity: 1; } }
```

### 3.2 The correct pattern

```css
@keyframes reveal-in {
  from { opacity: 0; transform: translateY(2rem); }
  to   { opacity: 1; transform: none; }
}

/* Feature gate + motion gate. If either fails, .reveal has NO styles at all. */
@media (prefers-reduced-motion: no-preference) {
  @supports (animation-timeline: view()) {
    .reveal {
      animation: reveal-in linear both;  /* `both` fill applies the from-state only within/around the range */
      animation-timeline: view();        /* MUST come after the shorthand */
      animation-range: entry 0% cover 40%;
    }
  }
  @supports not (animation-timeline: view()) {   /* JS/IntersectionObserver fallback layer */
    .js-reveal .reveal { opacity: 0; transform: translateY(1rem); }
    .js-reveal .reveal.is-revealed {
      opacity: 1; transform: none;
      transition: opacity .5s, transform .5s;
    }
  }
}
```

Every failure mode resolves to *visible*:

| Failure | Result |
|---|---|
| No feature support (Firefox stable, older Safari) | `@supports` block dropped → visible |
| `prefers-reduced-motion: reduce` | `@media` dropped → visible, no motion (WCAG 2.3.3 satisfied for free) |
| Element already in the viewport at load | Past `entry`; backwards fill does not apply → **no flash of hidden content** |
| Inactive timeline (no scrollbar on that axis, unresolved name) | No animation → visible |
| Non-JS crawler | The CSS never hides it → visible in rendered and raw HTML alike |
| CSP blocks the stylesheet | No rule at all → visible |

If you must hide via JS, make **JS itself apply the hiding**:

```html
<!-- in <head>, before the stylesheet -->
<script>document.documentElement.classList.add('js-reveal');</script>
<noscript><style>.reveal{opacity:1!important;transform:none!important}</style></noscript>
```

### 3.3 Explicit test procedure — run all six

*The `Sev` column below uses a local scale — P0 = blocking defect, P1 = should fix, P2 = advisory — that does not ship. Translate to the canonical `blocker | major | minor | advisory` before writing a finding; crosswalk in `reporting.md` §2. Note that `P2` here means advisory, whereas `P2` in `seo/triage.md` means High.*

| # | Test | Steps | Pass criterion | Sev |
|---|---|---|---|---|
| 1 | **JS disabled** | DevTools → Command Menu → "Disable JavaScript" → hard reload → scroll the full page | Every section of content is visible and readable | **P0** |
| 2 | **No scroll-driven animation support** | Load in Firefox stable (154), or set `layout.css.scroll-driven-animations.enabled = false` in `about:config` | All content visible | **P0** |
| 3 | **Deep link mid-page** | Load `https://site/page#some-section-far-down` in a fresh tab | The target section and everything around it is visible; the browser lands on the right element | **P0** |
| 4 | **Scroll restoration / back-navigation** | Scroll to the bottom → navigate away → press Back | Content is visible at the restored position, not blank. Check for a `pageshow` / `event.persisted` handler | **P1** |
| 5 | **Slow network** | Network throttling → "Slow 4G" or a custom 400 kbps profile; reload and immediately scroll | Content is visible before the reveal script finishes; no long blank stretch | **P1** |
| 6 | **Raw HTML** | `curl -s URL \| grep -c "<known body phrase>"` | Non-zero. ~28% of crawl traffic never executes JS (§4) | **P0** |

Additional resilience checks:

| # | Check | How | Sev |
|---|---|---|---|
| 7 | No static `opacity: 0` / `visibility: hidden` / `transform: scale(0)` on JS-only-revealed content | `grep -rn "opacity:\s*0" *.css` then confirm each hit is inside `@supports`, a `@keyframes from`, or a JS-applied class | **P0** |
| 8 | The hiding class is applied by JS in `<head>`, not shipped in the stylesheet | Read the head; check for a `<noscript>` override | P1 |
| 9 | No `opacity: 0` element is keyboard-focusable ("phantom focus") | Tab through the page **before** scrolling; watch for the focus ring disappearing | P1 |
| 10 | Elements already in the viewport on load are visible immediately with no flash | Record a Performance trace with screenshots; inspect the first 3 frames | P1 |

---

## 4. SEO and assistive-tech consequences

Cross-reference `seo/L1-foundations.md` for crawl and indexation fundamentals; this section covers only the scroll-reveal delta.

### 4.1 What Google does and does not do

Google ([lazy-loading docs](https://developers.google.com/search/docs/crawling-indexing/javascript/lazy-loading), updated 2025-12-10): *"Google Search does not interact with your page."*
[JS SEO basics](https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics) (2026-03-04): *"If the content isn't visible in the rendered HTML, Google won't be able to index it."*

Googlebot's rendering viewport is approximately **768 × 9,307 desktop** and **431 × 12,140 mobile** — very tall, which is why some reveal mechanisms work by accident.

| Reveal mechanism | Indexed? | Why |
|---|---|---|
| IntersectionObserver reveal | ✅ | The tall render viewport intersects everything, so the observer fires |
| CSS `animation-timeline: view()` reveal | ✅ | The content is never statically hidden if authored per §3.2 |
| **Scroll-*event* reveal** | ❌ | Googlebot does not scroll. This is the architecture of the abandoned scroll-reveal libraries |
| Click-to-load / "Load more" button | ❌ | *"Google Search does not interact with your page"* |
| Infinite scroll without paginated URLs | ❌ | Same |

**`opacity: 0` and `display: none` are both indexed** with no demotion (Illyes 2016, Mueller Mar 2020). ⚠️ Pre-2016 "hidden content is discounted" advice is obsolete — do not repeat it. The problem with `opacity: 0` is not ranking; it is that the content may never become visible to a *human* (§3) and that it stays in the accessibility tree (§4.3).

### 4.2 The AI-crawler finding

[Vercel × MERJ](https://vercel.com/blog/the-rise-of-the-ai-crawler), across ~1.3 billion fetches: *"none of the major AI crawlers currently render JavaScript"* — OpenAI, Anthropic, Meta, ByteDance, Perplexity. Monthly volumes: GPTBot **569M**, ClaudeBot **370M**, AppleBot **314M**, PerplexityBot **24.4M** ≈ **28% of Googlebot's 4.5B**. They download JS (GPTBot 11.50% of fetches, ClaudeBot 23.84%) but never execute it. Gemini and AppleBot **do** render; CCBot does not.

> **Audit rule: if it is not in `curl`'s output, it does not exist for roughly a third of your crawl traffic.**

Verification: `curl -s URL | grep -c "<distinctive phrase from a scroll-revealed section>"`. Zero is a P0 finding, not an SEO nicety.

### 4.3 The "phantom focus" accessibility trap

| Technique | Visually hidden | In the accessibility tree | Focusable |
|---|---|---|---|
| `opacity: 0` | ✅ | ✅ **still announced** | ✅ **still tabbable** |
| `visibility: hidden` | ✅ | ❌ | ❌ |
| `display: none` | ✅ | ❌ | ❌ |
| `.sr-only` clip | ✅ | ✅ | ✅ |

`opacity: 0` links and buttons are tabbable and invisible — a keyboard user tabs into nowhere and the focus ring vanishes. **Verify by Tabbing through the page from the top before scrolling at all** and watching where focus goes. If a section is genuinely meant to be hidden until revealed, it needs `visibility: hidden` alongside the opacity (and then it must still fail open per §3) — but the better answer is not to hide it.

---

## 5. `content-visibility: auto` + `contain-intrinsic-size`

A rendering-work optimisation for long pages, not a reveal mechanism. It belongs here because it interacts badly with the same things reveals do: find-in-page, anchor links, the accessibility tree, and IntersectionObserver.

| Value | Containment applied | Contents skipped | Find-in-page / tab / focus / selection |
|---|---|---|---|
| `visible` | none | no | normal |
| `auto` | **layout + style + paint always**; **+ size while skipping** | only when **not relevant to the user** | **must remain fully available** |
| `hidden` | layout + style + paint + size, always | always | **must NOT be accessible** |

For `auto`, layout/style/**paint** containment persist **even when the element is not skipped**, so the element permanently becomes a **stacking context**, a **containing block for `fixed`/`absolute`** descendants, and **clips ink overflow**. "Relevant to the user" means it intersects the viewport or the UA margin (~50%), is focused, is selected, or is in the top layer. While skipped, transitions and animations do not advance and there is no `innerText` contribution. Hook: **`contentvisibilityautostatechange`**.

```
contain-intrinsic-size = [ auto? [ none | <length [0,∞]> ] ]{1,2}
```
```css
contain-intrinsic-size: none;
contain-intrinsic-size: 1000px;                 /* both axes */
contain-intrinsic-size: 1000px 1.5em;           /* width then height */
contain-intrinsic-size: auto 300px;             /* last-remembered-size, fallback 300px */
contain-intrinsic-size: auto none;
contain-intrinsic-size: auto 300px auto 4rem;   /* two auto pairs */
```

**Percentages are not allowed.** Initial value `none`. `auto` means "last remembered size." MDN: `auto none` is "almost always preferred to setting `0px`". ⚠️ **Chrome 117+ and Firefox force `contain-intrinsic-size: auto` when `content-visibility: auto` is set** ([csswg-drafts#8407](https://github.com/w3c/csswg-drafts/issues/8407#issuecomment-1440466558)).

**Find-in-page:** skipped `auto` content must stay findable and tabbable — Chrome has done this since 85. ⚠️ **Safari 18–25 was broken** ([WebKit 283846](https://bugs.webkit.org/show_bug.cgi?id=283846)); **fixed in Safari 26.** ⚠️ **`hidden-matchable` never shipped** — the real feature is **`hidden="until-found"` + `beforematch`**: Chrome/Edge 102 (2022-05-24), Firefox 148 (2026-02-24), Safari none → **Baseline `limited`**. Detect with `'onbeforematch' in HTMLElement.prototype`.

**Anchor links:** spec §4.4 note 5 — skipping elements are scrolled to **with size containment still active**, so "it might not align in the viewport exactly as requested." Note 6 — focus un-skips first, so **prefer `.focus()` over a raw `scrollIntoView()`**. `#:~:text=` fragments work; they cannot reach `content-visibility: hidden` unless `hidden="until-found"` is used.

**Scrollbar jumpiness**, fixes best-first: (1) `contain-intrinsic-size: auto <length>` with a median estimate; (2) apply per card, not per section; (3) server-render heights inline; (4) **never above the fold** — `.item:nth-child(n+10) { content-visibility: auto; }`; (5) keep scroll anchoring enabled. ⚠️ Alex Russell's Dec 2020 IntersectionObserver + MutationObserver/ResizeObserver workarounds are superseded.

**Accessibility:** `hidden` is removed from the accessibility tree; `auto` stays. ⚠️ **Trap:** styles inside skipped subtrees are not computed, so `display:none` / `visibility:hidden` descendants **still appear in the accessibility tree** — add `aria-hidden="true"` explicitly. Keep headings and landmarks outside skipped subtrees. ⚠️ The 2021 "screen readers can't reach it" claim is outdated ([csswg-drafts#5857](https://github.com/w3c/csswg-drafts/issues/5857) resolved).

**Support:** `content-visibility` — Chrome/Edge **85** (2020-08-25), Firefox **125** (2024-04-16; 109–124 shipped disabled), Safari/iOS **18.0** partial → **26.0** full. **Global 93.19%, Baseline Newly available 2025-09-15.** `contain-intrinsic-size` — Chrome **83**, Firefox **107**, Safari **17.0**; `auto none` from Chrome/Firefox 117 and Safari 17. **Global 93.25%, Baseline Widely available since 2026-03-18.**

**Evidence it is worth it:** web.dev's travel blog demo went from **232 ms → 30 ms rendering (~7×)**, with a stated expectation of "reduction of 50% or more." Facebook reported "up to 250 ms improvement in navigation times." ⚠️ **In-viewport use can DELAY LCP** — this is the most common way teams make things worse with it.

**Verify:** measure rendering time before and after in the Performance panel (Rendering/Recalculate Style + Layout totals), test Ctrl+F in **both** Chrome and Safari 26, screen-record a full scroll looking for scrollbar jumps, and inspect the accessibility tree inside a skipped subtree.

---

## 6. Infinite scroll, scroll-jacking, and smooth-scroll libraries

### 6.1 Infinite scroll

| When it is defensible | When it is wrong |
|---|---|
| Homogeneous, endless, low-commitment feeds where no item is a destination (social timelines, image walls) | Anything with a **footer** the user needs |
| Content the user browses rather than searches | Anything a user needs to **return to** a specific item in |
| Contexts where no deep link into position is expected | Search results, product listings, documentation, news archives |

**The footer-unreachable problem.** New content loads as the user approaches the bottom, so the footer is pushed away every time they get close. Contact details, legal links, accessibility statements, and the site map become unreachable — a usability failure and, where those links are the only route to required information, a conformance problem too.

**Other failure modes to check for:**

| Failure | Test |
|---|---|
| No paginated URLs → nothing beyond page 1 is indexed | `curl` a deep item's expected URL; check for `<link rel="next">` or real `?page=N` routes (`seo/L1-foundations.md`) |
| Back-navigation loses position and re-fetches from the top | Scroll deep, click an item, press Back |
| Screen reader is never told new items arrived | Check for a live region announcing "20 more items loaded" |
| Focus is not managed after a load | Tab after new content appears; focus should not jump to the top |
| Unbounded DOM growth → per-frame cost climbs forever | Performance trace at item 20 vs item 500; check DOM node count |
| The auto-loading itself is "auto-updating content in parallel with other content" | WCAG 2.2.2 — **no 5-second exception for auto-updating**; see `ada/media-and-motion.md` §4 |

**Accessible alternatives, best first:**

1. **Real pagination** with distinct URLs, `rel="prev"`/`rel="next"` semantics, and a visible page control. Indexable, linkable, restorable.
2. **"Load more" button** appending to the same view — user-initiated, focus moved to the first new item, count announced in a live region, and the footer stays reachable. Combine with paginated URLs so crawlers still see everything.
3. **Infinite scroll + a persistent footer bar** — the compromise. Pin the essential footer links to the viewport, or provide them in the header/nav as well.
4. If infinite scroll stays: paginated URLs behind it, `history.replaceState` to keep the URL in sync with scroll position, a live region for load announcements, and DOM recycling to bound node count.

### 6.2 Smooth-scroll libraries and scroll-jacking

**`scroll-behavior: smooth` is not scroll-jacking.** It is a one-shot easing on a programmatic jump and is fine (gate it behind `no-preference`). Scroll-jacking is taking over the *continuous* scroll gesture.

**Generation 1** (Locomotive ≤4, `smooth-scrollbar`, old fullPage.js): collapse the document to `100vh`, `preventDefault()` on wheel/touch, drive `translate3d`. Catastrophic — no real scrollbar, no find-in-page, broken keyboard, broken screen reader cursor.

**Generation 2** (Lenis, Locomotive v5): the shipped `lenis.mjs` v1.3.26 has **zero occurrences of `transform`**; it calls `wrapper.scrollTo({top, behavior:'instant'})`, i.e. it drives the *real* scroll position. ⚠️ **Articles claiming "Lenis breaks the scrollbar / Ctrl+F / screen readers" describe Gen-1 and are wrong for Lenis ≥1.x and Locomotive ≥5.** Do not copy those claims into a report — verify against the version actually shipped.

| Concern | Gen 1 | Gen 2 (Lenis / Locomotive 5) |
|---|---|---|
| Scrollbar tracks position | ❌ | ✅ real, but **lags** by the lerp |
| Scrollbar drag | ❌ | ⚠️ fought mid-animation ([lenis#107](https://github.com/darkroomengineering/lenis/issues/107)) |
| Find-in-page | ❌ | ✅ |
| Space / PageDown / Home / End | ❌ | ⚠️ native + instant, unsmoothed; cannot interrupt an in-flight animation |
| Keyboard on a custom wrapper | — | ❌ [lenis#356](https://github.com/darkroomengineering/lenis/issues/356) |
| Anchor links / `#hash` | ❌ | ⚠️ **broken BY DEFAULT** — the README requires `anchors: true` |
| Screen reader cursor | ❌ | ✅ |
| CSS `scroll-snap` | ❌ | ❌ needs the separate `lenis/snap` |
| Trackpad momentum | ❌ severe | ⚠️ compounded easing |
| Scroll restoration | ❌ | ⚠️ opt-in `stopInertiaOnNavigate: true` (v1.3.17+) |
| Safari frame rate | — | ⚠️ **60 fps cap, 30 fps in Low Power Mode** ([WebKit #173434](https://bugs.webkit.org/show_bug.cgi?id=173434)) |

**Reduced-motion history — the decisive finding.** Occurrences of `respectReducedMotion` in shipped Lenis bundles:

```
1.1.20 → 0    1.2.3 → 0    1.3.4 → 0    1.3.11 → 0
1.3.18 → 0    1.3.22 → 0   1.3.24 → 0   1.3.25 → 0
1.3.26 → 3    ← FIRST APPEARANCE
```

**Lenis 1.3.26 (2026-08-05) is the first version ever to respect `prefers-reduced-motion`** (`respectReducedMotion`, default `true`). The Framer build (`lenis/framer`, 1.3.25-framer) and **Locomotive v5.0.1 (built on Lenis 1.3.17) still do not.** Any site on an earlier version has an unconditional scroll-motion override for reduced-motion users.

Sizes (measured artifacts): **Lenis** MIT, 1.3.26 — `lenis.min.js` 18,722 B min / **5,431 B gzip** (⚠️ the README's "under 4kb" is stale). **Locomotive Scroll** 5.0.1 (2026-01-15; 7 months without a release) — 36,955 B min / **9,590 B gzip**.

**The usability evidence.** [NN/g Scrolljacking 101](https://www.nngroup.com/articles/scrolljacking-101/) (2023-08-06): *"The majority of our study participants were at least mildly disoriented… users sometimes interpreted the scrolljack as a bug."* Cite **WCAG 2.3.3 (AAA)** for this, not 2.2.2. **Genuinely justified only for WebGL↔DOM synchronisation.**

**What to check if a smooth-scroll library is present:**

| # | Check | Sev |
|---|---|---|
| 1 | Version is **≥1.3.26** for Lenis (first version to respect `prefers-reduced-motion`). `lenis/framer` and Locomotive v5.0.1 still do not — those need a manual `matchMedia` teardown | **P0** |
| 2 | `anchors: true` is set — **anchor links are broken by default** | **P0** |
| 3 | Space / PageDown / Home / End behave correctly; keyboard works on custom wrappers ([lenis#356](https://github.com/darkroomengineering/lenis/issues/356)) | P1 |
| 4 | `stopInertiaOnNavigate: true` for scroll restoration | P1 |
| 5 | `scroll-snap` handled (`lenis/snap` is a separate module) | P2 |
| 6 | Justification exists beyond aesthetics — genuinely WebGL↔DOM sync only | P1 |
| 7 | Bundle cost acknowledged: Lenis 5.4 kB gzip; Locomotive 9.6 kB | P2 |
| 8 | Scrollbar drag is not fought mid-animation; test dragging the bar rapidly | P1 |
| 9 | Safari behaviour verified — 60 fps cap, 30 fps in Low Power Mode makes the lerp feel worse | P1 |

**Verification for all of the above:** actually use the page with keyboard only, with Ctrl+F, with a screen reader, by dragging the scrollbar, and by loading a `#hash` URL cold. None of this is inspectable from source.

---

## 7. When a JS animation library is justified vs bloat

Discuss the **category**, not the brand. The question is always: what does this buy that CSS + IntersectionObserver does not, and is it worth the bytes and the main-thread cost?

| Requirement | Right answer | Cost (measured gzip, Aug 2026) |
|---|---|---|
| Fade / slide in on scroll | IntersectionObserver + a class, or `animation-timeline: view()` | **0 kB** |
| One-liner "reveal when in view" | A micro `inView`-style helper from a modular library | **442 B** |
| Scroll-linked progress value | A modular `scroll`-binding helper | **6.2 kB** |
| Springs, keyframes, staggers | Minimal build of a modular library, escalating to the full build only if needed | **3.0 → 22.2 kB** |
| Declarative component transitions in a framework | The framework-idiomatic package **with lazy-loading of the feature set** | **8.7 vs 41.4 kB** — the lazy variant is ~4.7× smaller |
| Complex timelines, SVG morphing (Safari-safe), draggable | A full timeline library | **~45 kB**, plus licence considerations |
| WebGL ↔ DOM scroll synchronisation | A Gen-2 smooth-scroll library | **5.4 kB** |
| Anything, if the library is abandoned or scroll-listener-based | **Never** | — |

**Concrete measurements to hold vendors to (19 Aug 2026):**

| Artifact | min | **gzip** |
|---|---|---|
| GSAP 3.15.0 `gsap.min.js` core | 72,927 | **28,268 (27.6 kB)** |
| GSAP `ScrollTrigger.min.js` | 44,575 | **17,998 (17.6 kB)** |
| **GSAP core + ScrollTrigger** | 117,502 | **≈45.2 kB** |
| GSAP tree-shaken ESM | 115,212 | 45,115 — **barely tree-shakes** |
| GSAP ScrollSmoother / SplitText | 13,373 / 7,732 | 5,545 / 3,658 |
| GSAP MorphSVG / Draggable / Flip / Observer | 21,195 / 35,762 / 25,534 / 10,014 | 9,553 / 13,490 / 9,706 / 4,320 |
| Motion 13.1.0 `animate` from `motion/mini` | 7,832 | **3,115 (3.0 kB)** |
| Motion `animate` from `motion` | 63,370 | **22,698 (22.2 kB)** |
| Motion `scroll` from `motion` | 15,522 | **6,310 (6.2 kB)** |
| Motion `inView` from `motion` | 715 | **442 B** |
| Motion + `AnimatePresence` (react) | 128,065 | **42,414 (41.4 kB)** |
| `m` + `LazyMotion` + `AnimatePresence` | 22,284 | **8,863 (8.7 kB)** |
| Lenis 1.3.26 | 18,722 | **5,431 (5.4 kB)** |
| Locomotive Scroll 5.0.1 | 36,955 | **9,590 (9.6 kB)** |
| Anime.js 4.5.0, tree-shaken | — | **12,692 B** |

⚠️ **`scroll()` / `inView()` are NOT in `motion/mini` in v13** — importing them pulls the full build. Check which entry point is actually imported, not which package is installed.

**Architecture matters more than size.** Motion routes `opacity`/`transform`/`filter`/`clipPath` through **WAAPI** (so those can be composited); GSAP is **rAF-on-main-thread** for everything. A 45 kB library that runs every tween on the main thread is a different risk profile from a 3 kB one that hands off to WAAPI. Verify which by recording a trace and checking whether the animation appears on the **Animations track** (composited) or as main-thread script work.

**Licence risk is an audit item.** GSAP 3.13.0 (2025-04-30) made everything free: *"100% FREE including ALL of the bonus plugins."* Club GSAP is gone; Webflow acquired GreenSock in Oct 2024. ⚠️ **Free ≠ open source, and it is revocable**: *"Webflow may terminate this GSAP License and revoke your access in its discretion"* ([standard-license](https://gsap.com/community/standard-license/)). Latest is **3.15.0 (2026-04-13)**. Note this in the report where a client's build depends on it.

**Abandonment is a P0.** AOS's last release, `2.3.4`, was published **2018-10-03** — roughly 7 years 10 months with zero releases. ~6.9 kB gzip, **scroll-listener architecture (therefore invisible to Googlebot)**, and **no `prefers-reduced-motion` support at all**. Its presence is a straight P0 finding on three independent grounds. Anime.js 4.5.0 (2026-06-22) is maintained; Theatre.js 0.7.2 (2024-05-19) is dormant.

**The general rule:** if the only requirement is "fade in when scrolled to", any library at all is bloat — the CSS + IntersectionObserver pattern in `../examples/scroll-reveal.md` costs 0 kB and fails open. Escalate only when timelines, interruptible springs, morphing, or WebGL sync are genuinely required, and record the specific requirement in the report so it can be re-litigated later. See `viz-libraries.md` for the data-visualisation equivalent of this decision.

---

## 8. Testable checklist

**P0 = blocking defect, P1 = should fix, P2 = advisory.** Run the P0s first; §8.1 is the highest-value block in the whole skill.

### 8.1 Fail-open / resilience — check first

| # | Check | How to verify | Sev |
|---|---|---|---|
| 1 | No static `opacity: 0`, `visibility: hidden`, `transform: scale(0)`, or `translateY(...)` on content revealed only by JS | `grep -rn "opacity:\s*0" *.css` then confirm each hit is inside `@supports`, a `@keyframes from`, or a JS-applied class | **P0** |
| 2 | Page renders all content with JS disabled | DevTools → Command Menu → "Disable JavaScript", hard reload, scroll the full page | **P0** |
| 3 | Page renders all content when `@supports (animation-timeline: view())` is false | Test in Firefox stable (154), or set `layout.css.scroll-driven-animations.enabled=false` | **P0** |
| 4 | Content is present in the raw HTML | `curl -s URL \| grep -c "<known body phrase>"` — ~28% of crawl traffic (GPTBot, ClaudeBot, PerplexityBot, CCBot) never executes JS | **P0** |
| 5 | Deep link to a far-down `#anchor` in a cold tab shows the target and surrounding content | Load `URL#section-id` in a new tab | **P0** |
| 6 | Scroll restoration / Back-navigation leaves content visible | Scroll deep → navigate away → Back. Check for a `pageshow` / `event.persisted` handler | P1 |
| 7 | Content visible before the reveal script finishes on a slow connection | Network throttling → Slow 4G, reload, scroll immediately | P1 |
| 8 | The hiding class is applied by JS in `<head>` (`document.documentElement.classList.add('js-reveal')`), not shipped in the stylesheet, with a `<noscript>` override | Read the head | P1 |
| 9 | No `opacity: 0` element is keyboard-focusable ("phantom focus") | Tab through the page **before** scrolling; watch for focus disappearing | P1 |
| 10 | Elements already in the viewport on load are visible immediately, with no flash | Record a Performance trace with screenshots; inspect the first 3 frames | P1 |

### 8.2 Scroll reveal implementation

| # | Check | How to verify | Sev |
|---|---|---|---|
| 11 | `animation-timeline` is declared **after** the `animation` shorthand (the shorthand resets it) | grep; verify rule order visually | **P0** if broken |
| 12 | Every named timeline resolves to exactly one element (zero or >1 = silently inactive) | grep `*-timeline-name` against `animation-timeline: --*`; check `timeline-scope` | P1 |
| 13 | No `timeline-scope: all` (**removed in Chrome 138**) | grep | P1 |
| 14 | Scroll-driven animations are not described as "Baseline" — they are **Baseline Limited**, 85.43%, no Firefox stable | Documentation / PR review | P2 |
| 15 | `scroll-timeline-polyfill` is not relied on (latest 1.1.0, 2024-05-15) | Check `package.json` | P1 |
| 16 | IntersectionObserver calls `unobserve()` after reveal | grep `unobserve` vs the number of observed elements; confirm per-frame cost does not climb in a trace | P1 |
| 17 | No non-zero `threshold` combined with a bare `isIntersecting` check (Blink/WebKit implement it as `thresholdIndex > 0`) | Read the callback | P1 |
| 18 | No section taller than the viewport uses `threshold: 0.5`/`1.0` (can never fire) | Measure section heights against an 812 px mobile viewport | P1 |
| 19 | `rootMargin` uses only `px`/`%` (`em`/`rem`/`vh`/`calc()` throw `SyntaxError`), and any percentage author knows **top/bottom resolve against root WIDTH** | grep `rootMargin` | P1 |
| 20 | `'scrollMargin' in IntersectionObserver.prototype` guarded if used (Baseline only since Sept 2025) | grep | P2 |
| 21 | No abandoned scroll-listener-architecture reveal library (e.g. AOS, last release 2018-10-03: invisible to Googlebot, no reduced-motion support) | `package.json` / script tags | **P0** |
| 22 | Reveal animations animate only `transform`/`opacity` | Performance → Animations track: no red triangle | **P0** |
| 23 | Every `animation-timeline` / `scroll-timeline` / `view-timeline` declaration sits inside a `@media (prefers-reduced-motion: no-preference)` block | grep | P1 |
| 24 | Reveal motion that loops or runs >5 s in parallel with other content has a pause mechanism (28 June 2026 Understanding 2.2.2 update: "scrolling an element into view" counts as starting automatically) | Time each; look for a control. `ada/media-and-motion.md` §4 | **P0** |

### 8.3 content-visibility

| # | Check | How to verify | Sev |
|---|---|---|---|
| 25 | `content-visibility: auto` never applied above the fold (it can delay LCP) | Check the first ~2 viewports | P1 |
| 26 | Every `content-visibility: auto` has a `contain-intrinsic-size`, preferably `auto <length>` (not `0px`; **percentages are invalid**) | grep the pairing | P1 |
| 27 | Ctrl+F finds text inside skipped subtrees (broken in Safari 18–25, fixed in 26) | Manual test in **both** Safari 26 and Chrome | P1 |
| 28 | Anchor links and `#:~:text=` fragments land correctly; `.focus()` preferred over a raw `scrollIntoView()` | Manual test | P1 |
| 29 | The scrollbar does not visibly jump while scrolling | Screen-record a full scroll | P1 |
| 30 | Elements hidden by `display:none`/`visibility:hidden` **inside** skipped subtrees also carry `aria-hidden="true"` (styles are not computed, so they leak into the a11y tree) | Accessibility tree inspection | P1 |
| 31 | IntersectionObserver targets inside a skipped subtree still fire (zero-area targets report `isIntersecting: false` in Blink/WebKit) | Manual test | P1 |
| 32 | Measured benefit exists — rendering time actually dropped | Performance panel Recalculate Style + Layout totals, before vs after | P2 |

### 8.4 Infinite scroll

| # | Check | How to verify | Sev |
|---|---|---|---|
| 33 | The footer remains reachable | Scroll to the bottom repeatedly and try to click a footer link | **P0** if legal/contact links are footer-only |
| 34 | Paginated URLs exist behind the infinite scroll and are crawlable | `curl` a page-2 URL; check `<link rel="next">` / real routes (`seo/L1-foundations.md`) | **P0** |
| 35 | Back-navigation restores scroll position without re-fetching from the top | Scroll deep → open an item → Back | P1 |
| 36 | New items are announced in a live region and focus is managed | Screen reader test; Tab after a load | P1 |
| 37 | DOM node count is bounded (recycling), and per-frame cost does not climb with list length | Performance trace at item 20 vs item 500 | P1 |
| 38 | Auto-loading content has a pause/stop mechanism or is user-initiated — **no 5-second exception applies to auto-updating** (WCAG 2.2.2) | Manual; `ada/media-and-motion.md` §4 | **P0** |
| 39 | An accessible alternative is offered (real pagination or a "Load more" button) | Manual | P1 |

### 8.5 Smooth-scroll libraries and scroll-jacking

| # | Check | How to verify | Sev |
|---|---|---|---|
| 40 | If a smooth-scroll library is present, its version honours `prefers-reduced-motion` (Lenis: **≥1.3.26** only; `lenis/framer` and Locomotive v5.0.1 do not) | Read the installed version; Firefox `ui.prefersReducedMotion=1` then scroll | **P0** |
| 41 | Anchor links work (`anchors: true` — **broken by default** in Lenis) | Load a `#hash` URL cold; click an in-page anchor | **P0** |
| 42 | Space / PageDown / Home / End work; keyboard works on custom wrappers | Keyboard only, no mouse | P1 |
| 43 | `stopInertiaOnNavigate: true` for scroll restoration | grep config; Back-navigation test | P1 |
| 44 | Scroll-snap handled if used (`lenis/snap` is separate) | Manual | P2 |
| 45 | Scrollbar drag is not fought mid-animation | Drag the scrollbar rapidly | P1 |
| 46 | Ctrl+F, screen reader cursor, and the real scrollbar all work (confirm the actual generation — Gen-1 claims are wrong for Gen-2 libraries) | Manual, per tool | P1 |
| 47 | Safari behaviour verified (60 fps cap; 30 fps in Low Power Mode) | Test on Safari, then again in Low Power Mode | P1 |
| 48 | A written justification exists beyond aesthetics — genuinely WebGL↔DOM sync. NN/g: *"The majority of our study participants were at least mildly disoriented"* | Ask; record the answer | P1 |
| 49 | No parallax, multi-speed, or direction-mismatched scroll motion (Head's trigger factors; 2.3.3 names parallax explicitly) | Design review; `animation-and-motion.md` §7.6 | P1 |
| 50 | Bundle cost acknowledged and proportionate (Lenis 5.4 kB gzip — the README's "under 4kb" is stale; Locomotive 9.6 kB) | Measure the served artifact, not the README | P2 |

### 8.6 Library justification

| # | Check | How to verify | Sev |
|---|---|---|---|
| 51 | Any animation library present has a stated requirement CSS + IntersectionObserver cannot meet | Ask; record the requirement in the report | P1 |
| 52 | The imported **entry point** is the minimal one (e.g. `scroll()`/`inView()` are **not** in `motion/mini` in v13) | Read the import statements, not `package.json` | P1 |
| 53 | Framework packages use the lazy/partial feature-loading variant where one exists (8.7 kB vs 41.4 kB) | Read the imports | P1 |
| 54 | The library's animations are actually composited — check whether they appear on the Animations track or as main-thread script work | Performance trace | P1 |
| 55 | Licence risk noted where the licence is free-but-revocable | Read the licence terms | P2 |
| 56 | The library is maintained (no multi-year gap since the last release) | npm publish dates | **P0** if abandoned |
