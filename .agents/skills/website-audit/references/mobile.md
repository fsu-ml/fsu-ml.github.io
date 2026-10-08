# Mobile-friendliness: what fails, and how you prove it

**Covers:** the objective, framework-agnostic criteria that separate a mobile-friendly page from a broken one — viewport, reflow, touch targets, typography, responsive technique, mobile-specific interaction, and the verification pass that produces citable evidence.
**Load when:** the audit profile marks mobile as in scope (nearly always). Also load on any "it looks wrong on my phone" complaint, any WCAG 2.1/2.2 AA engagement, and any Core Web Vitals investigation.
**Siblings:** performance depth → `performance.md` · transitions, parallax, reduced-motion → `animation-and-motion.md` · zoom/reflow/focus in the a11y frame → `ada/html-core.md` · form semantics → `ada/html-forms.md` · **colour themes and `forced-colors`** → `code-quality.md` §10 · **RTL and bidirectional text** (`site.multilingual: true` only) → `code-quality.md` §11 · what search engines do with this → `seo/L2-technical-performance.md` · reference implementations → `../examples/responsive-foundations.md` · automation → `../scripts/audit_responsive.py`.

Everything below is stated as something measurable against a **rendered DOM + CSSOM in a controlled emulation environment**, not against source code. You are never auditing a framework; you are auditing what the browser produced.

---

## 0. The framing: there is no verdict any more

"Mobile-friendly" stopped being a pass/fail label issued by Google.

| Retired | Date |
|---|---|
| Mobile-Friendly Test (standalone tool) | **4 December 2023** |
| Mobile-Friendly Test API | **4 December 2023** |
| Search Console → Mobile Usability report | **4 December 2023** |
| Mobile-first indexing rollout | announced complete **31 October 2023**; final stragglers migrated **5 July 2024** |

There is **no replacement mobile-friendly verdict**. Google's stated rationale: mobile-friendliness is now a standard part of web design and better third-party tooling exists. What replaced the single label is a split:

| Concern | Authority now |
|---|---|
| Can a touch user operate it? | **WCAG 2.2 AA** (2.5.8, 1.4.10, 1.4.4, 1.4.12, 1.3.4, 2.4.11) — legally enforceable in the EU/US |
| Does it render and index? | **Mobile-first indexing** — Googlebot Smartphone is the *only* crawl; content parity is the failure mode |
| Is it fast enough? | **Core Web Vitals** field data (CrUX p75), not lab scores |
| Is it *good*? | Apple HIG / Material 3 / platform conventions — advisory, not testable-as-law |

### What you may call a violation

| Tier | What it is | How it appears in a client report |
|---|---|---|
| **Tier 1 — hard failure** | Maps to a named WCAG Success Criterion, or is a deterministic, reproducible platform behaviour | **"Violation of WCAG 2.2 SC x.y.z (Level AA)."** Cite the SC, the measured value, the threshold, the viewport, and the section. Defensible in a demand letter. |
| **Tier 2 — quality signal** | Platform guidance (HIG, Material), performance heuristics, technique quality | **"Recommendation."** Never "violation", never "required". State the source and that it is advisory. |

Mixing the tiers is the single most common way an audit report gets discredited. A finding that says "touch targets must be 44×44 per WCAG" is wrong (see §1.4) and invites the client to dismiss the rest of the document.

---

## 1. Tier 1 — hard failures

Run every recipe under mobile emulation. Run **§5.1's sanity gate first** — if touch/hover emulation is not actually active, every hover and target finding below is invalid.

### 1.1 The meta viewport

**Correct baseline:**
```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```
Add `viewport-fit=cover` **only if** you then handle safe-area insets (§2.5).

| Condition | Why it fails | Threshold | WCAG SC | Rule ID |
|---|---|---|---|---|
| No `meta[name=viewport]` at all | Page renders in the ~980 px desktop fallback viewport; all CSS `px` are scaled down; also reinstates the 300–350 ms tap delay | presence | — | Lighthouse `viewport-insight` (was `viewport`) |
| `width` is a fixed number (e.g. `width=1024`) | Same as above | `width` must equal `device-width` | — | same |
| `user-scalable=no` / `user-scalable=0` | Blocks pinch-zoom | any occurrence | **1.4.4 Resize Text (AA)** | axe `meta-viewport` |
| `maximum-scale` < 2 (usually `=1`) | Caps zoom below 200% | `maximum-scale` ≥ 2 | **1.4.4 (AA)** | axe `meta-viewport` |
| `maximum-scale` < 5 | Caps below 500% | ≥ 5 preferred | — (best practice) | axe `meta-viewport-large` |
| `minimum-scale` > 1 | Prevents zoom-out recovery | ≤ 1 | — | manual |

**⚠ CONTESTED — "iOS ignores `user-scalable=no`, so it's harmless."** iOS Safari has overridden `user-scalable=no` since iOS 10 (2016). But Chrome on Android **does** honour it unless the user has turned on the *Force enable zoom* accessibility setting, and axe-core/WCAG treat it as a flat failure regardless of UA behaviour. **Report it as a failure, always.**

**⚠ Note on `initial-scale`:** `initial-scale=1` is conventional but strictly optional when `width=device-width` is set. Values ≠ 1 are a smell. Do not hard-fail on its absence.

**Detection — paste into the DevTools console:**
```js
(() => {
  const m = document.querySelector('meta[name="viewport" i]');
  if (!m) return { pass: false, errors: ['MISSING_VIEWPORT_META'] };
  const p = {};
  for (const kv of m.content.toLowerCase().split(',')) {
    const [k, v] = kv.split('=').map(s => (s || '').trim());
    if (k) p[k] = v;
  }
  const errors = [], warnings = [];
  if (p.width !== 'device-width') errors.push('WIDTH_NOT_DEVICE_WIDTH:' + p.width);
  if (p['user-scalable'] === 'no' || p['user-scalable'] === '0')
    errors.push('USER_SCALABLE_NO');                                  // WCAG 1.4.4
  const max = parseFloat(p['maximum-scale']);
  if (!isNaN(max) && max < 2) errors.push('MAXIMUM_SCALE_LT_2:' + max); // WCAG 1.4.4
  if (!isNaN(max) && max >= 2 && max < 5) warnings.push('MAXIMUM_SCALE_LT_5:' + max);
  const min = parseFloat(p['minimum-scale']);
  if (!isNaN(min) && min > 1) warnings.push('MINIMUM_SCALE_GT_1:' + min);
  const init = parseFloat(p['initial-scale']);
  if (!isNaN(init) && init !== 1) warnings.push('INITIAL_SCALE_NOT_1:' + init);
  return { content: m.content, parts: p, errors, warnings,
           optedIntoCover: p['viewport-fit'] === 'cover',
           interactiveWidget: p['interactive-widget'] || null,
           pass: errors.length === 0 };
})()
```

### 1.2 Horizontal overflow

The canonical failure. **Gate:** `document.documentElement.scrollWidth > document.documentElement.clientWidth + 1`.

Test at **320 CSS px** (the WCAG 1.4.10 floor), then **360** and **390** (real device widths), then **412**.

Common culprits, in rough order of frequency:

1. A flex or grid **child** with default `min-width: auto` refusing to shrink below its content (long words, long URLs, `<pre>`, `<code>`, tables). Fix: `min-width: 0` / `overflow-wrap: anywhere`.
2. Fixed `width: NNNpx` without `max-width: 100%`.
3. `100vw` used for a full-bleed element on a page with a classic (non-overlay) scrollbar — **`vw` deliberately ignores scrollbars per spec**, so `100vw` > available width. Use `100%`, or `width: calc(100vw - (100vw - 100%))`.
4. Negative margins / absolutely-positioned decorations extending right.
5. `<table>` without a scroll wrapper; `<iframe>` with a hardcoded `width` attribute.
6. Off-canvas menus parked at `left: 100%` instead of `transform: translateX(100%)` + `overflow-x: clip` on a wrapper.

**Detection nuance:** naive scans produce false positives on elements *visually clipped* by an ancestor with `overflow-x: hidden|clip|auto|scroll`. A correct detector walks ancestors.

```js
(() => {
  const de = document.documentElement, limit = de.clientWidth;
  if (de.scrollWidth <= limit + 1) return { overflows: false, culprits: [] };
  const path = el => { const seg = [];
    for (let n = el; n && n.nodeType === 1 && seg.length < 5; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { seg.unshift(s + '#' + n.id); break; }
      if (n.classList.length) s += '.' + [...n.classList].slice(0, 3).join('.');
      seg.unshift(s);
    } return seg.join(' > '); };
  const clipsX = el => /^(hidden|clip|auto|scroll)$/.test(getComputedStyle(el).overflowX);
  const culprits = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const over = Math.max(r.right - limit, -r.left);
    if (over <= 1) continue;
    let clipped = false;
    for (let p = el.parentElement; p && p !== de; p = p.parentElement)
      if (clipsX(p)) { clipped = true; break; }
    if (clipped) continue;                       // contained: not a real overflow
    culprits.push({
      selector: path(el), tag: el.tagName.toLowerCase(),
      rect: { left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width) },
      overflowPx: Math.round(over),
      cssWidth: cs.width, cssMinWidth: cs.minWidth, cssMaxWidth: cs.maxWidth, position: cs.position,
      suspectMinWidthAuto: (cs.minWidth === 'auto' || cs.minWidth === '0px') &&
        /flex|grid/.test(getComputedStyle(el.parentElement || de).display),
      suspect100vw: /100vw/.test(el.getAttribute('style') || '')
    });
  }
  culprits.sort((a, b) => b.overflowPx - a.overflowPx);
  return { overflows: true, scrollWidth: de.scrollWidth, clientWidth: limit,
           excessPx: de.scrollWidth - limit, culprits: culprits.slice(0, 25) };
})()
```

Scope it to one section by replacing `document.querySelectorAll('body *')` with `sectionRoot.querySelectorAll('*')` — see §5.

### 1.3 Reflow — WCAG 1.4.10 (AA)

> Content can be presented without loss of information or functionality, and without requiring scrolling in two dimensions for:
> - Vertical scrolling content at a width equivalent to **320 CSS pixels**
> - Horizontal scrolling content at a height equivalent to **256 CSS pixels**
>
> Except for parts of the content which require two-dimensional layout for usage or meaning.

- **320 px is chosen because it equals 1280 px at 400% zoom.** The criterion is really about desktop zoom, but it is measured identically by narrowing the viewport. Testing at 320 CSS px and testing at 1280 px with `Emulation.setPageScaleFactor(4)` are equivalent by the criterion's own definition.
- The **"two-dimensional layout" exception** legitimately covers: data tables, maps, diagrams, video, games, complex toolbars, code editors with preserved whitespace. It does **not** cover "our hero is 1200 px wide."
- 1.4.10 also implicitly bans content that is *hidden or clipped* at 320 px, not just content that scrolls. A `@media (max-width: …) { .x { display: none } }` that removes information is a 1.4.10 failure, not a responsive technique.

**Harness recipe:** set the viewport to **320 × 512** (`isMobile: true, hasTouch: true`, DPR 2). Run §1.2. Then **diff the visible text content against a 1280 px render** to catch hidden content (loss of information), and screenshot-diff for clipping.

```js
// Run at 1280 and at 320; diff the two strings. Non-trivial loss = 1.4.10 failure.
(() => [...document.body.querySelectorAll('*')]
  .filter(e => { const cs = getComputedStyle(e);
    return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) !== 0; })
  .map(e => (e.childNodes[0]?.nodeType === 3 ? e.childNodes[0].nodeValue.trim() : ''))
  .filter(Boolean).join('\n'))()
```

### 1.4 Target size — the four competing numbers

| Source | Number | Status | Notes |
|---|---|---|---|
| **WCAG 2.2 SC 2.5.8** Target Size (Minimum) | **24 × 24 CSS px** | **Level AA — normative, legally relevant** | 5 exceptions (below) |
| **WCAG 2.1/2.2 SC 2.5.5** Target Size (Enhanced) | **44 × 44 CSS px** | Level **AAA** | Rarely a conformance target; good design floor |
| **Apple HIG** | **44 × 44 pt** | Platform guidance — advisory | pt ≈ CSS px at 1× logical scale |
| **Material Design 3** | **48 × 48 dp** | Platform guidance — advisory | Visual icon may be 24 dp; padding carries the rest |
| **Lighthouse `tap-targets`** (SEO) | **48 × 48 px** | Tooling heuristic | Fails only if < 48 px **AND** ≥ 25% of the target area within 48 px of centre overlaps another target |
| **axe-core `target-size`** | **24 × 24 px** | Automated, maps to 2.5.8 | The correct automated check in 2026 |

**⚠ STALE — "WCAG requires 44×44."** That is 2.5.5, **AAA**. The AA requirement introduced in WCAG 2.2 is **24 × 24**. Auditors conflate these constantly. Report both, separately: *fails AA at 24* (violation) and *below the 44/48 platform recommendation* (advisory).

**The five 2.5.8 exceptions — encode them or the audit is noise:**

1. **Spacing** — an undersized target passes if a **24 px-diameter circle centred on its minimum bounding box** does not intersect any other target, nor the circle of any other undersized target. (W3C's worked example: 20 × 20 buttons with 4 px gaps *pass*; 20 × 20 buttons touching *fail*.)
2. **Equivalent** — the same function is reachable via a conforming control elsewhere on the same page.
3. **Inline** — the target sits within a sentence, or its size is constrained by the line-height of surrounding non-target text. *This exempts most in-prose links.*
4. **User agent control** — native, unstyled UA rendering (e.g. `<input type="date">` internals, default scrollbars). **If the author restyles it, the exception is lost.**
5. **Essential** — map pins, dense data visualisations, legally-mandated form replicas.

Additional normative details:

- The 24 × 24 square must be **axis-aligned and fully inside the target**. Rounded corners can make a nominally-24 px control *undersized*.
- Overlapping targets: the overlapped area does not count toward size — **unless both targets do the same thing**.
- Sliders, colour pickers, and text-editing areas count as **one** target.
- The rule is **zoom-independent**. "The user can zoom in" is not a defence.
- Targets **obscured by author-triggered overlays** (open dropdown, modal, cookie banner) are exempt *while obscured*; targets **inside** the new overlay are in scope.

```js
((MIN = 24) => {
  const SEL = 'a[href], button, input:not([type=hidden]), select, textarea, summary,' +
    '[role=button], [role=link], [role=checkbox], [role=radio], [role=tab],' +
    '[role=menuitem], [role=switch], [role=option], [onclick], [tabindex]:not([tabindex="-1"])';
  const visible = el => { const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.pointerEvents === 'none') return false;
    if (parseFloat(cs.opacity) === 0) return false;
    const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  // Exception 3 "Inline": target sits inside a sentence of non-target text.
  const isInlineInText = el => { const cs = getComputedStyle(el);
    if (!/^inline/.test(cs.display)) return false;
    const p = el.parentElement; if (!p) return false;
    const own = (el.textContent || '').trim().length;
    const all = (p.textContent || '').trim().length;
    return all > own + 20; };
  const targets = [...document.querySelectorAll(SEL)].filter(visible).map(el => {
    const r = el.getBoundingClientRect();
    return { el, r, cx: r.left + r.width / 2, cy: r.top + r.height / 2,
             undersized: r.width < MIN || r.height < MIN, inline: isInlineInText(el) }; });
  const intersectsRect = (cx, cy, rad, r) =>
    Math.hypot(Math.max(r.left - cx, 0, cx - r.right),
               Math.max(r.top - cy, 0, cy - r.bottom)) < rad;
  const findings = [];
  for (const t of targets) {
    if (!t.undersized || t.inline) continue;
    const r0 = MIN / 2;                                   // 24px-diameter circle, radius 12
    const conflicts = targets.filter(o => {
      if (o === t) return false;
      if (o.el.contains(t.el) || t.el.contains(o.el)) return false;
      if (o.undersized) return Math.hypot(o.cx - t.cx, o.cy - t.cy) < MIN;  // circle vs circle
      return intersectsRect(t.cx, t.cy, r0, o.r);                            // circle vs target
    });
    if (conflicts.length === 0) continue;                 // passes via the Spacing exception
    findings.push({ tag: t.el.tagName.toLowerCase(),
      text: (t.el.innerText || t.el.getAttribute('aria-label') || '').trim().slice(0, 60),
      size: [Math.round(t.r.width), Math.round(t.r.height)], conflicts: conflicts.length,
      belowAA24: true,
      belowHIG44: t.r.width < 44 || t.r.height < 44,
      belowMaterial48: t.r.width < 48 || t.r.height < 48 });
  }
  const softWarnings = targets.filter(t =>
    !t.undersized && !t.inline && (t.r.width < 44 || t.r.height < 44)).length;
  return { total: targets.length, aaFailures: findings, below44Advisory: softWarnings };
})()
```

This approximates the normative rule. It does **not** implement the *Equivalent*, *User Agent Control*, or *Essential* exceptions — those need human judgement. Surface maps, canvases, and native `date`/`color`/`file` inputs as **"review"**, not "fail". Sizing pattern that hits 44 px without visually bloating the control: `../examples/responsive-foundations.md`.

### 1.5 Resize text to 200% — WCAG 1.4.4 (AA)

Text must be resizable to **200%** without assistive technology and without loss of content or functionality. Browser **full-page zoom** satisfies this in every major browser, so the real risks are:

| Risk | Check |
|---|---|
| Fixed-height containers with `overflow: hidden` clipping enlarged text | §1.6 clipping detector at 2× root font-size |
| Text baked into images | 1.4.5 / 1.4.9 — see `ada/html-core.md` |
| **Viewport-unit-only font sizing** | §2.2 — `font-size` in `vw` never grows under zoom |
| `maximum-scale` / `user-scalable` | §1.1 |

**Harness recipe:** either set zoom via CDP `Emulation.setPageScaleFactor`, or — more portably — `document.documentElement.style.fontSize = '32px'` (2× the 16 px default) and re-run the overflow and clipping detectors. Any content that disappears, clips, or requires 2-D scrolling is a failure.

```js
(() => { const de = document.documentElement, prev = de.style.fontSize;
  const clipCount = () => [...document.querySelectorAll('*')].filter(e =>
    (e.scrollHeight > e.clientHeight + 2 || e.scrollWidth > e.clientWidth + 2) &&
    /hidden|clip/.test(getComputedStyle(e).overflow)).length;
  const before = { clipped: clipCount(), sw: de.scrollWidth };
  de.style.fontSize = '32px'; void document.body.offsetHeight;
  const after = { clipped: clipCount(), sw: de.scrollWidth };
  de.style.fontSize = prev;
  return { before, after, newlyClipped: after.clipped - before.clipped,
           newHorizontalOverflow: after.sw > before.sw + 1 };
})()
```

### 1.6 Text spacing — WCAG 1.4.12 (AA)

No loss of content or functionality when the user overrides to **all** of:

| Property | Minimum |
|---|---|
| `line-height` | **1.5 ×** font size |
| Spacing after paragraphs | **2 ×** font size |
| `letter-spacing` | **0.12 ×** font size |
| `word-spacing` | **0.16 ×** font size |

Author code using `!important` on those properties blocks user override and is itself a smell — flag it.

```js
(() => {
  const clipped = () => [...document.querySelectorAll('*')].filter(e =>
    e.scrollHeight > e.clientHeight + 2 && /hidden|clip/.test(getComputedStyle(e).overflowY)).length;
  const before = { sw: document.documentElement.scrollWidth, clipped: clipped() };
  const s = document.createElement('style'); s.id = '__wcag1412';
  s.textContent = `* { line-height: 1.5 !important; letter-spacing: 0.12em !important;
                       word-spacing: 0.16em !important; }
                   p { margin-bottom: 2em !important; }`;
  document.head.appendChild(s);
  void document.body.offsetHeight;
  const after = { sw: document.documentElement.scrollWidth, clipped: clipped() };
  s.remove();
  return { before, after, newlyClipped: after.clipped - before.clipped,
           newHorizontalOverflow: after.sw > before.sw + 1 };
})()
```

### 1.7 Orientation — WCAG 1.3.4 (AA)

> Content does not restrict its view and operation to a single display orientation, such as portrait or landscape, unless a specific display orientation is essential.

**"Essential" is narrow:** a piano-keyboard app, a cheque-scanning camera view, a bank-check imaging flow, content targeting a fixed-orientation device. **"It looks better in landscape" is not essential.**

This is a real barrier, not a nicety: users mount devices in fixed orientations (wheelchair mounts), and low-vision users often need landscape to increase text size.

Failure signals: orientation-conditional `display: none` plus a "please rotate your device" overlay; `screen.orientation.lock(...)` calls; web app manifest `"orientation": "portrait"|"landscape"` on a general-purpose site (this also affects installed PWAs); CSS transforms that force-rotate the layout.

```js
(() => { const findings = [];
  for (const sheet of document.styleSheets) {
    let rs; try { rs = sheet.cssRules; } catch { continue; }         // cross-origin
    const walk = list => { for (const r of list) {
      if (r.media && /orientation\s*:/.test(r.conditionText || r.media.mediaText || '')) {
        const text = r.cssText;
        if (/display\s*:\s*none|visibility\s*:\s*hidden/.test(text))
          findings.push({ media: r.conditionText || r.media.mediaText, snippet: text.slice(0, 200) });
      }
      if (r.cssRules) walk(r.cssRules);
    }}; walk(rs);
  }
  return { orientationHidingRules: findings,
    rotatePromptTextFound:
      /rotate your (device|phone)|please rotate|landscape mode required|portrait mode only/i
        .test(document.body.innerText),
    manifestUrl: document.querySelector('link[rel=manifest]')?.href || null, // fetch, check .orientation
    screenOrientationLockAvailable: typeof screen.orientation?.lock === 'function' };
})()
```
Also hook `screen.orientation.lock` **before page load** and record calls; fetch the manifest and read `"orientation"`.

### 1.8 Hover-only affordances

A **hard failure** when the only path to information or function is a mouse hover. Typical patterns:

- Dropdown nav that opens on `:hover` with no click/tap/keyboard handler.
- "Reveal on hover" action buttons in list rows / cards / tables.
- Tooltips carrying non-redundant content, exposed only via `:hover` or `mouseover`.
- Image overlays with the caption or CTA only on `:hover`.

WCAG hooks: **2.1.1 Keyboard**, **1.3.1** / **4.1.2** (info not programmatically available), **1.4.13 Content on Hover or Focus** (hover-revealed content must be dismissible, hoverable, persistent — and reachable by *focus*).

**Related, separate defect — sticky `:hover` on touch.** On iOS and Android, tapping an element applies `:hover` and it **stays applied** until the user taps elsewhere: buttons stuck in their hover colour, submenus that will not close, "hover" overlays permanently covering images. The correct pattern is **additive, never subtractive**:

```css
@media (hover: hover) and (pointer: fine) {
  .card:hover .card__actions { opacity: 1; }
}
```

- `hover` / `pointer` describe the **primary** input mechanism only.
- `any-hover` / `any-pointer` describe the union of **all** available input mechanisms.

**⚠ CONTESTED:** touchscreen laptops and Windows tablets report `pointer: fine, hover: hover` **and** `any-pointer: coarse`. **There is no reliable "is this a touch device" query.** Two consequences: (a) never use `pointer: coarse` as a proxy for "small screen"; (b) never use `@media (hover: none)` to *remove* functionality — always use `@media (hover: hover)` to *add* the enhancement on top of a fully touch-operable baseline.

```js
(() => {
  const VISUAL = /^(display|visibility|opacity|max-height|height|transform|clip-path|pointer-events|content-visibility|width|max-width)$/;
  const rules = [];
  for (const sheet of document.styleSheets) {
    let list; try { list = sheet.cssRules; } catch { continue; }      // cross-origin
    const walk = rs => { for (const r of rs) {
      if (r.cssRules) { walk(r.cssRules); continue; }
      if (r.selectorText) rules.push(r);
    }}; walk(list);
  }
  const hoverRules = rules.filter(r => /:hover\b/.test(r.selectorText));
  const focusSelectors = new Set(rules
    .filter(r => /:focus(-within|-visible)?\b|\[aria-expanded|\[data-(open|expanded|active)|\.(is-|has-)?(open|active|expanded)/.test(r.selectorText))
    .map(r => r.selectorText.replace(/:focus(-within|-visible)?|\[aria-expanded[^\]]*\]|\[data-[^\]]*\]|\.(is-|has-)?(open|active|expanded)/g, '').trim()));
  const findings = [];
  for (const r of hoverRules) {
    const props = [...r.style].filter(p => VISUAL.test(p));
    if (!props.length) continue;                    // ignore decorative colour/shadow hovers
    const base = r.selectorText.replace(/:hover/g, '').trim();
    const hasFocusEquivalent = [...focusSelectors].some(f => f && (f.includes(base) || base.includes(f)));
    const reveals = props.some(p => { const v = r.style.getPropertyValue(p);
      return (p === 'display' && v !== 'none') || (p === 'visibility' && v === 'visible') ||
             (p === 'opacity' && parseFloat(v) > 0) || (p === 'max-height' && v !== '0px') ||
             (p === 'pointer-events' && v === 'auto'); });
    if (reveals && !hasFocusEquivalent) {
      let matched = 0; try { matched = document.querySelectorAll(base).length; } catch {}
      findings.push({ selector: r.selectorText, properties: props, matchedElements: matched,
                      href: r.parentStyleSheet?.href || 'inline' });
    }
  }
  const listenerSuspects = [...document.querySelectorAll('[onmouseover],[onmouseenter]')]
    .filter(el => !el.onclick && !el.getAttribute('onfocus') && !el.getAttribute('ontouchstart'))
    .map(el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : ''));
  return { hoverOnlyRules: findings, mouseOnlyInlineHandlers: listenerSuspects };
})()
```

For richer runtime coverage use CDP `DOMDebugger.getEventListeners` on each interactive node and flag nodes with `mouseover`/`mouseenter` but no `click`/`pointerdown`/`focusin`/`keydown`. **Then confirm behaviourally:** dispatch a real touch sequence (`pointerdown`/`touchstart`/`touchend`/`click`) at the element's centre and assert the target content becomes visible. A CSSOM finding alone is a hypothesis; the touch dispatch is the proof.

### 1.9 iOS input auto-zoom below 16 px

Safari on iOS **automatically zooms the page** when the user focuses an `<input>`, `<select>`, or `<textarea>` whose **computed, rendered** font-size is **< 16 px**. The page does not reliably zoom back out on blur — the user is left zoomed and horizontally scrolled.

**Threshold: 16 CSS px.** No WCAG SC maps to it directly; report it as a **deterministic, reproducible platform defect** (Tier 1 by reproducibility, not by citation), and note that the widespread "fix" for it *is* a WCAG 1.4.4 violation.

- The threshold is the **rendered** size, after inheritance *and after transforms*. `font-size: 16px; transform: scale(0.875)` still triggers it.
- The zoom does *not* fire if `user-scalable=no` / `maximum-scale=1` — which is exactly why that anti-pattern proliferated. **The fix is 16 px, not disabling zoom.**
- Legitimate workaround when the design demands smaller inputs: declare 16 px and shrink visually with `transform: scale()` + compensating negative margins. Note the caveat above about computed rendered size; in practice this works because the *declared* size drives Safari's heuristic in most builds. **Verify on device.**
- `text-size-adjust` does **not** fix this. `-webkit-text-size-adjust: 100%` is still worth setting to stop iOS text inflation in landscape; `none` is discouraged as it can suppress Android text autosizing.

```js
(() => [...document.querySelectorAll(
  'input:not([type=hidden]):not([type=checkbox]):not([type=radio]):not([type=range]):not([type=color]), select, textarea')]
  .filter(el => getComputedStyle(el).display !== 'none')
  .map(el => ({ tag: el.tagName.toLowerCase(), type: el.type || null,
                name: el.name || el.id || null,
                fontSizePx: parseFloat(getComputedStyle(el).fontSize),
                transform: getComputedStyle(el).transform !== 'none'
                  ? getComputedStyle(el).transform : null }))
  .filter(r => r.fontSizePx < 16))()          // hard failure if non-empty
```

### 1.10 Also hard failures — owned by sibling references

| Failure | Threshold | WCAG SC | Where |
|---|---|---|---|
| Focused element **entirely** hidden by sticky header/footer | `scroll-padding-top` on `:root` ≥ sticky header height | **2.4.11 Focus Not Obscured (Minimum), AA — new in 2.2** (2.4.12 AAA forbids even partial obscuring) | §2.6 here; `ada/wcag22-new.md` |
| Personal-data field with no `autocomplete` token | every matching field has a valid token | **1.3.5 Identify Input Purpose (AA)** | §2.9 here; `ada/html-forms.md` |
| Form control with no accessible name | `labels.length \|\| aria-label \|\| aria-labelledby` | **1.3.1 / 4.1.2** | `ada/html-forms.md` |
| Scrollable region unreachable by keyboard (wrapped tables, `overflow: auto` panes) | `tabindex="0"` + accessible name on the wrapper | **2.1.1 Keyboard** | `ada/html-core.md` |
| Motion/parallax with no `prefers-reduced-motion` honour | — | **2.3.3 (AAA) / 2.2.2** | `animation-and-motion.md` |
| **Contrast, clipping or overflow present in only one colour theme** — the site renders a dark theme and it was never audited | run the contrast sweep **and** §1.2/§1.3 once per colour state; every finding names its theme | **1.4.3 / 1.4.11 (AA)**; **1.4.10 (AA)** where the theme changes layout — different font stacks, added borders, swapped images and taller banners all shift the box model | `code-quality.md` §10 (detection, forcing recipes, `color-scheme`, `forced-colors`); `ada/html-core.md` §8 (ratios) |
| Layout breaks under `dir="rtl"` — hardcoded physical CSS, unmirrored affordances, inverted scroll maths | re-render each section with `dir="rtl"` and diff the screenshots | no SC of its own; **1.3.2 / 2.4.3** where the flip desynchronises focus order from visual order | `code-quality.md` §11 — **only when `site.multilingual: true`** |

---

## 2. Tier 2 — quality signals

Advisory. Report as recommendations with a named source. Do not call these violations.

### 2.1 Typography and line length

| Measure | Value | Source |
|---|---|---|
| Body copy floor | **16 CSS px** (browser default; typical phone reading distance 25–35 cm) | Design/readability standard |
| Audit thresholds | **fail** body text < 14 px · **warn** 14–15 px · **pass** ≥ 16 px · secondary/legal/caption ≥ 12 px | Practical |
| Line length, classic | **45–75 characters**, optimum ~**66** | Bringhurst 1992 — a typographer's aesthetic recommendation for printed books, stated without empirical citation **[F]** |
| Line length, screen | ~**55 CPL** for effective reading at normal and fast speeds | Dyson & Haselgrove (2001) **[E]** — and Shaikh & Chaparro (2005) found 95 CPL read *fastest* with no comprehension penalty. The experiments disagree with each other |
| Line length, mobile target | **~35–50 characters** (desktop editorial 60–75) | Practitioner consensus **[C]** — capping measure is a sensible convention; the specific number is not measured |
| Line length hard cap | **80 characters** (40 for CJK) | **WCAG 1.4.8 Visual Presentation (AAA)** — normative, and the only line-length figure in this table that is |

**Line length must not appear as a report finding.** The 45–75/66 number is `[F]` folklore and never ships at any severity (`site-categories.md` §1, `reporting.md` §6); the practice of capping measure is `[C]` convention and may only be offered as an explicitly-labelled recommendation, never as a violation. `site-categories.md` §1 names line length as the canonical example of an indefensible finding. Audit font size, line-height and contrast instead — those have direct WCAG backing (1.4.3, 1.4.4, 1.4.12).

**⚠ STALE / misattributed — "Google requires 16 px."** Google's Lighthouse `font-size` audit threshold was **12 px**, and it flagged only when **≥ 40% of page text** fell below it. That audit was **removed entirely in Lighthouse 13 (10 October 2025, ships in Chrome 143)**, with the stated reason: *"While small fonts is a legibility issue, there are no signals that this remains an SEO concern today."* So 16 px stands on **readability and iOS input-zoom grounds only** — no Google tool checks font size any more.

CSS: `max-width: 60ch` (or `65ch`) on prose containers. `ch` is the width of `0` in the current font — a rough but adequate proxy.

**On mobile the usual failure is the opposite of too-long:** full-bleed text with **zero horizontal padding**, running edge to edge. Check for **≥ 16 px inline padding** on text blocks at 320 px.

Report the *computed* size and the **% of visible text affected**:
```js
(() => { const seen = new Map();
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n; (n = w.nextNode()); ) {
    const t = n.nodeValue.trim(); if (t.length < 4) continue;
    const el = n.parentElement;
    if (!el || /^(SCRIPT|STYLE|NOSCRIPT)$/.test(el.tagName)) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const px = parseFloat(cs.fontSize);
    const key = el.tagName + '|' + px + '|' + cs.fontWeight;
    const rec = seen.get(key) || { tag: el.tagName.toLowerCase(), px, chars: 0, sample: t.slice(0, 60) };
    rec.chars += t.length; seen.set(key, rec);
  }
  const all = [...seen.values()], total = all.reduce((a, r) => a + r.chars, 0) || 1;
  all.forEach(r => r.pctOfText = +(100 * r.chars / total).toFixed(1));
  const small = all.filter(r => r.px < 14).sort((a, b) => b.pctOfText - a.pctOfText);
  return { smallText: small,
           pctTextUnder14: +small.reduce((a, r) => a + r.pctOfText, 0).toFixed(1) };
})()
```

### 2.2 Responsive technique quality

**Breakpoints**

- **Content-driven, not device-driven.** Add a breakpoint where *your layout* breaks, not because a phone model exists. Named-device stacks (`320/480/768/1024/1440`) are a 2013 artifact.
- Quality signal: **fewer** breakpoints, expressed in `rem`/`em` (so they respond to root font-size), placed at odd non-device numbers.
- Poor-quality signal: breakpoints exactly at 320/480/768/1024; `max-width`-only cascades; any `@media` block that **hides** content rather than reflowing it (→ 1.4.10 loss of information, which promotes it to Tier 1).

**Container queries** — Baseline since **February 2023** (Chrome/Edge 105, Safari 16, Firefox 110); widely available in 2026. **Complementary to, not a replacement for, media queries:**

| Use | For |
|---|---|
| Media queries | Page-level structure, plus what containers cannot see: `print`, `orientation`, `prefers-reduced-motion`, `prefers-color-scheme`, `hover`/`pointer` |
| Container queries | Component-level adaptation, so the same component works in a sidebar and a full-width slot |

A component that only responds to viewport width but is reused at multiple widths is a **soft** finding.

**Fluid sizing primitives** — `clamp()`, `min()`, `max()` are all Baseline. Preferred over breakpoint-stepped ladders. Watch: `min()`/`max()` are the inverse of what people expect (`width: min(60ch, 100%)` = "at most 60ch, and never wider than the container").

**⚠ Important, commonly wrong — fluid type and the zoom trap.** `font-size: 4vw` (or a `clamp()` whose *preferred* value is pure `vw`) **fails WCAG 1.4.4** — viewport units are unaffected by browser zoom, so the text never grows. This is Tier 1, not Tier 2. Always include a `rem` term:

```css
font-size: clamp(1rem, 0.75rem + 1.5vw, 2rem);
```

`rem` responds to zoom and to the user's root font-size preference; `vw` only controls the *rate of change* across viewport widths. Heuristic from the fluid-type literature: if **max ≤ 2.5 × min** (both in `rem`), the text can always reach 200% at 500% zoom on modern browsers and passes 1.4.4.

**Fixed px audit:** any element with a computed `width` in px **and** no `max-width` (or a `max-width` also in px, larger than 320); `min-width` in px > 320 on any layout container; flex/grid items missing `min-width: 0`; long unbroken strings without `overflow-wrap: anywhere` / `hyphens: auto`; `<table>` without an `overflow-x: auto` wrapper (and if wrapped, the wrapper needs `tabindex="0"` + an accessible name — WCAG 2.1.1).

**Static CSSOM smell scan** (covers §2.2, §2.3 and §2.5):
```js
(() => { const out = { vh100: [], fontSizeViewportOnly: [], vw100: [],
                       safeAreaUsed: false, dvhUsed: false };
  for (const sheet of document.styleSheets) {
    let rs; try { rs = sheet.cssRules; } catch { continue; }
    const walk = list => { for (const r of list) {
      if (r.cssRules) { walk(r.cssRules); continue; }
      if (!r.style) continue;
      const css = r.cssText;
      if (/env\(\s*safe-area-/.test(css)) out.safeAreaUsed = true;
      if (/\b\d+(\.\d+)?(dvh|svh|lvh|dvi|svi|lvi)\b/.test(css)) out.dvhUsed = true;
      for (const prop of ['height', 'min-height', 'max-height']) {
        const v = r.style.getPropertyValue(prop);
        if (/\b100vh\b/.test(v) && !/\b(dvh|svh|lvh)\b/.test(css))
          out.vh100.push({ sel: r.selectorText, prop, value: v });
      }
      const w = r.style.getPropertyValue('width');
      if (/\b100vw\b/.test(w)) out.vw100.push({ sel: r.selectorText, value: w });
      const fs = r.style.getPropertyValue('font-size');
      if (fs && /\b\d+(\.\d+)?(vw|vi|vmin|vmax)\b/.test(fs) && !/(rem|em)\b/.test(fs))
        out.fontSizeViewportOnly.push({ sel: r.selectorText, value: fs });   // WCAG 1.4.4 risk
    }}; walk(rs);
  }
  return out;
})()
```

### 2.3 Responsive images and CLS prevention

**Required for correctness:**

- `srcset` with `w` descriptors **plus** `sizes` for resolution switching. `srcset` with only `x` descriptors is fine for fixed-size images (icons, avatars) only.
- `<picture>` + `<source type>` for format negotiation (AVIF/WebP) and for **art direction** (different crops per breakpoint). Art direction is the *only* thing `srcset` cannot do.
- **`width` and `height` attributes on every `<img>`.** The browser derives the intrinsic aspect ratio from them and reserves space, eliminating CLS. They set the *ratio*, not the render size; CSS still controls layout (`width: 100%; height: auto`).
- When `<picture>` sources have **different** aspect ratios, the HTML attributes are insufficient — add CSS `aspect-ratio` per source/breakpoint.
- Same for `<video>`, `<iframe>` embeds, and ad/widget slots: reserve space or you ship CLS.

**`loading="lazy"` — when NOT to use it:**

- **Never on the LCP image or anything in the initial viewport.** Lazy-loading defers the request until after layout, directly and severely regressing LCP.
- On the probable LCP image: **no `loading` attribute** (or `loading="eager"`) **+ `fetchpriority="high"`**, and ideally a matching `<link rel="preload" as="image" imagesrcset … fetchpriority="high">` if the image is discovered late (CSS background, or JS-injected).
- `loading="lazy"` + `fetchpriority="high"` on the same element is self-contradictory — the fetch is still deferred until near-viewport, then issued at high priority. Flag it.
- `decoding="async"` is a safe default for non-LCP images.
- **⚠ Newly stale:** Lighthouse's `offscreen-images` audit was **removed in Lighthouse 13**: *"offscreen images are already deprioritized by the browser, so while lazy loading helps reduce bandwidth, it is unlikely to have an impact on what Lighthouse measures."* Lazy-loading below-fold images is still correct for bandwidth on mobile data — just do not expect a score for it.
- **Google Search:** do not lazy-load primary content that requires user interaction to load. Content that only appears after a click or a scroll-triggered fetch may not be indexed.

```js
(() => { const vh = innerHeight, out = [];
  const imgs = [...document.querySelectorAll('img')];
  let lcpGuess = null, lcpArea = 0;
  for (const img of imgs) {
    const r = img.getBoundingClientRect();
    const inViewport = r.top < vh && r.bottom > 0;
    const area = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0)) * r.width;
    if (inViewport && area > lcpArea) { lcpArea = area; lcpGuess = img; }
    out.push({ src: (img.currentSrc || img.src || '').slice(-90), inViewport,
      renderedCss: [Math.round(r.width), Math.round(r.height)],
      intrinsic: [img.naturalWidth, img.naturalHeight],
      hasWidthHeightAttrs: img.hasAttribute('width') && img.hasAttribute('height'),
      cssAspectRatio: getComputedStyle(img).aspectRatio,
      loading: img.getAttribute('loading'), fetchpriority: img.getAttribute('fetchpriority'),
      decoding: img.getAttribute('decoding'),
      hasSrcset: img.hasAttribute('srcset') || !!img.closest('picture')?.querySelector('source[srcset]'),
      hasSizes: img.hasAttribute('sizes'),
      oversuppliedX: r.width ? +(img.naturalWidth / (r.width * devicePixelRatio)).toFixed(2) : null });
  }
  const findings = [];
  for (const i of out) {
    if (!i.hasWidthHeightAttrs && i.cssAspectRatio === 'auto') findings.push(['CLS_NO_RESERVED_SPACE', i.src]);
    if (i.inViewport && i.loading === 'lazy')               findings.push(['LAZY_ABOVE_FOLD', i.src]);
    if (i.loading === 'lazy' && i.fetchpriority === 'high')  findings.push(['CONTRADICTORY_LAZY_HIGH', i.src]);
    if (i.oversuppliedX > 2)                                findings.push(['OVERSIZED_' + i.oversuppliedX + 'x', i.src]);
    if (i.renderedCss[0] > 320 && !i.hasSrcset)             findings.push(['NO_SRCSET_LARGE_IMAGE', i.src]);
    if (i.hasSrcset && !i.hasSizes)                         findings.push(['SRCSET_WITHOUT_SIZES', i.src]);
  }
  if (lcpGuess) {
    if (lcpGuess.getAttribute('loading') === 'lazy') findings.push(['LCP_LAZY_LOADED', lcpGuess.currentSrc]);
    if (lcpGuess.getAttribute('fetchpriority') !== 'high') findings.push(['LCP_NO_FETCHPRIORITY_HIGH', lcpGuess.currentSrc]);
  }
  return { images: out, findings };
})()
```

Pair this with a real `PerformanceObserver` on `largest-contentful-paint` for the authoritative LCP element, and on `layout-shift` (`hadRecentInput === false`) for CLS sources.

### 2.4 Dynamic viewport units and the URL-bar resize problem

The problem: on mobile the address/tab bar retracts on scroll. `vh` is fixed and equals the **large** viewport, so `100vh` overflows the screen on load and only "fits" once the toolbar retracts.

| Unit | Definition (CSS Values 4 / web.dev) |
|---|---|
| `lvh`/`lvw`/`lvmin`/`lvmax`/`lvi`/`lvb` | Viewport with dynamic UA UI **retracted** (largest). `100vh === 100lvh`. |
| `svh`/… | Viewport with dynamic UA UI **expanded** (smallest). |
| `dvh`/… | Live value, clamped between `sv*` and `lv*`. |

Support: **Chrome/Edge 108, Firefox 101, Safari 15.4** → Baseline widely available (all three engines).

**Caveats that matter for an audit:**

- **None of the viewport units account for scrollbars** (per spec). `100vw` overflow is a real, spec-sanctioned bug.
- `dvh` **does not update at 60 fps** — updates are throttled and, on some gestures, debounced. Animating or transitioning to `100dvh` produces jitter. Use fixed values, `svh`, or `lvh` in animations. (See `animation-and-motion.md`.)
- **The virtual keyboard is not part of the UA UI** and therefore does *not* affect any viewport unit, unless you opt in via Chrome's `interactive-widget`.

**Recommended default:** `min-height: 100svh` for "must be fully visible on load" (heroes, splash, login). `100dvh` when the layout must exactly fill the visible area at all times *and* you have verified on-device. **`100vh` on a full-screen element is a defect.** Keep `100vh` as a progressive-enhancement fallback line *before* the `svh`/`dvh` line.

**⚠ CONTESTED — "just replace every `100vh` with `100dvh`."** `dvh` causes reflow-on-scroll for any element sized by it: jank, and a CLS source if it changes layout. `svh` is the safer default for most heroes.

Detection: §2.2 scan (`vh100`, `vw100`, `dvhUsed`).

### 2.5 Safe-area insets (notch / Dynamic Island / home indicator / rounded corners)

- `env(safe-area-inset-top | right | bottom | left)` — **Baseline widely available.**
- They only return non-zero **when `viewport-fit=cover` is set**. Without it, the browser letterboxes you into the safe area automatically — safe, but with visible bars.
- `viewport-fit=cover` is an **opt-in to responsibility**: you get edge-to-edge, and you must then inset fixed/sticky/full-screen elements yourself. **Adding `cover` without adding `env()` handling is a regression, not an improvement.**
- Insets provide **zero margin** — they are exactly the system-UI extent. Always `calc(env(safe-area-inset-bottom) + 1rem)`.
- **Landscape moves the insets to left/right.** Portrait-only handling is a common bug.
- **Desktop always returns 0, and Chrome DevTools responsive/device mode also returns 0.** This is precisely why these bugs ship. Automated audits in headless Chrome **cannot observe** the failure; they can only check whether `env(safe-area-inset-*)` appears in the CSSOM at all when `viewport-fit=cover` is present. Treat as a **static-signal check, not a rendered check**, and say so in the report.
- `safe-area-max-inset-*` (stable value that does not shrink when browser chrome collapses) is **Chromium-only as of 2026**. Use with the `env()` fallback chain. Good for cookie banners and persistent zones; `safe-area-inset-*` is better for a floating button that should hug the home indicator.

**Must be checked for inset handling:** fixed headers/navbars, bottom tab bars, FABs/chat bubbles, full-screen dialogs and drawers, media controls near corners.

Pass rule: `viewport-fit=cover` present ⟺ `env(safe-area-inset-*)` used somewhere. Both, or neither.

### 2.6 Fixed / sticky chrome eating the viewport

Measure the total height of visible `position: fixed|sticky` elements at scroll-top as a **percentage of `100svh`** — the small viewport, the worst case with browser chrome expanded.

| Persistent chrome as % of `svh` | Verdict |
|---|---|
| ≤ 25% | ok |
| > 25% | finding (advisory) |
| > 40% | severe |

On a 667 px-tall iPhone SE with a 60 px browser bar, a 64 px header + 56 px tab bar + 120 px cookie banner leaves **under half the screen** for content.

Two hard correctness items that ride along:

- **`scroll-padding-top` on `:root`** must be ≥ the sticky header height, or in-page anchor links and browser scroll-into-view land *underneath* the header.
- **WCAG 2.4.11 Focus Not Obscured (Minimum) — AA, new in 2.2:** a focused element must not be **entirely** hidden by author content. Sticky headers/footers are the #1 cause. (2.4.12 AAA forbids even partial obscuring.) Browsers scroll focused elements into view but **do not account for your sticky elements** — `scroll-padding` is the fix. This one is Tier 1.

```js
(() => { const vh = innerHeight, fixed = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.position !== 'fixed' && cs.position !== 'sticky') continue;
    const r = el.getBoundingClientRect();
    if (r.height < 8 || r.width < innerWidth * 0.5) continue;      // full-width bars only
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    fixed.push({ tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || null,
      position: cs.position, zIndex: cs.zIndex,
      top: Math.round(r.top), bottom: Math.round(r.bottom), height: Math.round(r.height),
      pctOfViewport: +(100 * r.height / vh).toFixed(1),
      usesSafeArea: /env\(\s*safe-area/.test(el.getAttribute('style') || '') ||
                    cs.paddingBottom !== '0px' });
  }
  const total = fixed.reduce((a, f) => a + f.height, 0);
  const scrollPadTop = getComputedStyle(document.documentElement).scrollPaddingTop;
  const topBarHeight = Math.max(0, ...fixed.filter(f => f.top <= 2).map(f => f.height));
  return { bars: fixed, totalChromePx: total,
    totalChromePctOfViewport: +(100 * total / vh).toFixed(1),
    scrollPaddingTop: scrollPadTop,
    scrollPaddingSufficient: parseFloat(scrollPadTop) >= topBarHeight - 1,   // WCAG 2.4.11
    verdict: total / vh > 0.40 ? 'severe' : total / vh > 0.25 ? 'warn' : 'ok' };
})()
```

### 2.7 Modals and iOS scroll locking

Prefer native **`<dialog>` + `showModal()`**: top layer, `::backdrop`, automatic `inert` on the rest of the document, focus containment, Esc to close. This is the correct 2026 baseline and removes a large class of hand-rolled bugs.

**Scroll locking is still not solved by the platform:**

| Approach | Behaviour |
|---|---|
| `overflow: hidden` on `<body>` | Does **not** reliably block touch scrolling in iOS Safari |
| `position: fixed` on `<body>` | Works, but **resets scroll position** — you must capture and restore `scrollY` |
| `overscroll-behavior: contain` | Prevents scroll **chaining**, not scrolling itself. Will not stop a touch that starts on the backdrop |
| `overscroll-behavior: contain` on both `dialog` and `::backdrop` | Blocks page scroll in recent Chromium (≈ 144). **Other engines have not caught up.** |

Detectable failure modes: background scrolls behind an open modal; scroll position jumps on close; modal taller than `100svh` with no internal scroll; modal content clipped by `100vh` sizing; focus not moved into the dialog; no `inert`/`aria-hidden` on the background.

**Verification is behavioural, not static:** open the modal under touch emulation, dispatch a touch-drag on the backdrop, and assert `window.scrollY` is unchanged; close it and assert `scrollY` returns to its pre-open value.

### 2.8 Virtual keyboard overlap

Default behaviours — all different, all relevant:

- **Chrome on Android:** by default the on-screen keyboard resizes the **visual viewport** only; the layout viewport (and hence `position: fixed` and all viewport units) is unchanged.
- **iOS Safari:** resizes the visual viewport; `position: fixed` elements drift and can end up under or above the keyboard unpredictably. Bottom-anchored bars are the classic casualty.
- **No viewport unit responds to the keyboard** (`svh`/`dvh`/`lvh` included) — the keyboard is not "UA UI".

Tools, in order of portability:

| # | Mechanism | Support |
|---|---|---|
| 1 | **`window.visualViewport`** — `.height`, `.offsetTop`, `.pageTop`, plus `resize`/`scroll` events | **Baseline**, works everywhere including iOS. Portable overlap heuristic: `window.innerHeight - visualViewport.height > ~150` |
| 2 | **`interactive-widget`** in the viewport meta — `resizes-visual` (default), `resizes-content` (resizes the layout viewport too, so `position: fixed` behaves), `overlays-content` | **Chrome 108+, Firefox 132+, no Safari/WebKit as of 2026.** An enhancement, not a solution |
| 3 | **VirtualKeyboard API** — `navigator.virtualKeyboard.overlaysContent = true` + `env(keyboard-inset-top/bottom/height/…)` + `geometrychange` | **Chromium (Android) only** |

**Audit signal:** any `position: fixed` bottom bar or submit button on a page with form inputs, and **no** `visualViewport` listener anywhere in the page's JS, is a probable keyboard-overlap defect.

```js
// Run BEFORE focusing an input; then focus a bottom-of-page input in the harness.
(() => new Promise(res => {
  if (!window.visualViewport) return res({ supported: false });
  const start = { inner: innerHeight, vv: visualViewport.height };
  const on = () => {
    const occluded = innerHeight - visualViewport.height - visualViewport.offsetTop;
    res({ supported: true, start,
          now: { inner: innerHeight, vv: visualViewport.height, offsetTop: visualViewport.offsetTop },
          keyboardLikelyOpen: occluded > 150, occludedPx: Math.round(occluded) });
  };
  visualViewport.addEventListener('resize', on, { once: true });
  setTimeout(() => res({ supported: true, start, keyboardLikelyOpen: false, note: 'no resize fired' }), 1500);
}))()
```
Then, with the keyboard open, re-run §2.6 and check whether the focused input's rect and any bottom-fixed submit button fall inside `visualViewport.offsetTop … offsetTop + visualViewport.height`.

### 2.9 Form inputs: `type`, `inputmode`, `autocomplete`

Three attributes, three jobs:

- **`type`** — validation, UI affordance, submitted value semantics.
- **`inputmode`** — *only* the virtual keyboard layout. No validation, no format change.
- **`autocomplete`** — browser/password-manager autofill. **WCAG 1.3.5 Identify Input Purpose (AA)** requires it on fields collecting information *about the user* — that row is Tier 1. Baymard's research consistently shows autofill has a larger mobile conversion effect than keyboard type.

| Field | `type` | `inputmode` | `autocomplete` | other |
|---|---|---|---|---|
| Email | `email` | `email` | `email` (or `username`) | `autocapitalize="off" autocorrect="off" spellcheck="false"` |
| Phone | `tel` | `tel` | `tel` | |
| URL | `url` | `url` | `url` | |
| Search | `search` | `search` | — | `enterkeyhint="search"` |
| Numeric code / OTP | `text` | `numeric` | `one-time-code` | `pattern="[0-9]*"` |
| Postal code | `text` | `numeric` (locale-dependent) | `postal-code` | |
| Credit card | `text` | `numeric` | `cc-number` | |
| Quantity / true number | `number` | — | — | |
| Price / decimal | `text` | `decimal` | — | |
| Name | `text` | — | `given-name` / `family-name` / `name` | |
| Address | `text` | — | `street-address`, `address-level1/2`, `country-name` | |
| Password | `password` | — | `current-password` / `new-password` | |

**Anti-patterns to flag:**

- `type="number"` on phone numbers, OTPs, postal codes, credit cards, or IDs. It shows a spinner, silently strips leading zeros, mutates on scroll-wheel, and rejects `+`/`-`/spaces. Use `type="text" inputmode="numeric"`.
- Missing `autocomplete` on any recognisable personal-data field → **WCAG 1.3.5 failure** (Tier 1).
- `autocomplete="off"` on address/payment/name fields — mostly ignored by browsers now, and an accessibility negative.
- `<input>` with no `<label>`/`aria-label` (WCAG 1.3.1, 3.3.2, 4.1.2 — Tier 1).
- `enterkeyhint` (`go`/`next`/`send`/`done`/`search`) missing on multi-field forms — cheap mobile win.

```js
(() => { const PERSONAL = /name|email|phone|tel|address|city|state|zip|postal|country|card|cc-|birth|dob|company|organi/i;
  return [...document.querySelectorAll('input:not([type=hidden]), select, textarea')].map(el => {
    const type = (el.getAttribute('type') || el.tagName.toLowerCase()).toLowerCase();
    const name = (el.name || el.id || el.getAttribute('aria-label') || el.placeholder || '');
    const labelled = !!(el.labels?.length || el.getAttribute('aria-label') || el.getAttribute('aria-labelledby'));
    const f = [];
    if (type === 'number' && /phone|tel|zip|postal|code|otp|card|ssn|pin/i.test(name))
      f.push('TYPE_NUMBER_MISUSE__use_text_inputmode_numeric');
    if (/email/i.test(name) && type !== 'email') f.push('EMAIL_WRONG_TYPE');
    if (/phone|tel/i.test(name) && type !== 'tel') f.push('PHONE_WRONG_TYPE');
    if (PERSONAL.test(name) && !el.getAttribute('autocomplete')) f.push('MISSING_AUTOCOMPLETE__WCAG_1_3_5');
    if (el.getAttribute('autocomplete') === 'off' && PERSONAL.test(name)) f.push('AUTOCOMPLETE_OFF_ON_PERSONAL_FIELD');
    if (!labelled) f.push('NO_ACCESSIBLE_NAME');
    if (parseFloat(getComputedStyle(el).fontSize) < 16) f.push('IOS_AUTOZOOM_LT_16PX');
    if (/email|url|search/i.test(name) && el.getAttribute('autocapitalize') !== 'off' && type === 'text')
      f.push('MISSING_AUTOCAPITALIZE_OFF');
    return { name, type, inputmode: el.getAttribute('inputmode'),
             autocomplete: el.getAttribute('autocomplete'),
             enterkeyhint: el.getAttribute('enterkeyhint'), findings: f };
  }).filter(r => r.findings.length);
})()
```

### 2.10 Tap delay and `touch-action` — mostly a stale defect

**⚠ Largely STALE.** The 300–350 ms double-tap-zoom delay was removed in **Chrome 32 (2014)**, Firefox/IE-Edge shortly after, and **iOS Safari 9.3 (March 2016)** — **conditional on the page declaring `width=device-width`**. (Safari also accepts `user-scalable=no`, but that is an a11y failure.)

Where it still matters:

- A page **missing or with an incorrect viewport meta** reinstates the delay. Fold this into §1.1 rather than testing tap latency directly.
- `touch-action: manipulation` remains useful on custom controls (canvas UIs, sliders, tap-heavy grids) to suppress double-tap-zoom without suppressing pinch-zoom. **⚠ STALE advice: "not supported in Safari."** Safari has supported `touch-action` since **13**; that warning is a decade old.
- **`touch-action: none` on a scrollable region is a defect** — it kills native panning.
- **⚠ STALE / actively harmful: FastClick.** Archived library, unnecessary since 2016, causes ghost-click and double-fire bugs on modern browsers. Flag its presence.
- `-webkit-tap-highlight-color: transparent` with no replacement `:active` state removes the only tap feedback the platform gives.
- **⚠ STALE: `-webkit-overflow-scrolling: touch`.** Obsolete since iOS 13, removed from WebKit. Harmless, but indicates unmaintained CSS.

---

## 3. Mobile performance

`performance.md` owns the deep treatment. What follows is only the part that changes *because* the device is mobile.

### 3.1 What "mobile" means in the lab

Lighthouse mobile defaults (from Lighthouse `docs/throttling.md`):

| Setting | Value |
|---|---|
| Network — "Slow 4G" preset | **150 ms RTT, 1.6 Mbps down, 750 Kbps up**, no packet loss |
| Chosen to emulate | ~**85th percentile** mobile connection speed — roughly the bottom 25% of 4G and top 25% of 3G. Historically labelled "Fast 3G"; identical to WebPageTest's "Mobile 3G – Fast" |
| CPU | constant **4× slowdown multiplier** — "moves a typical run in the high-end desktop bracket somewhere into the mid-tier mobile bracket" |
| Device emulation | mid-tier Android (**Moto G Power** in recent versions; Moto G4 historically), narrow viewport, touch enabled, mobile UA |
| Throttling method | **simulated** by default in the CLI, the DevTools Lighthouse panel, and PageSpeed Insights lab data |

Lighthouse's own CPU class table (`benchmarkIndex`, Chrome m86 basis):

| | High-End Desktop | Low-End Desktop | High-End Mobile | Mid-Tier Mobile | Low-End Mobile |
|---|---|---|---|---|---|
| BenchmarkIndex | 1500–2000 | 1000–1500 | 800–1200 | 125–800 | < 125 |
| Speedometer 2.0 | 90–200 | 50–90 | 20–50 | 10–20 | < 10 |
| **JS execution, news site** | **2–4 s** | 4–8 s | 4–8 s | **8–20 s** | **20–40 s** |

That last row is the whole argument: the same bundle costs **2–4 s** on a dev laptop and **8–20 s** on a mid-tier phone, **20–40 s** on a low-end one.

### 3.2 Why desktop Lighthouse misleads

1. **No comparable CPU throttling** — the 4× mid-tier-mobile penalty is simply not applied on the desktop config.
2. **Different network preset** (much higher throughput, lower latency).
3. **Different LCP element.** A wide viewport often makes a different image or heading the LCP candidate, and `srcset` serves a *different file*.
4. **No touch, no `pointer: coarse`** — hover-gated content and touch-only code paths never execute.
5. **Hidden-by-media-query content is still downloaded** — `display: none` does not prevent `<img src>` fetches. Desktop weight ≠ mobile weight, in either direction.
6. **INP is dominated by main-thread contention**, which is a CPU story, so desktop INP is structurally optimistic.

### 3.3 Lab vs field

- **PageSpeed Insights lab always uses simulated throttling**, and in **December 2024 PSI changed its mobile CPU multiplier** (reported as moving from 4× toward ~1.2×) because PSI's own production machines are already slow — the multiplier is calibrated to the host, not absolute. **Consequence: PSI lab scores are not comparable across tools, and not comparable to PSI scores from before that change. Do not trend them.**
- The defensible number is **field data: CrUX / RUM at the 75th percentile over a 28-day window, segmented by form factor.**
- Core Web Vitals "good" thresholds (unchanged in 2026, identical on mobile and desktop):

| Metric | Good | Needs improvement | Poor |
|---|---|---|---|
| **LCP** | ≤ 2.5 s | ≤ 4.0 s | > 4.0 s |
| **INP** | ≤ 200 ms | ≤ 500 ms | > 500 ms |
| **CLS** | ≤ 0.1 | ≤ 0.25 | > 0.25 |

Mobile is materially harder: per HTTP Archive's Web Almanac, roughly **43% of mobile origins** pass all three vs **54% of desktop**.

**TTI is retired**; **TBT** is the lab proxy for INP. `first-meaningful-paint` was removed in Lighthouse 13.

### 3.4 JS budgets

**⚠ These are heuristics, not standards.** Cite them as such.

| Resource | Starting budget, mobile first load |
|---|---|
| JavaScript | **≤ ~150–200 KB compressed** on the critical path (lineage: Alex Russell's "Performance Inequality Gap" work; the older ~170 KB PRPL figure) |
| CSS | **≤ ~50 KB compressed** — the **mobile device-floor** figure. `../references/performance.md` §6 and `code-quality.md` use **≤ ~150 KB**, the general budget. Both are deliberate: 50 KB is what a mid-tier phone on Slow 4G can absorb without pushing TBT past 200 ms; 150 KB is the sitewide ceiling against a median web CSS payload of 82 KB. Audit mobile against 50, the site as a whole against 150, and say which you used. **`performance.md` §6 governs the budget file** — if the two disagree in a report, the budget file wins |
| Total first-load transfer | **≤ ~300–500 KB** |
| DOM | **≤ ~1,400 nodes**, depth ≤ 32 (Lighthouse `dom-size-insight` territory) |

The **objective, non-arbitrary** test is not a byte count:

- **TBT < 200 ms** on the Lighthouse mobile config (Slow 4G, 4× CPU)
- **main-thread work < ~2 s**, script evaluation < ~1 s
- no single **long task > 50 ms** during the first interaction window
- field **INP p75 ≤ 200 ms**

Enforce byte budgets via Lighthouse **LightWallet** (`budget.json` — `resourceSizes`/`resourceCounts`/`timings`) in CI, but treat TBT/INP as the pass criterion.

Mobile-specific amplifiers worth flagging: unbundled legacy transpilation shipped to modern browsers (`legacy-javascript-insight`), duplicated dependencies across chunks (`duplicated-javascript-insight`), third-party tag managers (`third-parties-insight`), hydrating the entire page when only islands are interactive, and web fonts without `font-display: swap|optional`.

### 3.5 Lighthouse's own drift — matters for audit tooling

**Lighthouse 13 (10 October 2025, Chrome 143):**

- `font-size` (legible font sizes) — **removed outright**: *"there are no signals that this remains an SEO concern today. Additionally, this audit was very expensive to run and maintain."*
- `viewport` (best-practices) → replaced by **`viewport-insight`** in Performance.
- `offscreen-images`, `no-document-write`, `uses-passive-event-listeners`, `third-party-facades`, `uses-rel-preload`, `preload-fonts`, `first-meaningful-paint` — removed.
- `prioritize-lcp-image` and `lcp-lazy-loaded` → folded into **`lcp-discovery-insight`**.
- `tap-targets` remains documented under SEO with the legacy heuristic (**< 48 × 48 px AND ≥ 25% of the area within 48 px of centre overlapping another target**) — note it is **not** the WCAG rule.

**Consequence: do not build a mobile audit on Lighthouse's mobile-specific SEO audits.** They are being dismantled. Build on **axe-core `target-size`** (24 px, WCAG 2.5.8), **axe `meta-viewport` / `meta-viewport-large`**, and your own rendered-page checks. axe-core has had `target-size` since **4.5**, and Deque notes it is likely the *only* WCAG 2.2 criterion automatable without excessive false positives.

### 3.6 Search-side consequences (summary only — see `seo/L2-technical-performance.md`)

Mobile-first indexing is done: **Googlebot Smartphone is the crawl**, so whatever the mobile render produces *is* what gets indexed. The parity checklist: same primary content (the #1 real-world failure — truncated mobile copy, collapsed sections not in the DOM, "read more" that fetches on click), same structured data, same `meta robots`, same titles/meta descriptions, same headings, all resources crawlable and renderable, no primary content gated behind interaction, images with stable URLs and the same `alt`. **Intrusive full-screen interstitials on mobile** remain a live negative signal (legally-required cookie/age banners and reasonably-sized banners are exempt). **Responsive design is Google's recommended configuration**; `m.` subdomains and dynamic serving are legacy patterns needing `rel=canonical`/`rel=alternate` pairs and `Vary: User-Agent`.

---

## 4. Advice that is now wrong

Flag these when you see them in a client's existing documentation, a previous vendor's audit, or a developer's justification.

| Claim you will see | Status |
|---|---|
| "Run Google's Mobile-Friendly Test" | **⚠ STALE — dead.** Tool + API retired **4 December 2023** |
| "Fix issues in Search Console's Mobile Usability report" | **⚠ STALE — dead.** Retired same date |
| "Mobile-friendly is a ranking boost" | **⚠ STALE.** Mobile-first indexing means the mobile render *is* the index; the discrete 2015 signal and the SERP label are gone |
| "WCAG requires 44 × 44 touch targets" | **⚠ STALE — wrong level.** **24 × 24 is AA (2.5.8)**; 44 × 44 is **AAA (2.5.5)**; 44 pt is Apple HIG; 48 dp is Material |
| "Google requires 16 px body text" | **⚠ STALE.** Google's audit threshold was **12 px**, and Lighthouse **removed** the `font-size` audit in v13 (Oct 2025). 16 px stands on readability + iOS input-zoom grounds |
| "Use FastClick to kill the 300 ms delay" | **⚠ STALE and harmful.** Library archived; delay gone since 2014–2016 with `width=device-width`; FastClick now causes ghost clicks |
| "`touch-action` isn't supported in Safari" | **⚠ STALE** by ~10 years (supported since Safari 13) |
| "`-webkit-overflow-scrolling: touch`" | **⚠ STALE.** Obsolete since iOS 13; removed from WebKit |
| "iOS ignores `user-scalable=no`, so it's fine" | **⚠ CONTESTED.** Android Chrome honours it; axe/WCAG fail it unconditionally |
| "Just swap every `100vh` for `100dvh`" | **⚠ CONTESTED.** `dvh` reflows on scroll (throttled, debounced) → jank and potential CLS. `svh` is the safer default for heroes |
| "Add `viewport-fit=cover` to every project" | **⚠ CONTESTED.** It is an opt-in to *handling* insets. Without matching `env()` usage it makes content worse |
| "`dvh` fixes keyboard overlap" | **False.** The virtual keyboard is not part of UA UI and affects no viewport unit |
| "Use `interactive-widget=resizes-content`" | Chrome 108+ / Firefox 132+ only; **no Safari**. Still need `visualViewport` JS |
| "Fluid type with `clamp()` is a best practice" | True **only** if the preferred value contains a `rem` term — `vw`-only sizing fails WCAG 1.4.4 |
| "Use 320/768/1024 breakpoints" | Device-driven breakpoints are legacy; content-driven + container queries is the 2026 pattern |
| "Lazy-load all images" | **Never** the LCP/above-fold image. Lighthouse also **removed** `offscreen-images` (browsers already deprioritise offscreen images) |
| "PageSpeed Insights mobile score = mobile reality" | Simulated throttling; PSI's CPU multiplier changed **Dec 2024**, so scores are not comparable across time or tools. Use CrUX p75 field data |
| "Optimise TTI" | **⚠ STALE.** TTI is retired; use **TBT** (lab) → **INP** (field) |
| "`@media (hover: none)` detects touch devices" | **⚠ CONTESTED.** Unreliable on hybrid devices. Use `(hover: hover)` additively; never remove function based on it |
| "Separate `m.` site / dynamic serving" | Legacy. Responsive is Google's recommended configuration |

---

## 5. The verification procedure

**Run this section-by-section, not page-by-page.** A page can pass every page-level gate — no document-level horizontal scroll, viewport meta correct, Lighthouse mobile score respectable — while three of its eleven sections are individually broken. Page-level `scrollWidth` is masked by any ancestor with `overflow-x: hidden`, which most site shells set. Page-level screenshots at 390 px hide a section that only fails at 320 px. **Every finding must name a section**, or it is not actionable and the developer will close it as "works on my phone."

### 5.0 Enumerate the sections first

Build the section list **before** running any check, and reuse the same list at every viewport so findings are comparable.

```js
(() => {
  const roots = [...document.querySelectorAll(
    'main > *, section, article, aside, header, footer, nav, [role=region], [data-section], form')]
    .filter(el => { const r = el.getBoundingClientRect();
      return r.height > 40 && getComputedStyle(el).display !== 'none'; });
  const seen = new Set(), out = [];
  for (const el of roots) {                          // drop nested duplicates
    if ([...seen].some(a => a.contains(el))) continue;
    seen.add(el);
    const r = el.getBoundingClientRect();
    out.push({ id: el.id || null,
      selector: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
        (el.className && typeof el.className === 'string'
          ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : ''),
      heading: el.querySelector('h1,h2,h3')?.innerText.trim().slice(0, 60) || null,
      topPx: Math.round(r.top + scrollY), heightPx: Math.round(r.height) });
  }
  return out;
})()
```

If the markup has no usable sectioning elements, that is itself a finding (`ada/html-core.md` §2 landmarks) — fall back to visual bands captured from a full-page screenshot, and say in the report that section boundaries were assigned manually.

### 5.1 Environment sanity gate — run first, every time

If `emulatingTouch` or `emulatingNoHover` is false, **every hover and target finding below is invalid**. Hover/pointer media queries only resolve to touch values when the emulation sets touch **and** mobile: CDP `Emulation.setTouchEmulationEnabled` + `setDeviceMetricsOverride({ mobile: true })`, or Playwright `hasTouch: true, isMobile: true`.

```js
(() => ({
  layoutWidth: document.documentElement.clientWidth,
  innerWidth: window.innerWidth,
  visualViewport: window.visualViewport
    ? { w: visualViewport.width, h: visualViewport.height, scale: visualViewport.scale } : null,
  dpr: devicePixelRatio,
  emulatingTouch: matchMedia('(pointer: coarse)').matches,
  emulatingNoHover: matchMedia('(hover: none)').matches,
  anyHover: matchMedia('(any-hover: hover)').matches,
  reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches
}))()
```

### 5.2 The viewport matrix

| # | Viewport (CSS px) | DPR | Why this one |
|---|---|---|---|
| V1 | **320 × 512** | 2 | **WCAG 1.4.10 floor.** Equivalent to 1280 px at 400% zoom. Also iPhone SE 1st gen, Galaxy Fold cover screen. If a section fails anywhere, it usually fails here first. |
| V2 | **360 × 640** | 3 | The most common Android logical width worldwide. |
| V3 | **390 × 844** | 3 | iPhone 12/13/14/15/16 baseline logical width — the "it looks fine to me" width. |
| V4 | **412 × 915** | 2.6 | Large Android (Pixel class). Catches layouts tuned only for the 375–390 band. |
| V5 | **568 × 320** (landscape) | 2 | Orientation (**1.3.4**) and the **256 px** horizontal-scroll height floor in 1.4.10. Also moves safe-area insets to left/right. |
| V6 | **768 × 1024** | 2 | Tablet portrait — the band where a desktop layout is commonly jammed in without a breakpoint. |
| V7 | **1280 × 1024 @ 400% page zoom** | 1 | The desktop equivalence for 1.4.10. Confirms V1 findings are zoom findings, not just narrow-screen findings. |
| V8 | **V3 with root `font-size: 32px`** | 3 | **WCAG 1.4.4** at 200%. |

All of V1–V6 with `isMobile: true`, `hasTouch: true`, mobile UA. The performance pass (§3) runs separately at the Lighthouse mobile config: **Slow 4G (150 ms / 1.6 Mbps / 750 Kbps) + 4× CPU.**

### 5.3 The ordered pass

1. **Gate.** Run §5.1. Abort and fix the harness if touch/hover emulation is not active.
2. **Page-level statics, once.** §1.1 viewport meta, §2.2 CSSOM smell scan, §1.7 orientation, §2.5 safe-area static signal. These are document-wide; record once.
3. **Enumerate sections.** §5.0. Freeze the list.
4. **For each viewport V1 → V6, for each section:**
   a. Scroll the section into view. Capture a **section-clipped screenshot**.
   b. Run §1.2 scoped to the section root. Record `overflowPx` and the top culprit selector.
   c. Run §1.4 target size scoped to the section root. Record AA failures and the below-44 advisory count separately.
   d. Run §1.9 and §2.9 if the section contains form controls.
   e. Run §2.1 font-size/measure scoped to the section.
   f. Note any content present at V6 but absent at V1 → candidate **1.4.10 loss of information**.
5. **Reflow content diff.** §1.3 at 320 vs 1280. Attribute each missing string to a section.
6. **V7 (400% zoom).** Confirm the V1 failures reproduce. A failure that reproduces at both is unambiguous 1.4.10.
7. **V8 (200% text).** §1.5. Attribute new clipping to a section.
8. **Text spacing.** §1.6 at V3. Attribute new clipping to a section.
9. **Interaction, per section, under touch emulation:**
   - §1.8 hover-only — CSSOM findings, then **dispatch a real touch sequence** on each candidate and assert the content reveals. Record the before/after.
   - Tab through the section and assert no focused element is **entirely** obscured (**2.4.11**). Record `scroll-padding-top` vs sticky header height (§2.6).
   - Open every modal/drawer the section can trigger; run the §2.7 behavioural scroll-lock test.
   - Focus the last input in the section; run §2.8 and assert the input and any bottom-fixed submit remain inside the visual viewport.
10. **Chrome budget.** §2.6 at V1 (worst case, `svh`). Record the percentage.
11. **Images.** §2.3 at V3. Confirm the LCP element with a real `PerformanceObserver`, not the heuristic guess.
12. **Performance.** Lighthouse mobile config + CrUX p75 field data. See `performance.md`.
13. **On-device confirmation for the three things emulation cannot see:** safe-area insets (DevTools device mode always returns 0), iOS input auto-zoom, and iOS modal scroll-locking. State explicitly in the report which findings were confirmed on a real device and which are static-signal only.

### 5.4 Evidence to record per finding

| Field | Example |
|---|---|
| Section | `section#pricing` — "Plans & pricing" |
| Viewport | 320 × 512, DPR 2, `isMobile: true, hasTouch: true` |
| Check | §1.2 horizontal overflow |
| Measured value | `scrollWidth 412` vs `clientWidth 320` → `excessPx 92` |
| Threshold | `scrollWidth ≤ clientWidth + 1` |
| Culprit | `div.pricing-grid > table.plans` — `cssMinWidth: 380px` |
| Tier / SC | **Tier 1 — WCAG 2.2 SC 1.4.10 Reflow (AA)** |
| Evidence | section-clipped screenshot at 320; console output JSON |
| Reproduced at | V1 (320), V7 (1280 @ 400%) |
| Confirmed on device | no — emulation only |

A finding without a measured value and a threshold is an opinion. Do not ship it as a violation.

---

## 6. Testable checklist

Run under mobile emulation at **320 / 360 / 390 / 412** widths, `isMobile: true`, `hasTouch: true`, `deviceScaleFactor: 2–3`, mobile UA; Slow-4G + 4× CPU for the performance pass. Verify §5.1 before trusting any result. **Every box is answered per section, not per page.**

### Tier 1 — hard failures (WCAG-mapped, defensible)

- [ ] Viewport meta present — §1.1 snippet — `meta[name=viewport]` exists
- [ ] Viewport uses `width=device-width` — §1.1 snippet — exact match, not a fixed number
- [ ] Zoom not disabled — §1.1 snippet — no `user-scalable=no`; `maximum-scale` ≥ 2 (prefer ≥ 5 or absent) — **WCAG 1.4.4**
- [ ] No horizontal scroll at 320 px — §1.2 snippet, scoped per section — `scrollWidth ≤ clientWidth + 1` — **WCAG 1.4.10**
- [ ] No content loss or 2-D scrolling at 320 × 256 — §1.3 text diff 320 vs 1280 + screenshot diff — nothing clipped, hidden or requiring two-axis scroll — **WCAG 1.4.10**
- [ ] Touch targets ≥ 24 × 24 CSS px, or pass the 24 px-circle spacing test — §1.4 snippet — zero unexcepted failures — **WCAG 2.5.8 (AA)** / axe `target-size`
- [ ] Every hover-revealed function has a focus/tap equivalent — §1.8 snippet **plus** a dispatched touch sequence — content reveals on tap — **WCAG 2.1.1, 1.4.13, 1.3.1**
- [ ] Text resizable to 200% with no loss — §1.5 snippet (root `font-size: 32px`) — no new clipping or overflow — **WCAG 1.4.4**
- [ ] Text spacing overrides survive — §1.6 snippet — `newlyClipped === 0` and `newHorizontalOverflow === false` — **WCAG 1.4.12**
- [ ] No `font-size` sized purely in viewport units — §2.2 snippet, `fontSizeViewportOnly` — every fluid `font-size` includes a `rem`/`em` term — **WCAG 1.4.4**
- [ ] No orientation lock — §1.7 snippet + fetch the manifest — no orientation-conditional `display: none`, no `screen.orientation.lock`, no manifest `"orientation"` on a general site — **WCAG 1.3.4 (AA)**
- [ ] Focused elements not entirely hidden by sticky chrome — §2.6 snippet + tab through each section — `scroll-padding-top` ≥ sticky header height — **WCAG 2.4.11 (AA)**
- [ ] Personal-data fields have `autocomplete` — §2.9 snippet — every matching field has a valid token — **WCAG 1.3.5 (AA)**
- [ ] Every form control has an accessible name — §2.9 snippet — `labels.length \|\| aria-label \|\| aria-labelledby` — **WCAG 1.3.1 / 4.1.2**
- [ ] Scrollable regions are keyboard-reachable — manual tab-through — wrapped tables and `overflow: auto` panes have `tabindex="0"` + accessible name — **WCAG 2.1.1**
- [ ] No form control renders below 16 px — §1.9 snippet — result array empty (iOS auto-zoom)
- [ ] **Every colour theme audited, not just the one that loaded** — enumerate states with `code-quality.md` §10.1, then re-run the contrast sweep and §1.2/§1.3 per state — zero failures in each; every finding names its theme — **WCAG 1.4.3 / 1.4.11**, and **1.4.10** where the theme alters layout
- [ ] **RTL layout intact** (only when `site.multilingual: true`) — re-render each section with `dir="rtl"` and diff screenshots, per `code-quality.md` §11.5 — no new overflow, no unmoved elements, focus order still matches visual order (**1.3.2 / 2.4.3**)

### Tier 2 — mobile correctness (recommendations, not violations)

- [ ] Body copy ≥ 16 px — §2.1 snippet — < 5% of visible text below 14 px
- [ ] Text blocks have ≥ 16 px inline padding at 320 px — computed `padding-inline` on prose containers — no edge-to-edge text
- [ ] No full-screen element sized with `100vh` — §2.2 snippet, `vh100` — `svh`/`dvh` used, with `vh` only as a fallback line
- [ ] No `width: 100vw` full-bleed — §2.2 snippet, `vw100` — uses `100%` or a scrollbar-safe calc
- [ ] `viewport-fit=cover` ⟺ `env(safe-area-inset-*)` used — §1.1 + §2.2 snippets — both present, or neither (static signal only; confirm on device)
- [ ] Fixed/sticky elements near screen edges use safe-area insets — §2.6 snippet, `usesSafeArea` — insets in the calc chain
- [ ] Persistent chrome ≤ 25% of `svh` — §2.6 snippet at 320 × 512 — `verdict: 'ok'`
- [ ] Modals: background does not scroll, scroll position restored, modal fits `svh` with internal scroll — §2.7 behavioural test under touch emulation — `scrollY` unchanged while open and restored on close
- [ ] Modals use `<dialog>` + `showModal()`, or provide `inert` + focus trap + Esc — manual — all three present if hand-rolled
- [ ] Bottom-fixed submit buttons reachable with the keyboard open — §2.8 snippet — button rect inside `visualViewport.offsetTop … +height`
- [ ] Correct `type`/`inputmode`; no `type="number"` for phone/OTP/postal/card — §2.9 snippet — zero misuse findings
- [ ] Every `<img>` reserves space — §2.3 snippet — zero `CLS_NO_RESERVED_SPACE`
- [ ] LCP image not lazy, has `fetchpriority="high"` — §2.3 snippet + `PerformanceObserver('largest-contentful-paint')` — zero `LCP_LAZY_LOADED` / `LCP_NO_FETCHPRIORITY_HIGH`
- [ ] No above-fold image with `loading="lazy"` — §2.3 snippet — zero `LAZY_ABOVE_FOLD`
- [ ] Images wider than ~320 CSS px have `srcset` + `sizes` — §2.3 snippet — zero `NO_SRCSET_LARGE_IMAGE` / `SRCSET_WITHOUT_SIZES`
- [ ] No image over-supplied > 2× the needed pixel dimensions — §2.3 snippet — zero `OVERSIZED_*`
- [ ] Hover styles gated behind `@media (hover: hover)` — §1.8 snippet — all visual-reveal `:hover` rules gated
- [ ] `-webkit-tap-highlight-color: transparent` has an `:active`/`:focus-visible` replacement — CSSOM grep — replacement rule exists
- [ ] No `touch-action: none` on scrollable regions — CSSOM grep + manual pan — native panning works
- [ ] No FastClick, no `-webkit-overflow-scrolling: touch` — source/CSSOM scan — zero occurrences
- [ ] No flex/grid children overflowing due to `min-width: auto` — §1.2 snippet, `suspectMinWidthAuto` — zero
- [ ] Breakpoints are content-driven and in `rem`/`em` — CSSOM scan of `@media` conditions — no exact 320/480/768/1024 stack; no `@media` that hides content
- [ ] Reused components adapt by container query where appropriate — CSSOM scan for `@container` — advisory

### Tier 3 — performance, mobile config (see `performance.md`)

- [ ] Lighthouse run on the mobile config — Slow 4G (150 ms / 1.6 Mbps / 750 Kbps) + 4× CPU — this config, never desktop
- [ ] **TBT** — Lighthouse mobile — < 200 ms
- [ ] Lab **LCP** — Lighthouse mobile — < 2.5 s
- [ ] Lab **CLS** — Lighthouse mobile — < 0.1
- [ ] Field **LCP / INP / CLS** — CrUX p75, 28-day window, mobile form factor — ≤ 2.5 s / ≤ 200 ms / ≤ 0.1
- [ ] JS transfer, first load — coverage/network panel — ≤ ~200 KB compressed (heuristic)
- [ ] CSS transfer, first load — coverage/network panel — ≤ ~50 KB compressed (heuristic)
- [ ] Main-thread work — performance trace — < ~2 s; no long task > 50 ms during the first interaction
- [ ] Web fonts — CSSOM — `font-display: swap` or `optional`; preloaded if render-critical
- [ ] Budget enforced in CI — repo scan — Lighthouse LightWallet `budget.json` present

### Tier 4 — search and indexing (see `seo/L2-technical-performance.md`)

- [ ] Mobile render (Googlebot Smartphone UA, JS enabled) contains the full primary content — text diff vs desktop render — ≈ 0 meaningful loss
- [ ] Same structured data on mobile and desktop — JSON-LD diff — identical
- [ ] Same `meta robots` and canonical on both — diff — identical
- [ ] No render-blocking resources disallowed in `robots.txt` (CSS/JS/images) — robots check — zero
- [ ] No primary content gated behind user interaction to load — manual — none
- [ ] Images use stable URLs and have `alt` — crawl twice, diff URLs — zero volatile URLs
- [ ] No intrusive full-screen interstitial on mobile landing — manual, from a search referrer — none
- [ ] Responsive, not `m.` subdomain or dynamic serving; if legacy, `rel=canonical`/`alternate` + `Vary: User-Agent` correct — header + markup check

---

## Primary sources

**W3C / WCAG**
- Understanding SC 2.5.8 Target Size (Minimum) — https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html
- Understanding SC 1.4.10 Reflow — https://www.w3.org/WAI/WCAG22/Understanding/reflow.html
- Understanding SC 2.5.5 Target Size (Enhanced) — https://w3c.github.io/wcag/understanding/target-size-enhanced.html
- Understanding SC 1.3.4 Orientation — https://www.w3.org/WAI/WCAG21/Understanding/orientation.html
- F110: Failure due to sticky footer/header obscuring focus — https://www.w3.org/WAI/WCAG22/Techniques/failures/F110.html
- C42: min-height/min-width for target spacing — https://www.w3.org/WAI/WCAG22/Techniques/css/C42
- CSS Viewport Module Level 1 — https://www.w3.org/TR/css-viewport-1/
- CSS Values 4, viewport-relative lengths — https://www.w3.org/TR/css-values-4/#viewport-relative-lengths

**Google / Chrome / web.dev**
- Mobile-first indexing has landed (Oct 2023) — https://developers.google.com/search/blog/2023/10/mobile-first-is-here
- Mobile-first Indexing Best Practices — https://developers.google.com/search/docs/crawling-indexing/mobile/mobile-sites-mobile-first-indexing
- What's new in Lighthouse 13 — https://developer.chrome.com/blog/lighthouse-13-0
- Lighthouse throttling docs — https://github.com/GoogleChrome/lighthouse/blob/main/docs/throttling.md
- Lighthouse: legible font sizes (removed in LH13) — https://developer.chrome.com/docs/lighthouse/seo/font-size
- Lighthouse: tap targets are not sized appropriately — https://developer.chrome.com/docs/lighthouse/seo/tap-targets
- The large, small, and dynamic viewport units — https://web.dev/blog/viewport-units
- Browser-level image lazy loading — https://web.dev/articles/browser-level-image-lazy-loading
- Optimize resource loading with the Fetch Priority API — https://web.dev/articles/fetch-priority
- Common misconceptions about how to optimize LCP — https://web.dev/blog/common-misconceptions-lcp
- Responsive images (Learn Design) — https://web.dev/learn/design/responsive-images
- Interaction (Learn Design) — https://web.dev/learn/design/interaction
- Incorporate performance budgets into your build process — https://web.dev/articles/incorporate-performance-budgets-into-your-build-tools
- Full control with the VirtualKeyboard API — https://developer.chrome.com/docs/web-platform/virtual-keyboard
- 300ms tap delay, gone away — https://developer.chrome.com/blog/300ms-tap-delay-gone-away
- PageSpeed Insights release notes — https://developers.google.com/speed/docs/insights/release_notes

**MDN**
- `env()` CSS function — https://developer.mozilla.org/en-US/docs/Web/CSS/env
- `inputmode` global attribute — https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/inputmode
- `VirtualKeyboard.overlaysContent` — https://developer.mozilla.org/en-US/docs/Web/API/VirtualKeyboard/overlaysContent
- `Element.scrollWidth` — https://developer.mozilla.org/en-US/docs/Web/API/Element/scrollWidth

**Practitioner / tooling**
- Polypane — Using safe-area-inset to build mobile-safe layouts (May 2026) — https://polypane.app/blog/using-safe-area-inset-to-build-mobile-safe-layouts/
- DebugBear — Simulated throttling in Lighthouse and PSI — https://www.debugbear.com/blog/simulated-throttling
- DebugBear — CPU throttling in Chrome DevTools and Lighthouse — https://www.debugbear.com/blog/cpu-throttling-in-chrome-devtools-and-lighthouse
- Deque — axe-core 4.5: first WCAG 2.2 support (`target-size`) — https://www.deque.com/blog/axe-core-4-5-first-wcag-2-2-support-and-more/
- Deque University — axe rule: meta-viewport — https://dequeuniversity.com/rules/axe/4.4/meta-viewport
- Search Engine Land — Google drops Mobile Usability report and Mobile-Friendly Test — https://searchengineland.com/google-officially-drops-mobile-usability-report-mobile-friendly-test-tool-and-mobile-friendly-test-api-435377
- Bramus — `interactive-widget` viewport resize behavior explainer — https://github.com/bramus/viewport-resize-behavior/blob/main/explainer.md
- HTMHell — Control viewport resize behavior with `interactive-widget` — https://www.htmhell.dev/adventcalendar/2024/4/
- Ahmad Shadeed — The virtual keyboard API — https://ishadeed.com/article/virtual-keyboard-api/
- Ahmad Shadeed — New viewport units — https://ishadeed.com/article/new-viewport-units/
- Smashing — Addressing accessibility concerns with fluid type — https://www.smashingmagazine.com/2023/11/addressing-accessibility-concerns-fluid-type/
- Smashing — A guide to hover and pointer media queries — https://www.smashingmagazine.com/2022/03/guide-hover-pointer-media-queries/
- Adrian Roselli — Responsive type and zoom — https://adrianroselli.com/2019/12/responsive-type-and-zoom.html
- CSS-Tricks — Finding/fixing unintended body overflow — https://css-tricks.com/findingfixing-unintended-body-overflow/
- CSS-Tricks — 16px or larger text prevents iOS form zoom — https://css-tricks.com/16px-or-larger-text-prevents-ios-form-zoom/
- CSS-Tricks — Better form inputs for better mobile UX — https://css-tricks.com/better-form-inputs-for-better-mobile-user-experiences/
- CSS-Tricks — Prevent page scrolling when a modal is open — https://css-tricks.com/prevent-page-scrolling-when-a-modal-is-open/
- Jay Freestone — Locking body scroll for modals on iOS — https://www.jayfreestone.com/writing/locking-body-scroll-ios/
- TPGi — Prevent focused elements from being obscured by sticky headers — https://www.tpgi.com/prevent-focused-elements-from-being-obscured-by-sticky-headers/
- TetraLogical — Foundations: target sizes — https://tetralogical.com/blog/2022/12/20/foundations-target-size/
- Baymard — Touch keyboard types cheat sheet — https://baymard.com/labs/touch-keyboard-types
- Baymard — Readability: the optimal line length — https://baymard.com/blog/line-length-readability
- LogRocket — Container queries in 2026: powerful, but not a silver bullet — https://blog.logrocket.com/container-queries-2026/
