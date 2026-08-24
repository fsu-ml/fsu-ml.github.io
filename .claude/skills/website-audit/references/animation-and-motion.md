# Animation and motion

**Covers:** the rendering pipeline, which properties are cheap and which are not, SVG animation, motion accessibility and restraint, and — most importantly — **how to prove a site actually renders smoothly** rather than asserting it.
**Load when:** the page has any CSS `animation`/`transition`, WAAPI, scroll-linked motion, animated SVG/Lottie/canvas, a carousel, a hover-scale, a parallax layer, or the brief contains the words "smooth", "polished", "buttery", "clean".
**Companion files:** `dynamic-loading.md` (scroll reveals, progressive content), `../examples/scroll-reveal.md`, `../examples/motion-system.md`, `performance.md`, `mobile.md`, `viz-libraries.md`, `ada/media-and-motion.md` (WCAG criterion text — do not restate it here), `seo/L1-foundations.md`, `../scripts/audit_motion.py`.

**Research date of the underlying source material: 19 August 2026.** Verified against Chromium `main`, MDN BCD JSON, caniuse (StatCounter July 2026), W3C/WAI, WebKit release blogs, the Firefox release train, and measured npm/CDN artifacts. Anything marked ⚠️ *unverified* is a practitioner heuristic, not a standard — say so in the report.

**The prime directive of this file:** every quality claim must come with a stated verification method. "The animations are smooth" is not a finding. "Frames track shows 2.1% dropped frames over a 6-second wheel scroll at calibrated 4× CPU throttling" is a finding.

---

## 1. Rendering fundamentals

### 1.1 The pixel pipeline

Classic teaching model (`web.dev/articles/rendering-performance`, updated 2023-12-13):

`JavaScript / CSS → Style → Layout → Paint → Composite`

| Path | Stages executed | Triggered by |
|---|---|---|
| 1 | Style → **Layout** → Paint → Composite | `width`, `height`, `top`, `left`, `margin`, `padding` |
| 2 | Style → **Paint** → Composite | `background-image`, `color`, `box-shadow`, `border-radius` |
| 3 | Style → **Composite** | compositor-only properties |

Paint is two tasks: building a display list, then **rasterization**. "Whenever you see paint records in DevTools, you should think of it as including rasterization."

Modern Chromium (RenderingNG) has 12 stages: **Animate, Style, Layout, Pre-paint, Scroll, Paint, Commit, Layerize, Raster, Activate, Aggregate, Draw**. Pre-paint+Paint map to "Paint"; Commit→Draw map to "Composite". **`Animate` and `Scroll` can run on either thread — that duality is the whole game.**

> "If layout, pre-paint, and paint can be skipped for visual effects, they can be run entirely on the compositor thread and skip the main thread."

### 1.2 Main thread vs compositor thread

**Main thread** — "runs scripts, the rendering event loop, the document lifecycle, hit testing, script event dispatching, and parsing of HTML, CSS."
**Compositor thread** — "processes input events, performs scrolling and animations of web content, computes optimal layerization, and coordinates image decodes, paint worklets and raster tasks."
There is exactly **one main thread and one compositor thread per render process**.

> "**If any animation triggers paint, layout, or both, the main thread will be required to do work.**"

| Cause of main-thread involvement | Mechanism |
|---|---|
| Non-composited animation | Property outside `kCompositableProperties` → Style→Layout/Paint→Commit on main, every frame |
| Non-passive `touchstart`/`touchmove`/`wheel` listener | Compositor round-trips to main and **waits** to learn whether `preventDefault()` was called |
| `scroll` listener doing work | Scroll runs on the compositor; the handler on main lands a frame late |
| `background-attachment: fixed` | Historically forces main-thread scrolling |
| Large paint areas | Saturates raster and CPU↔GPU upload bandwidth |
| Forced synchronous layout, sticky+JS, custom scrollbars | Surfaced by Rendering ▸ Scrolling Performance Issues |

Chrome 56 intervention: "if the target of a `touchstart` or `touchmove` listener is the `window`, `document` or `body` we default `passive` to `true`." Measured effect: "1% of scrolls took just over 400ms… now just over 250ms; **a reduction of about 38%**." ~**80%** of cancelable root-target listeners were "conceptually passive but were not registered as such."

Prefer **`touch-action`** (`none`, or `pan-y pinch-zoom` for a horizontal carousel) over `preventDefault()`. `{ passive: false }` is "Last (and discouraged)." Suppress synthetic clicks with `preventDefault()` in **`touchend`**, not `touchstart`.

### 1.3 Frame budget

> "the browser has **16.66 milliseconds**… **In reality, the browser has its own overhead, so all of your work needs to be completed inside 10 milliseconds.**"

| Refresh rate | Frame period | Realistic budget for your work |
|---|---|---|
| 60 Hz | 16.67 ms | **~10 ms** |
| 90 Hz | 11.11 ms | ~7 ms |
| **120 Hz** (ProMotion / flagship Android) | **8.33 ms** | **~5 ms** |
| 144 Hz | 6.94 ms | ~4 ms |

Compositing alone during scroll costs **~4–5 ms**; web.dev's stated compositing target is "around 4-5ms." ProMotion ranges 10–120 Hz, Android 1–144 Hz, both dropping to 30 Hz on low battery. **Never hardcode 16.67 ms** — there is no `screen.refreshRate` API; derive the interval from the median `requestAnimationFrame` timestamp delta.

`rAF` runs in the event loop's "update the rendering" step, exactly once per frame, immediately before style/layout/paint, with a shared `DOMHighResTimeStamp`; it tracks VRR and 120 Hz and suspends in background tabs. `setTimeout(fn, 16)` has no vsync relationship (drift and beat frequencies), is clamped to ≥4 ms after 5 nestings and ≥1000 ms in background tabs, and hardcodes 60 Hz. **`setTimeout` is always wrong for animation.**

```js
let last = null;
function frame(now) {
  const dt = last === null ? 16.67 : now - last;   // seed only; never assume 60 Hz after
  last = now;
  x += velocity * (dt / 1000);
  el.style.transform = `translate3d(${x}px,0,0)`;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

### 1.4 The definitive compositor-only property list

Literal array in Chromium `main`, `third_party/blink/renderer/core/animation/compositor_animations.cc`:

```cpp
constexpr auto kCompositableProperties = std::to_array<CSSPropertyID>(
    {CSSPropertyID::kBackdropFilter, CSSPropertyID::kFilter,
     CSSPropertyID::kOpacity, CSSPropertyID::kRotate, CSSPropertyID::kScale,
     CSSPropertyID::kTransform, CSSPropertyID::kTranslate,
     CSSPropertyID::kBackgroundColor, CSSPropertyID::kClipPath});
```

Nine properties *can* start on the compositor, with sharply different guarantees:

| Property | Verdict |
|---|---|
| `transform`, `translate`, `rotate`, `scale` | **True compositor-only.** `cc::TargetProperty::TRANSFORM/TRANSLATE/ROTATE/SCALE` |
| `opacity` | **True compositor-only.** `cc::TargetProperty::OPACITY` |
| `filter` | **Composited, hard caveat.** Bails to main if `HasFilterThatMovesPixels()` → sets `kFilterRelatedPropertyMayMovePixels`. **`blur()` and `drop-shadow()` are NOT composited.** `brightness`, `contrast`, `saturate`, `grayscale`, `sepia`, `invert`, `hue-rotate`, `opacity()` are. |
| `backdrop-filter` | **Composited even when pixel-moving** — "pixel moving filters do not change the layer bounds like regular filters do, so they can still be composited." But **very expensive to raster**: it reads back the backdrop every frame. |
| `background-color` | Composited via a native paint worklet, `CompositeBGColorAnimation` **`status: "stable"`**. This is compositor-*paint*, not free like transform. Falls back to main without a `NativePaintImageGenerator`. |
| `clip-path` | **`CompositeClipPathAnimation` = `status: "experimental"`, OFF by default.** In stable Chrome in 2026, animating `clip-path` still costs paint. |

Present in the source but not enabled: `CompositeBoxShadowAnimation` (no `status:` key → test-only).

Additional gates: SVG targets bail for anything other than `transform` (`kSVGTargetHasIndependentTransformProperty`); transform-related properties also require `layout_object->IsTransformApplicable()`.

> **Audit rule.** `transform` + `opacity` are the only two that are compositor-only **across all engines**. Treat non-pixel-moving `filter` and `background-color` as Chromium-only bonuses. Safari 26.4's threaded scroll-driven animation list is `opacity`, `transform`, `translate`, `scale`, `rotate`, `filter`, `backdrop-filter`, and Motion Path — animate only those if you want a cross-engine compositor guarantee.

### 1.5 ⚠️ csstriggers.com is dead — do not cite it

The original Paul Lewis site is gone. The domain is now a **WordPress 7.0.4 content farm** with SEO-spam sidebars, and its remaining data is **wrong** — it claims `box-shadow` "alters the geometry of the element… require the browser to perform layout operations" (false), and repeats the same bogus claim for `border-image-source`. The property list is truncated at `box-sizing` and the per-engine breakdown is gone.

**Paul Irish (gist comment, 2023-10-26):** "tbh generally I think the csstriggers data is mostly a distraction/oversimplification and **you should just use the devtools performance panel instead**."

Authoritative replacements, in order: (1) DevTools Performance panel → Animations track compositing-failure reasons; (2) Chromium `compositor_animations.cc` `kCompositableProperties`; (3) `web.dev/articles/rendering-performance`; (4) Rendering tab Paint Flashing + Layer Borders + Layers panel; (5) LoAF `forcedStyleAndLayoutDuration`.

---

## 2. The decision table: is this animation cheap or expensive?

Apply this to **every** animated property found on the site. Column 3 is the verdict you write in the report; column 4 is how you confirm it rather than guessing.

| Property animated | Deepest pipeline stage triggered | Verdict | Confirm by |
|---|---|---|---|
| `transform`, `translate`, `rotate`, `scale` | Composite | ✅ **Cheap.** Safe for continuous/looping/scroll-linked | Animations track: no red triangle; Layer Borders shows a border on the element |
| `opacity` | Composite | ✅ **Cheap.** Safe for continuous | as above |
| `filter` — `brightness`/`contrast`/`saturate`/`grayscale`/`sepia`/`invert`/`hue-rotate`/`opacity()` | Composite (Chromium + Safari 26.4) | 🟡 **Conditionally cheap**, engine-dependent | Animations track Summary — check for `kFilterRelatedPropertyMayMovePixels` |
| `filter: blur()`, `filter: drop-shadow()` | **Paint** (pixel-moving, explicitly not composited) | ❌ **Expensive.** Never loop or scroll-link | Animations track red triangle; Paint flashing goes green every frame |
| `backdrop-filter` | Composite, but re-reads the backdrop each frame | 🟡 **Composited yet costly.** ≤1–2 per page, never animated | Frame rendering stats → GPU memory; Frames track during scroll |
| `background-color` | Compositor-paint (Chromium `status: stable`) | 🟡 **Chromium-only bonus.** Assume main-thread paint elsewhere | Animations track in Safari/Firefox too |
| `clip-path` | **Paint** (`CompositeClipPathAnimation` experimental, off by default) | 🟡 **Costs paint in stable Chrome 2026** | Animations track |
| `box-shadow` | Paint → Composite; blur makes raster expensive | ❌ **Expensive.** Animate a pseudo-element's `opacity` instead | Paint flashing |
| `border-radius` | Paint; antialiased rounded-rect clipping on every re-raster | ❌ Expensive | Paint flashing |
| `background-position` | Paint; re-rasters the whole background box | ❌ Expensive | Paint flashing |
| `width`, `height`, `top`, `left`, `right`, `bottom`, `margin`, `padding` | **Layout** → Paint → Composite | ❌ **Worst case.** web.dev measured **50% dropped frames vs 1%** for the `transform` equivalent | Frames track red frames; purple Layout bars every frame |
| SVG `cx`, `cy`, `r`, `x`, `y`, `width`, `height`, `viewBox`, `points`, `d` | **Layout** + Paint (invalidates bbox → stroke region, gradient/pattern bbox units, markers, clip/mask, filter region) | ❌ **Worst case.** Orders of magnitude worse than the equivalent `translate()` | Performance trace: Layout on every frame |
| SVG `fill`, `stroke`, `stroke-dashoffset`, `stop-color` | Paint | 🟡 Repaints the dirty rect every frame; acceptable for a short one-shot, not a loop | Paint flashing on the SVG bbox |
| Any CSS custom property `--*` driven by JS and read into a layout property | Whatever the consuming property triggers | ❌ Judge by the consumer, not by `--x` | Read the CSS |
| `height: auto` ↔ fixed, accordion opens | Layout every frame | ❌ Use `grid-template-rows: 0fr → 1fr`, `transform: scaleY()` on a wrapper, or the `interpolate-size`/`calc-size()` route where supported | Frames track |

> web.dev: "over 28 milliseconds is spent inside layout for each frame, which, when we have 16 milliseconds to get a frame on screen in an animation, is far too high."

**Report phrasing.** Do not write "avoid animating width." Write: *"`.card:hover` animates `width` (styles.css:412). Layout runs on every frame; Frames track shows 34% dropped frames on hover at calibrated 4× throttling. Replace with `transform: scaleX()` on a wrapper. Re-verify: Animations track must show no red triangle."*

---

## 3. Layout thrashing / forced synchronous reflow

Source of truth: **Paul Irish, "What forces layout / reflow. The comprehensive list."** — `https://gist.github.com/paulirish/5d52fb081b3570c81e3a` — **"Last active May 14, 2026."**

> "All of the below properties or methods, when requested/called in JavaScript, will trigger the browser to synchronously calculate the style and layout… Generally, all APIs that synchronously provide layout metrics will trigger forced reflow / layout."

| Group | Members |
|---|---|
| Element box metrics | `offsetLeft`, `offsetTop`, `offsetWidth`, `offsetHeight`, `offsetParent`, `clientLeft`, `clientTop`, `clientWidth`, `clientHeight`, `getClientRects()`, `getBoundingClientRect()` |
| Element scroll (also when **setting**) | `scrollBy()`, `scrollTo()`, `scrollIntoView()`, `scrollIntoViewIfNeeded()`, `scrollWidth`, `scrollHeight`, `scrollLeft`, `scrollTop` |
| Other element | `elem.focus()`, `elem.computedRole`, `elem.computedName`, `elem.innerText` |
| Window | `scrollX`, `scrollY`, `innerHeight`, `innerWidth`, `visualViewport.height/width/offsetTop/offsetLeft` |
| Document | `document.scrollingElement` (style only), `document.elementFromPoint` |
| Forms | `inputElem.focus()`, `inputElem.select()`, `textareaElem.select()` |
| Mouse events | `layerX`, `layerY`, `offsetX`, `offsetY` |
| Range | `range.getClientRects()`, `range.getBoundingClientRect()` |
| SVG | `computeCTM()`, `getBBox()`, `getCharNumAtPosition()`, `getComputedTextLength()`, `getEndPositionOfChar()`, `getExtentOfChar()`, `getNumberOfChars()`, `getRotationOfChar()`, `getStartPositionOfChar()`, `getSubStringLength()`, `selectSubString()`, `SVGUse.instanceRoot`. (`getTotalLength()` is absent from the gist body but does force layout — batch it.) |
| Canvas 2D | setting `ctx.filter` (Firefox + Chrome); `ctx.fillText()`/`strokeText()` — **every browser, unconditionally** |

> ⚠️ **Correction (jantimon, Mar 2025, confirmed May 2025):** `window.innerHeight` **does not** trigger reflow in Chromium ≥133 (100,000-iteration benchmark). The gist body still lists it.

**`window.getComputedStyle()`** always forces style recalc, and forces **layout** in three conditions: (1) the element is in a **shadow tree**; (2) viewport media queries exist (`min-width`, `max-width`, `width`, `height`, `aspect-ratio`, `device-pixel-ratio`, `resolution`, `orientation`, …); (3) the requested property is `height`/`width`, `top`/`right`/`bottom`/`left`, `margin*`/`padding*` (only if fixed), `transform`, `transform-origin`, `perspective-origin`, `translate`, `rotate`, `scale`, `grid`, `grid-template*`.

**`element.computedStyleMap()`** — the call itself is free; **`.get()` forces layout** in current Chrome.

> "If layout is forced, style must be recalculated first. So forced layout triggers both operations… `for` loops that force layout & change the DOM are the worst, avoid them."

### 3.1 The read-then-write batching pattern

Bad (web.dev, verbatim) — a read-write-read-write cycle:

```js
function resizeAllParagraphsToMatchBlockWidth () {
  // Puts the browser into a read-write-read-write cycle.
  for (let i = 0; i < paragraphs.length; i++) {
    paragraphs[i].style.width = `${box.offsetWidth}px`;
  }
}
```

Good — read once, outside the loop:

```js
const width = box.offsetWidth;
function resizeAllParagraphsToMatchBlockWidth () {
  for (let i = 0; i < paragraphs.length; i++) {
    paragraphs[i].style.width = `${width}px`;   // Now write.
  }
}
```

Ordering matters just as much for a single element:

```js
// ❌ write-then-read forces layout NOW
box.classList.add('super-big');
console.log(box.offsetHeight);

// ✅ read first — uses last frame's already-computed layout
console.log(box.offsetHeight);
box.classList.add('super-big');
```

Paul Irish's rule: *"Read your metrics at the beginning of the frame (very very start of `rAF`, scroll handler, etc), when the numbers are still identical to the last time layout was done."*

Dependency-free batcher (this is the pattern to recommend; do not recommend a library):

```js
const reads = [], writes = [];
let scheduled = false;
function flush() {
  scheduled = false;
  const r = reads.splice(0), w = writes.splice(0);
  for (const fn of r) fn();   // ALL reads first
  for (const fn of w) fn();   // then ALL writes
}
function schedule() { if (!scheduled) { scheduled = true; requestAnimationFrame(flush); } }
export const measure = fn => { reads.push(fn);  schedule(); };
export const mutate  = fn => { writes.push(fn); schedule(); };
```

**FastDom in 2026:** the gist still recommends it, but the library has been dormant since ~2019. The *pattern* is mandatory; the *library* is not — frameworks batch DOM writes, and `ResizeObserver`/`IntersectionObserver` deliver geometry **without forcing layout**. **Audit rule: FastDom in a 2026 codebase is a smell that geometry is being read imperatively where an Observer would do.**

### 3.2 How thrashing surfaces (this is your verification method)

- Console: `[Violation] Forced reflow while executing JavaScript took <N>ms`.
- Performance panel: **purple bars** = Layout / Recalculate Style. A **warning triangle** on a Recalculate Style/Layout event means script forced it synchronously, with a link to the causing stack frame.
- Chrome 134+: the **Forced reflow insight** in the Insights sidebar names the offending function, its stack, and total reflow time.
- In the field: LoAF `scripts[].forcedStyleAndLayoutDuration` (§7.3).

---

## 4. `will-change` — correct use and its memory cost

MDN (modified 2026-08-04), verbatim: it "enables optimizing animations by providing a **rendering hint**." **Warning: "Use as a last resort… Don't use it to anticipate performance problems."** "Don't apply to too many elements… Overusing can cause the page to slow down." "**Excessive use will result in excessive memory use and will cause more complex rendering.**" It applies to the element's **entire subtree** — on `<body>` it is actively harmful.

The decisive line:

> "add the property **before the animation starts**, not within the `@keyframes`… **Animated properties are treated as if they're already included in a `will-change`, so there's no reason to add them there.**"

**Consequences for an audit:**

1. For **CSS animation / transition / WAAPI of `transform` or `opacity`**, `will-change` is **redundant** — Chromium promotes automatically. This is the single most common misuse and should be flagged wherever it appears.
2. It earns its keep only for **imperative JS-driven mutation** (a `rAF` loop assigning `style.transform`) and for **paint-heavy** subtrees.
3. Dynamic add/remove is correct **only with real lead time** — `mouseenter`/`focusin`/`pointerenter`, i.e. ~100 ms+ ahead. Adding it in the same task that starts the animation is useless and lands the texture allocation inside the INP presentation window.

```js
const el = document.getElementById("element");
el.addEventListener("mouseenter", hintBrowser);
el.addEventListener("animationend", removeHint);
function hintBrowser() { this.style.willChange = "transform, opacity"; }
function removeHint()  { this.style.willChange = "auto"; }
```

⚠️ MDN's own sample writes `"animationEnd"` — that is a bug; DOM event names are case-sensitive. If you see `animationEnd` in a codebase, the cleanup never runs and the layer is permanent.

### 4.1 Layer memory math

```
bytes ≈ cssWidth × cssHeight × devicePixelRatio² × 4     (RGBA8888)
```

| Layer | DPR 1 | DPR 2 | DPR 3 |
|---|---|---|---|
| 300 × 200 card | 240 KB | 960 KB | 2.16 MB |
| 1440 × 900 hero | 5.2 MB | 20.7 MB | 46.7 MB |
| 390 × 844 phone viewport | 1.3 MB | 5.3 MB | 11.8 MB |

**DPR² is the killer** — a 3× device pays **9×**. `* { will-change: transform }` on 500 elements at DPR 3 is a multi-gigabyte ask; web.dev calls this a **"layer explosion."** Chromium tiles large layers and rasters only near-viewport tiles, so treat the formula as an **upper bound / relative-cost heuristic**, and confirm with the real number: **Layers panel → Memory estimate**, and **Chrome 136+ status bar total layer count + total memory**.

### 4.2 ⚠️ `translateZ(0)` / `translate3d(0,0,0)` — outdated and harmful

web.dev still publishes this advice ("Stick to Compositor-Only Properties", last updated **2015-03-20**; "How to create high-performance CSS animations", **2020-10-06**). It is obsolete: **`will-change` has been Baseline widely available since January 2020** (Chrome 36, Edge 79, Firefox 36, Safari 9.1).

Why the hack is actively harmful:

1. A real transform creates a permanent **stacking context** and a **containing block** for `fixed`/`absolute` descendants.
2. It creates a **3D rendering context** — layer explosion by contagion.
3. The layer is pinned **forever**. `will-change` is a hint the UA may drop; a transform must be honoured.
4. It **occupies the `transform` property**, so you cannot animate transform cleanly.
5. **Text rendering degrades** — loses LCD/subpixel antialiasing unless the background is known-opaque.
6. Layers raster at a **fixed scale** → blurry on scale-up.

**Audit rule: flag every `translateZ(0)`, `translate3d(0,0,0)`, `backface-visibility: hidden`, and `perspective: 1000px`-as-hack. Replace with `will-change`, or — usually — delete, because `transform`/`opacity` animations self-promote.**

---

## 5. Scheduling: long tasks, yielding, and INP

> "**Any task that takes longer than 50 milliseconds is a long task.** …total time minus 50 milliseconds is known as the task's **blocking period**." The web is run-to-completion. DevTools marks long tasks with a **red triangle** and the blocking portion in **diagonal red stripes**.

**`scheduler.yield()` — Chrome 129, Edge 129, Firefox 142, Safari ✗.** Its key property: "its **continuation is prioritized**… the continuation of the current task will run *before* any other similar tasks are started." That is precisely what `setTimeout(0)` cannot do.

```js
function yieldToMain () {
  if (globalThis.scheduler?.yield) { return scheduler.yield(); }
  return new Promise(resolve => { setTimeout(resolve, 0); });
}
async function runJobs(jobQueue, deadline = 50) {
  let lastYield = performance.now();
  for (const job of jobQueue) {
    job();
    if (performance.now() - lastYield > deadline) {
      await yieldToMain();
      lastYield = performance.now();
    }
  }
}
```

"A common deadline is **50 milliseconds**." ⚠️ Do not forget the `await`, and beware iterative methods such as `Array.prototype.forEach`. Polyfill: `scheduler-polyfill` on npm.

**`scheduler.postTask()` priorities:** `user-blocking`, `user-visible` (**default**, also the default for `yield()`), `background`. Tasks run in priority order, then insertion order. Options `{ priority, signal, delay }`; setting `priority` makes it **immutable** — it is mutable only if you pass a `TaskSignal` and omit `priority` (then `TaskController.setPriority()` + the `prioritychange` event). `yield()` inside `postTask()` **inherits** the priority.

⚠️ **`isInputPending()` (Chrome 87, Edge 87, Firefox ✗, Safari ✗) is discouraged.** web.dev has a section headed **"Don't use `isInputPending()`"**: "**We no longer recommend using this API**… may incorrectly return `false`… **Input isn't the only case where tasks should yield.**" Not formally deprecated, but flag its use and recommend `scheduler.yield()`.

`setTimeout(0)`: after five nested `setTimeout()`s a **minimum 5 ms delay** applies and the task goes to the **end** of the queue — fallback only. `requestIdleCallback`: `deadline.timeRemaining()` is capped at **50 ms** and returns 0 if input arrives; "It does not run during heavy load" — prefer `postTask(fn, {priority:'background'})`.

### 5.1 INP and animation

p75 of field page loads: **≤ 200 ms Good**, **> 200 and ≤ 500 Needs improvement**, **> 500 Poor**. Chrome ignores one highest interaction for every 50 interactions. Only **click, tap, keypress** count — **scrolling, hovering, and zooming are NOT measured.** (So a site with catastrophic scroll jank can have a perfect INP. See `performance.md`.)

```
[gesture] ──input delay──▶ [handlers] ──processing duration──▶ [style/layout/paint/composite] ──presentation delay──▶ [frame]
```

| Phase | Animation-caused failure |
|---|---|
| **Input delay** | A `rAF` loop or animation-setup long task occupies the main thread at tap time |
| **Processing duration** | Handler synchronously measures geometry (FLIP done wrong, `getBoundingClientRect()` in a loop) → forced style+layout |
| **Presentation delay** | The first frame needs Layout+Paint (`height`/`width`/`top`/`box-shadow` instead of `transform`/`opacity`); or `will-change` added **on click**, forcing texture allocation + raster + GPU upload inside the presentation window |

> "**In order to keep your website's INP as low as possible, it's important to avoid layout when possible.**"

---

## 6. SVG animation

### 6.1 SMIL vs CSS vs WAAPI

**SMIL is NOT deprecated and is safe to ship.** Chrome filed an Intent to Deprecate in 2015 (Chrome 45 console warning), but **the deprecation was suspended and never executed** — confirmed by a 2023 reply on the original blink-dev thread. **caniuse `svg-smil`: 96.42% global.** Chrome 5+, Firefox 4+, Safari 6+, Edge 79+. Chrome usage is ~2.5% of page loads and **rising**.

⚠️ **Outdated claims to strike from any document:** "SMIL is deprecated", "SMIL is being removed from Chrome", "SMIL never worked in Edge".

**The real reason to prefer CSS/WAAPI: SMIL is not GPU-composited.** Nikolas Zimmermann (WebKit engineer), April 2025: *"In WebKit … transform-related animations, such as SMIL won't be hardware accelerated, unlike the CSS Animation counterpart… while it is functional, don't expect performance wonders."* **SMIL is correctness-safe, performance-unsafe.** For anything looping or scroll-linked, use CSS or WAAPI.

| Capability | CSS | WAAPI | SMIL |
|---|---|---|---|
| `transform`, `opacity`, `fill`, `stroke`, `stroke-dashoffset` | ✅ | ✅ | ✅ |
| GPU compositing of transform/opacity | ✅ (Chromium 89+, Firefox, Safari) | ✅ | ❌ |
| Animate arbitrary XML attributes (`viewBox`, `points`, `stdDeviation`, `startOffset`) | ❌ | ❌ | ✅ |
| Path morphing (`d`) | ✅ Chrome/Firefox/Edge — **not Safari** | same limits | ✅ everywhere |
| Declarative sequencing (`begin="a.end+0.5s"`), event triggers | ❌ | ✅ imperatively | ✅ |
| Runs inside `<img src=".svg">` | ✅ (if `<style>` is inline) | ❌ | ✅ |
| Responds to the `prefers-reduced-motion` media query | ✅ | via JS | ❌ |

**Only SMIL can:** (a) animate non-CSS XML attributes, notably `viewBox` and filter-primitive parameters; (b) morph `d` in a way that works in Safari.

SVG2 geometry properties `x`, `y`, `width`, `height`, `cx`, `cy`, `r`, `rx`, `ry` **are** CSS properties in all three engines, so CSS/WAAPI can animate them — but they still force layout. `viewBox`, `points`, `offset`, `stdDeviation`, `pathLength`, `startOffset` are not CSS properties.

`Element.animate()` animates **CSS properties only** — it cannot touch XML attributes. That is a hard limitation of WAAPI, not a browser gap.

### 6.2 Performance characteristics

**Per-frame SVG pipeline:** DOM update → style resolve → **layout (transforms + `viewBox` scaling + fill/stroke-region/bbox computation)** → text shaping → paint-layer generation (gradients, patterns, markers — **each pattern/marker is a whole nested SVG render**) → clip/mask → **filters** → composite. Dirtying anything high in that list forces everything below it.

**Node count.** SVG creates a DOM node per drawing instruction, retained for styling and hit-testing. 2026 benchmarking: smooth to ~1–2k elements, degrading past ~5,000, unusable at 50k+. **Audit rule: >1,500 elements in a single animated SVG → flag; >5,000 → hard fail, recommend canvas.** Note that Lighthouse's "Avoid an excessive DOM size" counts inline SVG children and `<use>`-expanded sprite nodes.

**Attributes vs transform:**

- `transform` / `opacity` — **composited**, no layout or paint. Since **Chromium 89** (Feb 2021) SVG transform/opacity animations are hardware-accelerated by default, matching Firefox.
- `fill`, `stroke`, `stroke-dashoffset`, `stop-color` — **paint only**; repaints the dirty rect every frame.
- `x`, `y`, `width`, `height`, `cx`, `cy`, `r`, `viewBox`, `points`, `d` — **layout + paint**. Changing geometry invalidates the bounding box, which invalidates the stroke region, gradient/pattern bbox units, markers, clip/mask scaling, and any filter region. **`cx`/`cy` animation is orders of magnitude worse than `translate()` for an identical visual result.**

```css
/* BAD — layout every frame */
@keyframes bad { to { cx: 200px; } }
/* GOOD — composited */
@keyframes good { to { transform: translateX(120px); } }
```

**The `transform-box` gotcha (extremely common audit finding).** `transform-box`'s initial value is `view-box`, so an SVG child's transform origin defaults to the **SVG viewport's** (0,0), not the shape's own box. `transform-origin: center` alone does not fix it:

```css
.spinner-blade {
  transform-box: fill-box;      /* required */
  transform-origin: center;
  animation: spin 1s linear infinite;
}
```

⚠️ **Outdated:** "Chrome never divides an SVG graphic into different GPU layers" (O'Reilly ch.19, pre-2021) and the accompanying advice to stack separate `<svg>` elements as manual layers. Superseded by Chromium 89.

**`will-change: transform` on SVG** works (it promotes in Chromium and Firefox) with caveats: one layer per element, so broad application (`svg * { will-change: transform }`) is a documented anti-pattern; it creates a stacking context that can reorder painting and break `mix-blend-mode` on groups; a promoted layer rasterizes once and is GPU-scaled, so an animated `scale()` can end up blurry. Safari-specific: `will-change` sometimes *drops* the composited layer for the subtree, and composited SVG layers are sometimes rasterized at 1× and scaled, producing soft edges.

**Rasterization cost of scaling.** SVG is resolution-independent in *source* only. A composited layer holds a raster at one scale, so animating `scale()` either re-rasterizes each frame (sharp, CPU-expensive) or GPU-scales a cached bitmap (cheap, blurry); browsers choose heuristically. **Scaling a path-heavy SVG from 0.1 → 1 is the worst case.** Prefer `opacity` + `translate`, or pre-scale so the animated range is small.

**Filters.** Any change to a filtered element or its children forces the **entire filter graph to re-run**; filters defeat compositing. `feGaussianBlur` cost scales with `stdDeviation` × filter-region area; `feTurbulence` and `feDisplacementMap` are worse; `feMorphology` with a large radius is pathological. The default filter region is `x="-10%" y="-10%" width="120%" height="120%"` — **always tighten it.** CSS `filter: blur()` is generally cheaper than an SVG `<filter>` for the same visual. **Audit rule: any `<filter>` on an element that animates every frame → flag; animating `stdDeviation` → hard flag.** Masking is more expensive than clipping; Firefox has a known bug where CSS-animated content inside a `<mask>` does not update. Preferred pattern: pre-bake statically-masked layers and animate their `opacity`.

### 6.3 Line drawing (`stroke-dasharray` / `stroke-dashoffset`)

**Prefer the `pathLength` normalization trick — no JS required.** `pathLength` tells the browser to pretend the path is N units long; all dash values then scale automatically.

```html
<path d="M10 80 C 40 10, 65 10, 95 80" pathLength="1"
      fill="none" stroke="currentColor" stroke-width="4" class="draw"/>
```
```css
.draw {
  stroke-dasharray: 1;
  stroke-dashoffset: 1;
  animation: draw 1.2s ease forwards;
}
@keyframes draw { to { stroke-dashoffset: 0; } }

@media (prefers-reduced-motion: reduce) {
  .draw { animation: none; stroke-dashoffset: 0; }   /* substitute: show the finished line */
}
```

This is robust against path edits, responsive scaling, and floating-point drift. **Prefer it over `getTotalLength()` in every new build.**

```js
const p = document.querySelector('path');
const len = p.getTotalLength();           // SVGGeometryElement.getTotalLength()
p.style.strokeDasharray  = len;
p.style.strokeDashoffset = len;
```

⚠️ Outdated: "only `<path>` supports `getTotalLength()`" — that was SVG 1.1. It now lives on `SVGGeometryElement`, so `<circle>`, `<rect>`, `<line>`, `<polyline>`, `<polygon>` work too. It **forces layout** — batch all calls, never call it inside a rAF loop. It returns **user units**, not CSS px.

**Performance:** `stroke-dashoffset` is a **paint** property — it repaints the stroke region every frame and is never composited. Long, thick, complex, or blurred strokes are the expensive case. Multiple simultaneous draw-ins are a common jank source; stagger with `animation-delay`. Firefox renders a 1-unit dash on a closed path as a dot at the start — use `stroke-dasharray: 0.999 1` if you see that artifact. **Audit rule: >~10 concurrently drawing paths, or any dash animation on a filtered/masked element → flag.**

### 6.4 Path morphing (`d`)

Browser support for CSS/WAAPI `d` animation (caniuse `mdn-svg_elements_path_d_path`, Aug 2026):

| Browser | Status |
|---|---|
| Chrome | ✅ 52+ (current: 154) |
| Edge | ✅ 79+ |
| Firefox | ✅ 97+ (current: 157) |
| **Safari / iOS Safari** | ❌ **not supported through Safari 27 / iOS 26.6** |
| **Global** | **79.89%** |

WebKit has had it behind a flag / in STP since roughly 2024–2025 and has repeatedly cited performance as the blocker. **Not Baseline, not production-safe cross-browser in 2026.**

```css
/* Chrome/Firefox/Edge only — Safari renders the static `d` and never animates */
@supports (d: path("M0 0")) {
  .blob { transition: d 400ms ease; }
  .blob:hover { d: path("M20 20 C ..."); }
}
```

**Structural requirement (all techniques):** interpolation is per-number and pairwise, so both `d` strings need the **same number of path commands, in the same order, of the same type**. `M C C C Z` → `M C C C Z` works; `M C C Z` → `M L L L Z` snaps. Convert everything to all-cubic with identical command counts.

**Libraries — discuss the category, not the brand.** A morph plugin that **writes the `d` attribute from JS** works in Safari, where CSS `d` does not; that is the only reason to reach for one. GSAP's MorphSVGPlugin became free with GSAP 3.13 (30 April 2025, Webflow "No Charge" license): it auto-normalizes command counts, handles differing segment counts, supports `shapeIndex` for rotation artifacts, converts `<circle>`/`<rect>`/`<polygon>` automatically, and writes the attribute — hence the pragmatic 2026 default. `flubber` (~4 KB) is the dependency-light alternative for arbitrary shapes including topology changes.

**Audit position: treat `d` morphing as an expensive, non-composited effect regardless of technique. Acceptable for a sub-second interaction; flag it for anything looping.**

### 6.5 Delivery method

| Method | Internal CSS/SMIL animates? | External CSS applies? | `currentColor` | JS access | HTTP caching |
|---|---|---|---|---|---|
| **Inline `<svg>`** | ✅ | ✅ full | ✅ | ✅ | ❌ cached with the HTML |
| **`<img src="x.svg">`** | ✅ if `<style>`/SMIL is *inside the file* | ❌ | ❌ | ❌ | ✅ separate cacheable file |
| **`<use href="sprite.svg#id">`** | ✅ inside the sprite | ⚠️ shadow boundary: only **inherited** properties (`fill`, `stroke`, `color`) and **custom properties** cross | ✅ | ⚠️ limited | ✅ one request |
| **`<object>` / `<iframe>`** | ✅ including scripts | ❌ | ❌ | ✅ same-origin | ✅ |
| **CSS `background-image`** | ✅ internal only | ❌ | ❌ | ❌ | ✅ |

**`<img>` is not "no animation."** A self-contained SVG with an internal `<style>` block or SMIL animates perfectly inside `<img>`, `background-image`, and `<use>`. It simply cannot be themed or controlled from the page. **This is the cheapest way to ship a decorative loop.**

For `<use>` sprites, hardcode nothing: use `fill="currentColor"` or expose `var(--icon-color, currentColor)` — custom properties are the only reliable styling channel through the shadow tree. Data URIs in CSS must be URL-encoded (base64 adds 33%) and kept under ~4 KB.

### 6.6 When to abandon SVG

| Replace with | When |
|---|---|
| **Plain CSS on HTML elements** | Spinners, pulses, bars. A `border-radius` + `transform` div beats a 40-node SVG. |
| **CSS `filter`, `clip-path`, `mask-image`** | Simple effects — these sit on the accelerating path |
| **Lottie / dotLottie** | Complex After Effects output. Costs: `lottie-web` ≈ **60 KB gzip** runtime, JSON 15–40 KB for what SVG does in 1–5 KB, and the SVG renderer mutates DOM on the **main thread** every frame. Prefer `dotlottie-web` (WASM/ThorVG; `.lottie` is ≈30–50% smaller) or the canvas renderer. **Never more than 2–3 concurrent players; never autoplay one above the fold.** |
| **Video / animated WebP / AVIF** | Photographic, >5 s, gradient/blur-heavy, or thousands of nodes. Hardware-decoded, off-main-thread. Requires `muted playsinline` + `poster`, and must pause under reduced motion. |
| **Canvas / WebGL / state-machine runtimes** | >5,000 marks, particle systems, state-machine-driven interactivity. See `viz-libraries.md`. |

### 6.7 SVG accessibility

```html
<!-- Informative -->
<svg role="img" aria-labelledby="chart-t chart-d" viewBox="0 0 100 100">
  <title id="chart-t">Revenue by quarter</title>
  <desc id="chart-d">Bar chart; Q4 is highest at 4.2 million.</desc>
  ...
</svg>

<!-- Decorative -->
<svg aria-hidden="true" focusable="false" ...>
```

`role="img"` collapses the subtree so AT announces one image instead of walking every shape — **required**; without it, AT mapping is inconsistent (W3C *SVG Accessibility API Mappings*). `<title>` must be the **first child** of the element it names. `aria-labelledby` + `aria-describedby` has the widest AT support; `aria-label` is a fine simpler alternative — but do not use both. For `<img src="x.svg">`, use `alt` instead.

**`focusable="false"` in 2026:** an IE/EdgeHTML-era attribute; **no engine shipping in 2026 needs it.** It costs one attribute, is still recommended by Orange's accessibility guidelines and most icon libraries, and SVGO will not strip it. **Audit stance: not a defect either way — do not flag its presence or absence.** Flag the actual modern bug instead: `tabindex="0"` on a non-interactive SVG.

Interactive SVG children need `tabindex="0"`, `role="button"`, a keyboard handler, and a **visible focus indicator** — `outline` on SVG elements is unreliable across engines, so use `stroke` or `filter: drop-shadow()` on `:focus-visible`. They must also meet the 24×24 CSS px target minimum (WCAG 2.5.8 — see `ada/media-and-motion.md` and `ada/targets.md`).

⚠️ **SMIL does not respond to media queries.** `display: none` is not an acceptable fix. Use the SVG DOM timeline API:

```js
const mq = matchMedia('(prefers-reduced-motion: reduce)');
const apply = () => document.querySelectorAll('svg').forEach(svg => {
  if (mq.matches) { svg.pauseAnimations(); svg.setCurrentTime(0); }
  else svg.unpauseAnimations();
});
mq.addEventListener('change', apply); apply();
```

For WAAPI: `document.getAnimations().forEach(a => a.cancel())`.

### 6.8 SVGO optimisation

**Current: `svgo@4.0.2`** (Node ≥16, ESM-first, `.mjs`/`.cjs` config). v4 shipped mid-2025.

⚠️ **v4 changed `preset-default`:** `removeViewBox` and `removeTitle` were **removed** from the default set. But **`removeDesc` is still in preset-default** — it silently strips `<desc>`, an accessibility regression. Disable it.

```js
// svgo.config.mjs
export default {
  multipass: true,
  floatPrecision: 2,               // 3 is the safe default; 2 for icons
  js2svg: { indent: 0, pretty: false },
  plugins: [
    { name: 'preset-default',
      params: { overrides: {
        removeViewBox: false,      // already off in v4; explicit = future-proof
        removeDesc:    false,      // a11y
        cleanupIds:    false,      // REQUIRED for <use>, gradients, filters, clip-path, sprites
        inlineStyles:  false,      // REQUIRED if the file has CSS animations/media queries
        minifyStyles:  false,      // ditto — can drop @media (prefers-reduced-motion)
      }}},
    'removeDimensions',            // drop width/height, keep viewBox → fluid scaling
    { name: 'addAttributesToSVGElement',
      params: { attributes: [{ 'aria-hidden': 'true' }, { focusable: 'false' }] } }, // decorative only
  ],
};
```

⚠️ Outdated: "SVGO removes your viewBox by default" (no longer true in v4; keeping the override is harmless); "SVGO config is `module.exports`" (v4 is ESM).

**Precision:** **2 decimals** is safe for icons and UI graphics (24–48 px viewBoxes) and visually lossless. **3** is safe for essentially all artwork. **1 or 0** visibly deforms curves. **Never reduce precision on paths targeted by morphing or `getTotalLength()`** — rounding changes total length and can collapse near-coincident control points.

**Embedded base64 rasters — the #1 cause of huge SVGs.** An `<image xlink:href="data:image/png;base64,...">` is a raster smuggled in as XML text, +33% for base64, and **not compressible** by gzip/brotli. A 4 KB "vector icon" becomes 2 MB. **Audit rule: grep for `data:image/` inside any `.svg`. Any hit >10 KB → fail.** Also watch for outlined text (Illustrator "Convert to Outlines" turns a 200-byte string into 40 KB of paths).

**`viewBox` is mandatory** for anything that scales or is delivered via `<img>`/`background-image` — without it there is no intrinsic aspect ratio. Keep `viewBox`, drop `width`/`height`, size from CSS — then supply explicit CSS dimensions or `aspect-ratio` to avoid CLS. `preserveAspectRatio="none"` is almost always a bug.

| Asset | Green | Flag | Fail |
|---|---|---|---|
| Single icon (optimized) | < 2 KB | 2–10 KB | > 10 KB |
| Logo / spot illustration | < 15 KB | 15–50 KB | > 50 KB |
| Full-width hero illustration | < 50 KB | 50–150 KB | > 150 KB |
| `<use>` sprite (all icons) | < 30 KB | 30–80 KB | > 100 KB |
| Any SVG containing `data:image/` | — | any | > 10 KB embedded |
| Data URI in CSS | < 2 KB | 2–4 KB | > 4 KB |

Additional flags: element count >1,500; any single `d` attribute >10 KB (usually a traced raster or a map that belongs in TopoJSON/canvas); missing gzip/brotli on `.svg` responses — SVG is XML and compresses 60–80%, and servers frequently omit `image/svg+xml` from the compressible-MIME list.

---

## 7. Motion accessibility and restraint

**Criterion text lives in `ada/media-and-motion.md`.** Do not restate 2.2.2 / 2.3.1 / 2.3.3 / 1.4.2 here — cross-reference it. What follows is the *motion-engineering* delta: the parts of those criteria that change how animation code must be written, plus the 2026 updates.

| SC | Level | The motion-engineering consequence | Full text |
|---|---|---|---|
| **1.4.2** Audio Control | A | Threshold is **3 seconds**, not 5. Autoplaying video *with sound* must be checked against **both** 1.4.2 (audio) and 2.2.2 (visual motion). OS mute does not count. | `ada/media-and-motion.md` §3 |
| **2.2.2** Pause, Stop, Hide | A | See §7.1 below — the 28 June 2026 Understanding update pulls **scroll-triggered motion into Level A**. | `ada/media-and-motion.md` §4 |
| **2.3.1** Three Flashes or Below Threshold | A | ≤3 flashes/second **automatically passes, no tool needed**. Above that, PEAT or Harding. | `ada/media-and-motion.md` §5 |
| **2.3.3** Animation from Interactions | AAA | **Parallax is named explicitly** in the Understanding doc. `prefers-reduced-motion` satisfies it via **C39** (CSS) / **SCR40** (JS). An in-page control must be a **site-wide setting**. | `ada/media-and-motion.md` §6 |
| **2.5.8** Target Size (Minimum) | AA | Targets must be ≥24×24 CSS px **throughout** motion — a hover-scale growing from an undersized rest state fails. Auto-advancing carousels compound 2.5.8 with 2.2.2 and 2.2.1. | `ada/targets.md` |

**Status notes.** WCAG 2.2 is a W3C Recommendation, **republished 12 December 2024** (originally Rec 5 October 2023). **`4.1.1 Parsing` is "(Obsolete and removed)"** — flag any audit template still testing it. **WCAG 3.0 is still a Working Draft (03 March 2026) — do not audit against it.**

**Legal.** The **European Accessibility Act — Directive (EU) 2019/882**, compliance date **28 June 2025**, is in force and enforceable across Member States as of August 2026, public *and* private sector, **WCAG Version Used: WCAG 2.2** — covering e-commerce, banking, transport, e-books, telecoms. Directive (EU) 2016/2102 (public sector) still cites **WCAG 2.1 AA** via **EN 301 549 v3.2.1** (March 2021, still the newest *published* version; v4.1.0 drafts exist Nov 2025 and Jun 2026, none final). **Net position: 2.2.2 (A), 2.3.1 (A), 1.4.2 (A), 2.5.8 (AA) are legally required in the EU; 2.3.3 is AAA and is not** — but 2.3.3 is the criterion that addresses parallax and scroll-jacking, and `prefers-reduced-motion` is the cheapest way to satisfy it. Auditing to WCAG 2.2 AA satisfies both directives.

### 7.1 🔴 The 28 June 2026 update to Understanding 2.2.2

> *"…starts automatically either when it starts without direct user activation or interaction… or when it starts as a result of an indirect interaction (such as focusing/hovering over an element, or **scrolling an element into view**). Content that starts automatically from an indirect interaction also potentially fails 2.3.3."*

**Consequence: scroll-triggered reveals and hover-triggered motion can fail 2.2.2 at Level A** if they run longer than 5 seconds in parallel with other content. A 300 ms one-shot reveal is not caught; an infinite scroll-linked decorative loop is the clear failure. This is new, it is legally material in the EU, and most audit templates predate it.

Two further points that catch people out:
- **The 5-second exception applies ONLY to moving/blinking/scrolling.** W3C: *"there is no five second exception for auto-updating."* Live stock/score/notification feeds need a control immediately.
- **Focus/hover-to-pause is NOT a valid mechanism.** W3C: *"Having an animation stop only so long as a user has focus on it… would not be considered a 'mechanism for the user to pause'."*
- **Non-interference (Conformance Requirement 5)** applies to all content, including third-party ads and embeds. One un-pausable ad ticker fails the page.
- The spec endorses a site-wide toggle: *"a **single** mechanism to pause, stop, hide … that affects all these elements at the same time."*
- **Preloader exception:** a full-screen loading spinner is fine (not in parallel with other content) even beyond 5 s.

### 7.2 `prefers-reduced-motion`, done properly

```css
@media (prefers-reduced-motion: reduce)        { /* user asked for less motion */ }
@media (prefers-reduced-motion: no-preference) { /* user has expressed nothing */ }
@media (prefers-reduced-motion)                { /* identical to `: reduce` */ }
```

Two values only. MDN: `no-preference` "evaluates as false"; `reduce` "evaluates as true." **`no-preference` is not consent.**

```js
const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
function apply() {
  if (mq.matches) { /* stop JS + WAAPI + library + Lottie + canvas + video motion */ }
  else            { /* full motion */ }
}
mq.addEventListener('change', apply); // modern
apply();                               // and run once on load
```

⚠️ **Parentheses are mandatory** — `matchMedia('prefers-reduced-motion: reduce')` silently never matches. ⚠️ `addListener()` (WebKit's 2017 sample) is deprecated → `addEventListener('change', …)`.

Non-JS delivery, worth recommending because it avoids the download entirely:

```html
<link rel="stylesheet" href="animations.css" media="(prefers-reduced-motion: no-preference)">

<picture>
  <source srcset="cat.avifs" type="image/avif" media="(prefers-reduced-motion: no-preference)">
  <source srcset="cat.gif"   type="image/gif"  media="(prefers-reduced-motion: no-preference)">
  <img src="cat.png" alt="…" width="250" height="250">
</picture>
```

Server-side: the `Sec-CH-Prefers-Reduced-Motion` client hint (requires an `Accept-CH` opt-in).

**Where the setting lives, for testing:**

| Platform | Path |
|---|---|
| macOS ≤ 15 (Sequoia) | System Settings → Accessibility → Display → **Reduce motion** |
| macOS 26 (Tahoe)+ | System Settings → Accessibility → **Motion** → Reduce motion |
| iOS / iPadOS | Settings → Accessibility → **Motion** → Reduce Motion |
| Windows 10 | Settings → Ease of Access → Display → **Show animations in Windows** |
| Windows 11 | Settings → Accessibility → Visual Effects → **Animation Effects** |
| Android 9+ | Settings → Accessibility → **Remove animations** |
| GNOME/GTK | Accessibility → Seeing → **Reduced animation**; `gsettings set org.gnome.desktop.interface enable-animations false` |
| KDE Plasma | Workspace Behavior → Animation speed → **Instant** |
| **Firefox (best test lever)** | `about:config` → integer pref `ui.prefersReducedMotion` → `1` = reduce, `0` = full. **Immediate effect, no restart.** |
| Chrome DevTools | Rendering → Emulate CSS media feature `prefers-reduced-motion: reduce` |

### 7.3 Substitute, don't delete

WebKit (James Craig), verbatim: *"If your site uses a vestibular trigger animation to convey some essential meaning… removing the animation entirely may make the interface confusing or unusable… only remove the animations you know to be vestibular triggers… consider serving an alternate, simpler animation, or display another visual indicator to convey the intended meaning."* MDN's own example swaps a **`pulse` (scaling)** animation for **`dissolve` (opacity)** under `reduce` — not `animation: none`.

⚠️ **Partially outdated:** Val Head's 2015 ALA guidance that *"Animation that involves only non-moving properties, like opacity, color, and blurs, are unlikely to be problematic"* has been **superseded on blur**. A **WCAG erratum amended the `motion animation` definition "to not exclude blurring."** Current text: *"Motion animation does not include changes of color, blurring, or opacity which do not change the perceived size, shape, or position."* WebKit also lists animated blur as a zoom-class trigger. **Substitute opacity/colour cross-fades. Do not substitute blur.**

Hierarchy for a reduced variant:

1. Keep the state change, remove the travel → `opacity` cross-fade instead of `translate`/`scale`.
2. Shrink amplitude and duration (24 px slide → 4 px, or 0).
3. `animation: none` **only** for pure decoration.
4. If motion *is* the information, gate it behind an explicit play control plus a static or text equivalent.

### 7.4 ⚠️ The blunt global reset is a user-side tool, not an author-side one

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

web.dev files this under **"(Bonus) Forcing reduced motion on all websites"** — to *"inject … into every web page you visit"* via an extension, *"use at your own risk."* **Andy Bell's 2023 "A (more) Modern CSS Reset" no longer contains the reduced-motion block** (his 2020 reset did).

Known breakage:

- **`animationend` / `transitionend` never fire** with `animation: none`. web.dev: *"As some websites depend on an animation to be run in order to work correctly (maybe because a certain step depends on the firing of the `animationend` event), the more radical `animation: none !important;` approach wouldn't work."* The `0.01ms`/`1ms` values exist precisely to keep those events firing.
- **JS-driven motion is unaffected.** web.dev: *"it can't stop motion that was initiated using the Web Animations API."* Same for every JS animation library, Lottie, `rAF`, canvas/WebGL, `<video>`, and smooth-scroll libraries.
- `transition-delay: -1ms !important` kills hover-intent tooltips; `!important` on `*` is unoverridable downstream.
- `background-attachment: initial` (in web.dev's variant) kills `fixed` backgrounds — that is a layout change, not a motion change.
- Does nothing for `<video autoplay>`, animated GIF/WebP/AVIF, or `<marquee>`.

**Audit recommendation: a blanket `*` reset is YELLOW, not a pass.** It is useful as a net *in addition to* per-component handling. A site whose only implementation is the global reset scores non-compliant — it proves nothing about JS motion and makes no substitution decisions.

### 7.5 The CSS-only compliance failure (the most common real defect)

| Motion source | Honoured by the CSS media query alone? | Required gate |
|---|---|---|
| CSS `animation` / `transition` | Yes | `@media` |
| **Scroll-driven animations** (`animation-timeline`) | Yes, but **frequently forgotten** | `@media` |
| **View Transitions** (`::view-transition-*`) | **UA default cross-fade/morph is NOT auto-disabled** | explicit override, below |
| WAAPI / `element.animate()` | **No** | `matchMedia` |
| JS animation libraries (timeline, spring, scroll-trigger categories) | **No** | the library's own reduced-motion API, or a `matchMedia` teardown |
| Lottie / dotLottie / state-machine runtimes | **No** — `autoplay` fires regardless | render frame 0 as a still |
| `<video autoplay loop muted>` hero | **No** | swap to `poster` |
| Animated GIF / APNG / WebP / AVIF | **No** | `<picture>` + `media="(prefers-reduced-motion: no-preference)"` |
| Autoplaying carousels / tickers / marquees | **No** | `matchMedia` **plus** a 2.2.2 pause control regardless |
| Canvas / WebGL loops | **No** | pause the `rAF` loop |
| Smooth-scroll libraries | **No** in general — see `dynamic-loading.md` §6 | destroy/disable |
| `scroll-behavior: smooth` | **No, unless gated** | below |
| `scrollIntoView({behavior:'smooth'})` | **No** | `behavior: mq.matches ? 'auto' : 'smooth'` |
| SMIL | **No** | `svg.pauseAnimations()` |

```css
html { scroll-behavior: auto; }
@media (prefers-reduced-motion: no-preference) {
  html { scroll-behavior: smooth; }
}

@media (prefers-reduced-motion: reduce) {
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) { animation: none !important; }
}
```

**Reduced motion should be the DEFAULT.** Tatiana Mac: *"this is operating on a no-consent model… The user hasn't necessarily opted into animations. They just haven't checked 'Reduce motion.'… it's equally possible the user doesn't know about this setting."* And: *"Defaulting to this latter approach will mean that all users will default to no animation, including users whose browsers won't recognise the media query."*

This reconciles with WebKit's "don't reduce too much": **author the reduced/static state as the base, add motion inside `no-preference`; within `reduce`, substitute rather than delete where motion carried meaning.** See `../examples/motion-system.md` for the full pattern.

### 7.6 Vestibular triggers

**Prevalence.** WebKit (citing vestibular.org): *"affecting as many as **69 million people** in the United States alone"* (lifetime). Val Head / ALA: *"approximately **8 million American adults** report a chronic problem with balance, while an additional **2.4 million** report a chronic problem with dizziness"* (chronic, self-reported). These are different measures and are often conflated — quote them separately.

**WebKit's five trigger categories (verbatim headings):**

1. **Scaling and zooming** — *"give the illusion that the viewer is moving forward or backward in physical space. Some animated blurring effects give a similar illusion."* Exempt: *"real-time, user-controlled direct manipulation effects such as pinch-to-zoom."*
2. **Spinning and vortex effects** — *"can cause some people with vestibular disorders to lose their balance or vertical orientation."*
3. **Multi-speed or multi-directional movement** — parallax; background and foreground moving at different rates.
4. **Dimensionality or plane shifting (2.5D)** — *"the illusion of moving two-dimensional (2D) planes in three-dimensional (3D) space."*
5. **Peripheral motion** — *"Horizontal movement in the peripheral field of vision can cause disorientation or queasiness. Think back to the last time you read a book while in a moving vehicle."*

**Val Head's three quantitative factors:**

- **Relative size of movement** — *"The physical size of screen matters less than the size of the motion relative to the screen space available — so a small button with a 3D rotation probably won't cause trouble, but a full-screen wipe transition covering the entire screen likely would."*
- **Mismatched directions and speed** — *"Exaggerated parallax and scrolljacking animations are highly likely to be triggering… Animations that move a different direction than the user is scrolling, or in a way not directly linked to the speed at which the user is scrolling, also tend to be problematic."*
- **Distance covered** — *"animations covering a large perceived distance can be triggering. For example, iOS 7's 3D zoom transitions caused trouble because of the amount of virtual space covered so quickly."*

⚠️ **No source publishes a hard "% of viewport = large-scale" number.** Circulating heuristics (motion covering >25–30% of viewport area, travel >20–25% of a viewport dimension) are **practitioner rules of thumb, not sourced standards** — label them as such in the report. Express the rule as a ratio to viewport per Head, anchored by her two endpoints (small-button 3D rotation = safe; full-screen wipe = unsafe). The only WCAG-defined number in this space is 2.3.1's 25%-of-a-10-degree-field.

**Named offenders (verified):** iOS 7 (2013) parallax + 3D zoom (Apple shipped Reduce Motion in response); the **Apple.com homepage carousel** — Craig Grannell: *"a big problem, especially when it flicks back to the first picture. There are no controls to pause or stop that particular carousel"* (a textbook 2.2.2 failure); **wired.co.uk** horizontal scroll; Vimeo Cameo, Ice and Sky. Apple's own remediations are documented with before/after video on the WebKit blog (apple.com/macos/sierra's 3D dolly zoom + animated blur → a simple scroll view; apple.com/environment's 2.5D solar-array tilt → a still).

⚠️ **Even *implied* motion triggers.** Greg Tarnoff on the ALA article's static hero: *"the great header illustration… actually triggers me with the 'double line' 3Dish thing going on in the middle tweening steps. Sometimes even the allusion to motion can be enough."* Motion-blur and multiple-exposure static imagery is not automatically safe.

⚠️ **"Facebook's parallax bug reports"** could not be verified in any primary source. **Do not cite it** — use iOS 7, the Apple.com carousel, or wired.co.uk instead.

**Do not recommend banning animation.** Head: *"Not one person I spoke with said that they want to see all interface animation eliminated."* Cited benefits: reduces cognitive load, improves decision-making, aids learning of spatial relationships, and prevents **change blindness**. WebKit: *"Remember that the Web belongs to the user, not the author."*

### 7.7 Duration and easing, tied to size and distance

Carbon's scaling principle: *"Motion's duration should be dynamic based on the size of the animation; the larger the change in distance (traveled) or size (scaling) of the element, the longer the animation takes."* Duration grows **sub-linearly** with distance.

**IBM Carbon tokens (live docs, updated 13 Aug 2026):**

| Token | Usage | Value |
|---|---|---|
| `duration-fast-01` | Micro-interactions such as button and toggle | **70 ms** |
| `duration-fast-02` | Micro-interactions such as fade | **110 ms** |
| `duration-moderate-01` | Micro-interactions, small expansion, short distance | **150 ms** |
| `duration-moderate-02` | Expansion, system communication, toast | **240 ms** |
| `duration-slow-01` | Large expansion, important system notifications | **400 ms** |
| `duration-slow-02` | Background dimming | **700 ms** |

Carbon checklist: *"Do micro-interactions fall within a static duration ranging from **90–120 ms**?"* and *"Do micro-interactions use **`ease-out`** on user input?"*

**Material Design 3 tokens** (`material-web` v0.192): `short1` 50 · `short2` 100 · `short3` 150 · `short4` 200 · `medium1` 250 · `medium2` 300 · `medium3` 350 · `medium4` 400 · `long1` 450 · `long2` 500 · `long3` 550 · `long4` 600 · `extra-long1` 700 · `extra-long2` 800 · `extra-long3` 900 · `extra-long4` 1000.

**Synthesised ranges — this is the audit table:**

| Category | Recommended | Hard ceiling |
|---|---|---|
| Micro-interaction / hover / focus / button press | **70–120 ms** | 150 ms |
| Small element enter/exit (dropdown, tooltip, checkbox) | **150–200 ms** | 250 ms |
| Toast / snackbar / accordion / inline expansion | **200–300 ms** | 400 ms |
| Modal / dialog / side panel / sheet | **250–400 ms** | 500 ms |
| Full-page / view transition | **300–500 ms** | 600 ms |
| Background scrim / dim | **~700 ms** | — |

Under **~70 ms reads as instantaneous**; over **~500 ms on a routine interaction feels sluggish**; **>1000 ms** is only for one-time onboarding, never a repeated action.

⚠️ **Apple HIG deliberately does NOT publish a millisecond duration table** — its guidance is qualitative plus a strong "support Reduce Motion" directive. **Do not fabricate Apple ms numbers.**

**Easing conventions:**

| Situation | Easing | Carbon rationale (verbatim) |
|---|---|---|
| Element **entering** | **ease-out** | *"an element quickly appears and slows down to a stop. Use `entrance-ease` when adding elements to the view… Elements moving in response to the user's input, such as a dropdown opening or toggle switching should also use this style."* |
| Element **exiting** | **ease-in** | *"The element speeds up as it exits from view, implying that its departure from the screen is permanent."* |
| Element **moving on-screen** | **ease-in-out / standard** | *"Use `standard-easing` when an element is visible from the beginning to the end of a motion."* |
| Leaving view but staying nearby | **standard**, not exit | *"A good example of this is a side panel… implying that it would come to rest just outside the view, and ready to be recalled."* |

**IBM Carbon `cubic-bezier()`:**

| Curve | Productive | Expressive |
|---|---|---|
| Standard | `cubic-bezier(0.2, 0, 0.38, 0.9)` | `cubic-bezier(0.4, 0.14, 0.3, 1)` |
| Entrance | `cubic-bezier(0, 0, 0.38, 0.9)` | `cubic-bezier(0, 0, 0.3, 1)` |
| Exit | `cubic-bezier(0.2, 0, 1, 0.9)` | `cubic-bezier(0.4, 0.14, 1, 1)` |

**Material Design 3:**

```css
--md-sys-motion-easing-standard:              cubic-bezier(0.2, 0, 0, 1);
--md-sys-motion-easing-standard-accelerate:   cubic-bezier(0.3, 0, 1, 1);
--md-sys-motion-easing-standard-decelerate:   cubic-bezier(0, 0, 0, 1);
--md-sys-motion-easing-emphasized-accelerate: cubic-bezier(0.3, 0, 0.8, 0.15);
--md-sys-motion-easing-emphasized-decelerate: cubic-bezier(0.05, 0.7, 0.1, 1);
/* Material 2 legacy — still the most-copied values on the web */
--md-sys-motion-easing-legacy:                cubic-bezier(0.4, 0, 0.2, 1);
```

⚠️ **Outdated:** `cubic-bezier(0.4, 0, 0.2, 1)` is the **Material 2** standard curve, now labelled **`easing-legacy`** by Google. M3's standard is `cubic-bezier(0.2, 0, 0, 1)`.

Carbon anti-patterns, verbatim: *"Avoid easing curves that are unnatural, distracting, or purely decorative… Do not use easing curves that suggest bounce, stretch, or sudden stops."* and *"Strictly linear movement appears unnatural to the human eye."* Exception: `linear` is correct for continuous indefinite motion (spinners, progress, marquees).

**Bounce/overshoot is an accessibility concern**, not just taste — it adds travel distance and a direction reversal, mapping onto Head's "distance covered" and "mismatched direction" triggers. Gate it behind `no-preference`.

**`linear()` easing — Baseline widely available since 2026-06-11** (newly available 2023-12-11): Chrome/Edge **113**, Firefox **112**, Safari/iOS **17.2**. Spring-like easing is now safe in plain CSS with no JS library.

```css
.spring {
  transition: translate 500ms linear(
    0, 0.006, 0.025, 0.101, 0.539, 0.826, 0.949, 1.001,
    1.017, 1.012, 1.001, 0.997, 0.999, 1
  );
}
@supports not (transition-timing-function: linear(0, 1)) {
  .spring { transition-timing-function: cubic-bezier(0.05, 0.7, 0.1, 1); }
}
```

Values passing 1.0 are overshoot — under `reduce`, swap to a monotonic curve.

⚠️ **CSS `spring()` is NOT a shipping web platform feature.** A webstatus.dev query for `spring` returns **zero tracked features**. Use `linear()`, or a JS library for interruptible physical springs.

**Stagger delays** ⚠️ *(unverified numbers — no design system publishes a stagger table; these are practitioner rules of thumb, and must be labelled as such)*: ~20–50 ms per item feels intentional; <15 ms reads as simultaneous; >80–100 ms per item feels slow. **The total stagger budget matters more** — cap the whole sequence at roughly one modal transition (~300–500 ms end-to-end). 12 items × 50 ms = 550 ms before the last item even starts, which is too slow; reduce the per-item delay as the count rises, or stagger only the first 4–6. Under `reduce`, set all delays to 0 and cross-fade the group together — stagger creates directional flow across a large area, a Head "relative size of movement" trigger even when each individual element barely moves.

### 7.8 When animation is decorative noise

Carbon's sharpest check, verbatim: *"**Is your motion unobtrusive?** The best interface motion may go unnoticed… **Is your motion frequently noticed by average users? If so, consider removing or minimizing it.**"*

**Kill-list — recommend deleting the animation if any is true:**

1. **No state change behind it** (fails WCAG's `essential` test — and note the definition's conjunction: `essential` = "if removed, would fundamentally change the information or functionality of the content, **and** information and functionality cannot be achieved in another way that would conform." "It looks nicer" and "it's our brand" both fail).
2. **Seen more than ~5 times per session.** The cost is paid every time; the novelty benefit is paid once.
3. **It delays access to content the user asked for** — entrance animations above the fold, scroll-reveal on body copy, page-transition curtains.
4. **Infinite loop and not a progress indicator.** Directly implicates 2.2.2.
5. **In peripheral vision while reading** — floating particles, ambient loops, animated gradients beside body text.
6. **Moves at a rate or direction the user did not command** — scroll-jacking, parallax, scroll-velocity-decoupled reveals.
7. **Scales, spins, blurs, or plane-shifts across a large fraction of the viewport.**
8. **You cannot answer "why is this here?" in one non-aesthetic sentence.** Mac's escalation: *"Why is this animation critical? / What other ways can we serve up the critical animation? / If the animation can't load for someone or they can't see it, what was your plan?"*
9. **It would fail at 3× duration on a slow device.** Load-bearing motion must survive being slow.

---

## 8. Measurement and verification

**This is the section that makes the audit worth anything.** Everything above tells you what is *likely* to be slow. This section is how you *prove* it, and how you prove the fix worked. Never accept a claim of smoothness without one of these artifacts.

### 8.1 Why Lighthouse cannot tell you this

**Lighthouse cannot measure INP.** From `core/config/default-config.js`:

```js
{id: 'first-contentful-paint',      weight: 10, group: 'metrics', acronym: 'FCP'},
{id: 'largest-contentful-paint',    weight: 25, group: 'metrics', acronym: 'LCP'},
{id: 'total-blocking-time',         weight: 30, group: 'metrics', acronym: 'TBT'},
{id: 'cumulative-layout-shift',     weight: 25, group: 'metrics', acronym: 'CLS'},
{id: 'speed-index',                 weight: 10, group: 'metrics', acronym: 'SI'},
{id: 'interaction-to-next-paint',   weight:  0, group: 'metrics', acronym: 'INP'},   // ← ZERO
```

`interaction-to-next-paint` only produces a value in **timespan** mode with scripted interactions. TBT is a proxy for main-thread blocking **during load**, not for interaction responsiveness during steady-state animation.

**`non-composited-animations` is `INFORMATIVE`, weight 0**, and its own source comments away its impact:

```js
metricSavings: {
  // We do not have enough information to accurately predict the impact of individual
  // animations on CLS. It is also not worth the effort since only a small percentage
  // of sites have their CLS affected by non-composited animations.
  CLS: 0,
},
```

It reads `failureReasonsMask` from `TraceElements` and filters to only **six** "actionable" reasons: `1 << 3` unsupported timing parameters; `1 << 4` composite mode other than "replace"; `1 << 6` target has another incompatible animation; `1 << 11` transform-related property depends on box size; `1 << 12` filter-related property may move pixels; `1 << 13` unsupported CSS property.

**False negatives:** all other mask bits are silently discarded; only animations inside the trace window are seen (hover, scroll and modal transitions are invisible); **JS-driven `style.top`, scroll handlers, IntersectionObserver churn, and canvas work are entirely out of scope**; there is no frame data at all. **False positives:** `transformDependsBoxSize`, `filterMayMovePixels`, 200 ms imperceptible animations, and `--*` custom properties.

> **Why a 100 score coexists with terrible scroll jank:** every scored metric is a **load** metric. `long-tasks`, `non-composited-animations`, `mainthread-work-breakdown`, `bootup-time`, `layout-shifts`, `inp-breakdown-insight`, `forced-reflow-insight`, and `dom-size-insight` are all **weight 0**. **A page that loads in 800 ms and drops 40% of frames on every scroll scores 100.**

What Lighthouse *does* give you (unscored but useful as leads, not as evidence): `non-composited-animations`, `forced-reflow-insight` (Chrome 134+), `dom-size-insight`, `cls-culprits-insight`, `long-tasks`, `mainthread-work-breakdown`, `unsized-images`. TBT is scored at weight 30. Lighthouse 13 (Chrome 143) unified insights with DevTools.

### 8.2 DevTools Performance panel workflow

**Opening Performance lands on the Live metrics screen, not an empty recorder.** LCP and CLS are captured immediately; **INP only once you interact.** Interactions and Layout shifts tabs sit under the metric cards. The **Interaction phases breakdown** shipped in Chrome 132. **Field data (CrUX)** is available via *Next steps > Field data > Set up*, with dev→prod origin mapping. **Environment settings** gives CrUX-derived recommendations; Chrome 133's default CPU recommendation is `4x slowdown`.

**The workflow, in order:**

1. **Calibrate throttling first** — *Settings > Throttling > CPU throttling presets > Calibrate > Continue* (Chrome 134+). Wait ~5 seconds while DevTools navigates away and reloads. Never hardcode 4×.
2. **Record while performing the exact motion under audit** — a full-page wheel scroll, the hover, the modal open. Record 5–10 s; longer traces get hard to read.
3. **Read the Frames track. This is the ground truth for lab jank.** Four frame types, verbatim:
   - **Idle frame (white)** — "No changes."
   - **Frame (green)** — "Rendered as expected and in time."
   - **Partially presented frame (yellow, sparse wide dash-line pattern)** — *"Chrome did its best to render at least some visual updates in time. For example, in case the work of the main thread of the renderer process (canvas animation) is late but the compositor thread (scrolling) is in time."*
   - **Dropped frame (red, dense solid-line pattern)** — *"Chrome can't render the frame in reasonable time."*

   **Audit rule: many yellow frames during scroll = main-thread animation work losing to a compositor-driven scroll** — exactly what Lighthouse cannot see.
4. **Read the Animations track** — the empirical replacement for Lighthouse's audit: *"Animations are named as corresponding CSS properties or elements if any, for example, `transform` or `my-element`. **Non-compositing animations are marked with red triangles in the top right corner**. Select an animation to see more details in the **Summary** tab, **including reasons for compositing failures**."*
5. **Read the Interactions track** — whiskers mark input delay and presentation delay. **Interactions over 200 ms get a red triangle in the top-right corner** plus an INP warning.
6. **Read the Insights sidebar** (failing insights expand; passing ones collapse under *Passed insights (N)*): **Forced reflow** (Chrome 134 — names the top function, stack trace, and total reflow time), **Optimize DOM size** (134), **Layout shift culprits** (133/139), **INP breakdown**, **Duplicated/Legacy JavaScript** (137).

**Notable 2025–2026 changes worth knowing so you do not follow stale instructions:** Chrome 149 upgraded the bundled `web-vitals` to v5.2.0 (**fixes memory leaks during INP monitoring**); 148 surfaces CrUX-recommended network presets; 145 added soft-navigation markers (*"The live metrics view and Performance Insights don't reflect soft navigations"*); 138 **removed "Disable JavaScript samples"**; 136 added **total layer count + memory to the Layers panel status bar**; 134 added **calibrated CPU throttling presets**; **132 deprecated and removed the Performance insights panel** and *"marks the end of support for the Web Vitals extension."* **Zero Rendering-tab changes across Chrome 132–151.**

### 8.3 The Rendering tab

Open via `Ctrl/Cmd+Shift+P` → "Show Rendering". Options: **Paint flashing**, **Layout Shift Regions**, **Layer Borders**, **Frame rendering stats**, **Scrolling Performance Issues**, **Highlight ad frames**, **Emulate a focused page**, **Emulate CSS media type**, **Emulate CSS media feature** (including `prefers-reduced-motion`, `prefers-reduced-transparency`, `forced-colors`, `prefers-contrast`), **Emulate vision deficiencies**, **Disable local fonts / AVIF / WebP**, **Enable automatic dark mode**.

| Tool | What it proves | Red flags |
|---|---|---|
| **Paint flashing** — *"Chrome flashes the screen green whenever repainting happens."* | Whether an animation is repainting or only compositing | Whole screen green during scroll; a fixed header flashing while the body scrolls; the whole viewport flashing on hover of one element. Test page: `googlechrome.github.io/devtools-samples/jank/` |
| **Layer Borders** — *"layer borders in orange and olive and tiles in cyan."* Legend in `cc/debug/debug_colors.cc` | Whether the element is actually composited | **No border on an animated element → it is not composited.** Hundreds of orange borders → layer explosion |
| **Frame rendering stats** ⚠️ the old "FPS meter" was renamed/replaced by this — the perf reference still carries a stale "FPS meter" heading | Real-time estimated FPS; a frame timeline plot with **blue = successfully rendered, yellow = partially presented, red = dropped**; **GPU raster state: on/off**; **GPU memory usage: used MB / max MB** | **GPU memory near max is the layer-explosion tell** |
| **Scrolling Performance Issues** | Highlights (in **teal**) elements with scroll listeners | Your non-passive-listener detector |
| **Emulate CSS media feature `prefers-reduced-motion`** — docs name only `reduce`; the sibling options start with "No emulation" | Whether the reduced-motion implementation is real | A `@media (prefers-reduced-motion: reduce)` block that only sets `animation: none` while JS scroll animations, `scroll-behavior: smooth`, or WAAPI calls keep running is a **failed** implementation |

⚠️ **OUTDATED: the "Core Web Vitals" overlay checkbox is GONE.** *"January 2025 update: The merger of the Web Vitals extension and DevTools is complete and support for the extension has ended."* The replacement is Performance → Live metrics. **Flag any guidance saying "enable the Core Web Vitals overlay in the Rendering tab."**

### 8.4 The Layers panel

Command Menu → "Show Layers panel". Also available as a Layers tab inside the Performance panel if **Capture settings > Enable advanced paint instrumentation (slow)** is on and you select a frame in the Frames track.

Details pane: **Size**, **Compositing Reasons** (the load-bearing one — it tells you *why* a layer exists), **Memory estimate**, **Paint count** (climbing during animation = repainting, not re-compositing), **Slow scroll regions**, **Sticky position constraint**. Toolbar: **Paints** (enables the Paint Profiler), **Slow scroll rects** (pink highlight). **Chrome 136 added total layer count + total memory to the bottom status bar** — that single number is your layer-explosion check.

### 8.5 `PerformanceObserver` — programmatic detection

#### Long Animation Frames (LoAF)

MDN BCD: `PerformanceLongAnimationFrameTiming`, `blockingDuration`, `firstUIEventTimestamp`, `renderStart`, `scripts`, `styleAndLayoutStart` → **Chrome 123**, **Firefox false**, **Safari false**. `paintTime` / `presentationTime` → **Chrome 145**. **Experimental / Limited availability — not Baseline.**

Semantics: `duration` runs until *"all that's left is painting & compositing"* and **does not include presentation time**; the threshold is **50 ms**. `renderStart` = *"Equivalent to BeginMainFrame in Chromium"*. `styleAndLayoutStart` **includes ResizeObserver callbacks**. `blockingDuration` = long tasks **plus** (longest task + rendering time) if their sum exceeds 50 ms, summed with 50 ms subtracted from each. `scripts[]` includes **only scripts ≥ 5 ms**, same-origin windows only.

```js
if (PerformanceObserver.supportedEntryTypes?.includes('long-animation-frame')) {
  new PerformanceObserver((list) => {
    for (const loaf of list.getEntries()) {
      const renderDuration      = loaf.startTime + loaf.duration - loaf.renderStart;
      const styleLayoutDuration = loaf.startTime + loaf.duration - loaf.styleAndLayoutStart;
      const preRenderDuration   = loaf.renderStart ? loaf.renderStart - loaf.startTime : loaf.duration;

      console.groupCollapsed(
        `LoAF ${loaf.duration.toFixed(1)}ms  blocking=${loaf.blockingDuration.toFixed(1)}ms`
      );
      console.log({ preRenderDuration, renderDuration, styleLayoutDuration,
                    firstUIEventTimestamp: loaf.firstUIEventTimestamp });

      for (const s of loaf.scripts) {
        console.log(
          `  [${s.invokerType}] ${s.invoker}\n` +
          `    ${s.duration.toFixed(1)}ms  forcedStyleAndLayout=${s.forcedStyleAndLayoutDuration.toFixed(1)}ms\n` +
          `    ${s.sourceURL}:${s.sourceCharPosition}  fn=${s.sourceFunctionName}\n` +
          `    window=${s.windowAttribution}`
        );
      }
      console.groupEnd();
    }
  }).observe({ type: 'long-animation-frame', buffered: true });
}
```

`scripts[]` fields: `invokerType` (`"user-callback"` | `"event-listener"` | `"resolve-promise"` | `"reject-promise"` | `"classic-script"` | `"module-script"`), `invoker` (e.g. `"IMG#id.onload"`, `"Window.requestAnimationFrame"`), `startTime`, `executionStart`, `duration`, **`forcedStyleAndLayoutDuration`**, `pauseDuration`, `sourceURL`, `sourceFunctionName`, `sourceCharPosition` (**character position, not line/column** — *"to avoid overhead of line splitting"*), `windowAttribution` (`"self"` | `"descendant"` | `"ancestor"` | `"same-page"` | `"other"`).

**Triage table:**

| Signal | Diagnosis |
|---|---|
| High `forcedStyleAndLayoutDuration` | **Layout thrashing** — go to §3 |
| Dominant style/layout time with low script time | Expensive selectors or a huge DOM |
| `invokerType: "event-listener"` with `"DIV#hero.onscroll"` | Heavy scroll handler — go to §1.2 and `dynamic-loading.md` §1 |
| `windowAttribution: "descendant"` / `"other"` | Third-party iframe or ad |

#### `longtask` — fallback only

⚠️ From the explainer: *"developers can game their long task timing by moving long operations into a `requestAnimationFrame` callback"*; *"a task is an incomplete and inaccurate cadence to measure main-thread blocking."* **A site that moves heavy work into rAF shows zero long tasks and still drops every frame.** Use only where LoAF is unavailable (Firefox/Safari), and say so in the report.

#### `event` timing / INP

| Subpart | Formula |
|---|---|
| **Input delay** | `processingStart - startTime` |
| **Processing duration** | `processingEnd - processingStart` |
| **Presentation delay** | `startTime + duration - processingEnd` |

`interactionId === 0` means the entry is not part of a tracked interaction. **`duration` is rounded to the nearest 8 ms** and can round *down*. `durationThreshold`: **spec default 104 ms**; `web-vitals` overrides to **40 ms**. The 104 ms spec default still applies to entries emitted **before** your observer initializes, even with `buffered: true`.

```js
const interactions = new Map();
new PerformanceObserver((list) => {
  for (const e of list.getEntries()) {
    if (!e.interactionId) continue;
    const inputDelay        = e.processingStart - e.startTime;
    const processing        = e.processingEnd - e.processingStart;
    const presentationDelay = e.startTime + e.duration - e.processingEnd;
    const prev = interactions.get(e.interactionId);
    if (!prev || e.duration > prev.duration) {
      interactions.set(e.interactionId,
        { type: e.name, duration: e.duration, inputDelay, processing, presentationDelay, target: e.target });
    }
  }
}).observe({ type: 'event', buffered: true, durationThreshold: 16 });
```

**If `presentationDelay` dominates, the handler is fast but the *frame* is slow** — rendering/animation cost being misdiagnosed as "slow JavaScript". That is the single most common misattribution in animation work.

#### Dropped frames — there is NO standardized API

[WICG/frame-timing](https://github.com/WICG/frame-timing) README, verbatim: **"This effort is no longer being pursued."** `w3c/frame-timing` was **archived 2018-12-18**. LoAF reports *long* frames, not *dropped* ones — a frame the compositor never produced generates no LoAF entry.

rAF delta timing is the portable fallback, with three caveats: (1) it measures **when the callback ran, not when the frame was presented** (use LoAF `presentationTime`, Chrome 145+); (2) it is **blind to compositor-thread-only jank** — the most common false negative; (3) rAF does not fire in background tabs, and **do not hardcode 60 Hz**. Calibrate to the observed median delta and count `Math.round(delta / median) - 1` missed frames per sample. **Ground truth requires a trace (§8.7).**

### 8.6 CPU throttling and real devices

**Throttling is relative to the host, not absolute:** *"Throttling is relative to your computer's capabilities... DevTools can't truly simulate the CPUs of mobile devices, because the architecture of mobile devices is very different from that of desktops and laptops."* Hence `benchmarkIndex` in every Lighthouse report.

| | High-End Desktop | Low-End Desktop | High-End Mobile | **Mid-Tier Mobile** | Low-End Mobile |
|---|---|---|---|---|---|
| Example device | 16" MacBook Pro | Intel NUC i3 | Samsung S10 | **Moto G4** | Galaxy J2 |
| benchmarkIndex | 1500–2000 | 1000–1500 | 800–1200 | **125–800** | <125 |
| Speedometer 2.0 | 90–200 | 50–90 | 20–50 | 10–20 | <10 |
| JS exec, news site | 2–4 s | 4–8 s | 4–8 s | **8–20 s** | 20–40 s |

From a high-end desktop host: **4×** targets mid-tier mobile (the Lighthouse default, sourced); **10×** targets low-end mobile. **6×** is inference from WebPageTest's Nexus 5 (6.5×) / Nexus 5X (6.0×). **20× is unsourced** — WPT's slowest real entries are 15×.

**Real-device multipliers (WPT `mobile_devices.ini`):** Motorola E 15× · Android One 15× · Galaxy S5 9× · **Moto G4 8×** · **Moto G Power (2022) 7.5×** · Nexus 5 6.5× · Galaxy S7 4.5× · Pixel 3A 4.2× · Pixel 5 2× · **Pixel 6/7/Galaxy S23 1×**. All modern iPhones/iPads are 1×.

DebugBear defaults (updated 2026-05-12): **Mobile** 1.6 Mbps / 150 ms RTT / **4×** / 412×660; **Mobile Fast** 12 Mbps / 70 ms / **2×**; **Desktop** 8 Mbps / 40 ms / 1× / 1350×940.

⚠️ **Any specific 2025–2026 model name (Moto G Power 2025, Galaxy A16/A55, Pixel 9a, Redmi Note 14) is unsourced inference. Do not hardcode model names.** Read `benchmarkIndex` and pick a multiplier that lands in the **125–800** band.

⚠️ **Gotcha:** the WebPageTest agent disables `IsolateOrigins` and `site-per-process` whenever CPU throttling is >1× — *"It is disabled on actual mobile Chrome (and breaks Chrome's CPU throttling)."*

See `mobile.md` for device-class strategy beyond throttling.

### 8.7 Field measurement and automated gates

**`web-vitals` current version is 6.0.1.** ⚠️ **Every CDN example in the current README is still pinned to `@5`** — substitute `@6`.

```js
import {onINP} from 'web-vitals/attribution';

onINP(({name, value, rating, id, attribution: a}) => {
  navigator.sendBeacon('/analytics', JSON.stringify({
    name, value, rating, id,
    target: a.interactionTarget, type: a.interactionType,
    inputDelay: a.inputDelay,
    processingDuration: a.processingDuration,
    presentationDelay: a.presentationDelay,
    longestScript: a.longestScript && {
      subpart: a.longestScript.subpart,
      url: a.longestScript.entry.sourceURL,
      fn: a.longestScript.entry.sourceFunctionName,
    },
    totalScriptDuration: a.totalScriptDuration,
    totalStyleAndLayoutDuration: a.totalStyleAndLayoutDuration,
    totalPaintDuration: a.totalPaintDuration,
  }));
}, { durationThreshold: 40 });
```

~2 KB brotli standard, +~1.5 KB for the attribution build. **Never call `onINP()` / `onCLS()` / `onLCP()` more than once per page load** (memory leak). Flush on `visibilitychange` **and** `pagehide` — never `beforeunload`/`unload`.

```ts
import {CLSThresholds, INPThresholds, LCPThresholds} from 'web-vitals';
console.log(CLSThresholds); // [ 0.1, 0.25 ]
console.log(INPThresholds); // [ 200, 500 ]
console.log(LCPThresholds); // [ 2500, 4000 ]
```

| Metric | good | needs improvement | poor |
|---|---|---|---|
| **INP** | **≤ 200 ms** | 200–500 ms | **> 500 ms** |
| **LCP** | **≤ 2500 ms** | 2500–4000 ms | **> 4000 ms** |
| **CLS** | **≤ 0.1** | 0.1–0.25 | **> 0.25** |

Support: `onCLS()` **Chromium only**; `onFCP()`, `onINP()`, `onLCP()`, `onTTFB()` **Chromium, Firefox, Safari** (a change from older docs that list INP as Chromium-only). LoAF-derived attribution fields are Chromium-only in practice.

Limitations, verbatim: *"they have no visibility into `<iframe>` content (not even same-origin iframes)"*; *"the `onCLS()` function technically measures DCLS (Document Cumulative Layout Shift) rather than CLS, if the page includes iframes."* bfcache is **handled, not excluded** — re-reported with a new `id` and `navigationType: 'back-forward-cache'`.

**Thresholds apply at the 75th percentile, segmented across mobile and desktop.** ⚠️ **Jank concentrates in the tail** — p75 INP of 180 ms with p95 of 900 ms shows green in CrUX. Always look at p95 too.

#### The only ground truth is a trace

Required categories: **`disabled-by-default-devtools.timeline.frame`** (emits `BeginFrame`, `DrawFrame`, **`DroppedFrame`**, `ActivateLayerTree`, `PipelineReporter`) **plus `disabled-by-default-devtools.timeline`** (emits **`Commit`**, `SetLayerTreeId`) — **`Commit` is on the plain category while the frame events are on `.frame`, so you need both.**

```js
import fs from 'node:fs';
import puppeteer from 'puppeteer';

const FRAME_CATEGORIES = [
  '-*', 'devtools.timeline',
  'disabled-by-default-devtools.timeline',
  'disabled-by-default-devtools.timeline.frame',
  'disabled-by-default-devtools.timeline.stack',
  'toplevel', 'blink.user_timing', 'latencyInfo',
  'v8.execute', 'disabled-by-default-v8.cpu_profiler',
];

const browser = await puppeteer.launch({
  headless: false,            // headful gives a real vsync-driven compositor
  args: ['--enable-gpu'],
});
const page = await browser.newPage();
await page.setViewport({ width: 412, height: 823, deviceScaleFactor: 2.6, isMobile: true, hasTouch: true });

const cdp = await page.createCDPSession();
await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });

await page.goto(URL, { waitUntil: 'networkidle2' });
await new Promise(r => setTimeout(r, 1000));   // let post-load work settle

await page.tracing.start({
  path: 'trace.json', screenshots: true,
  categories: [...FRAME_CATEGORIES],   // fresh array: start() mutates it via categories.push()
  bufferSize: 400_000,
});

// Real compositor scroll via trusted input.
// NOTE: window.scrollTo/scrollBy in page.evaluate BYPASSES the input pipeline
// and will UNDER-REPORT scroll jank.
await page.mouse.move(200, 400);
const deadline = Date.now() + 6000;
while (Date.now() < deadline) {
  await page.mouse.wheel({ deltaY: 60 });
  await new Promise(r => setTimeout(r, 16));
}
await page.tracing.stop();
await browser.close();

const { traceEvents } = JSON.parse(fs.readFileSync('trace.json', 'utf8'));
const layerTreeId = traceEvents.find(e => e.name === 'SetLayerTreeId')?.args?.data?.layerTreeId;
const mine = e => layerTreeId == null || e.args.layerTreeId === layerTreeId;
const has  = e => e.args && 'frameSeqId' in e.args;   // the correct compat check

const dropped = traceEvents.filter(e => e.name === 'DroppedFrame' && has(e) && mine(e));
const drawn   = traceEvents.filter(e => e.name === 'DrawFrame'    && has(e) && mine(e));
const partial = dropped.filter(e => e.args.hasPartialUpdate);
const total   = dropped.length + drawn.length;

console.log({
  drawnFrames: drawn.length,
  droppedFrames: dropped.length,
  partialUpdates: partial.length,
  droppedPct: total ? +(100 * dropped.length / total).toFixed(2) : 0,
});
```

**Caveats:** exclude idle periods or a static page reads as "0 drawn, 0 dropped" (`NeedsBeginFrameChanged` with `needsBeginFrame=1` marks the start of an idle period); `hasPartialUpdate: true` is **its own bucket**, not a drop; ignore every compositor event with a different `layerTreeId`; `--disable-frame-rate-limit` makes frame counts meaningless as a 60 fps denominator.

**Playwright:** `browser.startTracing()`/`stopTracing()` exist (v1.11+, not in C#). ⚠️ **`context.tracing.start()` is a completely different thing and is NOT a performance trace** — no Chrome trace events, no frame events, no compositor timing. Playwright has **no `page.metrics()`**; use `Performance.getMetrics` or raw `Tracing.start` with a structured `traceConfig` via CDP.

**Best tooling in 2026: `@paulirish/trace_engine`** (npm v0.0.65, ~Jun 2026, BSD-3) — *"This package contains the trace engine implementation used by the DevTools Performance Panel."* Identical dropped/partial/idle classification, plus insights, LoAF, and third-party attribution. **Perfetto trace processor** for SQL over `PipelineReporter`/`ChromeFrameReporter` at scale. ⚠️ **`tracium` is legacy** — main-thread task attribution only, no frame model.

`chrome.gpuBenchmarking` (`--enable-gpu-benchmarking`) still exists in Chromium `main` and offers `smoothScrollBy()`, `smoothDrag()`, `swipe()`, `pinchBy()`, `pointerActionSequence()`, and **`addSwapCompletionEventListener(cb)`, which fires per compositor frame swap** — a JS-visible present signal. **Private, unstable, testing-only**; often needs `--no-sandbox`.

See `../scripts/audit_motion.py` for the packaged version of these gates.

---

## 9. Outdated advice to flag and remove on sight

| Pattern found | Why it is wrong now | Replace with |
|---|---|---|
| `translateZ(0)` / `translate3d(0,0,0)` / `backface-visibility: hidden` / `perspective: 1000px` as a promotion hack | `will-change` has been Baseline widely available since **January 2020**. The hack pins a layer forever, creates a stacking context and containing block, degrades text antialiasing, and rasters at a fixed scale | Delete (transform/opacity self-promote), or `will-change` with real lead time |
| Blanket `* { will-change: transform }` or `will-change` on `body` | Applies to the whole subtree; DPR² memory; layer explosion | Remove; promote only what a `rAF` loop mutates imperatively |
| `will-change` next to a CSS `@keyframes` that animates `transform`/`opacity` | Redundant — "Animated properties are treated as if they're already included in a `will-change`" | Delete |
| Animating inside a `scroll` listener | Scroll events are delivered **asynchronously** from a separate process; the compositor has already painted a frame you have not caught up to. Events fire off-cadence, many per frame. The handler runs on main and "cannot interrupt" | `animation-timeline: scroll()/view()`, IntersectionObserver, or a rAF-throttled read/write split (see `dynamic-loading.md` §1) |
| jQuery `.animate()` / `$(window).scroll()` patterns | `.animate()` drives `setTimeout`/`setInterval` style mutation of layout properties on the main thread | CSS transitions/animations, or WAAPI |
| `setTimeout(fn, 16)` / `setInterval` as an animation clock | No vsync relationship, clamped ≥4 ms after 5 nestings and ≥1000 ms in background tabs, hardcodes 60 Hz | `requestAnimationFrame`, with `dt` derived from the timestamp |
| `csstriggers.com` cited as the property-cost reference | Now a WordPress content farm with **wrong** data (claims `box-shadow` triggers layout) | Chromium `kCompositableProperties` + the DevTools Animations track |
| `isInputPending()` for yielding | web.dev: **"Don't use `isInputPending()`"** — "We no longer recommend using this API" | `scheduler.yield()` (Chrome/Edge 129, Firefox 142) with a `setTimeout(0)` fallback |
| "SMIL is deprecated / being removed from Chrome" | The 2015 Intent to Deprecate was **suspended and never executed**; caniuse `svg-smil` is 96.42% | Say instead: SMIL is safe but **not GPU-composited** — avoid for loops and scroll-linked motion |
| "Chrome never divides an SVG graphic into different GPU layers" (+ stacking separate `<svg>` elements as manual layers) | Superseded by **Chromium 89** (Feb 2021), which hardware-accelerates SVG transform/opacity | Single SVG, animate `transform`/`opacity`, `transform-box: fill-box` |
| "SVGO removes your viewBox by default" | `removeViewBox` was **removed from `preset-default` in SVGO v4** | Keep the explicit override as future-proofing; the real v4 hazard is `removeDesc` |
| "only `<path>` supports `getTotalLength()`" | SVG 1.1 era. It lives on `SVGGeometryElement` now | Prefer `pathLength="1"` normalization and skip JS entirely |
| "Hidden content is discounted by Google" | Pre-2016 advice. `opacity:0` and `display:none` are **both indexed** with no demotion | See `seo/L1-foundations.md` and `dynamic-loading.md` §4 |
| "Enable the Core Web Vitals overlay in the Rendering tab" | **Gone.** The Web Vitals extension merger completed and extension support ended **7 January 2025** | Performance → Live metrics |
| "FPS meter" | Renamed/replaced by **Frame rendering stats**; the perf reference still carries the stale heading | Frame rendering stats |
| `window.performance.timing`, `performance.navigation` | Deprecated | `performance.getEntriesByType('navigation')[0]`, `navigationEntry.type` |
| FID as the responsiveness metric | Superseded by **INP** in March 2024; Lighthouse still emits `max-potential-fid` at weight 0, group `hidden` | INP |
| `frame-timing` API | *"This effort is no longer being pursued."* `w3c/frame-timing` archived 2018-12-18 | Trace-based `DroppedFrame`/`DrawFrame` counting |
| Performance insights panel | Deprecated **and removed in Chrome 132** | Performance panel Insights sidebar |
| "Disable JavaScript samples" checkbox | Removed in Chrome 138 | — |
| `cubic-bezier(0.4, 0, 0.2, 1)` described as "the Material standard curve" | That is **Material 2**, now labelled `easing-legacy` | M3 standard is `cubic-bezier(0.2, 0, 0, 1)` |
| CSS `spring()` referenced as a real feature | **Zero tracked web-features entries** | `linear()` (Baseline widely available 2026-06-11) or a JS spring |
| `mq.addListener(fn)` | Deprecated (WebKit's 2017 sample) | `mq.addEventListener('change', fn)` |
| `matchMedia('prefers-reduced-motion: reduce')` without parentheses | **Silently never matches** | `matchMedia('(prefers-reduced-motion: reduce)')` |
| Audit template testing **4.1.1 Parsing** | "(Obsolete and removed)" in WCAG 2.2 | Drop it |
| Auditing against **WCAG 3.0** | Still a Working Draft (03 March 2026) | WCAG 2.2 AA |
| FastDom in `package.json` | Dormant since ~2019 | The read/write batcher in §3.1, or `ResizeObserver`/`IntersectionObserver` |
| An abandoned scroll-animation library with a scroll-*listener* architecture | Invisible to Googlebot (which does not scroll), no reduced-motion support | 0 kB CSS/IO pattern — see `dynamic-loading.md` §7 |

---

## 10. Testable checklist

**P0 = blocking defect, P1 = should fix, P2 = advisory.** Every row states how to verify. Do not mark a row passed on inspection alone where a tool is named.

*This P-scale is local to this file's checklists and does not ship. Translate to the canonical `blocker | major | minor | advisory` before writing a finding — crosswalk in `reporting.md` §2. Note that `P2` here means advisory, whereas `P2` in `seo/triage.md` means High.*

### 10.1 Compositor discipline

| # | Check | How to verify | Sev |
|---|---|---|---|
| A1 | Every continuous/looping animation animates only `transform` and/or `opacity` | Performance → **Animations track**; non-composited animations carry a **red triangle**, with compositing-failure reasons in Summary | **P0** |
| A2 | No animation of `width`, `height`, `top`, `left`, `right`, `bottom`, `margin`, `padding` | grep `@keyframes` and `transition-property` for these | **P0** |
| A3 | No animated `filter: blur()` / `drop-shadow()` (pixel-moving → not composited) | grep; confirm on the Animations track | P1 |
| A4 | `backdrop-filter` on ≤1–2 elements, none animated | grep; Frame rendering stats → GPU memory | P1 |
| A5 | Animated `clip-path` flagged (`CompositeClipPathAnimation` is experimental, OFF by default in stable Chrome) | grep | P1 |
| A6 | No `translateZ(0)`, `translate3d(0,0,0)`, `backface-visibility: hidden`, `perspective: 1000px` used as a layer hack | grep across CSS | P1 |
| A7 | `will-change` is not applied to elements whose `transform`/`opacity` is already CSS-animated | grep; cross-reference `@keyframes` targets | P1 |
| A8 | `will-change` is not on `*`, `body`, or a large subtree; total promoted layers is bounded | Rendering → **Layer Borders**; Layers panel **status bar total count + memory** (Chrome 136+) | **P0** if layer explosion |
| A9 | JS-toggled `will-change` is added on `pointerenter`/`focusin` (≥~100 ms lead), not in the task that starts the animation; cleanup listener spelled `animationend` not `animationEnd` | Read the source | P2 |
| A10 | Estimated layer memory sane: `w × h × DPR² × 4` per layer, checked at DPR 3 | Layers panel **Memory estimate**; Frame rendering stats **GPU memory used/max** | P1 |
| A11 | Animated SVG children that rotate/scale set `transform-box: fill-box` | grep for `transform-origin` on SVG without `transform-box` | P1 |

### 10.2 Layout thrashing / main-thread hygiene

| # | Check | How to verify | Sev |
|---|---|---|---|
| B1 | No `getBoundingClientRect()`, `offsetTop`, `offsetHeight`, `scrollTop`, `getComputedStyle()` read inside a scroll/resize handler or rAF loop **after** a write | grep against the Paul Irish list; Performance → **Forced reflow insight** (Chrome 134+) names the function | **P0** |
| B2 | Zero `[Violation] Forced reflow while executing JavaScript took Nms` in the console during a full-page scroll | Open the console, scroll top to bottom | **P0** |
| B3 | Field LoAF shows `forcedStyleAndLayoutDuration === 0` for animation-related scripts | Paste the §8.5 observer; scroll and interact | P1 |
| B4 | All `scroll`, `touchstart`, `touchmove`, `wheel` listeners are `{ passive: true }` (or replaced by `touch-action`) | Rendering → **Scrolling Performance Issues** (teal highlights); grep `addEventListener('scroll'` | **P0** |
| B5 | No animation driven directly from a `scroll` handler — use scroll-driven CSS, IntersectionObserver, or a rAF-throttled read/write split | grep for `style.` assignment inside a scroll listener | **P0** |
| B6 | No `setTimeout`/`setInterval` driving animation frames | grep `setInterval`, `setTimeout(.*1[0-9])` | P1 |
| B7 | Long tasks yield via `scheduler.yield()` (Chrome/Edge 129, Firefox 142) with a `setTimeout(0)` fallback, not `isInputPending()` | grep `isInputPending` — discouraged by the Chrome team | P1 |
| B8 | FastDom present? → smell that geometry is read imperatively where an Observer would do | grep `fastdom` in `package.json` | P2 |

### 10.3 Motion accessibility

| # | Check | How to verify | Sev |
|---|---|---|---|
| C1 | `prefers-reduced-motion: reduce` actually stops motion — **CSS *and* WAAPI *and* JS libraries *and* Lottie *and* SMIL *and* `<video autoplay>` *and* animated GIF/WebP *and* canvas/rAF *and* any smooth-scroll library** | Firefox `about:config` → `ui.prefersReducedMotion=1`, **then scroll the entire page with JS enabled**. This is the step everyone skips and where nearly all real failures live | **P0** |
| C2 | The implementation is not *only* a blanket `*` reset — per-component handling exists | grep for the reset; read component CSS | P1 (yellow, not a pass) |
| C3 | The reset uses `0.01ms`/`1ms`, never `animation: none !important` on `*` (breaks `animationend`/`transitionend` state machines) | grep | P1 |
| C4 | `scroll-behavior: smooth` is gated inside `@media (prefers-reduced-motion: no-preference)`; `scrollIntoView` passes `behavior: mq.matches ? 'auto' : 'smooth'` | grep | P1 |
| C5 | View Transitions gated — the UA default is **not** auto-disabled: `::view-transition-group/old/new(*) { animation: none !important }` under `reduce` | grep `view-transition` | P1 |
| C6 | Every `animation-timeline` / `scroll-timeline` / `view-timeline` declaration sits inside a `no-preference` block | grep | P1 |
| C7 | Reduced motion **substitutes** (opacity cross-fade) rather than deleting where motion carried meaning; **does not substitute blur** (WCAG erratum amended `motion animation` "to not exclude blurring") | Design review | P1 |
| C8 | The reduced/static state is the **base**, with motion added inside `no-preference` (no-consent model) | CSS structure review | P2 |
| C9 | Uses `mq.addEventListener('change', …)`, not `addListener()`; the media query string has **parentheses** | grep `matchMedia(` | P1 |
| C10 | **2.2.2 (A):** every auto-starting motion >5 s in parallel with other content has a pause/stop/hide **mechanism**. Hover/focus-to-pause does **not** count. Includes third-party ads (Conformance Req. 5). Apply the **28 June 2026** Understanding update: "scrolling an element into view" counts as starting automatically | Watch the page 10 s untouched; list everything that moves; time each; Tab for a control. `document.getAnimations().filter(a=>a.playState==='running')` | **P0** |
| C11 | **2.2.2 (A) auto-updating:** live tickers/feeds have a control **immediately** — there is **no 5-second exception** for auto-updating | Manual | **P0** |
| C12 | **2.3.1 (A):** nothing flashes >3×/second. ≤3/s **automatically passes**; above that run PEAT or Harding. (341×256 @1024×768 is the *10-degree field*; the limit is **25% of it ≈ 21,824 px²**) | Screen-record and count; see `ada/media-and-motion.md` §5 for the console sweep | **P0** |
| C13 | **1.4.2 (A):** audio auto-playing >**3 seconds** has a pause/stop or independent volume control | Load with headphones; Tab up to 5 times | **P0** |
| C14 | **2.5.8 (AA):** interactive targets are ≥24×24 CSS px **throughout** any motion (hover-scale from an undersized rest state fails) | Measure the rest state, not the hover state | P1 |
| C15 | **2.3.3 (AAA):** interaction-triggered motion animation is disableable; parallax is named explicitly | `prefers-reduced-motion` satisfies via C39/SCR40 | P2 (P0 if AAA is claimed) |
| C16 | A site-wide motion toggle exists (W3C endorses "a **single** mechanism … that affects all these elements at the same time"), defaulting to the OS preference and persisting overrides | Manual | P2 |
| C17 | No large-scale zoom, spin, 2.5D plane-shift, or multi-speed parallax; motion is small relative to viewport (Head: small-button 3D rotation safe, full-screen wipe unsafe) | Design review; express as a ratio to viewport, flagged as a heuristic | P1 |
| C18 | No ambient/looping motion in the peripheral field beside body copy | Design review | P1 |
| C19 | Audit template does not test **4.1.1 Parsing** ("Obsolete and removed" in WCAG 2.2) and does not audit against **WCAG 3.0** (Working Draft, 03 Mar 2026) | Template review | P2 |

### 10.4 Duration, easing, restraint

Durations are measured, not eyeballed. Two methods, either acceptable: **static** — `grep -rn 'transition-duration\|animation-duration\|transition:\|animation:' CSS` and read the declared value (watch for shorthand, where duration is the first time value and delay the second); **runtime** — Performance panel → record the interaction → **Animations track**, click the animation and read its duration, or run `document.getAnimations().map(a => ({ name: a.animationName ?? a.transitionProperty, ms: a.effect.getTiming().duration }))` in the console while it plays. The runtime reading is authoritative where JS or a library sets the duration and no CSS declaration exists.

| # | Check | Target | Sev |
|---|---|---|---|
| D1 | Micro-interactions (hover, focus, button) | **70–120 ms**, ceiling 150 ms — grep `transition-duration` on the hover/focus rule, or Animations track after hovering | P2 |
| D2 | Small element enter/exit | **150–200 ms**, ceiling 250 ms — grep `animation-duration` on the enter/exit `@keyframes` user, or Animations track while toggling the element | P2 |
| D3 | Modal / dialog / panel | **250–400 ms**, ceiling 500 ms — Animations track while opening the dialog; `document.getAnimations()` during the open transition catches JS-driven durations | P2 |
| D4 | Full-page / view transition | **300–500 ms**, ceiling 600 ms — Animations track across the navigation; for View Transitions read the `::view-transition-*` pseudo-element durations (grep, or the Animations track, which lists them by name) | P2 |
| D5 | Entering uses `ease-out`; exiting uses `ease-in`; on-screen movement uses `ease-in-out`/standard | grep `transition-timing-function` / `animation-timing-function` | P2 |
| D6 | `cubic-bezier(0.4, 0, 0.2, 1)` flagged as **Material 2 legacy**; M3 standard is `cubic-bezier(0.2, 0, 0, 1)` | grep | P2 |
| D7 | `linear()` spring approximations have a `@supports not (transition-timing-function: linear(0, 1))` fallback (Baseline widely available 2026-06-11) | grep | P2 |
| D8 | No claim that CSS `spring()` exists — **zero tracked web-features entries** | Doc review | P2 |
| D9 | Stagger ~20–50 ms per item, total sequence ≤~300–500 ms; delays zeroed under `reduce` *(unverified practitioner numbers — label them)* | Read the code | P2 |
| D10 | Kill-list applied (§7.8): no animation with no state change, seen >5×/session, delaying requested content, looping infinitely without being a progress indicator, or failing "why is this here?" in one non-aesthetic sentence | Design review | P1 |

### 10.5 SVG

| # | Check | Threshold / method | Sev |
|---|---|---|---|
| E1 | No looping SMIL on transform (not GPU-composited in WebKit) | grep `<animateTransform`, `<animate` | P1 |
| E2 | SMIL reduced-motion handled via `svg.pauseAnimations()` (media queries do **not** reach SMIL) | grep | **P0** |
| E3 | No animation of `cx`, `cy`, `r`, `x`, `y`, `width`, `height`, `viewBox`, `points`, `d` in a loop | grep `@keyframes` and `attributeName=` | **P0** |
| E4 | `<filter>` does not intersect animated content; filter region tightened from the default `-10% / 120%` | grep `<filter` | P1 |
| E5 | Line drawing uses `pathLength` normalization rather than JS `getTotalLength()`; ≤~10 concurrent draws | grep `stroke-dasharray` / `getTotalLength` | P1 |
| E6 | `d` morphing has a Safari path (CSS `d` is **unsupported through Safari 27 / iOS 26.6**, 79.89% global) — an attribute-writing morph library works | grep `d: path(` | **P0** if Safari-facing |
| E7 | Element count per animated SVG | <1,500 pass · >1,500 flag · **>5,000 fail** (recommend canvas) | P1 |
| E8 | No `data:image/` payloads inside `.svg` | any hit >10 KB → **fail** | **P0** |
| E9 | File sizes: icon <2 KB · logo <15 KB · hero <50 KB · sprite <30 KB | bands in §6.8 | P1 |
| E10 | `viewBox` present on every SVG that scales or ships via `<img>`/`background-image`; `preserveAspectRatio="none"` is almost always a bug | grep | P1 |
| E11 | SVGO 4.x in the pipeline with `removeDesc: false`, `cleanupIds: false` (sprites), `inlineStyles`/`minifyStyles` off for animated files, precision 2–3 | Read `svgo.config.mjs` | P1 |
| E12 | Informative SVG has `role="img"` + `<title>` as first child; decorative has `aria-hidden="true"`; no stray `tabindex="0"` on non-interactive SVG. **`focusable="false"` is neutral — do not flag either way** | Accessibility tree inspection | P1 |
| E13 | `.svg` responses are gzip/brotli compressed (`image/svg+xml` is frequently missing from server compressible-MIME lists; SVG compresses 60–80%) | Response headers | P1 |
| E14 | ≤2–3 concurrent Lottie players; none autoplaying above the fold; WASM/canvas renderer preferred over the DOM-mutating SVG renderer (`lottie-web` ≈60 KB gzip) | Script audit | P1 |

### 10.6 Measurement gates — run these; do not accept assertions

| # | Gate | Pass criterion |
|---|---|---|
| F1 | **Calibrate CPU throttling** — Settings > Throttling > CPU throttling presets > Calibrate | Never hardcode 4×; target `benchmarkIndex` **125–800** (mid-tier mobile) |
| F2 | **Record a trace and read the Frames track** during a full-page scroll and during each animation | **<5% red (dropped) frames**; investigate any sustained yellow (partially presented) run |
| F3 | **Animations track** — check every animation for a red triangle | Zero non-compositing animations, or each has a written justification |
| F4 | **Rendering → Paint flashing** during scroll | No full-viewport green; no fixed header flashing while the body scrolls |
| F5 | **Rendering → Layer Borders** + **Layers panel status bar** (Chrome 136+) | Bounded layer count; GPU memory well under max in Frame rendering stats |
| F6 | **Rendering → Scrolling Performance Issues** | No teal-highlighted non-passive scroll listeners |
| F7 | **LoAF observer** in-page while scrolling and interacting | `blockingDuration` low; `scripts[].forcedStyleAndLayoutDuration === 0` |
| F8 | **Field INP** via `web-vitals@6` attribution build | **p75 ≤ 200 ms** — and check **p95** too; jank concentrates in the tail |
| F9 | **Automated regression gate** — Puppeteer trace counting `DroppedFrame` vs `DrawFrame` (both the `.frame` and plain `devtools.timeline` categories; drive scroll with `page.mouse.wheel`, **never `window.scrollTo` inside `page.evaluate`** — it bypasses the input pipeline and under-reports), or `@paulirish/trace_engine` | Dropped % under an agreed budget |
| F10 | **Never accept as evidence:** a Lighthouse score alone · a `longtask`-only measurement · an unthrottled desktop run · an rAF-delta number alone · the removed "Core Web Vitals overlay" | — |

---

## 11. Highest-leverage findings, ranked

1. **Fail-open reveals are the #1 audit item** — see `dynamic-loading.md` §3. A static `.reveal { opacity: 0 }` base undone only by JS is a permanent-blank-page hazard.
2. **Lighthouse cannot see animation jank.** All five scored metrics are load metrics; every animation-relevant audit is weight 0. A page can score 100 and drop 40% of frames on every scroll. The Frames track and the Animations track are the real evidence.
3. **Reduced-motion compliance almost always fails on the JS side.** The CSS media query does not reach WAAPI, JS libraries, Lottie, video, canvas, SMIL, or smooth-scroll libraries. Test by emulating `reduce` **and then scrolling the whole page with JS enabled.**
4. **The 28 June 2026 WCAG 2.2.2 Understanding update** makes scroll-into-view-triggered motion a potential **Level A** failure, not just AAA — legally material in the EU under the EAA (enforceable since 28 June 2025).
5. **Of Chromium's nine `kCompositableProperties`, only `transform`/`translate`/`rotate`/`scale`/`opacity` are truly free.** `csstriggers.com` is now a content farm with wrong data.
