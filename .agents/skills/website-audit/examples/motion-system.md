# A small, opinionated motion system

Drop-in CSS custom properties plus the rules for using them. Framework-agnostic, no build step.

The point of a motion system in an audit context is not aesthetics — it is that **an auditor can hold a site to it**. A site with 40 different durations scattered across 12 stylesheets cannot be reviewed; a site with six tokens can be reviewed in ten minutes. §4 below is written as a budget statement precisely so it can be checked.

Sources: IBM Carbon (live docs, updated 13 Aug 2026) and Material Design 3 (`material-web` v0.192). Cross-references: `../references/animation-and-motion.md` §7.7 (the full duration/easing research), §7.3 (substitute vs delete), §2 (the cheap/expensive decision table), `scroll-reveal.md`, `../references/dynamic-loading.md`.

⚠️ **Apple HIG deliberately does not publish a millisecond duration table.** Its guidance is qualitative plus a strong "support Reduce Motion" directive. Do not fabricate Apple ms numbers.

---

## 1. The tokens

```css
:root {
  /* ── Durations ─────────────────────────────────────────────────────────
     Derived from IBM Carbon's published token values. Six is enough;
     more than eight and nobody can tell them apart or remember which is which. */
  --motion-duration-xs: 70ms;    /* Carbon duration-fast-01     */
  --motion-duration-sm: 110ms;   /* Carbon duration-fast-02     */
  --motion-duration-md: 150ms;   /* Carbon duration-moderate-01 */
  --motion-duration-lg: 240ms;   /* Carbon duration-moderate-02 */
  --motion-duration-xl: 400ms;   /* Carbon duration-slow-01     */
  --motion-duration-2xl: 700ms;  /* Carbon duration-slow-02     */

  /* ── Easings ───────────────────────────────────────────────────────────
     Carbon "productive" curves. Use the expressive set only for
     marketing surfaces, never for tools people use all day. */
  --motion-ease-standard: cubic-bezier(0.2, 0, 0.38, 0.9);  /* visible start → end */
  --motion-ease-entrance: cubic-bezier(0, 0, 0.38, 0.9);    /* appearing           */
  --motion-ease-exit:     cubic-bezier(0.2, 0, 1, 0.9);     /* leaving permanently */
  --motion-ease-linear:   linear;                            /* continuous, indefinite */

  /* Expressive alternatives (Carbon), for hero/marketing only */
  --motion-ease-standard-expressive: cubic-bezier(0.4, 0.14, 0.3, 1);
  --motion-ease-entrance-expressive: cubic-bezier(0, 0, 0.3, 1);
  --motion-ease-exit-expressive:     cubic-bezier(0.4, 0.14, 1, 1);

  /* ── Travel ────────────────────────────────────────────────────────────
     Cap distance, not just duration. Vestibular triggers scale with
     distance covered relative to the viewport, not with absolute pixels. */
  --motion-travel-sm: 0.5rem;
  --motion-travel-md: 1rem;
  --motion-travel-lg: 2rem;

  /* ── Stagger ───────────────────────────────────────────────────────────
     ⚠️ Unverified practitioner heuristic — no design system publishes a
     stagger table. Label it as a heuristic in any report. */
  --motion-stagger: 40ms;
  --motion-stagger-max: 5;   /* cap the sequence: 5 × 40ms = 200ms of offset, max */
}
```

### If you prefer the Material Design 3 easing set

```css
:root {
  --md-sys-motion-easing-standard:              cubic-bezier(0.2, 0, 0, 1);
  --md-sys-motion-easing-standard-accelerate:   cubic-bezier(0.3, 0, 1, 1);
  --md-sys-motion-easing-standard-decelerate:   cubic-bezier(0, 0, 0, 1);
  --md-sys-motion-easing-emphasized-accelerate: cubic-bezier(0.3, 0, 0.8, 0.15);
  --md-sys-motion-easing-emphasized-decelerate: cubic-bezier(0.05, 0.7, 0.1, 1);
}
```

M3's full duration scale, if you want finer granularity: `short1` 50 · `short2` 100 · `short3` 150 · `short4` 200 · `medium1` 250 · `medium2` 300 · `medium3` 350 · `medium4` 400 · `long1` 450 · `long2` 500 · `long3` 550 · `long4` 600 · `extra-long1` 700 · `extra-long2` 800 · `extra-long3` 900 · `extra-long4` 1000.

⚠️ **`cubic-bezier(0.4, 0, 0.2, 1)` is the Material 2 curve**, now labelled **`easing-legacy`** by Google. It is still the most-copied easing value on the web. If you find it, it is not wrong, but it is not "the Material standard" — M3's standard is `cubic-bezier(0.2, 0, 0, 1)`.

---

## 2. Which token for what

| Situation | Duration | Easing | Travel | Rationale |
|---|---|---|---|---|
| **Micro-interaction** — hover, focus ring, button press, toggle, checkbox tick | `--motion-duration-xs` … `--motion-duration-sm` (**70–120 ms**, ceiling 150 ms) | `--motion-ease-entrance` (`ease-out`) on user input | none | Carbon: *"Do micro-interactions fall within a static duration ranging from 90–120 ms?"* and *"Do micro-interactions use `ease-out` on user input?"* Under ~70 ms reads as instantaneous — which is often correct |
| **Element transition, small** — dropdown, tooltip, popover, inline expand | `--motion-duration-md` … `240ms` (**150–200 ms**, ceiling 250 ms) | `--motion-ease-entrance` entering, `--motion-ease-exit` leaving | `--motion-travel-sm` | Carbon: entering *"quickly appears and slows down to a stop"*; exiting *"speeds up as it exits… implying that its departure is permanent"* |
| **Element transition, medium** — toast, snackbar, accordion, card expand | `--motion-duration-lg` (**200–300 ms**, ceiling 400 ms) | `--motion-ease-standard` if visible throughout | `--motion-travel-md` | Carbon: *"Use `standard-easing` when an element is visible from the beginning to the end of a motion"* |
| **Element transition, large** — modal, dialog, side panel, bottom sheet | `--motion-duration-xl` (**250–400 ms**, ceiling 500 ms) | `--motion-ease-standard` — **not** exit easing for a side panel | `--motion-travel-lg` | Carbon on panels: *"implying that it would come to rest just outside the view, and ready to be recalled"* |
| **Page / view transition** | `--motion-duration-xl` (**300–500 ms**, ceiling 600 ms) | `--motion-ease-standard` | ≤ `--motion-travel-lg`; **never a full-screen wipe** | Head: a full-screen wipe *"covering the entire screen likely would"* trigger |
| **Attention** — error shake, new-item highlight, save confirmation | `--motion-duration-sm` … `--motion-duration-lg`, **one iteration only** | `--motion-ease-standard` | `--motion-travel-sm` | Attention motion must be brief and finite. An infinite attention loop directly implicates WCAG 2.2.2 |
| **Background scrim / dim** | `--motion-duration-2xl` (**~700 ms**) | `--motion-ease-standard` | none — `opacity` only | Carbon `duration-slow-02` |
| **Continuous indefinite** — spinner, progress, marquee | n/a (loops) | `--motion-ease-linear` | none | Carbon: *"Strictly linear movement appears unnatural to the human eye"* — **except** here, where linear is correct. But a loop >5 s in parallel with other content needs a pause control (2.2.2) |

**Scale duration with size and distance, sub-linearly.** Carbon: *"Motion's duration should be dynamic based on the size of the animation; the larger the change in distance (traveled) or size (scaling) of the element, the longer the animation takes."* A tooltip appearing 8 px away and a sheet sliding 600 px should not share a duration. Doubling the distance does **not** double the duration.

**The two hard bounds:** under **~70 ms reads as instantaneous**; over **~500 ms on a routine interaction feels sluggish**; **>1000 ms** is only ever acceptable for one-time onboarding, never for a repeated action.

### Easing anti-patterns

Carbon, verbatim: *"Avoid easing curves that are unnatural, distracting, or purely decorative… Do not use easing curves that suggest bounce, stretch, or sudden stops."*

**Bounce and overshoot are an accessibility concern, not just taste** — they add travel distance and a direction reversal, mapping directly onto Val Head's "distance covered" and "mismatched direction" vestibular triggers. If you ship them, gate them behind `no-preference`.

Spring-like easing without a JS library, using `linear()` — **Baseline widely available since 2026-06-11** (Chrome/Edge 113, Firefox 112, Safari/iOS 17.2):

```css
:root {
  --motion-ease-spring: linear(
    0, 0.006, 0.025, 0.101, 0.539, 0.826, 0.949, 1.001,
    1.017, 1.012, 1.001, 0.997, 0.999, 1
  );
}
@supports not (transition-timing-function: linear(0, 1)) {
  :root { --motion-ease-spring: cubic-bezier(0.05, 0.7, 0.1, 1); }
}
```

Values passing 1.0 are overshoot — swap to a monotonic curve under `reduce`.

⚠️ **CSS `spring()` is NOT a shipping web platform feature.** A webstatus.dev query for `spring` returns **zero tracked features**. Use `linear()`, or a JS library for genuinely interruptible physical springs.

### One non-negotiable constraint on every token above

**Only apply these to `transform` and `opacity`.** Every duration and easing token in this file is worthless if it is attached to `width`, `height`, `top`, `left`, `box-shadow`, or `filter: blur()` — those animate through Layout or Paint on the main thread and will drop frames regardless of how tasteful the curve is. See `../references/animation-and-motion.md` §2 for the full decision table, and verify with the DevTools Animations track (a red triangle = non-composited).

---

## 3. The `prefers-reduced-motion` strategy: substitute, don't delete

**Reduced motion is the default state; motion is the enhancement.** Tatiana Mac: *"this is operating on a no-consent model… The user hasn't necessarily opted into animations. They just haven't checked 'Reduce motion.'… it's equally possible the user doesn't know about this setting."* And: *"Defaulting to this latter approach will mean that all users will default to no animation, including users whose browsers won't recognise the media query."*

This reconciles with WebKit's warning against reducing too much (*"removing the animation entirely may make the interface confusing or unusable… consider serving an alternate, simpler animation, or display another visual indicator to convey the intended meaning"*): **author the static/reduced state as the base, add motion inside `no-preference`, and within `reduce` substitute rather than delete wherever motion carried meaning.**

### 3.1 The substitution hierarchy

Apply the first rule that fits:

| # | Rule | Example |
|---|---|---|
| 1 | **Keep the state change, remove the travel** | `translate: 0 16px → 0` becomes an `opacity` cross-fade |
| 2 | **Shrink amplitude and duration** | 24 px slide → 4 px slide; 400 ms → 150 ms |
| 3 | **`animation: none`** — only for pure decoration with no state behind it | Ambient background loop |
| 4 | **Gate behind an explicit play control plus a static/text equivalent** — when the motion *is* the information | A process animation explaining a workflow |

⚠️ **Never substitute blur.** A WCAG erratum amended the `motion animation` definition **"to not exclude blurring."** Current text: *"Motion animation does not include changes of color, blurring, or opacity which do not change the perceived size, shape, or position."* WebKit also lists animated blur as a zoom-class vestibular trigger. Substitute **opacity or colour cross-fades** only. Val Head's 2015 guidance that blur is safe has been superseded on this point.

### 3.2 The token override

```css
/* Base = reduced. Motion is opt-in. */
:root {
  --motion-travel-sm: 0;
  --motion-travel-md: 0;
  --motion-travel-lg: 0;
  --motion-stagger: 0ms;
  --motion-ease-spring: var(--motion-ease-standard);   /* no overshoot by default */
}

@media (prefers-reduced-motion: no-preference) {
  :root {
    --motion-travel-sm: 0.5rem;
    --motion-travel-md: 1rem;
    --motion-travel-lg: 2rem;
    --motion-stagger: 40ms;
    --motion-ease-spring: linear(
      0, 0.006, 0.025, 0.101, 0.539, 0.826, 0.949, 1.001,
      1.017, 1.012, 1.001, 0.997, 0.999, 1
    );
  }
}
```

Because travel and stagger are tokens, a component written once works in both modes: it keeps its cross-fade, loses its slide, and needs no `reduce` branch of its own.

```css
/* Written once. Correct in both modes. */
.toast {
  opacity: 0;
  translate: 0 calc(-1 * var(--motion-travel-md));
  transition: opacity var(--motion-duration-lg) var(--motion-ease-entrance),
              translate var(--motion-duration-lg) var(--motion-ease-entrance);
}
.toast[data-visible] { opacity: 1; translate: 0 0; }
```

### 3.3 The safety net — and why it is not the strategy

```css
/* Net, NOT strategy. A site whose only implementation is this scores YELLOW, not pass. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

web.dev files this under **"(Bonus) Forcing reduced motion on all websites"** — as something to *"inject … into every web page you visit"* via an extension, *"use at your own risk."* Andy Bell's 2023 "A (more) Modern CSS Reset" **no longer contains** the reduced-motion block his 2020 reset had.

Why it is a net and not a strategy:

- **`0.01ms`, never `animation: none`.** With `none`, `animationend` and `transitionend` **never fire** and any state machine depending on them breaks. web.dev: *"the more radical `animation: none !important;` approach wouldn't work."*
- **It reaches CSS and nothing else.** web.dev: *"it can't stop motion that was initiated using the Web Animations API."* Same for every JS animation library, Lottie, `rAF`, canvas/WebGL, `<video autoplay>`, animated GIF/WebP/AVIF, `<marquee>`, SMIL, and smooth-scroll libraries.
- `!important` on `*` is unoverridable downstream, and web.dev's variant's `background-attachment: initial` kills `fixed` backgrounds — a layout change smuggled into a motion rule.
- It makes **no substitution decisions**. Every animation is flattened identically, meaning and decoration alike.

### 3.4 The JS half — mandatory, not optional

The CSS media query does not reach any of these. Each needs its own gate:

| Motion source | Gate |
|---|---|
| WAAPI / `element.animate()` | `matchMedia`, then `document.getAnimations().forEach(a => a.cancel())` |
| JS animation libraries (timeline / spring / scroll-trigger categories) | the library's own reduced-motion API, or a `matchMedia` teardown |
| Lottie / dotLottie / state-machine runtimes | render frame 0 as a still — `autoplay` fires regardless |
| `<video autoplay loop muted>` hero | swap to `poster` |
| Animated GIF / APNG / WebP / AVIF | `<picture>` + `media="(prefers-reduced-motion: no-preference)"` |
| Autoplaying carousels, tickers, marquees | `matchMedia` **plus** a 2.2.2 pause control regardless |
| Canvas / WebGL loops | pause the `rAF` loop |
| Smooth-scroll libraries | destroy/disable the instance |
| SMIL | `svg.pauseAnimations()` — media queries do **not** reach SMIL |
| `scroll-behavior: smooth` | gate it (below) |
| `scrollIntoView({behavior:'smooth'})` | `behavior: mq.matches ? 'auto' : 'smooth'` |
| **View Transitions** | the UA default cross-fade/morph is **not** auto-disabled (below) |
| **Scroll-driven animations** (`animation-timeline`) | honoured by CSS, but **frequently forgotten** — put every declaration inside `no-preference` |

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

```js
// One shared signal for every JS motion system on the page.
export const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');  // parentheses MANDATORY
export const prefersReducedMotion = () => motionQuery.matches;

motionQuery.addEventListener('change', apply);   // addEventListener, not deprecated addListener
apply();                                          // and run once on load

function apply() {
  document.documentElement.classList.toggle('reduced-motion', motionQuery.matches);
  if (!motionQuery.matches) return;
  document.getAnimations().forEach(a => a.cancel());
  document.querySelectorAll('svg').forEach(svg => { svg.pauseAnimations(); svg.setCurrentTime(0); });
  document.querySelectorAll('video[autoplay]').forEach(v => { v.pause(); v.currentTime = 0; });
  // …plus: stop rAF loops, render Lottie frame 0, destroy smooth-scroll instances.
}
```

⚠️ `matchMedia('prefers-reduced-motion: reduce')` **without parentheses silently never matches** — a bug that looks exactly like "the user hasn't enabled it."

Avoid downloading the motion CSS at all for `reduce` users:

```html
<link rel="stylesheet" href="animations.css" media="(prefers-reduced-motion: no-preference)">
```

Server-side: the `Sec-CH-Prefers-Reduced-Motion` client hint (requires an `Accept-CH` opt-in).

---

## 4. The motion budget

**Statement an auditor can hold a site to.** Each line is a testable assertion; the right-hand column says how to falsify it. Any line that fails is a finding.

| # | Budget statement | How to falsify |
|---|---|---|
| 1 | **Every animated property is `transform` or `opacity`.** Exceptions are individually justified in writing. | DevTools → Performance → **Animations track**: any red triangle is a violation; Summary gives the compositing-failure reason |
| 2 | **No animation exceeds its category ceiling** — 150 ms micro, 250 ms small, 400 ms medium, 500 ms modal, 600 ms page transition. | grep every `transition-duration` / `animation-duration` and bucket them; anything over 600 ms must be a scrim, a spinner, or a written exception |
| 3 | **Total distinct duration values on the site ≤ 8.** More than that and the system is decorative, not systematic. | `grep -rhoE '[0-9]+m?s' *.css \| sort -u \| wc -l` |
| 4 | **Total distinct easing curves ≤ 5** (standard, entrance, exit, linear, one spring). | `grep -rhoE 'cubic-bezier\([^)]*\)\|linear\([^)]*\)' *.css \| sort -u` |
| 5 | **No looping animation runs longer than 5 seconds in parallel with other content without a pause/stop/hide mechanism.** Hover- or focus-to-pause does **not** count. | Watch the page 10 s untouched, list everything that moves, time each, Tab for a control. `document.getAnimations().filter(a => a.playState === 'running')` |
| 6 | **No motion is triggered purely by scrolling an element into view and then loops.** Per the **28 June 2026** Understanding 2.2.2 update, "scrolling an element into view" counts as starting automatically — this is a **Level A** exposure, not just AAA. | Scroll to each reveal and watch for 6 s |
| 7 | **Nothing flashes more than 3 times per second.** ≤3/s automatically passes with no tool. | Screen-record and count; above 3/s run PEAT or Harding |
| 8 | **Reduced motion is honoured by every motion system, not just CSS.** | Firefox `about:config` → `ui.prefersReducedMotion = 1`, **then scroll the entire page with JS enabled**. This is the step everyone skips |
| 9 | **The reduced state substitutes rather than deletes wherever motion carried meaning, and never substitutes blur.** | Side-by-side review of each component in both modes |
| 10 | **No motion covers a large fraction of the viewport** — no full-screen wipes, large-scale zooms, spins, or 2.5D plane shifts; no multi-speed parallax. *(Expressed as a ratio to viewport per Val Head; the circulating "25–30% of viewport" numbers are ⚠️ practitioner heuristics, not standards.)* | Design review against Head's endpoints: small-button 3D rotation = safe; full-screen wipe = unsafe |
| 11 | **No ambient or looping motion in the peripheral field beside body copy.** | Design review — floating particles, animated gradients, ambient loops next to text |
| 12 | **Stagger sequences complete within ~300–500 ms end-to-end**, with per-item delay reduced as item count rises. *(⚠️ heuristic)* | Read the code: `itemCount × perItemDelay + duration` |
| 13 | **Every animation survives the "why is this here?" test in one non-aesthetic sentence.** | Ask the team; write down the answer. Carbon: *"Is your motion frequently noticed by average users? If so, consider removing or minimizing it"* |
| 14 | **No animation is seen more than ~5 times per session.** The cost is paid every time; the novelty benefit is paid once. | Count occurrences on a typical task path |
| 15 | **No animation delays access to content the user asked for.** No above-the-fold entrance animations, no scroll-reveal on body copy, no page-transition curtains. | Time from navigation to readable content with motion enabled |
| 16 | **Under a full-page scroll at calibrated CPU throttling, dropped frames are <5%.** | Performance panel Frames track (red = dropped, yellow = partially presented), or a Puppeteer trace counting `DroppedFrame` vs `DrawFrame`. Calibrate throttling first: Settings > Throttling > CPU throttling presets > Calibrate; target `benchmarkIndex` **125–800** |
| 17 | **No forced-reflow violations in the console during a full-page scroll.** | Open the console, scroll top to bottom, watch for `[Violation] Forced reflow while executing JavaScript took Nms` |
| 18 | **Promoted layer count and GPU memory are bounded.** | Rendering → Layer Borders; Layers panel status bar total count + memory (Chrome 136+); Frame rendering stats GPU memory used/max |

**Do not accept as evidence for any line above:** a Lighthouse score (every scored metric is a load metric; every animation-relevant audit is weight 0 — a page can score 100 and drop 40% of frames on every scroll), a `longtask`-only measurement, an unthrottled desktop run, an rAF-delta number alone, or the removed "Core Web Vitals overlay". See `../references/animation-and-motion.md` §8.

---

## 5. The kill-list

Recommend **deleting** the animation if any of these is true. This is the fastest way to shrink a bloated motion layer, and it is usually more valuable than optimising it.

1. **No state change behind it.** Fails WCAG's `essential` test, whose definition uses a conjunction: `essential` = "if removed, would fundamentally change the information or functionality of the content, **and** information and functionality cannot be achieved in another way that would conform." "It looks nicer" and "it's our brand" both fail.
2. **Seen more than ~5 times per session.**
3. **It delays access to content the user asked for.**
4. **Infinite loop and not a progress indicator.** Directly implicates WCAG 2.2.2.
5. **In peripheral vision while reading.**
6. **Moves at a rate or direction the user did not command** — scroll-jacking, parallax, scroll-velocity-decoupled reveals.
7. **Scales, spins, blurs, or plane-shifts across a large fraction of the viewport.**
8. **You cannot answer "why is this here?" in one non-aesthetic sentence.** Tatiana Mac's escalation: *"Why is this animation critical? / What other ways can we serve up the critical animation? / If the animation can't load for someone or they can't see it, what was your plan?"*
9. **It would fail at 3× duration on a slow device.** Load-bearing motion must survive being slow.

Carbon's one-line version, verbatim: *"**Is your motion unobtrusive?** The best interface motion may go unnoticed."*

And the counterweight, so the recommendation stays proportionate — Val Head: *"Not one person I spoke with said that they want to see all interface animation eliminated."* Motion reduces cognitive load, improves decision-making, aids learning of spatial relationships, and prevents change blindness. The goal is a small, deliberate, verifiable motion layer — not zero motion.
