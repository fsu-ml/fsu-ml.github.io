# Scroll reveal — reference implementations

Copy-pasteable, plain HTML/CSS/JS. No framework, no build step, no dependencies.

**Governing principle: fail open.** Content is visible by default. Motion is opt-in, gated on both feature support and motion preference. Every failure path — no JS, no scroll-driven-animation support, `prefers-reduced-motion: reduce`, CSP block, ad-blocker, 3G timeout, non-rendering crawler, element already in the viewport, deep link mid-page, back-navigation — resolves to **visible content**.

Reference: `../references/dynamic-loading.md` §3 (the fail-open requirement and its test procedure), §2 (scroll-driven animation API and support), §1 (IntersectionObserver). Compositing rules: `../references/animation-and-motion.md` §2. Duration/easing tokens: `motion-system.md`.

**Support reality check (19 Aug 2026):** scroll-driven animations are **Baseline Limited, 85.43% global, Chrome/Edge 115+, Safari 26.0+ (threaded in 26.4), and never shipped to Firefox stable** — still flagged off in Firefox 154. Every example below therefore ships a real fallback rather than treating CSS-only as sufficient.

---

## 1. CSS-only scroll-driven reveal, with `@supports` progressive enhancement

```html
<section class="reveal">
  <h2>Section heading</h2>
  <p>Body copy that must be readable even if every script on this page fails.</p>
</section>
```

```css
@keyframes reveal-in {
  from { opacity: 0; transform: translateY(2rem); }
  to   { opacity: 1; transform: none; }
}

/* Motion gate OUTSIDE, feature gate INSIDE.
   If either fails, `.reveal` receives no declarations at all. */
@media (prefers-reduced-motion: no-preference) {
  @supports (animation-timeline: view()) {
    .reveal {
      animation: reveal-in linear both;   /* `both` fill supplies the from-state around the range */
      animation-timeline: view();         /* MUST come after the shorthand — it is reset-only in it */
      animation-range: entry 0% cover 40%;
    }
  }
}
```

**Prevents:** the permanent-blank-page failure — the hidden state lives only inside `@keyframes`, so no engine, crawler or blocked-script scenario can leave content at `opacity: 0`. It also prevents the flash-of-hidden-content on load, because an element already past `entry` never gets the backwards fill. And it satisfies WCAG 2.3.3 for free: under `reduce` the whole block is dropped and content simply appears.

**Two traps this encodes.** `animation-timeline` is **reset-only in the `animation` shorthand**, so writing it *before* `animation:` silently disables the timeline (and `animation-duration` is ignored on a scroll timeline regardless). And an axis with no scrollbar produces an **inactive timeline with zero progress and no error** — if nothing animates, check that the scroller actually scrolls on that axis before suspecting the keyframes.

**Verify:** DevTools → Rendering → Emulate CSS media feature `prefers-reduced-motion: reduce` → reload → content visible, no motion. Then Firefox stable 154 → content visible, no motion. Then Performance → Animations track → no red triangle (composited).

---

## 2. IntersectionObserver fallback, written to fail open

The hidden-then-reveal behaviour is **opted into by JavaScript**. Without JS, the `.js-reveal` class is never added, the descendant selectors never match, and everything is visible.

```html
<!doctype html>
<html lang="en">
<head>
  <!-- BEFORE the stylesheet. If this line never runs, nothing is ever hidden. -->
  <script>document.documentElement.classList.add('js-reveal');</script>
  <link rel="stylesheet" href="site.css">
  <noscript><style>
    .reveal { opacity: 1 !important; transform: none !important; }
  </style></noscript>
</head>
<body>
  <section class="reveal">…</section>
  <section class="reveal">…</section>
</body>
</html>
```

```css
@keyframes reveal-in {
  from { opacity: 0; transform: translateY(2rem); }
  to   { opacity: 1; transform: none; }
}

@media (prefers-reduced-motion: no-preference) {

  /* Preferred path: no JS at all. */
  @supports (animation-timeline: view()) {
    .reveal {
      animation: reveal-in linear both;
      animation-timeline: view();
      animation-range: entry 0% cover 40%;
    }
  }

  /* Fallback path: only when the CSS feature is missing AND JS ran. */
  @supports not (animation-timeline: view()) {
    .js-reveal .reveal {
      opacity: 0;
      transform: translateY(1rem);
    }
    .js-reveal .reveal.is-revealed {
      opacity: 1;
      transform: none;
      transition: opacity var(--motion-duration-md, 240ms) var(--motion-ease-entrance, cubic-bezier(0, 0, 0.38, 0.9)),
                  transform var(--motion-duration-md, 240ms) var(--motion-ease-entrance, cubic-bezier(0, 0, 0.38, 0.9));
    }
  }
}
```

```js
// reveal.js — load with `defer`. Safe to run twice; safe to run with no matches.
(function () {
  'use strict';

  // Bail out entirely if the CSS path is handling it, or motion is unwanted,
  // or the browser is too old for IntersectionObserver. In all three cases the
  // stylesheet above leaves content visible.
  if (CSS.supports('animation-timeline', 'view()')) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (!('IntersectionObserver' in window)) {
    document.documentElement.classList.remove('js-reveal');   // un-hide, then stop
    return;
  }

  var targets = document.querySelectorAll('.reveal');
  if (!targets.length) return;

  var io = new IntersectionObserver(function (entries, observer) {
    for (var i = 0; i < entries.length; i++) {
      var entry = entries[i];
      // threshold is 0, so a bare isIntersecting check is safe HERE.
      // With any non-zero threshold, compare intersectionRatio explicitly:
      // Blink and WebKit implement isIntersecting as `thresholdIndex > 0`.
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('is-revealed');
      observer.unobserve(entry.target);   // one-shot: stop paying per-frame geometry cost
    }
  }, {
    threshold: 0,                       // never 0.5/1.0 — sections taller than the
                                        // viewport can never reach those ratios
    rootMargin: '0px 0px -10% 0px'      // px and % ONLY. em/rem/vh/calc() throw SyntaxError.
                                        // ⚠️ percentages resolve against the root rect's
                                        // WIDTH, including top/bottom.
  });

  for (var j = 0; j < targets.length; j++) io.observe(targets[j]);

  // bfcache: IO registrations survive a back-navigation but emit no new entries.
  window.addEventListener('pageshow', function (e) {
    if (!e.persisted) return;
    for (var k = 0; k < targets.length; k++) targets[k].classList.add('is-revealed');
  });

  // If the user turns Reduce Motion on mid-session, reveal everything immediately.
  window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', function (e) {
    if (!e.matches) return;
    io.disconnect();
    for (var k = 0; k < targets.length; k++) targets[k].classList.add('is-revealed');
  });
}());
```

**Prevents:** four distinct blank-content failures. (a) JS 404 / CSP block / ad-blocker — `.js-reveal` is never added, so `.js-reveal .reveal { opacity: 0 }` never matches. (b) Back-navigation from bfcache, where IntersectionObserver produces no new entries and content would otherwise stay hidden. (c) Unbounded observer cost on long pages — `unobserve()` stops the per-frame geometry work that would otherwise run forever for every card. (d) Mid-session Reduce Motion changes leaving un-revealed sections stuck at `opacity: 0`.

**Also note:** the `<noscript>` block is belt-and-braces for the case where the head script is stripped but the stylesheet loads. And `opacity: 0` elements stay **focusable and in the accessibility tree** — that is why the fallback layer must never be the permanent state (`../references/dynamic-loading.md` §4.3).

**Verify:** run all six tests in `../references/dynamic-loading.md` §3.3 — JS disabled, Firefox 154, `#anchor` deep link, back-navigation, Slow 4G, and `curl | grep`.

---

## 3. Staggered group reveal

Stagger with CSS custom properties rather than per-element JS delays. The stagger applies only on the CSS path *and* the JS path, and is zeroed under `reduce`.

```html
<ul class="card-grid">
  <li class="reveal" style="--i:0">…</li>
  <li class="reveal" style="--i:1">…</li>
  <li class="reveal" style="--i:2">…</li>
  <li class="reveal" style="--i:3">…</li>
  <li class="reveal" style="--i:4">…</li>
  <li class="reveal" style="--i:5">…</li>
</ul>
```

```css
@keyframes reveal-in {
  from { opacity: 0; transform: translateY(1.5rem); }
  to   { opacity: 1; transform: none; }
}

.card-grid { --stagger: 40ms; --stagger-max: 5; }   /* cap the sequence, don't scale it forever */

@media (prefers-reduced-motion: no-preference) {

  @supports (animation-timeline: view()) {
    /* Scroll-driven: "stagger" by offsetting each card's own view range.
       Each card has its own view() timeline, so a time delay is meaningless —
       shift the range instead. */
    .card-grid .reveal {
      animation: reveal-in linear both;
      animation-timeline: view();
      animation-range: entry 0% entry 60%;
    }
  }

  @supports not (animation-timeline: view()) {
    .js-reveal .card-grid .reveal { opacity: 0; transform: translateY(1.5rem); }
    .js-reveal .card-grid .reveal.is-revealed {
      opacity: 1;
      transform: none;
      transition: opacity 240ms cubic-bezier(0, 0, 0.38, 0.9),
                  transform 240ms cubic-bezier(0, 0, 0.38, 0.9);
      /* clamp() caps the total sequence: with --stagger 40ms and a cap of 5,
         the last card starts at 200ms, so the whole group finishes ~440ms. */
      transition-delay: calc(min(var(--i, 0), var(--stagger-max)) * var(--stagger));
    }
  }
}
```

```js
// Reveal the group together rather than card-by-card: observe the CONTAINER,
// not each card. One observer, one geometry computation, and the stagger is
// pure CSS.
(function () {
  if (CSS.supports('animation-timeline', 'view()')) return;
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (!('IntersectionObserver' in window)) {
    document.documentElement.classList.remove('js-reveal');
    return;
  }
  document.querySelectorAll('.card-grid').forEach(function (grid) {
    var io = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        grid.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('is-revealed'); });
        obs.unobserve(entry.target);
      });
    }, { threshold: 0, rootMargin: '0px 0px -10% 0px' });
    io.observe(grid);
  });
}());
```

**Prevents:** the runaway stagger — 12 items × 50 ms means the last item does not *start* until 550 ms, which reads as broken rather than elegant. `min(var(--i), var(--stagger-max))` caps the sequence at ~200 ms of offset regardless of item count. Observing the container rather than each card also prevents N observers doing per-frame geometry on a long grid, and prevents the visually random reveal order that happens when cards in the same row cross the threshold a frame apart.

⚠️ **Stagger numbers are practitioner heuristics, not sourced standards** — ~20–50 ms per item reads as intentional, <15 ms as simultaneous, >80–100 ms as slow; total sequence budget ~300–500 ms. Label them as heuristics in any report. Under `reduce`, the whole block is dropped, so all delays are effectively 0 and the group appears together — which is the correct substitution, because stagger creates directional flow across a large area and that is a Head "relative size of movement" trigger even when each element barely moves.

---

## 4. Reduced-motion handling: substitute, don't delete

The examples above drop motion entirely under `reduce`, which is correct when the motion is decoration. When the motion **carries meaning** — a panel that slides from the right to tell you where it came from, a toast that enters from the top — deleting it makes the interface confusing. Substitute instead.

```css
/* Base = the reduced state. Motion is added, never subtracted. */
.panel {
  opacity: 0;
  visibility: hidden;
  transition: opacity 200ms cubic-bezier(0, 0, 0.38, 0.9),
              visibility 0s linear 200ms;
}
.panel[data-open] {
  opacity: 1;
  visibility: visible;
  transition-delay: 0s;
}

/* Travel is added only when motion is welcome. */
@media (prefers-reduced-motion: no-preference) {
  .panel {
    translate: 2rem 0;
    transition: opacity 240ms cubic-bezier(0, 0, 0.38, 0.9),
                translate 240ms cubic-bezier(0, 0, 0.38, 0.9),
                visibility 0s linear 240ms;
  }
  .panel[data-open] { translate: 0 0; transition-delay: 0s; }
}
```

```js
// One shared signal for every JS motion system on the page.
// The CSS media query does NOT reach WAAPI, animation libraries, Lottie,
// <video autoplay>, canvas/rAF loops, SMIL, or smooth-scroll libraries.
export const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');   // parentheses are mandatory
export const prefersReducedMotion = () => motionQuery.matches;

motionQuery.addEventListener('change', () => {                                // not addListener()
  document.documentElement.classList.toggle('reduced-motion', motionQuery.matches);
  // Then actually tear down: cancel WAAPI animations, stop rAF loops,
  // pause SMIL (svg.pauseAnimations()), render Lottie frame 0, swap <video> to poster,
  // and destroy any smooth-scroll instance.
  if (motionQuery.matches) document.getAnimations().forEach(a => a.cancel());
});
```

**Prevents:** the two commonest reduced-motion failures. (a) Deleting meaning — WebKit: *"removing the animation entirely may make the interface confusing or unusable… consider serving an alternate, simpler animation."* Here the state change survives; only the travel is removed. (b) The CSS-only illusion of compliance — the media query reaches CSS and nothing else, so the JS signal exists to tear down every other motion system.

⚠️ **Do not substitute blur.** A WCAG erratum amended the `motion animation` definition "to not exclude blurring", and WebKit lists animated blur as a zoom-class vestibular trigger. Substitute opacity or colour cross-fades only.

⚠️ **Do not rely on the blanket `*` reset alone.** `*, *::before, *::after { animation-duration: 0.01ms !important; … }` is a **user-side** tool (web.dev files it under "Forcing reduced motion on all websites", to inject via an extension, "use at your own risk"). It is a useful safety net *in addition to* per-component handling, never instead of it. Note it uses `0.01ms` rather than `animation: none` specifically so `animationend`/`transitionend` still fire.

---

## 5. `content-visibility` for long pages

```css
/* Never above the fold — in-viewport use can DELAY LCP.
   nth-child(n+10) keeps the first nine items eligible for LCP. */
.feed-item:nth-child(n+10) {
  content-visibility: auto;
  contain-intrinsic-size: auto 420px;   /* auto = last remembered size, 420px = median fallback */
}
```

```html
<!-- Anchor targets inside skipped subtrees: focus un-skips before scrolling,
     so prefer .focus() over a raw scrollIntoView(). -->
<script>
  document.addEventListener('click', function (e) {
    var a = e.target.closest('a[href^="#"]');
    if (!a) return;
    var target = document.getElementById(decodeURIComponent(a.hash.slice(1)));
    if (!target) return;
    e.preventDefault();
    if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });   // un-skips the subtree first
    target.scrollIntoView({
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'
    });
    history.pushState(null, '', a.hash);
  });
</script>
```

**Prevents:** four things. (a) LCP regression — `content-visibility: auto` on an above-the-fold element delays the largest paint, which is the most common way teams make this optimisation net-negative. (b) Scrollbar jumping — `contain-intrinsic-size: auto <length>` with a realistic median means the scrollbar does not lurch as skipped items resolve their real heights; `0px` guarantees it will. (c) Anchor links landing in the wrong place — spec §4.4 note 5 says skipped elements are scrolled to **with size containment still active**, so "it might not align in the viewport exactly as requested"; note 6 says focus un-skips first, hence `.focus()` before `scrollIntoView()`. (d) Un-gated smooth scrolling overriding a reduced-motion preference.

⚠️ **Percentages are invalid** in `contain-intrinsic-size`. Chrome 117+ and Firefox force `contain-intrinsic-size: auto` when `content-visibility: auto` is set.

⚠️ **Accessibility trap:** styles inside skipped subtrees are not computed, so `display:none` / `visibility:hidden` descendants **still appear in the accessibility tree**. Add `aria-hidden="true"` explicitly to anything meant to be hidden inside a skipped region, and keep headings and landmarks outside skipped subtrees.

⚠️ **Test find-in-page in both Chrome and Safari 26.** Chrome has supported find-in-page inside skipped `auto` subtrees since 85; **Safari 18–25 was broken** ([WebKit 283846](https://bugs.webkit.org/show_bug.cgi?id=283846)) and was fixed in 26.

Support: `content-visibility` Chrome/Edge 85, Firefox 125, Safari 18.0 partial → 26.0 full; **93.19% global, Baseline Newly available 2025-09-15**. `contain-intrinsic-size` Chrome 83, Firefox 107, Safari 17.0; **93.25% global, Baseline Widely available since 2026-03-18**.

---

## 6. Scroll progress bar — the correct scroll-linked pattern

```css
@keyframes grow-progress { from { transform: scaleX(0); } to { transform: scaleX(1); } }

#progress {
  position: fixed;
  inset: 0 auto auto 0;
  width: 100%;
  height: 0.25rem;
  background: currentColor;
  transform: scaleX(0);
  transform-origin: 0 50%;
}

@media (prefers-reduced-motion: no-preference) {
  @supports (animation-timeline: scroll()) {
    #progress {
      animation: grow-progress auto linear;
      animation-timeline: scroll();   /* = scroll(nearest block); use scroll(root) for page scroll */
    }
  }
}
```

```html
<div id="progress" aria-hidden="true"></div>
```

**Prevents:** the scroll-listener rubber-band. Scroll events are delivered **asynchronously from a separate process**, so by the time a handler reads `scrollY` the compositor has already painted a frame it has not caught up to; events also fire off-cadence, many per frame, with every write but the last discarded. This version runs on the compositor with zero JS. It also animates `transform: scaleX()` rather than `width`, keeping it off the layout path entirely.

`aria-hidden="true"` because a decorative progress bar is redundant to assistive technology — it duplicates information the scroll position already conveys. If the bar is genuinely informative, use `role="progressbar"` with live `aria-valuenow`, which requires JS and is usually not worth it.

**Verify:** Performance → Animations track shows `transform` with no red triangle; Rendering → Paint flashing shows no green during scroll.
