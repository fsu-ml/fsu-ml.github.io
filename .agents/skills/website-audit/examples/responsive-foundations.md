# Responsive foundations — reference implementations

**Covers:** eight copy-pasteable patterns that satisfy the Tier 1 and Tier 2 criteria in `../references/mobile.md`. Plain HTML and CSS, no framework syntax, no build step.
**Load when:** a finding needs a "fix looks like this" alongside it, or you are writing remediation guidance.
**Rule:** every snippet is here because it prevents a *specific, named* failure. Recheck against the detection recipe in `../references/mobile.md`, not by eye.

---

## 1. The meta viewport

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

Prevents the ~980 px desktop-fallback render and the reinstated 300–350 ms tap delay. Omitting `user-scalable` and `maximum-scale` entirely is what keeps **WCAG 1.4.4 Resize Text (AA)** satisfied — any `user-scalable=no` or `maximum-scale=1` is a flat failure regardless of whether iOS honours it.

Only add `viewport-fit=cover` if §4 below is also implemented:

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
```

`viewport-fit=cover` is an opt-in to *handling* insets. Adding it without `env(safe-area-inset-*)` usage puts content under the notch and the home indicator — a regression, not an improvement.

---

## 2. Fluid type scale with `clamp()`

```css
:root {
  /* Every preferred value carries a rem term. The vw term only sets the RATE of change. */
  --step--1: clamp(0.833rem, 0.80rem + 0.17vw, 0.938rem);
  --step-0:  clamp(1rem,     0.95rem + 0.25vw, 1.125rem);
  --step-1:  clamp(1.2rem,   1.12rem + 0.40vw, 1.406rem);
  --step-2:  clamp(1.44rem,  1.31rem + 0.65vw, 1.758rem);
  --step-3:  clamp(1.728rem, 1.53rem + 0.99vw, 2.197rem);
  --step-4:  clamp(2.074rem, 1.78rem + 1.47vw, 2.75rem);
}

body { font-size: var(--step-0); line-height: 1.5; }
h2   { font-size: var(--step-3); line-height: 1.2; }
```

Prevents **WCAG 1.4.4** failure: `font-size: 4vw`, or a `clamp()` whose preferred value is pure `vw`, never grows under browser zoom because the viewport's CSS-pixel width does not change when you zoom. The `rem` term responds to zoom *and* to the user's root font-size preference.

Sanity heuristic: keep **max ≤ 2.5 × min** (both in `rem`) and the text can always reach 200% at 500% zoom.

```css
/* WRONG — fails 1.4.4 */
h1 { font-size: clamp(2rem, 5vw, 4rem); }   /* preferred value has no rem term */
```

---

## 3. A container-query component

```css
.card-slot {
  container-type: inline-size;
  container-name: card;
}

.card {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: 1fr;              /* stacked baseline */
}

@container card (min-width: 30rem) {
  .card {
    grid-template-columns: 12rem 1fr;      /* side-by-side once the SLOT is wide enough */
    align-items: start;
  }
}
```

```html
<div class="card-slot">
  <article class="card">
    <img src="…" alt="" width="480" height="320">
    <div><h3>Title</h3><p>Body…</p></div>
  </article>
</div>
```

Prevents the "works in the main column, breaks in the sidebar" defect: a component sized by viewport width has no idea it was dropped into a 280 px slot at 1440 px. Container queries are Baseline since **February 2023** (Chrome/Edge 105, Safari 16, Firefox 110).

Container queries **do not replace** media queries. Keep media queries for page-level structure and for what containers cannot see: `print`, `orientation`, `prefers-reduced-motion`, `prefers-color-scheme`, `hover`/`pointer`.

---

## 4. Safe-area insets

```css
.site-header {
  position: sticky;
  top: 0;
  /* Insets provide ZERO margin — they are exactly the system-UI extent. Always add your own. */
  padding-block-start: calc(env(safe-area-inset-top, 0px) + 0.75rem);
  padding-inline: calc(env(safe-area-inset-left, 0px) + 1rem)
                  calc(env(safe-area-inset-right, 0px) + 1rem);
}

.bottom-bar {
  position: fixed;
  inset-inline: 0;
  bottom: 0;
  padding-block-end: calc(env(safe-area-inset-bottom, 0px) + 0.75rem);
}
```

Prevents content sliding under the notch / Dynamic Island / home indicator / rounded corners once `viewport-fit=cover` is set. **Both inline sides are handled** because landscape moves the insets to left/right — portrait-only handling is the common bug.

Must be applied to: fixed headers and navbars, bottom tab bars, FABs and chat bubbles, full-screen dialogs and drawers, media controls near corners.

**You cannot verify this in emulation.** Desktop always returns 0, and Chrome DevTools responsive/device mode also returns 0. An audit can only confirm the static signal — that `env(safe-area-inset-*)` appears in the CSSOM when `viewport-fit=cover` is present. Confirm on a real notched device and say so in the report.

Chromium-only, use behind the fallback chain:

```css
.cookie-bar {
  padding-block-end: calc(env(safe-area-inset-bottom, 0px) + 1rem);
  padding-block-end: calc(env(safe-area-max-inset-bottom, 0px) + 1rem); /* stable when chrome collapses */
}
```

---

## 5. The `dvh` / `svh` pattern

```css
.hero {
  min-height: 100vh;    /* fallback for anything pre-Chrome 108 / Firefox 101 / Safari 15.4 */
  min-height: 100svh;   /* small viewport: fully visible on load, browser chrome expanded */
}
```

Prevents the "hero overflows the screen until you scroll" defect. `vh` equals the **large** viewport, so `100vh` is taller than what the user can actually see on load.

| Unit | Viewport with dynamic UA UI |
|---|---|
| `lvh` | **retracted** (largest). `100vh === 100lvh` |
| `svh` | **expanded** (smallest) |
| `dvh` | live value, clamped between `sv*` and `lv*` |

Use `100dvh` only when the layout must exactly fill the visible area at all times, and only after verifying on-device — `dvh` does not update at 60 fps (throttled, and debounced on some gestures), so anything sized by it reflows on scroll. **Never animate or transition to a `dvh` value**; use fixed values, `svh`, or `lvh` there.

Two things this pattern does *not* fix:

```css
/* WRONG — vw ignores scrollbars per spec, so this overflows on any classic-scrollbar page */
.full-bleed { width: 100vw; }
/* Use instead: */
.full-bleed { width: 100%; }
```

And it does not help with the virtual keyboard: the keyboard is not UA UI, so **no** viewport unit responds to it. See §8 and `../references/mobile.md` §2.8.

---

## 6. Responsive `<picture>` with reserved space

Resolution switching — one crop, several sizes:

```html
<img
  src="/img/hero-800.jpg"
  srcset="/img/hero-400.jpg 400w,
          /img/hero-800.jpg 800w,
          /img/hero-1600.jpg 1600w"
  sizes="(min-width: 60rem) 50vw, 100vw"
  width="1600" height="900"
  alt="Two engineers reviewing a wiring diagram"
  fetchpriority="high">
```

`width` and `height` give the browser the intrinsic **ratio** so it reserves space before the bytes arrive — this is what prevents CLS. They do not set the render size; CSS still does:

```css
img { max-width: 100%; height: auto; }
```

No `loading` attribute plus `fetchpriority="high"` because this is the probable LCP image. **`loading="lazy"` on the LCP or any above-fold image directly and severely regresses LCP.** `loading="lazy"` together with `fetchpriority="high"` is self-contradictory — the fetch is still deferred until near-viewport.

Art direction — different crops per breakpoint, which `srcset` alone cannot do:

```html
<picture>
  <source media="(min-width: 60rem)" srcset="/img/wide.avif" type="image/avif" width="1600" height="600">
  <source media="(min-width: 60rem)" srcset="/img/wide.jpg"  width="1600" height="600">
  <source srcset="/img/square.avif" type="image/avif" width="800" height="800">
  <img src="/img/square.jpg" width="800" height="800" alt="…">
</picture>
```

When sources have **different aspect ratios** the HTML attributes are not enough — pin the ratio in CSS per breakpoint or the layout still shifts:

```css
.art-directed { aspect-ratio: 1 / 1; }
@media (min-width: 60rem) { .art-directed { aspect-ratio: 8 / 3; } }
```

Below-fold images: `loading="lazy" decoding="async"` — still correct for mobile bandwidth even though Lighthouse removed its `offscreen-images` audit in v13.

Reserve space for `<video>`, `<iframe>` embeds and ad/widget slots the same way, or you ship CLS from third-party content:

```css
.embed { aspect-ratio: 16 / 9; width: 100%; }
```

---

## 7. Touch targets: 44 px hit area without a 44 px-looking control

```css
.icon-btn {
  position: relative;
  display: inline-grid;
  place-items: center;
  inline-size: 24px;          /* visual size — meets WCAG 2.5.8 (AA) 24×24 on its own */
  block-size: 24px;
  border: 0;
  background: none;
  padding: 0;
}

/* Pseudo-element extends the HIT AREA to 44×44 without changing the visual footprint. */
.icon-btn::after {
  content: "";
  position: absolute;
  inset: 50% auto auto 50%;
  translate: -50% -50%;
  inline-size: 44px;
  block-size: 44px;
  /* no background — invisible, but hit-tested */
}

/* Tap feedback, because the platform highlight is often suppressed. */
.icon-btn:active { opacity: 0.6; }
.icon-btn:focus-visible { outline: 2px solid currentColor; outline-offset: 4px; }
```

```html
<button class="icon-btn" type="button" aria-label="Close">
  <svg aria-hidden="true" focusable="false" width="16" height="16"><!-- … --></svg>
</button>
```

Satisfies **WCAG 2.5.8 Target Size (Minimum), 24 × 24 CSS px, Level AA** — the normative, legally relevant number — and additionally reaches the **44 × 44 pt** Apple HIG / **44 × 44 px** WCAG 2.5.5 AAA / **48 × 48 dp** Material recommendation without visually bloating a dense toolbar.

Notes that decide whether this actually passes:

- The 24 × 24 square must be **axis-aligned and fully inside** the target. A `border-radius: 50%` on a nominally 24 px control makes it undersized — enlarge to 28 px or rely on the pseudo-element.
- Give adjacent icon buttons **≥ 44 px centre-to-centre** so the enlarged hit areas do not overlap. Overlapping area does not count toward size unless both targets do the same thing.
- Alternative when a pseudo-element is unavailable: `min-block-size: 44px; min-inline-size: 44px;` on the control plus negative margin on the row. Same result, more layout risk.
- Do not restyle native `<input type="date">`, `<input type="color">` or `<input type="file">` internals — the *User agent control* exception to 2.5.8 is lost the moment you do.

---

## 8. Mobile form inputs

```html
<form>
  <p>
    <label for="email">Email</label>
    <input id="email" name="email" type="email" inputmode="email"
           autocomplete="email" autocapitalize="off" autocorrect="off"
           spellcheck="false" enterkeyhint="next">
  </p>

  <p>
    <label for="phone">Phone</label>
    <input id="phone" name="phone" type="tel" inputmode="tel"
           autocomplete="tel" enterkeyhint="next">
  </p>

  <p>
    <label for="zip">Postal code</label>
    <input id="zip" name="zip" type="text" inputmode="numeric"
           autocomplete="postal-code" enterkeyhint="next">
  </p>

  <p>
    <label for="otp">One-time code</label>
    <input id="otp" name="otp" type="text" inputmode="numeric" pattern="[0-9]*"
           autocomplete="one-time-code" enterkeyhint="done">
  </p>
</form>
```

```css
/* iOS Safari auto-zooms the whole page when a focused control renders under 16px,
   and does not reliably zoom back out on blur. 16px is the fix — NOT user-scalable=no. */
input, select, textarea { font-size: 16px; }
```

What each attribute prevents:

- **`type`** — wrong validation, wrong submitted semantics, wrong affordance. `type="number"` on a phone, OTP, postal code, card number or ID is the classic defect: spinner UI, silently stripped leading zeros, mutation on scroll-wheel, rejection of `+`/`-`/spaces. Use `type="text" inputmode="numeric"`.
- **`inputmode`** — the virtual keyboard layout, and nothing else. No validation, no format change.
- **`autocomplete`** — required by **WCAG 1.3.5 Identify Input Purpose (AA)** on any field collecting information *about the user*. Also the single largest mobile conversion lever in Baymard's testing — larger than keyboard type. Never `autocomplete="off"` on address, payment or name fields.
- **`<label for>`** — **WCAG 1.3.1 / 3.3.2 / 4.1.2**. A placeholder is not a label.
- **`enterkeyhint`** — labels the return key (`go`/`next`/`send`/`done`/`search`). Cheap win on multi-field forms.
- **16 px font-size** — prevents the iOS auto-zoom trap. The reason `user-scalable=no` proliferated is that it also suppresses this zoom; that "fix" is a **WCAG 1.4.4** violation.

Field-by-field reference table: `../references/mobile.md` §2.9. Detection recipe for all of the above: same section.
