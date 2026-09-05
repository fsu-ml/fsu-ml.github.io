/**
 * Winter — December.
 *
 * Snow is fixed-count DOM particles on CSS keyframes, so there is no canvas
 * and no frame loop. The one thing that reacts per frame is the light string,
 * and it is deliberately cheap: a rAF-throttled scroll listener that only
 * touches the DOM on the frames where the number of lit bulbs changes.
 *
 * The artwork here is ported from Winter.dc.html rather than invented — the
 * frost corners, frost ring, snow cap, icicles and the footer treeline are all
 * the artboard's own paths.
 */

import { Disposer, buildParticles, decorate, make, pick, range, seeded } from "./engine.js";

/* One flake per three density points, hard-capped. Forty animated elements is
   the budget for the whole layer; past that the page stops being a seminar
   site with snow on it and starts being a snow globe. */
const MAX_FLAKES = 40;
const flakeCount = (density) => Math.min(MAX_FLAKES, Math.round(density / 3));

const FLAKE_COLORS = ["#ffffff", "#f1f6fa", "#dce9f2"];

/* ---------------------------------------------------------------------------
   Artwork
   ---------------------------------------------------------------------------
   Static, author-written SVG. Assembled as markup rather than through the DOM
   builder because these are drawings, and a drawing is far easier to read and
   correct as a path than as thirty createElementNS calls.
   -------------------------------------------------------------------------- */

/* Snow cap for the brand mark: an arc following the top of the circle with
   melt drips hanging off it. The artboard's own path, drawn for a 72px mark. */
const SNOW_CAP = `
  <svg class="wn-mark-cap" viewBox="0 0 72 72" aria-hidden="true" focusable="false">
    <path fill="#f7f9fc"
          d="M0.7 22A38 38 0 0 1 71.3 22Q68 34 63 25Q58 38 52 27Q47 36 42 26Q38 42 33 27Q28 35 23 26Q18 38 13 27Q9 33 5 25Q3 30 0.7 22Z"></path>
    <path d="M16 15c6-7 14-10 22-10" stroke="rgba(255,255,255,.75)" stroke-width="1.6"
          fill="none" stroke-linecap="round"></path>
  </svg>`;

/* Frost creeping from a corner: a lattice of straight runs with small
   branchings where they cross. */
const FROST_CORNER = `
  <svg viewBox="0 0 120 120" aria-hidden="true" focusable="false">
    <g fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
      <path d="M0 30 L60 30 M0 55 L40 55 M30 0 L30 55 M55 0 L55 40 M8 8 L48 48 M15 0 L15 22 M0 15 L22 15 M70 0 L70 22 M0 70 L22 70"></path>
      <path d="M30 30 l-6 -6 M30 30 l6 -6 M30 30 l-6 6 M30 30 l6 6 M55 15 l-5 -5 M55 15 l5 -5 M15 55 l-5 -5 M15 55 l5 -5"></path>
    </g>
  </svg>`;

/* Frost ring: twenty-four radial spikes around a thin circle, alternating long
   and short. The artboard's logo treatment, used here on the portraits. */
const FROST_RING = (() => {
  const spikes = [];
  for (let i = 0; i < 24; i += 1) {
    const long = i % 2 === 0;
    spikes.push(
      `<g transform="rotate(${i * 15} 65 65)"><path d="${
        long
          ? "M65 25V10M65 18l-4-4M65 18l4-4M65 13l-2.5-2.5M65 13l2.5-2.5"
          : "M65 25V16M65 20l-3-3M65 20l3-3"
      }"/></g>`
    );
  }
  return `
    <svg class="wn-frost-ring" viewBox="0 0 130 130" aria-hidden="true" focusable="false">
      <g fill="none" stroke="#9fbfd6" stroke-width="1.2" stroke-linecap="round">
        <circle cx="65" cy="65" r="40" stroke-width="1"></circle>
        ${spikes.join("")}
      </g>
    </svg>`;
})();

const SNOWFLAKE = `
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <g stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none">
      <path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9L4.9 19.1"></path>
      <path d="M12 6l-2.4-2.4M12 6l2.4-2.4M12 18l-2.4 2.4M12 18l2.4 2.4"></path>
      <path d="M6 12l-2.4-2.4M6 12l-2.4 2.4M18 12l2.4-2.4M18 12l2.4 2.4"></path>
    </g>
  </svg>`;

/* Snow settled along a card's top edge, and the icicles hanging under it.
   Together these are the artboard's "Snow and icicles" frame. */
const CARD_SNOW = `
  <svg class="wn-card-snow" viewBox="0 0 400 30" preserveAspectRatio="none"
       aria-hidden="true" focusable="false">
    <path fill="#dce9f2"
          d="M0 0H400V12c-20 0-30 12-55 12s-30-14-60-14-35 14-70 14-35-16-70-16-30 12-55 12S25 10 0 12z"></path>
  </svg>`;

const icicles = (seed) => {
  const rand = seeded(seed);
  const W = 400;
  const n = 22;
  const w = W / n;
  let d = `M0 0H${W}V3`;
  for (let i = n; i > 0; i -= 1) {
    const x = i * w;
    const len = 8 + rand() * 26;
    d += `L${(x - w * 0.5).toFixed(1)} ${len.toFixed(1)}L${(x - w).toFixed(1)} 3`;
  }
  return `
    <svg class="wn-icicles" viewBox="0 0 ${W} 40" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <path d="${d}Z" fill="rgba(247,249,252,.96)" stroke="rgba(159,191,214,.6)"
            stroke-width=".8"></path>
    </svg>`;
};

/* Rolling snow bank for the bottom of the hero.
   One silhouette in one colour, and that colour is the exact white
   `.section-overview` starts with, so the bank reads as that section rising
   into the hero rather than a shape laid over it. */
const DRIFT_PATH =
  "M0 30c150-18 260 6 400-6s220-22 340-6 240 24 300 8 120-10 160 2V60H0z";

const driftSvg = (height) => `
  <svg class="wn-drift" viewBox="0 0 1200 60" preserveAspectRatio="none"
       aria-hidden="true" focusable="false" style="height:${height}px">
    <path class="wn-drift-front" d="${DRIFT_PATH}"></path>
  </svg>`;

const starfield = (seed, n, maxTop) => {
  const rand = seeded(seed);
  return Array.from({ length: n }, () => {
    const size = 1.5 + rand() * 2.5;
    return `<span class="wn-star" style="left:${(rand() * 100).toFixed(1)}%;top:${(
      6 +
      rand() * maxTop
    ).toFixed(1)}%;width:${size.toFixed(1)}px;height:${size.toFixed(1)}px;opacity:${(
      0.4 +
      rand() * 0.6
    ).toFixed(2)};--delay:${(-rand() * 3).toFixed(1)}s"></span>`;
  }).join("");
};

/* The artboard's footer treeline: a run of sharp firs cut straight from the
   baseline, with a snow ground in front. */
const TREELINE = `
  <svg class="wn-treeline" viewBox="0 0 1200 88" preserveAspectRatio="none"
       aria-hidden="true" focusable="false">
    <path fill="#1e2e26" d="M0 88 V70 l15 -26 l15 26 l19 -40 l19 40 l23 -32 l23 32 l15 -48 l15 48 l19 -36 l19 36 l23 -30 l23 30 l15 -44 l15 44 l19 -38 l19 38 l23 -52 l23 52 l15 -34 l15 34 l19 -42 l19 42 l23 -28 l23 28 l15 -46 l15 46 l19 -36 l19 36 l23 -50 l23 50 l15 -30 l15 30 l19 -40 l19 40 l23 -34 l23 34 l15 -44 l15 44 l19 -38 l19 38 l23 -26 l23 26 l15 -40 l15 40 l19 -32 l19 32 l23 -48 l23 48 l15 -36 l15 36 l19 -30 l19 30 l23 -44 l23 44 l15 -38 l15 38 l19 -52 l19 52 l23 -34 l23 34 l15 -42 l15 42 l19 -28 l19 28 V88 Z"></path>
    <path fill="#f7f9fc" d="M0 88V76c100-10 200 6 300-2s200-12 300-4 200 12 300 2 200-10 300-2V88z"></path>
  </svg>`;

/* ---------------------------------------------------------------------------
   Light string
   -------------------------------------------------------------------------- */

const BULBS = 14;
const WIRE_W = 1200;
const WIRE_H = 58;
const WIRE_HOOKS = 4;
const WIRE_SAG = 17;
const WIRE_TOP = 2;
const BULB_COLORS = ["#ceb888", "#c1273b", "#dce9f2", "#3e6b52"];

/* The wire is four parabolic sags between hooks. Both the path and the bulb
   positions come from the same formula, so a bulb always sits on the wire no
   matter how the strip is scaled. */
const wireY = (x) => {
  const seg = WIRE_W / WIRE_HOOKS;
  const t = (x % seg) / seg;
  return WIRE_TOP + 4 * WIRE_SAG * t * (1 - t);
};

const WIRE_D = (() => {
  const seg = WIRE_W / WIRE_HOOKS;
  let d = `M0 ${WIRE_TOP}`;
  for (let k = 0; k < WIRE_HOOKS; k += 1) {
    d += ` Q${(k + 0.5) * seg} ${WIRE_TOP + 2 * WIRE_SAG} ${(k + 1) * seg} ${WIRE_TOP}`;
  }
  return d;
})();

const lightStringHtml = (count = BULBS, cls = "") => {
  const bulbs = [];
  for (let i = 0; i < count; i += 1) {
    const x = 30 + i * ((WIRE_W - 60) / (count - 1));
    bulbs.push(
      `<span class="wn-bulb" style="left:${((x / WIRE_W) * 100).toFixed(3)}%;` +
        `top:${wireY(x).toFixed(1)}px;--bulb:${BULB_COLORS[i % BULB_COLORS.length]};` +
        `--order:${i}"></span>`
    );
  }
  return `
    <svg class="wn-wire ${cls}" viewBox="0 0 ${WIRE_W} ${WIRE_H}" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <path d="${WIRE_D}" fill="none" stroke="#ceb888" stroke-width="1.4" opacity=".7"></path>
    </svg>
    ${bulbs.join("")}`;
};

/**
 * Lights the header string one bulb at a time as the page is scrolled.
 *
 * The listener runs on every scroll frame but the DOM is only touched when
 * the number of lit bulbs changes — fourteen class toggles at most fourteen
 * times over a whole page, rather than per frame.
 *
 * Returns its `update`, which the mounted engine re-exposes as `syncLights()`.
 * Hidden documents never fire requestAnimationFrame, so without a way to
 * advance this by hand the string cannot be verified anywhere the page is not
 * actually on screen — the same reason `Loop` makes `step` and `draw` public.
 */
const bindLightString = (disposer, host) => {
  const bulbs = Array.from(host.querySelectorAll(".wn-bulb"));
  if (bulbs.length === 0) {
    return () => {};
  }
  let lit = -1;
  let ticking = false;

  const update = () => {
    ticking = false;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 1;
    /* One bulb is lit from the very top so the string never reads as broken. */
    const next = Math.max(1, Math.round(ratio * bulbs.length));
    if (next === lit) {
      return;
    }
    lit = next;
    bulbs.forEach((bulb, index) => bulb.classList.toggle("is-lit", index < next));
  };

  const onScroll = () => {
    if (ticking) {
      return;
    }
    ticking = true;
    requestAnimationFrame(update);
  };

  disposer.listen(window, "scroll", onScroll, { passive: true });
  disposer.listen(window, "resize", onScroll, { passive: true });
  update();
  return update;
};

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

const buildSnow = (overlay, density, motion) => {
  const count = flakeCount(density);
  if (count === 0) {
    return;
  }
  overlay.appendChild(
    buildParticles(count, 11, (index, rand) => {
      const size = range(rand, 3, 9);
      const duration = range(rand, 12, 26);
      const left = rand() * 100;
      const opacity = range(rand, 0.3, 0.9);
      const color = pick(rand, FLAKE_COLORS);
      const swayDur = range(rand, 2.5, 5.5);

      /* With motion off the flakes are not hidden, they are *landed*: seeded
         positions spread down the viewport so the scene still reads as
         snowfall, just a photograph of one. */
      const outer = make("div", {
        class: `season-particle${motion ? " wn-flake-fall" : ""}`,
        style: {
          left: `${left.toFixed(2)}%`,
          top: motion ? "-40px" : `${range(rand, 2, 96).toFixed(2)}vh`,
          width: `${size.toFixed(1)}px`,
          height: `${size.toFixed(1)}px`,
          opacity: opacity.toFixed(2),
          "--dur": `${duration.toFixed(1)}s`,
          /* A negative delay starts each flake partway through its fall, so
             the first frame is already a full snowfall rather than an empty
             sky that fills over the next twenty seconds. */
          "--delay": `${(-rand() * duration).toFixed(1)}s`
        }
      });

      const inner = make("div", {
        class: `wn-flake${motion ? " wn-flake-sway" : ""}`,
        style: {
          "--flake": color,
          "--sway": `${swayDur.toFixed(1)}s`,
          "--sway-delay": `${(-rand() * 3).toFixed(1)}s`
        }
      });

      outer.appendChild(inner);
      return outer;
    })
  );
};

export const mount = ({ overlay, density, motion }) => {
  const disposer = new Disposer();

  buildSnow(overlay, density, motion);

  /* Header: a string of lights hanging off the bar's bottom edge, coming on a
     bulb at a time as the page is scrolled. No snow on the bar itself. */
  const [lights] = decorate(
    disposer,
    ".site-header",
    "season-scene wn-lights",
    lightStringHtml()
  );
  const syncLights = lights ? bindLightString(disposer, lights) : () => {};

  const brand = document.querySelector(".site-header .brand");
  if (brand) {
    const cap = make("span", { class: "wn-cap-host", "aria-hidden": "true" });
    cap.innerHTML = SNOW_CAP;
    brand.appendChild(cap);
    disposer.node(cap);
  }

  /* Hero: night sky, stars, frost creeping in from the upper corners, and a
     bank along the bottom that bleeds into the section below. */
  decorate(
    disposer,
    ".hero",
    "season-scene wn-hero",
    `<div class="season-sky"></div>
     ${starfield(3, 30, 52)}
     <div class="wn-frost wn-frost-left">${FROST_CORNER}</div>
     <div class="wn-frost wn-frost-right">${FROST_CORNER}</div>
     ${driftSvg(64)}`
  );

  /* Footer: the artboard's own night — garnet gradient, stars, a fir treeline
     with snow on the ground. */
  decorate(
    disposer,
    ".site-footer",
    "season-scene wn-footer",
    `<div class="season-sky"></div>
     ${starfield(23, 22, 48)}
     ${TREELINE}`
  );

  /* Section seam between the overview and the dashboard.
     The adjacent-sibling selector matters: the subpages reuse
     `.section-dashboard` as their only section, with no overview before it, so
     a bare class selector hangs a seam divider directly under the header on
     every one of them. */
  decorate(
    disposer,
    ".section-overview + .section-dashboard",
    "season-divider wn-divider",
    `<span class="wn-rule"></span>
     <span class="wn-rule-mark">${SNOWFLAKE}</span>
     <span class="wn-rule"></span>`,
    { first: true }
  );

  /* Portraits: the frost ring, on hover only. */
  decorate(
    disposer,
    ".speaker-directory-photo, .seminar-speaker-photo",
    "season-ring wn-ring",
    FROST_RING
  );

  /* A snowflake resting in the corner of each speaker card. Kept very faint —
     it is a watermark, not a badge. */
  decorate(
    disposer,
    ".speaker-directory-card",
    "season-card-art wn-card-flake",
    `<span class="wn-corner-flake">${SNOWFLAKE}</span>`
  );

  /* The community cards get the artboard's "Snow and icicles" frame: snow
     settled on the top edge with icicles hanging under it. */
  decorate(
    disposer,
    ".community-card",
    "wn-frame wn-frame-icicles",
    `${CARD_SNOW}${icicles(31)}`
  );

  /* Each talk card gets the artboard's "Seminar speakers" container: a light
     string across its top edge that switches on bulb by bulb when the card is
     hovered. This is the card the schedule page is built from — it has no
     table wrapper to hang a single string off. */
  decorate(
    disposer,
    ".talk-card",
    "wn-frame wn-frame-lights",
    lightStringHtml(9, "wn-wire-card")
  );

  return {
    /* Applies the current scroll position to the light string immediately,
       bypassing the rAF throttle. Used to verify the string where rAF is
       suspended. */
    syncLights,
    destroy() {
      disposer.dispose();
    }
  };
};
