/**
 * Winter — December.
 *
 * The simplest of the three: snow is fixed-count DOM particles on CSS
 * keyframes, so there is no canvas and no frame loop. Everything the theme
 * adds is decoration appended to chrome that exists on all five pages, and
 * every node is registered with the disposer so teardown is total.
 */

import { Disposer, buildParticles, make, pick, range, seeded } from "./engine.js";

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

/* Rolling snow edge. One path, no fill rule tricks: the curve is the top of
   the snow and the shape closes off the bottom of its box. */
const DRIFT_PATH =
  "M0 30c150-18 260 6 400-6s220-22 340-6 240 24 300 8 120-10 160 2V60H0z";

/* A second, flatter crest for the layer behind, so the two silhouettes do not
   trace each other. */
const DRIFT_PATH_BACK =
  "M0 22c180 14 300-12 460 2s250 18 360 0 200-14 260-4 90 8 120 4V60H0z";

/* `layered` is off for the header's 20px strip. At that height the back path
   sits almost entirely below the visible band, so all it contributes is a
   translucent grey patch that reads as a smudge rather than as depth. */
const driftSvg = (className, height, layered = false) => `
  <svg class="${className}" viewBox="0 0 1200 60" preserveAspectRatio="none"
       aria-hidden="true" focusable="false" style="height:${height}px">
    ${layered ? `<path class="wn-drift-back" fill="currentColor" d="${DRIFT_PATH_BACK}"></path>` : ""}
    <path fill="currentColor" d="${DRIFT_PATH}"></path>
  </svg>`;

/* A frost fern: one spine with many short paired barbs, each carrying two
   smaller barbs of its own. Density is what makes this read as a crystal —
   with few long barbs it reads as a bare branch instead, which is the wrong
   holiday entirely. */
const frostFern = (rand) => {
  const strokes = ["M60 6v100"];
  for (let i = 1; i <= 13; i += 1) {
    const t = i / 14;
    const y = 8 + t * 94;
    const len = 22 * (1 - t * 0.75) + 4;
    const rise = len * 0.62;
    [-1, 1].forEach((dir) => {
      const tipX = 60 + dir * len;
      const tipY = y - rise;
      strokes.push(`M60 ${y.toFixed(1)}L${tipX.toFixed(1)} ${tipY.toFixed(1)}`);
      /* Secondary barbs at the midpoint, the detail that turns a line into
         a crystal at a glance. */
      const midX = 60 + dir * len * 0.55;
      const midY = y - rise * 0.55;
      const sub = len * 0.3;
      strokes.push(`M${midX.toFixed(1)} ${midY.toFixed(1)}l${(dir * sub).toFixed(1)} ${(
        sub * 0.9
      ).toFixed(1)}`);
      strokes.push(`M${midX.toFixed(1)} ${midY.toFixed(1)}l${(dir * sub * 0.7).toFixed(1)} ${(
        -sub
      ).toFixed(1)}`);
    });
  }
  const tilt = range(rand, -14, 14).toFixed(1);
  return `
    <g transform="rotate(${tilt} 60 60)">
      ${strokes.map((d) => `<path d="${d}"></path>`).join("")}
    </g>`;
};

const frostSvg = (className, seed) => {
  const rand = seeded(seed);
  return `
    <svg class="${className}" viewBox="0 0 120 120" aria-hidden="true" focusable="false">
      <!-- Rotated so the cluster grows diagonally out of the corner it is
           pinned to, the way frost actually creeps in from the edge of a
           pane, rather than standing upright in the middle of its box. -->
      <g fill="none" stroke="currentColor" stroke-width="0.9" stroke-linecap="round"
         transform="rotate(-38 34 18)">
        ${frostFern(rand)}
        <g transform="translate(-30 -22) rotate(-34 60 60) scale(.6)">${frostFern(rand)}</g>
        <g transform="translate(40 -10) rotate(28 60 60) scale(.46)">${frostFern(rand)}</g>
        <g transform="translate(-6 34) rotate(64 60 60) scale(.4)">${frostFern(rand)}</g>
      </g>
    </svg>`;
};

/* Snow cap hugging the brand mark: an arc that follows the top of the 48px
   circle, with two small lumps so it does not read as a crescent. */
const MARK_CAP = `
  <svg class="wn-mark-cap" viewBox="0 0 60 30" aria-hidden="true" focusable="false">
    <path fill="#f7fbff"
          d="M6 21C6 11 15 4 30 4s24 7 24 17c-4-2-6 1-9-1s-5 2-8 0-5-3-8-1-5-2-8 0-5 3-9 2z"></path>
  </svg>`;

const ornament = (x, drop, color) => `
  <div class="wn-ornament" style="left:${x}">
    <svg viewBox="0 0 24 ${drop + 22}" width="26" aria-hidden="true" focusable="false">
      <line x1="12" y1="0" x2="12" y2="${drop}" stroke="#ceb888" stroke-width="1.2"></line>
      <rect x="9" y="${drop - 1}" width="6" height="4" rx="1" fill="#ceb888"></rect>
      <circle cx="12" cy="${drop + 11}" r="8" fill="${color}"></circle>
      <ellipse cx="9" cy="${drop + 8}" rx="2.4" ry="1.6" fill="rgba(255,255,255,.5)"></ellipse>
    </svg>
  </div>`;

/* A treeline of overlapping firs. Generated so the silhouette differs from the
   hero drift rather than mirroring it. */
const treelineSvg = (seed) => {
  const rand = seeded(seed);
  const trees = [];
  for (let x = -20; x < 1240; x += range(rand, 34, 62)) {
    const h = range(rand, 56, 118);
    const w = h * range(rand, 0.42, 0.58);
    /* Three stacked tiers, each narrower, gives a fir rather than a cone. */
    const tiers = [0, 0.3, 0.58]
      .map((t) => {
        const top = 132 - h + h * t;
        const half = (w / 2) * (1 - t * 0.55);
        const base = top + h * 0.42;
        return `M${(x - half).toFixed(1)} ${base.toFixed(1)}L${x.toFixed(1)} ${top.toFixed(
          1
        )}L${(x + half).toFixed(1)} ${base.toFixed(1)}Z`;
      })
      .join("");
    trees.push(tiers);
  }
  return `
    <svg class="wn-treeline" viewBox="0 0 1200 132" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <path fill="currentColor" d="${trees.join("")}"></path>
      <path fill="#f1f6fa" opacity=".9"
            d="M0 108c140-14 240 8 380-4s210-16 330-2 230 18 290 6 140-8 200 0v24H0z"></path>
    </svg>`;
};

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

/**
 * Appends a decoration container to a host, if the host is on this page.
 * Returns the container so callers can keep decorating it, or null.
 */
const scene = (disposer, selector, className, html) => {
  const host = document.querySelector(selector);
  if (!host) {
    return null;
  }
  const node = make("div", { class: className, "aria-hidden": "true" });
  node.innerHTML = html;
  host.appendChild(node);
  disposer.node(node);
  return node;
};

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

  /* Header: snow settling on the bar's bottom edge, a cap on the mark, and two
     ornaments hanging into the hero below. */
  scene(
    disposer,
    ".site-header",
    "season-edge-strip wn-header-edge",
    `${driftSvg("wn-edge", 20)}
     ${ornament("18%", 26, "#c1273b")}
     ${ornament("72%", 40, "#ceb888")}`
  );
  const brand = document.querySelector(".site-header .brand");
  if (brand) {
    const cap = make("span", { "aria-hidden": "true" });
    cap.innerHTML = MARK_CAP;
    brand.appendChild(cap);
    disposer.node(cap);
  }

  /* Hero: frost in the upper corners and a drift along the bottom that bleeds
     into the section below. */
  scene(
    disposer,
    ".hero",
    "season-scene wn-hero",
    `<div class="season-sky"></div>
     <div class="wn-frost wn-frost-left">${frostSvg("", 3)}</div>
     <div class="wn-frost wn-frost-right">${frostSvg("", 7)}</div>
     ${driftSvg("wn-drift", 64, true)}`
  );

  /* Footer: night treeline with snow on the ground. The link columns sit in
     front of it untouched. */
  scene(
    disposer,
    ".site-footer",
    "season-scene wn-footer",
    `<div class="season-sky"></div>
     ${treelineSvg(23)}
     ${Array.from({ length: 14 }, (_, i) => {
       const rand = seeded(100 + i);
       return `<span class="wn-star" style="left:${(rand() * 100).toFixed(
         1
       )}%;top:${(rand() * 46).toFixed(1)}%;--delay:${(-rand() * 3).toFixed(1)}s"></span>`;
     }).join("")}`
  );

  return {
    destroy() {
      disposer.dispose();
    }
  };
};
