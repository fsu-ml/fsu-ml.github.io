/**
 * Winter — December.
 *
 * Snow is fixed-count DOM particles on CSS keyframes, so there is no canvas
 * and no frame loop. The one thing here that reacts per frame is the light
 * string, and it is deliberately cheap: a rAF-throttled scroll listener that
 * only touches the DOM on the frames where the number of lit bulbs actually
 * changes.
 *
 * Everything the theme adds is decoration appended to chrome that exists on
 * all five pages, and every node and listener is registered with the disposer
 * so teardown is total.
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

/* Snow ledge for the bottom edge of the header bar.
 *
 * Solid across the top with scalloped lumps hanging off the bottom, so it
 * reads as snow sitting *on* the bar. A plain drift silhouette was tried
 * first and, squashed into a 26px strip, it rendered as a solid white stripe
 * floating under the header instead.
 */
const snowLedge = (seed, height) => {
  const W = 1200;
  const H = 30;
  const rand = seeded(seed);
  /* Right to left along the underside: each span dips to its own depth, so no
     two lumps match and the edge never looks tiled. */
  let d = `M0 0H${W}V7`;
  let x = W;
  while (x > 0) {
    const span = range(rand, 46, 124);
    const nx = Math.max(0, x - span);
    const dip = range(rand, 10, H - 2);
    d +=
      `C${(x - span * 0.22).toFixed(1)} ${dip.toFixed(1)} ` +
      `${(nx + span * 0.22).toFixed(1)} ${dip.toFixed(1)} ${nx.toFixed(1)} 7`;
    x = nx;
  }
  d += "Z";
  return `
    <svg class="wn-edge" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         aria-hidden="true" focusable="false" style="height:${height}px">
      <path fill="currentColor" d="${d}"></path>
    </svg>`;
};

/* Rolling snow bank for the bottom of the hero.
 *
 * Deliberately a single silhouette in a single colour. Two earlier attempts at
 * depth both failed the same way: a translucent back layer let the hero art
 * through and split the bank into two tones, and an opaque one in a darker
 * blue read as a grey smudge floating above the white. One clean crest in the
 * exact white that `.section-overview` starts with is what makes the bank look
 * like that section rising into the hero rather than a shape laid over it.
 */
const DRIFT_PATH =
  "M0 30c150-18 260 6 400-6s220-22 340-6 240 24 300 8 120-10 160 2V60H0z";

const driftSvg = (className, height) => `
  <svg class="${className}" viewBox="0 0 1200 60" preserveAspectRatio="none"
       aria-hidden="true" focusable="false" style="height:${height}px">
    <path class="wn-drift-front" d="${DRIFT_PATH}"></path>
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
      strokes.push(
        `M${midX.toFixed(1)} ${midY.toFixed(1)}l${(dir * sub).toFixed(1)} ${(sub * 0.9).toFixed(1)}`
      );
      strokes.push(
        `M${midX.toFixed(1)} ${midY.toFixed(1)}l${(dir * sub * 0.7).toFixed(1)} ${(-sub).toFixed(1)}`
      );
    });
  }
  const tilt = range(rand, -14, 14).toFixed(1);
  return `
    <g transform="rotate(${tilt} 60 60)">
      ${strokes.map((d) => `<path d="${d}"></path>`).join("")}
    </g>`;
};

const frostSvg = (seed) => {
  const rand = seeded(seed);
  return `
    <svg viewBox="0 0 120 120" aria-hidden="true" focusable="false">
      <!-- Rotated so the cluster grows diagonally out of the corner it is
           pinned to, the way frost creeps in from the edge of a pane. -->
      <g fill="none" stroke="currentColor" stroke-width="0.9" stroke-linecap="round"
         transform="rotate(-38 34 18)">
        ${frostFern(rand)}
        <g transform="translate(-30 -22) rotate(-34 60 60) scale(.6)">${frostFern(rand)}</g>
        <g transform="translate(40 -10) rotate(28 60 60) scale(.46)">${frostFern(rand)}</g>
        <g transform="translate(-6 34) rotate(64 60 60) scale(.4)">${frostFern(rand)}</g>
      </g>
    </svg>`;
};

/* Holly sprig tucked behind the brand mark.
   The artboard offers three logo treatments; this is the one that dresses the
   mark without covering it. The snow cap covered the badge artwork, which is
   the one thing the mark cannot afford to lose. */
const HOLLY_LEAF =
  "M0 7C2 4 4 2 6 2L8 0l2 3c3-2 6-2 9-1L21 0l1 3c3 1 5 2 6 4-1 2-3 3-6 4l-1 3-2-2c-3 1-6 1-9-1l-2 3-2-2c-2 0-4-2-6-5z";

const HOLLY = `
  <svg class="wn-holly" viewBox="0 0 44 30" aria-hidden="true" focusable="false">
    <g fill="#2f6b45">
      <path d="${HOLLY_LEAF}" transform="translate(12 2) rotate(-22)"></path>
      <path d="${HOLLY_LEAF}" transform="translate(9 12) rotate(18)"></path>
    </g>
    <g fill="#c1273b">
      <circle cx="10" cy="12" r="3.1"></circle>
      <circle cx="15.6" cy="10.4" r="2.7"></circle>
      <circle cx="13" cy="16" r="2.5"></circle>
    </g>
    <circle cx="9.2" cy="11" r="0.9" fill="rgba(255,255,255,.55)"></circle>
  </svg>`;

/* ---------------------------------------------------------------------------
   Light string
   -------------------------------------------------------------------------- */

const BULBS = 14;
const WIRE_W = 1200;
const WIRE_H = 58;
const WIRE_HOOKS = 4;
const WIRE_SAG = 16;
const WIRE_TOP = 2;
const BULB_COLORS = ["#ceb888", "#c1273b", "#dce9f2", "#3e6b52"];

/* The wire is four parabolic sags between hooks. Both the path and the bulb
   positions are derived from the same formula, so a bulb always sits on the
   wire no matter how the strip is scaled. */
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

const lightStringHtml = () => {
  const bulbs = [];
  for (let i = 0; i < BULBS; i += 1) {
    const x = 30 + i * ((WIRE_W - 60) / (BULBS - 1));
    bulbs.push(
      `<span class="wn-bulb" style="left:${((x / WIRE_W) * 100).toFixed(3)}%;` +
        `top:${wireY(x).toFixed(1)}px;--bulb:${BULB_COLORS[i % BULB_COLORS.length]}"></span>`
    );
  }
  return `
    <svg class="wn-wire" viewBox="0 0 ${WIRE_W} ${WIRE_H}" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <path d="${WIRE_D}" fill="none" stroke="#ceb888" stroke-width="1.4" opacity=".7"></path>
    </svg>
    ${bulbs.join("")}`;
};

/**
 * Lights the string one bulb at a time as the page is scrolled.
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
   Treeline
   -------------------------------------------------------------------------- */

/* Overlapping firs, generated so the silhouette differs from the hero drift
   rather than mirroring it. Evergreen, not navy: painted in the footer's own
   dark blue they read as a jagged shape rather than as trees. */
const treelineSvg = (seed) => {
  const rand = seeded(seed);
  const back = [];
  const front = [];
  for (let x = -20; x < 1240; x += range(rand, 32, 58)) {
    const far = rand() > 0.5;
    const h = far ? range(rand, 52, 84) : range(rand, 74, 122);
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
    (far ? back : front).push(tiers);
  }
  return `
    <svg class="wn-treeline" viewBox="0 0 1200 132" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <path class="wn-tree-far" d="${back.join("")}"></path>
      <path class="wn-tree-near" d="${front.join("")}"></path>
      <path class="wn-tree-snow"
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
const scene = (disposer, selector, className, html, { first = false } = {}) => {
  const host = document.querySelector(selector);
  if (!host) {
    return null;
  }
  const node = make("div", { class: className, "aria-hidden": "true" });
  node.innerHTML = html;
  /* `first` puts the scene behind the host's own content by tree order, which
     is how the header snow ends up under the logo and nav rather than over
     them. Stacking contexts are not involved, so no z-index is needed. */
  if (first) {
    host.prepend(node);
  } else {
    host.appendChild(node);
  }
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

  /* Header: snow lying on the bar, and a string of lights hanging below it
     that comes on a bulb at a time as the page is scrolled. */
  scene(disposer, ".site-header", "season-edge-strip wn-header-edge", snowLedge(41, 30), {
    first: true
  });
  const lights = scene(disposer, ".site-header", "season-scene wn-lights", lightStringHtml());
  const syncLights = lights ? bindLightString(disposer, lights) : () => {};

  const brand = document.querySelector(".site-header .brand");
  if (brand) {
    const sprig = make("span", { "aria-hidden": "true" });
    sprig.innerHTML = HOLLY;
    brand.appendChild(sprig);
    disposer.node(sprig);
  }

  /* Hero: frost in the upper corners and a bank along the bottom that bleeds
     into the section below. */
  scene(
    disposer,
    ".hero",
    "season-scene wn-hero",
    `<div class="season-sky"></div>
     <div class="wn-frost wn-frost-left">${frostSvg(3)}</div>
     <div class="wn-frost wn-frost-right">${frostSvg(7)}</div>
     ${driftSvg("wn-drift", 64)}`
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
    /* Applies the current scroll position to the light string immediately,
       bypassing the rAF throttle. Used to verify the string where rAF is
       suspended. */
    syncLights,
    destroy() {
      disposer.dispose();
    }
  };
};
