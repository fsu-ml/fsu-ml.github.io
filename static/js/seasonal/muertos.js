/**
 * Día de Muertos — 1 to 5 November.
 *
 * Warm and celebratory, never spooky. Two ideas carry the whole layer, both
 * taken from DiaDeMuertos.dc.html: papel picado — cut-paper banners strung
 * across the top of dark chrome — and cempasúchil petals drifting down.
 *
 * Like Winter, nothing here needs a canvas or a frame loop. The petals are
 * fixed-count DOM particles on CSS keyframes, the flags flutter on their own
 * clocks, and the candle flames are keyframed SVG. Every arrangement is
 * generated from a seed, so the same density always draws the same scene.
 *
 * The season sits inside Thanksgiving's month. `seasonForDate` gives it the
 * first five days of November and hands the rest back.
 */

import { Disposer, buildParticles, decorate, make, pick, range, seeded } from "./engine.js";

/* Sparse on purpose. The artboard's own note is that the petals mark a path,
   they do not fill the sky: one petal per four density points against Winter's
   one flake per three, and a lower cap, because a petal is several times the
   area of a flake and reads much louder for the same count. */
const MAX_PETALS = 26;
const petalCount = (density) => Math.min(MAX_PETALS, Math.round(density / 4));

const ORANGE = "#f28c1b";
const YELLOW = "#ffb627";
const MAGENTA = "#d81b60";
const TEAL = "#1fa9a0";
const PURPLE = "#6b2c91";
const BONE = "#f5efe0";
const NIGHT = "#1a1013";
const DEEP = "#c96a0c";

const PETAL_COLORS = [ORANGE, YELLOW];
const FLAG_COLORS = [MAGENTA, ORANGE, TEAL, PURPLE, YELLOW];

/* ---------------------------------------------------------------------------
   Artwork
   ---------------------------------------------------------------------------
   Static, author-written SVG, assembled as markup rather than through the DOM
   builder: these are drawings, and a drawing is far easier to read and correct
   as a path than as thirty createElementNS calls.
   -------------------------------------------------------------------------- */

/* One cempasúchil petal, drawn in a 12x18 box with the tip at the top. */
const petalPath = (color) => `<path d="M6 1C10 5 11 11 6 17 1 11 2 5 6 1z" fill="${color}"></path>`;

const petalSvg = (color) =>
  `<svg viewBox="0 0 12 18" aria-hidden="true" focusable="false">${petalPath(color)}</svg>`;

/* A papel picado flag: one path with even-odd holes, on a scalloped bottom.
   Three hole patterns — diamond and dots, circle and diamonds, heart and dots
   — so a string never reads as one shape repeated. */
const FLAG_OUTLINE =
  "M0 0H60V32Q55 41 50 32Q45 41 40 32Q35 41 30 32Q25 41 20 32Q15 41 10 32Q5 41 0 32Z";

const FLAG_HOLES = [
  "M30 7L41 18L30 29L19 18Z M8 8a3 3 0 1 0 .1 0z M52 8a3 3 0 1 0 .1 0z M8 26a3 3 0 1 0 .1 0z M52 26a3 3 0 1 0 .1 0z",
  "M30 18m-8 0a8 8 0 1 0 16 0a8 8 0 1 0-16 0z M9 12l4 5-4 5-4-5z M51 12l4 5-4 5-4-5z M30 4l3 3-3 3-3-3z M30 26l3 3-3 3-3-3z",
  "M30 27C22 21 17 15 21 10c3-3 7-1 9 2 2-3 6-5 9-2 4 5-1 11-9 17z M10 6a2.5 2.5 0 1 0 .1 0z M20 6a2.5 2.5 0 1 0 .1 0z M40 6a2.5 2.5 0 1 0 .1 0z M50 6a2.5 2.5 0 1 0 .1 0z M8 22a2.5 2.5 0 1 0 .1 0z M52 22a2.5 2.5 0 1 0 .1 0z"
];

const flagSvg = (color, pattern) =>
  `<svg viewBox="0 0 60 42" aria-hidden="true" focusable="false">` +
  `<path d="${FLAG_OUTLINE} ${FLAG_HOLES[pattern % 3]}" fill="${color}" fill-rule="evenodd"></path></svg>`;

/* ---------------------------------------------------------------------------
   Papel picado
   ---------------------------------------------------------------------------
   One drawing, used three times: off the header's bottom edge, over the top of
   the footer, and strung inside a card's top edge on hover.

   Same construction as Winter's light strings, for the same reason. The wire
   is a single parabola drawn as an SVG stretched to the host's width — a
   stretched curve is still a smooth curve. The flags are not in the SVG: they
   are positioned elements at a percentage across and a pixel height from the
   same formula, so they keep their shape at any width and always hang off the
   wire.
   -------------------------------------------------------------------------- */

const WIRE_W = 1200;

/**
 * `top` is where the string meets its two ends, `sag` how far it dips at the
 * centre. Returns the markup and the height the host has to reserve for it.
 */
const banner = ({ seed, count, flagW, sag, top = 0, motion }) => {
  const rand = seeded(seed);
  const wireY = (t) => top + 4 * sag * t * (1 - t);

  const points = [];
  for (let i = 0; i <= 24; i += 1) {
    const t = i / 24;
    points.push(`${(t * WIRE_W).toFixed(0)} ${wireY(t).toFixed(1)}`);
  }

  const height = top + sag + flagW * 0.7 + 6;
  const flags = [];
  for (let i = 0; i < count; i += 1) {
    const t = (i + 0.5) / count;
    flags.push(
      `<span class="dm-flag${motion ? " dm-flag-live" : ""}" style="left:${(t * 100).toFixed(3)}%;` +
        `top:${wireY(t).toFixed(1)}px;width:${flagW}px;margin-left:${(-flagW / 2).toFixed(1)}px;` +
        `--flutter:${(2.8 + rand() * 1.6).toFixed(2)}s;--flutter-delay:${(-rand() * 3).toFixed(2)}s">` +
        flagSvg(FLAG_COLORS[(i + Math.floor(rand() * 5)) % 5], Math.floor(rand() * 3)) +
        `</span>`
    );
  }

  return {
    height,
    html:
      `<svg class="dm-wire" viewBox="0 0 ${WIRE_W} ${height.toFixed(1)}" preserveAspectRatio="none" ` +
      `aria-hidden="true" focusable="false"><path d="M${points.join("L")}" fill="none" ` +
      `stroke="rgba(247,242,232,.55)" stroke-width="1.4"></path></svg>` +
      flags.join("")
  };
};

/* ---------------------------------------------------------------------------
   Marigolds
   -------------------------------------------------------------------------- */

/* Three rings of petals around a dark centre, in two colourways. Drawn in a
   40x40 box with `overflow: visible`, so the outer ring is free to sit proud
   of the box the way a real bloom does. */
const marigoldSvg = (variant) => {
  const outer = variant === "yellow" ? YELLOW : ORANGE;
  const mid = variant === "yellow" ? ORANGE : YELLOW;
  const petals = [];
  for (let i = 0; i < 12; i += 1) {
    petals.push(
      `<ellipse cx="20" cy="9.5" rx="4.6" ry="9.5" fill="${outer}" transform="rotate(${i * 30} 20 20)"/>`
    );
  }
  for (let i = 0; i < 8; i += 1) {
    petals.push(
      `<ellipse cx="20" cy="12.5" rx="3.6" ry="7" fill="${mid}" transform="rotate(${i * 45 + 22} 20 20)"/>`
    );
  }
  for (let i = 0; i < 6; i += 1) {
    petals.push(
      `<ellipse cx="20" cy="15.5" rx="2.4" ry="4.4" fill="${outer}" transform="rotate(${i * 60 + 10} 20 20)"/>`
    );
  }
  petals.push(`<circle cx="20" cy="20" r="3.2" fill="${DEEP}"/>`);
  return `<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false">${petals.join("")}</svg>`;
};

/**
 * A bed of marigolds cascading along the bottom edge of a scene: blooms of
 * varying size, sunk into the edge so only their tops show, with loose petals
 * scattered between them.
 */
const marigoldBed = (seed, count) => {
  const rand = seeded(seed);
  const parts = [];
  for (let i = 0; i < count; i += 1) {
    const size = range(rand, 26, 56);
    /* Centred on its point rather than hung off it, and stopped short of the
       right edge. The hero's bed deliberately sits outside a clipping host, so
       a bloom anchored by its left edge at 99% widens the document by its own
       width and the page picks up a horizontal scrollbar. */
    parts.push(
      `<span class="dm-bloom" style="left:${((i / count) * 96 + rand() * 3).toFixed(2)}%;` +
        `bottom:${(-size * 0.35 + rand() * 12).toFixed(1)}px;width:${size.toFixed(1)}px;` +
        `height:${size.toFixed(1)}px;margin-left:${(-size / 2).toFixed(1)}px;` +
        `transform:rotate(${(rand() * 90).toFixed(0)}deg)">` +
        marigoldSvg(rand() > 0.5 ? "yellow" : "orange") +
        `</span>`
    );
  }
  for (let i = 0; i < count; i += 1) {
    const size = range(rand, 8, 14);
    parts.push(
      `<span class="dm-loose-petal" style="left:${(rand() * 96).toFixed(2)}%;` +
        `bottom:${(4 + rand() * 30).toFixed(1)}px;width:${(size * 0.67).toFixed(1)}px;` +
        `transform:rotate(${(rand() * 360).toFixed(0)}deg)">` +
        petalSvg(rand() > 0.5 ? ORANGE : YELLOW) +
        `</span>`
    );
  }
  return `<span class="dm-bed" aria-hidden="true">${parts.join("")}</span>`;
};

/* A ring of petals radiating from behind a portrait. The inner ends tuck under
   the photograph, so the bloom looks like it is growing out from behind the
   person rather than being a collar drawn around them.

   Deliberately short. A long petal reaches past the portrait far enough to
   land on the speaker's name below it, and the ring stops being a detail on a
   photograph and becomes the loudest thing on the card. */
const PETAL_RING = (() => {
  const petals = [];
  for (let i = 0; i < 18; i += 1) {
    petals.push(
      `<g transform="rotate(${i * 20} 65 65) translate(59 -1)">${petalPath(
        i % 2 ? YELLOW : ORANGE
      )}</g>`
    );
  }
  return `<svg class="dm-petal-ring" viewBox="0 0 130 130" aria-hidden="true" focusable="false">${petals.join(
    ""
  )}</svg>`;
})();

/* Three blooms bunched into a corner, the artboard's "marigold corners" card
   frame. Drawn once and reflected into the opposite corner by CSS. */
const MARIGOLD_CLUSTER =
  `<span class="dm-cluster-bloom dm-cluster-a">${marigoldSvg("orange")}</span>` +
  `<span class="dm-cluster-bloom dm-cluster-b">${marigoldSvg("yellow")}</span>` +
  `<span class="dm-cluster-bloom dm-cluster-c">${marigoldSvg("orange")}</span>`;

/* ---------------------------------------------------------------------------
   Candles and calavera
   -------------------------------------------------------------------------- */

/* A candle in a coloured holder. The flame and its glow are separate layers so
   the halo can breathe without the flame scaling with it. */
const candleSvg = (holder) =>
  `<span class="dm-candle" aria-hidden="true">` +
  `<span class="dm-candle-glow"></span>` +
  `<svg viewBox="0 0 24 60" aria-hidden="true" focusable="false">` +
  `<g class="dm-flame">` +
  `<path d="M12 4c4 6 6 10 6 14a6 6 0 0 1-12 0c0-4 2-8 6-14z" fill="#f2a65a"></path>` +
  `<path d="M12 11c2 3 3 5 3 7.5a3 3 0 0 1-6 0c0-2.5 1-4.5 3-7.5z" fill="#ffe39a"></path>` +
  `</g>` +
  `<rect x="11.2" y="22" width="1.6" height="5" fill="${NIGHT}"></rect>` +
  `<rect x="6" y="26" width="12" height="32" rx="2" fill="${BONE}"></rect>` +
  `<path d="M8 30c2 4 1 8 3 11s3 2 4 5" stroke="rgba(42,17,24,.18)" stroke-width="1" fill="none"></path>` +
  `<ellipse cx="12" cy="58" rx="9" ry="2.5" fill="${holder}"></ellipse>` +
  `</svg></span>`;

/* A sugar skull: bone white, marigold eye rings, magenta and turquoise
   flourishes. Decorative, not a memento mori — the artboard's own drawing. */
const calaveraSvg = () => {
  const eye = (cx) => {
    const petals = [];
    for (let i = 0; i < 10; i += 1) {
      petals.push(
        `<ellipse cx="${cx}" cy="36" rx="2.6" ry="4.6" fill="${
          i % 2 ? YELLOW : ORANGE
        }" transform="rotate(${i * 36} ${cx} 52)"/>`
      );
    }
    return (
      `<circle cx="${cx}" cy="52" r="12" fill="${NIGHT}"/>` +
      petals.join("") +
      `<circle cx="${cx}" cy="52" r="3" fill="${TEAL}"/>`
    );
  };
  const crown = [];
  for (let i = 0; i < 10; i += 1) {
    crown.push(
      `<ellipse cx="50" cy="18.2" rx="1.96" ry="3.85" fill="${
        i % 2 ? YELLOW : ORANGE
      }" transform="rotate(${i * 36} 50 22)"/>`
    );
  }
  return (
    `<svg class="dm-calavera" viewBox="0 0 100 120" aria-hidden="true" focusable="false">` +
    `<path d="M50 6C24 6 12 26 12 52c0 14 6 24 14 32v16c0 6 4 10 10 10h28c6 0 10-4 10-10V84c8-8 14-18 14-32C88 26 76 6 50 6z" fill="${BONE}"></path>` +
    eye(34) +
    eye(66) +
    `<path d="M50 66c-4-6-8-6-8-2 0 3 4 6 8 10 4-4 8-7 8-10 0-4-4-4-8 2z" fill="${NIGHT}"></path>` +
    `<g fill="none" stroke="${MAGENTA}" stroke-width="1.6" stroke-linecap="round">` +
    `<path d="M22 44q-6 6-2 12M78 44q6 6 2 12M26 72q-4 6 2 8M74 72q4 6-2 8"></path></g>` +
    `<g fill="none" stroke="${TEAL}" stroke-width="1.6" stroke-linecap="round">` +
    `<path d="M40 20q10-8 20 0M32 26q4-6 8-2M68 26q-4-6-8-2"></path></g>` +
    `<g stroke="${NIGHT}" stroke-width="1.2" fill="none">` +
    `<path d="M36 88h28v10H36zM43 88v10M50 88v10M57 88v10M36 93h28"></path></g>` +
    `<path d="M50 110c-3-4-6-4-6-1 0 2 3 4 6 7 3-3 6-5 6-7 0-3-3-3-6 1z" fill="${MAGENTA}"></path>` +
    `<g fill="${MAGENTA}"><circle cx="20" cy="62" r="2"/><circle cx="80" cy="62" r="2"/>` +
    `<circle cx="26" cy="94" r="1.8"/><circle cx="74" cy="94" r="1.8"/></g>` +
    crown.join("") +
    `<circle cx="50" cy="22" r="1.54" fill="${DEEP}"></circle>` +
    `</svg>`
  );
};

/* ---------------------------------------------------------------------------
   Ambient petals
   -------------------------------------------------------------------------- */

const buildPetals = (overlay, density, motion) => {
  const count = petalCount(density);
  if (count === 0) {
    return;
  }
  overlay.appendChild(
    buildParticles(count, 17, (index, rand) => {
      const size = range(rand, 11, 21);
      const duration = range(rand, 14, 30);

      /* With motion off the petals are not hidden, they have *landed*: seeded
         positions spread down the viewport so the scene still reads as a fall
         of petals, just a photograph of one. */
      const outer = make("div", {
        class: `season-particle${motion ? " dm-petal-fall" : ""}`,
        style: {
          left: `${(rand() * 100).toFixed(2)}%`,
          top: motion ? "-40px" : `${range(rand, 2, 96).toFixed(2)}vh`,
          width: `${(size * 0.67).toFixed(1)}px`,
          height: `${size.toFixed(1)}px`,
          opacity: range(rand, 0.55, 1).toFixed(2),
          "--dur": `${duration.toFixed(1)}s`,
          /* A negative delay starts each petal partway through its fall, so the
             first frame is already a full drift rather than an empty sky that
             fills over the next half-minute. */
          "--delay": `${(-rand() * duration).toFixed(1)}s`
        }
      });

      const inner = make("div", {
        class: `dm-petal${motion ? " dm-petal-sway" : ""}`,
        style: {
          "--sway": `${range(rand, 3.4, 6.5).toFixed(1)}s`,
          "--sway-delay": `${(-rand() * 4).toFixed(1)}s`,
          "--tilt": `${range(rand, -40, 40).toFixed(0)}deg`
        }
      });
      inner.innerHTML = petalSvg(pick(rand, PETAL_COLORS));

      outer.appendChild(inner);
      return outer;
    })
  );
};

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

export const mount = ({ overlay, density, motion }) => {
  const disposer = new Disposer();

  buildPetals(overlay, density, motion);

  /* Header: a small string of papel picado hanging off the bar's bottom edge.
     The nav links' own marigold dot is CSS; it needs nothing here. */
  decorate(
    disposer,
    ".site-header",
    "season-scene dm-header-banner",
    banner({ seed: 2, count: 16, flagW: 34, sag: 8, motion }).html
  );

  /* Hero: the artboard's night — a purple-to-garnet sky, candles standing low
     on the left, and a bed of marigolds cascading over the bottom edge.
     Everything sits behind `.hero-inner`, so the copy and the seminar card keep
     the contrast they were audited with.

     No string of flags here. The header hangs its own and the header is sticky
     over the hero's top edge, so a second one lands in the same eighty pixels
     as the first and the two read as one crowded knot. */
  decorate(
    disposer,
    ".hero",
    "season-scene dm-hero",
    `<div class="season-sky"></div>
     <div class="dm-hero-candles">
       ${candleSvg(MAGENTA)}${candleSvg(TEAL)}${candleSvg(PURPLE)}
     </div>
     ${marigoldBed(5, 26)}`
  );

  /* Footer: the same night with a banner overhead and the marigold path along
     the bottom, and a small calavera resting in it. No candles down here — the
     footer already carries the hero's, and a second set of live flames under
     the link columns is one flicker too many on a page. */
  decorate(
    disposer,
    ".site-footer",
    "season-scene dm-footer",
    `<div class="season-sky"></div>
     <div class="dm-footer-banner">${
       banner({ seed: 10, count: 16, flagW: 58, sag: 18, motion }).html
     }</div>
     ${marigoldBed(11, 28)}
     <span class="dm-footer-calavera">${calaveraSvg()}</span>`
  );

  /* Section seam between the overview and the dashboard: the artboard's gold
     rule with a slowly turning marigold at its centre.

     The adjacent-sibling selector matters: the subpages reuse
     `.section-dashboard` as their only section, with no overview before it, so
     a bare class selector puts a divider directly under the header on every
     one of them. */
  decorate(
    disposer,
    ".section-overview + .section-dashboard",
    "season-divider dm-divider",
    `<span class="dm-rule"></span>
     <span class="dm-rule-mark">${marigoldSvg("orange")}</span>
     <span class="dm-rule"></span>`,
    { first: true }
  );

  /* Portraits: the petal ring, on hover only. A ring permanently around every
     face is decoration the reader cannot switch off. */
  decorate(disposer, ".speaker-directory-photo, .seminar-speaker-photo", "season-ring dm-ring", PETAL_RING);

  /* The About cards get the artboard's "marigold corners" frame: a bunch of
     blooms over one corner and its opposite, on hover. */
  decorate(
    disposer,
    ".feature-card",
    "season-frame dm-frame dm-corners",
    `<span class="dm-cluster dm-cluster-tl">${MARIGOLD_CLUSTER}</span>
     <span class="dm-cluster dm-cluster-br">${MARIGOLD_CLUSTER}</span>`
  );

  /* Talk and community cards get a string of papel picado strung across the
     inside of their top edge on hover, the way the artboard hangs one off its
     light card. Small flags and a lot of them: at card width a full-size flag
     is a poster, not a garland. The string is centred and width-capped in CSS,
     so the same markup reads the same on a half-width community card and on a
     full-width talk card. */
  const cardBanner = banner({ seed: 6, count: 12, flagW: 28, sag: 6, motion });
  decorate(
    disposer,
    ".talk-card, .community-card",
    "season-frame dm-frame dm-card-banner",
    cardBanner.html
  );

  return {
    destroy() {
      disposer.dispose();
    }
  };
};
