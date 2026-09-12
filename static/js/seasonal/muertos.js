/**
 * Día de Muertos — 1 to 5 November.
 *
 * Warm and celebratory, never spooky. Papel picado strung across the top of
 * dark chrome, cempasúchil petals drifting down, a garden of marigolds along
 * the bottom of the hero and the footer, and an ofrenda standing in the
 * hero's left margin: three tiers under an arch of marigolds, with candles,
 * pan de muerto, photographs with nobody in them yet, and sugar skulls in
 * their flower crowns.
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
const PINK = "#ff6f9c";
const RED = "#c8322b";
const TEAL = "#1fa9a0";
const TEAL_DEEP = "#12736d";
const BLUE = "#2a7fa8";
const PURPLE = "#6b2c91";
const PURPLE_LT = "#8a45b5";
const GREEN = "#0f4a45";
const BONE = "#f5efe0";
const NIGHT = "#1a1013";
const DEEP = "#c96a0c";
const GOLD = "#ceb888";

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
   Marigolds and leaves
   ---------------------------------------------------------------------------
   The bloom is defined once as bare children of a 40x40 box, so the same
   drawing can be wrapped as its own <svg> or dropped into a larger drawing
   with a transform. Three rings of petals around a dark centre, in two
   colourways.
   -------------------------------------------------------------------------- */

const marigoldPetals = (variant) => {
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
  return petals.join("");
};

/* Drawn in a 40x40 box with `overflow: visible`, so the outer ring is free to
   sit proud of the box the way a real bloom does. */
const marigoldSvg = (variant) =>
  `<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false">${marigoldPetals(variant)}</svg>`;

/* A bloom placed inside a larger drawing: centred on (x, y), scaled so its
   outer petals reach 20 * s from the centre. */
const bloomAt = (x, y, s, variant, rot = 0) =>
  `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${rot.toFixed(0)}) scale(${s.toFixed(3)}) translate(-20 -20)">` +
  marigoldPetals(variant) +
  `</g>`;

/* A leaf with its stalk at the origin and its tip 28 units out, drawn to the
   upper right; `rot` swings it round. A lighter midrib keeps it from reading
   as a flat wedge. */
const leafAt = (x, y, s, rot, color) =>
  `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${rot.toFixed(0)}) scale(${s.toFixed(2)})">` +
  `<path d="M0 0C6-12 18-16 28-12C22-2 12 3 0 0Z" fill="${color}"/>` +
  `<path d="M2-1C10-7 18-10 26-11" fill="none" stroke="rgba(245,239,224,.35)" stroke-width="1"/>` +
  `</g>`;

/* A five-petal folk flower, the kind painted on everything in the third
   artboard: five discs around a centre. */
const flowerAt = (x, y, r, color, center = YELLOW) => {
  const discs = [];
  for (let i = 0; i < 5; i += 1) {
    const a = ((i * 72 - 90) * Math.PI) / 180;
    discs.push(
      `<circle cx="${(x + Math.cos(a) * r * 0.62).toFixed(1)}" cy="${(y + Math.sin(a) * r * 0.62).toFixed(1)}" r="${(r * 0.46).toFixed(1)}" fill="${color}"/>`
    );
  }
  return discs.join("") + `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r * 0.26).toFixed(1)}" fill="${center}"/>`;
};

/* A rose seen from above: a disc with a spiral drawn into it. */
const roseAt = (x, y, r, color) =>
  `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${color}"/>` +
  `<path d="M${(x + r * 0.7).toFixed(1)} ${y.toFixed(1)}A${(r * 0.7).toFixed(1)} ${(r * 0.7).toFixed(1)} 0 1 1 ${(x - r * 0.7).toFixed(1)} ${y.toFixed(1)}` +
  `A${(r * 0.35).toFixed(1)} ${(r * 0.35).toFixed(1)} 0 1 1 ${x.toFixed(1)} ${y.toFixed(1)}" fill="none" stroke="rgba(26,16,19,.3)" stroke-width="${(r * 0.16).toFixed(2)}"/>`;

/* A curl — two arcs of halving radius. */
const swirlAt = (x, y, r, color, w = 1.6, flip = false) => {
  const s = flip ? -1 : 1;
  return (
    `<path d="M${(x + r * s).toFixed(1)} ${y.toFixed(1)}A${r} ${r} 0 1 ${flip ? 0 : 1} ${(x - r * s).toFixed(1)} ${y.toFixed(1)}` +
    `A${r / 2} ${r / 2} 0 1 ${flip ? 0 : 1} ${x.toFixed(1)} ${y.toFixed(1)}` +
    `A${r / 4} ${r / 4} 0 1 ${flip ? 0 : 1} ${(x - (r / 2) * s).toFixed(1)} ${y.toFixed(1)}" ` +
    `fill="none" stroke="${color}" stroke-width="${w}" stroke-linecap="round"/>`
  );
};

const heartAt = (x, y, s, color) =>
  `<path d="M0-4C-6-12-14-4 0 8C14-4 6-12 0-4z" fill="${color}" transform="translate(${x} ${y}) scale(${s})"/>`;

/* ---------------------------------------------------------------------------
   The marigold garden
   ---------------------------------------------------------------------------
   A continuous bed along the bottom of the hero and the footer: two rows of
   blooms packed shoulder to shoulder over a base of foliage, with teal and
   purple leaves standing up between them and loose petals lying on top. It
   used to be a scatter of separate blooms, which read as random.

   The bed is a user-unit <pattern>, so it repeats across any width rather
   than stretching. Anything near a tile's edge is drawn again one tile over,
   so the seam is invisible.

   The tile is taller than the band it fills: the foliage base stops at
   GARDEN_EDGE, which the stylesheet lines up with the host's bottom edge,
   and only the front row's blooms and a few drooping leaves reach into the
   strip below it. Hung over the hero's edge, that strip is what spills onto
   the white of the section beneath — a scalloped hedge, not a cut line.
   -------------------------------------------------------------------------- */

const GARDEN_W = 560;
const GARDEN_EDGE = 120;
const GARDEN_H = 152;

const gardenSvg = (seed, id) => {
  const rand = seeded(seed);
  const parts = [];
  const put = (x, draw) => {
    parts.push(draw(x));
    if (x < 70) {
      parts.push(draw(x + GARDEN_W));
    }
    if (x > GARDEN_W - 70) {
      parts.push(draw(x - GARDEN_W));
    }
  };

  /* Foliage base with a gently rolling top, so nothing shows through under
     the front row. It ends at the host's edge, not the tile's. */
  const crest = [];
  for (let x = 0; x <= GARDEN_W; x += 40) {
    crest.push(`${x} ${(90 + Math.sin((x / GARDEN_W) * Math.PI * 4) * 5).toFixed(1)}`);
  }
  parts.push(`<path d="M0 ${GARDEN_EDGE}L${crest.join("L")}L${GARDEN_W} ${GARDEN_EDGE}Z" fill="${GREEN}"/>`);

  /* Tall leaves at the back, in the artboard's purples and teals. */
  for (let i = 0; i < 18; i += 1) {
    const x = (i / 18) * GARDEN_W + rand() * 24;
    const y = 78 + rand() * 18;
    const left = rand() > 0.5;
    const rot = left ? -100 - rand() * 30 : -50 - rand() * 30;
    const color = pick(rand, [PURPLE, TEAL_DEEP, BLUE, PURPLE_LT]);
    const s = 1.2 + rand() * 0.6;
    put(x, (px) => leafAt(px, y, s, rot, color));
  }

  /* Back row: smaller blooms, a touch darker with distance. */
  for (let i = 0; i < 12; i += 1) {
    const x = ((i + 0.5) / 12) * GARDEN_W + rand() * 14 - 7;
    const y = 78 + rand() * 10;
    const s = 0.72 + rand() * 0.22;
    const v = rand() > 0.35 ? "orange" : "yellow";
    const rot = rand() * 90;
    put(x, (px) => `<g opacity=".88">${bloomAt(px, y, s, v, rot)}</g>`);
  }

  /* Small teal leaves poking up between the rows. */
  for (let i = 0; i < 14; i += 1) {
    const x = (i / 14) * GARDEN_W + rand() * 30;
    const y = 94 + rand() * 12;
    const rot = -130 + rand() * 80;
    put(x, (px) => leafAt(px, y, 0.9 + rand() * 0.4, rot, rand() > 0.5 ? TEAL : TEAL_DEEP));
  }

  /* Leaves drooping over the edge, drawn before the front row so the blooms
     sit on their stalks. */
  for (let i = 0; i < 9; i += 1) {
    const x = (i / 9) * GARDEN_W + rand() * 40;
    const y = 110 + rand() * 8;
    const rot = rand() > 0.5 ? 20 + rand() * 40 : 120 + rand() * 40;
    put(x, (px) => leafAt(px, y, 0.9 + rand() * 0.4, rot, rand() > 0.5 ? TEAL : TEAL_DEEP));
  }

  /* A row of smaller blooms just under the edge, behind the front row, so a
     gap between two front petals shows another flower rather than the line
     where the host's colour ends. */
  for (let i = 0; i < 16; i += 1) {
    const x = ((i + 0.5) / 16) * GARDEN_W + rand() * 8 - 4;
    const y = GARDEN_EDGE + 2 + rand() * 6;
    const s = 0.78 + rand() * 0.2;
    const v = rand() > 0.5 ? "orange" : "yellow";
    const rot = rand() * 90;
    put(x, (px) => bloomAt(px, y, s, v, rot));
  }

  /* Front row: the big blooms, packed so that neighbours overlap, with their
     centres on the host's edge — so the widest part of every bloom lies
     across the line and no straight edge shows between them. */
  for (let i = 0; i < 18; i += 1) {
    const x = (i / 18) * GARDEN_W + rand() * 8 - 4;
    const y = GARDEN_EDGE - 6 + rand() * 12;
    const s = 1.1 + rand() * 0.3;
    const v = rand() > 0.45 ? "orange" : "yellow";
    const rot = rand() * 90;
    put(x, (px) => bloomAt(px, y, s, v, rot));
  }

  /* Loose petals lying on the blooms. */
  for (let i = 0; i < 12; i += 1) {
    const x = rand() * GARDEN_W;
    const y = 64 + rand() * 40;
    const rot = rand() * 360;
    put(x, (px) =>
      `<g transform="translate(${px.toFixed(1)} ${y.toFixed(1)}) rotate(${rot.toFixed(0)}) scale(.55)">${petalPath(
        rand() > 0.5 ? YELLOW : ORANGE
      )}</g>`
    );
  }

  return `
    <svg class="dm-garden-art" width="100%" height="${GARDEN_H}" aria-hidden="true" focusable="false">
      <defs>
        <pattern id="${id}" patternUnits="userSpaceOnUse" width="${GARDEN_W}" height="${GARDEN_H}">${parts.join("")}</pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#${id})"></rect>
    </svg>`;
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
   Sugar skulls
   ---------------------------------------------------------------------------
   Four styles on one skull: a bone-white classic with marigold eyes, a blue
   one with orange work, a bone one in magenta and purple with jewelled eyes,
   and a purple one in yellow. Each wears a crown of flowers across the top
   the way the face-painted celebrants do — roses or marigolds with leaves
   between — which is what makes them read as calaveras rather than as
   Halloween's leftovers.

   Decorative, not a memento mori.
   -------------------------------------------------------------------------- */

const SKULL_STYLES = {
  classic: {
    base: BONE,
    ring: [YELLOW, ORANGE],
    iris: TEAL,
    line1: MAGENTA,
    line2: TEAL,
    eye: "petals",
    brow: "petals",
    crown: "rose",
    crownColors: [RED, ORANGE, RED, ORANGE, RED, ORANGE],
    leaf: TEAL
  },
  azul: {
    base: BLUE,
    ring: [ORANGE, YELLOW],
    iris: BONE,
    line1: YELLOW,
    line2: BONE,
    eye: "petals",
    brow: "flower",
    crown: "rose",
    crownColors: [RED, RED, ORANGE, RED, RED, ORANGE],
    leaf: TEAL
  },
  rosa: {
    base: BONE,
    ring: [MAGENTA, PINK],
    iris: PURPLE,
    line1: PURPLE,
    line2: MAGENTA,
    eye: "dots",
    brow: "heart",
    crown: "rose",
    crownColors: [PINK, MAGENTA, PINK, MAGENTA, PINK, MAGENTA],
    leaf: TEAL_DEEP
  },
  noche: {
    base: PURPLE,
    ring: [YELLOW, ORANGE],
    iris: BONE,
    line1: YELLOW,
    line2: ORANGE,
    eye: "petals",
    brow: "flower",
    crown: "marigold",
    crownColors: ["yellow", "orange", "yellow", "orange", "yellow", "orange"],
    leaf: TEAL
  }
};

/* Where the crown's flowers sit along the skull's top curve. */
const CROWN_SPOTS = [
  [16, 40, 6.5],
  [28, 21, 8],
  [42, 9, 7],
  [58, 9, 8],
  [72, 21, 7],
  [84, 40, 6.5]
];

/**
 * The skull is drawn in a 100x120 box; the viewBox carries extra headroom for
 * the crown. `attrs` lets it be nested inside another drawing as an inner
 * <svg> with its own x, y and width.
 */
const calaveraSvg = (style = "classic", attrs = 'class="dm-calavera"') => {
  const s = SKULL_STYLES[style] || SKULL_STYLES.classic;
  const eye = (cx) => {
    const marks = [];
    if (s.eye === "dots") {
      for (let i = 0; i < 12; i += 1) {
        const a = (i / 12) * Math.PI * 2;
        marks.push(
          `<circle cx="${(cx + Math.cos(a) * 15).toFixed(1)}" cy="${(52 + Math.sin(a) * 15).toFixed(1)}" r="2" fill="${
            i % 2 ? s.ring[0] : s.ring[1]
          }"/>`
        );
      }
    } else {
      for (let i = 0; i < 10; i += 1) {
        marks.push(
          `<ellipse cx="${cx}" cy="36" rx="2.6" ry="4.6" fill="${i % 2 ? s.ring[0] : s.ring[1]}" transform="rotate(${
            i * 36
          } ${cx} 52)"/>`
        );
      }
    }
    return (
      `<circle cx="${cx}" cy="52" r="12" fill="${NIGHT}"/>` +
      marks.join("") +
      `<circle cx="${cx}" cy="52" r="3" fill="${s.iris}"/>`
    );
  };

  let brow = "";
  if (s.brow === "petals") {
    for (let i = 0; i < 10; i += 1) {
      brow += `<ellipse cx="50" cy="26.2" rx="1.96" ry="3.85" fill="${i % 2 ? s.ring[0] : s.ring[1]}" transform="rotate(${
        i * 36
      } 50 30)"/>`;
    }
    brow += `<circle cx="50" cy="30" r="1.6" fill="${DEEP}"/>`;
  } else if (s.brow === "flower") {
    brow = flowerAt(50, 30, 7, s.ring[0], s.ring[1]);
  } else {
    brow = heartAt(50, 29, 0.7, s.line1);
  }

  const crown = CROWN_SPOTS.map(([x, y, r], i) => {
    const c = s.crownColors[i % s.crownColors.length];
    return s.crown === "marigold" ? bloomAt(x, y, r / 20, c, i * 25) : roseAt(x, y, r, c);
  }).join("");
  const crownLeaves =
    leafAt(20, 30, 0.55, -140, s.leaf) +
    leafAt(36, 13, 0.5, -120, s.leaf) +
    leafAt(64, 13, 0.5, -60, s.leaf) +
    leafAt(80, 30, 0.55, -40, s.leaf);

  return (
    `<svg ${attrs} viewBox="0 -12 100 132" aria-hidden="true" focusable="false">` +
    `<path d="M50 6C24 6 12 26 12 52c0 14 6 24 14 32v16c0 6 4 10 10 10h28c6 0 10-4 10-10V84c8-8 14-18 14-32C88 26 76 6 50 6z" fill="${s.base}"></path>` +
    eye(34) +
    eye(66) +
    `<path d="M50 66c-4-6-8-6-8-2 0 3 4 6 8 10 4-4 8-7 8-10 0-4-4-4-8 2z" fill="${NIGHT}"></path>` +
    `<g fill="none" stroke="${s.line1}" stroke-width="1.6" stroke-linecap="round">` +
    `<path d="M22 44q-6 6-2 12M78 44q6 6 2 12M26 72q-4 6 2 8M74 72q4 6-2 8"></path></g>` +
    `<g fill="none" stroke="${s.line2}" stroke-width="1.6" stroke-linecap="round">` +
    `<path d="M32 26q4-6 8-2M68 26q-4-6-8-2M40 40q10-6 20 0"></path></g>` +
    `<g stroke="${NIGHT}" stroke-width="1.2" fill="none">` +
    `<path d="M36 88h28v10H36zM43 88v10M50 88v10M57 88v10M36 93h28"></path></g>` +
    `<path d="M50 110c-3-4-6-4-6-1 0 2 3 4 6 7 3-3 6-5 6-7 0-3-3-3-6 1z" fill="${s.line1}"></path>` +
    `<g fill="${s.line1}"><circle cx="20" cy="62" r="2"/><circle cx="80" cy="62" r="2"/>` +
    `<circle cx="26" cy="94" r="1.8"/><circle cx="74" cy="94" r="1.8"/></g>` +
    brow +
    crownLeaves +
    crown +
    flowerAt(50, 0, 4.2, BONE, s.ring[0]) +
    `</svg>`
  );
};

/* ---------------------------------------------------------------------------
   The ofrenda
   ---------------------------------------------------------------------------
   Three tiers under an arch of marigolds, drawn in a 260x300 box. On the
   tiers: candles in coloured holders, photographs in gold frames — blank,
   they are nobody's in particular — pan de muerto, a glass of water, sugar
   skulls, and marigolds and petals over everything. The cloth is purple with
   a magenta hem and a row of teal dots, from the artboard's palette.
   -------------------------------------------------------------------------- */

const candleAt = (x, base, h, holder, delay) =>
  `<g class="dm-candle" style="--delay:${delay}s">` +
  `<circle class="dm-candle-halo" cx="${x}" cy="${base - h - 6}" r="${(h * 0.45 + 10).toFixed(1)}" fill="url(#dm-halo)"/>` +
  `<rect x="${x - 4}" y="${base - h}" width="8" height="${h}" rx="1.5" fill="${BONE}"/>` +
  `<path d="M${x - 2} ${base - h + 4}c1 4 0 8 1 12s2 3 2 6" stroke="rgba(42,17,24,.16)" stroke-width="1" fill="none"/>` +
  `<rect x="${x - 0.6}" y="${base - h - 4}" width="1.2" height="4" fill="${NIGHT}"/>` +
  `<g class="dm-flame">` +
  `<path d="M${x} ${base - h - 16}c3 4.5 4.5 7.5 4.5 10.5a4.5 4.5 0 0 1-9 0c0-3 1.5-6 4.5-10.5z" fill="#f2a65a"/>` +
  `<path d="M${x} ${base - h - 10.5}c1.5 2.2 2.2 3.7 2.2 5.6a2.2 2.2 0 0 1-4.4 0c0-1.9.7-3.4 2.2-5.6z" fill="#ffe39a"/>` +
  `</g>` +
  `<ellipse cx="${x}" cy="${base}" rx="6.5" ry="2" fill="${holder}"/>` +
  `</g>`;

const frameAt = (x, y, w, h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2" fill="${GOLD}"/>` +
  `<rect x="${x + 3}" y="${y + 3}" width="${w - 6}" height="${h - 6}" fill="#efe6cf"/>` +
  `<rect x="${x + 6}" y="${y + 6}" width="${w - 12}" height="${h - 12}" fill="#e2d7bd"/>`;

const panAt = (x, y) =>
  `<circle cx="${x}" cy="${y - 10}" r="13" fill="#c98a4a"/>` +
  `<ellipse cx="${x - 4}" cy="${y - 14}" rx="6" ry="4" fill="rgba(255,255,255,.14)"/>` +
  `<g stroke="#ecc78d" stroke-width="3.6" stroke-linecap="round" fill="none">` +
  `<path d="M${x - 9} ${y - 17}L${x + 9} ${y - 3}M${x + 9} ${y - 17}L${x - 9} ${y - 3}"/></g>` +
  `<circle cx="${x}" cy="${y - 10}" r="3.6" fill="#ecc78d"/>`;

const tier = (x, y, w, h) =>
  `<rect x="${x}" y="${y}" width="${w}" height="10" fill="${PURPLE_LT}"/>` +
  `<rect x="${x}" y="${y + 10}" width="${w}" height="${h - 10}" fill="${PURPLE}"/>` +
  `<rect x="${x}" y="${y + h - 7}" width="${w}" height="7" fill="${MAGENTA}"/>` +
  Array.from({ length: Math.floor(w / 12) }, (_, i) =>
    `<circle cx="${x + 6 + i * 12}" cy="${y + h - 3.5}" r="1.6" fill="${TEAL}"/>`
  ).join("");

const ofrendaSvg = (seed) => {
  const rand = seeded(seed);

  /* The arch: a thick stem of foliage with blooms along it. */
  const arch = [];
  const cx = 130;
  const cy = 244;
  const rx = 112;
  const ry = 212;
  arch.push(
    `<path d="M${cx - rx} ${cy}A${rx} ${ry} 0 0 1 ${cx + rx} ${cy}" fill="none" stroke="${GREEN}" stroke-width="16" stroke-linecap="round"/>`
  );
  for (let i = 0; i <= 18; i += 1) {
    const a = Math.PI + (i / 18) * Math.PI;
    const x = cx + Math.cos(a) * rx;
    const y = cy + Math.sin(a) * ry;
    const deg = (a * 180) / Math.PI;
    arch.push(leafAt(x, y, 0.7, deg - 30, i % 2 ? TEAL : TEAL_DEEP));
    arch.push(leafAt(x, y, 0.7, deg + 120, i % 2 ? TEAL_DEEP : TEAL));
    arch.push(bloomAt(x, y, 0.5 + rand() * 0.22, i % 3 === 1 ? "yellow" : "orange", rand() * 90));
  }

  const petals = [];
  const scatter = (x0, x1, y0, y1, n) => {
    for (let i = 0; i < n; i += 1) {
      petals.push(
        `<g transform="translate(${range(rand, x0, x1).toFixed(1)} ${range(rand, y0, y1).toFixed(1)}) rotate(${(
          rand() * 360
        ).toFixed(0)}) scale(.4)">${petalPath(rand() > 0.5 ? YELLOW : ORANGE)}</g>`
      );
    }
  };
  scatter(78, 184, 120, 128, 6);
  scatter(44, 216, 178, 186, 8);
  scatter(12, 248, 238, 246, 10);

  const skull = (style, x, y, w) => calaveraSvg(style, `x="${x}" y="${y}" width="${w}" height="${(w * 1.32).toFixed(1)}"`);

  return `
    <svg class="dm-ofrenda-art" viewBox="0 0 260 300" aria-hidden="true" focusable="false">
      <defs>
        <radialGradient id="dm-halo">
          <stop offset="0" stop-color="rgba(255,182,39,.4)"/>
          <stop offset="1" stop-color="rgba(255,182,39,0)"/>
        </radialGradient>
      </defs>
      ${arch.join("")}
      ${tier(72, 118, 116, 58)}
      ${tier(40, 176, 180, 60)}
      ${tier(8, 236, 244, 54)}
      ${frameAt(110, 70, 40, 54)}
      ${bloomAt(112, 122, 0.32, "orange", 20)}
      ${candleAt(92, 126, 30, MAGENTA, -0.4)}
      ${candleAt(168, 126, 26, TEAL, -1.3)}
      ${frameAt(80, 146, 28, 38)}
      ${frameAt(152, 146, 28, 38)}
      ${skull("azul", 40, 142, 32)}
      ${skull("rosa", 190, 142, 32)}
      ${panAt(130, 184)}
      ${candleAt(118, 184, 20, PURPLE, -2.1)}
      ${bloomAt(64, 182, 0.4, "yellow", 0)}
      ${bloomAt(196, 182, 0.4, "orange", 40)}
      ${candleAt(26, 244, 36, TEAL, -0.9)}
      ${skull("noche", 44, 196, 38)}
      ${bloomAt(92, 242, 0.42, "orange", 10)}
      ${candleAt(114, 244, 22, MAGENTA, -1.7)}
      <rect x="138" y="214" width="12" height="30" rx="2" fill="rgba(191,224,245,.42)" stroke="rgba(255,255,255,.5)" stroke-width="1"/>
      <rect x="140" y="220" width="8" height="22" fill="rgba(191,224,245,.35)"/>
      ${skull("classic", 156, 200, 34)}
      ${panAt(214, 244)}
      ${candleAt(238, 244, 30, PURPLE, -2.6)}
      ${bloomAt(14, 242, 0.42, "yellow", 30)}
      ${bloomAt(248, 240, 0.4, "orange", 50)}
      ${petals.join("")}
    </svg>`;
};

/* ---------------------------------------------------------------------------
   Folk-art sprays
   ---------------------------------------------------------------------------
   The third artboard's border language: curling stems with leaves, dotted
   lines, small flowers and hearts, in teal, orange and magenta on the night.
   One spray is drawn for a top-left corner and reflected into the others.
   -------------------------------------------------------------------------- */

const FOLK_SPRAY = (() => {
  const dots = (pts, color) =>
    pts.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="1.7" fill="${color}"/>`).join("");
  return (
    `<svg class="dm-spray-art" viewBox="0 0 240 200" aria-hidden="true" focusable="false">` +
    `<g fill="none" stroke="${TEAL}" stroke-width="2.2" stroke-linecap="round">` +
    `<path d="M0 12C50 14 90 40 112 90C124 118 118 150 96 172"/>` +
    `<path d="M0 30C40 44 66 78 74 122"/>` +
    `<path d="M20 0C70 6 130 4 176 26C200 38 214 54 224 76"/>` +
    `</g>` +
    leafAt(40, 22, 1.1, -150, TEAL) +
    leafAt(74, 34, 1.1, 20, TEAL_DEEP) +
    leafAt(96, 58, 1.2, -140, TEAL) +
    leafAt(112, 90, 1.1, 30, TEAL_DEEP) +
    leafAt(118, 128, 1.0, -160, TEAL) +
    leafAt(56, 72, 0.9, 40, PURPLE_LT) +
    leafAt(140, 14, 1.1, 30, TEAL_DEEP) +
    leafAt(178, 28, 1.0, -150, TEAL) +
    leafAt(206, 46, 0.9, 40, PURPLE_LT) +
    swirlAt(150, 62, 12, ORANGE, 2) +
    swirlAt(38, 100, 10, MAGENTA, 1.8, true) +
    swirlAt(128, 168, 9, TEAL, 1.6) +
    flowerAt(112, 36, 12, RED) +
    flowerAt(170, 90, 9, MAGENTA, ORANGE) +
    flowerAt(60, 128, 8, ORANGE, MAGENTA) +
    heartAt(200, 20, 0.7, MAGENTA) +
    heartAt(90, 140, 0.6, RED) +
    dots(
      [
        [86, 96],
        [92, 108],
        [96, 120],
        [98, 132],
        [150, 30],
        [162, 42],
        [174, 56],
        [26, 50],
        [30, 62],
        [32, 74]
      ],
      YELLOW
    ) +
    dots(
      [
        [190, 70],
        [204, 88],
        [216, 106],
        [224, 124]
      ],
      PINK
    ) +
    `</svg>`
  );
})();

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

  /* Hero: the artboard's night — a purple-to-garnet sky, folk-art sprays in
     the upper corners, the ofrenda standing in the left margin, and the
     marigold garden along the bottom. Everything sits behind `.hero-inner`,
     so the copy and the seminar card keep the contrast they were audited
     with.

     No string of flags here. The header hangs its own and the header is sticky
     over the hero's top edge, so a second one lands in the same eighty pixels
     as the first and the two read as one crowded knot. */
  decorate(
    disposer,
    ".hero",
    "season-scene dm-hero",
    `<div class="season-sky"></div>
     <div class="dm-spray dm-spray-tl">${FOLK_SPRAY}</div>
     <div class="dm-spray dm-spray-tr">${FOLK_SPRAY}</div>
     <div class="dm-ofrenda">${ofrendaSvg(5)}</div>
     <div class="dm-garden">${gardenSvg(8, "dm-garden-hero")}</div>`
  );

  /* Footer: the same night with a banner overhead, sprays in the lower
     corners and the garden along the bottom. No candles or skulls down here
     — the hero's ofrenda already carries them, and a second set under the
     link columns looked out of place. */
  decorate(
    disposer,
    ".site-footer",
    "season-scene dm-footer",
    `<div class="season-sky"></div>
     <div class="dm-footer-banner">${
       banner({ seed: 10, count: 16, flagW: 58, sag: 18, motion }).html
     }</div>
     <div class="dm-spray dm-spray-bl">${FOLK_SPRAY}</div>
     <div class="dm-spray dm-spray-br">${FOLK_SPRAY}</div>
     <div class="dm-garden">${gardenSvg(11, "dm-garden-footer")}</div>`
  );

  /* Section seam between the overview and the dashboard: the artboard's gold
     rule with a slowly turning marigold at its centre and a leaf either side.

     The adjacent-sibling selector matters: the subpages reuse
     `.section-dashboard` as their only section, with no overview before it, so
     a bare class selector puts a divider directly under the header on every
     one of them. */
  decorate(
    disposer,
    ".section-overview + .section-dashboard",
    "season-divider dm-divider",
    `<span class="dm-rule"></span>
     <span class="dm-rule-leaf dm-rule-leaf-l"><svg viewBox="-2 -20 32 24" aria-hidden="true" focusable="false">${leafAt(0, 0, 1, -30, TEAL)}</svg></span>
     <span class="dm-rule-mark">${marigoldSvg("orange")}</span>
     <span class="dm-rule-leaf dm-rule-leaf-r"><svg viewBox="-2 -20 32 24" aria-hidden="true" focusable="false">${leafAt(0, 0, 1, -30, TEAL)}</svg></span>
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

  /* A small sugar skull resting in the corner of each speaker card. Kept very
     faint — it is a watermark, not a badge — and a different style on each
     card so a grid of them is not one stamp repeated. */
  const styles = Object.keys(SKULL_STYLES);
  document.querySelectorAll(".speaker-directory-card").forEach((card, i) => {
    const node = make("div", { class: "season-card-art dm-card-skull", "aria-hidden": "true" });
    node.innerHTML = `<span class="dm-corner-skull">${calaveraSvg(styles[i % styles.length])}</span>`;
    card.appendChild(node);
    disposer.node(node);
  });

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
