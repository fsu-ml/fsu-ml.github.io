/**
 * Lunar New Year — the eve, the day, and the day after.
 *
 * Red and gold, lanterns and plum blossom. The one season on the site that
 * cannot be derived from a month, so its dates live in `lunar-dates.js` and
 * both this file and the orchestrator read them from there.
 *
 * Two ideas carry the layer. Lanterns, strung off the header and the footer,
 * each carrying the zodiac animal whose year is beginning — so the decoration
 * says *which* new year it is, and says something different in 2032 than it
 * did in 2031. And a plum tree in bloom across the hero, because 梅花 opens in
 * the cold just before the new year and is the flower the holiday is drawn
 * with.
 *
 * Like Winter and Muertos, nothing here needs a canvas or a frame loop. The
 * petals are fixed-count DOM particles on CSS keyframes, the lanterns sway on
 * their own clocks, and the tree is generated once from a seed — so the same
 * density always draws the same scene.
 */

import { Disposer, buildParticles, decorate, make, pick, range, seeded } from "./engine.js";
import { lunarAnimalForDate } from "./lunar-dates.js";

/* Sparse, for the same reason Muertos' petals are: a blossom petal is several
   times the area of a snowflake and reads much louder for the same count. One
   per four density points, against Winter's one flake per three. */
const MAX_PETALS = 26;
const petalCount = (density) => Math.min(MAX_PETALS, Math.round(density / 4));

const RED = "#c8102e";
const RED_DEEP = "#7d0a1c";
const GOLD = "#e8b64c";
const GOLD_LT = "#ffd98a";
const PETAL = "#ffd7e2";
const PETAL_DEEP = "#ff9fbb";
const BARK = "#2c1620";

const PETAL_COLORS = [PETAL, PETAL_DEEP, GOLD_LT];

/* `?date=2032-02-11` previews another year's animal. Parsed by parts so it
   lands in local time; a UTC-parsed ISO date is the day before for everyone
   west of Greenwich. Same trick Winter uses to preview a Hanukkah night. */
const today = () => {
  const raw = new URLSearchParams(window.location.search).get("date");
  const match = raw && /^\d{4}-\d{2}-\d{2}$/.test(raw) ? raw.split("-").map(Number) : null;
  return match ? new Date(match[0], match[1] - 1, match[2]) : new Date();
};

/* ---------------------------------------------------------------------------
   Zodiac glyphs
   ---------------------------------------------------------------------------
   Ten heads, front on, in a 120x120 box: one head shape, the one feature that
   names the animal, two eyes. Nothing else earns its place — inside a lantern
   these are drawn at about forty pixels, where a shoulder or a paw is noise
   but a horn or an ear is the whole identification.

   Each is built as a mask: shapes in white build the head, shapes in black cut
   it, and one filled rect shows through. Cutting rather than overpainting is
   what lets a lantern read as backlit — the light comes through the eyes.
   -------------------------------------------------------------------------- */

const W = "#fff";
const B = "#000";
const cir = (cx, cy, r, f = W) => `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${f}"/>`;
const ell = (cx, cy, rx, ry, rot = 0, f = W) =>
  `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"` +
  (rot ? ` transform="rotate(${rot} ${cx} ${cy})"` : "") + ` fill="${f}"/>`;
const pth = (d, f = W) => `<path d="${d}" fill="${f}"/>`;
const bar = (d, w, f = W) =>
  `<path d="${d}" fill="none" stroke="${f}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;
const ring = (cx, cy, rx, ry, w, f = B) =>
  `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="none" stroke="${f}" stroke-width="${w}"/>`;

const ZODIAC = {
  /* Rat — two big ears on a tapered face. */
  rat:
    cir(28, 38, 21) + cir(92, 38, 21) +
    pth("M24 52 C24 32 96 32 96 52 C96 78 80 104 60 104 C40 104 24 78 24 52 Z") +
    bar("M48 88 L14 96", 3) + bar("M72 88 L106 96", 3) +
    cir(28, 38, 11, B) + cir(92, 38, 11, B) +
    cir(47, 62, 5.5, B) + cir(73, 62, 5.5, B) + cir(60, 86, 5, B),

  /* Ox — the horns are the whole drawing. */
  ox:
    pth("M34 44 C14 44 2 30 6 14 C18 18 20 34 36 32 Z") +
    pth("M86 44 C106 44 118 30 114 14 C102 18 100 34 84 32 Z") +
    ell(22, 60, 14, 9, 18) + ell(98, 60, 14, 9, -18) +
    pth("M26 46 C26 30 94 30 94 46 C94 78 82 102 60 102 C38 102 26 78 26 46 Z") +
    cir(46, 60, 5.5, B) + cir(74, 60, 5.5, B) +
    cir(52, 84, 4.5, B) + cir(68, 84, 4.5, B),

  /* Tiger — round ears and the 王 brow. */
  tiger:
    cir(28, 34, 18) + cir(92, 34, 18) +
    pth("M20 50 C20 28 100 28 100 50 C100 82 84 104 60 104 C36 104 20 82 20 50 Z") +
    bar("M44 86 L12 94", 3) + bar("M76 86 L108 94", 3) +
    cir(28, 34, 9, B) + cir(92, 34, 9, B) +
    bar("M60 33 L60 53", 4, B) + bar("M48 36 L72 36", 4, B) + bar("M48 50 L72 50", 4, B) +
    cir(44, 64, 5.5, B) + cir(76, 64, 5.5, B) +
    pth("M53 78 L67 78 L60 87 Z", B) +
    bar("M22 62 L10 60", 4, B) + bar("M24 74 L13 76", 4, B) +
    bar("M98 62 L110 60", 4, B) + bar("M96 74 L107 76", 4, B),

  /* Rabbit — two tall ears. */
  rabbit:
    pth("M34 48 C27 30 27 12 36 6 C47 5 48 24 46 42 Z") +
    pth("M86 48 C93 30 93 12 84 6 C73 5 72 24 74 42 Z") +
    ell(60, 68, 35, 31) +
    pth("M37 43 C33 30 33 17 38 12 C43 18 42 31 41 42 Z", B) +
    pth("M83 43 C87 30 87 17 82 12 C77 18 78 31 79 42 Z", B) +
    cir(47, 64, 5.5, B) + cir(73, 64, 5.5, B) +
    pth("M54 80 L66 80 L60 88 Z", B),

  /* Dragon — swept horns, mane spikes, whiskers. */
  dragon:
    pth("M36 38 C26 24 12 18 4 22 C12 30 16 42 32 48 Z") +
    pth("M84 38 C94 24 108 18 116 22 C108 30 104 42 88 48 Z") +
    pth("M26 60 L6 58 L22 74 Z") + pth("M94 60 L114 58 L98 74 Z") +
    pth("M24 52 C24 34 96 34 96 52 C96 68 90 78 78 82 C74 94 66 100 60 100" +
        "C54 100 46 94 42 82 C30 78 24 68 24 52 Z") +
    bar("M42 86 C26 98 12 94 10 82", 3.4) + bar("M78 86 C94 98 108 94 110 82", 3.4) +
    cir(44, 58, 6.5, B) + cir(76, 58, 6.5, B) +
    bar("M34 46 C40 41 50 41 55 46", 3.4, B) + bar("M65 46 C70 41 80 41 86 46", 3.4, B) +
    cir(53, 80, 3.8, B) + cir(67, 80, 3.8, B),

  /* Goat — swept horns, side ears, a tapered face and a beard. */
  goat:
    bar("M42 34 C34 14 17 11 10 23", 7) +
    bar("M78 34 C86 14 103 11 110 23", 7) +
    ell(21, 58, 16, 8, 28) + ell(99, 58, 16, 8, -28) +
    pth("M30 50 C30 34 90 34 90 50 C90 76 78 100 60 100 C42 100 30 76 30 50 Z") +
    pth("M54 96 C52 113 68 113 66 96 Z") +
    cir(46, 60, 5.5, B) + cir(74, 60, 5.5, B) +
    cir(54, 84, 3.6, B) + cir(66, 84, 3.6, B),

  /* Monkey — side ears and a muzzle that breaks the chin. */
  monkey:
    cir(18, 56, 18) + cir(102, 56, 18) +
    ell(60, 58, 32, 31) +
    ell(60, 80, 22, 16) + ring(60, 80, 22, 16, 3.4) +
    cir(18, 56, 9, B) + cir(102, 56, 9, B) +
    cir(49, 52, 5.5, B) + cir(71, 52, 5.5, B) +
    cir(53, 75, 3, B) + cir(67, 75, 3, B) +
    bar("M50 86 C55 91 65 91 70 86", 3.2, B),

  /* Rooster — the one profile in the set. A comb, a beak and a wattle only
     resolve side on; drawn front on it reads as a lump with a nose. */
  rooster:
    pth("M42 32 C36 16 47 6 53 18 C57 4 71 6 71 19 C79 10 89 19 83 32" +
        "C72 27 53 27 42 32 Z") +
    cir(64, 54, 26) +
    pth("M42 42 L10 52 L42 62 Z") +
    cir(45, 75, 7.5) + cir(55, 83, 6.5) +
    cir(60, 46, 6, B) + cir(34, 51, 2.6, B),

  /* Dog — long ears and a muzzle. */
  dog:
    pth("M32 44 C14 46 6 66 13 84 C22 94 35 86 37 68 Z") +
    pth("M88 44 C106 46 114 66 107 84 C98 94 85 86 83 68 Z") +
    ell(60, 60, 31, 30) +
    ell(60, 84, 20, 15) + ring(60, 84, 20, 15, 3) +
    ell(60, 74, 7.5, 5.5, 0, B) +
    bar("M60 79 L60 88", 3, B) +
    bar("M60 88 C55 93 50 92 48 88", 2.8, B) + bar("M60 88 C65 93 70 92 72 88", 2.8, B) +
    cir(47, 54, 5.5, B) + cir(73, 54, 5.5, B),

  /* Pig — the snout disc. */
  pig:
    pth("M30 50 C16 30 28 12 44 22 C50 32 48 46 43 54 Z") +
    pth("M90 50 C104 30 92 12 76 22 C70 32 72 46 77 54 Z") +
    ell(60, 66, 34, 30) +
    ell(60, 82, 18, 13) + ring(60, 82, 18, 13, 3.4) +
    cir(53, 82, 4, B) + cir(67, 82, 4, B) +
    cir(46, 58, 5.5, B) + cir(74, 58, 5.5, B)
};

/* A mask id has to be unique per instance, not per animal. A page carries the
   same animal on a dozen lanterns, and `url(#id)` resolves to whichever comes
   first in the document — so sharing one id means removing that first lantern
   silently unmasks every other one, and each glyph fills its whole box. */
let maskSeq = 0;

/* ---------------------------------------------------------------------------
   Lanterns
   ---------------------------------------------------------------------------
   Drawn in a 64x104 box, hung from the top edge so a string can place them by
   their hook rather than by their middle. The glyph, when there is one, is
   scaled into the belly; a plain lantern between the marked ones keeps a
   string from reading as one shape repeated.
   -------------------------------------------------------------------------- */

/* The glyph as bare children, to be dropped inside a <g> on the lantern rather
   than nested as its own <svg>: a nested svg re-establishes its own viewport
   and would ignore the parent's transform. */
const zodiacGlyph = (animal) => {
  const id = `ln-mask-${(maskSeq += 1)}`;
  return `<mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="120" height="120">` +
    `<rect width="120" height="120" fill="#000"/>${ZODIAC[animal]}</mask>` +
    `<rect width="120" height="120" fill="${GOLD_LT}" mask="url(#${id})"/>`;
};

const lanternSvg = (animal) =>
  `<svg class="ln-lantern-art" viewBox="0 0 64 104" aria-hidden="true" focusable="false">` +
  `<path d="M32 0 V10" stroke="${GOLD}" stroke-width="2"/>` +
  `<rect x="25" y="8" width="14" height="7" rx="1.5" fill="${GOLD}"/>` +
  `<ellipse cx="32" cy="52" rx="27" ry="36" fill="${RED}"/>` +
  `<path d="M32 16 C20 26 20 78 32 88 C44 78 44 26 32 16 Z" fill="${RED_DEEP}" opacity=".22"/>` +
  `<rect x="6" y="14" width="52" height="7" rx="2" fill="${GOLD}"/>` +
  `<rect x="6" y="83" width="52" height="7" rx="2" fill="${GOLD}"/>` +
  (animal
    ? `<g transform="translate(15 35) scale(0.28)">${zodiacGlyph(animal)}</g>`
    : `<circle cx="32" cy="52" r="11" fill="none" stroke="${GOLD_LT}" stroke-width="2.5" opacity=".8"/>` +
      `<circle cx="32" cy="52" r="4" fill="${GOLD_LT}" opacity=".8"/>`) +
  `<path d="M32 90 V96" stroke="${GOLD}" stroke-width="3"/>` +
  `<path d="M28 96 h8 l-2 8 h-4 z" fill="${GOLD}"/>` +
  `</svg>`;

/* ---------------------------------------------------------------------------
   The string
   ---------------------------------------------------------------------------
   Same construction as Winter's light strings and Muertos' papel picado, for
   the same reason. The wire is a single parabola drawn as an SVG stretched to
   the host's width — a stretched curve is still a smooth curve. The lanterns
   are not in that SVG: they are positioned elements at a percentage across and
   a pixel height from the same formula, so they keep their shape at any width
   and always hang off the wire.
   -------------------------------------------------------------------------- */

const WIRE_W = 1200;

/**
 * `top` is where the string meets its two ends, `sag` how far it dips at the
 * centre. Returns the markup and the height the host has to reserve.
 */
const lanternString = ({ seed, count, animal, width, sag, top = 0, motion }) => {
  const rand = seeded(seed);
  const wireY = (t) => top + 4 * sag * t * (1 - t);

  const points = [];
  for (let i = 0; i <= 24; i += 1) {
    const t = i / 24;
    points.push(`${(t * WIRE_W).toFixed(0)} ${wireY(t).toFixed(1)}`);
  }

  const height = top + sag + width * 1.7 + 8;
  const lanterns = [];
  for (let i = 0; i < count; i += 1) {
    const t = (i + 0.5) / count;
    /* Every other lantern carries the animal; the plain ones in between stop
       the string reading as one shape stamped out N times. */
    const marked = i % 2 === 0;
    const w = marked ? width : width * 0.74;
    lanterns.push(
      `<span class="ln-lantern${motion ? " ln-lantern-live" : ""}" ` +
        `style="left:${(t * 100).toFixed(3)}%;top:${wireY(t).toFixed(1)}px;` +
        `width:${w.toFixed(1)}px;margin-left:${(-w / 2).toFixed(1)}px;` +
        `--sway:${(3.4 + rand() * 2).toFixed(2)}s;--sway-delay:${(-rand() * 4).toFixed(2)}s">` +
        lanternSvg(marked ? animal : null) +
        `</span>`
    );
  }

  return {
    height,
    html:
      `<svg class="ln-wire" viewBox="0 0 ${WIRE_W} ${height.toFixed(1)}" preserveAspectRatio="none" ` +
      `aria-hidden="true" focusable="false"><path d="M${points.join("L")}" fill="none" ` +
      `stroke="rgba(232,182,76,.5)" stroke-width="1.4"></path></svg>` +
      lanterns.join("")
  };
};

/* ---------------------------------------------------------------------------
   Plum blossom
   ---------------------------------------------------------------------------
   梅花 opens in the cold weeks before the new year, which is exactly why it is
   the flower the holiday is drawn with.

   The tree is grown recursively from a seed rather than drawn by hand: a hand
   drawing of a hundred branches is unreadable as source and impossible to
   retune, and the generator gives the same tree every time anyway. Limbs are
   round-capped strokes that thin with depth; blossoms cluster on the outer
   wood, where they actually grow.
   -------------------------------------------------------------------------- */

const TREE_W = 1200;
const TREE_H = 400;

/**
 * `roots` are the limbs growth starts from — a trunk rising off the bottom for
 * a tree, or a bough entering from a corner for a branch. Everything after the
 * first generation is identical either way, which is the point: one generator,
 * and the composition is a matter of where it is told to start.
 */
const blossom = ({ seed, roots, align = "xMinYMax", fit = "meet", cls = "ln-tree" }) => {
  const rand = seeded(seed);
  const limbs = [];
  const sites = [];

  const grow = (x, y, ang, len, wid, depth) => {
    /* Nothing in the tree is ever a ruler-straight stick: the control point is
       nudged off the straight line by a seeded amount. */
    const bend = (rand() - 0.5) * 0.5;
    const x2 = x + Math.cos(ang) * len;
    const y2 = y + Math.sin(ang) * len;
    const mx = x + Math.cos(ang + bend) * len * 0.55;
    const my = y + Math.sin(ang + bend) * len * 0.55;
    limbs.push({
      d: `M${x.toFixed(1)} ${y.toFixed(1)}Q${mx.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`,
      w: wid
    });

    if (depth >= 5 || len < 14) {
      sites.push([x2, y2, depth]);
      return;
    }
    if (depth >= 3) {
      sites.push([mx, my, depth]);
    }
    const kids = depth < 2 ? 2 : rand() < 0.32 ? 3 : 2;
    for (let i = 0; i < kids; i += 1) {
      const spread = 0.3 + rand() * 0.42;
      const dir = i === 0 ? -spread : i === 1 ? spread * 0.85 : spread * 0.15;
      grow(x2, y2, ang + dir, len * (0.68 + rand() * 0.14), wid * 0.66, depth + 1);
    }
  };

  roots.forEach(({ x, y, ang, len, wid, depth = 0 }) => grow(x, y, ang, len, wid, depth));

  /* One blossom, defined once at unit radius and instanced. Spelling each one
     out costs twelve circles apiece and ran the finished tree past 150 KB of
     markup; a definition plus a transform draws the identical picture for a
     fraction of that. Petals take currentColor so one definition serves every
     tint, and only the `color` on each <use> changes. */
  const def =
    `<defs><g id="ln-bloom">` +
    Array.from({ length: 5 }, (_, i) => {
      const a = (i * 72 * Math.PI) / 180;
      return `<circle cx="${(Math.cos(a) * 0.62).toFixed(3)}" cy="${(Math.sin(a) * 0.62).toFixed(3)}" r=".52" fill="currentColor"/>`;
    }).join("") +
    Array.from({ length: 6 }, (_, i) => {
      const a = ((i * 60 + 18) * Math.PI) / 180;
      return `<circle cx="${(Math.cos(a) * 0.44).toFixed(3)}" cy="${(Math.sin(a) * 0.44).toFixed(3)}" r=".1" fill="${GOLD}"/>`;
    }).join("") +
    `<circle r=".2" fill="${GOLD}"/></g></defs>`;

  const blooms = sites
    .map(([x, y, depth]) => {
      if (rand() < 0.12) {
        return "";
      }
      const r = (depth >= 4 ? 7.5 : depth >= 3 ? 9.5 : 11) * (0.78 + rand() * 0.5);
      if (rand() < 0.2) {
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r * 0.42).toFixed(1)}" fill="${PETAL_DEEP}"/>`;
      }
      return `<use href="#ln-bloom" color="${rand() < 0.34 ? PETAL_DEEP : PETAL}" ` +
        `transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${(rand() * 72).toFixed(0)}) scale(${r.toFixed(2)})"/>`;
    })
    .join("");

  /* Widest limbs first, so a thick branch never paints over a thin one that
     grew out of it. */
  const wood = limbs
    .sort((a, b) => b.w - a.w)
    .map((l) => `<path d="${l.d}" fill="none" stroke="${BARK}" stroke-width="${l.w.toFixed(1)}" stroke-linecap="round"/>`)
    .join("");

  /* `meet` rather than `slice`: slice scales the drawing up to cover its box
     and crops whatever does not fit, which lops the crown or the trunk off a
     tree whose box is not exactly 3:1. Fitting the whole drawing inside and
     letting the box letterbox it is what keeps a tree a whole tree. */
  return `<svg class="${cls}" viewBox="0 0 ${TREE_W} ${TREE_H}" preserveAspectRatio="${align} ${fit}" ` +
    `aria-hidden="true" focusable="false">${def}${wood}${blooms}</svg>`;
};

/* A tree: a trunk off the bottom edge with a second, lower stem beside it. */
const blossomTree = (seed) =>
  blossom({
    seed,
    roots: [
      { x: 232, y: TREE_H - 4, ang: -Math.PI / 2 + 0.3, len: 108, wid: 26 },
      { x: 248, y: TREE_H - 2, ang: -Math.PI / 2 + 0.64, len: 62, wid: 14, depth: 1 }
    ]
  });

/* A single petal, drawn in a 14x12 box, for the drift and for card corners. */
const petalSvg = (color) =>
  `<svg viewBox="0 0 14 12" aria-hidden="true" focusable="false">` +
  `<path d="M7 0.5 C11.5 3 12.5 8 7 11.5 C1.5 8 2.5 3 7 0.5 Z" fill="${color}"></path></svg>`;

/* One blossom at a fixed size, for card corners and the portrait ring. */
const blossomSvg = (color = PETAL, r = 9) =>
  `<svg viewBox="-14 -14 28 28" aria-hidden="true" focusable="false">` +
  Array.from({ length: 5 }, (_, i) => {
    const a = (i * 72 * Math.PI) / 180;
    return `<circle cx="${(Math.cos(a) * r * 0.62).toFixed(2)}" cy="${(Math.sin(a) * r * 0.62).toFixed(2)}" r="${(r * 0.52).toFixed(2)}" fill="${color}"/>`;
  }).join("") +
  `<circle r="${(r * 0.22).toFixed(2)}" fill="${GOLD}"/></svg>`;

/* The portrait outline: a gold ring on the photograph's own edge with a finer
   red one inside it, and four blossoms set on the circle at the diagonals.
   Drawn as a circle rather than a scatter because it is framing a face — the
   shape it traces should be the shape of the thing it is framing. */
const PORTRAIT_RING = (() => {
  const blooms = [45, 135, 225, 315]
    .map((deg) => {
      const a = (deg * Math.PI) / 180;
      return `<g transform="translate(${(Math.cos(a) * 60).toFixed(1)} ${(Math.sin(a) * 60).toFixed(1)})">` +
        Array.from({ length: 5 }, (_, i) => {
          const p = (i * 72 * Math.PI) / 180;
          return `<circle cx="${(Math.cos(p) * 4.4).toFixed(2)}" cy="${(Math.sin(p) * 4.4).toFixed(2)}" r="3.7" fill="${PETAL}"/>`;
        }).join("") +
        `<circle r="1.6" fill="${GOLD}"/></g>`;
    })
    .join("");
  return `<svg class="ln-ring-art" viewBox="-72 -72 144 144" aria-hidden="true" focusable="false">` +
    `<circle r="60" fill="none" stroke="${GOLD}" stroke-width="2.4"/>` +
    `<circle r="54" fill="none" stroke="${RED}" stroke-width="1.4" opacity=".85"/>` +
    `${blooms}</svg>`;
})();

/* ---------------------------------------------------------------------------
   Petal drift
   -------------------------------------------------------------------------- */

const buildPetals = (overlay, density, motion) => {
  const count = petalCount(density);
  if (count === 0) {
    return;
  }
  overlay.appendChild(
    buildParticles(count, 19, (index, rand) => {
      const size = range(rand, 9, 17);
      const duration = range(rand, 13, 26);
      const spin = range(rand, 3.5, 8);

      /* With motion off the petals are not hidden, they have *fallen*: seeded
         positions spread down the viewport so the scene still reads as a
         drift, just a photograph of one. */
      const outer = make("div", {
        class: `season-particle${motion ? " ln-petal-fall" : ""}`,
        style: {
          left: `${(rand() * 100).toFixed(2)}%`,
          top: motion ? "-40px" : `${range(rand, 2, 96).toFixed(2)}vh`,
          width: `${size.toFixed(1)}px`,
          opacity: range(rand, 0.35, 0.85).toFixed(2),
          "--dur": `${duration.toFixed(1)}s`,
          /* A negative delay starts each petal partway down, so the first
             frame is already a drift rather than an empty sky that fills over
             the next twenty seconds. */
          "--delay": `${(-rand() * duration).toFixed(1)}s`
        }
      });

      const inner = make("div", {
        class: `ln-petal${motion ? " ln-petal-spin" : ""}`,
        style: {
          "--spin": `${spin.toFixed(1)}s`,
          "--spin-delay": `${(-rand() * 5).toFixed(1)}s`
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

  /* Falls back to the rat rather than to nothing: the switcher can force this
     season on any day of the year, and a lantern with an empty belly is worse
     than one carrying an animal that is merely out of season. */
  const animal = lunarAnimalForDate(today()) ?? "rat";

  buildPetals(overlay, density, motion);

  /* Header: a gold rule under the bar and nothing else.

     Lanterns were strung off this edge first, the way Winter hangs its lights,
     and it does not work here: the header is sticky, so a string hung off its
     bottom travels down the page with it, and a 34px lantern dragged over body
     copy is very different from a 6px bulb doing the same. The lanterns moved
     to the hero, which does not move. */
  decorate(disposer, ".site-header", "season-edge-strip ln-edge", "");

  /* Hero: a night that goes to red at the horizon, a moon, a string of
     lanterns across the top, and a bough of plum reaching in from the left. */
  const heroString = lanternString({ seed: 13, count: 8, animal, width: 40, sag: 22, motion });
  decorate(
    disposer,
    ".hero",
    "season-scene ln-hero",
    `<div class="season-sky"></div>
     <span class="ln-moon"></span>
     <div class="ln-hero-string">${heroString.html}</div>
     ${blossomTree(7)}`
  );

  /* Footer: the same night, mirrored, with a second tree and its own string. */
  const footerString = lanternString({ seed: 29, count: 7, animal, width: 30, sag: 14, motion });
  decorate(
    disposer,
    ".site-footer",
    "season-scene ln-footer",
    `<div class="season-sky"></div>
     <div class="ln-footer-string">${footerString.html}</div>
     <div class="ln-footer-tree">${blossomTree(23)}</div>`
  );

  /* The seam under the hero: a shallower string draped over it, hanging into
     the section below. It lives in the overview rather than the hero, because
     the hero clips its children.

     Deliberately small. A lantern is 1.6x as tall as it is wide, and these
     hang into a section's top padding — about 50px before the kicker starts.
     At the size the hero uses they land squarely on the heading. */
  const seam = lanternString({ seed: 37, count: 7, animal, width: 20, sag: 12, motion });
  decorate(disposer, ".section-overview", "season-scene ln-string ln-string-seam", seam.html, {
    first: true
  });

  /* The subpages reuse `.section-dashboard` as their only section with no
     overview before it, so a bare class selector would hang a second string
     directly under the header on every one of them. */
  const seam2 = lanternString({ seed: 41, count: 6, animal, width: 17, sag: 9, motion });
  decorate(
    disposer,
    ".section-overview + .section-dashboard",
    "season-scene ln-string ln-string-seam",
    seam2.html,
    { first: true }
  );

  /* Portraits: a ring of blossom, on hover only. */
  decorate(disposer, ".speaker-directory-photo, .seminar-speaker-photo", "season-ring ln-ring", PORTRAIT_RING);

  /* A blossom resting in the corner of each speaker card. Kept very faint —
     it is a watermark, not a badge. */
  decorate(
    disposer,
    ".speaker-directory-card",
    "season-card-art ln-card-blossom",
    `<span class="ln-corner-blossom">${blossomSvg(PETAL, 9)}</span>`
  );

  /* A red and gold frame on the community and feature cards, on hover: two
     inset rules with a blossom set into each corner. Both families are grids
     of several cards, so it stays off until the pointer is on one — a frame
     drawn on every card at rest turns the grid into a chessboard. */
  decorate(
    disposer,
    ".community-card, .feature-card",
    "season-card-art ln-card-frame",
    `<span class="ln-frame">
       <span class="ln-frame-gold"></span>
       <span class="ln-frame-red"></span>
       <span class="ln-frame-bloom ln-frame-bloom-tl">${blossomSvg(PETAL, 7)}</span>
       <span class="ln-frame-bloom ln-frame-bloom-tr">${blossomSvg(PETAL_DEEP, 6)}</span>
       <span class="ln-frame-bloom ln-frame-bloom-bl">${blossomSvg(PETAL_DEEP, 6)}</span>
       <span class="ln-frame-bloom ln-frame-bloom-br">${blossomSvg(PETAL, 7)}</span>
     </span>`
  );

  decorate(
    disposer,
    ".talk-card",
    "season-card-art ln-card-corner",
    `<span class="ln-corner ln-corner-tr">${blossomSvg(PETAL, 7)}</span>
     <span class="ln-corner ln-corner-bl">${blossomSvg(PETAL_DEEP, 6)}</span>`
  );

  return {
    destroy() {
      disposer.dispose();
    }
  };
};
