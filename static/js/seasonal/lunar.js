/**
 * Lunar New Year — the eve, the day, and the day after.
 *
 * Red and gold, lanterns and plum blossom. The one season on the site that
 * cannot be derived from a month, so its dates live in `lunar-dates.js` and
 * both this file and the orchestrator read them from there.
 *
 * Two ideas carry the layer. Lanterns, strung off the eave of a glazed roof
 * across the top of the hero and off the footer, each carrying the zodiac
 * animal whose year is beginning — so the decoration says *which* new year
 * it is, and says something different in 2032 than it did in 2031. And a
 * plum tree in bloom across the hero, because 梅花 opens in the cold just
 * before the new year and is the flower the holiday is drawn with. Behind
 * the tree a city; over it, fireworks and sky lanterns.
 *
 * The fireworks are the one canvas engine here, a small particle system on
 * the shared Surface and Loop. Everything else is markup on CSS keyframes:
 * the petals are fixed-count DOM particles, the lanterns sway on their own
 * clocks, and the tree and the city are generated once from a seed — so the
 * same density always draws the same scene.
 */

import { Disposer, Loop, Surface, buildParticles, decorate, make, pick, range, seeded } from "./engine.js";
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
const blossom = ({ seed, roots, align = "xMinYMax", fit = "meet", cls = "ln-tree", lush = false, box = `0 0 ${TREE_W} ${TREE_H}` }) => {
  const rand = seeded(seed);
  const limbs = [];
  const sites = [];
  /* A lush tree grows one generation further and puts two or three blooms
     at every site instead of one, so the crown fills in the way a tree in
     full flower does rather than a tree just coming into it. */
  const maxDepth = lush ? 6 : 5;

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

    if (depth >= maxDepth || len < (lush ? 10 : 14)) {
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

  const bloomAt = (x, y, depth) => {
    const r = (depth >= 4 ? 7.5 : depth >= 3 ? 9.5 : 11) * (0.78 + rand() * 0.5);
    if (rand() < 0.2) {
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r * 0.42).toFixed(1)}" fill="${PETAL_DEEP}"/>`;
    }
    return `<use href="#ln-bloom" color="${rand() < 0.34 ? PETAL_DEEP : PETAL}" ` +
      `transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${(rand() * 72).toFixed(0)}) scale(${r.toFixed(2)})"/>`;
  };
  const blooms = sites
    .map(([x, y, depth]) => {
      if (rand() < (lush ? 0.05 : 0.12)) {
        return "";
      }
      let out = bloomAt(x, y, depth);
      if (lush) {
        const extra = rand() < 0.65 ? (rand() < 0.4 ? 2 : 1) : 0;
        for (let i = 0; i < extra; i += 1) {
          out += bloomAt(x + (rand() - 0.5) * 22, y + (rand() - 0.5) * 22, depth);
        }
      }
      return out;
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
  return `<svg class="${cls}" viewBox="${box}" preserveAspectRatio="${align} ${fit}" ` +
    `aria-hidden="true" focusable="false">${def}${wood}${blooms}</svg>`;
};

/* A tree: a trunk off the bottom edge with a second, lower stem beside it.
   The lush one is cropped to the tree's own extent — the generator's canvas
   is three times wider than the tree, and fitting all of that into a box
   leaves the crown a third of the size the box suggests. */
const blossomTree = (seed, lush = false) =>
  blossom({
    seed,
    lush,
    box: lush ? "0 0 720 400" : undefined,
    roots: [
      { x: 232, y: TREE_H - 4, ang: -Math.PI / 2 + 0.3, len: 108, wid: 26 },
      { x: 248, y: TREE_H - 2, ang: -Math.PI / 2 + 0.64, len: 62, wid: 14, depth: 1 }
    ]
  });

/* ---------------------------------------------------------------------------
   The eave
   ---------------------------------------------------------------------------
   The top of the hero is the edge of a glazed roof seen from just under it:
   columns of yellow barrel tiles running up and away in perspective, rows of
   channel tiles stepping down between them, the round end caps and pointed
   drip tiles along the lip, and beneath that the painted beam — blue with a
   gold fret — and the red timber the lanterns hang from.

   Drawn in a fixed 2000-unit canvas because the columns converge on a
   vanishing point, which a repeating pattern cannot do. `xMidYMax slice`
   keeps the lip on the container's bottom edge and crops the sides on a
   narrow screen, where the middle columns are the near-vertical ones.
   -------------------------------------------------------------------------- */

const EAVE_W = 2000;
const EAVE_H = 214;
const EAVE_LIP = 160;
const EAVE_VP = { x: 1000, y: -700 };

const eaveSvg = () => {
  const parts = [];
  const toward = (xb, y) => EAVE_VP.x + (xb - EAVE_VP.x) * ((y - EAVE_VP.y) / (EAVE_LIP - EAVE_VP.y));

  /* Channel tiles: the field between the barrels. */
  parts.push(`<rect x="0" y="0" width="${EAVE_W}" height="${EAVE_LIP}" fill="#b97d1e"/>`);

  /* Barrel columns. */
  const step = 40;
  for (let xb = 20; xb < EAVE_W; xb += step) {
    const half = 9;
    const xt = toward(xb, 0);
    const halfT = half * ((0 - EAVE_VP.y) / (EAVE_LIP - EAVE_VP.y));
    parts.push(
      `<path d="M${(xb - half).toFixed(1)} ${EAVE_LIP}L${(xb + half).toFixed(1)} ${EAVE_LIP}L${(xt + halfT).toFixed(1)} 0L${(xt - halfT).toFixed(1)} 0Z" fill="url(#ln-barrel)"/>`
    );
  }

  /* Rows of tiles stepping down, closer together toward the top. */
  let y = EAVE_LIP;
  let gap = 28;
  while (gap > 2.5) {
    parts.push(`<line x1="0" y1="${y.toFixed(1)}" x2="${EAVE_W}" y2="${y.toFixed(1)}" stroke="rgba(80,44,8,.55)" stroke-width="${Math.max(1, gap / 9).toFixed(1)}"/>`);
    parts.push(`<line x1="0" y1="${(y - gap * 0.12).toFixed(1)}" x2="${EAVE_W}" y2="${(y - gap * 0.12).toFixed(1)}" stroke="rgba(255,228,150,.35)" stroke-width="${Math.max(0.6, gap / 16).toFixed(1)}"/>`);
    y -= gap;
    gap *= 0.78;
  }

  /* Distance: the field darkens as it climbs away. */
  parts.push(`<rect x="0" y="0" width="${EAVE_W}" height="${EAVE_LIP}" fill="url(#ln-eave-far)"/>`);

  /* Drip tiles between the columns, then the end caps over them. */
  for (let xb = 20; xb < EAVE_W; xb += step) {
    const x = xb + step / 2;
    parts.push(
      `<path d="M${x - 13} ${EAVE_LIP - 6}Q${x} ${EAVE_LIP - 2} ${x + 13} ${EAVE_LIP - 6}L${x + 9} ${EAVE_LIP + 8}Q${x} ${EAVE_LIP + 18} ${x - 9} ${EAVE_LIP + 8}Z" fill="#d9a02b" stroke="#7a4a10" stroke-width="1.5"/>` +
        `<path d="M${x - 5} ${EAVE_LIP + 2}Q${x} ${EAVE_LIP + 10} ${x + 5} ${EAVE_LIP + 2}" fill="none" stroke="#7a4a10" stroke-width="1"/>`
    );
  }
  for (let xb = 20; xb < EAVE_W; xb += step) {
    parts.push(
      `<circle cx="${xb}" cy="${EAVE_LIP}" r="13" fill="#d9a02b" stroke="#7a4a10" stroke-width="2"/>` +
        `<circle cx="${xb}" cy="${EAVE_LIP}" r="7.5" fill="none" stroke="#7a4a10" stroke-width="1.2"/>` +
        `<circle cx="${xb}" cy="${EAVE_LIP}" r="2.4" fill="#7a4a10"/>` +
        [45, 135, 225, 315]
          .map((deg) => {
            const a = (deg * Math.PI) / 180;
            return `<circle cx="${(xb + Math.cos(a) * 10).toFixed(1)}" cy="${(EAVE_LIP + Math.sin(a) * 10).toFixed(1)}" r="1.5" fill="#7a4a10"/>`;
          })
          .join("")
    );
  }

  /* The beam: shadow under the lip, the painted band with its gold fret, a
     green fillet, and the red timber. */
  const beamTop = EAVE_LIP + 18;
  parts.push(`<rect x="0" y="${beamTop}" width="${EAVE_W}" height="${EAVE_H - beamTop}" fill="#17365a"/>`);
  parts.push(`<rect x="0" y="${beamTop}" width="${EAVE_W}" height="7" fill="rgba(0,0,0,.45)"/>`);
  for (let x = 6; x < EAVE_W; x += 36) {
    parts.push(
      `<rect x="${x}" y="${beamTop + 11}" width="12" height="10" fill="none" stroke="#e8b64c" stroke-width="1.4"/>` +
        `<rect x="${x + 4}" y="${beamTop + 15}" width="4" height="2" fill="#e8b64c"/>` +
        `<circle cx="${x + 24}" cy="${beamTop + 16}" r="2.2" fill="#3fa88f"/>`
    );
  }
  parts.push(`<rect x="0" y="${EAVE_H - 12}" width="${EAVE_W}" height="3" fill="#1f6b5a"/>`);
  parts.push(`<rect x="0" y="${EAVE_H - 9}" width="${EAVE_W}" height="9" fill="#6b1a12"/>`);
  parts.push(`<rect x="0" y="${EAVE_H - 9}" width="${EAVE_W}" height="1" fill="#e8b64c" opacity=".7"/>`);

  return `
    <svg class="ln-eave-art" viewBox="0 0 ${EAVE_W} ${EAVE_H}" preserveAspectRatio="xMidYMax slice"
         aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="ln-barrel" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stop-color="#9c6414"/>
          <stop offset=".3" stop-color="#f2c14e"/>
          <stop offset=".65" stop-color="#d9a02b"/>
          <stop offset="1" stop-color="#8a5510"/>
        </linearGradient>
        <linearGradient id="ln-eave-far" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stop-color="rgba(22,7,9,.7)"/>
          <stop offset=".6" stop-color="rgba(22,7,9,.15)"/>
          <stop offset="1" stop-color="rgba(22,7,9,0)"/>
        </linearGradient>
      </defs>
      ${parts.join("")}
    </svg>`;
};

/* ---------------------------------------------------------------------------
   The city
   ---------------------------------------------------------------------------
   Skyscrapers in silhouette along the bottom, in two rows for depth, with
   lit windows. A user-unit pattern like Winter's, so it repeats rather than
   stretches. Nothing fancy: it is the background the tree stands against.
   -------------------------------------------------------------------------- */

const CITY_W = 960;
const CITY_H = 190;

const cityRow = (rand, { minH, maxH, windows }) => {
  const bodies = [];
  const lit = { bright: [], dim: [] };
  let x = 0;
  while (x < CITY_W) {
    const w = range(rand, 26, 80);
    const h = range(rand, minH, maxH);
    const top = CITY_H - h;
    const x1 = Math.min(x + w, CITY_W);
    const roof = rand();
    let d = `M${x.toFixed(1)} ${CITY_H}V${top.toFixed(1)}`;
    if (roof < 0.2) {
      const mx = x + (x1 - x) / 2;
      d += `H${(mx - 1.2).toFixed(1)}V${(top - range(rand, 10, 28)).toFixed(1)}h2.4V${top.toFixed(1)}`;
    } else if (roof < 0.42) {
      const inset = (x1 - x) * range(rand, 0.18, 0.3);
      const rise = range(rand, 8, 18);
      d += `H${(x + inset).toFixed(1)}V${(top - rise).toFixed(1)}H${(x1 - inset).toFixed(1)}V${top.toFixed(1)}`;
    } else if (roof < 0.52) {
      d += `L${(x + (x1 - x) / 2).toFixed(1)} ${(top - range(rand, 14, 30)).toFixed(1)}`;
    }
    d += `H${x1.toFixed(1)}V${CITY_H}Z`;
    bodies.push(d);

    if (windows) {
      const cols = Math.floor((x1 - x - 8) / 11);
      const rows = Math.floor((h - 12) / 15);
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const on = rand();
          if (on > 0.5) {
            continue;
          }
          const rect = `M${(x + 5 + c * 11).toFixed(1)} ${(top + 8 + r * 15).toFixed(1)}h4v6h-4z`;
          (on < 0.2 ? lit.bright : lit.dim).push(rect);
        }
      }
    }
    x = x1 + range(rand, 2, 14);
  }
  return { bodies: bodies.join(""), lit };
};

const citySvg = (seed, id) => {
  const rand = seeded(seed);
  const far = cityRow(rand, { minH: 80, maxH: 170, windows: true });
  const near = cityRow(rand, { minH: 30, maxH: 110, windows: true });
  return `
    <svg class="ln-city-art" width="100%" height="${CITY_H}" aria-hidden="true" focusable="false">
      <defs>
        <pattern id="${id}" patternUnits="userSpaceOnUse" width="${CITY_W}" height="${CITY_H}">
          <path d="${far.bodies}" fill="#3a0a16" opacity=".85"/>
          <path d="${far.lit.bright.join("")}${far.lit.dim.join("")}" fill="#c9773d" opacity=".45"/>
          <path d="${near.bodies}" fill="#15030a"/>
          <path d="${near.lit.bright.join("")}" fill="#ffcf6b" opacity=".95"/>
          <path d="${near.lit.dim.join("")}" fill="#e09a4a" opacity=".55"/>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#${id})"></rect>
    </svg>`;
};

/* ---------------------------------------------------------------------------
   Fireworks
   ---------------------------------------------------------------------------
   A particle system on a canvas, on the shared `Surface` and `Loop` so it
   inherits DPR handling, the zero-size guard and the hidden-document pause.

   Rockets launch from behind the city at random intervals, trailing embers,
   and burst at their apex into a shell of sparks. Four shell types: a peony
   (an even sphere), a ring, a willow (gold, heavy, long-lived, so it droops)
   and a crackle (short-lived and twinkling). Sparks are drawn as short
   streaks from where they were to where they are, with `lighter` blending
   so overlapping sparks glow rather than paint over each other.

   Budget: a few hundred sparks at most, one path per spark per frame. On a
   phone that is a fraction of a millisecond.
   -------------------------------------------------------------------------- */

const SHELL_COLORS = ["#ff4d5e", "#ffd98a", "#ff9fbb", "#ffffff", "#e8b64c", "#ff7a3d"];
const MAX_SPARKS = 900;

class Fireworks {
  constructor(surface) {
    this.surface = surface;
    this.rockets = [];
    this.sparks = [];
    this.next = 0.6;
    this.t = 0;
  }

  launch() {
    const W = this.surface.width;
    const H = this.surface.height;
    const x = W * range(Math.random, 0.06, 0.94);
    const apex = H * range(Math.random, 0.14, 0.42);
    const start = H - 60;
    const g = 240;
    this.rockets.push({
      x,
      y: start,
      vx: range(Math.random, -12, 12),
      vy: -Math.sqrt(2 * g * (start - apex)),
      g,
      color: pick(Math.random, SHELL_COLORS),
      kind: Math.random() < 0.55 ? "peony" : Math.random() < 0.45 ? "willow" : Math.random() < 0.5 ? "ring" : "crackle"
    });
  }

  burst(r) {
    const kind = r.kind;
    const n = kind === "ring" ? 54 : kind === "crackle" ? 70 : kind === "willow" ? 90 : 110;
    const color = kind === "willow" ? "#ffd98a" : r.color;
    const second = kind === "peony" && Math.random() < 0.5 ? pick(Math.random, SHELL_COLORS) : null;
    /* A ring is tilted so it reads as a disc seen at an angle. */
    const tilt = range(Math.random, 0.35, 0.8);
    for (let i = 0; i < n; i += 1) {
      const a = kind === "ring" ? (i / n) * Math.PI * 2 : Math.random() * Math.PI * 2;
      let sp;
      if (kind === "ring") {
        sp = 170;
      } else if (kind === "willow") {
        sp = range(Math.random, 40, 150);
      } else if (kind === "crackle") {
        sp = range(Math.random, 60, 210);
      } else {
        /* Even sphere: speed by sqrt so the shell is not denser at the centre. */
        sp = 60 + 160 * Math.sqrt(Math.random());
      }
      const vx = Math.cos(a) * sp;
      const vy = Math.sin(a) * sp * (kind === "ring" ? tilt : 1);
      const ttl =
        kind === "willow" ? range(Math.random, 2.6, 3.6) : kind === "crackle" ? range(Math.random, 0.7, 1.4) : range(Math.random, 1.3, 2.1);
      this.sparks.push({
        x: r.x,
        y: r.y,
        px: r.x,
        py: r.y,
        vx: vx + r.vx * 0.3,
        vy,
        life: ttl,
        ttl,
        color: second && i % 2 ? second : color,
        size: kind === "willow" ? 1.6 : kind === "crackle" ? 1.3 : 1.9,
        g: kind === "willow" ? 210 : kind === "crackle" ? 90 : 120,
        drag: kind === "willow" ? 1.2 : 1.9,
        twinkle: kind === "crackle"
      });
    }
    /* A bright flash at the centre, gone in a blink. */
    this.sparks.push({ x: r.x, y: r.y, px: r.x, py: r.y, vx: 0, vy: 0, life: 0.12, ttl: 0.12, color: "#ffffff", size: 14, g: 0, drag: 0, twinkle: false, flash: true });
  }

  step(dt) {
    if (!this.surface.ready) {
      return;
    }
    this.t += dt;
    this.next -= dt;
    if (this.next <= 0) {
      this.launch();
      this.next = range(Math.random, 0.9, 2.4);
    }

    for (let i = this.rockets.length - 1; i >= 0; i -= 1) {
      const r = this.rockets[i];
      r.x += r.vx * dt;
      r.y += r.vy * dt;
      r.vy += r.g * dt;
      /* Embers off the tail. */
      if (this.sparks.length < MAX_SPARKS) {
        this.sparks.push({
          x: r.x,
          y: r.y,
          px: r.x,
          py: r.y,
          vx: range(Math.random, -18, 18),
          vy: range(Math.random, 20, 70),
          life: 0.45,
          ttl: 0.45,
          color: "#ffd98a",
          size: 1,
          g: 80,
          drag: 2,
          twinkle: false
        });
      }
      if (r.vy >= -20) {
        this.rockets.splice(i, 1);
        if (this.sparks.length < MAX_SPARKS - 130) {
          this.burst(r);
        }
      }
    }

    const k = Math.exp(-dt);
    for (let i = this.sparks.length - 1; i >= 0; i -= 1) {
      const s = this.sparks[i];
      s.life -= dt;
      if (s.life <= 0) {
        this.sparks.splice(i, 1);
        continue;
      }
      s.px = s.x;
      s.py = s.y;
      const d = Math.pow(k, s.drag);
      s.vx *= d;
      s.vy = s.vy * d + s.g * dt;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
    }
  }

  draw() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    for (const r of this.rockets) {
      ctx.strokeStyle = "rgba(255,225,160,.9)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(r.x, r.y + 16);
      ctx.lineTo(r.x, r.y);
      ctx.stroke();
    }
    for (const s of this.sparks) {
      const u = s.life / s.ttl;
      let alpha = u < 0.35 ? u / 0.35 : 1;
      if (s.twinkle && Math.random() < 0.35) {
        alpha *= 0.25;
      }
      if (s.flash) {
        ctx.globalAlpha = alpha * 0.9;
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size * (1.6 - u), 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.size;
      ctx.beginPath();
      ctx.moveTo(s.px, s.py);
      ctx.lineTo(s.x, s.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }

  /** Reduced motion: three shells hanging open in the sky. */
  drawStatic() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    const rand = seeded(5);
    const W = this.surface.width;
    const H = this.surface.height;
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    for (let k = 0; k < 3; k += 1) {
      const cx = W * (0.2 + k * 0.28 + rand() * 0.1);
      const cy = H * (0.2 + rand() * 0.2);
      const color = SHELL_COLORS[k * 2];
      const R = 46 + rand() * 24;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      for (let i = 0; i < 40; i += 1) {
        const a = (i / 40) * Math.PI * 2 + rand() * 0.1;
        const r = R * (0.8 + rand() * 0.2);
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a) * r * 0.55, cy + Math.sin(a) * r * 0.55);
        ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx + Math.cos(a) * r * 1.08, cy + Math.sin(a) * r * 1.08, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
}

/* ---------------------------------------------------------------------------
   Sky lanterns
   ---------------------------------------------------------------------------
   They rise slowly out of the city, drifting a little as they go, and fade
   high in the sky. Emitted before the city in the markup, so each one
   appears from behind a building rather than out of the ground. Negative
   delays spread them through their cycles, so the first frame already has
   several in the air.
   -------------------------------------------------------------------------- */

const SKY_LANTERN =
  `<svg viewBox="0 0 24 34" aria-hidden="true" focusable="false">` +
  `<path d="M5 4Q12 0 19 4L22 25Q12 30 2 25Z" fill="#ffb347"/>` +
  `<path d="M8 6Q12 4 16 6L17 22Q12 25 7 22Z" fill="#ffd98a" opacity=".85"/>` +
  `<path d="M5 4Q12 0 19 4L18.4 8Q12 5 5.6 8Z" fill="#e8892c"/>` +
  `<path d="M2 25Q12 30 22 25L21.6 28Q12 33 2.4 28Z" fill="#b85c1a"/>` +
  `<circle cx="12" cy="27" r="2" fill="#fff4d2"/>` +
  `</svg>`;

const skyLanternsHtml = (seed, n, motion) => {
  const rand = seeded(seed);
  return Array.from({ length: n }, () => {
    const dur = range(rand, 30, 52);
    const size = Math.round(range(rand, 16, 30));
    const left = range(rand, 4, 94);
    const rise = Math.round(range(rand, 640, 900));
    /* With motion off the lanterns are already up: each holds a seeded
       height, so the sky still has lanterns in it, just a still one. */
    const still = motion ? "" : `bottom:${Math.round(range(rand, 200, 620))}px;`;
    return (
      `<span class="ln-skylantern${motion ? " ln-skylantern-live" : ""}" ` +
      `style="left:${left.toFixed(1)}%;width:${size}px;${still}--rise:${rise}px;--dur:${dur.toFixed(1)}s;` +
      `--delay:${(-rand() * dur).toFixed(1)}s;--sway:${range(rand, 6, 10).toFixed(1)}s;--sway-delay:${(-rand() * 8).toFixed(1)}s">` +
      `<span class="ln-skylantern-sway">${SKY_LANTERN}</span></span>`
    );
  }).join("");
};

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

  /* Hero, from the back: the sky and the moon, the fireworks canvas, sky
     lanterns, the city, the tree, and in front of everything the eave along
     the top with the lantern string hanging from its beam. The lanterns and
     the fireworks come before the city so they start out behind it. */
  const heroString = lanternString({ seed: 13, count: 9, animal, width: 32, sag: 6, motion });
  const [heroScene] = decorate(
    disposer,
    ".hero",
    "season-scene ln-hero",
    `<div class="season-sky"></div>
     <span class="ln-moon"></span>
     <canvas class="ln-canvas" aria-hidden="true"></canvas>
     <div class="ln-skylanterns">${skyLanternsHtml(31, 8, motion)}</div>
     <div class="ln-city">${citySvg(3, "ln-city-hero")}</div>
     ${blossomTree(7, true)}
     <div class="ln-eave">${eaveSvg()}</div>
     <div class="ln-hero-string">${heroString.html}</div>`
  );

  /* The fireworks run on the hero's own canvas, so they scroll away with it
     rather than following the reader down the page. Subpages have no hero
     and simply get no fireworks. */
  let fireworks = null;
  let loop = null;
  const canvas = heroScene ? heroScene.querySelector(".ln-canvas") : null;
  if (canvas) {
    const surface = new Surface(canvas);
    disposer.add(() => surface.destroy());
    fireworks = new Fireworks(surface);
    loop = new Loop({
      motion,
      step: (dt) => fireworks.step(dt),
      /* With motion off, `Loop.start()` paints exactly one frame: three
         shells hanging open rather than an empty sky. */
      draw: () => (motion ? fireworks.draw() : fireworks.drawStatic())
    });
    disposer.add(() => loop.destroy());
    surface.onChange = () => loop.draw();
    loop.start();
  }

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
    /* Exposed for tuning and for verification: hidden documents never fire
       rAF, so this is the only way to advance the fireworks off-screen. */
    fireworks,
    loop,
    destroy() {
      disposer.dispose();
    }
  };
};
