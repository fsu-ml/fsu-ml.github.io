/**
 * Lunar New Year art — generator.
 *
 * Source of truth for the ten zodiac glyphs and the blossom tree. Run it to
 * rewrite the static review page and the individual SVGs:
 *
 *   node planning/build-art.mjs
 *
 * The page it writes carries no script of its own, so it survives being
 * emailed, opened from disk, or dropped into a viewer that strips JavaScript.
 * Nothing here is wired into the site yet.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const OUT = {};

const SHELL = String.raw`<title>Lunar New Year Art Plan</title>
<style>
  body{margin:0;background:#160709;color:#f4e8d6;
       font:14px/1.6 "Iowan Old Style",Palatino,Georgia,serif;padding:34px 24px 90px}
  h1{font-size:27px;margin:0 0 6px;color:#ffd98a;letter-spacing:.02em}
  h2{font-size:15px;margin:48px 0 6px;color:#e8b64c;border-bottom:1px solid #ffffff1f;
     padding-bottom:7px;text-transform:uppercase;letter-spacing:.15em;font-weight:600}
  p.note{color:#e8dccba8;max-width:74ch;margin:0 0 16px}
  code{font:12px ui-monospace,Menlo,monospace;color:#ffd98a}
  .wrap{max-width:1200px;margin:0 auto}

  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:16px}
  .cell{background:#ffffff0a;border:1px solid #ffffff14;border-radius:10px;padding:10px 8px 9px;text-align:center}
  .cell svg{width:100%;height:auto;display:block}
  .cell .nm{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:#e8b64c;margin-top:6px}
  .cell .yr{font-size:10px;color:#e8dccb59;letter-spacing:.05em}
  .art{color:#ffd98a}

  /* silhouette test: how it reads as one flat shape, small */
  .flip .cell{background:radial-gradient(circle at 50% 40%,#ffe3ad,#e0a63a 70%,#b3781f);padding:8px}
  .flip .art{color:#6e1119}
  .flip .nm{color:#5c1018}.flip .yr{color:#5c101880}
  .tiny .cell{padding:6px}
  .tiny{grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:10px}
  .tiny .nm,.tiny .yr{display:none}

  .lanterns{display:flex;gap:24px;flex-wrap:wrap;justify-content:center;padding:22px 0 6px}
  .lantern{width:124px;text-align:center}
  .lantern .cap{width:32px;height:11px;background:linear-gradient(#f2cd76,#8a5f14);margin:0 auto;border-radius:3px 3px 0 0}
  .lantern .body{position:relative;height:142px;border-radius:50%/36%;
      background:radial-gradient(ellipse at 34% 28%,#ff7a63,#d4192f 54%,#7d0a1c);
      box-shadow:0 0 38px #ff43335e, inset -11px 0 24px #0006;display:grid;place-items:center}
  .lantern .body::before,.lantern .body::after{content:"";position:absolute;left:5%;right:5%;height:9px;
      background:linear-gradient(#ffe6a8,#c9992f)}
  .lantern .body::before{top:-4px}.lantern .body::after{bottom:-4px}
  .lantern svg{width:76%;filter:drop-shadow(0 0 7px #ffdca188)}
  .lantern .art{color:#ffeec4}
  .lantern .tassel{width:5px;height:30px;background:linear-gradient(#e8b64c,#c8102e);margin:0 auto}
  .lantern .nm{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:#e8b64c;margin-top:9px}

  table{border-collapse:collapse;font-size:13px;margin:10px 0 22px}
  th,td{text-align:left;padding:5px 20px 5px 0;border-bottom:1px solid #ffffff12}
  th{color:#e8b64c;font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;font-weight:600}
  td.d{color:#e8dccbaa}
  pre{background:#ffffff0a;border:1px solid #ffffff14;border-radius:9px;padding:14px 16px;
      overflow-x:auto;font:12.5px/1.65 ui-monospace,Menlo,monospace;color:#f0e4d2}
  pre .c{color:#e8dccb66}
  .stage{border:1px solid #ffffff14;border-radius:12px;overflow:hidden;margin-bottom:18px}
  .stage svg{display:block;width:100%;height:auto}
</style>

<div class="wrap">
<h1>Lunar New Year — art plan</h1>
<p class="note">Papercut (剪纸) construction. Each animal is a mask: shapes in white build the body,
shapes in black cut it, and one <code>currentColor</code> rect shows through — so light passes through
every cut, which is what a lantern actually does. Each is a head seen front on: one head shape, the one
feature that names the animal, two eyes. All 120&times;120, all recolour from <code>currentColor</code>.</p>

<h2>The ten animals</h2>
<div class="grid" id="grid-a"></div>

<h2>Alternates</h2>
<p class="note">The rooster is the one animal that fights the front-on format &mdash; a comb, a beak and a
wattle only really resolve side on. Both are here; pick one.</p>
<div class="grid" id="grid-alt"></div>

<h2>Read as a flat silhouette</h2>
<p class="note">Colour reversed and the shape flattened — this is the test that matters.</p>
<div class="grid flip" id="grid-b"></div>

<h2>At lantern size (64px)</h2>
<div class="grid flip tiny" id="grid-c"></div>

<h2>In the lantern</h2>
<div class="lanterns" id="lanterns"></div>

<h2>Blossom tree — night</h2>
<div class="stage" id="tree-night"></div>
<h2>Blossom tree — gold on red</h2>
<div class="stage" id="tree-gold"></div>
<h2>Wiring — the three-day window</h2>
<p class="note">The date moves every year, so this is the one season that cannot be derived from a month.
A ten-entry table is the whole mechanism: find the year, and if today is within a day of it, the season is on.
That gives eve, day and the day after &mdash; three days, matching how the holiday is actually kept.</p>
<table id="dates"></table>
<pre id="code"></pre>
</div>`;

/* ===========================================================================
   Shape helpers. `f` is "#fff" to add to the body, "#000" to cut it away.
   =========================================================================== */
const W = "#fff", B = "#000";
const cir = (cx, cy, r, f = W) => `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${f}"/>`;
const ell = (cx, cy, rx, ry, rot = 0, f = W) =>
  `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"` +
  (rot ? ` transform="rotate(${rot} ${cx} ${cy})"` : "") + ` fill="${f}"/>`;
const pth = (d, f = W) => `<path d="${d}" fill="${f}"/>`;
const bar = (d, w, f = W) =>
  `<path d="${d}" fill="none" stroke="${f}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"/>`;
const ring = (cx, cy, rx, ry, w, f = B) =>
  `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="none" stroke="${f}" stroke-width="${w}"/>`;
/* A tapered limb: wide at the top, narrow at the foot, with a rounded pad. */
const leg = (x, y0, y1, w0, w1, f = W) =>
  pth(`M${x - w0 / 2} ${y0} L${x + w0 / 2} ${y0} L${x + w1 / 2} ${y1} L${x - w1 / 2} ${y1} Z`, f) +
  cir(x, y1, w1 / 2, f);

const ANIMALS = [];
const add = (id, label, year, parts) => ANIMALS.push({ id, label, year, parts });

/* The mask id has to be unique per instance, not per animal. Two copies of the
   same animal on one page would otherwise share one id, and `url(#id)` resolves
   to whichever comes first in the document — so hiding or removing that first
   copy silently unmasks every other one, and the glyph fills its whole box. */
let maskSeq = 0;
const svgOf = (a) => {
  const id = `lny-${a.id}-${(maskSeq += 1)}`;
  return `<svg viewBox="0 0 120 120" aria-hidden="true" focusable="false" class="art">` +
    `<mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="120" height="120">` +
    `<rect width="120" height="120" fill="#000"/>${a.parts}</mask>` +
    `<rect width="120" height="120" fill="currentColor" mask="url(#${id})"/></svg>`;
};

/* ---------------------------------------------------------------------------
   Ten heads, front on.
   ---------------------------------------------------------------------------
   One head shape, the one feature that names the animal, two eyes. Nothing
   else earns its place: at 64px inside a lantern a shoulder or a paw is just
   noise, while a horn or an ear is the whole identification.
   -------------------------------------------------------------------------- */

/* ---- Rat 鼠 — two big ears on a tapered face ----------------------------- */
add("rat", "Rat", "2032",
  cir(28, 38, 21) + cir(92, 38, 21) +
  pth("M24 52 C24 32 96 32 96 52 C96 78 80 104 60 104 C40 104 24 78 24 52 Z") +
  bar("M48 88 L14 96", 3) + bar("M72 88 L106 96", 3) +
  cir(28, 38, 11, B) + cir(92, 38, 11, B) +
  cir(47, 62, 5.5, B) + cir(73, 62, 5.5, B) +
  cir(60, 86, 5, B));

/* ---- Ox 牛 — the horns are the whole drawing ---------------------------- */
add("ox", "Ox", "2033",
  pth("M34 44 C14 44 2 30 6 14 C18 18 20 34 36 32 Z") +
  pth("M86 44 C106 44 118 30 114 14 C102 18 100 34 84 32 Z") +
  ell(22, 60, 14, 9, 18) + ell(98, 60, 14, 9, -18) +
  pth("M26 46 C26 30 94 30 94 46 C94 78 82 102 60 102 C38 102 26 78 26 46 Z") +
  cir(46, 60, 5.5, B) + cir(74, 60, 5.5, B) +
  cir(52, 84, 4.5, B) + cir(68, 84, 4.5, B));

/* ---- Tiger 虎 — round ears and the 王 brow ------------------------------ */
add("tiger", "Tiger", "2034",
  cir(28, 34, 18) + cir(92, 34, 18) +
  pth("M20 50 C20 28 100 28 100 50 C100 82 84 104 60 104 C36 104 20 82 20 50 Z") +
  bar("M44 86 L12 94", 3) + bar("M76 86 L108 94", 3) +
  cir(28, 34, 9, B) + cir(92, 34, 9, B) +
  bar("M60 33 L60 53", 4, B) + bar("M48 36 L72 36", 4, B) + bar("M48 50 L72 50", 4, B) +
  cir(44, 64, 5.5, B) + cir(76, 64, 5.5, B) +
  pth("M53 78 L67 78 L60 87 Z", B) +
  bar("M22 62 L10 60", 4, B) + bar("M24 74 L13 76", 4, B) +
  bar("M98 62 L110 60", 4, B) + bar("M96 74 L107 76", 4, B));

/* ---- Rabbit 兔 — two tall ears ------------------------------------------ */
add("rabbit", "Rabbit", "2035",
  pth("M34 48 C27 30 27 12 36 6 C47 5 48 24 46 42 Z") +
  pth("M86 48 C93 30 93 12 84 6 C73 5 72 24 74 42 Z") +
  ell(60, 68, 35, 31) +
  pth("M37 43 C33 30 33 17 38 12 C43 18 42 31 41 42 Z", B) +
  pth("M83 43 C87 30 87 17 82 12 C77 18 78 31 79 42 Z", B) +
  cir(47, 64, 5.5, B) + cir(73, 64, 5.5, B) +
  pth("M54 80 L66 80 L60 88 Z", B));

/* ---- Dragon 龙 — swept horns, mane spikes, whiskers --------------------- */
add("dragon", "Dragon", "2036",
  pth("M36 38 C26 24 12 18 4 22 C12 30 16 42 32 48 Z") +
  pth("M84 38 C94 24 108 18 116 22 C108 30 104 42 88 48 Z") +
  pth("M26 60 L6 58 L22 74 Z") + pth("M94 60 L114 58 L98 74 Z") +
  pth("M24 52 C24 34 96 34 96 52 C96 68 90 78 78 82 C74 94 66 100 60 100" +
      "C54 100 46 94 42 82 C30 78 24 68 24 52 Z") +
  bar("M42 86 C26 98 12 94 10 82", 3.4) + bar("M78 86 C94 98 108 94 110 82", 3.4) +
  cir(44, 58, 6.5, B) + cir(76, 58, 6.5, B) +
  bar("M34 46 C40 41 50 41 55 46", 3.4, B) + bar("M65 46 C70 41 80 41 86 46", 3.4, B) +
  cir(53, 80, 3.8, B) + cir(67, 80, 3.8, B));

/* ---- Goat 羊 — swept horns, side ears, a tapered face and a beard ------ */
add("goat", "Goat", "2027",
  bar("M42 34 C34 14 17 11 10 23", 7) +
  bar("M78 34 C86 14 103 11 110 23", 7) +
  ell(21, 58, 16, 8, 28) + ell(99, 58, 16, 8, -28) +
  pth("M30 50 C30 34 90 34 90 50 C90 76 78 100 60 100 C42 100 30 76 30 50 Z") +
  pth("M54 96 C52 113 68 113 66 96 Z") +
  cir(46, 60, 5.5, B) + cir(74, 60, 5.5, B) +
  cir(54, 84, 3.6, B) + cir(66, 84, 3.6, B));

/* ---- Monkey 猴 — side ears and a muzzle that breaks the chin ----------- */
add("monkey", "Monkey", "2028",
  cir(18, 56, 18) + cir(102, 56, 18) +
  ell(60, 58, 32, 31) +
  ell(60, 80, 22, 16) +
  ring(60, 80, 22, 16, 3.4) +
  cir(18, 56, 9, B) + cir(102, 56, 9, B) +
  cir(49, 52, 5.5, B) + cir(71, 52, 5.5, B) +
  cir(53, 75, 3, B) + cir(67, 75, 3, B) +
  bar("M50 86 C55 91 65 91 70 86", 3.2, B));

/* ---- Rooster 鸡 — the one profile in the set ---------------------------- */
add("rooster", "Rooster", "2029",
  pth("M42 32 C36 16 47 6 53 18 C57 4 71 6 71 19 C79 10 89 19 83 32" +
      "C72 27 53 27 42 32 Z") +
  cir(64, 54, 26) +
  pth("M42 42 L10 52 L42 62 Z") +
  cir(45, 75, 7.5) + cir(55, 83, 6.5) +
  cir(60, 46, 6, B) + cir(34, 51, 2.6, B));

/* ---- Dog 狗 — long ears and a muzzle ------------------------------------ */
add("dog", "Dog", "2030",
  pth("M32 44 C14 46 6 66 13 84 C22 94 35 86 37 68 Z") +
  pth("M88 44 C106 46 114 66 107 84 C98 94 85 86 83 68 Z") +
  ell(60, 60, 31, 30) +
  ell(60, 84, 20, 15) +
  ring(60, 84, 20, 15, 3) +
  ell(60, 74, 7.5, 5.5, 0, B) +
  bar("M60 79 L60 88", 3, B) +
  bar("M60 88 C55 93 50 92 48 88", 2.8, B) + bar("M60 88 C65 93 70 92 72 88", 2.8, B) +
  cir(47, 54, 5.5, B) + cir(73, 54, 5.5, B));

/* ---- Pig 猪 — the snout disc ------------------------------------------- */
add("pig", "Pig", "2031",
  pth("M30 50 C16 30 28 12 44 22 C50 32 48 46 43 54 Z") +
  pth("M90 50 C104 30 92 12 76 22 C70 32 72 46 77 54 Z") +
  ell(60, 66, 34, 30) +
  ell(60, 82, 18, 13) +
  ring(60, 82, 18, 13, 3.4) +
  cir(53, 82, 4, B) + cir(67, 82, 4, B) +
  cir(46, 58, 5.5, B) + cir(74, 58, 5.5, B));

/* ===========================================================================
   Blossom tree — a seeded recursive plum (梅) in bloom.
   Branches are round-capped strokes that thin with depth; blossoms are placed
   at the tips and scattered along the outer third of every limb.
   =========================================================================== */
const seeded = (s) => () => {
  s = (s * 1664525 + 1013904223) >>> 0;
  return s / 4294967296;
};

function tree({ key, bark, petal, petalAlt, heart, moon, sky, seed = 7 }) {
  const w = 1000, h = 560, rand = seeded(seed);
  const limbs = [], tips = [];

  /* One limb: a quadratic arc whose control point is nudged off the straight
     line, so nothing in the tree is ever a ruler-straight stick. */
  const grow = (x, y, ang, len, wid, depth) => {
    const bend = (rand() - 0.5) * 0.5;
    const x2 = x + Math.cos(ang) * len, y2 = y + Math.sin(ang) * len;
    const mx = x + Math.cos(ang + bend) * len * 0.55;
    const my = y + Math.sin(ang + bend) * len * 0.55;
    limbs.push({ d: `M${x.toFixed(1)} ${y.toFixed(1)}Q${mx.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`, w: wid, depth });

    if (depth >= 6 || len < 15) { tips.push([x2, y2, depth]); return; }
    /* Blossoms cling to the outer half of the older wood too, not only tips. */
    if (depth >= 3) tips.push([mx, my, depth]);

    const kids = depth < 2 ? 2 : rand() < 0.32 ? 3 : 2;
    for (let i = 0; i < kids; i += 1) {
      const spread = 0.30 + rand() * 0.42;
      const dir = i === 0 ? -spread : i === 1 ? spread * 0.85 : spread * 0.15;
      grow(x2, y2, ang + dir, len * (0.68 + rand() * 0.14), wid * 0.66, depth + 1);
    }
  };
  grow(196, 548, -Math.PI / 2 + 0.30, 132, 30, 0);
  grow(210, 552, -Math.PI / 2 + 0.62, 74, 16, 1);   // a low secondary trunk

  /* Five petals on a ring, a gold heart, a few stamens — drawn ONCE at unit
     radius and instanced with <use>. Spelling every bloom out in full costs
     twelve circles apiece and ran the finished tree past 170 KB; one definition
     plus a transform per bloom draws the identical picture for a fraction of
     that. Petals take currentColor, so the same definition serves both
     colourways and only the `color` on each <use> changes.

     The definition id is namespaced per tree: both colourways live on the
     review page at once, and `url(#id)`/`href="#id"` resolve to whichever
     comes first in the document. */
  const defId = `bloom-${key}`;
  const bloomDef = () => {
    let s = `<g id="${defId}">`;
    for (let i = 0; i < 5; i += 1) {
      const a = (i * 72 * Math.PI) / 180;
      s += `<circle cx="${(Math.cos(a) * 0.62).toFixed(3)}" cy="${(Math.sin(a) * 0.62).toFixed(3)}" r=".52" fill="currentColor"/>`;
    }
    for (let i = 0; i < 6; i += 1) {
      const a = ((i * 60 + 18) * Math.PI) / 180;
      s += `<circle cx="${(Math.cos(a) * 0.44).toFixed(3)}" cy="${(Math.sin(a) * 0.44).toFixed(3)}" r=".1" fill="${heart}"/>`;
    }
    return s + `<circle r=".2" fill="${heart}"/></g>`;
  };

  const bloom = (x, y, r, fill, rot) =>
    `<use href="#${defId}" color="${fill}" transform="translate(${x.toFixed(1)} ${y.toFixed(1)})` +
    ` rotate(${rot.toFixed(0)}) scale(${r.toFixed(2)})"/>`;

  const blooms = tips.map(([x, y, d]) => {
    if (rand() < 0.12) return "";
    const r = (d >= 5 ? 8.5 : d >= 4 ? 10.5 : 12.5) * (0.78 + rand() * 0.5);
    if (rand() < 0.2) {                                   // a closed bud
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r * 0.42).toFixed(1)}" fill="${petalAlt}"/>`;
    }
    return bloom(x, y, r, rand() < 0.34 ? petalAlt : petal, rand() * 72);
  }).join("");

  const falling = Array.from({ length: 26 }, () => {
    const x = 120 + rand() * 860, y = 120 + rand() * 420;
    return `<ellipse cx="${x.toFixed(0)}" cy="${y.toFixed(0)}" rx="${(3 + rand() * 3).toFixed(1)}" ry="${(2 + rand() * 2).toFixed(1)}" transform="rotate(${(rand() * 360).toFixed(0)} ${x.toFixed(0)} ${y.toFixed(0)})" fill="${petalAlt}" opacity="${(0.35 + rand() * 0.45).toFixed(2)}"/>`;
  }).join("");

  const bark2 = limbs.sort((a, b) => b.w - a.w)
    .map((l) => `<path d="${l.d}" fill="none" stroke="${bark}" stroke-width="${l.w.toFixed(1)}" stroke-linecap="round"/>`).join("");

  return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true" focusable="false">
    <defs>${bloomDef()}</defs>
    <rect width="${w}" height="${h}" fill="${sky}"/>
    ${moon ? `<circle cx="770" cy="150" r="96" fill="${moon}" opacity=".95"/>` : ""}
    <path d="M0 548 H1000 V560 H0 Z" fill="${bark}" opacity=".85"/>
    ${bark2}${falling}${blooms}</svg>`;
}

/* ===========================================================================
   Render
   =========================================================================== */
/* Built fresh per grid rather than once and reused: svgOf mints a new mask id
   on every call, and sharing one string across three grids would put the same
   id on all three copies. Identical definitions render fine, but it reinstates
   exactly the footgun the unique ids exist to remove. */
const cells = (list) => list.map((a) =>
  `<div class="cell">${svgOf(a)}<div class="nm">${a.label}</div><div class="yr">${a.year}</div></div>`).join("");
const ALTS = [];
const addAlt = (id, label, note, parts) => ALTS.push({ id, label, year: note, parts });

addAlt("rooster-front", "Rooster", "front on",
  pth("M36 36 C28 16 42 4 50 18 C53 2 69 2 71 18 C80 4 93 16 85 36" +
      "C77 28 68 30 63 38 C53 30 44 28 36 36 Z") +
  ell(60, 68, 25, 28) +
  cir(48, 92, 8) + cir(72, 92, 8) +
  pth("M53 80 L67 80 L60 100 Z") +
  bar("M53 80 L67 80", 3, B) +
  cir(48, 62, 5.5, B) + cir(72, 62, 5.5, B));

addAlt("rooster-side", "Rooster", "profile", ANIMALS.find((a) => a.id === "rooster").parts);

OUT["grid-alt"] = cells(ALTS);

["grid-a", "grid-b", "grid-c"].forEach((id) => { OUT[id] = cells(ANIMALS); });
OUT["lanterns"] = ANIMALS.map((a) =>
  `<div class="lantern"><div class="cap"></div><div class="body">${svgOf(a)}</div>` +
  `<div class="tassel"></div><div class="nm">${a.label}</div></div>`).join("");

const LNY = [
  ["2027-02-06", "Saturday",  "Goat",    "goat"],
  ["2028-01-26", "Wednesday", "Monkey",  "monkey"],
  ["2029-02-13", "Tuesday",   "Rooster", "rooster"],
  ["2030-02-03", "Sunday",    "Dog",     "dog"],
  ["2031-01-23", "Thursday",  "Pig",     "pig"],
  ["2032-02-11", "Wednesday", "Rat",     "rat"],
  ["2033-01-31", "Monday",    "Ox",      "ox"],
  ["2034-02-19", "Sunday",    "Tiger",   "tiger"],
  ["2035-02-08", "Thursday",  "Rabbit",  "rabbit"],
  ["2036-01-28", "Monday",    "Dragon",  "dragon"]
];
const fmt = (iso) => new Date(iso + "T12:00:00").toLocaleDateString("en-GB",
  { day: "numeric", month: "long" });
OUT["dates"] =
  "<tr><th>Year</th><th>New Year's Day</th><th>Day</th><th>Animal</th><th>Window</th></tr>" +
  LNY.map(([iso, day, animal]) => {
    const d = new Date(iso + "T12:00:00");
    const e = new Date(d); e.setDate(d.getDate() - 1);
    const a = new Date(d); a.setDate(d.getDate() + 1);
    return `<tr><td>${iso.slice(0, 4)}</td><td>${fmt(iso)}</td><td class="d">${day}</td>` +
      `<td>${animal}</td><td class="d">${fmt(e.toISOString().slice(0, 10))} &ndash; ` +
      `${fmt(a.toISOString().slice(0, 10))}</td></tr>`;
  }).join("");

OUT["code"] = `<span class="c">/* static/js/seasonal/season.js */</span>

<span class="c">/* Lunar New Year moves against the Gregorian calendar, so unlike every other
   season it cannot be read off a month. The dates are simply listed, and the
   layer is given the eve, the day and the day after. Past 2036 the table runs
   out and the function falls through — which is correct: better no season than
   a wrong one. */</span>
const LUNAR_NEW_YEAR = {
${LNY.map(([iso, , , id]) => `  "${iso}": "${id}"`).join(",\n")}
};

const LUNAR_WINDOW = new Map();
for (const [iso, animal] of Object.entries(LUNAR_NEW_YEAR)) {
  const day = new Date(\`\${iso}T00:00:00\`);
  for (let offset = -1; offset <= 1; offset += 1) {
    const d = new Date(day);
    d.setDate(day.getDate() + offset);
    LUNAR_WINDOW.set(d.toDateString(), animal);
  }
}

<span class="c">/* Called first in seasonForDate, ahead of the month checks: late January and
   February are otherwise unclaimed, but 2028 and 2031 fall inside Winter's
   1 Dec – 5 Jan run only if that range is ever widened. Checking first keeps
   the precedence explicit rather than incidental. */</span>
export const lunarAnimalForDate = (date) =&gt; LUNAR_WINDOW.get(date.toDateString()) ?? null;`;

OUT["tree-night"] = tree({
  key: "night", sky: "#150a12", bark: "#2c1620", petal: "#ffd7e2", petalAlt: "#ff9fbb",
  heart: "#ffd98a", moon: "#f6e3c4", seed: 7 });
OUT["tree-gold"] = tree({
  key: "gold", sky: "#8c0a1c", bark: "#4a0710", petal: "#ffd98a", petalAlt: "#e8b64c",
  heart: "#fff3d4", moon: "#c8102e", seed: 7 });

/* ---------------------------------------------------------------------------
   Emit
   -------------------------------------------------------------------------- */

/* Every placeholder in the shell is an empty element carrying an id. */
let html = SHELL;
for (const [id, content] of Object.entries(OUT)) {
  const at = new RegExp(`(id="${id}"[^>]*>)`);
  if (!at.test(html)) throw new Error(`no placeholder for "${id}" in the shell`);
  html = html.replace(at, `$1\n${content}\n`);
}
fs.writeFileSync(path.join(DIR, "lunar-art.html"), html + "\n");

/* Standalone files get a literal fill rather than currentColor, so they show
   something when opened on their own. The copies that go into the site keep currentColor. */
const RED = "#c8102e";
const standalone = (a) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="240" height="240"\n` +
  `     role="img" aria-label="${a.label}">\n` +
  `  <mask id="lny-${a.id}" maskUnits="userSpaceOnUse" x="0" y="0" width="120" height="120">\n` +
  `    <rect width="120" height="120" fill="#000"/>\n    ${a.parts.trim()}\n  </mask>\n` +
  `  <rect width="120" height="120" fill="${RED}" mask="url(#lny-${a.id})"/>\n</svg>\n`;

const svgDir = path.join(DIR, "svg");
fs.mkdirSync(svgDir, { recursive: true });
const written = [];
for (const a of [...ANIMALS, ...ALTS]) {
  const name = `${a.id}.svg`;
  fs.writeFileSync(path.join(svgDir, name), standalone(a));
  written.push(name);
}
for (const [name, key] of [["blossom-night.svg", "tree-night"], ["blossom-gold.svg", "tree-gold"]]) {
  fs.writeFileSync(path.join(svgDir, name),
    OUT[key].replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ') + "\n");
  written.push(name);
}
console.log(`lunar-art.html + svg/{${written.join(", ")}}`);
