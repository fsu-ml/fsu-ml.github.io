/**
 * Día de Muertos art — review sheet.
 *
 * The drawings live in static/js/seasonal/muertos.js; this pulls the ofrenda,
 * the garden tile, the four sugar skulls and the folk-art spray out of that
 * module and lays them on the theme's night at inspection sizes, so they can
 * be judged on their own. Run it after changing the art:
 *
 *   node planning/build-muertos-art.mjs
 *
 * On the site: `/?season=muertos`.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(DIR, "../static/js/seasonal/muertos.js"), "utf8");

/* Everything between the colour constants and the ambient petals is plain
   drawing code with no DOM in it, so it can be evaluated as-is; only the
   engine's `seeded`, `range` and `pick` have to be supplied. */
const from = src.indexOf("const ORANGE =");
const to = src.indexOf("/* ---------------------------------------------------------------------------\n   Ambient petals");
/* The engine's own generator, lifted from its source so the sheet draws the
   exact scene the site does. */
const engine = fs.readFileSync(path.join(DIR, "../static/js/seasonal/engine.js"), "utf8");
const lift = (name) => {
  const i = engine.indexOf(`export const ${name} =`);
  return engine.slice(i + 7, engine.indexOf("\n};\n", i) + 3);
};
const seeded = `${lift("seeded")}
const range = (rand, min, max) => min + rand() * (max - min);
const pick = (rand, list) => list[Math.min(list.length - 1, Math.floor(rand() * list.length))];`;
const api = new Function(
  `${seeded}\n${src.slice(from, to)}\nreturn { ofrendaSvg, gardenSvg, calaveraSvg, FOLK_SPRAY, SKULL_STYLES };`
)();

const skulls = (w) =>
  Object.keys(api.SKULL_STYLES)
    .map((k) => `<figure><div style="width:${w}px">${api.calaveraSvg(k)}</div><figcaption>${k}</figcaption></figure>`)
    .join("");

const html = `<meta charset="utf-8"><title>Día de Muertos art</title>
<style>
  body{margin:0;min-height:100vh;background:linear-gradient(180deg,#1a1013,#3a1640 55%,#4a1c27);font:14px system-ui;color:#f7f9fc;padding:30px}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .row{display:flex;gap:40px;align-items:flex-end;flex-wrap:wrap;margin-bottom:40px}
  figure{margin:0}figcaption{opacity:.7;margin-top:8px;text-align:center}
  p{opacity:.7;margin:0 0 8px}
  .garden{height:152px;line-height:0;margin-bottom:40px}.garden svg{height:100%}
  .edge{position:relative;height:200px;margin:0 -30px 40px;background:linear-gradient(#4a1c27 0 120px,#f7f3ee 120px)}
  .edge .garden{position:absolute;left:0;right:0;top:-32px;margin:0;height:152px}
  .garden.big{height:304px;overflow:hidden}.garden.big svg{width:200%;height:200%;transform-origin:0 0}
</style>
<p>The garden over the hero's edge, as on the site: night above, the section's cream below</p>
<div class="edge"><div class="garden">${api.gardenSvg(8, "g0")}</div></div>
<p>Ofrenda at 560px (2×) and at site size (280px)</p>
<div class="row"><div style="width:560px">${api.ofrendaSvg(5)}</div><div style="width:280px">${api.ofrendaSvg(5)}</div></div>
<p>Sugar skulls at 200px</p>
<div class="row">${skulls(200)}</div>
<p>…and at 34px, the size they are on the altar</p>
<div class="row">${skulls(34)}</div>
<p>Garden, site size</p>
<div class="garden">${api.gardenSvg(8, "g1")}</div>
<p>Garden at 2×</p>
<div class="garden big">${api.gardenSvg(8, "g2")}</div>
<p>Folk-art spray at 480px</p>
<div class="row"><div style="width:480px">${api.FOLK_SPRAY}</div></div>
`;

fs.writeFileSync(path.join(DIR, "muertos-art.html"), html);
console.log("wrote planning/muertos-art.html");
