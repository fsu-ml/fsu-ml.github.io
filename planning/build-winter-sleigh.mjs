/**
 * Winter sleigh art — review sheet.
 *
 * The drawing itself lives in static/js/seasonal/winter.js; this pulls the
 * sleigh and gift artwork out of that module and lays it on the theme's
 * garnet sky at three sizes, so it can be judged without waiting for the
 * night it flies on. Run it after changing the art:
 *
 *   node planning/build-winter-sleigh.mjs
 *
 * On the site the sleigh appears only on 24 and 25 December; preview it any
 * day with `/?season=winter&date=2026-12-24`.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(DIR, "../static/js/seasonal/winter.js"), "utf8");

/* Lift the definitions out by their names rather than importing the module:
   winter.js exports only `mount`, and the drawings are private to it. */
const grab = (name, end) => {
  const i = src.indexOf(name);
  return src.slice(i, src.indexOf(end, i) + end.length);
};
const code = [grab("const reindeer =", "</g>`;"), grab("const SLEIGH_ART =", "</svg>`;"), grab("const GIFT =", "</svg>`;")].join("\n");
const { SLEIGH_ART, GIFT } = new Function(`${code}\nreturn { SLEIGH_ART, GIFT };`)();

const GIFT_COLORS = ["#c1273b", "#ceb888", "#bfe0f5", "#fff1c4"];
const gifts = (size) =>
  GIFT_COLORS.map((c) => `<span class="gift" style="--gift:${c};width:${size}px">${GIFT}</span>`).join("");

const html = `<meta charset="utf-8"><title>Winter sleigh</title>
<style>
  body{margin:0;min-height:100vh;background:linear-gradient(180deg,#2a1118,#4a1c27);font:14px system-ui;color:#f7f9fc;padding:30px}
  .row{display:flex;gap:40px;align-items:flex-end;flex-wrap:wrap;margin-bottom:40px}
  .big svg{width:900px;max-width:100%;height:auto;display:block;filter:drop-shadow(0 2px 6px rgba(0,0,0,.45))}
  .site svg{width:300px;height:auto;display:block;filter:drop-shadow(0 2px 6px rgba(0,0,0,.45))}
  .phone svg{width:190px;height:auto;display:block}
  .gift{display:block;--gift:#c1273b}.gift svg{width:100%;height:auto;display:block}
  .gifts{display:flex;gap:12px;align-items:flex-end;margin-bottom:16px}
  p{opacity:.7;margin:0 0 8px}
</style>
<p>3× — for inspection</p><div class="big">${SLEIGH_ART}</div>
<p>Site size (300px) and phone size (190px)</p>
<div class="row"><div class="site">${SLEIGH_ART}</div><div class="phone">${SLEIGH_ART}</div></div>
<p>Gifts at 16px, then 3×</p>
<div class="gifts">${gifts(16)}</div>
<div class="gifts">${gifts(48)}</div>
`;

fs.writeFileSync(path.join(DIR, "winter-sleigh.html"), html);
console.log("wrote planning/winter-sleigh.html");
