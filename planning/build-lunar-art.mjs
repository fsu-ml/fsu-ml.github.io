/**
 * Lunar New Year art — review sheet.
 *
 * Pulls the city, the eave and the sky lantern out of static/js/seasonal/
 * lunar.js and lays them on the theme's night at inspection sizes. Run it
 * after changing the art:
 *
 *   node planning/build-lunar-art.mjs
 *
 * On the site: `/?season=lunar`.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(DIR, "../static/js/seasonal/lunar.js"), "utf8");
const engine = fs.readFileSync(path.join(DIR, "../static/js/seasonal/engine.js"), "utf8");

const lift = (name) => {
  const i = engine.indexOf(`export const ${name} =`);
  return engine.slice(i + 7, engine.indexOf("\n};\n", i) + 3);
};
const slice = (from, to) => src.slice(src.indexOf(from), src.indexOf(to));

const code = [
  lift("seeded"),
  "const range = (rand, min, max) => min + rand() * (max - min);",
  "const pick = (rand, list) => list[Math.min(list.length - 1, Math.floor(rand() * list.length))];",
  slice("const RED = ", "/* `?date="),
  slice("const ROOF_W = ", "/* ---------------------------------------------------------------------------\n   The city"),
  slice("const CITY_W = ", "/* ---------------------------------------------------------------------------\n   Fireworks"),
  slice("const SKY_LANTERN =", "const skyLanternsHtml")
].join("\n");
const api = new Function(`${code}\nreturn { citySvg, roofSvg, SKY_LANTERN };`)();

const html = `<meta charset="utf-8"><title>Lunar New Year art</title>
<style>
  body{margin:0;min-height:100vh;background:linear-gradient(180deg,#160709,#4a0d18 52%,#7d0a1c);font:14px system-ui;color:#f7f9fc;padding:30px}
  svg{display:block}
  p{opacity:.7;margin:0 0 8px}
  .city{height:240px;line-height:0;margin:0 -30px 40px}.city svg{width:100%;height:100%}
  .city.big{height:480px;overflow:hidden}.city.big svg{width:200%;height:200%;transform:scale(.5);transform-origin:0 0;width:200%}
  .roof{height:64px;line-height:0;margin:0 -30px 40px}.roof svg{width:100%;height:100%}
  .lantern{width:60px;margin-bottom:40px}.lantern svg{width:100%;height:auto}
</style>
<p>The city at site size (one 1200px tile repeats)</p>
<div class="city">${api.citySvg(3, "c1")}</div>
<p>The eave at site size</p>
<div class="roof">${api.roofSvg()}</div>
<p>A sky lantern at 60px</p>
<div class="lantern">${api.SKY_LANTERN}</div>
`;

fs.writeFileSync(path.join(DIR, "lunar-art.html"), html);
console.log("wrote planning/lunar-art.html");
