/**
 * Hero: misty ridges.
 *
 * Five ridges stepping back into a low autumn sun, each one hazier and
 * warmer than the one in front, with round trees in rust, ochre and gold
 * along the crests (a few already bare). Bands of fog drift between the
 * layers and are shoved along by gusts, slow rays turn behind the sun, and
 * the layers shift with the pointer for depth.
 *
 * Adapted from the theme library's "Misty morning hills" (fog-hills), moved
 * from morning to late afternoon so the copy keeps its contrast: the ridges
 * stay low on the copy's side and rise toward the seminar card.
 *
 * Built from SVG and CSS. The only per-frame work is two custom properties
 * for the parallax, written to the scene element.
 */

import { decorate, seeded, smoothstep } from "../engine.js";
import { f1 } from "./art.js";

const VW = 1600;
const VH = 420;

/* far -> near: [left y, right y, amplitude, fill, tree size, depth] */
const LAYERS = [
  [262, 120, 16, "#b2602e", 6, 4],
  [296, 176, 18, "#8c3e22", 9, 8],
  [328, 232, 16, "#6a2a1a", 13, 13],
  [356, 288, 14, "#431711", 18, 20],
  [388, 352, 10, "#200908", 25, 30]
];

const TREE_TINTS = [
  [200, 98, 40],
  [214, 150, 64],
  [168, 64, 36],
  [142, 31, 49],
  [224, 168, 60]
];

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const tint = (base, add, k) => `rgb(${base.map((v, i) => Math.round(v + (add[i] - v) * k)).join(",")})`;

const ridgeY = (layer, x, seed) => {
  const [yl, yr, amp] = layer;
  const u = x / VW;
  const lift = smoothstep((u - 0.3) / 0.7);
  return (
    yl +
    (yr - yl) * lift +
    amp * Math.sin(u * 7.1 + seed) +
    amp * 0.6 * Math.sin(u * 17.3 + seed * 2.3) +
    amp * 0.3 * Math.sin(u * 37 + seed * 0.7)
  );
};

const treeAt = (x, y, size, color, bare, rand) => {
  const trunk = size * 0.55;
  if (bare) {
    const s = size / 20;
    return `<path d="M${f1(x)} ${f1(y)}v${f1(-trunk * 1.6)}m0 ${f1(trunk * 0.6)}l${f1(-6 * s)} ${f1(-7 * s)}m${f1(6 * s)} ${f1(3 * s)}l${f1(7 * s)} ${f1(-9 * s)}"
              stroke="${color}" stroke-width="${f1(Math.max(1, size / 9))}" stroke-linecap="round" fill="none"/>`;
  }
  const w = size * (0.7 + rand() * 0.35);
  return `<rect x="${f1(x - size / 14)}" y="${f1(y - trunk)}" width="${f1(Math.max(1, size / 7))}" height="${f1(trunk + 2)}" fill="#1a0806"/>
          <ellipse cx="${f1(x)}" cy="${f1(y - trunk - w * 0.55)}" rx="${f1(w * 0.62)}" ry="${f1(w * 0.72)}" fill="${color}"/>`;
};

const layerSvg = (layer, i) => {
  const rand = seeded(90 + i);
  const seed = 1.3 + i * 2.1;
  let d = `M0 ${VH}`;
  for (let x = 0; x <= VW; x += 16) {
    d += `L${x} ${f1(ridgeY(layer, x, seed))}`;
  }
  d += `L${VW} ${VH}Z`;
  let trees = "";
  if (i > 0) {
    const base = hex(layer[3]);
    for (let x = rand() * 40; x < VW; x += 18 + rand() * (60 - i * 6)) {
      if (rand() < 0.25) {
        continue;
      }
      const size = layer[4] * (0.7 + rand() * 0.6);
      const bare = rand() < 0.14;
      const k = 0.22 + i * 0.08;
      const color = bare ? layer[3] : tint(base, TREE_TINTS[(rand() * TREE_TINTS.length) | 0], k);
      trees += treeAt(x, ridgeY(layer, x, seed) + 3, size, color, bare, rand);
    }
  }
  return `<svg viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="xMidYMax slice" aria-hidden="true" focusable="false">
            <g>${trees}</g><path d="${d}" fill="${layer[3]}"/></svg>`;
};

const fog = (n, bottom) =>
  `<div class="tg-fog" style="--bottom:${bottom}%;--dur:${70 + n * 24}s;--alpha:${(0.2 - n * 0.03).toFixed(2)};--h:${60 + n * 20}px"></div>`;

/* The V of geese from the original hero, reused as-is. */
const GEESE = (() => {
  const offsets = [
    [0, 0],
    [-16, 9],
    [16, 9],
    [-32, 18],
    [32, 18],
    [-48, 27],
    [48, 27]
  ];
  return `<div class="tg-geese"><svg viewBox="0 0 180 40" aria-hidden="true" focusable="false">${offsets
    .map(
      ([dx, dy], i) =>
        `<g class="tg-goose" style="animation-delay:${(-i * 0.07).toFixed(2)}s" transform="translate(${dx + 60} ${dy + 6})">
           <path d="M0 6 L7 0 L9 3 L11 0 L18 6" fill="none" stroke="#2a0d10" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
         </g>`
    )
    .join("")}</svg></div>`;
})();

export const mountRidgesHero = ({ disposer, pointer, rect }) => {
  const html = `
    <div class="season-sky"></div>
    <div class="tg-ridge-sun"><span class="tg-ridge-rays"></span><span class="tg-ridge-disc"></span></div>
    ${GEESE}
    ${LAYERS.map(
      (layer, i) =>
        `<div class="tg-ridge" style="--depth:${layer[5]}">${layerSvg(layer, i)}</div>${
          i < 3 ? fog(i, [36, 24, 12][i]) : ""
        }`
    ).join("")}
    <div class="tg-dusk"></div>
    <div class="tg-hero-shade"></div>`;
  const [scene] = decorate(disposer, ".hero", "season-scene tg-hero tg-hero-ridges", html);
  if (!scene) {
    return null;
  }
  const hero = scene.parentElement;
  const box = rect(hero);

  /* Put the sun in the gutter beside the seminar card, sitting on the far
     ridge. The ridge is drawn in viewBox units under `slice`, so its height
     at the sun's x has to go through the same mapping. */
  const placeSun = () => {
    const W = hero.clientWidth;
    const H = hero.clientHeight;
    const ridgeBox = scene.querySelector(".tg-ridge");
    if (!W || !ridgeBox) {
      return;
    }
    const RH = ridgeBox.clientHeight;
    const card = hero.querySelector(".next-seminar");
    const copy = hero.querySelector(".hero-copy");
    const hb = hero.getBoundingClientRect();
    const cb = card && card.getBoundingClientRect();
    const pb = copy && copy.getBoundingClientRect();
    const wide = cb && pb && cb.left > pb.left + pb.width * 0.5;
    const sunX = wide ? cb.left - hb.left - 10 : W * 0.74;
    const scale = Math.max(W / VW, RH / VH);
    const offX = (W - VW * scale) / 2;
    const vx = (sunX - offX) / scale;
    const ry = H - (VH - ridgeY(LAYERS[0], vx, 1.3)) * scale;
    scene.style.setProperty("--tg-sun-x", `${Math.round(sunX)}px`);
    scene.style.setProperty("--tg-sun-y", `${Math.round(H - ry - 14)}px`);
  };
  const ro = new ResizeObserver(placeSun);
  ro.observe(hero);
  disposer.add(() => ro.disconnect());
  placeSun();

  let sx = 0;
  let sy = 0;
  let lastX = "";
  let lastY = "";
  return {
    visible: true,
    el: hero,
    step(dt) {
      const local = box.local(pointer);
      const tx = local ? (local.x / Math.max(1, box.width)) * 2 - 1 : 0;
      const ty = local ? (local.y / Math.max(1, box.height)) * 2 - 1 : 0;
      sx += (tx - sx) * Math.min(1, dt * 2.4);
      sy += (ty - sy) * Math.min(1, dt * 2.4);
      const vx = (-sx).toFixed(3);
      const vy = (-sy).toFixed(3);
      if (vx !== lastX || vy !== lastY) {
        lastX = vx;
        lastY = vy;
        scene.style.setProperty("--tg-px", vx);
        scene.style.setProperty("--tg-py", vy);
      }
    },
    draw() {}
  };
};
