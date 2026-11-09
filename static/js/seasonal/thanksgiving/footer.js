/**
 * Footer: the harvest table.
 *
 * The table after dinner, under a night sky: plank top, a garnet and gold
 * runner, pumpkins, a steaming pie, a cornucopia spilling over, and candles
 * whose light flickers on the wood. Candle flames lean away from the pointer
 * and with the wind; a pumpkin wobbles when touched; the pie puffs steam.
 *
 * Adapted from the sc-ai Thanksgiving sheet's footer
 * (HolidayThemes/collections/sc-ai/themes/thanksgiving, 13-footer).
 *
 * The link columns are never touched. The table reserves its band through
 * `.footer-inner` padding (see thanksgiving-footer.css), so a stacked column
 * on a phone never lands in it.
 */

import { decorate, seeded } from "../engine.js";
import { LEAF, LEAF_COLORS, LEAF_KINDS, f1, pickOf } from "./art.js";

const NS = "http://www.w3.org/2000/svg";

const starfield = (seed, n) => {
  const rand = seeded(seed);
  return Array.from(
    { length: n },
    () =>
      `<span class="tg-star" style="left:${(rand() * 100).toFixed(1)}%;top:${(rand() * 52).toFixed(1)}%;--delay:${(
        -rand() * 3
      ).toFixed(1)}s"></span>`
  ).join("");
};

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
           <path d="M0 6 L7 0 L9 3 L11 0 L18 6" fill="none" stroke="#4a1c27" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
         </g>`
    )
    .join("")}</svg></div>`;
})();

/* A leaf lying on the table at (x, y), in the shared 100-unit box. */
const svgLeaf = (x, y, size, rot, kind, color, squash = 1) =>
  `<g transform="translate(${f1(x)} ${f1(y)}) scale(1 ${squash.toFixed(2)}) rotate(${Math.round(rot)})">
     <path d="${LEAF[kind]}" transform="scale(${(size / 100).toFixed(3)}) translate(-50 -52)" fill="${color}"/>
   </g>`;

/* Re-runs `build` whenever the footer's width changes by more than a few
   pixels, so artwork laid out in real pixels never stretches. */
const onWidth = (disposer, el, build) => {
  let last = 0;
  const run = () => {
    const w = Math.round(el.getBoundingClientRect().width);
    if (w < 50 || Math.abs(w - last) < 8) {
      return;
    }
    last = w;
    build(w);
  };
  const ro = new ResizeObserver(run);
  ro.observe(el);
  disposer.add(() => ro.disconnect());
  run();
};

const TABLE_H = 204;
const TOP = 112;
const FRONT = 166;

const pumpkin = (x, y, r, body, rib, stem = "#4a3a1a") => {
  const ribs = [-0.62, -0.3, 0, 0.3, 0.62]
    .map(
      (k) =>
        `<ellipse cx="${f1(k * r)}" cy="${f1(-r * 0.78)}" rx="${f1(r * (0.52 - Math.abs(k) * 0.18))}" ry="${f1(r * 0.78)}" fill="${
          Math.abs(k) > 0.5 ? rib : body
        }"/>`
    )
    .join("");
  return `<g class="tg-pumpkin" transform="translate(${f1(x)} ${f1(y)})">
    <ellipse cx="0" cy="1" rx="${f1(r * 1.05)}" ry="${f1(r * 0.22)}" fill="rgba(0,0,0,.35)"/>
    <g class="tg-wobble">${ribs}
      <path d="M${f1(-r * 0.25)} ${f1(-r * 1.2)}C${f1(-r * 0.2)} ${f1(-r * 1.25)} ${f1(-r * 0.05)} ${f1(-r * 1.55)} ${f1(r * 0.16)} ${f1(-r * 1.68)}"
            stroke="${stem}" stroke-width="${f1(Math.max(2.4, r * 0.16))}" fill="none" stroke-linecap="round"/>
      <ellipse cx="${f1(-r * 0.3)}" cy="${f1(-r * 1.0)}" rx="${f1(r * 0.18)}" ry="${f1(r * 0.36)}" fill="rgba(255,230,180,.18)"/>
    </g></g>`;
};

const pie = (x, y, r) => {
  const strips = [-0.6, -0.2, 0.2, 0.6]
    .map(
      (k) =>
        `<path d="M${f1(k * r - r * 0.12)} ${f1(-r * 0.62)}L${f1(k * r + r * 0.12)} ${f1(r * 0.02)}" stroke="#e0a85a" stroke-width="${f1(
          r * 0.09
        )}" stroke-linecap="round"/>
         <path d="M${f1(-r * 0.92)} ${f1(-r * 0.3 + k * r * 0.26)}L${f1(r * 0.92)} ${f1(-r * 0.3 + k * r * 0.26)}" stroke="#d8984a" stroke-width="${f1(
           r * 0.08
         )}" stroke-linecap="round"/>`
    )
    .join("");
  const wisps = [0, 1, 2]
    .map(
      (i) =>
        `<path class="tg-steam" style="--d:${i * 0.9}s" d="M${f1((i - 1) * r * 0.36)} ${f1(-r * 0.5)}c-6 -10 6 -18 0 -28s6 -18 0 -26"/>`
    )
    .join("");
  return `<g class="tg-pie" transform="translate(${f1(x)} ${f1(y)})">
    <ellipse cx="0" cy="2" rx="${f1(r * 1.16)}" ry="${f1(r * 0.36)}" fill="rgba(0,0,0,.35)"/>
    <ellipse cx="0" cy="${f1(-r * 0.16)}" rx="${f1(r * 1.12)}" ry="${f1(r * 0.42)}" fill="#e9e1cf"/>
    <ellipse cx="0" cy="${f1(-r * 0.3)}" rx="${f1(r)}" ry="${f1(r * 0.36)}" fill="#a8582a"/>
    <clipPath id="tg-pie-clip"><ellipse cx="0" cy="${f1(-r * 0.3)}" rx="${f1(r * 0.92)}" ry="${f1(r * 0.31)}"/></clipPath>
    <g clip-path="url(#tg-pie-clip)" fill="none">${strips}</g>
    <ellipse cx="0" cy="${f1(-r * 0.3)}" rx="${f1(r)}" ry="${f1(r * 0.36)}" fill="none" stroke="#d99a52" stroke-width="${f1(
      r * 0.12
    )}" stroke-dasharray="${f1(r * 0.1)} ${f1(r * 0.06)}"/>
    <g class="tg-steams" fill="none" stroke="rgba(255,240,220,.5)" stroke-width="2" stroke-linecap="round">${wisps}</g>
  </g>`;
};

const candle = (x, y, h, i) => `<g class="tg-candle" data-i="${i}" transform="translate(${f1(x)} ${f1(y)})">
    <circle class="tg-candle-glow" cx="0" cy="${f1(-h - 30)}" r="150" fill="url(#tg-candle-light)"/>
    <ellipse cx="0" cy="1" rx="16" ry="4" fill="rgba(0,0,0,.4)"/>
    <ellipse cx="0" cy="-2" rx="14" ry="4.4" fill="#a07a34"/>
    <path d="M-4 -3C-4 -10 -2 -12 -2 -14H2C2 -12 4 -10 4 -3Z" fill="#b48a3c"/>
    <ellipse cx="0" cy="-15" rx="9" ry="3" fill="#c9a04a"/>
    <rect x="-5.5" y="${f1(-15 - h)}" width="11" height="${f1(h)}" rx="2" fill="#efe4cc"/>
    <rect x="-5.5" y="${f1(-15 - h)}" width="4" height="${f1(h)}" rx="2" fill="rgba(255,255,255,.35)"/>
    <path d="M3 ${f1(-15 - h)}v${f1(h * 0.28)}c0 3 2.6 3 2.6 0V${f1(-15 - h + 2)}" fill="#e4d6b8"/>
    <path d="M0 ${f1(-15 - h)}v-4" stroke="#2a1a10" stroke-width="1.2"/>
    <g class="tg-flame">
      <path d="M0 ${f1(-19 - h)}C-6 ${f1(-24 - h)} -3 ${f1(-33 - h)} 0 ${f1(-40 - h)}C3 ${f1(-33 - h)} 6 ${f1(-24 - h)} 0 ${f1(-19 - h)}Z" fill="#f6b34a"/>
      <path d="M0 ${f1(-20 - h)}C-3 ${f1(-24 - h)} -1.5 ${f1(-29 - h)} 0 ${f1(-33 - h)}C1.5 ${f1(-29 - h)} 3 ${f1(-24 - h)} 0 ${f1(-20 - h)}Z" fill="#fff2c8"/>
    </g>
    <rect x="-24" y="${f1(-46 - h)}" width="48" height="${f1(h + 50)}" fill="transparent" class="tg-hit"/>
  </g>`;

const cornucopia = (x, y, s) => {
  let weave = "";
  for (let i = 0; i < 6; i += 1) {
    const k = i / 6;
    weave += `<path d="M${f1(-70 + k * 58)} ${f1(-8 - k * 22)}q${f1(4 + k * 6)} ${f1(10 + k * 14)} 0 ${f1(20 + k * 22)}" />`;
  }
  return `<g transform="translate(${f1(x)} ${f1(y)}) scale(${s.toFixed(2)})">
    <ellipse cx="-14" cy="2" rx="70" ry="7" fill="rgba(0,0,0,.35)"/>
    <path d="M-96 -34C-92 -40 -84 -34 -80 -26C-66 -6 -34 -40 2 -46C10 -30 12 -8 2 4C-30 0 -60 4 -78 -10C-86 -16 -90 -26 -96 -34Z" fill="#a8743a"/>
    <g stroke="rgba(70,40,15,.55)" stroke-width="2.2" fill="none">${weave}</g>
    <ellipse cx="3" cy="-21" rx="9" ry="25" fill="#5a3818" transform="rotate(-8 3 -21)"/>
    <circle cx="18" cy="-8" r="10" fill="#b0283a"/><circle cx="15" cy="-11" r="3" fill="rgba(255,255,255,.25)"/>
    <path d="M18 -18q2 -4 5 -5" stroke="#4a2a10" stroke-width="1.6" fill="none"/>
    <g fill="#5b2a5a">${[
      [30, -4],
      [36, -2],
      [33, 3],
      [27, 2],
      [39, 3],
      [35, 8],
      [30, 8]
    ]
      .map(([cx, cy]) => `<circle cx="${cx}" cy="${cy}" r="4"/>`)
      .join("")}</g>
    <ellipse cx="6" cy="-34" rx="9" ry="7" fill="#e0a83c"/><path d="M14 -36q10 -2 16 4" stroke="#c98a2a" stroke-width="5" stroke-linecap="round" fill="none"/>
    <ellipse cx="52" cy="-2" rx="11" ry="8" fill="#d9802f"/><path d="M52 -10v-4" stroke="#4a3a1a" stroke-width="2"/>
    <path d="M-2 -46q20 -22 34 -26M4 -44q16 -26 30 -34" stroke="#d6ae5a" stroke-width="1.6" fill="none"/>
    <ellipse cx="32" cy="-72" rx="2.4" ry="6" fill="#d6ae5a" transform="rotate(60 32 -72)"/>
    <ellipse cx="34" cy="-78" rx="2.4" ry="6" fill="#d6ae5a" transform="rotate(50 34 -78)"/>
  </g>`;
};

const buildTable = (W) => {
  const r = seeded(23);
  const narrow = W < 680;
  const at = (p) => p * W;
  let planks = "";
  for (const y of [TOP + 14, TOP + 30]) {
    planks += `<path d="M0 ${y}H${W}" stroke="rgba(20,8,4,.45)" stroke-width="1.2"/>`;
  }
  let grain = "";
  for (let i = 0; i < Math.round(W / 60); i += 1) {
    const x = r() * W;
    const y = TOP + 4 + r() * 42;
    const len = 60 + r() * 160;
    grain += `<path d="M${f1(x)} ${f1(y)}q${f1(len / 2)} ${f1((r() - 0.5) * 4)} ${f1(len)} 0" />`;
  }
  /* Candles, the pie and the produce, placed by fraction of the width. */
  const items = narrow
    ? [
        cornucopia(at(0.2), TOP + 30, 0.7),
        pumpkin(at(0.42), TOP + 32, 20, "#d9802f", "#b8601f"),
        candle(at(0.56), TOP + 28, 38, 0),
        pie(at(0.72), TOP + 34, 28),
        candle(at(0.9), TOP + 26, 52, 1)
      ]
    : [
        cornucopia(at(0.1), TOP + 34, 1.3),
        pumpkin(at(0.235), TOP + 38, 38, "#d9802f", "#b8601f"),
        pumpkin(at(0.29), TOP + 26, 21, "#ece2cc", "#cfc2a4", "#5a6a3a"),
        candle(at(0.36), TOP + 24, 76, 0),
        candle(at(0.395), TOP + 32, 48, 1),
        pie(at(0.5), TOP + 38, 50),
        candle(at(0.6), TOP + 26, 62, 2),
        pumpkin(at(0.69), TOP + 30, 18, "#5d6b3a", "#4a5630", "#3a2a12"),
        pumpkin(at(0.745), TOP + 40, 30, "#c8641f", "#a84e18"),
        candle(at(0.85), TOP + 28, 56, 3)
      ];
  let litter = "";
  for (let i = 0; i < Math.round(W / 70); i += 1) {
    litter += svgLeaf(r() * W, TOP + 10 + r() * 40, 14 + r() * 10, r() * 360, pickOf(r, LEAF_KINDS), pickOf(r, LEAF_COLORS), 0.45 + r() * 0.2);
  }
  for (let i = 0; i < Math.round(W / 120); i += 1) {
    litter += `<circle cx="${f1(r() * W)}" cy="${f1(TOP + 20 + r() * 26)}" r="${f1(2.4 + r())}" fill="#8e1f31"/>`;
  }
  return `<defs>
      <linearGradient id="tg-wood" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#6b3f24"/><stop offset="1" stop-color="#3e2215"/></linearGradient>
      <radialGradient id="tg-candle-light"><stop offset="0" stop-color="rgba(255,190,100,.55)"/><stop offset=".4" stop-color="rgba(255,160,70,.18)"/><stop offset="1" stop-color="rgba(255,160,70,0)"/></radialGradient>
    </defs>
    <path d="M0 ${TOP}H${W}V${FRONT}H0Z" fill="url(#tg-wood)"/>
    ${planks}
    <g stroke="rgba(255,210,160,.06)" stroke-width="1" fill="none">${grain}</g>
    <path d="M0 ${TOP + 8}H${W}V${TOP + 40}H0Z" fill="#6e1f2c"/>
    <path d="M0 ${TOP + 12}H${W}M0 ${TOP + 36}H${W}" stroke="#ceb888" stroke-width="1.6" opacity=".8"/>
    <path d="M0 ${TOP + 8}H${W}" stroke="rgba(0,0,0,.3)"/>
    <path d="M0 ${FRONT}H${W}V${TABLE_H}H0Z" fill="#2a150c"/>
    <path d="M0 ${FRONT}H${W}" stroke="#8a5632" stroke-width="1.5"/>
    <g>${litter}</g>
    <g class="tg-table-items">${items.join("")}</g>`;
};

export const mountFooter = ({ disposer, motion, pointer, rect }) => {
  const [scene] = decorate(
    disposer,
    ".site-footer",
    "season-scene tg-footer tg-footer-table",
    `<div class="season-sky"></div>${starfield(63, 22)}${GEESE}`
  );
  if (!scene) {
    return null;
  }
  const footer = scene.parentElement;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "tg-table");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  footer.appendChild(svg);
  disposer.node(svg);

  let candles = [];
  onWidth(disposer, footer, (W) => {
    svg.setAttribute("viewBox", `0 0 ${W} ${TABLE_H}`);
    svg.innerHTML = buildTable(W);
    candles = Array.from(svg.querySelectorAll(".tg-candle")).map((g) => ({
      g,
      flame: g.querySelector(".tg-flame"),
      x: Number(/translate\(([\d.]+)/.exec(g.getAttribute("transform"))[1]),
      lean: 0
    }));
  });

  if (!motion) {
    return null;
  }

  const onEnter = (event) => {
    const pumpkinEl = event.target.closest?.(".tg-pumpkin");
    if (pumpkinEl && !pumpkinEl.classList.contains("is-wobbling")) {
      pumpkinEl.classList.add("is-wobbling");
      window.setTimeout(() => pumpkinEl.classList.remove("is-wobbling"), 900);
    }
    const pieEl = event.target.closest?.(".tg-pie");
    if (pieEl && !pieEl.classList.contains("is-puffing")) {
      pieEl.classList.add("is-puffing");
      window.setTimeout(() => pieEl.classList.remove("is-puffing"), 2600);
    }
  };
  svg.addEventListener("pointerover", onEnter);
  disposer.add(() => svg.removeEventListener("pointerover", onEnter));

  /* Flames lean away from a nearby pointer. Written as one custom property
     per candle, only when it changes by a visible amount. */
  const box = rect(footer);
  return {
    visible: true,
    el: footer,
    step(dt) {
      const px = pointer.active ? pointer.x - box.left : -9999;
      const py = pointer.active ? pointer.y - (box.bottom - TABLE_H) : -9999;
      for (const c of candles) {
        const dx = c.x - px;
        const near = Math.abs(dx) < 140 && py > -120 && py < TABLE_H;
        const target = near ? Math.sign(dx) * (1 - Math.abs(dx) / 140) * 26 : 0;
        const next = c.lean + (target - c.lean) * Math.min(1, dt * 7);
        if (Math.abs(next - c.lean) > 0.2 || (target === 0 && Math.abs(next) > 0.2)) {
          c.lean = next;
          c.flame.style.setProperty("--lean", `${next.toFixed(1)}deg`);
        }
        c.g.classList.toggle("is-near", near && Math.abs(dx) < 60);
      }
    },
    draw() {}
  };
};
