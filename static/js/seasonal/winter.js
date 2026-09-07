/**
 * Winter — December.
 *
 * Snow is fixed-count DOM particles on CSS keyframes, so there is no canvas,
 * no frame loop and no scroll listener: everything that moves — flakes, the
 * light strings, the candle flames, a few city windows — is a declarative
 * animation the browser composites on its own.
 *
 * The frost corners, frost ring, snow cap, icicles and the footer treeline are
 * ported from Winter.dc.html. The skyline and the light strings are generated
 * here from a seed, so they are the same drawing on every visit.
 *
 * The theme is December's, not one holiday's: a moonlit city, snow, and
 * strings of light against the longest nights. The observances that fall in
 * the month each get a small addition to the footer on their own dates only —
 * see "The nights of December".
 */

import { Disposer, buildParticles, decorate, make, pick, range, seeded } from "./engine.js";

/* One flake per three density points, hard-capped. Forty animated elements is
   the budget for the whole layer; past that the page stops being a seminar
   site with snow on it and starts being a snow globe. */
const MAX_FLAKES = 40;
const flakeCount = (density) => Math.min(MAX_FLAKES, Math.round(density / 3));

const FLAKE_COLORS = ["#ffffff", "#f1f6fa", "#dce9f2"];

/* ---------------------------------------------------------------------------
   Artwork
   ---------------------------------------------------------------------------
   Static, author-written SVG. Assembled as markup rather than through the DOM
   builder because these are drawings, and a drawing is far easier to read and
   correct as a path than as thirty createElementNS calls.
   -------------------------------------------------------------------------- */

/* Snow cap for the brand mark: an arc following the top of the circle with
   melt drips hanging off it. The artboard's own path, drawn for a 72px mark. */
const SNOW_CAP = `
  <svg class="wn-mark-cap" viewBox="0 0 72 72" aria-hidden="true" focusable="false">
    <path fill="#f7f9fc"
          d="M0.7 22A38 38 0 0 1 71.3 22Q68 34 63 25Q58 38 52 27Q47 36 42 26Q38 42 33 27Q28 35 23 26Q18 38 13 27Q9 33 5 25Q3 30 0.7 22Z"></path>
    <path d="M16 15c6-7 14-10 22-10" stroke="rgba(255,255,255,.75)" stroke-width="1.6"
          fill="none" stroke-linecap="round"></path>
  </svg>`;

/* Frost creeping from a corner: a lattice of straight runs with small
   branchings where they cross. */
const FROST_CORNER = `
  <svg viewBox="0 0 120 120" aria-hidden="true" focusable="false">
    <g fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
      <path d="M0 30 L60 30 M0 55 L40 55 M30 0 L30 55 M55 0 L55 40 M8 8 L48 48 M15 0 L15 22 M0 15 L22 15 M70 0 L70 22 M0 70 L22 70"></path>
      <path d="M30 30 l-6 -6 M30 30 l6 -6 M30 30 l-6 6 M30 30 l6 6 M55 15 l-5 -5 M55 15 l5 -5 M15 55 l-5 -5 M15 55 l5 -5"></path>
    </g>
  </svg>`;

/* Frost ring: twenty-four radial spikes around a thin circle, alternating long
   and short. The artboard's logo treatment, used here on the portraits. */
const FROST_RING = (() => {
  const spikes = [];
  for (let i = 0; i < 24; i += 1) {
    const long = i % 2 === 0;
    spikes.push(
      `<g transform="rotate(${i * 15} 65 65)"><path d="${
        long
          ? "M65 25V10M65 18l-4-4M65 18l4-4M65 13l-2.5-2.5M65 13l2.5-2.5"
          : "M65 25V16M65 20l-3-3M65 20l3-3"
      }"/></g>`
    );
  }
  return `
    <svg class="wn-frost-ring" viewBox="0 0 130 130" aria-hidden="true" focusable="false">
      <g fill="none" stroke="#9fbfd6" stroke-width="1.2" stroke-linecap="round">
        <circle cx="65" cy="65" r="40" stroke-width="1"></circle>
        ${spikes.join("")}
      </g>
    </svg>`;
})();

const SNOWFLAKE = `
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <g stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none">
      <path d="M12 2v20M2 12h20M4.9 4.9l14.2 14.2M19.1 4.9L4.9 19.1"></path>
      <path d="M12 6l-2.4-2.4M12 6l2.4-2.4M12 18l-2.4 2.4M12 18l2.4 2.4"></path>
      <path d="M6 12l-2.4-2.4M6 12l-2.4 2.4M18 12l2.4-2.4M18 12l2.4 2.4"></path>
    </g>
  </svg>`;

/* Snow settled along a card's top edge, and the icicles hanging under it.
   Together these are the artboard's "Snow and icicles" frame. */
const CARD_SNOW = `
  <svg class="wn-card-snow" viewBox="0 0 400 30" preserveAspectRatio="none"
       aria-hidden="true" focusable="false">
    <path fill="#dce9f2"
          d="M0 0H400V12c-20 0-30 12-55 12s-30-14-60-14-35 14-70 14-35-16-70-16-30 12-55 12S25 10 0 12z"></path>
  </svg>`;

const icicles = (seed) => {
  const rand = seeded(seed);
  const W = 400;
  const n = 22;
  const w = W / n;
  let d = `M0 0H${W}V3`;
  for (let i = n; i > 0; i -= 1) {
    const x = i * w;
    const len = 8 + rand() * 26;
    d += `L${(x - w * 0.5).toFixed(1)} ${len.toFixed(1)}L${(x - w).toFixed(1)} 3`;
  }
  return `
    <svg class="wn-icicles" viewBox="0 0 ${W} 40" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <path d="${d}Z" fill="rgba(247,249,252,.96)" stroke="rgba(159,191,214,.6)"
            stroke-width=".8"></path>
    </svg>`;
};

/* ---------------------------------------------------------------------------
   Skyline
   ---------------------------------------------------------------------------
   A city along the bottom of the hero, in two layers: a paler, taller row far
   off and a darker row in front, with windows lit in the near one.

   The buildings live in an SVG <pattern> in user units rather than a scaled
   viewBox, so a building is the same width on a phone as on a monitor and the
   tile simply repeats across whatever the viewport is. Stretching a single
   drawing to the viewport turned every tower into a needle on narrow screens.
   -------------------------------------------------------------------------- */

const SKY_TILE = 960;
const SKY_H = 170;

/* One row of buildings across a tile. Returns the silhouette as a single path
   and the windows as one path per brightness bucket, so a whole city is a
   handful of nodes rather than hundreds of rects. */
const skylineLayer = (rand, { minH, maxH, windows, animate }) => {
  const bodies = [];
  const lit = { bright: [], dim: [] };
  const flicker = [];
  let x = 0;
  while (x < SKY_TILE) {
    const w = range(rand, 26, 84);
    const h = range(rand, minH, maxH);
    const top = SKY_H - h;
    const x1 = x + w;
    const roof = rand();
    let d = `M${x.toFixed(1)} ${SKY_H}V${top.toFixed(1)}`;
    if (roof < 0.22) {
      /* Antenna mast off the middle of the roof. */
      const mx = x + w / 2;
      d += `H${(mx - 1.2).toFixed(1)}V${(top - range(rand, 10, 26)).toFixed(1)}h2.4V${top.toFixed(1)}`;
    } else if (roof < 0.46) {
      /* A stepped penthouse. */
      const inset = w * range(rand, 0.18, 0.3);
      const rise = range(rand, 8, 18);
      d += `H${(x + inset).toFixed(1)}V${(top - rise).toFixed(1)}H${(x1 - inset).toFixed(1)}V${top.toFixed(1)}`;
    } else if (roof < 0.56) {
      /* A spire. */
      d += `L${(x + w / 2).toFixed(1)} ${(top - range(rand, 14, 30)).toFixed(1)}`;
    }
    d += `H${x1.toFixed(1)}V${SKY_H}Z`;
    bodies.push(d);

    if (windows) {
      const cols = Math.floor((w - 8) / 11);
      const rows = Math.floor((h - 12) / 15);
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const on = rand();
          if (on > 0.5) {
            continue;
          }
          const wx = x + 5 + c * 11;
          const wy = top + 8 + r * 15;
          const rect = `M${wx.toFixed(1)} ${wy.toFixed(1)}h4v6h-4z`;
          if (animate && on < 0.03 && flicker.length < 14) {
            flicker.push(
              `<path d="${rect}" fill="#f5d78a" opacity=".8">` +
                `<animate attributeName="opacity" values=".85;.1;.85" dur="${range(rand, 6, 14).toFixed(1)}s" ` +
                `begin="${(-rand() * 10).toFixed(1)}s" repeatCount="indefinite"/></path>`
            );
          } else {
            (on < 0.22 ? lit.bright : lit.dim).push(rect);
          }
        }
      }
    }
    x = x1 + range(rand, 2, 16);
  }
  return { bodies: bodies.join(""), lit, flicker: flicker.join("") };
};

const skylineSvg = (seed, motion) => {
  const rand = seeded(seed);
  const far = skylineLayer(rand, { minH: 70, maxH: 150, windows: true, animate: false });
  const near = skylineLayer(rand, { minH: 26, maxH: 104, windows: true, animate: motion });
  return `
    <svg class="wn-skyline" width="100%" height="${SKY_H}" aria-hidden="true" focusable="false">
      <defs>
        <pattern id="wn-sky-tile" patternUnits="userSpaceOnUse" width="${SKY_TILE}" height="${SKY_H}">
          <path fill="#3a1721" d="${far.bodies}"></path>
          <path fill="#f5d78a" opacity=".22" d="${far.lit.bright.join("")}${far.lit.dim.join("")}"></path>
          <path fill="#160709" d="${near.bodies}"></path>
          <path fill="#f5d78a" opacity=".85" d="${near.lit.bright.join("")}"></path>
          <path fill="#f5d78a" opacity=".42" d="${near.lit.dim.join("")}"></path>
          ${near.flicker}
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#wn-sky-tile)"></rect>
    </svg>`;
};

/* A full moon, high on the right, with its halo painted on the sky. */
const MOON = `<span class="wn-moon"></span>`;

const starfield = (seed, n, maxTop) => {
  const rand = seeded(seed);
  return Array.from({ length: n }, () => {
    const size = 1.5 + rand() * 2.5;
    return `<span class="wn-star" style="left:${(rand() * 100).toFixed(1)}%;top:${(
      6 +
      rand() * maxTop
    ).toFixed(1)}%;width:${size.toFixed(1)}px;height:${size.toFixed(1)}px;opacity:${(
      0.4 +
      rand() * 0.6
    ).toFixed(2)};--delay:${(-rand() * 3).toFixed(1)}s"></span>`;
  }).join("");
};

/* The artboard's footer treeline: a run of sharp firs cut straight from the
   baseline, with a snow ground in front. */
const TREELINE = `
  <svg class="wn-treeline" viewBox="0 0 1200 88" preserveAspectRatio="none"
       aria-hidden="true" focusable="false">
    <path fill="#1e2e26" d="M0 88 V70 l15 -26 l15 26 l19 -40 l19 40 l23 -32 l23 32 l15 -48 l15 48 l19 -36 l19 36 l23 -30 l23 30 l15 -44 l15 44 l19 -38 l19 38 l23 -52 l23 52 l15 -34 l15 34 l19 -42 l19 42 l23 -28 l23 28 l15 -46 l15 46 l19 -36 l19 36 l23 -50 l23 50 l15 -30 l15 30 l19 -40 l19 40 l23 -34 l23 34 l15 -44 l15 44 l19 -38 l19 38 l23 -26 l23 26 l15 -40 l15 40 l19 -32 l19 32 l23 -48 l23 48 l15 -36 l15 36 l19 -30 l19 30 l23 -44 l23 44 l15 -38 l15 38 l19 -52 l19 52 l23 -34 l23 34 l15 -42 l15 42 l19 -28 l19 28 V88 Z"></path>
    <path fill="#f7f9fc" d="M0 88V76c100-10 200 6 300-2s200-12 300-4 200 12 300 2 200-10 300-2V88z"></path>
  </svg>`;

/* ---------------------------------------------------------------------------
   Light strings
   ---------------------------------------------------------------------------
   One drawing, used three times: hanging off the header, and draped over the
   two section seams the way a garland goes over a railing.

   The wire is a row of parabolic sags between hooks, drawn as an SVG that is
   stretched to the host's width. A stretched curve is still a smooth curve.
   The bulbs are not in the SVG: they are positioned elements at a percentage
   across and a pixel height taken from the same formula, so they keep their
   shape at any width and always sit on the wire.
   -------------------------------------------------------------------------- */

const WIRE_W = 1200;

/* Warm white, gold, ice blue, garnet: winter lights rather than one holiday's,
   and the pair in the middle is the site's own. */
const BULB_COLORS = ["#fff1c4", "#ceb888", "#bfe0f5", "#c1273b"];

/* Every bulb is lit. The colour cycle and the breathing glow are CSS
   keyframes; the per-bulb delays here are what turn fourteen identical
   animations into a string that always shows all four colours, with the
   glow travelling slowly down the wire rather than the whole string
   pulsing at once. */
const CYCLE_PERIOD = 32;

const lightString = ({ hooks, sag, top, bulbs, bulbClass, colorAt, extra = "" }) => {
  const seg = WIRE_W / hooks;
  const wireY = (x) => {
    const t = (x % seg) / seg;
    return top + 4 * sag * t * (1 - t);
  };
  let d = `M0 ${top}`;
  for (let k = 0; k < hooks; k += 1) {
    d += ` Q${(k + 0.5) * seg} ${top + 2 * sag} ${(k + 1) * seg} ${top}`;
  }
  const height = top + sag + 22;
  const spans = [];
  for (let i = 0; i < bulbs; i += 1) {
    const x = 30 + i * ((WIRE_W - 60) / (bulbs - 1));
    spans.push(
      `<span class="${bulbClass}" style="left:${((x / WIRE_W) * 100).toFixed(3)}%;` +
        `top:${wireY(x).toFixed(1)}px;--bulb:${colorAt(i)};` +
        `--cycle-delay:${(-(((i % 4) * CYCLE_PERIOD) / 4 + i * 0.35)).toFixed(2)}s;` +
        `--breathe-delay:${(-i * 0.42).toFixed(2)}s${extra}"></span>`
    );
  }
  return {
    height,
    html:
      `<svg class="wn-wire" viewBox="0 0 ${WIRE_W} ${height}" preserveAspectRatio="none" ` +
      `aria-hidden="true" focusable="false">` +
      `<path d="${d}" fill="none" stroke="#ceb888" stroke-width="1.4" opacity=".7"></path></svg>` +
      spans.join("")
  };
};

const headerLights = () =>
  lightString({
    hooks: 4,
    sag: 17,
    top: 2,
    bulbs: 14,
    bulbClass: "wn-bulb",
    colorAt: (i) => BULB_COLORS[i % BULB_COLORS.length]
  });

/* Over a seam: two gentle swags, mostly warm white with a gold or a blue
   bulb here and there. No colour cycle — against a white page a string that
   changes colour is a lot of event; these just breathe. */
const seamLights = (seed, sag) => {
  const rand = seeded(seed);
  return lightString({
    hooks: 2,
    sag,
    top: 0,
    bulbs: 16,
    bulbClass: "wn-seam-bulb",
    colorAt: () => {
      const r = rand();
      return r < 0.62 ? "#fff1c4" : r < 0.84 ? "#ceb888" : "#bfe0f5";
    }
  });
};

/* ---------------------------------------------------------------------------
   The nights of December
   ---------------------------------------------------------------------------
   Small, date-specific additions to the footer and the hero. Each appears only
   on its own dates; on every other night the scene is simply winter.

   Hanukkah moves with the Hebrew calendar, so its first evening is a table
   rather than a rule. Extend it as years are added; a year missing from the
   table shows no menorah, nothing worse.
   -------------------------------------------------------------------------- */

const HANUKKAH_FIRST_EVENING = {
  2024: [11, 25],
  2025: [11, 14],
  2026: [11, 4],
  2027: [11, 24],
  2028: [11, 12],
  2029: [11, 1],
  2030: [11, 20]
};

const DAY_MS = 86400000;

/* Which candles are lit on a given date and time. Candles are lit at nightfall
   and stay in the picture through the following day, so the count switches
   at five in the afternoon rather than at midnight. */
const hanukkahNight = (date) => {
  /* A Hanukkah that starts late in December runs into January. */
  const year = date.getMonth() === 0 ? date.getFullYear() - 1 : date.getFullYear();
  const first = HANUKKAH_FIRST_EVENING[year];
  if (!first) {
    return 0;
  }
  const start = new Date(year, first[0], first[1]);
  const evening = new Date(date.getTime() - 17 * 3600000);
  const night = Math.floor((evening - start) / DAY_MS) + 1;
  return night >= 1 && night <= 8 ? night : 0;
};

/* Kwanzaa: 26 December through 1 January, a candle a day. */
const kwanzaaDay = (date) => {
  const m = date.getMonth();
  const d = date.getDate();
  if (m === 11 && d >= 26) {
    return d - 25;
  }
  if (m === 0 && d === 1) {
    return 7;
  }
  return 0;
};

const isSolstice = (date) => date.getMonth() === 11 && date.getDate() === 21;
const isNewYearsEve = (date) => date.getMonth() === 11 && date.getDate() === 31;

/* `?date=2026-12-31` previews another night. Parsed by parts so it lands in
   local time; `new Date("2026-12-31")` is UTC midnight, which is the evening
   before in Tallahassee. */
const tonight = () => {
  const raw = new URLSearchParams(window.location.search).get("date");
  const m = raw && /^(\d{4})-(\d{2})-(\d{2})$/.exec(raw);
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), 20) : new Date();
};

const FLAME = (x, y, motion, delay) =>
  `<g class="wn-flame${motion ? " wn-flame-live" : ""}" style="--delay:${delay}s" transform="translate(${x} ${y})">` +
  `<ellipse cx="0" cy="-5" rx="3.6" ry="6.5" fill="#f6b73c" opacity=".9"/>` +
  `<ellipse cx="0" cy="-4" rx="1.8" ry="3.6" fill="#fff3c4"/>` +
  `<circle cx="0" cy="-5" r="9" fill="#f6b73c" opacity=".16"/></g>`;

/* A hanukkiah: eight candles at one height with the shamash raised in the
   middle. Candles go in from the right, one more each night, so night one is
   the rightmost holder and the shamash. */
const menorahSvg = (night, motion) => {
  const rand = seeded(night);
  const holders = [0, 1, 2, 3, 5, 6, 7, 8];
  const lit = new Set(holders.slice().reverse().slice(0, night));
  const parts = [];
  for (let i = 0; i < 9; i += 1) {
    const x = 12 + i * 12;
    const shamash = i === 4;
    const top = shamash ? 22 : 34;
    /* Arms rise from the stem to each holder. */
    parts.push(`<path d="M60 74 L60 ${top + 12} L${x} ${top + 12} L${x} ${top + 6}" stroke="#ceb888" stroke-width="2.2" fill="none"/>`);
    parts.push(`<rect x="${x - 3}" y="${top + 4}" width="6" height="3" fill="#ceb888"/>`);
    if (shamash || lit.has(i)) {
      parts.push(`<rect x="${x - 2}" y="${top - 12}" width="4" height="16" fill="${shamash ? "#f4efe2" : "#e9edf6"}"/>`);
      parts.push(FLAME(x, top - 12, motion, (-rand() * 3).toFixed(1)));
    }
  }
  return `
    <svg class="wn-menorah" viewBox="0 0 120 84" aria-hidden="true" focusable="false">
      <path d="M44 84 Q60 72 76 84 Z" fill="#ceb888"/>
      ${parts.join("")}
    </svg>`;
};

/* A kinara: three red, one black, three green, lit one a day — the black
   first, then alternating from the outside in, red before green. */
const kinaraSvg = (day, motion) => {
  const rand = seeded(day + 40);
  const reds = Math.ceil((day - 1) / 2);
  const greens = Math.floor((day - 1) / 2);
  const parts = [];
  for (let i = 0; i < 7; i += 1) {
    const x = 12 + i * 16;
    const color = i < 3 ? "#b3232f" : i === 3 ? "#1c1a1a" : "#2f7d3a";
    const lit = i === 3 || (i < 3 && i < reds) || (i > 3 && i >= 7 - greens);
    parts.push(`<rect x="${x - 4}" y="54" width="8" height="5" fill="#8a6a3a"/>`);
    parts.push(`<rect x="${x - 3}" y="30" width="6" height="26" fill="${color}"/>`);
    if (lit) {
      parts.push(FLAME(x, 30, motion, (-rand() * 3).toFixed(1)));
    }
  }
  return `
    <svg class="wn-kinara" viewBox="0 0 120 64" aria-hidden="true" focusable="false">
      <path d="M4 64 L10 58 L110 58 L116 64 Z" fill="#6b4a25"/>
      ${parts.join("")}
    </svg>`;
};

/* Three bursts in the footer sky for the last night of the year. */
const fireworksHtml = (motion) => {
  const rand = seeded(99);
  const colors = ["#ceb888", "#c1273b", "#bfe0f5"];
  return Array.from({ length: 3 }, (_, k) => {
    const rays = [];
    for (let i = 0; i < 16; i += 1) {
      const a = (i / 16) * Math.PI * 2;
      const r = 26 + rand() * 10;
      rays.push(
        `<line x1="0" y1="0" x2="${(Math.cos(a) * r).toFixed(1)}" y2="${(Math.sin(a) * r).toFixed(1)}"/>` +
          `<circle cx="${(Math.cos(a) * (r + 4)).toFixed(1)}" cy="${(Math.sin(a) * (r + 4)).toFixed(1)}" r="1.6"/>`
      );
    }
    return (
      `<svg class="wn-firework${motion ? " wn-firework-live" : ""}" viewBox="-40 -40 80 80" ` +
      `style="left:${(18 + k * 28 + rand() * 8).toFixed(1)}%;top:${(10 + rand() * 18).toFixed(1)}%;--delay:${(k * 1.9).toFixed(1)}s" ` +
      `aria-hidden="true" focusable="false"><g stroke="${colors[k]}" fill="${colors[k]}" stroke-width="1.3" stroke-linecap="round">` +
      rays.join("") +
      `</g></svg>`
    );
  }).join("");
};

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

const buildSnow = (overlay, density, motion) => {
  const count = flakeCount(density);
  if (count === 0) {
    return;
  }
  overlay.appendChild(
    buildParticles(count, 11, (index, rand) => {
      const size = range(rand, 3, 9);
      const duration = range(rand, 12, 26);
      const left = rand() * 100;
      const opacity = range(rand, 0.3, 0.9);
      const color = pick(rand, FLAKE_COLORS);
      const swayDur = range(rand, 2.5, 5.5);

      /* With motion off the flakes are not hidden, they are *landed*: seeded
         positions spread down the viewport so the scene still reads as
         snowfall, just a photograph of one. */
      const outer = make("div", {
        class: `season-particle${motion ? " wn-flake-fall" : ""}`,
        style: {
          left: `${left.toFixed(2)}%`,
          top: motion ? "-40px" : `${range(rand, 2, 96).toFixed(2)}vh`,
          width: `${size.toFixed(1)}px`,
          height: `${size.toFixed(1)}px`,
          opacity: opacity.toFixed(2),
          "--dur": `${duration.toFixed(1)}s`,
          /* A negative delay starts each flake partway through its fall, so
             the first frame is already a full snowfall rather than an empty
             sky that fills over the next twenty seconds. */
          "--delay": `${(-rand() * duration).toFixed(1)}s`
        }
      });

      const inner = make("div", {
        class: `wn-flake${motion ? " wn-flake-sway" : ""}`,
        style: {
          "--flake": color,
          "--sway": `${swayDur.toFixed(1)}s`,
          "--sway-delay": `${(-rand() * 3).toFixed(1)}s`
        }
      });

      outer.appendChild(inner);
      return outer;
    })
  );
};

export const mount = ({ overlay, density, motion }) => {
  const disposer = new Disposer();

  buildSnow(overlay, density, motion);

  /* Header: a string of lights hanging off the bar's bottom edge, all lit,
     slowly trading colours. No snow on the bar itself. */
  decorate(disposer, ".site-header", "season-scene wn-lights", headerLights().html);

  const night = tonight();

  const brand = document.querySelector(".site-header .brand");
  if (brand) {
    const cap = make("span", { class: "wn-cap-host", "aria-hidden": "true" });
    cap.innerHTML = SNOW_CAP;
    brand.appendChild(cap);
    disposer.node(cap);
  }

  /* Hero: a moonlit night — sky, stars, the moon, frost creeping in from the
     upper corners, and a city skyline along the bottom. */
  decorate(
    disposer,
    ".hero",
    "season-scene wn-hero",
    `<div class="season-sky"></div>
     ${starfield(3, 30, 52)}
     ${MOON}
     <div class="wn-frost wn-frost-left">${FROST_CORNER}</div>
     <div class="wn-frost wn-frost-right">${FROST_CORNER}</div>
     ${skylineSvg(7, motion)}`
  );
  if (isSolstice(night)) {
    /* The longest night: the moon rides higher and larger. */
    document.querySelectorAll(".wn-hero").forEach((scene) => scene.classList.add("wn-solstice"));
  }

  /* The seam under the hero: a string of lights draped over it like a
     railing, hanging into the section below. It lives in the overview rather
     than the hero, because the hero clips its children. */
  decorate(disposer, ".section-overview", "season-scene wn-seam wn-seam-hero", seamLights(41, 34).html, {
    first: true
  });

  /* Footer: the artboard's own night — garnet gradient, stars, a fir treeline
     with snow on the ground — and, on the nights that have one, a candle
     stand on the snow or fireworks in the sky. */
  const hanukkah = hanukkahNight(night);
  const kwanzaa = kwanzaaDay(night);
  decorate(
    disposer,
    ".site-footer",
    "season-scene wn-footer",
    `<div class="season-sky"></div>
     ${starfield(23, 22, 48)}
     ${isNewYearsEve(night) ? fireworksHtml(motion) : ""}
     ${TREELINE}
     ${hanukkah ? `<div class="wn-vigil wn-vigil-left">${menorahSvg(hanukkah, motion)}</div>` : ""}
     ${kwanzaa ? `<div class="wn-vigil wn-vigil-right">${kinaraSvg(kwanzaa, motion)}</div>` : ""}`
  );

  /* Section seam between the overview and the dashboard: the same string,
     shallower. The adjacent-sibling selector matters: the subpages reuse
     `.section-dashboard` as their only section, with no overview before it, so
     a bare class selector hangs lights directly under the header on every one
     of them. */
  decorate(
    disposer,
    ".section-overview + .section-dashboard",
    "season-scene wn-seam wn-seam-dashboard",
    seamLights(43, 22).html,
    { first: true }
  );

  /* Portraits: the frost ring, on hover only. */
  decorate(
    disposer,
    ".speaker-directory-photo, .seminar-speaker-photo",
    "season-ring wn-ring",
    FROST_RING
  );

  /* A snowflake resting in the corner of each speaker card. Kept very faint —
     it is a watermark, not a badge. */
  decorate(
    disposer,
    ".speaker-directory-card",
    "season-card-art wn-card-flake",
    `<span class="wn-corner-flake">${SNOWFLAKE}</span>`
  );

  /* The community cards get the artboard's "Snow and icicles" frame: snow
     settled on the top edge with icicles hanging under it. */
  decorate(
    disposer,
    ".community-card",
    "season-frame wn-frame wn-frame-icicles",
    `${CARD_SNOW}${icicles(31)}`
  );

  /* Talk cards get the artboard's first card treatment: frost creeping into
     opposite corners on hover. */
  decorate(
    disposer,
    ".talk-card",
    "season-card-art wn-card-frost",
    `<span class="wn-frost-corner wn-frost-corner-tr">${FROST_CORNER}</span>
     <span class="wn-frost-corner wn-frost-corner-bl">${FROST_CORNER}</span>`
  );

  return {
    destroy() {
      disposer.dispose();
    }
  };
};
