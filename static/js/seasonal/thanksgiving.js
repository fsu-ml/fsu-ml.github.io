/**
 * Thanksgiving — November.
 *
 * The idea that separates this from Winter is the wind. One JS source writes a
 * single unitless `--wind` to the document root each frame; every consumer —
 * leaves, wheat, the garland — reads it through `calc()`. One variable, zero
 * per-element JavaScript, which is what keeps a whole page of reacting parts
 * cheap.
 *
 * Leaves are the exception and live on a canvas, because they need to tumble in
 * three axes, flee the pointer and settle individually along the bottom, none
 * of which a keyframe can do.
 */

import {
  Disposer,
  decorate,
  Loop,
  Surface,
  bindProgress,
  clamp,
  make,
  onButtonPress,
  range,
  seeded,
  smoothstep,
  trackPointer
} from "./engine.js";

const LEAF_COLORS = ["#d98e3f", "#b4562c", "#8e1f31", "#ceb888"];
const LEAF_KINDS = ["maple", "oak", "birch", "ovate"];
const LEAF_PATH = {
  maple: "M12 2l2 5 5-1-3 4 4 3-5 1 1 5-4-3-4 3 1-5-5-1 4-3-3-4 5 1z",
  oak: "M12 2c-3 0-4 3-3 5-3 0-4 3-2 5-2 2 0 5 3 4l2 6 2-6c3 1 5-2 3-4 2-2 1-5-2-5 1-2 0-5-3-5z",
  birch: "M12 2C7 6 5 11 6 16c1 3 4 5 6 6 2-1 5-3 6-6 1-5-1-10-6-14z",
  ovate: "M12 2C5 5 3 12 6 21c7-1 12-7 12-14 0-2-1-4-2-5-1 3-3 5-5 6 2-2 2-5 1-6z"
};

const leafCount = (density) => Math.min(30, Math.round(density / 4));

/* ---------------------------------------------------------------------------
   Wind
   -------------------------------------------------------------------------- */

/**
 * A gust every 8-18s: a smoothstep rise over 1.5s, a hold of 0.5-2s, then an
 * exponential decay, all over a faint idle breeze so the page is never
 * completely still between gusts.
 *
 * The only output is one number on one element. Nothing here touches the DOM
 * tree or triggers a render.
 */
class Wind {
  constructor(root) {
    this.root = root;
    this.t = 0;
    this.next = 2.5;
    this.gust = null;
    this.value = 0;
  }

  write(v) {
    this.value = v;
    this.root.style.setProperty("--wind", v.toFixed(2));
  }

  step(dt) {
    const t = (this.t += dt);
    if (!this.gust && t >= this.next) {
      this.gust = { t0: t, amp: range(Math.random, 0.4, 1), hold: range(Math.random, 0.5, 2) };
    }
    let env = 0;
    if (this.gust) {
      const u = t - this.gust.t0;
      const g = this.gust;
      if (u < 1.5) {
        env = smoothstep(u / 1.5);
      } else if (u < 1.5 + g.hold) {
        env = 1;
      } else {
        env = Math.exp(-(u - 1.5 - g.hold) * 1.1);
        if (env < 0.02) {
          this.gust = null;
          this.next = t + range(Math.random, 8, 18);
          env = 0;
        }
      }
      env *= g.amp;
    }
    /* Three incommensurate sines: a breeze that never repeats audibly. */
    const idle = 4 * Math.sin(0.7 * t) + 3 * Math.sin(1.3 * t + 1) + 2 * Math.sin(2.1 * t + 2);
    this.write(clamp(idle + env * 90, -10, 95));
  }

  /** A single representative gust value, for the motion-off still frame. */
  settle() {
    this.write(6);
  }
}

/* ---------------------------------------------------------------------------
   Leaves
   -------------------------------------------------------------------------- */

class LeafField {
  constructor(surface, { target, wind }) {
    this.surface = surface;
    this.target = target;
    this.wind = wind;
    this.leaves = [];
    this.settled = [];
    this.pointer = null;
    this.t = 0;
    this._filled = false;
    this.paths = {};
    for (const kind of LEAF_KINDS) {
      this.paths[kind] = new Path2D(LEAF_PATH[kind]);
    }
  }

  make(fromTop) {
    const d = range(Math.random, 0.4, 1.6);
    return {
      /* Depth drives size, fall speed, opacity and how hard the wind shoves
         it, so one number gives the field its parallax. */
      d,
      size: 10 + d * 9,
      x: Math.random() * this.surface.width,
      y: fromTop
        ? range(Math.random, -80, -20)
        : range(Math.random, -this.surface.height, this.surface.height),
      kind: LEAF_KINDS[Math.floor(Math.random() * LEAF_KINDS.length)],
      color: LEAF_COLORS[Math.floor(Math.random() * LEAF_COLORS.length)],
      vx: 0,
      vy: 0.6 + d * 0.9,
      rot: Math.random() * 6.28,
      rs: range(Math.random, -0.02, 0.02),
      ph: Math.random() * 6.28,
      tum: range(Math.random, 0.03, 0.06)
    };
  }

  step(dt) {
    if (!this.surface.ready) {
      return;
    }
    const k = dt * 60;
    const W = this.surface.width;
    const H = this.surface.height;
    const wind = this.wind.value;
    const P = this.pointer && this.pointer.active ? this.pointer : null;
    this.t += dt;

    /* The first fill spreads leaves through the whole column of air; refills
       after that drop in from above. */
    while (this.leaves.length < this.target) {
      this.leaves.push(this.make(this._filled));
    }
    this._filled = true;
    if (this.leaves.length > this.target) {
      this.leaves.length = this.target;
    }

    for (const l of this.leaves) {
      const vy0 = 0.6 + l.d * 0.9;
      let fx = 0;
      let fy = 0;
      if (P) {
        const dx = l.x - P.x;
        const dy = l.y - P.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 140 && dist > 0.01) {
          const f = 1 - dist / 140;
          fx += (dx / dist) * f * 1.6;
          /* The extra upward term is what makes a leaf lift out of the way
             rather than merely being pushed sideways. */
          fy += (dy / dist) * f * 1.2 - f * 1.5;
          l.rs += (Math.random() - 0.5) * f * 0.08;
        }
      }
      l.vx = (l.vx + fx * k) * 0.93;
      l.vy += fy * k;
      l.vy += (vy0 - l.vy) * 0.05 * k;

      const sway = Math.sin(this.t * 1.3 + l.ph) * 0.5 * l.d;
      const shove = wind * 0.03 * l.d;
      l.x += (l.vx + sway + shove) * k;
      l.y += l.vy * k;
      l.rot += (l.rs + l.vx * 0.01) * k;
      l.ph += (l.tum + Math.abs(l.vx) * 0.01) * k;
      l.rs *= 0.98;

      if (l.x < -40) {
        l.x = W + 30;
      } else if (l.x > W + 40) {
        l.x = -30;
      }
      if (l.y > H + 12) {
        /* Landed. It joins a bounded set of settled leaves along the bottom
           and a fresh one enters at the top, so the field stays at target. */
        if (this.settled.length >= 36) {
          this.settled.shift();
        }
        this.settled.push({ x: l.x, rot: l.rot, kind: l.kind, color: l.color, size: l.size, t: this.t });
        Object.assign(l, this.make(true));
      }
    }

    /* Settled leaves fade out after a couple of minutes, so the bottom of a
       long-open page does not silently accumulate a solid band. */
    for (let i = this.settled.length - 1; i >= 0; i -= 1) {
      if (this.t - this.settled[i].t > 150) {
        this.settled.splice(i, 1);
      }
    }
  }

  drawLeaf(ctx, x, y, rot, flip, size, kind, color, alpha) {
    const sc = size / 24;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rot);
    /* `flip` is the cosine of the tumble phase, so scaling y by it turns the
       leaf edge-on and back: a tumble in three dimensions from a 2D context. */
    ctx.scale(sc, sc * flip);
    ctx.globalAlpha = alpha;
    ctx.translate(-12, -12);
    ctx.fillStyle = color;
    ctx.fill(this.paths[kind]);
    ctx.strokeStyle = "rgba(42,17,24,.35)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(12, 4);
    ctx.lineTo(12, 20);
    ctx.stroke();
    ctx.restore();
  }

  draw() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    const H = this.surface.height;
    for (const s of this.settled) {
      const age = this.t - s.t;
      this.drawLeaf(
        ctx,
        s.x,
        H - s.size * 0.32,
        s.rot,
        1,
        s.size,
        s.kind,
        s.color,
        0.85 * Math.min(1, (150 - age) / 40)
      );
    }
    for (const l of this.leaves) {
      const flip = Math.cos(l.ph);
      this.drawLeaf(
        ctx,
        l.x,
        l.y,
        l.rot,
        Math.abs(flip) < 0.12 ? 0.12 : flip,
        l.size,
        l.kind,
        l.color,
        0.5 + l.d * 0.3
      );
    }
  }

  /**
   * Throws n leaves out of a point, for the burst a button fires on click.
   * They are ordinary field members from the next frame on, so they tumble,
   * catch the wind and settle exactly like the rest.
   */
  burst(x, y, n) {
    for (let i = 0; i < n; i += 1) {
      const leaf = this.make(true);
      const a = (i / n) * Math.PI * 2 + Math.random() * 0.6;
      const sp = range(Math.random, 2.4, 5.2);
      Object.assign(leaf, { x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 1.4 });
      this.leaves.push(leaf);
    }
    /* The extra leaves are over target, so `step` trims the field back down
       over the following frames rather than letting a click inflate it. */
  }

  /** Reduced motion: a scattering of leaves already lying along the bottom. */
  drawStatic() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    const rand = seeded(77);
    const H = this.surface.height;
    const W = this.surface.width;
    const n = Math.max(6, Math.round(this.target * 0.8));
    for (let i = 0; i < n; i += 1) {
      const size = range(rand, 12, 26);
      this.drawLeaf(
        ctx,
        rand() * W,
        H - size * 0.32,
        rand() * 6.28,
        1,
        size,
        LEAF_KINDS[Math.floor(rand() * LEAF_KINDS.length)],
        LEAF_COLORS[Math.floor(rand() * LEAF_COLORS.length)],
        0.85
      );
    }
  }
}

/* ---------------------------------------------------------------------------
   Artwork
   -------------------------------------------------------------------------- */

const leafSvg = (kind, color, size, extra = "") =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" aria-hidden="true" ${extra}>
     <path d="${LEAF_PATH[kind]}" fill="${color}"></path>
   </svg>`;

/* Garland of individual leaves and berries along the bar's bottom edge. Kept
   static: a whole row of leaves swaying under the navigation was movement
   exactly where the eye is trying to read. */
const garland = (seed, count = 26) => {
  const rand = seeded(seed);
  const parts = [];
  for (let i = 0; i < count; i += 1) {
    const x = 1.5 + i * (97 / count) + rand() * 2;
    const size = 15 + rand() * 9;
    const rot = -60 + rand() * 120;
    parts.push(
      `<span class="tg-garland-leaf" style="left:${x.toFixed(2)}%;top:${(-size * 0.42).toFixed(
        1
      )}px;transform:rotate(${rot.toFixed(0)}deg)">${leafSvg(
        LEAF_KINDS[Math.floor(rand() * 4)],
        LEAF_COLORS[Math.floor(rand() * 4)],
        size
      )}</span>`
    );
    if (rand() > 0.55) {
      parts.push(`<span class="tg-berry" style="left:${(x + 2).toFixed(2)}%"></span>`);
    }
  }
  return parts.join("");
};

/* The table after dinner: a plank surface, produce, a pie, two candles and a
   few leaves that drifted in, under a night sky. Straight from the artboard's
   footer, which is the composition the whole theme was designed around. */
const PLANK_BG =
  "repeating-linear-gradient(90deg,#6e3a22 0 160px,#7a4228 160px 300px,#63341f 300px 460px)";

const produce = {
  pumpkin: `<svg viewBox="0 0 60 52" aria-hidden="true"><path d="M28 8c-1-4 0-7 4-8 1 3 0 6-1 8z" fill="#3e6b52"/><path d="M8 16C2 20 0 34 6 44s16 8 24 8 18 2 24-8 4-24-2-28c-8-4-14-4-22-4S16 12 8 16z" fill="#e07a2f"/><path d="M30 12c-8 0-12 10-12 20s4 20 12 20 12-10 12-20-4-20-12-20z" fill="none" stroke="#b4562c" stroke-width="1.2"/></svg>`,
  apple: `<svg viewBox="0 0 34 36" aria-hidden="true"><path d="M17 8C8 4 2 12 4 22s8 14 13 12c5 2 11-2 13-12s-4-18-13-14z" fill="#a32638"/><path d="M17 9c0-4 2-7 5-8" stroke="#4a1c27" stroke-width="1.6" fill="none"/><path d="M18 7c3-3 6-3 8-1-3 2-6 2-8 1z" fill="#3e6b52"/></svg>`,
  pear: `<svg viewBox="0 0 32 40" aria-hidden="true"><path d="M16 6c-2 8-12 12-12 22 0 7 6 11 12 11s12-4 12-11c0-10-10-14-12-22z" fill="#ceb888"/><path d="M16 7c0-3 1-5 3-6" stroke="#4a1c27" stroke-width="1.4" fill="none"/></svg>`,
  grapes: `<svg viewBox="0 0 36 42" aria-hidden="true">${[
    [18, 8],
    [12, 14],
    [24, 14],
    [8, 22],
    [18, 22],
    [28, 22],
    [12, 30],
    [24, 30],
    [18, 37]
  ]
    .map(([x, y]) => `<circle cx="${x}" cy="${y}" r="5.5" fill="#6b4c9a" stroke="#4a2f6e" stroke-width=".8"/>`)
    .join("")}<path d="M18 4v-4" stroke="#3e6b52" stroke-width="1.6"/></svg>`,
  corn: `<svg viewBox="0 0 30 48" aria-hidden="true"><path d="M15 2c6 0 9 10 9 24s-3 20-9 20-9-6-9-20S9 2 15 2z" fill="#e8c55a"/><path d="M9 10h12M8 18h14M8 26h14M9 34h12M11 41h8" stroke="#c9a63a" stroke-width="1"/><path d="M6 20c-4 8-4 18 0 26 3-8 3-18 0-26zM24 20c4 8 4 18 0 26-3-8-3-18 0-26z" fill="#3e6b52"/></svg>`,
  squash: `<svg viewBox="0 0 46 34" aria-hidden="true"><path d="M4 20c0-10 8-16 19-16s19 6 19 16-8 12-19 12S4 30 4 20z" fill="#5b7a3a"/><path d="M12 8c0 8 0 16 2 22M23 4v28M34 8c0 8 0 16-2 22" stroke="#e8d5a8" stroke-width="1.6" fill="none"/><path d="M22 6c0-3 1-5 3-6" stroke="#4a1c27" stroke-width="1.4" fill="none"/></svg>`,
  pie: `<svg viewBox="0 0 80 40" aria-hidden="true"><ellipse cx="40" cy="30" rx="37" ry="8" fill="#8c4d2e"/><ellipse cx="40" cy="24" rx="34" ry="10" fill="#b4562c"/><path d="M12 20l16 10M22 16l22 14M34 14l20 14M46 14l14 12M8 26l14-12M18 32l22-18M32 34l24-20M48 32l14-14" stroke="#e8c55a" stroke-width="2.6" stroke-linecap="round"/><ellipse cx="40" cy="24" rx="36" ry="10" fill="none" stroke="#d9a65a" stroke-width="4"/></svg>`,
  candle: `<svg viewBox="0 0 24 60" aria-hidden="true" style="overflow:visible"><g class="tg-flame"><path d="M12 4c4 6 6 10 6 14a6 6 0 0 1-12 0c0-4 2-8 6-14z" fill="#f2a65a"/><path d="M12 11c2 3 3 5 3 7.5a3 3 0 0 1-6 0c0-2.5 1-4.5 3-7.5z" fill="#ffe39a"/></g><rect x="11.2" y="22" width="1.6" height="5" fill="#2a1118"/><rect x="6" y="26" width="12" height="32" rx="2" fill="#f7f2e8"/><ellipse cx="12" cy="58" rx="9" ry="2.5" fill="#ceb888"/></svg>`
};

const scatterLeaves = (seed, n, b0, bh) => {
  const rand = seeded(seed);
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const size = 14 + rand() * 10;
    out.push(
      `<span class="tg-table-leaf" style="left:${(rand() * 92).toFixed(1)}%;bottom:${(
        b0 +
        rand() * bh
      ).toFixed(0)}px;transform:rotate(${(rand() * 360).toFixed(0)}deg)">${leafSvg(
        LEAF_KINDS[Math.floor(rand() * 4)],
        LEAF_COLORS[Math.floor(rand() * 4)],
        size
      )}</span>`
    );
  }
  return out.join("");
};

const harvestTable = () => {
  const items = [
    ["pumpkin", 74, 5],
    ["squash", 58, 14],
    ["pie", 84, 25],
    ["candle", 18, 38],
    ["apple", 30, 44],
    ["grapes", 34, 50],
    ["pear", 30, 58],
    ["candle", 18, 66],
    ["corn", 30, 72],
    ["pumpkin", 56, 79],
    ["apple", 26, 88],
    ["squash", 44, 92]
  ]
    .map(
      ([id, w, x]) =>
        `<span class="tg-item" style="left:${x}%;width:${w}px">${produce[id]}</span>`
    )
    .join("");
  /* Warm pools under the two candles, so the flames actually light the table
     rather than floating on it. */
  const glow = [38, 66]
    .map((x) => `<span class="tg-candleglow" style="left:calc(${x}% - 60px)"></span>`)
    .join("");
  return `
    <div class="tg-table">
      ${glow}
      <div class="tg-plank" style="background:${PLANK_BG}"><span></span></div>
      ${items}
      ${scatterLeaves(41, 9, 14, 60)}
    </div>`;
};

const starfield = (seed, n) => {
  const rand = seeded(seed);
  return Array.from(
    { length: n },
    () =>
      `<span class="tg-star" style="left:${(rand() * 100).toFixed(1)}%;top:${(
        rand() * 52
      ).toFixed(1)}%;--delay:${(-rand() * 3).toFixed(1)}s"></span>`
  ).join("");
};

/* A V of seven geese. One shared flap phase with a per-bird delay, so the
   formation beats together without being in perfect lockstep. */
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
  const birds = offsets
    .map(
      ([dx, dy], i) =>
        `<g class="tg-goose" style="animation-delay:${(-i * 0.07).toFixed(2)}s"
            transform="translate(${dx + 60} ${dy + 6})">
           <path d="M0 6 L7 0 L9 3 L11 0 L18 6" fill="none" stroke="#4a1c27"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
         </g>`
    )
    .join("");
  return `<div class="tg-geese"><svg viewBox="0 0 180 40" aria-hidden="true" focusable="false">${birds}</svg></div>`;
})();

/* Section divider: a hairline rule with an acorn at its centre. */
const ACORN = `
  <svg viewBox="0 0 24 30" aria-hidden="true" focusable="false">
    <path d="M5 12h14c0 8-4 14-7 16-3-2-7-8-7-16z" fill="#b4562c"></path>
    <path d="M4 12c0-6 4-9 8-9s8 3 8 9z" fill="#8e3b2a"></path>
    <path d="M12 3V0" stroke="#8e3b2a" stroke-width="1.6" stroke-linecap="round"></path>
  </svg>`;

const DIVIDER = `
  <span class="tg-rule"></span>
  <span class="tg-rule-mark">${ACORN}</span>
  <span class="tg-rule"></span>`;

/* Logo treatment for the circular speaker photographs: the artboard's leaf
   wreath. Leaves are laid around the rim at even angles with a seeded tilt, so
   nothing lands on a face. */
const leafWreath = (seed) => {
  const rand = seeded(seed);
  const n = 14;
  const parts = [];
  for (let i = 0; i < n; i += 1) {
    const a = (i / n) * 360 + range(rand, -7, 7);
    const size = 15 + rand() * 6;
    parts.push(
      `<span class="tg-wreath-leaf" style="transform:rotate(${a.toFixed(
        1
      )}deg) translateY(-50%) rotate(${(90 + range(rand, -22, 22)).toFixed(0)}deg)">${leafSvg(
        LEAF_KINDS[Math.floor(rand() * 4)],
        LEAF_COLORS[Math.floor(rand() * 4)],
        size
      )}</span>`
    );
  }
  return `<div class="tg-wreath">${parts.join("")}</div>`;
};

/* Card treatment: three leaves slide into the top corners on hover, each from
   a slightly different angle and on its own delay. */
const cardLeaves = (seed) => {
  const rand = seeded(seed);
  const spots = [
    [4, -6, -28],
    [16, 2, 22],
    [30, -3, -8]
  ];
  return `<div class="tg-card-leaves">${spots
    .map(
      ([right, top, rot], i) =>
        `<span style="right:${right}px;top:${top}px;--turn:${rot}deg;--wait:${(i * 70).toFixed(
          0
        )}ms">${leafSvg(
          LEAF_KINDS[Math.floor(rand() * 4)],
          LEAF_COLORS[Math.floor(rand() * 4)],
          20 + rand() * 8
        )}</span>`
    )
    .join("")}</div>`;
};

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

export const mount = ({ overlay, density, motion, root }) => {
  const disposer = new Disposer();

  const canvas = make("canvas", { class: "season-canvas tg-canvas", "aria-hidden": "true" });
  overlay.appendChild(canvas);
  const surface = new Surface(canvas);
  disposer.add(() => surface.destroy());

  const wind = new Wind(root);
  /* The variable lives on the document root for the life of the theme, so it
     has to be cleaned up explicitly — the scene nodes going away would not
     remove it. */
  disposer.add(() => root.style.removeProperty("--wind"));

  const leaves = new LeafField(surface, { target: leafCount(density), wind });
  leaves.pointer = trackPointer(disposer);

  const loop = new Loop({
    motion,
    step: (dt) => {
      wind.step(dt);
      leaves.step(dt);
    },
    draw: () => (motion ? leaves.draw() : leaves.drawStatic())
  });
  disposer.add(() => loop.destroy());
  surface.onChange = () => loop.draw();

  if (!motion) {
    /* A held breeze rather than dead calm: the wheat and garland read as
       caught mid-gust instead of collapsed to zero. */
    wind.settle();
  }
  loop.start();

  /* Header: a garland hanging off the bar's bottom edge. */
  decorate(disposer, ".site-header", "season-edge-strip tg-garland", garland(6));

  /* Hero: the warm wash and nothing else. The tumbling leaves already cross
     it from the ambient layer, and every scenic element tried here — sun,
     hills, wheat, leaf bank — competed with the banner art and the
     next-seminar card rather than sitting behind them. */
  decorate(
    disposer,
    ".hero",
    "season-scene tg-hero",
    `<div class="season-sky"></div>
     <div class="tg-dusk"></div>`
  );

  /* Footer: the artboard's own composition — the table after dinner under a
     night sky, with the link columns sitting unchanged above it. */
  decorate(
    disposer,
    ".site-footer",
    "season-scene tg-footer",
    `<div class="season-sky"></div>
     ${starfield(63, 16)}
     ${GEESE}
     ${harvestTable()}`
  );

  /* Drives the hero's sunset: the sky deepens from gold to dusk as the page
     is scrolled. One variable, read by CSS. */
  const syncProgress = bindProgress(disposer, root);

  /* Click a primary action and a handful of leaves tumble out of it. */
  onButtonPress(disposer, ".button-primary", (x, y) => leaves.burst(x, y, 9));

  /* Section seam between the overview and the dashboard.
     The adjacent-sibling selector matters: the subpages reuse
     `.section-dashboard` as their only section, with no overview before it, so
     a bare class selector hangs a seam divider directly under the header on
     every one of them. */
  decorate(disposer, ".section-overview + .section-dashboard", "season-divider tg-divider", DIVIDER, { first: true });

  /* Every circular speaker photograph gets the leaf wreath. */
  decorate(
    disposer,
    ".speaker-directory-photo, .seminar-speaker-photo",
    "season-ring",
    leafWreath(43)
  );

  /* Cards get leaves sliding into the corner on hover. */
  decorate(
    disposer,
    ".speaker-directory-card, .feature-card, .community-card, .talk-card",
    "season-card-art",
    cardLeaves(87)
  );

  return {
    /* Exposed for tuning and for verification: hidden documents never fire
       rAF, so this is the only way to advance either engine off-screen. */
    wind,
    leaves,
    loop,
    syncProgress,
    destroy() {
      disposer.dispose();
    }
  };
};
