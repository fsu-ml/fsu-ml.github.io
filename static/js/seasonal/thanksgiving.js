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

/* Card treatment: leaves slide in on hover and settle on the card's top edge —
   one at the left corner and two at the right. They rest above the edge, which
   is what makes them read as having landed on the card rather than being
   printed on it. */
const cardLeaves = (seed) => {
  const rand = seeded(seed);
  /* [side, offset from that side, top, resting angle] */
  const spots = [
    ["left", 10, -16, -24],
    ["right", 34, -18, 20],
    ["right", 8, -8, -12]
  ];
  return `<div class="tg-card-leaves">${spots
    .map(
      ([side, offset, top, rot], i) =>
        `<span style="${side}:${offset}px;top:${top}px;--turn:${rot}deg;--from:${
          side === "left" ? "-22px" : "22px"
        };--wait:${i * 80}ms">${leafSvg(
          LEAF_KINDS[Math.floor(rand() * 4)],
          LEAF_COLORS[Math.floor(rand() * 4)],
          22 + rand() * 8
        )}</span>`
    )
    .join("")}</div>`;
};

/* Leaf-scatter border: leaves set all the way round the edge of a card, from
   the artboard's card-outline set. A frame, not a reaction, so it does not
   wait on hover. */
const leafScatter = (seed) => {
  const rand = seeded(seed);
  const out = [];
  const put = (css) => {
    const size = 14 + rand() * 8;
    out.push(
      `<span class="tg-edge-leaf" style="${css};width:${size.toFixed(1)}px;margin:${(
        -size / 2
      ).toFixed(1)}px 0 0 ${(-size / 2).toFixed(1)}px;transform:rotate(${(rand() * 360).toFixed(
        0
      )}deg)">${leafSvg(
        LEAF_KINDS[Math.floor(rand() * 4)],
        LEAF_COLORS[Math.floor(rand() * 4)],
        size
      )}</span>`
    );
  };
  for (let i = 0; i < 9; i += 1) {
    put(`left:${(4 + i * 11.5).toFixed(1)}%;top:0`);
    put(`left:${(9 + i * 11.5).toFixed(1)}%;top:100%`);
  }
  for (let i = 0; i < 4; i += 1) {
    put(`left:0;top:${18 + i * 22}%`);
    put(`left:100%;top:${12 + i * 24}%`);
  }
  return `<div class="tg-scatter">${out.join("")}</div>`;
};

/* Acorn-and-berry ring for the portraits: a dashed circle with eight acorns on
   it and eight berries between them. On hover the circle draws itself on and
   the acorns and berries pop in around it, each a beat after the last. */
const ACORN_RING = (() => {
  const acorn = (a) => `
    <g class="tg-ring-item" style="--i:${a / 45}"
       transform="rotate(${a} 65 65) translate(65 15) rotate(${-a})">
      <g transform="translate(-6 -7.5) scale(.5)">
        <path d="M5 12h14c0 8-4 14-7 16-3-2-7-8-7-16z" fill="#b4562c"></path>
        <path d="M4 12c0-6 4-9 8-9s8 3 8 9z" fill="#8e3b2a"></path>
        <path d="M12 3V0" stroke="#8e3b2a" stroke-width="1.6" stroke-linecap="round"></path>
      </g>
    </g>`;
  const berry = (a) => `
    <g class="tg-ring-item" style="--i:${((a - 22.5) / 45 + 0.5).toFixed(1)}"
       transform="rotate(${a} 65 65)">
      <circle cx="65" cy="15" r="3.5" fill="#8e1f31"></circle>
    </g>`;
  const items = [];
  for (let i = 0; i < 8; i += 1) {
    items.push(acorn(i * 45));
    items.push(berry(i * 45 + 22.5));
  }
  return `
    <svg class="tg-acorn-ring" viewBox="0 0 130 130" aria-hidden="true" focusable="false">
      <circle class="tg-ring-circle" cx="65" cy="65" r="50" fill="none" stroke="#8e3b2a"
              stroke-width="1.6"></circle>
      ${items.join("")}
    </svg>`;
})();

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
     ${GEESE}`
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

  /* Every circular speaker photograph gets the acorn-and-berry ring, on hover. */
  decorate(
    disposer,
    ".speaker-directory-photo, .seminar-speaker-photo",
    "season-ring tg-ring",
    ACORN_RING
  );

  /* Speaker and talk cards get leaves sliding onto their top edge on hover. */
  decorate(
    disposer,
    ".speaker-directory-card, .feature-card, .talk-card",
    "season-frame tg-frame tg-leaf-frame",
    cardLeaves(87)
  );

  /* Community cards get the artboard's leaf-scatter border. */
  decorate(disposer, ".community-card", "season-frame tg-frame tg-scatter-frame", leafScatter(61));

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
