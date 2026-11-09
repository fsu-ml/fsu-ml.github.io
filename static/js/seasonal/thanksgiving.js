/**
 * Thanksgiving — November.
 *
 * The idea that separates this from Winter is the wind. One JS source writes a
 * single unitless `--wind` to the document root each frame; every consumer —
 * leaves, fog, the garland, candle flames — reads it, either through `calc()`
 * or from `wind.value` in a canvas. One variable, zero per-element JavaScript
 * for the CSS side, which is what keeps a whole page of reacting parts cheap.
 *
 * The page gets misty autumn ridges in the hero, a vine that grows along each
 * card's edges on hover, and a candlelit harvest table in the footer. There is
 * one wind, one leaf field and one rAF loop; anything else that animates per
 * frame is a "part" with `step(dt)` and `draw()` that the loop drives, and
 * parts scrolled out of view are skipped.
 */

import {
  Disposer,
  Loop,
  Surface,
  bindProgress,
  clamp,
  decorate,
  make,
  onButtonPress,
  range,
  seeded,
  smoothstep,
  trackPointer
} from "./engine.js";
import { LEAF_COLORS, LEAF_KINDS, drawLeaf, leafSvg, pickOf } from "./thanksgiving/art.js";
import { mountCards } from "./thanksgiving/cards.js";
import { mountFooter } from "./thanksgiving/footer.js";
import { mountRidgesHero } from "./thanksgiving/hero-ridges.js";

/* Tuned so the hero reads as a leaf fall at the default density: at 45 that
   is 20 leaves. */
const leafCount = (density) => Math.min(48, Math.round(density / 2.2));

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
  }

  make(fromTop) {
    const d = range(Math.random, 0.4, 1.6);
    return {
      /* Depth drives size, fall speed, opacity and how hard the wind shoves
         it, so one number gives the field its parallax. */
      d,
      size: 14 + d * 11,
      x: Math.random() * this.surface.width,
      y: fromTop ? range(Math.random, -80, -20) : range(Math.random, -this.surface.height, this.surface.height),
      kind: pickOf(Math.random, LEAF_KINDS),
      color: pickOf(Math.random, LEAF_COLORS),
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
        if (this.settled.length >= 36) {
          this.settled.shift();
        }
        this.settled.push({ x: l.x, rot: l.rot, kind: l.kind, color: l.color, size: l.size, t: this.t });
        Object.assign(l, this.make(true));
      }
    }

    for (let i = this.settled.length - 1; i >= 0; i -= 1) {
      if (this.t - this.settled[i].t > 150) {
        this.settled.splice(i, 1);
      }
    }
  }

  draw() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    const H = this.surface.height;
    for (const s of this.settled) {
      const age = this.t - s.t;
      drawLeaf(ctx, s.kind, s.x, H - s.size * 0.32, s.size, s.rot, s.color, 0.85 * Math.min(1, (150 - age) / 40));
    }
    for (const l of this.leaves) {
      const flip = Math.cos(l.ph);
      drawLeaf(ctx, l.kind, l.x, l.y, l.size, l.rot, l.color, 0.55 + l.d * 0.28, Math.abs(flip) < 0.12 ? 0.12 : flip);
    }
  }

  /** Throws n leaves out of a point, for the burst a button fires on click. */
  burst(x, y, n) {
    for (let i = 0; i < n; i += 1) {
      const leaf = this.make(true);
      const a = (i / n) * Math.PI * 2 + Math.random() * 0.6;
      const sp = range(Math.random, 2.4, 5.2);
      Object.assign(leaf, { x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 1.4 });
      this.leaves.push(leaf);
    }
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
      const size = range(rand, 14, 28);
      drawLeaf(ctx, pickOf(rand, LEAF_KINDS), rand() * W, H - size * 0.32, size, rand() * 6.28, pickOf(rand, LEAF_COLORS), 0.85);
    }
  }
}

/* ---------------------------------------------------------------------------
   Part helpers
   -------------------------------------------------------------------------- */

/**
 * Caches an element's viewport box, refreshed on scroll and resize rather
 * than read inside the frame loop: the loop writes `--wind` every frame, so a
 * read there would force a style recalculation each time.
 */
const rectTracker = (disposer) => (el) => {
  const box = { left: 0, top: 0, width: 0, height: 0, bottom: 0 };
  const read = () => {
    const r = el.getBoundingClientRect();
    box.left = r.left;
    box.top = r.top;
    box.width = r.width;
    box.height = r.height;
    box.bottom = r.bottom;
  };
  disposer.listen(window, "scroll", read, { passive: true });
  disposer.listen(window, "resize", read, { passive: true });
  const ro = new ResizeObserver(read);
  ro.observe(el);
  disposer.add(() => ro.disconnect());
  read();
  /** The pointer in this element's coordinates, or null when it is away. */
  box.local = (pointer) => {
    if (!pointer.active) {
      return null;
    }
    const x = pointer.x - box.left;
    const y = pointer.y - box.top;
    return x < -40 || y < -40 || x > box.width + 40 || y > box.height + 40 ? null : { x, y };
  };
  return box;
};

/* Parts off screen are skipped entirely: no step, no draw. */
const watchVisible = (disposer, part) => {
  if (!part.el || typeof IntersectionObserver !== "function") {
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        part.visible = entry.isIntersecting;
      }
    },
    { rootMargin: "120px" }
  );
  io.observe(part.el);
  disposer.add(() => io.disconnect());
};

/* ---------------------------------------------------------------------------
   Chrome artwork
   -------------------------------------------------------------------------- */

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
      `<span class="tg-garland-leaf" style="left:${x.toFixed(2)}%;top:${(-size * 0.42).toFixed(1)}px;transform:rotate(${rot.toFixed(
        0
      )}deg)">${leafSvg(pickOf(rand, LEAF_KINDS), pickOf(rand, LEAF_COLORS.slice(0, 4)), size.toFixed(0), { veins: false })}</span>`
    );
    if (rand() > 0.55) {
      parts.push(`<span class="tg-berry" style="left:${(x + 2).toFixed(2)}%"></span>`);
    }
  }
  return parts.join("");
};

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
  disposer.add(() => root.style.removeProperty("--wind"));

  const pointer = trackPointer(disposer);
  const leaves = new LeafField(surface, { target: leafCount(density), wind });
  leaves.pointer = pointer;
  const rect = rectTracker(disposer);
  const opts = { disposer, wind, pointer, motion, rect };

  /* Everything else that animates per frame is a part the one loop drives. */
  const parts = [];
  const add = (part) => {
    if (part) {
      parts.push(part);
      watchVisible(disposer, part);
    }
    return part;
  };

  /* Hero: misty ridges in a low sun. Behind `.hero-inner`, so the copy and
     the seminar card keep the contrast they were audited with. */
  const hero = add(mountRidgesHero(opts));
  /* Footer: the harvest table under a night sky. */
  add(mountFooter(opts));
  /* Cards: a vine grows along the edges on hover. */
  mountCards(opts);

  const loop = new Loop({
    motion,
    step: (dt) => {
      wind.step(dt);
      leaves.step(dt);
      for (const part of parts) {
        if (part.visible) {
          part.step(dt);
        }
      }
    },
    draw: () => {
      if (!motion) {
        leaves.drawStatic();
        for (const part of parts) {
          part.draw();
        }
        return;
      }
      leaves.draw();
      for (const part of parts) {
        if (part.visible) {
          part.draw();
        }
      }
    }
  });
  disposer.add(() => loop.destroy());
  surface.onChange = () => {
    if (!motion) {
      loop.draw();
    }
  };

  if (!motion) {
    /* A held breeze rather than dead calm: the fog, flames and garland read
       as caught mid-gust instead of collapsed to zero. */
    wind.settle();
  }
  loop.start();

  /* Header: a garland hanging off the bar's bottom edge. */
  decorate(disposer, ".site-header", "season-edge-strip tg-garland", garland(6));

  /* Drives the hero's sunset: the sun sinks and the sky deepens to dusk as
     the page is scrolled. One variable, read by CSS. */
  const syncProgress = bindProgress(disposer, root);

  /* Click a primary action and a handful of leaves tumble out of it. */
  onButtonPress(disposer, ".button-primary", (x, y) => leaves.burst(x, y, 9));

  /* Section seam between the overview and the dashboard. The adjacent-sibling
     selector keeps it off the subpages, which reuse `.section-dashboard` as
     their only section. */
  decorate(disposer, ".section-overview + .section-dashboard", "season-divider tg-divider", DIVIDER, { first: true });

  return {
    /* Exposed for tuning and for verification: hidden documents never fire
       rAF, so this is the only way to advance the engine off-screen. */
    wind,
    leaves,
    hero,
    parts,
    loop,
    syncProgress,
    destroy() {
      disposer.dispose();
    }
  };
};
