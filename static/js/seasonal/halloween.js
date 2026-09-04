/**
 * Halloween — October.
 *
 * The centrepiece is the bat swarm: bats do not fall, they flock. It is the
 * one canvas engine in the layer, and it runs on the shared `Surface` and
 * `Loop` so it inherits DPR handling, the zero-size guard, the hidden-document
 * pause and hand-steppable frames rather than reimplementing them.
 *
 * Everything else is Tier A — fog, embers and the chrome decorations are CSS
 * keyframes on seeded, built-once elements.
 */

import {
  Disposer,
  decorate,
  Loop,
  Surface,
  buildParticles,
  bindProgress,
  clamp,
  make,
  onButtonPress,
  range,
  rangeInt,
  seeded,
  trackPointer
} from "./engine.js";

const NIGHT = "#0e0509";

/* Bats per three density points, so the switcher's 0-120 maps onto 0-40 —
   the same budget every theme works to. */
const batTarget = (density) => Math.min(40, Math.round(density / 3));

/* ---------------------------------------------------------------------------
   Bat swarm
   ---------------------------------------------------------------------------
   Classic boids, tuned toward "swarm" rather than "flock": separation,
   alignment and cohesion over a 90px neighbourhood, plus a wandering attractor
   that carries the whole group around the page, plus cursor repulsion.

   Neighbour search is brute-force O(n^2). At forty bats that is 1600 distance
   checks a frame, which costs nothing; a grid would be more code and slower at
   this size.
   -------------------------------------------------------------------------- */

class BatSwarm {
  constructor(surface, { target, roost }) {
    this.surface = surface;
    this.target = target;
    this.roostFn = roost;
    this.bats = [];
    this.pointer = null;
    this.t = 0;
    /* Seconds until the first burst. Short, so the page is not empty while
       someone is still looking at the hero. */
    this.nextBurst = 1.2;
    /* Counts down while the swarm is being thinned back to target. */
    this.recall = 0;
    this.noiseSeed = Math.random() * 100;
  }

  get width() {
    return this.surface.width;
  }

  get height() {
    return this.surface.height;
  }

  /** Spawn n bats at (x, y), fanned around the heading toward mid-viewport. */
  release(x, y, n) {
    const base = Math.atan2(this.height * 0.5 - y, this.width * 0.5 - x);
    for (let i = 0; i < n; i += 1) {
      /* Size doubles as depth: bigger bats are nearer, so they also fly
         faster and draw more opaque. */
      const s = 0.6 + Math.random() * 0.8;
      const a = base + (Math.random() - 0.5) * 1.6;
      const sp = 2.4 * s;
      this.bats.push({
        x,
        y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        s,
        ph: Math.random() * 6.28,
        a: 0,
        dying: false
      });
    }
  }

  /**
   * Wandering attractor: a slow Lissajous figure across the viewport plus
   * low-frequency noise. This is what makes the swarm roam the page as a unit
   * instead of dispersing or orbiting one fixed point.
   */
  attractor() {
    const { t, noiseSeed: n } = this;
    const W = this.width;
    const H = this.height;
    const nx = Math.sin(0.23 * t + n) * 0.08 + Math.sin(0.41 * t + n * 2) * 0.04;
    const ny = Math.cos(0.19 * t + n) * 0.08 + Math.sin(0.37 * t + n * 3) * 0.04;
    return {
      x: W * (0.5 + 0.35 * Math.sin(0.11 * t) + nx),
      y: H * (0.36 + 0.25 * Math.sin(0.07 * t + 1.3) + ny)
    };
  }

  step(dt) {
    if (!this.surface.ready) {
      return;
    }
    this.t += dt;
    /* Forces are authored against a 60fps frame; k rescales them so the
       simulation behaves the same on a 144Hz display. */
    const k = dt * 60;
    const bats = this.bats;
    const W = this.width;
    const H = this.height;
    const roost = this.roostFn();

    this.nextBurst -= dt;
    if (this.nextBurst <= 0) {
      if (bats.length < this.target * 1.3) {
        this.release(roost.x, roost.y, rangeInt(Math.random, 5, 12));
      }
      this.nextBurst = range(Math.random, 12, 30);
    }
    if (bats.length > this.target && this.recall <= 0) {
      this.recall = 6;
    }
    if (this.recall > 0) {
      this.recall -= dt;
      if (bats.length <= this.target) {
        this.recall = 0;
      }
    }

    /* While thinning, the attractor becomes the roost: the swarm heads home
       and bats that reach it fade out. Population stays bounded without bats
       ever popping out of existence mid-air. */
    const att = this.recall > 0 ? roost : this.attractor();
    const P = this.pointer && this.pointer.active ? this.pointer : null;

    for (let i = 0; i < bats.length; i += 1) {
      const b = bats[i];
      let sx = 0;
      let sy = 0;
      let ax = 0;
      let ay = 0;
      let cx = 0;
      let cy = 0;
      let n = 0;

      for (let j = 0; j < bats.length; j += 1) {
        if (i === j) {
          continue;
        }
        const o = bats[j];
        const dx = b.x - o.x;
        const dy = b.y - o.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > 8100) {
          continue;
        }
        const d = Math.sqrt(d2) || 0.001;
        const sepR = 28 * b.s;
        if (d < sepR) {
          const f = (1 - d / sepR) / d;
          sx += dx * f;
          sy += dy * f;
        }
        ax += o.vx;
        ay += o.vy;
        cx += o.x;
        cy += o.y;
        n += 1;
      }

      let fx = sx * 1.2;
      let fy = sy * 1.2;
      if (n) {
        fx += (ax / n - b.vx) * 0.05 + (cx / n - b.x) * 0.004;
        fy += (ay / n - b.vy) * 0.05 + (cy / n - b.y) * 0.004;
      }

      const tx = att.x - b.x;
      const ty = att.y - b.y;
      const td = Math.hypot(tx, ty) || 1;
      /* Pull grows with distance but caps, so a far bat hurries home and a
         near one does not orbit tightly. */
      const pull = Math.min(0.09, td * 0.0009);
      fx += (tx / td) * pull;
      fy += (ty / td) * pull;

      if (P) {
        const dx = b.x - P.x;
        const dy = b.y - P.y;
        const d = Math.hypot(dx, dy);
        if (d < 140 && d > 0.01) {
          const f = (2.6 * (1 - d / 140)) / d;
          fx += dx * f;
          fy += dy * f;
        }
      }

      /* A little rotational jitter keeps paths from looking computed.
         Both components come out of temporaries: writing b.vx first and then
         reading it back for b.vy is not a rotation, it is a shear, and the
         swarm slowly winds itself into a spiral. */
      const jit = (Math.random() - 0.5) * 0.12;
      const jc = Math.cos(jit);
      const js = Math.sin(jit);
      const nvx = b.vx * jc - b.vy * js;
      const nvy = b.vx * js + b.vy * jc;
      b.vx = nvx + fx * k;
      b.vy = nvy + fy * k;

      if (b.x < 60) b.vx += 0.12 * k;
      if (b.x > W - 60) b.vx -= 0.12 * k;
      if (b.y < 40) b.vy += 0.12 * k;
      if (b.y > H - 60) b.vy -= 0.12 * k;

      const sp = Math.hypot(b.vx, b.vy) || 0.001;
      const lo = 1.2 * b.s;
      const hi = 3.4 * b.s;
      const cl = sp < lo ? lo / sp : sp > hi ? hi / sp : 1;
      b.vx *= cl;
      b.vy *= cl;
      b.x += b.vx * k;
      b.y += b.vy * k;

      /* Wings beat faster the faster the bat is moving, so a bat accelerating
         out of the roost flaps and a cruising one glides. */
      b.ph += (0.16 + 0.07 * sp) * k;

      if (this.recall > 0 && !b.dying && Math.hypot(b.x - roost.x, b.y - roost.y) < 60) {
        b.dying = true;
      }
      b.a = b.dying ? b.a - dt * 1.6 : Math.min(1, b.a + dt * 2);
    }

    for (let i = bats.length - 1; i >= 0; i -= 1) {
      if (bats[i].a <= 0) {
        bats.splice(i, 1);
      }
    }
  }

  /**
   * One bat, in a local frame where heading is +x and the wings extend to ±y.
   * The leading edge sweeps forward to a point; the trailing edge has three
   * concave scallops, which is the membrane between the fingers and the thing
   * that reads as "bat" rather than "bird" at fifteen pixels.
   */
  drawBat(ctx, b, heading, wing) {
    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(heading);
    ctx.scale(b.s * 1.1, b.s * 1.1);
    ctx.globalAlpha = (0.35 + (0.5 * (b.s - 0.6)) / 0.8) * clamp(b.a, 0, 1);
    ctx.fillStyle = "#2a1118";
    for (const side of [1, -1]) {
      ctx.save();
      ctx.scale(1, side * wing);
      ctx.beginPath();
      ctx.moveTo(1.5, 1.5);
      ctx.lineTo(5, 7);
      ctx.lineTo(7.5, 15.5);
      ctx.quadraticCurveTo(3.4, 11, 1, 11.2);
      ctx.quadraticCurveTo(-0.8, 7.3, -3.2, 7.6);
      ctx.quadraticCurveTo(-2.4, 3.7, -3.8, 2.6);
      ctx.lineTo(-5.5, 0);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    ctx.beginPath();
    ctx.ellipse(-0.5, 0, 5, 1.9, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(4.2, 0, 1.8, 0, Math.PI * 2);
    ctx.fill();
    [1, -1].forEach((s) => {
      ctx.beginPath();
      ctx.moveTo(4.4, 1.1 * s);
      ctx.lineTo(6.6, 2.8 * s);
      ctx.lineTo(4.6, 3 * s);
      ctx.closePath();
      ctx.fill();
    });
    ctx.beginPath();
    ctx.moveTo(-5, 0.9);
    ctx.lineTo(-7.5, 0);
    ctx.lineTo(-5, -0.9);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  draw() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    for (const b of this.bats) {
      /* Wingspan foreshortens through the beat: 1 fully spread, 0.27 folded. */
      const wing = 0.65 + 0.38 * Math.sin(b.ph);
      this.drawBat(ctx, b, Math.atan2(b.vy, b.vx), wing);
    }
  }

  /** Reduced motion: three bats hanging at the roost, wings folded. */
  drawStatic() {
    const ctx = this.surface.begin();
    if (!ctx) {
      return;
    }
    const r = this.roostFn();
    [
      [-26, 8, 1],
      [0, 16, 0.85],
      [24, 6, 1.15]
    ].forEach(([dx, dy, s]) => {
      this.drawBat(ctx, { x: r.x + dx, y: r.y + dy, s, a: 1 }, Math.PI / 2, 0.32);
    });
  }
}

/* ---------------------------------------------------------------------------
   Artwork
   -------------------------------------------------------------------------- */

/* Cobweb: spokes from a corner, plus rings that sag toward it.
   `pathLength="1"` lets one dash animation draw every path regardless of its
   real length, so no path has to be measured. */
const cobweb = (size, { spokes = 7, rings = 5, stroke = "rgba(247,242,232,.5)" } = {}) => {
  const ox = 0;
  const oy = 0;
  const paths = [];
  const ang = [];
  for (let i = 0; i < spokes; i += 1) {
    ang.push((Math.PI / 2) * (i / (spokes - 1)));
  }
  const pt = (a, r) => [ox + Math.cos(a) * r, oy + Math.sin(a) * r];
  ang.forEach((a, i) => {
    const [x, y] = pt(a, size);
    paths.push({ d: `M${ox} ${oy}L${x.toFixed(1)} ${y.toFixed(1)}`, delay: i * 0.05 });
  });
  for (let k = 1; k <= rings; k += 1) {
    const r = size * (k / rings);
    const pts = ang.map((a) => pt(a, r));
    let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; i += 1) {
      const [x1, y1] = pts[i - 1];
      const [x2, y2] = pts[i];
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      d += `Q${(ox + (mx - ox) * 0.86).toFixed(1)} ${(oy + (my - oy) * 0.86).toFixed(
        1
      )} ${x2.toFixed(1)} ${y2.toFixed(1)}`;
    }
    paths.push({ d, delay: 0.3 + k * 0.14 });
  }
  return `
    <svg class="hw-web" viewBox="0 0 ${size} ${size}" aria-hidden="true" focusable="false">
      <g stroke="${stroke}" stroke-width="1" fill="none" stroke-linecap="round">
        ${paths
          .map((p) => `<path d="${p.d}" pathLength="1" style="--draw-delay:${p.delay}s"/>`)
          .join("")}
      </g>
    </svg>`;
};

/* A tangle of interconnected bare branches for the footer.
 *
 * Several trees are grown from points along the bottom with a wide branching
 * angle so their canopies genuinely interleave, and then tips from different
 * trees are joined by thin connectors. The interlock is what makes it gloomy —
 * a row of separate trees just reads as scenery.
 *
 * Segments are bucketed by stroke width and emitted as a handful of multi-part
 * paths rather than one element each: depth 5 across seven trees is on the
 * order of 1500 segments, which is nothing as geometry and far too much as DOM.
 */
const branchTangle = (seed, { W = 1200, H = 210, trees = 7 } = {}) => {
  const rnd = seeded(seed);
  const buckets = new Map();
  const tips = [];

  const push = (x, y, x2, y2, w) => {
    const key = Math.max(0.8, Math.round(w * 2) / 2).toFixed(1);
    if (!buckets.has(key)) {
      buckets.set(key, []);
    }
    buckets.get(key).push(`M${x.toFixed(1)} ${y.toFixed(1)}L${x2.toFixed(1)} ${y2.toFixed(1)}`);
  };

  const branch = (x, y, ang, len, w, d) => {
    const x2 = x + Math.cos(ang) * len;
    const y2 = y + Math.sin(ang) * len;
    push(x, y, x2, y2, w);
    if (d === 0 || len < 4) {
      /* Subsampled: every tip would make the pairwise connector scan
         quadratic in the thousands for no visible gain. */
      if (rnd() > 0.82) {
        tips.push([x2, y2]);
      }
      return;
    }
    const n = 2 + (rnd() > 0.5 ? 1 : 0);
    for (let i = 0; i < n; i += 1) {
      branch(
        x2,
        y2,
        ang + (rnd() - 0.5) * 1.5 + (i - (n - 1) / 2) * 0.72,
        len * (0.62 + rnd() * 0.22),
        w * 0.64,
        d - 1
      );
    }
  };

  for (let i = 0; i < trees; i += 1) {
    const x = ((i + 0.5) * W) / trees + (rnd() - 0.5) * (W / trees) * 0.7;
    const h = H * (0.58 + rnd() * 0.4);
    branch(x, H + 6, -Math.PI / 2 + (rnd() - 0.5) * 0.5, h * 0.34, 5 + rnd() * 4, 5);
  }

  /* Connectors: join tips that are near each other but far enough apart to be
     from different limbs. Only a fraction are taken, so the result reads as a
     tangle rather than a net. */
  const sample = tips.length > 150 ? tips.filter((_, i) => i % Math.ceil(tips.length / 150) === 0) : tips;
  const links = [];
  for (let i = 0; i < sample.length; i += 1) {
    for (let j = i + 1; j < sample.length; j += 1) {
      const dx = sample[i][0] - sample[j][0];
      const dy = sample[i][1] - sample[j][1];
      const d2 = dx * dx + dy * dy;
      if (d2 > 500 && d2 < 5200 && rnd() > 0.88) {
        const [x1, y1] = sample[i];
        const [x2, y2] = sample[j];
        /* A slight sag, so a connector looks like a branch that grew across
           rather than a straight wire. */
        links.push(
          `M${x1.toFixed(1)} ${y1.toFixed(1)}Q${((x1 + x2) / 2).toFixed(1)} ${(
            (y1 + y2) / 2 +
            6
          ).toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`
        );
      }
    }
  }

  const groups = [...buckets.entries()]
    .map(([w, ds]) => `<path stroke-width="${w}" d="${ds.join("")}"/>`)
    .join("");

  return `
    <svg class="hw-branches" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <g stroke="${NIGHT}" fill="none" stroke-linecap="round">
        ${groups}
        <g stroke-width="0.9" opacity=".75">${links.join("") && `<path d="${links.join("")}"/>`}</g>
      </g>
    </svg>`;
};

/* Fog banks. Deliberately scoped to the hero and the footer rather than laid
   across the viewport: in the fixed overlay it drifted over the light content
   sections too, where a blurred grey ellipse reads as a smudge on the page.
   A blend mode would have fixed that, but `.season-layer` is `contain: strict`
   and so blends against its own empty backdrop, not the page. */
const FOG = `
  <div class="hw-fog">
    <span class="hw-fog-a"></span>
    <span class="hw-fog-b"></span>
  </div>`;

/* Owl on a footer branch. The eyes blink on their own clocks, so the pair
   never reads as a single winking light. */
const OWL = `
  <div class="hw-owl">
    <svg viewBox="0 0 44 40" aria-hidden="true" focusable="false">
      <path fill="${NIGHT}" d="M22 2c9 0 15 7 15 16s-6 20-15 20S7 27 7 18 13 2 22 2z"></path>
      <path fill="${NIGHT}" d="M8 3l7 5-8 3zM36 3l-7 5 8 3z"></path>
      <g class="hw-owl-eyes" fill="#e07a2f">
        <circle class="hw-eye" cx="16" cy="16" r="4"></circle>
        <circle class="hw-eye hw-eye-b" cx="28" cy="16" r="4"></circle>
      </g>
    </svg>
  </div>`;

/* Section divider: a hairline rule with a spider hanging from its centre on a
   thread. Thread runs the full height so it can hang from any container. */
const SPIDER = `
  <svg viewBox="0 0 14 44" aria-hidden="true" focusable="false">
    <line x1="7" y1="0" x2="7" y2="24" stroke="rgba(206,184,136,.85)" stroke-width=".8"></line>
    <g stroke="#1a0b12" stroke-width="1" fill="none" stroke-linecap="round">
      <path d="M5 27l-4-4M9 27l4-4M5 29l-4 0M9 29l4 0M5 31l-4 3M9 31l4 3M6 33l-3 5M8 33l3 5"></path>
    </g>
    <circle cx="7" cy="27" r="2.4" fill="#1a0b12"></circle>
    <ellipse cx="7" cy="33" rx="3.2" ry="4" fill="#1a0b12"></ellipse>
  </svg>`;

const DIVIDER = `
  <span class="hw-rule"></span>
  <span class="hw-rule-mark">${SPIDER}</span>
  <span class="hw-rule"></span>`;

/* Logo treatment for the circular speaker photographs: the artboard's cobweb
   ring. It occupies one quadrant of the rim, so it never covers a face. */
const PHOTO_WEB = `
  <div class="hw-photo-web">${cobweb(64, { spokes: 6, rings: 4, stroke: "rgba(224,122,47,.75)" })}</div>`;

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

const buildEmbers = (overlay, density, motion) => {
  const count = Math.min(20, Math.round(density / 6));
  if (count === 0) {
    return;
  }
  overlay.appendChild(
    buildParticles(count, 29, (index, rand) => {
      const size = range(rand, 2, 5);
      const duration = range(rand, 14, 26);
      /* Embers are the one thing on the page that travels upward. Off, they
         hold a seeded height instead of stacking at the bottom. */
      return make("div", {
        class: `season-particle hw-ember${motion ? " hw-ember-rise" : ""}`,
        style: {
          left: `${(rand() * 100).toFixed(2)}%`,
          top: motion ? "100vh" : `${range(rand, 8, 92).toFixed(2)}vh`,
          width: `${size.toFixed(1)}px`,
          height: `${size.toFixed(1)}px`,
          "--dur": `${duration.toFixed(1)}s`,
          "--delay": `${(-rand() * duration).toFixed(1)}s`,
          "--flicker": `${range(rand, 1.4, 3.2).toFixed(1)}s`
        }
      });
    })
  );
};

export const mount = ({ overlay, density, motion, root }) => {
  const disposer = new Disposer();

  /* --- ambient --- */

  buildEmbers(overlay, density, motion);

  const canvas = make("canvas", { class: "season-canvas hw-canvas", "aria-hidden": "true" });
  overlay.appendChild(canvas);
  const surface = new Surface(canvas);
  disposer.add(() => surface.destroy());

  /* --- chrome --- */

  decorate(
    disposer,
    ".hero",
    "season-scene hw-hero",
    `<div class="season-sky"></div>
     <div class="hw-corner-web">${cobweb(150)}</div>
     ${FOG}`
  );

  decorate(
    disposer,
    ".site-footer",
    "season-scene hw-footer",
    `<div class="season-sky"></div>
     ${FOG}
     ${branchTangle(31)}
     ${OWL}`
  );

  /* The swarm roosts just behind the header's top-right corner, so bursts
     always arrive from somewhere off the edge rather than out of thin air. */
  const roost = () => ({ x: Math.max(80, surface.width - 90), y: 40 });

  const swarm = new BatSwarm(surface, { target: batTarget(density), roost });
  swarm.pointer = trackPointer(disposer);

  const loop = new Loop({
    motion,
    step: (dt) => swarm.step(dt),
    /* With motion off, `Loop.start()` paints exactly one frame and schedules
       nothing further — so this branch *is* the reduced-motion presentation:
       three bats perched at the roost rather than an empty canvas. */
    draw: () => (motion ? swarm.draw() : swarm.drawStatic())
  });
  disposer.add(() => loop.destroy());

  /* A canvas can be in the DOM before layout has run, so the first paint may
     land on a zero-size surface. Repainting whenever the surface resizes
     covers that and keeps the perched bats on the roost as the window changes
     — in the animated case the loop would have redrawn anyway. */
  surface.onChange = () => loop.draw();

  if (motion) {
    /* Seed the sky so the first frame already has bats in it rather than an
       empty page for the first twelve seconds. */
    swarm.release(roost().x, roost().y, Math.min(swarm.target, 8));
  }
  loop.start();

  /* Drives the fog, which rises as the page is scrolled. */
  const syncProgress = bindProgress(disposer, root);

  /* Click a primary action and three bats fly out of it and join the swarm. */
  onButtonPress(disposer, ".button-primary", (x, y) => swarm.release(x, y, 3));

  /* Section seam between the overview and the dashboard. */
  decorate(disposer, ".section-dashboard", "season-divider hw-divider", DIVIDER, { first: true });

  /* Every circular speaker photograph gets a web across one corner of its rim. */
  decorate(disposer, ".speaker-directory-photo, .seminar-speaker-photo", "season-ring", PHOTO_WEB);

  return {
    /* Exposed for tuning and for verification: hidden documents never fire
       rAF, so this is the only way to advance the swarm off-screen. */
    swarm,
    loop,
    syncProgress,
    destroy() {
      disposer.dispose();
    }
  };
};
