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
  Loop,
  Surface,
  buildParticles,
  clamp,
  make,
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

/* Bare tree by recursive branching. Each segment is a round-capped stroke that
   thins with depth, so the trunk tapers into twigs without any per-branch
   bookkeeping. */
const bareTree = (seed, { W = 200, H = 240, trunk = 9, depth = 6 } = {}) => {
  const rnd = seeded(seed);
  const segs = [];
  const branch = (x, y, ang, len, w, d) => {
    const x2 = x + Math.cos(ang) * len;
    const y2 = y + Math.sin(ang) * len;
    segs.push({ x, y, x2, y2, w });
    if (d === 0 || len < 5) {
      return;
    }
    const n = 2 + (rnd() > 0.55 ? 1 : 0);
    for (let i = 0; i < n; i += 1) {
      branch(
        x2,
        y2,
        ang + (rnd() - 0.5) * 1.2 + (i - (n - 1) / 2) * 0.55,
        len * (0.6 + rnd() * 0.22),
        w * 0.62,
        d - 1
      );
    }
  };
  branch(W / 2, H, -Math.PI / 2 + (rnd() - 0.5) * 0.3, H * 0.3, trunk, depth);
  return `
    <svg viewBox="0 0 ${W} ${H}" aria-hidden="true" focusable="false" preserveAspectRatio="none">
      <g stroke="${NIGHT}" stroke-linecap="round" fill="none">
        ${segs
          .map(
            (s) =>
              `<path d="M${s.x.toFixed(1)} ${s.y.toFixed(1)}L${s.x2.toFixed(1)} ${s.y2.toFixed(
                1
              )}" stroke-width="${Math.max(0.8, s.w).toFixed(1)}"/>`
          )
          .join("")}
      </g>
    </svg>`;
};

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

/* Drip edge for the bottom of the hero.
 *
 * Same trick as Winter's snow bank: one opaque shape in the exact white that
 * `.section-overview` starts with. Here its *top* edge is lobed, so what you
 * read is the night above dripping down into the section below. Drawing the
 * drips themselves in night colour would need them to escape the hero, which
 * clips its overflow. */
const dripEdge = (seed, height) => {
  const W = 1200;
  const H = 56;
  const rand = seeded(seed);
  const baseline = 10;
  let d = `M0 ${H}V${baseline}`;
  let x = 0;
  while (x < W) {
    const gap = range(rand, 8, 40);
    /* Wide and shallow. Narrow deep lobes pinched off at the neck and read as
       a row of dots sitting under the edge instead of as drips coming off it,
       so width is kept well above depth. */
    const w = range(rand, 26, 64);
    const depth = range(rand, 6, 30);
    const start = Math.min(W, x + gap);
    const end = Math.min(W, start + w);
    if (start >= W) {
      break;
    }
    d += `H${start.toFixed(1)}`;
    /* Control points pulled inboard so the lobe leaves the baseline on a slope
       rather than a vertical wall: a hanging teardrop, not a circle. */
    d += `C${(start + w * 0.22).toFixed(1)} ${(baseline + depth).toFixed(1)} ${(
      end -
      w * 0.22
    ).toFixed(1)} ${(baseline + depth).toFixed(1)} ${end.toFixed(1)} ${baseline}`;
    x = end;
  }
  d += `H${W}V${H}Z`;
  return `
    <svg class="hw-drip" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         aria-hidden="true" focusable="false" style="height:${height}px">
      <path fill="#ffffff" d="${d}"></path>
    </svg>`;
};

/* Graveyard edge: ground line, leaning headstones, the occasional cross. */
const graveyard = (seed) => {
  const rand = seeded(seed);
  const W = 1200;
  const H = 96;
  const n = 11;
  const parts = [
    `<path d="M0 ${H}V${H - 12}C${W * 0.2} ${H - 18} ${W * 0.4} ${H - 8} ${W * 0.6} ${H - 14}S${
      W * 0.9
    } ${H - 6} ${W} ${H - 12}V${H}Z"/>`
  ];
  for (let i = 0; i < n; i += 1) {
    const x = ((i + 0.5) * W) / n + (rand() - 0.5) * (W / n) * 0.7;
    const w = range(rand, 16, 34);
    const h = range(rand, 22, 52);
    const tilt = (rand() - 0.5) * 10;
    const base = H - 10;
    const g = `transform="rotate(${tilt.toFixed(1)} ${x.toFixed(1)} ${base})"`;
    if (rand() > 0.72) {
      parts.push(
        `<g ${g}><rect x="${(x - 3).toFixed(1)}" y="${(base - h).toFixed(
          1
        )}" width="6" height="${h.toFixed(1)}"/><rect x="${(x - w / 2).toFixed(1)}" y="${(
          base -
          h +
          9
        ).toFixed(1)}" width="${w.toFixed(1)}" height="6"/></g>`
      );
    } else {
      parts.push(
        `<g ${g}><path d="M${(x - w / 2).toFixed(1)} ${base}V${(base - h + w / 2).toFixed(
          1
        )}A${(w / 2).toFixed(1)} ${(w / 2).toFixed(1)} 0 0 1 ${(x + w / 2).toFixed(1)} ${(
          base -
          h +
          w / 2
        ).toFixed(1)}V${base}Z"/></g>`
      );
    }
  }
  return `
    <svg class="hw-graveyard" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
         aria-hidden="true" focusable="false">
      <g fill="${NIGHT}">${parts.join("")}</g>
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

const MOON = `
  <div class="hw-moon" data-hw-roost>
    <div class="hw-moon-glow"></div>
    <svg viewBox="0 0 100 100" aria-hidden="true" focusable="false">
      <!-- Crescent by subtraction: a full disc with a second disc punched out
           of it, which keeps the terminator a true circular arc. -->
      <defs>
        <mask id="hw-crescent">
          <rect width="100" height="100" fill="black"></rect>
          <circle cx="50" cy="50" r="34" fill="white"></circle>
          <circle cx="66" cy="40" r="30" fill="black"></circle>
        </mask>
      </defs>
      <circle cx="50" cy="50" r="34" fill="#f4e3b8" mask="url(#hw-crescent)"></circle>
    </svg>
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

/* ---------------------------------------------------------------------------
   Mount
   -------------------------------------------------------------------------- */

const scene = (disposer, selector, className, html, { first = false } = {}) => {
  const host = document.querySelector(selector);
  if (!host) {
    return null;
  }
  const node = make("div", { class: className, "aria-hidden": "true" });
  node.innerHTML = html;
  if (first) {
    host.prepend(node);
  } else {
    host.appendChild(node);
  }
  disposer.node(node);
  return node;
};

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

export const mount = ({ overlay, density, motion }) => {
  const disposer = new Disposer();

  /* --- ambient --- */

  buildEmbers(overlay, density, motion);

  const canvas = make("canvas", { class: "season-canvas hw-canvas", "aria-hidden": "true" });
  overlay.appendChild(canvas);
  const surface = new Surface(canvas);
  disposer.add(() => surface.destroy());

  /* --- chrome --- */

  scene(disposer, ".site-header", "season-scene hw-header", cobweb(96), { first: true });

  scene(
    disposer,
    ".hero",
    "season-scene hw-hero",
    `<div class="season-sky"></div>
     ${MOON}
     ${FOG}
     <div class="hw-tree hw-tree-left">${bareTree(5, { W: 200, H: 260 })}</div>
     <div class="hw-tree hw-tree-right">${bareTree(12, { W: 200, H: 300 })}</div>
     ${dripEdge(17, 56)}`
  );

  scene(
    disposer,
    ".site-footer",
    "season-scene hw-footer",
    `<div class="season-sky"></div>
     ${FOG}
     <div class="hw-tree hw-tree-footer">${bareTree(31, { W: 200, H: 220 })}</div>
     ${OWL}
     ${graveyard(9)}`
  );

  /* The swarm roosts on the hero moon while it is on screen, and on the
     top-right corner behind the header once it has scrolled away — so bursts
     always arrive from somewhere, never out of thin air. */
  const moon = document.querySelector("[data-hw-roost]");
  const roost = () => {
    if (moon) {
      const r = moon.getBoundingClientRect();
      if (r.bottom > 0 && r.top < surface.height) {
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
      }
    }
    return { x: Math.max(80, surface.width - 90), y: 46 };
  };

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

  return {
    /* Exposed for tuning and for verification: hidden documents never fire
       rAF, so this is the only way to advance the swarm off-screen. */
    swarm,
    loop,
    destroy() {
      disposer.dispose();
    }
  };
};
