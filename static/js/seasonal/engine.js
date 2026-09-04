/**
 * Shared runtime for the seasonal layer.
 *
 * Every theme engine (snow, bats, leaves) is built from these pieces so the
 * three of them behave identically where it matters: deterministic layout,
 * one rAF driver, no per-frame DOM writes, and a `destroy()` that leaves the
 * page exactly as it found it.
 *
 * The hard rule this module exists to enforce: high-frequency updates
 * (pointer, scroll, rAF) never touch the DOM tree and never call into a
 * render. They write CSS custom properties on a cached element reference, or
 * they draw to a canvas. Anything else and forty particles start costing
 * layout on every frame.
 */

/* ---------------------------------------------------------------------------
   Environment
   -------------------------------------------------------------------------- */

const REDUCE_QUERY = "(prefers-reduced-motion: reduce)";

export const prefersReducedMotion = () =>
  typeof window.matchMedia === "function" && window.matchMedia(REDUCE_QUERY).matches;

/**
 * Fires whenever the OS-level reduced-motion preference flips. Returns an
 * unsubscribe. Used by the orchestrator to remount engines in static mode
 * without a page reload.
 */
export const onMotionPreferenceChange = (handler) => {
  if (typeof window.matchMedia !== "function") {
    return () => {};
  }
  const query = window.matchMedia(REDUCE_QUERY);
  const listener = () => handler(query.matches);
  // Safari < 14 only has the deprecated form.
  if (typeof query.addEventListener === "function") {
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }
  query.addListener(listener);
  return () => query.removeListener(listener);
};

/* ---------------------------------------------------------------------------
   Determinism
   -------------------------------------------------------------------------- */

/**
 * mulberry32. Small, fast, and good enough for scattering particles.
 *
 * Every ambient layout is generated from one of these rather than
 * `Math.random`, so a given (season, density) pair always produces the same
 * arrangement. Re-mounting a theme therefore does not reshuffle the page, and
 * a screenshot taken during review matches the next one.
 */
export const seeded = (seed) => {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** Uniform sample in [min, max) from a seeded generator. */
export const range = (rand, min, max) => min + rand() * (max - min);

/** Uniform integer in [min, max]. */
export const rangeInt = (rand, min, max) => Math.floor(range(rand, min, max + 1));

/** Pick one entry from a list. */
export const pick = (rand, list) => list[Math.min(list.length - 1, Math.floor(rand() * list.length))];

/* ---------------------------------------------------------------------------
   DOM helpers
   -------------------------------------------------------------------------- */

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Terse element builder. `attrs` values that are `null`/`undefined` are
 * skipped, so callers can pass conditionals inline. A `style` value may be an
 * object, in which case entries are set as properties (custom properties
 * included, which `Object.assign(el.style, ...)` cannot do).
 */
export const make = (tag, attrs = {}, children = []) => {
  const node = document.createElement(tag);
  applyAttrs(node, attrs);
  appendAll(node, children);
  return node;
};

/** Same contract as `make`, in the SVG namespace. */
export const makeSvg = (tag, attrs = {}, children = []) => {
  const node = document.createElementNS(SVG_NS, tag);
  applyAttrs(node, attrs);
  appendAll(node, children);
  return node;
};

const applyAttrs = (node, attrs) => {
  Object.entries(attrs).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      return;
    }
    if (key === "style" && typeof value === "object") {
      setVars(node, value);
      return;
    }
    if (key === "text") {
      node.textContent = value;
      return;
    }
    node.setAttribute(key, String(value));
  });
};

const appendAll = (node, children) => {
  const list = Array.isArray(children) ? children : [children];
  list.forEach((child) => {
    if (child === null || child === undefined || child === false) {
      return;
    }
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  });
};

/**
 * Writes a map of style properties onto an element. Keys beginning `--` go
 * through `setProperty` so custom properties actually land; everything else is
 * a normal style property.
 *
 * This is the only sanctioned way for a hot path to update the page.
 */
export const setVars = (node, values) => {
  Object.entries(values).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      return;
    }
    if (key.startsWith("--")) {
      node.style.setProperty(key, String(value));
    } else {
      node.style[key] = String(value);
    }
  });
};

/**
 * Builds `count` particles once and returns them in a fragment. The caller's
 * factory receives the index and a seeded generator, so particle layout is
 * reproducible. Nothing here re-runs on resize: particles are positioned in
 * viewport-relative units and simply reflow.
 */
export const buildParticles = (count, seed, factory) => {
  const rand = seeded(seed);
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < count; i += 1) {
    const node = factory(i, rand);
    if (node) {
      fragment.appendChild(node);
    }
  }
  return fragment;
};

/* ---------------------------------------------------------------------------
   Canvas
   -------------------------------------------------------------------------- */

/**
 * A canvas that keeps itself sized to its own box in device pixels.
 *
 * Two things this guards against, both learned the hard way on the artboards
 * (see THEME-PLAN.md section 9):
 *
 * 1. A canvas can be in the DOM before layout has run, so the first read is
 *    0x0. A zero-size measurement is ignored rather than latched, and the
 *    surface reports `ready === false` until it gets a real one.
 * 2. The element can be replaced or moved between mounts, so sizing is driven
 *    by a ResizeObserver rather than a one-time read.
 */
export class Surface {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.ready = false;
    /* Called after any measurement that actually changed the size, including
       the first real one. An engine that paints a single static frame has no
       loop to notice that it drew into a zero-size canvas, so this is how it
       gets told to paint again. */
    this.onChange = null;
    this._observer = new ResizeObserver(() => this.measure());
    this._observer.observe(canvas);
    this.measure();
  }

  /** Re-reads the element box. Returns true when the size changed. */
  measure() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width <= 0 || height <= 0) {
      return false;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (width === this.width && height === this.height && dpr === this.dpr) {
      return false;
    }
    this.width = width;
    this.height = height;
    this.dpr = dpr;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ready = true;
    if (this.onChange) {
      this.onChange(this);
    }
    return true;
  }

  /** Clears and resets the transform to CSS pixels. Call once per frame. */
  begin() {
    if (!this.ready) {
      return null;
    }
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.ctx.clearRect(0, 0, this.width, this.height);
    return this.ctx;
  }

  destroy() {
    this._observer.disconnect();
    if (this.ready) {
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }
}

/* ---------------------------------------------------------------------------
   Frame loop
   -------------------------------------------------------------------------- */

/* A frame longer than this means the tab was away or the main thread stalled.
   Advancing physics by the real elapsed time would teleport every particle, so
   the step is clamped and the simulation simply loses that time. */
const MAX_STEP = 1 / 20;

/**
 * The single rAF driver. One per engine; the orchestrator guarantees only one
 * engine exists at a time.
 *
 * `step(dt)` and `draw()` are also public methods, deliberately. Hidden
 * documents never fire requestAnimationFrame, and the Claude browser pane
 * counts as hidden almost all the time, so the only way to verify a canvas
 * engine there is to advance it by hand:
 *
 *     window.__season.engine.loop.step(1 / 60);
 *     window.__season.engine.loop.draw();
 */
export class Loop {
  constructor({ step, draw, motion = true }) {
    this._step = step;
    this._draw = draw;
    this.motion = motion;
    this.running = false;
    this.frames = 0;
    this._raf = 0;
    this._last = 0;
    this._onVisibility = () => {
      if (document.hidden) {
        this._stopFrames();
      } else if (this.running) {
        this._startFrames();
      }
    };
    document.addEventListener("visibilitychange", this._onVisibility);
  }

  /** Advance the simulation. Safe to call by hand. */
  step(dt) {
    this.frames += 1;
    if (this._step) {
      this._step(Math.min(dt, MAX_STEP));
    }
  }

  /** Paint one frame. Safe to call by hand. */
  draw() {
    if (this._draw) {
      this._draw();
    }
  }

  /**
   * Starts the loop. With motion disabled this paints exactly one static frame
   * and never schedules another — that is the reduced-motion presentation, not
   * a blank page.
   */
  start() {
    if (this.running) {
      return;
    }
    this.running = true;
    if (!this.motion) {
      this.draw();
      return;
    }
    this._startFrames();
  }

  stop() {
    this.running = false;
    this._stopFrames();
  }

  _startFrames() {
    if (this._raf || !this.motion) {
      return;
    }
    this._last = performance.now();
    const tick = (now) => {
      this._raf = requestAnimationFrame(tick);
      const dt = (now - this._last) / 1000;
      this._last = now;
      this.step(dt);
      this.draw();
    };
    this._raf = requestAnimationFrame(tick);
  }

  _stopFrames() {
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = 0;
    }
  }

  destroy() {
    this.stop();
    document.removeEventListener("visibilitychange", this._onVisibility);
    this._step = null;
    this._draw = null;
  }
}

/* ---------------------------------------------------------------------------
   Listener bookkeeping
   -------------------------------------------------------------------------- */

/**
 * Collects teardown callbacks so an engine's `destroy()` is a single call
 * rather than a list of paired add/remove that drifts out of sync. Every
 * listener, timer and observer an engine creates goes through here.
 */
export class Disposer {
  constructor() {
    this._items = [];
  }

  add(fn) {
    this._items.push(fn);
    return fn;
  }

  listen(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    this.add(() => target.removeEventListener(type, handler, options));
  }

  interval(fn, ms) {
    const id = window.setInterval(fn, ms);
    this.add(() => window.clearInterval(id));
    return id;
  }

  timeout(fn, ms) {
    const id = window.setTimeout(fn, ms);
    this.add(() => window.clearTimeout(id));
    return id;
  }

  node(element) {
    this.add(() => element.remove());
    return element;
  }

  dispose() {
    // Reverse order so nodes are removed after the listeners bound to them.
    for (let i = this._items.length - 1; i >= 0; i -= 1) {
      try {
        this._items[i]();
      } catch (error) {
        console.error("[seasonal] teardown step failed", error);
      }
    }
    this._items.length = 0;
  }
}

/* ---------------------------------------------------------------------------
   Pointer
   -------------------------------------------------------------------------- */

/**
 * Tracks the pointer in viewport coordinates for engines that react to it
 * (bats scatter, leaves flee). Writes to plain fields only — consumers read
 * them during their own frame, so a fast mouse cannot cause more work than
 * one frame's worth.
 *
 * `active` goes false when the pointer leaves the window, so engines relax
 * instead of being repelled by a stale corner position.
 */
export const trackPointer = (disposer) => {
  const state = { x: 0, y: 0, active: false };
  disposer.listen(
    window,
    "pointermove",
    (event) => {
      state.x = event.clientX;
      state.y = event.clientY;
      state.active = true;
    },
    { passive: true }
  );
  disposer.listen(document, "pointerleave", () => {
    state.active = false;
  });
  return state;
};

/* ---------------------------------------------------------------------------
   Geometry
   -------------------------------------------------------------------------- */

export const clamp = (value, min, max) => (value < min ? min : value > max ? max : value);

/** Hermite smoothstep, used for gust envelopes and fades. */
export const smoothstep = (t) => {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
};

export const TAU = Math.PI * 2;
