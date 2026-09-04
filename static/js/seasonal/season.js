/**
 * Seasonal layer orchestrator.
 *
 * Owns exactly one job: decide which season is active, and make sure exactly
 * one theme engine is mounted for it. Themes know nothing about each other or
 * about how they were selected.
 *
 * The layer is strictly additive. With season "off" nothing is imported,
 * nothing is injected, and no attribute is left on the document — the page is
 * identical to one built without this module.
 */

import { Disposer, make, onMotionPreferenceChange, prefersReducedMotion } from "./engine.js";

/* Flip to false to hide the floating switcher from visitors while leaving the
   date-driven selection running. The control is a review tool, not a feature. */
const SHOW_SWITCHER = true;

const STORE_SEASON = "scaiSeason";
const STORE_DENSITY = "scaiSeasonDensity";
const STORE_MOTION = "scaiSeasonMotion";

export const OFF = "off";

/** Display metadata, also consumed by the switcher. */
export const SEASONS = [
  { id: OFF, label: "Off", note: "No seasonal layer", swatch: "#5f6673" },
  { id: "halloween", label: "Halloween", note: "October · bats, fog, embers", swatch: "#e07a2f" },
  { id: "thanksgiving", label: "Thanksgiving", note: "November · wind, leaves, harvest", swatch: "#d98e3f" },
  { id: "winter", label: "Winter", note: "December · snow", swatch: "#dce9f2" }
];

const SEASON_IDS = SEASONS.map((entry) => entry.id);

/* Static specifiers, so a bundler could see them and a plain server resolves
   them without a manifest. Only the selected one is ever fetched. */
const LOADERS = {
  halloween: () => import("./halloween.js"),
  thanksgiving: () => import("./thanksgiving.js"),
  winter: () => import("./winter.js")
};

export const DENSITY_MIN = 0;
export const DENSITY_MAX = 120;
export const DENSITY_DEFAULT = 45;

/* ---------------------------------------------------------------------------
   Selection
   -------------------------------------------------------------------------- */

/**
 * Calendar default, used when the visitor has expressed no preference:
 * October is Halloween, November is Thanksgiving, and Winter runs from
 * 1 December through 5 January so it covers the break rather than stopping
 * dead on New Year's Day.
 */
export const seasonForDate = (date = new Date()) => {
  const month = date.getMonth();
  const day = date.getDate();
  if (month === 9) {
    return "halloween";
  }
  if (month === 10) {
    return "thanksgiving";
  }
  if (month === 11 || (month === 0 && day <= 5)) {
    return "winter";
  }
  return OFF;
};

/* Storage can throw outright in private modes and embedded webviews, so every
   access is guarded. A visitor who cannot persist a choice still gets a
   working switcher for the length of the page view. */
const readStore = (key) => {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
};

const writeStore = (key, value) => {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* preference simply does not persist */
  }
};

const clearStore = (key) => {
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* nothing to do */
  }
};

const isSeason = (value) => SEASON_IDS.includes(value);

/* Absent and empty both mean "no preference recorded" and must land on the
   default. Coercing them would not: `Number(null)` and `Number("")` are both a
   perfectly finite 0, which is a legitimate density and would silently pin a
   first-time visitor to no particles at all. */
const clampDensity = (value) => {
  if (value === null || value === undefined || value === "") {
    return DENSITY_DEFAULT;
  }
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return DENSITY_DEFAULT;
  }
  return Math.round(Math.min(DENSITY_MAX, Math.max(DENSITY_MIN, n)));
};

/* ---------------------------------------------------------------------------
   Controller
   -------------------------------------------------------------------------- */

export class SeasonController {
  constructor() {
    this.season = OFF;
    /** True when the current season came from the calendar, not a choice. */
    this.auto = true;
    this.density = DENSITY_DEFAULT;
    this.motion = true;
    this.engine = null;
    this.overlay = null;
    this._disposer = new Disposer();
    this._listeners = new Set();
    /* Incremented on every apply. A dynamic import that resolves after the
       season changed again compares tokens and drops its result. */
    this._token = 0;
  }

  /** Motion actually in effect: the OS preference always wins. */
  get motionEnabled() {
    return this.motion && !prefersReducedMotion();
  }

  init() {
    const params = new URLSearchParams(window.location.search);

    const fromUrl = params.get("season");
    const fromStore = readStore(STORE_SEASON);
    if (isSeason(fromUrl)) {
      this.season = fromUrl;
      this.auto = false;
    } else if (isSeason(fromStore)) {
      this.season = fromStore;
      this.auto = false;
    } else {
      this.season = seasonForDate();
      this.auto = true;
    }

    const densityParam = params.get("density");
    this.density = clampDensity(densityParam !== null ? densityParam : readStore(STORE_DENSITY));

    const motionParam = params.get("motion");
    if (motionParam !== null) {
      this.motion = motionParam !== "0" && motionParam !== "false";
    } else {
      this.motion = readStore(STORE_MOTION) !== "0";
    }

    /* A preference change mid-session remounts the engine so it swaps between
       the animated and the single-static-frame presentation live. */
    this._disposer.add(onMotionPreferenceChange(() => this._apply()));

    return this._apply();
  }

  setSeason(next) {
    const season = isSeason(next) ? next : OFF;
    if (season === this.season && !this.auto) {
      return Promise.resolve();
    }
    this.season = season;
    this.auto = false;
    writeStore(STORE_SEASON, season);
    return this._apply();
  }

  /** Drops the stored choice and returns to the calendar default. */
  setAuto() {
    clearStore(STORE_SEASON);
    this.season = seasonForDate();
    this.auto = true;
    return this._apply();
  }

  setDensity(next) {
    const density = clampDensity(next);
    if (density === this.density) {
      return Promise.resolve();
    }
    this.density = density;
    writeStore(STORE_DENSITY, String(density));
    return this._apply();
  }

  setMotion(next) {
    const motion = Boolean(next);
    if (motion === this.motion) {
      return Promise.resolve();
    }
    this.motion = motion;
    writeStore(STORE_MOTION, motion ? "1" : "0");
    return this._apply();
  }

  /** Subscribe to state changes. Returns an unsubscribe. */
  subscribe(listener) {
    this._listeners.add(listener);
    listener(this);
    return () => this._listeners.delete(listener);
  }

  _emit() {
    this._listeners.forEach((listener) => {
      try {
        listener(this);
      } catch (error) {
        console.error("[seasonal] subscriber failed", error);
      }
    });
  }

  async _apply() {
    const token = (this._token += 1);
    this._unmount();

    const root = document.documentElement;
    if (this.season === OFF) {
      /* Leave no trace: an "off" page must be indistinguishable from one
         where this module was never loaded. */
      delete root.dataset.season;
      root.style.removeProperty("--season-density");
      root.classList.remove("season-motion", "season-static");
      this._emit();
      return;
    }

    root.dataset.season = this.season;
    root.style.setProperty("--season-density", String(this.density));
    const motion = this.motionEnabled;
    root.classList.toggle("season-motion", motion);
    root.classList.toggle("season-static", !motion);
    this._emit();

    let module;
    try {
      module = await LOADERS[this.season]();
    } catch (error) {
      console.error(`[seasonal] could not load the ${this.season} theme`, error);
      return;
    }
    if (token !== this._token) {
      return;
    }

    const overlay = make("div", { class: "season-layer", "aria-hidden": "true" });
    document.body.appendChild(overlay);
    this.overlay = overlay;

    try {
      this.engine = module.mount({
        overlay,
        density: this.density,
        motion,
        root,
        document
      });
    } catch (error) {
      console.error(`[seasonal] the ${this.season} theme failed to mount`, error);
      overlay.remove();
      this.overlay = null;
    }
  }

  _unmount() {
    if (this.engine && typeof this.engine.destroy === "function") {
      try {
        this.engine.destroy();
      } catch (error) {
        console.error("[seasonal] theme teardown failed", error);
      }
    }
    this.engine = null;
    if (this.overlay) {
      this.overlay.remove();
      this.overlay = null;
    }
  }

  /** Full teardown, including the controller's own listeners. */
  destroy() {
    this._token += 1;
    this._unmount();
    this._disposer.dispose();
    this._listeners.clear();
    const root = document.documentElement;
    delete root.dataset.season;
    root.style.removeProperty("--season-density");
    root.classList.remove("season-motion", "season-static");
  }
}

/* ---------------------------------------------------------------------------
   Entry point
   -------------------------------------------------------------------------- */

/**
 * Called once per page, after the page's own chrome has rendered so that
 * themes can decorate real elements. Never rejects: a broken seasonal layer
 * must not take the page down with it.
 */
export const bindSeasons = async () => {
  const controller = new SeasonController();
  window.__season = controller;

  try {
    await controller.init();
  } catch (error) {
    console.error("[seasonal] initialisation failed", error);
  }

  if (SHOW_SWITCHER) {
    try {
      const { mountSeasonSwitcher } = await import("../ui/season-switcher.js");
      mountSeasonSwitcher(controller);
    } catch (error) {
      console.error("[seasonal] switcher failed to mount", error);
    }
  }

  return controller;
};
