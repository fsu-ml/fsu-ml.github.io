/**
 * Seasonal theme panel.
 *
 * An easter egg rather than a visible control: nothing on the page advertises
 * it. The panel slides in from the right edge when something marked
 * `data-season-trigger` is activated — today that is the department wordmark
 * in the footer — and closes on its own button, Escape, or a click outside.
 *
 * It writes through the SeasonController, so it holds no state of its own and
 * stays correct if the season is changed from the console or by a `?season=`
 * link.
 *
 * Built from native controls on purpose. A fieldset of radios gives arrow-key
 * navigation, grouping and announcement for free; hand-rolled `role="radio"`
 * buttons would need all three reimplemented and would still be worse.
 */

import { DENSITY_MAX, DENSITY_MIN, SEASONS } from "../seasonal/season.js";

const ID = "season-switcher";
const TRIGGER_SELECTOR = "[data-season-trigger]";

export const mountSeasonSwitcher = (controller) => {
  if (document.getElementById(ID)) {
    return null;
  }

  const root = document.createElement("div");
  root.className = "season-switcher";
  root.id = ID;

  const panelId = `${ID}-panel`;
  const titleId = `${ID}-title`;
  root.innerHTML = `
    <div
      class="season-switcher-panel"
      id="${panelId}"
      role="dialog"
      aria-modal="false"
      aria-labelledby="${titleId}"
      hidden
    >
      <div class="season-switcher-head">
        <h2 class="season-switcher-title" id="${titleId}">Seasonal theme</h2>
        <button class="season-switcher-close" type="button" data-close aria-label="Close theme switcher">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>

      <fieldset class="season-switcher-group">
        <legend class="sr-only">Seasonal theme</legend>
        ${SEASONS.map(
          (season) => `
          <label class="season-switcher-option">
            <input type="radio" name="season-choice" value="${season.id}">
            <span class="season-switcher-swatch" style="--swatch:${season.swatch}" aria-hidden="true"></span>
            <span class="season-switcher-text">
              <span class="season-switcher-label">${season.label}</span>
              <span class="season-switcher-note">${season.note}</span>
            </span>
          </label>`
        ).join("")}
      </fieldset>

      <div class="season-switcher-control">
        <label for="${ID}-density">Density <output for="${ID}-density" data-density-output></output></label>
        <input
          id="${ID}-density"
          type="range"
          min="${DENSITY_MIN}"
          max="${DENSITY_MAX}"
          step="5"
          data-density
        >
      </div>

      <div class="season-switcher-control season-switcher-check">
        <input id="${ID}-motion" type="checkbox" data-motion>
        <label for="${ID}-motion">Motion</label>
      </div>

      <p class="season-switcher-hint" data-hint></p>

      <button class="season-switcher-auto" type="button" data-auto>Use today&rsquo;s date</button>
    </div>
  `;

  document.body.appendChild(root);

  const panel = root.querySelector(".season-switcher-panel");
  const closeButton = root.querySelector("[data-close]");
  const radios = Array.from(root.querySelectorAll('input[name="season-choice"]'));
  const density = root.querySelector("[data-density]");
  const densityOut = root.querySelector("[data-density-output]");
  const motion = root.querySelector("[data-motion]");
  const auto = root.querySelector("[data-auto]");
  const hint = root.querySelector("[data-hint]");

  /* ---- open / close ---- */

  /* Remembered so Escape and the close button hand focus back to whatever
     opened the panel, which may be a different trigger on each page. */
  let opener = null;

  const markTriggers = (open) => {
    document.querySelectorAll(TRIGGER_SELECTOR).forEach((trigger) => {
      trigger.setAttribute("aria-expanded", String(open));
    });
  };

  const isOpen = () => !panel.hidden;

  const open = (trigger = null) => {
    if (isOpen()) {
      return;
    }
    opener = trigger;
    panel.hidden = false;
    root.classList.add("is-open");
    markTriggers(true);
    const checked = radios.find((radio) => radio.checked) || radios[0];
    checked.focus();
  };

  const close = ({ restoreFocus = false } = {}) => {
    if (!isOpen()) {
      return;
    }
    panel.hidden = true;
    root.classList.remove("is-open");
    markTriggers(false);
    if (restoreFocus && opener && document.contains(opener)) {
      opener.focus();
    }
    opener = null;
  };

  closeButton.addEventListener("click", () => close({ restoreFocus: true }));

  /* Delegated so triggers can be rendered before or after this mount, and so a
     re-rendered footer keeps working without re-binding. */
  document.addEventListener("click", (event) => {
    const trigger = event.target.closest?.(TRIGGER_SELECTOR);
    if (!trigger) {
      return;
    }
    event.preventDefault();
    if (isOpen()) {
      close({ restoreFocus: true });
    } else {
      open(trigger);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isOpen()) {
      event.stopPropagation();
      close({ restoreFocus: true });
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (isOpen() && !root.contains(event.target) && !event.target.closest?.(TRIGGER_SELECTOR)) {
      close();
    }
  });

  /* Closing on focus loss keeps the panel from lingering behind the page once
     the user tabs back into content. */
  document.addEventListener("focusin", (event) => {
    if (isOpen() && !root.contains(event.target) && !event.target.closest?.(TRIGGER_SELECTOR)) {
      close();
    }
  });

  /* ---- writes ---- */

  radios.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.checked) {
        controller.setSeason(radio.value);
      }
    });
  });

  /* `input` rather than `change` so dragging the slider previews live; each
     step remounts the engine, which is cheap because particle layout is
     seeded and the whole layer is one subtree. */
  density.addEventListener("input", () => {
    densityOut.textContent = density.value;
    controller.setDensity(density.value);
  });

  motion.addEventListener("change", () => controller.setMotion(motion.checked));
  auto.addEventListener("click", () => controller.setAuto());

  /* ---- reads ---- */

  controller.subscribe((state) => {
    radios.forEach((radio) => {
      radio.checked = radio.value === state.season;
    });
    if (document.activeElement !== density) {
      density.value = String(state.density);
    }
    densityOut.textContent = String(state.density);
    motion.checked = state.motion;

    auto.hidden = state.auto;

    /* The OS preference overrides the checkbox, so say so rather than letting
       the control look broken when motion is on but nothing moves. */
    if (state.motion && !state.motionEnabled) {
      hint.textContent = "Your system asks for reduced motion, so the layer is drawn static.";
      hint.hidden = false;
    } else if (state.auto && state.season !== "off") {
      hint.textContent = "Chosen from today’s date.";
      hint.hidden = false;
    } else if (state.auto) {
      hint.textContent = "No season for today’s date.";
      hint.hidden = false;
    } else {
      hint.hidden = true;
    }
  });

  return { root, open, close, isOpen };
};
