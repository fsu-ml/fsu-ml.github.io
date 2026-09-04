/**
 * Floating seasonal switcher.
 *
 * A review tool: a tab pinned to the right edge that opens a small panel for
 * flipping between the three holiday themes and tuning the two knobs each of
 * them takes. It writes through the SeasonController, so it holds no state of
 * its own and stays correct if the season is changed from the console or by a
 * `?season=` link.
 *
 * Built from native controls on purpose. A fieldset of radios gives arrow-key
 * navigation, grouping and announcement for free; hand-rolled `role="radio"`
 * buttons would need all three reimplemented and would still be worse.
 */

import { DENSITY_MAX, DENSITY_MIN, SEASONS } from "../seasonal/season.js";

const ID = "season-switcher";

export const mountSeasonSwitcher = (controller) => {
  if (document.getElementById(ID)) {
    return null;
  }

  const root = document.createElement("div");
  root.className = "season-switcher";
  root.id = ID;

  const panelId = `${ID}-panel`;
  root.innerHTML = `
    <button class="season-switcher-toggle" type="button" aria-expanded="false" aria-controls="${panelId}">
      <span class="season-switcher-dot" aria-hidden="true"></span>
      <span class="season-switcher-toggle-text">Season</span>
    </button>
    <div class="season-switcher-panel" id="${panelId}" hidden>
      <fieldset class="season-switcher-group">
        <legend>Seasonal theme</legend>
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

  const toggle = root.querySelector(".season-switcher-toggle");
  const panel = root.querySelector(".season-switcher-panel");
  const radios = Array.from(root.querySelectorAll('input[name="season-choice"]'));
  const density = root.querySelector("[data-density]");
  const densityOut = root.querySelector("[data-density-output]");
  const motion = root.querySelector("[data-motion]");
  const auto = root.querySelector("[data-auto]");
  const hint = root.querySelector("[data-hint]");
  const dot = root.querySelector(".season-switcher-dot");

  /* ---- open / close ---- */

  const setOpen = (open) => {
    toggle.setAttribute("aria-expanded", String(open));
    panel.hidden = !open;
    root.classList.toggle("is-open", open);
  };

  const close = ({ restoreFocus = false } = {}) => {
    if (panel.hidden) {
      return;
    }
    setOpen(false);
    if (restoreFocus) {
      toggle.focus();
    }
  };

  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") === "true";
    setOpen(!open);
    if (!open) {
      const checked = radios.find((radio) => radio.checked) || radios[0];
      checked.focus();
    }
  });

  root.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) {
      event.stopPropagation();
      close({ restoreFocus: true });
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (!panel.hidden && !root.contains(event.target)) {
      close();
    }
  });

  /* Closing on focus loss keeps the panel from lingering behind the page once
     the user tabs back into content. */
  document.addEventListener("focusin", (event) => {
    if (!panel.hidden && !root.contains(event.target)) {
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

    const active = SEASONS.find((season) => season.id === state.season);
    dot.style.setProperty("--swatch", active ? active.swatch : "#5f6673");
    toggle.setAttribute(
      "aria-label",
      `Seasonal theme: ${active ? active.label : "Off"}. Open theme switcher`
    );

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

  return root;
};
