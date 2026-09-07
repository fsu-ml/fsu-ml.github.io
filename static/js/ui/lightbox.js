/**
 * Flyer lightbox.
 *
 * One dialog per page, mounted lazily the first time something is opened, and
 * driven by a single delegated listener. Renderers therefore emit plain
 * `[data-flyer-set]` buttons and never have to re-bind after a render pass -
 * the same reason the season switcher delegates its trigger.
 *
 * Built on a native `<dialog>` opened with `showModal()`. That buys the focus
 * trap, Escape, background inertness and - the reason it matters here - the top
 * layer, which sits above the seasonal particle canvas without joining the
 * z-index ladder at all. A hand-rolled overlay would have to out-stack
 * `.season-layer` (15), `.render-error` (40) and `.season-switcher` (45) and
 * would still let snow drift over the poster.
 *
 * A talk may carry several flyers - typically a wide banner and a vertical
 * poster of the same event - so the dialog is a small gallery: variant pills,
 * arrow keys, and a counter. With a single flyer none of that chrome renders.
 */

import { escapeHtml } from "../utils/html.js";

const TRIGGER_SELECTOR = "[data-flyer-set]";
const ID = "flyer-lightbox";

/* Long enough for the exit keyframe in lightbox.css to finish. Only a fallback:
   `animationend` normally closes the dialog first, and under reduced motion the
   global 0.01ms override makes that fire almost immediately. */
const EXIT_FALLBACK_MS = 400;

/* Read per call rather than cached, so a preference changed mid-session is
   honoured. reveal.js keeps its own private copy of this check; it is three
   lines and exporting it would couple the motion engine to the lightbox. */
const prefersReducedMotion = () =>
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let view = null;

const parseSet = (trigger) => {
  try {
    const parsed = JSON.parse(trigger.getAttribute("data-flyer-set") || "");
    const items = Array.isArray(parsed?.items) ? parsed.items.filter((item) => item?.href) : [];
    return items.length ? { title: parsed.title || "", items } : null;
  } catch {
    return null;
  }
};

const mount = () => {
  const dialog = document.createElement("dialog");
  dialog.className = "flyer-lightbox";
  dialog.id = ID;
  dialog.innerHTML = `
    <div class="flyer-lightbox-frame">
      <button class="flyer-lightbox-close" type="button" data-close aria-label="Close flyer">
        <span aria-hidden="true">&times;</span>
      </button>
      <button class="flyer-lightbox-step is-prev" type="button" data-step="-1" aria-label="Previous flyer">
        <span aria-hidden="true">&lsaquo;</span>
      </button>
      <button class="flyer-lightbox-step is-next" type="button" data-step="1" aria-label="Next flyer">
        <span aria-hidden="true">&rsaquo;</span>
      </button>
      <figure class="flyer-lightbox-figure">
        <img class="flyer-lightbox-image" data-image alt="">
        <figcaption class="flyer-lightbox-caption">
          <span class="flyer-lightbox-title" data-title></span>
          <span class="flyer-lightbox-meta" data-meta aria-live="polite"></span>
          <span class="flyer-lightbox-actions">
            <a class="flyer-lightbox-download" data-download target="_blank" rel="noopener">Open full size</a>
          </span>
        </figcaption>
      </figure>
      <div class="flyer-lightbox-variants" data-variants></div>
    </div>
  `;
  document.body.appendChild(dialog);
  return dialog;
};

const build = () => {
  const dialog = mount();
  const image = dialog.querySelector("[data-image]");
  const titleEl = dialog.querySelector("[data-title]");
  const metaEl = dialog.querySelector("[data-meta]");
  const download = dialog.querySelector("[data-download]");
  const variants = dialog.querySelector("[data-variants]");
  const steps = Array.from(dialog.querySelectorAll("[data-step]"));

  let items = [];
  let title = "";
  let index = 0;
  let closing = false;
  /* Bumped by every open and every close. A close schedules two ways to finish
     (animationend and a timer) and either may land after the dialog has already
     been reopened - without this the stale one would shut the new flyer. */
  let generation = 0;

  /* Fading the swap rather than blanking it keeps the frame from flashing
     white between two large posters. Cached images are `complete` on arrival
     and skip straight to the loaded state. */
  const showImage = (item) => {
    image.classList.remove("is-loaded");
    image.src = item.href;
    image.alt = item.alt || item.label || "Flyer";
    if (image.complete) {
      image.classList.add("is-loaded");
    }
  };

  const render = () => {
    const item = items[index];
    if (!item) {
      return;
    }

    showImage(item);
    titleEl.textContent = title;
    metaEl.textContent =
      items.length > 1 ? `${item.label} — ${index + 1} of ${items.length}` : item.label;
    download.href = item.href;

    const many = items.length > 1;
    steps.forEach((step) => {
      step.hidden = !many;
    });
    variants.hidden = !many;
    if (!many) {
      // Cleared rather than just hidden, so a single-flyer talk opened after a
      // multi-flyer one cannot leave the previous talk's pills in the DOM.
      variants.innerHTML = "";
    } else {
      variants.innerHTML = items
        .map(
          (entry, position) =>
            `<button class="flyer-lightbox-variant${
              position === index ? " is-current" : ""
            }" type="button" data-go="${position}"${
              position === index ? ' aria-current="true"' : ""
            }>${escapeHtml(entry.label)}</button>`
        )
        .join("");
    }

    /* The next poster is a few hundred KB; fetching it now makes the arrow
       keys feel instant instead of showing the fade on every step. */
    const next = items[(index + 1) % items.length];
    if (next && next !== item) {
      new Image().src = next.href;
    }
  };

  const go = (position) => {
    if (items.length < 2) {
      return;
    }
    index = (position + items.length) % items.length;
    render();
  };

  const close = () => {
    if (closing || !dialog.open) {
      return;
    }
    closing = true;
    const token = (generation += 1);
    dialog.classList.add("is-closing");

    const finish = () => {
      if (token !== generation) {
        return;
      }
      dialog.classList.remove("is-closing");
      closing = false;
      document.body.classList.remove("flyer-open");
      if (dialog.open) {
        dialog.close();
      }
    };

    /* Under reduced motion the global rule in motion.css collapses the exit to
       0.01ms, and the browser then fires no `animationend` at all - waiting for
       one would leave the dialog up for the whole fallback timer, which is the
       opposite of what the preference asks for. So: close now. */
    if (prefersReducedMotion()) {
      finish();
      return;
    }

    /* Whichever lands first wins; `finish` is guarded by its token. The timer
       covers a browser that never fires `animationend` for a hidden or
       interrupted animation, which would otherwise strand the page behind a
       locked scroll. */
    dialog.addEventListener("animationend", finish, { once: true });
    window.setTimeout(finish, EXIT_FALLBACK_MS);
  };

  const open = (set) => {
    // Retires any close still in flight, including the entrance animation's own
    // animationend, which would otherwise be read as that close finishing.
    generation += 1;
    closing = false;
    dialog.classList.remove("is-closing");
    items = set.items;
    title = set.title;
    index = 0;
    render();
    if (!dialog.open) {
      dialog.showModal();
      /* `showModal` does not lock the page behind it; without this the flyer
         stays put while the document scrolls underneath. */
      document.body.classList.add("flyer-open");
    }
    dialog.querySelector("[data-close]").focus();
  };

  image.addEventListener("load", () => image.classList.add("is-loaded"));
  image.addEventListener("error", () => image.classList.add("is-loaded"));

  dialog.addEventListener("click", (event) => {
    if (event.target.closest("[data-close]")) {
      close();
      return;
    }
    const step = event.target.closest("[data-step]");
    if (step) {
      go(index + Number(step.dataset.step));
      return;
    }
    const jump = event.target.closest("[data-go]");
    if (jump) {
      go(Number(jump.dataset.go));
      return;
    }
    /* Anything that is not the poster, the caption or a control is backdrop.
       `::backdrop` is not itself an event target, so the dialog element sees
       those clicks and the frame is what tells them apart. */
    if (!event.target.closest(".flyer-lightbox-frame") || event.target === dialog) {
      close();
    }
  });

  /* Escape already closes a modal dialog, but the default close skips the exit
     animation and leaves the scroll lock on, so it is handled here instead. */
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    close();
  });

  dialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      go(index + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      go(index - 1);
    }
  });

  return { open, close };
};

/**
 * Bind the delegated trigger. Safe to call on every page; the dialog itself is
 * not built until something is actually opened, so pages with no flyers pay
 * nothing.
 */
export const bindFlyerLightbox = () => {
  if (document.documentElement.dataset.flyerLightbox === "bound") {
    return;
  }
  document.documentElement.dataset.flyerLightbox = "bound";

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest?.(TRIGGER_SELECTOR);
    if (!trigger) {
      return;
    }
    const set = parseSet(trigger);
    if (!set) {
      return;
    }
    event.preventDefault();
    if (!view) {
      view = build();
    }
    view.open(set);
  });
};
