/**
 * Scroll-reveal engine.
 *
 * Elements marked `data-reveal` start transparent (see motion.css) and animate
 * in the first time they cross into the viewport. Reveals are one-shot: an
 * element that re-enters on the way back up stays put, because re-running the
 * entrance every scroll pass reads as a glitch rather than as polish.
 *
 * Stagger is computed per *wave* rather than per container. Every element the
 * observer reports in a single callback is one wave, sorted into document
 * order and given an increasing delay. That means a grid scrolled into view
 * cascades, while the thirtieth row of a long table - which arrives in its own
 * later wave - animates immediately instead of inheriting a two-second delay.
 */

const MAX_STAGGER_STEPS = 8;

const prefersReducedMotion = () =>
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const supported = typeof IntersectionObserver === "function";

const reveal = (element, index = 0) => {
  if (index > 0) {
    element.style.setProperty("--reveal-index", String(index));
  }
  element.classList.add("is-revealed");
};

const revealImmediately = (elements) => {
  elements.forEach((element) => reveal(element));
};

const inDocumentOrder = (left, right) =>
  left.compareDocumentPosition(right) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;

const handleEntries = (entries, observer) => {
  const wave = entries
    .filter((entry) => entry.isIntersecting)
    .map((entry) => entry.target)
    .sort(inDocumentOrder);

  wave.forEach((element, index) => {
    const explicit = Number(element.dataset.revealIndex);
    reveal(element, Number.isFinite(explicit) && explicit > 0 ? explicit : Math.min(index, MAX_STAGGER_STEPS));
    observer.unobserve(element);
  });
};

let observer = null;

const getObserver = () => {
  if (!observer) {
    observer = new IntersectionObserver(handleEntries, {
      // Start the entrance just before the element is fully on screen, so the
      // motion has finished by the time it reaches comfortable reading height.
      rootMargin: "0px 0px -6% 0px",
      threshold: 0.04
    });
  }
  return observer;
};

/**
 * Register every unrevealed `[data-reveal]` inside `root`. Safe to call again
 * after each render pass - already-tracked elements are skipped.
 */
export const observeReveals = (root = document) => {
  const scope = root instanceof Element || root instanceof Document ? root : document;
  const elements = Array.from(scope.querySelectorAll("[data-reveal]")).filter(
    (element) => !element.classList.contains("is-revealed") && !element.dataset.revealBound
  );

  if (!elements.length) {
    return;
  }

  if (!supported || prefersReducedMotion()) {
    revealImmediately(elements);
    return;
  }

  const activeObserver = getObserver();
  elements.forEach((element) => {
    element.dataset.revealBound = "true";
    activeObserver.observe(element);
  });
};

/**
 * Fade portraits in as they decode. Images already in cache are `complete` on
 * arrival and skip straight to the loaded state, so nothing flashes.
 */
export const observeImageFades = (root = document) => {
  const scope = root instanceof Element || root instanceof Document ? root : document;
  scope.querySelectorAll("img[data-fade]:not(.is-loaded)").forEach((image) => {
    if (image.complete) {
      image.classList.add("is-loaded");
      return;
    }
    image.addEventListener("load", () => image.classList.add("is-loaded"), { once: true });
    image.addEventListener("error", () => image.classList.add("is-loaded"), { once: true });
  });
};

/** Convenience wrapper for the common "just rendered some markup" case. */
export const activateMotion = (root = document) => {
  // Tells the inline boot script's timer to stand down. Without this the
  // failsafe would fire on every page and force-reveal everything below the
  // fold, cancelling the scroll animations it exists to protect.
  document.documentElement.classList.add("motion-ready");
  observeReveals(root);
  observeImageFades(root);
};
