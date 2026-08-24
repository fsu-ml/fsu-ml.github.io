/**
 * Header chrome behaviour: condense-on-scroll and the reading-progress
 * hairline. Both read scroll position, so they share a single rAF-throttled
 * listener rather than each attaching their own.
 */

const CONDENSE_AT = 24;

export const bindHeaderChrome = () => {
  const header = document.querySelector("[data-site-header]");
  if (!header) {
    return;
  }

  const progress = header.querySelector("[data-scroll-progress]");
  let ticking = false;

  const update = () => {
    ticking = false;
    const offset = window.scrollY;
    header.classList.toggle("is-scrolled", offset > CONDENSE_AT);

    if (progress) {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = scrollable > 0 ? Math.min(1, Math.max(0, offset / scrollable)) : 0;
      progress.style.setProperty("--scroll-progress", ratio.toFixed(4));
    }
  };

  const onScroll = () => {
    if (ticking) {
      return;
    }
    ticking = true;
    requestAnimationFrame(update);
  };

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  update();
};
