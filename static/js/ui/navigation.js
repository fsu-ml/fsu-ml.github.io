import { qs } from "../utils/dom.js";

export const bindNavigation = () => {
  const toggle = qs(".nav-toggle");
  const links = qs("[data-nav-links]");

  const setOpen = (open) => {
    document.body.classList.toggle("nav-open", open);
    toggle.setAttribute("aria-expanded", String(open));
  };

  const isOpen = () => document.body.classList.contains("nav-open");

  toggle.addEventListener("click", () => {
    setOpen(!isOpen());
  });

  links.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      setOpen(false);
    }
  });

  // Escape closes the panel and hands focus back to the control that opened
  // it, per the ARIA disclosure pattern. Without this the only ways out are
  // re-pressing the toggle or following a link, which navigates away.
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isOpen()) {
      setOpen(false);
      toggle.focus();
    }
  });

  // A click outside the panel closes it. Focus is deliberately NOT moved here:
  // the pointer has already established where the user's attention is, and
  // yanking focus back to the toggle would be disorienting.
  document.addEventListener("click", (event) => {
    if (!isOpen()) {
      return;
    }
    const target = event.target instanceof Element ? event.target : null;
    if (target && (target.closest(".nav-toggle") || target.closest("[data-nav-links]"))) {
      return;
    }
    setOpen(false);
  });
};
