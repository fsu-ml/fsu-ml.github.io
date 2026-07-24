const getHeaderOffset = () => {
  const header = document.querySelector(".site-header");
  return header ? header.getBoundingClientRect().height + 16 : 88;
};

export const scrollToHashIfPresent = ({ behavior = "auto" } = {}) => {
  const { hash } = window.location;
  if (!hash) {
    return false;
  }

  const target = document.querySelector(hash);
  if (!target) {
    return false;
  }

  const top = target.getBoundingClientRect().top + window.scrollY - getHeaderOffset();
  window.scrollTo({ top: Math.max(0, top), behavior });
  return true;
};

export const bindHashScroll = () => {
  window.addEventListener("hashchange", () => {
    scrollToHashIfPresent({ behavior: "smooth" });
  });
};

export const scrollToHashAfterPaint = (options) => {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => scrollToHashIfPresent(options));
  });
};
