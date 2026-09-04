import { qs } from "./js/utils/dom.js";
import { renderFooter, renderNavigation } from "./js/render/sections.js";
import { bindNavigation } from "./js/ui/navigation.js";
import { bindHeaderChrome } from "./js/ui/chrome.js";
import { activateMotion } from "./js/ui/reveal.js";
import { bindSeasons } from "./js/seasonal/season.js";

const bindCopyPrompt = () => {
  const button = qs("[data-copy-prompt]");
  const prompt = qs("[data-llm-prompt]");
  if (!button || !prompt) {
    return;
  }
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(prompt.value);
    } catch {
      prompt.select();
      prompt.setSelectionRange(0, prompt.value.length);
      document.execCommand("copy");
    }
    const original = button.textContent;
    button.textContent = "Copied!";
    button.classList.add("is-copied");
    setTimeout(() => {
      button.textContent = original;
      button.classList.remove("is-copied");
    }, 1400);
  });
};

const init = () => {
  renderNavigation("");
  renderFooter();
  bindNavigation();
  bindHeaderChrome();
  activateMotion();
  // Mounted after the chrome exists so themes decorate real elements. Not
  // awaited: the layer is decorative and must never delay the page settling.
  bindSeasons();
  bindCopyPrompt();
};

try {
  init();
} catch (error) {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<div class="render-error">This page could not load. Please serve this folder with a local web server.</div>'
  );
}
