import { qs } from "./js/utils/dom.js";
import { renderFooter, renderNavigation } from "./js/render/sections.js";
import { bindNavigation } from "./js/ui/navigation.js";

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
    setTimeout(() => {
      button.textContent = original;
    }, 1400);
  });
};

const init = () => {
  renderNavigation("");
  renderFooter();
  bindNavigation();
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
