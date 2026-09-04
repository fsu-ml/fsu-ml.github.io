import { pageData } from "./js/data/page-data.js";
import { loadTemplates } from "./js/data/templates.js";
import { qs } from "./js/utils/dom.js";
import {
  renderCommunity,
  renderFooter,
  renderHero,
  renderNavigation,
  renderOverview,
  renderSchedule,
  renderSpeakers
} from "./js/render/sections.js";
import { bindNavigation } from "./js/ui/navigation.js";
import { bindHashScroll, scrollToHashAfterPaint } from "./js/ui/scroll-to-hash.js";
import { bindHeaderChrome } from "./js/ui/chrome.js";
import { activateMotion } from "./js/ui/reveal.js";
import { bindSeasons } from "./js/seasonal/season.js";

const init = async () => {
  document.title = pageData.page.title;
  qs('meta[name="description"]').setAttribute("content", pageData.page.description);

  const templates = await loadTemplates();
  renderNavigation("home");
  await renderHero(templates);
  renderOverview(templates);
  await renderSchedule(templates);
  await renderSpeakers(templates);
  renderCommunity(templates);
  renderFooter();
  bindNavigation();
  bindHeaderChrome();
  activateMotion();
  // Mounted after the chrome exists so themes decorate real elements. Not
  // awaited: the layer is decorative and must never delay the page settling.
  bindSeasons();
  bindHashScroll();
  scrollToHashAfterPaint();
};

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<div class="render-error">The page could not load its templates. Please serve this folder with a local web server.</div>'
  );
});
