import { pageData } from "./js/data/page-data.js";
import { loadTemplates } from "./js/data/templates.js";
import { qs } from "./js/utils/dom.js";
import { renderFooter, renderFullSchedule, renderNavigation } from "./js/render/sections.js";
import { bindNavigation } from "./js/ui/navigation.js";
import { bindHeaderChrome } from "./js/ui/chrome.js";
import { activateMotion } from "./js/ui/reveal.js";
import { bindSeasons } from "./js/seasonal/season.js";

const init = async () => {
  document.title = `Schedule | ${pageData.page.title}`;
  qs('meta[name="description"]').setAttribute(
    "content",
    "Upcoming FSU SC Artificial Intelligence Seminar talks, dates, speakers, and topics."
  );

  await loadTemplates();
  renderNavigation("schedule");
  await renderFullSchedule();
  renderFooter();
  bindNavigation();
  bindHeaderChrome();
  activateMotion();
  // Mounted after the chrome exists so themes decorate real elements. Not
  // awaited: the layer is decorative and must never delay the page settling.
  bindSeasons();
};

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<div class="render-error">The schedule could not load. Please serve this folder with a local web server.</div>'
  );
});
