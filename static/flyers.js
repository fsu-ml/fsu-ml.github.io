import { pageData } from "./js/data/page-data.js";
import { renderFlyers } from "./js/render/flyers.js";
import { renderFooter, renderNavigation } from "./js/render/sections.js";
import { qs } from "./js/utils/dom.js";
import { bindNavigation } from "./js/ui/navigation.js";
import { bindHeaderChrome } from "./js/ui/chrome.js";
import { bindFlyerLightbox } from "./js/ui/lightbox.js";
import { activateMotion } from "./js/ui/reveal.js";
import { bindSeasons } from "./js/seasonal/season.js";

const init = async () => {
  document.title = `Flyers | ${pageData.page.title}`;
  qs('meta[name="description"]').setAttribute(
    "content",
    "Flyers and announcement artwork for FSU SC Artificial Intelligence Seminar talks, by semester."
  );

  renderNavigation("flyers");
  await renderFlyers();
  renderFooter();
  bindNavigation();
  bindHeaderChrome();
  bindFlyerLightbox();
  activateMotion();
  // Mounted after the chrome exists so themes decorate real elements. Not
  // awaited: the layer is decorative and must never delay the page settling.
  bindSeasons();
};

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<div class="render-error">The flyers could not load. Please serve this folder with a local web server.</div>'
  );
});
