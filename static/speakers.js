import { pageData } from "./js/data/page-data.js";
import { loadTemplates } from "./js/data/templates.js";
import { qs } from "./js/utils/dom.js";
import { renderFooter, renderNavigation, renderSpeakerDirectory } from "./js/render/sections.js";
import { bindNavigation } from "./js/ui/navigation.js";

const init = async () => {
  document.title = `Speakers | ${pageData.page.title}`;
  qs('meta[name="description"]').setAttribute(
    "content",
    "Speaker directory for the FSU SC Artificial Intelligence Seminar."
  );

  const templates = await loadTemplates();
  renderNavigation("speakers");
  await renderSpeakerDirectory(templates);
  renderFooter();
  bindNavigation();
};

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<div class="render-error">The speaker directory could not load. Please serve this folder with a local web server.</div>'
  );
});
