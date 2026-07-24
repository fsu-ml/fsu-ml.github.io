import { pageData } from "./js/data/page-data.js";
import { renderArchive } from "./js/render/archive.js";
import { renderFooter, renderNavigation } from "./js/render/sections.js";
import { qs } from "./js/utils/dom.js";
import { bindNavigation } from "./js/ui/navigation.js";

const init = async () => {
  document.title = `Archive | ${pageData.page.title}`;
  qs('meta[name="description"]').setAttribute(
    "content",
    "Archive of past FSU SC Artificial Intelligence Seminar talks by semester."
  );

  renderNavigation("archive");
  await renderArchive();
  renderFooter();
  bindNavigation();
};

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML(
    "afterbegin",
    '<div class="render-error">The archive could not load. Please serve this folder with a local web server.</div>'
  );
});
