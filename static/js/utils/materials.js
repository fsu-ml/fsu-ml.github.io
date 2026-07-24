const materialLabel = (href = "") => {
  const value = href.trim();
  if (!value) {
    return "Resource";
  }

  if (/^https?:\/\//i.test(value)) {
    try {
      const url = new URL(value);
      const host = url.hostname.replace(/^www\./i, "");
      if (/youtube\.com|youtu\.be/i.test(host)) {
        return "Video";
      }
      if (/docs\.google\.com/i.test(host)) {
        return "Slides";
      }
      if (/github\.com/i.test(host)) {
        return "GitHub";
      }
      if (/calendar\.fsu\.edu/i.test(host)) {
        return "Event";
      }
      const pathSegment = url.pathname.split("/").filter(Boolean).pop() || host;
      return pathSegment.length > 28 ? `${pathSegment.slice(0, 25)}...` : pathSegment;
    } catch {
      return "Link";
    }
  }

  const fileName = value.split("/").pop() || value;
  if (/\.(pdf|pptx?|ipynb|md|txt)$/i.test(fileName)) {
    return fileName;
  }
  return fileName.length > 28 ? `${fileName.slice(0, 25)}...` : fileName;
};

export const parseMaterialLinks = (materials = "") =>
  materials
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((href) => {
      const url = /^https?:\/\//i.test(href) ? href : `/${href.replace(/^\//, "")}`;
      return { href: url, label: materialLabel(href) };
    });
