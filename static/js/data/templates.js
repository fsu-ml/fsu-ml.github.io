const templateFiles = {
  button: "../../../templates/button-link.html",
  featureCard: "../../../templates/feature-card.html",
  scheduleItem: "../../../templates/schedule-item.html",
  speakerCard: "../../../templates/speaker-card.html",
  speakerDirectoryCard: "../../../templates/speaker-directory-card.html",
  communityCard: "../../../templates/community-card.html"
};

export const loadTemplates = async () => {
  const entries = await Promise.all(
    Object.entries(templateFiles).map(async ([key, path]) => {
      const templateUrl = new URL(path, import.meta.url);
      const response = await fetch(templateUrl);
      if (!response.ok) {
        throw new Error(`Unable to load template: ${templateUrl.href}`);
      }
      return [key, await response.text()];
    })
  );

  return Object.fromEntries(entries);
};
