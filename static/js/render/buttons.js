import { icon } from "../ui/icons.js";
import { escapeHtml, renderTemplate } from "../utils/html.js";

export const renderButton = (template, item) =>
  renderTemplate(template, {
    href: escapeHtml(item.href),
    label: escapeHtml(item.label),
    icon: icon(item.icon),
    variant: `button-${escapeHtml(item.variant)}`
  });
