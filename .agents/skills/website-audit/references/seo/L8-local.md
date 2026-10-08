# Layer 8 — Local SEO

Covers: Google Business Profile, NAP consistency, location pages, local citations.
Load: **CONDITIONAL — `seo.local_business: true` only.** A physical location or a defined service area. Default is skip.
Skip when: `seo.local_business: false`. Portfolios, SaaS, publishers, lab sites, anything with no geographic service constraint.
Depends on: `L4-onpage.md` (location page quality) and `L6-structured-data.md` (`LocalBusiness` schema).

Most of this happens outside the website. The site is the corroborating source, not the primary surface.

---

## 8.1 Google Business Profile

- [ ] **GBP claimed, verified, and fully completed** — sign in at `business.google.com`. Pass: verified badge, no "incomplete" prompts remaining. Also search the business name + city in Google Maps and confirm exactly one listing (duplicates suppress each other).
- [ ] **Correct primary category, plus relevant secondary categories** — GBP → Business information → Category. Pass: the primary category matches what you most want to be found for. **This is the single highest-leverage GBP field.** Verify by searching the category term from a nearby location and seeing whether the map pack composition makes sense.
- [ ] **Accurate hours, including holiday hours** — Pass: current, and special hours set for upcoming holidays. Wrong hours generate "permanently closed" user reports, which is the fastest way to lose a listing.
- [ ] **Service areas defined if you travel to customers** — Pass: set, and matching the geography recorded in `business-context.md`.
- [ ] **Products/services listed in GBP** — Pass: populated, matching the service pages on the site.
- [ ] **Photos uploaded regularly; posts and updates published** — Pass: photos within the last 90 days, posts on a stated cadence.
- [ ] **Q&A section seeded and monitored** — Pass: the questions customers actually ask are answered by the business, and new ones get answered within days. Anyone can answer a GBP question, including a competitor.
- [ ] **Reviews actively requested, and every review responded to professionally** — Pass: a review-request process exists, and the response rate is ~100% including negatives. Never buy reviews (`L7-authority.md`).

---

## 8.2 Consistency

- [ ] **NAP consistency — Name, Address, Phone identical across website, GBP, directories, and social profiles** — search the phone number in quotes and the address in quotes; compare every result byte for byte. Pass: identical formatting everywhere, including suite notation, abbreviations, and phone punctuation. `Suite 4` vs `Ste. 4` vs `#4` are three different addresses to a matching algorithm.
- [ ] **`LocalBusiness` schema on the site matching GBP exactly** — `curl -s https://DOMAIN/ | grep -A30 LocalBusiness`. Pass: `name`, `address`, `telephone`, `geo`, `openingHoursSpecification` present and matching the GBP values character for character. See `L6-structured-data.md`.
- [ ] **Embedded map and address in the footer** — Pass: present on every page, marked up, and matching NAP.

---

## 8.3 Location pages

- [ ] **Location pages with genuinely unique content per location** — read three of them side by side. Pass: each contains information true only of that location — the actual team, actual local projects, actual parking, actual service radius. **Fail: templated pages with a swapped city name. Those are doorway pages and an explicit spam policy violation.**
- [ ] **One location page per location you actually serve, and no more** — Pass: no pages for cities you don't operate in. A page for every town within 50 miles is the classic doorway pattern.

---

## 8.4 Citations and other surfaces

- [ ] **Local citations built on legitimate directories** — the ones a human would actually use in this category and region: chambers of commerce, trade bodies, established local directories. Pass: present and NAP-consistent. Volume is not the goal; consistency is.
- [ ] **Bing Places claimed** 🔒 — `bingplaces.com`. Pass: claimed and verified. Feeds Bing Maps and, indirectly, Copilot and ChatGPT search.
- [ ] **Apple Business Connect claimed** 🔒 — `businessconnect.apple.com`. Pass: claimed and verified. Feeds Apple Maps and Siri.

---

## Questions to ask

1. Does my address appear byte-identical everywhere it appears online?
2. When did I last post to GBP, and when did I last earn a review?
3. Do my location pages say anything true and specific about each location?
4. If a competitor searched my category from three miles away, would I show in the map pack?

---

Next: `L9-ai-search.md`. Related: `L6-structured-data.md` for `LocalBusiness`; `L7-authority.md` for review platform management, which overlaps heavily with this layer.
