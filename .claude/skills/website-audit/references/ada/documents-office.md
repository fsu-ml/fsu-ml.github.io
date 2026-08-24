# Office and other document sources

**Covers:** Word, PowerPoint, Excel; plus InDesign, Google Docs/Slides, Markdown, EPUB and HTML email.
**Load when:** `content.has_office_docs`, `content.has_epub` or `content.has_html_email` is true. **Conditional — do not load for a site that serves none of these.**
**Prerequisite:** the tagged-PDF model and the WCAG2ICT rules in `documents-pdf.md`. In short: for any standalone document, target all Level A and AA criteria **except 2.4.1 and 2.4.5**, treat 3.2.3 / 3.2.4 / 3.2.6 as set-level obligations, and add PDF/UA on top for anything exported to PDF.

**The governing rule for all of these:** accessibility starts in the **source**, not in Acrobat. Fix the document, export correctly, and the tags come out right. **Never "Print to PDF"** — it rasterizes structure away entirely.

---

## 1. Word

### Structure

| Do | Not |
|---|---|
| **Built-in heading styles** (Heading 1, 2, 3…) from the Styles pane — this is what becomes `<H1>`/`<H2>` tags in the PDF and what lets users navigate | Manually bolded and enlarged text |
| One Heading 1 per document, usually the title; don't skip levels | |
| **Built-in list tools** | Manually typed hyphens or numbers |
| **Insert → Table**, header row marked (*Table Tools → Design → Header Row*, and *Table Properties → Row → Repeat as header row at the top of each page*) | Merged and split cells; nested tables; blank rows/columns used for spacing |
| **Page breaks** | Repeated Enter presses |
| **Insert → Page Number** — these become artifacts on export, correctly | Typed page numbers in the footer |
| Real columns (*Layout → Columns*) | Tabs or tables faking columns |

### Images

- Set alt text: right-click → **View Alt Text**. **Delete Word's auto-generated description** and write your own. Mark decorative images as **Decorative**.
- **Set every image to "In Line with Text."** Floating images are frequently skipped entirely by screen readers and lose their position in the reading order. **This is one of the most common and most damaging Word errors.**
- Don't group multiple images/shapes into a picture that carries information — flatten and describe, or provide the content in text.

### Links, language, colour

- Meaningful link text (right-click → Link → *Text to display*), not raw URLs.
- Set the document language: *Review → Language → Set Proofing Language*. Set it on foreign-language passages too.
- Check contrast; don't use colour alone.
- **Avoid text boxes** — a reading-order hazard. Use styles instead.
- Add a document **Title** in *File → Info → Properties*.

### Check, then export

1. **Review → Check Accessibility.** Fix everything. Word feeds this information into the tagging process on export.
2. Export correctly:

| Platform | Route |
|---|---|
| **Windows** | *File → Save As → PDF* → **Options** → ensure **"Document structure tags for accessibility"** is checked and **"Create bookmarks using: Headings"** |
| **Mac** | *File → Save As → PDF*, select **"Best for electronic distribution and accessibility"** |
| **With Acrobat installed** | The *Acrobat* ribbon tab → **Create PDF**. Generally the most reliable route; best preserves heading levels and alt text. |
| **Never** | *Print → Save as PDF* |

3. **Validate the resulting PDF** with PAC/veraPDF. Word's export is good but not perfect — **link tags and complex tables often need touching up in Acrobat**.

### Word pre-export checklist

- [ ] Built-in heading styles; no skipped levels
- [ ] Built-in list tools
- [ ] Tables via Insert → Table, header row marked and repeating, no merged cells
- [ ] All images **In Line with Text**, with alt text or marked Decorative
- [ ] Meaningful hyperlink text
- [ ] Document language set (and on foreign passages)
- [ ] Document Title set in Properties
- [ ] Contrast checked; colour not the sole cue
- [ ] No text boxes carrying content
- [ ] Review → Check Accessibility returns clean
- [ ] Exported with "Document structure tags for accessibility" / "Best for electronic distribution and accessibility"
- [ ] Resulting PDF validated

---

## 2. PowerPoint

- **Every slide needs a title.** Use the built-in title placeholder. If it must be hidden visually, use *Home → Arrange → Selection Pane* to keep it present but move it off-slide, or use title-less layouts sparingly. **Titles are how screen reader users navigate a deck.**
- **Use built-in slide layouts.** Content added outside placeholders lands at the **end** of the reading order.
- **Check reading order on every slide:** *Home → Arrange → Selection Pane* (bottom-to-top = reading order), or the Accessibility ribbon's Reading Order pane.
- Alt text on every image, chart, SmartArt and grouped shape.
- Don't put essential information **only** in speaker notes — but *do* use notes for extended descriptions of complex visuals, and publish the notes.
- Avoid animations that convey meaning; avoid auto-advance (2.2.2).
- Ensure contrast in the **theme**, not just in individual slides.
- **Video in decks still needs captions** — see `media-and-motion.md`.
- Run **Review → Check Accessibility**, then export with structure tags as for Word.

---

## 3. Excel

- Give every sheet a **meaningful tab name**; delete empty sheets.
- Use a single, contiguous data region per sheet where possible; avoid blank rows/columns inside data.
- Mark the header row: *Insert → Table* (which sets a header row), or define a Print Title / named range.
- Set alt text on charts and images.
- Don't encode meaning in **cell colour alone** — add a text or symbol column (1.4.1).
- Avoid merged cells.
- Add cell-level context where needed; avoid relying on comments/notes for essential information.
- **Excel → accessible PDF is the weakest of the three exports.** For data that must be distributed as a document, consider publishing the data as **CSV plus an accessible HTML or Word summary** instead.

---

## 4. Adobe InDesign

- Build with **paragraph styles**, then map them to PDF tags: *Paragraph Styles menu → Edit All Export Tags → PDF*.
- Alt text: *Object → Object Export Options → Alt Text*.
- Reading order: the **Articles panel** (*Window → Articles*) — drag content into the order it should be read. **Content not in an article is at the mercy of the layout order.**
- Anchor inline graphics to the text they belong with.
- Export via **File → Export → Adobe PDF (Interactive or Print)** with **"Create Tagged PDF"** checked and Compatibility Acrobat 6 (PDF 1.5) or higher; include bookmarks and hyperlinks.
- **Expect to finish the job in Acrobat** — InDesign's table tagging in particular usually needs correction.

---

## 5. Google Docs / Slides

- Built-in heading styles (*Format → Paragraph styles*).
- Alt text: right-click image → *Alt text* (or <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>Y</kbd>).
- Built-in list and table tools; set a header row.
- Document language: *File → Language*.
- **Export quality:** *File → Download → PDF* produces a tagged PDF, but the tagging is **weaker than Word's** — tables and links in particular. **For anything with a compliance obligation, export to `.docx`, open in Word, verify, and export from there.**

---

## 6. Markdown

Markdown is an excellent accessible **source** format because it is structurally constrained — headings, lists and links are semantic by construction.

| Rule | Detail |
|---|---|
| Heading order | Use `#`/`##`/`###` in order; don't skip levels |
| Alt text | Always supply it: `![Chart showing 40% growth](chart.png)`. Use `![](decorative.png)` sparingly and prefer CSS backgrounds for decoration. |
| Link text | Descriptive, not `[click here]` |
| Tables | Real tables with a header row; pipe tables produce `<th>` in most renderers |
| Raw HTML | Avoid unless it is accessible HTML |

**Markdown has no mechanism for `scope`, complex table headers, `lang` on passages, or figure/caption association** — for those, drop to HTML or post-process.

**Converting to PDF:** go **Markdown → HTML → PDF** with a tagging-aware engine (e.g. Prince, or Pandoc → LaTeX with `\DocumentMetadata` — see `documents-latex.md`). **Naive Markdown→PDF converters (wkhtmltopdf and most Chromium-print pipelines) produce untagged output.**

---

## 7. EPUB

- **EPUB Accessibility 1.1** (ISO/IEC 23761) is the governing standard; it references WCAG 2.x for content.
- The **EAA makes ebook accessibility a legal requirement in the EU as of June 2025** (`targets.md`).
- Requirements: semantic HTML in the content documents, a proper `nav` document, `epub:type` semantics, page-list mapping to print pages where a print equivalent exists, MathML for maths, alt text throughout, and **accessibility metadata** — `schema:accessibilityFeature`, `accessMode`, `accessibilitySummary`, `accessModeSufficient`.
- Validate with **EPUBCheck** plus **Ace by DAISY**.
- **EPUB is genuinely reflowable**, which makes it a better accessible format than PDF for long-form reading.

---

## 8. HTML email

- Semantic headings, real lists, alt text, sufficient contrast.
- **Provide a plain-text alternative part.**
- **Avoid image-only emails** — an image of a poster with no text is unreadable to a screen reader and blocked by default in many clients.
- Don't use tables for layout without `role="presentation"` — though email client support is inconsistent, so keep layouts simple.
- Don't rely on background images to carry information.

---

## How to verify

| Format | Check |
|---|---|
| **Word / PowerPoint / Excel** | *Review → Check Accessibility* must return clean — but it is a floor, not a ceiling; it does not judge alt-text quality or reading-order sense. Then **manually review reading order**: Selection Pane on every slide, and read the Word document with the Navigation Pane (*View → Navigation Pane*) as a heading outline. Finally **validate the exported PDF** with veraPDF and PAC (`documents-pdf.md`). |
| **Word images floating vs inline** | *Home → Select → Selection Pane*, or *Layout Options* on each image — the anchor icon means floating. Inline is the only safe setting. |
| **Any exported PDF** | `verapdf --flavour ua1 --format text out.pdf` and `qpdf --qdf --object-streams=disable out.pdf - \| grep -c StructTreeRoot` (0 = untagged, the export option was off). |
| **InDesign** | Open the Articles panel — content not listed there will not be in the tag tree. Then validate the exported PDF. |
| **Google Docs** | Export to PDF and run veraPDF. If tables or links fail, re-route through `.docx` → Word → export. |
| **Markdown pipelines** | Convert one representative document and check the output PDF for `StructTreeRoot` as above. Untagged output means the pipeline needs replacing, not the source. |
| **EPUB** | `epubcheck book.epub` then Ace by DAISY (`ace book.epub -o ./report`), then open it in a real reading system and navigate by headings. |
| **Email** | Send it to yourself, disable image loading, and read it. If the message is gone, it was an image-only email. Check the multipart body has a `text/plain` part. |
