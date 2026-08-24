# PDF accessibility

**Covers:** how tagged PDF actually works, PDF/UA versions, the WCAG2ICT translation layer shared by all non-web documents, what a conformant PDF requires, remediating existing PDFs, validation with veraPDF/PAC, and the pre-publication checklist.
**Load when:** `content.has_pdfs: true`. **Conditional — do not load for a site that serves no PDFs.**
**Also carries** the WCAG2ICT rules referenced by `documents-office.md` and `documents-latex.md`.

---

## 1. The core concept

A PDF file, at bottom, is a set of instructions: "place this glyph at these coordinates in this font." There is **no inherent notion** of a paragraph, a heading, a list, a reading order, or a table. A screen reader given a raw PDF is guessing.

**Tags** are a parallel structure — the **structure tree** — layered on top of the page content: "these glyphs constitute an `<H1>`; these constitute a `<P>`; this group is a `<Table>` with these `<TH>` and `<TD>` cells; this line-art is an `<Artifact>` and should be ignored."

Everything about PDF accessibility reduces to whether that tree exists and whether it is correct.

---

## 2. WCAG2ICT — reading WCAG for non-web documents

WCAG was written for web pages. Applying it verbatim to a standalone PDF, Word file or `.tex` output produces nonsense in a few places ("set of web pages", "web page", "user agent"). The W3C's **WCAG2ICT** Group Note — updated **October 2024** to cover WCAG 2.0, 2.1 *and* 2.2 — provides the translation. It is **informative, not normative**, but it is the reference that ADA Title II regulations and EN 301 549 chapter 10 lean on.

### Core substitutions

| WCAG term | Non-web reading |
|---|---|
| "web page" | **non-web document** |
| "set of web pages" | **set of documents** — documents published together as a body, e.g. a manual in volumes |
| "user agent" | the document reader / player / viewer application |
| "programmatically determined" | determinable by AT via the document format's own structures — i.e. **tags** |

### Criteria that behave differently for documents

| SC | Treatment for non-web documents |
|---|---|
| **2.4.1 Bypass Blocks** | Generally **not applicable** to a single non-web document — there is no repeated cross-page navigation block to skip. Headings and bookmarks still serve the underlying need. |
| **2.4.5 Multiple Ways** | Generally **not applicable** to an individual document; applies to *sets* of documents. |
| **3.2.3 Consistent Navigation** | Applies only across a **set of documents**. |
| **3.2.4 Consistent Identification** | Applies only across a **set of documents**. |
| **3.2.6 Consistent Help** | Applies across a set of documents. |
| **2.4.2 Page Titled** | Read as **"document titled"** — the PDF `Title` metadata entry, displayed in preference to the filename. |
| **4.1.1 Parsing** | Removed in 2.2; for 2.0/2.1 claims WCAG2ICT limited it to markup-based formats. |
| **4.1.2 Name, Role, Value** | Applies to documents containing **interactive elements** — form fields, links, buttons. |
| **1.4.13 Content on Hover/Focus** | Applies where the format supports hover/focus content, e.g. PDF form-field tooltips. |
| **3.3.8 Accessible Authentication** | Rarely applicable to documents; applicable to document-based workflows requiring login. |

> **Practical rule for any standalone document:** target **all Level A and AA criteria except 2.4.1 and 2.4.5**, and treat 3.2.3 / 3.2.4 / 3.2.6 as set-level obligations. Then add PDF/UA on top for the format-specific machinery WCAG does not cover.

**What WCAG2ICT does not cover:** WCAG2ICT is explicit that WCAG 2 was not designed for non-web ICT and that fully addressing document accessibility requires **requirements beyond WCAG**. Those gaps are exactly what PDF/UA fills: tag-to-content correspondence, artifact marking, the structure tree, `/Lang` at the object level, embedded font requirements, `/StructTreeRoot`, `/MarkInfo`, role mapping. **A PDF can pass a WCAG checklist and still be broken for a screen reader if the PDF/UA machinery is wrong.**

*WCAG2ICT — https://www.w3.org/TR/wcag2ict/ · overview: https://www.w3.org/WAI/standards-guidelines/wcag/non-web-ict/*

---

## 3. The two standards you are working to

| | WCAG 2.x (via WCAG2ICT) | PDF/UA (ISO 14289) |
|---|---|---|
| Governs | **Content decisions** — is the alt text meaningful? is contrast sufficient? is the language plain? | **Format machinery** — does the tag tree exist, is it correctly nested, are artifacts marked, is metadata present? |
| Referenced by | ADA, §508, EN 301 549, EAA | §508 (via WCAG), procurement specs, EN 301 549 clause 10 |
| Testable by | Human review + some tooling | Largely machine-checkable — **89 of the 136 Matterhorn conditions** |

**Do both.** Neither alone produces a usable document.

**Versions.** **PDF/UA-1 (ISO 14289-1:2014)**, based on PDF 1.7, is the pragmatic production target today — best validator support, and what most procurement language names. **PDF/UA-2 (ISO 14289-2:2024)**, based on PDF 2.0, adds MathML support, improved annotation handling and richer metadata, and derives from the PDF Association's Well-Tagged PDF (WTPDF) spec; tooling support is still maturing. Files can conform to both. **If you can only name one in a contract today, name PDF/UA-1**; if you are producing new technical or scientific content with mathematics, PDF/UA-2 is worth targeting (see `documents-latex.md`).

---

## 4. What a conformant PDF requires

### Structural
- [ ] `/StructTreeRoot` present; document is tagged (`/MarkInfo` with `/Marked true`)
- [ ] Every piece of real content is inside the structure tree
- [ ] Every piece of non-content — page numbers, running heads, decorative rules, background images, watermarks — is marked as an **`/Artifact`**
- [ ] Tag types are semantically correct: `H1`–`H6`, `P`, `L`/`LI`/`Lbl`/`LBody`, `Table`/`TR`/`TH`/`TD`, `Figure`, `Link`, `Form`, `TOC`/`TOCI`, `Note`
- [ ] Heading levels properly nested and not skipped
- [ ] Tag order = logical reading order. **This is what a screen reader follows, *not* the Content/Order panel.**
- [ ] Nesting is valid — no `<P>` inside `<P>`, no orphan `<TD>`, list items structured correctly

### Content
- [ ] `/Alt` on every `<Figure>` that carries information; decorative images artifacted instead
- [ ] `/ActualText` where glyphs don't match the text — ligatures, drop caps, decorative letterforms
- [ ] Table `<TH>` cells with correct `/Scope` (`Row`, `Column`, `Both`); `/Headers` and `/ID` for complex tables
- [ ] Link annotations wrapped in `<Link>` tags with matching `/Link-OBJR`, and a meaningful `/Contents` on the annotation
- [ ] Form fields have `/TU` tooltips serving as accessible names, and a defined tab order (`/Tabs /S`)
- [ ] Text is real text with correct Unicode mappings — not an image of text; all fonts embedded

### Document-level
- [ ] `/Lang` set at the document level; `/Lang` overrides on foreign-language passages
- [ ] XMP metadata `dc:title` populated, and `/ViewerPreferences << /DisplayDocTitle true >>` so the title, not the filename, shows in the window
- [ ] `pdfuaid:part` identifier in XMP (`1` or `2`) — **without it, validators will not treat the file as PDF/UA at all**
- [ ] No security settings that block content extraction for assistive technology
- [ ] Bookmarks/outline for documents over ~10 pages — not strictly required by PDF/UA but expected by 2.4.5 at set level and by most procurement

### Visual (WCAG side)
- [ ] Contrast **4.5:1 / 3:1**
- [ ] Colour not the sole information carrier
- [ ] No reliance on sensory characteristics
- [ ] **Reflow:** PDFs are fixed-layout and structurally fight SC 1.4.10. A well-tagged PDF reflows in Acrobat's Reflow view; a multi-column untagged PDF does not. **If reflow genuinely matters to your audience, publish HTML as the primary format and PDF as a secondary.**

---

## 5. The single most important rule

> **Accessibility starts in the source document, not in Acrobat.**

Remediating a PDF by hand in Acrobat is slow, expensive, error-prone, and has to be redone from scratch every time the source changes. Fix the Word/LaTeX/InDesign source, export correctly, and the tags come out right automatically. See `documents-office.md` and `documents-latex.md`.

**Never use "Print to PDF."** It rasterizes structure away entirely and produces a flat, untagged, essentially inaccessible file. The same goes for most third-party PDF printer drivers (CutePDF and similar).

---

## 6. Remediating an existing PDF

When there is no source:

| # | Step | Detail |
|---|---|---|
| 1 | **OCR if scanned** | Acrobat: *Scan & OCR → Recognize Text*. Open source: OCRmyPDF / Tesseract. **OCR alone is not accessibility** — it gives you text, not structure. |
| 2 | **Autotag** | Acrobat: *Accessibility → Autotag Document*. Expect it to be **60–80% right and 100% in need of review**. |
| 3 | **Fix the tag tree** | In the Tags panel: correct heading levels, merge/split tags, fix table cells, artifact the decorations. |
| 4 | **Set reading order** | Verify in the **Tags panel order**, not just the Reading Order tool. |
| 5 | **Add alt text** | On figures; artifact decoration. |
| 6 | **Set document properties** | Title, language, and *Show: Document Title* in Initial View. |
| 7 | **Validate** | PAC and veraPDF; fix; re-validate. |
| 8 | **Manual check** | Read the document top to bottom with NVDA or VoiceOver. |

**Budget realistically:** a complex 40-page report with tables and charts can take a skilled remediator a full day or more. Factor this into any backlog plan (`program.md`).

---

## 7. Validation tools

| Tool | Cost | Notes |
|---|---|---|
| **PAC 2024** (PDF Accessibility Checker) | Free, Windows | The de facto reference. Implements the Matterhorn Protocol plus WCAG checks and a screen-reader preview. From the PDF/UA Foundation. |
| **veraPDF** | Free, OSS, cross-platform | `verapdf --flavour ua1 file.pdf`. CLI/Docker/CI-friendly. PDF/A and PDF/UA profiles. PDF Association-backed reference implementation. |
| **Adobe Acrobat Pro** | Paid | Built-in Accessibility Checker + Preflight PDF/UA profile. Weakest of the three on standards conformance but essential for actually *fixing* files. |
| **axesPDF / CommonLook / PDFix** | Paid | Production remediation at volume. |
| **ngPDF** (ngpdf.com) | Free web | Inspect the tag structure and view the HTML derived from the tags. Excellent for sanity-checking generated output. |

**Important caveat: validators disagree.** One published comparison of 155 files found four leading validators producing inconsistent results in **over half of cases**, because the ISO clauses admit different interpretations. Use **two** tools plus a manual screen-reader pass, and never treat a green checkmark from any single tool as proof.

---

## How to verify

```bash
# Machine-checkable PDF/UA conformance
verapdf --flavour ua1 --format html report.pdf > verapdf-report.html
verapdf --flavour ua1 --format text report.pdf | tail -5     # pass/fail summary

# Batch a whole crawled document set
verapdf --flavour ua1 --recurse ./downloaded-pdfs/ --format mrr > batch.xml

# Is it tagged at all? (fast triage before spending time on it)
pdfinfo report.pdf | grep -i tagged
qpdf --qdf --object-streams=disable report.pdf - | grep -c StructTreeRoot   # 0 = untagged

# Is the text real text or a scan?
pdftotext report.pdf - | head -20     # empty output = image-only, needs OCR
```

Then, by hand:

| Check | How |
|---|---|
| Reading order is correct | Acrobat → Tags panel, walk the tree top to bottom. Not the Order panel. |
| Title shows instead of the filename | Open the PDF — read the window/tab title. If it shows `final_v3_APPROVED.pdf`, `DisplayDocTitle` is off. |
| Alt text is meaningful | Acrobat → Tags panel → each `<Figure>` → Properties → Alternate Text. Read it against the alt decision tree in `html-core.md` §5. |
| Reflow works | Acrobat → *View → Zoom → Reflow*. Multi-column text that interleaves is a broken tag order. |
| It is actually usable | Read it end to end with NVDA or VoiceOver. This is the only check that catches "technically tagged, practically unreadable". |
| Contrast | Colour Contrast Analyser eyedropper on the rendered page — 4.5:1 / 3:1. |

---

## Pre-publication checklist

- [ ] Produced from an accessible source, exported with structure tags (**never** "Print to PDF")
- [ ] Tagged: `/StructTreeRoot` present
- [ ] Heading tags correct and properly nested
- [ ] Reading order correct in the **Tags** panel
- [ ] All decoration, headers, footers and page numbers marked as `/Artifact`
- [ ] Alt text on every informative figure
- [ ] Table headers tagged `<TH>` with `/Scope`
- [ ] Links tagged, with meaningful text
- [ ] Form fields have tooltips and a logical tab order
- [ ] Document `/Lang` set; language overrides on passages
- [ ] Title in metadata; *Show Document Title* enabled
- [ ] `pdfuaid:part` identifier present
- [ ] Fonts embedded; text is real text (OCR applied if scanned)
- [ ] No security settings blocking AT
- [ ] Bookmarks for longer documents
- [ ] Contrast and colour-independence verified
- [ ] Passes **PAC and veraPDF**
- [ ] Read end to end with a screen reader

*PDF/UA — https://pdfa.org/resource/iso-14289-pdfua/ · PDF/UA-2 — https://pdfa.org/iso-14289-2-pdfua-2/ · Matterhorn Protocol — https://pdfa.org/resource/matterhorn-protocol/ · WCAG PDF techniques PDF1–PDF23 — https://www.w3.org/WAI/WCAG22/Techniques/*
