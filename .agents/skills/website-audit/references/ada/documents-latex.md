# LaTeX → accessible PDF

**Covers:** where LaTeX kernel tagging stands, the minimum viable accessible document, what must still be done by hand, mathematics and MathML, package compatibility, validation commands, and realistic expectations.
**Load when:** `content.has_latex_pdfs: true` — academic, scientific, journal or mathematics-bearing PDFs, or the client compiles from `.tex`. **Conditional — do not load otherwise.**
**Prerequisite:** the tagged-PDF model and the WCAG2ICT rules in `documents-pdf.md`. In short: target all Level A and AA criteria **except 2.4.1 and 2.4.5**, treat 3.2.3 / 3.2.4 / 3.2.6 as set-level, and add PDF/UA on top.

---

## 1. Where things stand

Historically LaTeX output was one of the **least** accessible document formats in existence — untagged, with mathematics rendered as unlabelled glyph soup and no defined reading order. That has changed substantially. The **LaTeX Tagged PDF Project** (Frank Mittelbach, Ulrike Fischer, David Carlisle, Joseph Wright) has been building tagging support directly into the LaTeX kernel, and as of the **2025-11-01** and **2026-06-01** releases it is usable for real documents.

It is **not finished.** Package coverage is incomplete, output should always be validated, and some classes — notably `beamer` and many journal styles — lag. **Plan for verification, not fire-and-forget.**

---

## 2. The minimum viable accessible document

Put `\DocumentMetadata` **before** `\documentclass`. This is the switch that turns everything on.

```latex
\DocumentMetadata{
  lang         = en,
  pdfstandard  = ua-2,        % PDF/UA-2 (ISO 14289-2)
  pdfstandard  = a-4f,        % PDF/A-4f — or a-4; omit if not archiving
  tagging      = on,
  tagging-setup = {math/setup=mathml-SE},
  % pdfversion = 2.0 is the default once \DocumentMetadata is used
}
\documentclass{article}

\title{Quarterly Structural Analysis}
\author{A. Engineer}

\begin{document}
\maketitle
\tableofcontents

\section{Introduction}
Ordinary text is tagged automatically as \texttt{<P>} inside the
section structure.

\end{document}
```

| Requirement | Detail |
|---|---|
| **Engine** | **LuaLaTeX.** `pdfLaTeX` is supported, but LuaLaTeX is the recommended engine for new documents and is **required for automatic MathML generation**. **XeLaTeX is not supported for tagging.** |
| **Release** | LaTeX **2025-11-01 or later**; **2026-06-01 preferred**. The `tagging=on` key does not exist in older releases. |
| **Overleaf** | Enable the **"Rolling TeX Live"** compiler option. To get newer tagging features ahead of Overleaf's default, add a `latexmkrc`: |

```
$max_repeat  = 1;
$force_mode  = 1;
$pdflatex    = 'pdflatex-dev -synctex=1 -interaction=nonstopmode';
$lualatex    = 'lualatex-dev -synctex=1 -interaction=nonstopmode';
```

---

## 3. What you still have to do by hand

Tagging is automatic for structure. Three things are not.

### Graphics — every image described or artifacted

```latex
% Informative image
\includegraphics[width=.6\textwidth,
  alt={Stress distribution across the beam, peaking at 240 MPa near the
       left support and falling to 40 MPa at midspan}]{stress.png}

% Purely decorative
\includegraphics[height=2cm, artifact]{flourish.pdf}

% Glyph-like image standing in for a character
\includegraphics[height=\baselineskip, actualtext=A]{fancy-A.pdf}
```

The `alt`, `artifact` and `actualtext` keys also work on `\tikz`, the `tikzpicture` environment, and the `picture` environment.

For a complex figure, put the short description in `alt` and the full explanation in the caption or an adjacent paragraph — the same short-alt-plus-long-description pattern as HTML (`html-core.md` §5).

### Tables — declare header rows

```latex
% Globally in the preamble, or immediately before a table
\tagpdfsetup{table/header-rows={1}}
\begin{tabular}{lrr}
  Component & Load (kN) & Factor \\
  Beam A    & 120       & 1.4    \\
  Beam B    & 95        & 1.4    \\
\end{tabular}
```

Multiple header rows: `table/header-rows={1,2}`.

Tables used purely for **visual alignment** (not data) must be declared presentational, or a screen reader announces a meaningless table:

```latex
\tagpdfsetup{table/tagging=presentation}
\begin{tabular}{ccc} … \end{tabular}
```

### Mathematics — see §4.

---

## 4. Mathematics

The hardest problem and the most interesting solution. The goal is to embed **MathML** so assistive technology can read, navigate and re-render the expression rather than reciting glyphs.

Two mechanisms exist; they are not mutually exclusive.

| Mechanism | What it is | Engines |
|---|---|---|
| **Associated Files (AF)** | An embedded XML stream carrying the MathML for each formula, exactly as you'd write it on a web page | LuaLaTeX (auto) and pdfLaTeX (manual) |
| **MathML Structure Elements (SE)** | A PDF 2.0 feature extending the PDF tag set with MathML-corresponding tags | **LuaLaTeX only** |

**With LuaLaTeX**, the `luamml` package (loaded automatically when needed) converts TeX to MathML. With `unicode-math` loaded, Associated Files are produced by default:

```latex
\DocumentMetadata{lang=en, tagging=on, pdfstandard=ua-2, pdfstandard=a-4f}
\documentclass{article}
\usepackage{unicode-math}
\begin{document}
If $x$ is real, then $x^{2} \geq 0$.
\[ \begin{pmatrix}0&1\\1&0\end{pmatrix}
   \begin{pmatrix}a&b\\c&d\end{pmatrix}
 = \begin{pmatrix}c&d\\a&b\end{pmatrix} \]
\end{document}
```

Structure Element tagging instead:

```latex
tagging-setup = {math/setup=mathml-SE}
```

Both:

```latex
tagging-setup = {math/setup={mathml-SE,mathml-AF}}
```

**With pdfLaTeX**, only the Associated File route is available and the MathML must be supplied externally. LaTeX will write a dummy file with a slot per formula:

```latex
\tagpdfsetup{math/mathml/write-dummy}
```

Each slot contains the LaTeX source, an MD5 checksum, and an empty `<math></math>` element. Fill in the MathML, rename the file to `<jobname>-mathml.html`, and re-run. **If you edit the maths afterwards, the checksum stops matching and that formula's MathML is silently dropped** — regenerate.

*Practical shortcut:* even for a pdfLaTeX final build, run LuaLaTeX once to have `luamml` generate the MathML file, edit if needed, then build with pdfLaTeX.

**Reality check.** Screen reader support for MathML **in PDF** is still uneven — better in some JAWS/NVDA configurations than others. HTML with MathML (now natively supported across all major browsers) plus MathJax remains the most reliable route for mathematically heavy content. **If you have a genuinely maths-dense document and a genuine accessibility obligation, publish HTML alongside the PDF.**

---

## 5. Other structures

| Structure | Status |
|---|---|
| **Lists** | `enumerate`, `itemize`, `description` tag automatically. The new list implementation supports a key-value interface similar to `enumitem`: `\begin{enumerate}[start=5]`, `\begin{enumerate}[resume=true]`. |
| **Cross-references and links** | `hyperref` works; links are tagged. Give links meaningful text rather than bare numbers where you can (2.4.4). |
| **Footnotes** | Tagged as `Note`. |
| **Bibliography** | Tags as a list; `biblatex` support is improving — **verify**. |
| **Multi-column** (`multicol`) | Verify reading order carefully in ngPDF. |

---

## 6. Package compatibility

The LaTeX Project maintains a **tagging status page** covering **over 1,000** contributed packages and classes, driven by an automated CI test suite:

> https://latex3.github.io/tagging-project/tagging-status

**Check every package in your preamble before committing to a tagged workflow.** Packages that manipulate output routines, boxes, or floats are the most likely to break tagging. If a critical package is unsupported: find an alternative, wait, or fall back to producing HTML.

---

## 7. Alternative and complementary routes

| Approach | Use when |
|---|---|
| **LaTeX → HTML** via `make4ht`/`tex4ht`, **LaTeXML**, or **Pandoc** | Content is maths-heavy, or the audience needs reflow/zoom. Produces MathML that works well in browsers. Often the *better* accessible deliverable. |
| **`axessibility`** | Legacy pragmatic approach: embeds the LaTeX source of formulas as hidden text so AT can read `\frac{a}{b}` aloud. Superseded by kernel tagging but still seen in the wild. |
| **`accessibility` package** (Andy Clifton) | Older tagging package aimed at KOMA-Script classes. Predates kernel support. |
| **`tagpdf` standalone** | The underlying engine (v1.0c, 2026-05-17). You normally do **not** load it directly any more — `\DocumentMetadata{tagging=on}` handles it. Load directly only for low-level experimentation. |
| **`latex2nemeth`** | Direct LaTeX-to-Braille (Nemeth) transcription for braille output. |

---

## 8. Realistic expectations

| Document | Expect |
|---|---|
| Plain `article` with sections, paragraphs, lists, described figures, declared table headers | **Tags well, validates cleanly.** |
| Custom journal class + `tikz` + `multicol` + exotic packages | **Expect problems.** Test early, not at submission. |
| `beamer` presentations | **Tagging support is limited.** For accessible slides today, PowerPoint (with proper layouts and reading order) or an HTML slide framework is more reliable — see `documents-office.md`. |
| Anything shipped under a legal obligation | **Validate every build**, ideally in CI. |

---

## How to verify

```bash
# 1. Compile — twice, for refs and TOC
lualatex --interaction=nonstopmode main.tex
lualatex --interaction=nonstopmode main.tex

# 2. Validate PDF/UA
verapdf --flavour ua1 --format html main.pdf > report.html
# (check your veraPDF version for a ua2 profile if targeting PDF/UA-2)

# 3. Validate PDF/A if claimed
verapdf --flavour 4f main.pdf

# 4. Confirm tagging actually happened (0 = the DocumentMetadata switch didn't take)
qpdf --qdf --object-streams=disable main.pdf - | grep -c StructTreeRoot

# 5. Confirm MathML made it in
qpdf --qdf --object-streams=disable main.pdf - | grep -c '<math'
```

Then: open in **PAC 2024**, and inspect the structure and derived HTML at **ngpdf.com** — the derived HTML is the fastest way to see what a screen reader will actually encounter. Finally, read it with a screen reader.

The **`uncompress`** key in `\DocumentMetadata` makes the PDF internals human-readable while debugging — **remove it for the final build.**

Wire steps 2 and 4 into CI so a package upgrade that silently breaks tagging fails the build rather than shipping.

---

## Pre-compile checklist

- [ ] LaTeX release **2025-11-01 or later**; **LuaLaTeX**
- [ ] `\DocumentMetadata{lang=…, tagging=on, pdfstandard=ua-2, pdfstandard=a-4f}` **before** `\documentclass`
- [ ] Every package checked against the tagging-status page
- [ ] Structural commands used throughout (`\section`, `enumerate`, `tabular`) — no manual formatting
- [ ] `alt=` or `artifact` on every `\includegraphics`, `\tikz` and `picture`
- [ ] `table/header-rows` declared for every data table
- [ ] `table/tagging=presentation` on every layout-only table
- [ ] Math strategy chosen (`mathml-SE` and/or `mathml-AF`); MathML verified
- [ ] Meaningful link text
- [ ] `uncompress` removed for the final build
- [ ] veraPDF + PAC + ngPDF verified
- [ ] Screen-reader read-through

*LaTeX Tagging Project — https://latex3.github.io/tagging-project/ · usage instructions — https://latex3.github.io/tagging-project/documentation/usage-instructions · package tagging status — https://latex3.github.io/tagging-project/tagging-status · `tagpdf` on CTAN — https://ctan.org/pkg/tagpdf · TUG Accessibility TWG — https://www.tug.org/twg/accessibility/*
