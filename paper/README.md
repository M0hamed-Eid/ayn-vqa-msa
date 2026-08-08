# AynVQA-MSA / Digilians paper

LaTeX source for the Digilians submission to ImageEval 2026 Task 1a (MSA track).
Built on the **official ACL style files** (`acl-org/acl-style-files`, downloaded
2026-08-07), since the organizers have not released a task-specific template yet.
`acl.sty` and `acl_natbib.bst` in this folder are the unmodified upstream files;
if an official template appears later, only `main.tex` should need to change.

## Files

- `main.tex` — the paper. Sourced from `Digilians_ImageEval2026_Task1a.docx`
  (the team's existing draft) as the primary content, enriched with
  evidence-grounded detail from `AynVQA-MSA_Book.docx` (the fuller project
  writeup) — notably the full per-configuration metrics table including the
  best/worst no-repair runs, the three-conceptual-layers framing, the
  engineering-infrastructure section, and the statistical-significance /
  McNemar's-test discussion.
- `references.bib` — Qwen-VL, Qwen2.5-VL, and Whisper citations, plus two
  **placeholder entries that need action before submission** (see below).
- `acl.sty`, `acl_natbib.bst` — unmodified official ACL style files.

## Building

No LaTeX distribution is installed in this environment, so the file has been
written carefully and checked for balanced braces/math-mode and non-ASCII
characters that commonly break `pdflatex`, but it has **not been compiled
end-to-end**. Before relying on it, build it once, either:

**Overleaf (recommended for a shared/team paper)** — create a new blank
project, upload `main.tex`, `references.bib`, `acl.sty`, `acl_natbib.bst`,
set the compiler to pdfLaTeX with bibtex, and share the project link with the
team. This also gives you real-time collaborative editing, which fits a
"shared paper" better than passing `.tex` files around.

**Local build** (if you install a TeX distribution, e.g. MiKTeX or TeX Live):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Before submitting — open items

1. **Two placeholder bibliography entries** in `references.bib`:
   - `qwen3vl2025` — the production selector (§4.6, "Multimodal Reasoning")
     uses Qwen3-VL, but its technical-report citation was not verified before
     writing this (avoided guessing an arXiv ID). Confirm the real citation
     and un-comment the `\citep{qwen3vl2025}` call flagged with a `TODO`
     footnote at that point in `main.tex`.
   - `imageeval2026task1a` — placeholder for the organizers' own task
     citation, already wired into the Introduction; swap in the real one once
     released.
2. **Author block** (`\author{Anonymous ACL submission}` near the top of
   `main.tex`) — currently anonymized because `\usepackage[review]{acl}` is
   active and the venue's blind-review policy for the system-description
   paper hasn't been confirmed. Once confirmed, either fill in real
   names/affiliations/emails and switch to `\usepackage{acl}` (drop the
   `review` option) for a camera-ready build, or leave as-is for review
   submission.
3. **Hardware specifics** (GPU model, VRAM) are referenced as missing in the
   Limitations section — fill in if available.
4. The results in this paper are **dev-split only**, matching every scored
   run in `experiments.md` at time of writing. No devtest/test predictions
   have been generated yet.
