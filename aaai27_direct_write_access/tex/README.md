# AAAI-27 TeX Manuscript

This folder contains the active AAAI-27 LaTeX manuscript.

Files:

- `main.tex`: anonymous AAAI-27 submission draft.
- `references.bib`: current bibliography.
- `aaai2027.sty`: copied from the official AAAI-27 Author Kit.
- `aaai2027.bst`: copied from the official AAAI-27 Author Kit.
- `reproducibility_checklist.tex`: copied from the official kit for later
  completion.
- `Makefile`: local compile helper.

Compile from this directory:

```bash
make
```

Intermediate files are written to:

```text
.build/
```

For editor preview compatibility, `make` also copies the final PDF to:

```text
main.pdf
```

The source directory is still kept clean: `.aux`, `.bbl`, `.log`, `.fls`, and
other intermediate files live under `.build/`. The top-level `main.pdf` is
generated for viewing only and is ignored by git.

AAAI style constraints already reflected here:

- `\usepackage[submission]{aaai2027}` for anonymous submission;
- no author identities;
- no `hyperref`;
- no margin or spacing packages;
- `aaai2027.bst` bibliography style.

Before submission:

- fill in the open-model inventory and attribution results;
- complete or prepare the reproducibility checklist according to AAAI-27
  instructions;
- clean PDF metadata before upload;
- verify page limit and supplementary-material rules on the official AAAI-27
  pages.
