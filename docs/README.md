# Documentation

The documentation is a [Sphinx](https://www.sphinx-doc.org/) project with the
[PyData theme](https://pydata-sphinx-theme.readthedocs.io/). Everything lives in `docs/source/`.
This file is a map: what is where, and why a few things are set up the way they are.

How to run the checks and the live preview is described in [CONTRIBUTING.md](../CONTRIBUTING.md)
(`nox -s docs`, "Live documentation"). Documenting a new model is step 6 of "Adding a model"
there.

## Layout

| Path (in `docs/source/`)  | What it holds                                                             |
|:--------------------------|:--------------------------------------------------------------------------|
| `conf.py`                 | Every Sphinx setting. Custom extensions are put on `sys.path` and enabled here. |
| `index.rst`, `philosophy.rst`, `known_limitations.rst`, `404.rst` | Top-level pages.          |
| `quick_start/`            | Quick start, installation, and one example page per model in `examples/<family>/`. |
| `api/pyml/`               | API reference: one page per class (`autoclass`), grouped by family. The order is deliberate, see the "API Reference isn't alphabetical" limitation. |
| `changelog/`              | One file per release.                                                     |
| `_ext/`                   | Custom Sphinx extensions (below).                                         |
| `_plot_styles/`           | `light.mplstyle` and `dark.mplstyle`: colors of the plots.                |
| `_latex/pyml.sty`         | Style of the PDF: fonts, colors, cover page, chapter headings.            |
| `_static/`                | `css/` (site styling) and `branding/` (logos, social image).              |
| `_extra/`                 | Files copied to the site root as they are (`html_extra_path`).            |

## Custom extensions (`_ext/`)

Each module has a docstring that explains its usage and limits. In short:

| Extension            | What it does                                                          | Config (`conf.py`)   |
|:---------------------|:----------------------------------------------------------------------|:---------------------|
| `pyml_hide_bases`    | Hides internal base classes from the "Bases:" line of a class: names starting with `_` and the names in `pyml_hidden_bases`. A hidden base is replaced by its own parents, one level only. | `pyml_hidden_bases` |
| `pyml_themed_plot`   | Replaces the `plot` directive: every block is drawn twice, once with `light.mplstyle` and once with `dark.mplstyle`, and the theme shows the one that matches the visitor's mode. Other builders (LaTeX, text) get the light image only. | `pyml_plot_styles` |
| `pyml_empty_sidebar` | Hides the empty left sidebar on pages whose top-level section has no sub-pages. | none |

The extensions are plain modules, not a package. `conf.py` appends `_ext/` to `sys.path`, so
`extensions = [..., "pyml_hide_bases"]` can import them by name.

### Plots

Write `.. plot::` as usual. Text, axes and grid colors follow the theme through the two style
files. Sizes and fonts are shared by both themes and live in `plot_rcparams` in `conf.py`. A color
that carries meaning (for example the red centroids of KMeans) may be set explicitly in the plot
code. The `:context:` option of `plot` is not supported by `pyml_themed_plot`.

Editing a style file or `plot_rcparams` redraws every plot on the next build.

## PDF

The PDF is built with XeLaTeX. Settings: `latex_elements` and the other `latex_*` values in
`conf.py`, and `_latex/pyml.sty`. `.readthedocs.yaml` asks Read the Docs for a PDF
(`formats: [pdf]`).

Locally (any TeX distribution with `xelatex`; tested with MiKTeX on Windows):

```text
python -m sphinx -b latex -E docs/source docs/build/latex
cd docs/build/latex
xelatex -interaction=nonstopmode pyml.tex
makeindex -s python.ist -o pyml.ind pyml.idx
xelatex -interaction=nonstopmode pyml.tex
xelatex -interaction=nonstopmode pyml.tex
```

The result is `docs/build/latex/pyml.pdf`. Delete `docs/build/latex` first if an earlier build
failed. Three `xelatex` runs are needed so that the table of contents, the index and the
cross-references settle. MiKTeX may ask to install missing packages on the first run. `latexmk`
works too, but needs Perl.

### Why the PDF is set up this way

- **Cover and chapter headings** (`_latex/pyml.sty`): the cover replaces Sphinx's `\sphinxmaketitle`
  and the `∇ pyml` wordmark is typeset, not an image. The tagline is `_HOME_DESCRIPTION` from
  `conf.py`, passed by `\newcommand{\pymltagline}` in `"preamble"` before `\usepackage{pyml}`.
  Chapter headings use `titlesec`, so `"fncychap"` is empty. Colors come from the site theme
  (palette at the top of the file).
- **Fonts** (`_latex/pyml.sty`, `"fontpkg": ""` in `conf.py`): only fonts that ship with every TeX
  distribution (TeX Gyre, Latin Modern), so MiKTeX, TeX Live and Read the Docs give the same
  result. Sphinx's default FreeSerif OTF files depend on the system.
- **`latex_use_xindy = False`**: `xindy` is not available on Windows/MiKTeX. `makeindex` ships with
  every distribution.
- **`latex_table_style = ["booktabs"]`**: the default `colorrows` style fails with
  `TeX capacity exceeded` on TeX distributions from 2026 (new `colortbl`), see
  [sphinx-doc/sphinx#14465](https://github.com/sphinx-doc/sphinx/issues/14465). The price is the
  missing grey row shading in tables. When Sphinx fixes it, remove the line and test a PDF build
  on MiKTeX.
- **`"tableofcontents"`** must keep `\sphinxtableofcontents`. Replacing it with only
  `\setcounter{tocdepth}{2}` removes the table of contents.
- **`figure_align: "H"`**: a figure stays right after its code. Without it LaTeX moves figures
  to a later page.
- **toctree position**: in the PDF, pages included by a `toctree` are nested under the section
  that contains it. Keep the toctree in its own last section (see "Next steps" in
  `quick_start/index.rst`), otherwise the pages appear as sub-sections of an unrelated section.
- **`.. only:: html` blocks** (the grid cards) do not exist in the PDF.

## Things that break quietly

- `pyml_hide_bases` is connected with `priority=400` on purpose: `sphinx_autodoc_typehints` also
  rewrites the bases of a class and runs at the default 500. Run later, the hook has no effect.
- The theme shows the left sidebar even when it is empty if the toctrees are `:hidden:` (they are
  here). `pyml_empty_sidebar` exists for that reason; turning `sidebar_includehidden` off would
  empty the sidebar everywhere.
