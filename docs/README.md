# Documentation

The documentation is a [Sphinx](https://www.sphinx-doc.org/) project with the
[PyData theme](https://pydata-sphinx-theme.readthedocs.io/). Everything lives in `docs/source/`.
This file is a map: what is where, and why a few things are set up the way they are.

How to run the checks and the live preview is described in [CONTRIBUTING.md](../CONTRIBUTING.md)
(`nox -s docs`, "Live documentation"). How to add a page, an API entry or an example is in
[AUTHORING.md](AUTHORING.md). Documenting a new model is step 6 of "Adding a model" in
CONTRIBUTING.md.

## Layout

| Path (in `docs/source/`)  | What it holds                                                             |
|:--------------------------|:--------------------------------------------------------------------------|
| `conf.py`                 | Every Sphinx setting. Custom extensions are put on `sys.path` and enabled here. |
| `index.rst`, `getting-started.rst`, `about.rst`, `404.rst` | Top-level pages. The `toctree` of `index.rst` sets the order of the header links. |
| `examples/`               | `index.rst` (tracked) and one generated folder per family, made by sphinx-gallery from the scripts in `<repo>/examples/` (ignored by git). |
| `api-reference/`          | API reference: one folder per family with one page per class (`autoclass`). The order is deliberate, see "Known trade-offs" below. |
| `changelog/`              | One file per release.                                                     |
| `_ext/`                   | Custom Sphinx extensions (below).                                         |
| `_plot_styles/`           | `light.mplstyle` and `dark.mplstyle`: colors of the plots.                |
| `_latex/pyml.sty`         | Style of the PDF: fonts, colors, cover page, chapter headings.            |
| `_static/`                | `css/` (site styling), `fonts/` (self-hosted Inter and JetBrains Mono, OFL licenses), `branding/` (logos, social image, author monogram). |
| `_extra/`                 | Files copied to the site root as they are (`html_extra_path`).            |

## Custom extensions (`_ext/`)

Each module has a docstring that explains its usage and limits. In short:

| Extension            | What it does                                                          | Config (`conf.py`)   |
|:---------------------|:----------------------------------------------------------------------|:---------------------|
| `pyml_hide_bases`    | Hides internal base classes from the "Bases:" line of a class: names starting with `_` and the names in `pyml_hidden_bases`. A hidden base is replaced by its own parents, one level only. | `pyml_hidden_bases` |
| `pyml_themed_plot`   | Replaces the `plot` directive: every block is drawn twice, once with `light.mplstyle` and once with `dark.mplstyle`, and the theme shows the one that matches the visitor's mode. Other builders (LaTeX, text) get the light image only. | `pyml_plot_styles` |
| `pyml_empty_sidebar` | Hides the empty left sidebar on pages whose top-level section has no sub-pages. | none |
| `pyml_dataset_info`  | Adds `.. dataset-info:: <name>`: a table, notes and the citation of a `pyml-datasets` dataset, read from the package while the docs are built. | none |
| `pyml_gallery_scraper` | Light and dark images for the sphinx-gallery examples (the gallery counterpart of `pyml_themed_plot`). | `image_scrapers` and `reset_modules` in `sphinx_gallery_conf` |

The extensions are plain modules, not a package. `conf.py` appends `_ext/` to `sys.path`, so
`extensions = [..., "pyml_hide_bases"]` can import them by name.

### Plots

Write `.. plot::` as usual. Text, axes and grid colors follow the theme through the two style
files. Sizes and fonts are shared by both themes and live in `plot_rcparams` in `conf.py`. A color
that carries meaning (for example the red centroids of KMeans) may be set explicitly in the plot
code. The `:context:` option of `plot` is not supported by `pyml_themed_plot`.

Editing a style file or `plot_rcparams` redraws every plot on the next build. The plots of the
examples in `examples/` are not made by this directive; they come from `pyml_gallery_scraper`
and use the same style files, see [AUTHORING.md](AUTHORING.md#light-and-dark-plots).

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
  that contains it. The toctrees of the top-level pages are `:hidden:` and stand before the
  first section; a visible toctree must be in its own last section, otherwise the pages appear
  as sub-sections of an unrelated section.
- **`.. only:: html` blocks** (the grid cards) do not exist in the PDF.

## Things that break quietly

- `pyml_hide_bases` is connected with `priority=400` on purpose: `sphinx_autodoc_typehints` also
  rewrites the bases of a class and runs at the default 500. Run later, the hook has no effect.
- The theme shows the left sidebar even when it is empty if the toctrees are `:hidden:` (they are
  here). `pyml_empty_sidebar` exists for that reason; turning `sidebar_includehidden` off would
  empty the sidebar everywhere.

## Known trade-offs

Decisions where the obvious fix was not taken, and why. They are written for the maintainer; the
public site has no "known limitations" page because none of these is something a user of the
library needs to work around.

- **Branding files are duplicated, not shared.** `logo.svg` and `logo-dark.svg` exist in
  `assets/branding/` (used by the GitHub README's light/dark `<picture>` switch) and in
  `docs/source/_static/branding/` (used by Sphinx). pydata-sphinx-theme checks that the logo file
  exists directly under Sphinx's own `_static/` before the extra `html_static_path` entries are
  resolved, so one shared location is not possible. Keep the two copies in sync by hand when the
  logo changes.
- **API Reference is not alphabetical.** Pages under `api-reference/` go supervised, then unsupervised,
  then utilities (Core, Linear Model, Naive Bayes, Neighbors, Tree, Cluster, Preprocessing, Model
  Selection, Metrics). The `examples/` gallery uses the same order, for consistency.
- **Only integer hyper-parameters are checked for type.** `fit` rejects out-of-range values with
  `InvalidParameterError` (a negative `alpha`, `n_clusters=0`), and integer parameters are also
  checked for type, so `n_clusters=2.5`, `max_iter="10"` and `max_depth=3.0` are rejected too.
  Float and string parameters get no dedicated type check: `alpha="1"` fails with a plain
  `TypeError` from the comparison. Type checks on every constructor were left out to keep the
  estimators small. `fit_intercept` and `outlier_label` are not validated at all, since any value
  is meaningful for them.
- **`train_test_split` raises `ValueError`, not `InvalidParameterError`.** It is a function, not an
  estimator, so it does not use the `_validate_params` hook; it validates its own arguments and
  raises `ValueError` (and `TypeError`). `InvalidParameterError` is not a subclass of
  `ValueError`, so code that handles errors from both an estimator's `fit` and `train_test_split`
  has to catch both. A NaN `test_size` or a negative `random_state` is also a plain `ValueError`.
  This may be unified in a later release. The `Raises` section of the docstring is the public
  documentation of this behaviour.
