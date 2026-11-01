# Authoring the documentation

How to add and change things in the docs. [docs/README.md](README.md) is the map of the
documentation (layout, extensions, the PDF build, known trade-offs); this file holds the recipes.
Running the checks and the live preview is described in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Where things go

| I want to ...                          | I change                                                                 |
|:---------------------------------------|:-------------------------------------------------------------------------|
| Document a new class                   | `docs/source/api-reference/<family>/`, see [API reference](#api-reference) |
| Add a worked example for a model       | `examples/<family>/<model>.py`, see [Examples](#examples)                |
| Add a whole new family                 | both of the above, plus a card and a toctree entry in the two index pages |
| Change the order of the header links   | the `toctree` in `docs/source/index.rst`                                 |
| Write a release page                   | `docs/source/changelog/X.Y.Z.rst`, see "Changelog" in CONTRIBUTING.md    |
| Change colors or fonts of the site     | `docs/source/_static/css/`, see [Styling](#styling)                      |
| Change the colors of the plots         | `docs/source/_plot_styles/light.mplstyle` and `dark.mplstyle`            |
| Describe a dataset on a page           | `.. dataset-info:: <name>`, see [Examples](#examples)                    |

The top-level pages, in the order of the header: Getting Started (`getting-started.rst`),
Examples (`examples/`), API Reference (`api-reference/`), Changelog (`changelog/`) and About
(`about.rst`). The order is the order of the `toctree` in `docs/source/index.rst`.

All folder and file names in `docs/source/` use hyphens (`linear-model`, `elastic-net`), because
they become the URLs. Python names (`pyml.linear_model`) keep their underscores.

## API reference

One folder per family in `docs/source/api-reference/`, with an `index.rst` and one page per
class. Copy a page of the same family and change the class name, so that the heading, the
`autoclass` options and the plot are the same everywhere.

- Add the page to the `toctree` of the family's `index.rst`, and to the family's grid. A page
  that is in no toctree makes the build fail (`-W`).
- The content comes from the class docstring, which is where the math, the parameters and the
  `.. plot::` demo are written, see "Docstrings" in CONTRIBUTING.md. The page itself stays short.
- A new family also needs a card in `docs/source/api-reference/index.rst`.
- Cards in an index page are written as header, body and footer button:

  ```rst
  .. grid-item-card::
      :text-align: center
      :shadow: md

      Title
      ^^^^^
      One sentence about what is inside.

      +++
      .. button-ref:: link
          :ref-type: doc
          :color: primary
          :expand:

          Open Title
  ```

  The grid is `1 2 2 2` columns (`1 2 2 3` when a page has three or more cards), with
  `:gutter: 3` and `:padding: 0`.

## Examples

An example is a plain Python script in `examples/<family>/`, not a page. The build runs it with
[sphinx-gallery](https://sphinx-gallery.github.io/) and writes the page, so the code, its output
and its plots on the page always agree. The generated pages go to
`docs/source/examples/<family>/` and are ignored by git.

Why the scripts are in `examples/` and not in `docs/`: they are real Python files, so the IDE,
ruff and mypy work on them, and they are checked like the rest of the code (`nox -s lint` and
`nox -s types`; interrogate skips them).

### Writing an example

Copy [examples/cluster/kmeans.py](../examples/cluster/kmeans.py). The structure, one "scene" of
the learning process per cell (`# %%` starts a cell, and the comment block under it becomes text
on the page):

1. **Module docstring**: title (`=` underline), what the model does in a few sentences, when to
   use it, and links to the class and to the neighbouring models with `:class:`.
2. **Parameters**: the parameters in plain words, grouped by what they decide. Do not repeat the
   default values; they are in the API reference and would go out of date in two places. Link to
   the class.
3. **Dataset**: one sentence on how the data is used, then `.. dataset-info:: <name>`.
4. **Preparation**: scaling or any other step, and why the model needs it.
5. **Fit**: fit the model and show what it learned, with a plot.
6. **A scene that checks the result** (compare with the known answer, a score, the effect of a
   hyper-parameter).
7. **Pitfall**, when the model has a real one (for example, K Means without scaling).

Rules:

- No cell titles with a number, no "Step 1". The titles say what happens.
- The first code cell holds all imports: standard library, then third party (`pyml_datasets`
  is third party), then `pyml`. This is what ruff's isort rule expects.
- Randomness only through an explicit `random_state`, so the page is the same on every build.
- End every cell that draws with `plt.show()`. Without it the last expression is printed
  (`<matplotlib.legend.Legend ...>`) under the figure.
- The scripts are type-checked with mypy in strict mode. Annotate helper functions.
- A failing example fails the build (`abort_on_example_error`).
- Read `get_info(name).categorical` before choosing a dataset for a distance-based model: a
  categorical feature is stored as integer codes, and the model would treat the codes as numbers.
  `auto_mpg` and `penguins` contain `NaN`; pyml does not impute, so drop or avoid them.
- Files that start with an underscore (`_helpers.py`) are not run as examples.
- **Bar charts with text labels** are drawn at numeric positions and named with
  `ax.set_yticks(positions, labels)`, not with `ax.barh(["a", "b"], ...)`. Python 3.14 cannot
  pickle a figure with a categorical axis, so the dark copy of such a figure is not made and the
  build warns.
- The card of an example shows its first figure. Another one is chosen with
  `# sphinx_gallery_thumbnail_number = N` in the first code cell. Every figure is saved twice
  (light and dark), so the light copy of figure `n` is number `2 * n - 1`: figure 2 is `3`.

Run one script on its own:

```bash
uv run --group examples python examples/cluster/kmeans.py
```

### Describing a dataset

`.. dataset-info:: wine` asks `pyml_datasets` for the dataset while the docs are built and
prints a table (task, shape, features, classes with counts, categorical features, missing
values, license and source), the notes of the dataset, and a collapsed "Citation, credits and
changes made to the data" block. Nothing is typed by hand. The datasets are CC BY 4.0 (or CC0),
which asks for attribution and a statement of the changes, so the block must stay. An unknown
name fails the build and lists the available ones.

### A new example

1. Add the script to `examples/<family>/`.
2. Add its file name (without `.py`) to `EXAMPLE_ORDER` in `docs/source/_ext/pyml_gallery_scraper.py`.
   This sets the order in the sidebar, and a script that is not in the list goes last.
3. If the family has groups, add it to a group in `pyml_gallery_groups` in `conf.py` (see below).
   The build warns about an example that is in no group.

### A new family

1. Create `examples/<family>/GALLERY_HEADER.rst` and the first script. The header is the title,
   a short introduction and, when the family has many pages, a "Where to start" list.
   `conf.py` finds every folder of `examples/` that has a script, nothing to register there.
2. Add `<family>/index` to the `toctree` and a card to `docs/source/examples/index.rst`. Without
   them the build warns that the generated index is in no toctree.

### Groups on a family page

sphinx-gallery writes the cards of a family as one grid. `pyml_gallery_groups` in `conf.py`
splits them under headings, each with a sentence of its own:

```python
pyml_gallery_groups = {
    "linear-model": [
        ("Regression", "Predict a number.", ["linear-regression", "ridge"]),
        ("Classification", "Predict a class.", ["logistic-regression"]),
    ],
}
```

A family that is not in the dictionary keeps the single grid. The work is done by
`docs/source/_ext/pyml_gallery_groups.py`, which rewrites the generated `index.rst` when Sphinx
reads it. It depends on the format that sphinx-gallery writes, so it warns (and shows the single
grid) when that format is not what it expects, for example after a sphinx-gallery upgrade.

### Light and dark plots

Figures in a script need no special code. `reset_matplotlib` applies the plot style before each
example, and `themed_scraper` saves every figure twice, light and dark, in containers with the
classes `only-light` and `only-dark`; the theme shows the one that matches the visitor's mode.
Both live in `docs/source/_ext/pyml_gallery_scraper.py` and read the colors from the same two
`.mplstyle` files as the `.. plot::` directive of the API reference
(`docs/source/_ext/pyml_themed_plot.py`). A color that carries meaning (the red centroids) may be
set in the plot code.

### When an example is run again

sphinx-gallery runs a script again only when the file changes (it compares an MD5). It does not
notice that **pyml** or **pyml-datasets** changed. After such a change, delete the generated
folder of the family, not `docs/source/examples/` itself, because `index.rst` there is tracked:

```bash
# Windows (PowerShell)
Remove-Item -Recurse -Force docs\source\examples\cluster
# Linux / macOS
rm -rf docs/source/examples/cluster
```

To take a newer pyml-datasets, update the lock first, `uv lock --upgrade-package pyml-datasets`
(the source is the `main` branch, `uv.lock` pins the commit), then delete the generated folders.
Do this before every release, so the examples show the current data and notes.

## Plots outside the gallery

`.. plot::` in a docstring or an `.rst` page works as usual, and is drawn in the light and the
dark theme by `pyml_themed_plot`. See "Plots" in [docs/README.md](README.md#plots).

## Styling

One CSS file per part of the site, all listed in `html_css_files` in `conf.py`:

| File                | What it styles                                                                |
|:--------------------|:------------------------------------------------------------------------------|
| `fonts.css`         | Self-hosted Inter and JetBrains Mono.                                         |
| `header.css`        | The top navigation, the logo, the search field.                               |
| `left_sidebar.css`  | The section tree on the left.                                                 |
| `right_sidebar.css` | "On this page" on the right.                                                  |
| `breadcrumb.css`    | The breadcrumb above the title.                                               |
| `footer_nav.css`    | The previous / next links at the bottom of a page.                            |
| `cards.css`         | The grid cards of `sphinx-design`.                                            |
| `code_blocks.css`   | Code blocks and the copy button (colors of the syntax are in `pygments.css`). |
| `gallery.css`       | The pages made by sphinx-gallery: images, thumbnails, download buttons.       |
| `about.css`         | The author's photo on the About page.                                         |

A link in the header, in the left sidebar and in "On this page" shows three different signs,
and they are the same everywhere:

- **hover**: teal text and an underline (you are about to go there),
- **current page**: teal text only, no background, no border, no bar (you are here),
- **keyboard focus**: an outline.

Hover rules are inside `@media (hover: hover)`, so touch screens do not keep a hover state.
The theme draws its own borders and backgrounds on these links, so the rules use `!important`
and `content: none` for the theme's pseudo-elements; do not remove them when tidying.

## Things that break quietly

- **Windows paths in `conf.py`.** sphinx-gallery matches `filename_pattern` against the full
  path, which has `\` on Windows. A pattern with only `/` matches nothing there: the page is
  generated but the script does not run, and the build prints
  `executed 0 out of 0 files`. The patterns in `conf.py` allow both separators.
- **A new example family that is in no toctree** fails the build with `-W`.
- **`uv sync --group <name>` removes the other groups** from `.venv`. Use
  `uv sync --all-groups` to get everything back.
- **Old results after a library change**: see "When an example is run again" above.

## Checks

```bash
uv run nox -s docs      # sphinx-lint, doc8, rstcheck and the build with warnings as errors
uv run nox -s lint      # ruff, includes the example scripts
uv run nox -s types     # mypy strict, includes the example scripts
```

Read the Docs builds with `fail_on_warning: true`, so a warning that you ignore locally fails
there. Open a changed page in both the light and the dark mode before you commit.
