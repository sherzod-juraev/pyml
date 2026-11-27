"""Configuration file for the Sphinx documentation builder."""

import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyml

# The custom extensions live in docs/source/_ext/. This makes them importable.
sys.path.append(str(Path(__file__).parent / "_ext"))

# --- Project Configurations --- #
project = "pyml"
# The first year of the project. The copyright shows a range such as "2026-2027" once the
# current year is later.
year = 2026
current_year = datetime.now(UTC).year
year_str = str(year) if current_year == year else f"{year}-{current_year}"
author = pyml.__author__
copyright = f"{year_str}, {author}"
release = pyml.__version__
# The full version and the short version are the same.
version = release

# --- RST Substitutions --- #
_HOME_DESCRIPTION = (
    "A machine learning library built from scratch on NumPy and SciPy — "
    "no TensorFlow, no PyTorch, just the math."
)

# Text that Sphinx adds to the start of every .rst file, so the substitutions |home_description|
# and |nuu_link| and the link target nuu_link can be used on any page.
rst_prolog = f"""
.. |home_description| replace:: {_HOME_DESCRIPTION}
.. |nuu_link| replace:: National University of Uzbekistan
.. _nuu_link: https://nuu.uz/en/
"""

# --- Canonical URL --- #
# Read the Docs sets READTHEDOCS_CANONICAL_URL on its builds. A local build uses the latest
# version of the site.
_READTHEDOCS_CANONICAL_URL = os.environ.get("READTHEDOCS_CANONICAL_URL")
_CANONICAL_URL = _READTHEDOCS_CANONICAL_URL or "https://pyml-edu.readthedocs.io/en/latest/"

# --- Extensions --- #
extensions = [
    "sphinx.ext.autodoc",  # the API pages, from the docstrings
    "sphinx.ext.intersphinx",  # links to the Python, NumPy and SciPy documentation
    "sphinx.ext.napoleon",  # the numpy-style sections of the docstrings
    "sphinx.ext.viewcode",  # [source] links to the highlighted code
    "sphinx_autodoc_typehints",  # type hints in the descriptions of the parameters
    "sphinx_copybutton",  # a copy button on the code blocks
    "sphinx_design",  # cards, grids, tabs and dropdowns
    "notfound.extension",  # the 404 page
    "matplotlib.sphinxext.plot_directive",  # the .. plot:: directive
    "sphinxext.opengraph",  # preview tags for a link that is shared in a social network
    "sphinx_sitemap",  # sitemap.xml for the search engines
    "autodocsumm",  # the summary tables at the top of the class pages
    "sphinx_gallery.gen_gallery",  # the gallery of examples, built from examples/
    # custom extensions
    "pyml_hide_bases",  # hides the internal base classes from the "Bases:" lines
    "pyml_themed_plot",  # draws each .. plot:: once per colour theme (light and dark)
    "pyml_empty_sidebar",  # hides the primary sidebar on a section without sub-pages
    "pyml_dataset_info",  # the .. dataset-info:: directive
    "pyml_gallery_scraper",  # light and dark images, and the order of the gallery examples
    "pyml_gallery_groups",  # the cards on the page of an example family
]

# --- Nitpicky --- #
# Warn about every cross-reference that cannot be resolved. The docs build runs with -W, so a
# warning fails it.
nitpicky = True
# References that are known not to resolve and are accepted: numpy.typing.NDArray, the type of the
# arrays in the signatures, numpy.float64 and numpy.int64, the element types of the arrays that
# `cross_val_score` and `KFold.split` return (the NumPy inventory lists them as attributes, not
# classes), and P, the type variable of `clone`, which has no page (the signature
# `clone[P: SupportsParams]` shows its bound).
nitpick_ignore = [
    ("py:class", "numpy.typing.NDArray"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int64"),
    ("py:class", "P"),
]

# --- Autodoc --- #
autodoc_default_options = {
    # The members in the order of the source file, not alphabetical.
    "member-order": "bysource",
    # A member without a docstring and a private member (_name) are not documented.
    "undoc-members": False,
    "private-members": False,
    # Left out: _abc_impl (an internal attribute of the abstract classes) and __init__.
    "exclude-members": "_abc_impl,__init__",
    # Document the inherited members, except those that come from `object`.
    "inherited-members": "object",
    # The summary table (autodocsumm) at the top of each class.
    "autosummary": True,
}
# Display the signature of a class as a method, not together with the class name.
autodoc_class_signature = "separated"
# No stub pages from `autosummary` directives. autodocsumm draws the summary tables.
autosummary_generate = False

# --- Napoleon --- #
# The docstrings are in the numpy style only.
napoleon_google_docstring = False
# The return type stays in the description of "Returns", not in a separate field.
napoleon_use_rtype = False
# An "Attributes" section is rendered like a "Parameters" section: a list of names and types.
napoleon_custom_sections = [
    ("Attributes", "params_style"),
]

# --- Code Syntax & Styling --- #
# In the table of contents of a page, show the name of an object without its parents (`fit`, not
# `Ridge.fit`).
toc_object_entries_show_parents = "hide"
# The colours of the highlighted code.
pygments_style = "friendly"
# `text` in single backticks is shown as code, like ``text``.
default_role = "literal"
# `Ridge`, not `pyml.linear_model.Ridge`, in the headings of the API pages.
add_module_names = False

# --- Sphinx Autodoc Typehints --- #
# The types of the parameters, and the return type, appear in the signature as well as in the
# fields below it.
typehints_use_signature = True
typehints_use_signature_return = True
# The default of a parameter follows its type, after a comma.
typehints_defaults = "comma"

# --- Intersphinx --- #
# The documentation of other projects that the cross-references can point to.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# --- CopyButton --- #
# The copy button leaves out the prompts ">>> " and "$ " (a regular expression) from the copied
# text.
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True

# --- Matplotlib --- #
# Show the source code under each plot, without the link to the source file.
plot_include_source = True
plot_html_show_source_link = False
# Each plot is saved as a png (100 dpi) and as a pdf.
plot_formats = [
    ("png", 100),
    "pdf",
]

# The golden ratio: a plot is 3 high and 3 * phi wide.
phi = (math.sqrt(5) + 1) / 2
# The matplotlib settings of the .. plot:: directive: small text (8 pt) for a small figure, the
# margins of the axes, and no LaTeX.
plot_rcparams = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

# --- OpenGraph --- #
# The address of the site, and the image of the link preview.
ogp_site_url = _CANONICAL_URL
ogp_image = "_static/og-image.png"

# --- Sphinx Sitemap --- #
# The address that the sitemap puts before each page. The canonical URL already holds the language
# and the version, so the default scheme "{lang}{version}{link}" is shortened to the link.
html_baseurl = _CANONICAL_URL
sitemap_url_scheme = "{link}"
# Pages that do not belong in a search engine.
sitemap_excludes = [
    "search.html",
    "genindex.html",
]

# --- Options For Html Output --- #
# The folder of custom page templates (it does not exist at present).
templates_path = [
    "_templates",
]
# Files that are copied unchanged to the root of the site (the verification file of Google Search
# Console).
html_extra_path = [
    "_extra",
]

# --- Theme & Branding --- #
html_theme = "pydata_sphinx_theme"
# The title of the site, in the browser tab and in the page titles.
html_title = "pyml"
# The theme that a visitor sees first: light.
html_context = {
    "default_mode": "light",
}
# No generated index pages of the modules.
html_domain_indices = False
# The format of the "last updated" date in the footer.
html_last_updated_fmt = "%b %d, %Y"
# Do not copy the .rst sources to the site.
html_copy_source = False

# --- Theme Customizations --- #
# The options of pydata_sphinx_theme.
html_theme_options = {
    # The logo in the light and in the dark mode, and no text next to it.
    "logo": {
        "image_light": "../../assets/branding/logo.svg",
        "image_dark": "../../assets/branding/logo-dark.svg",
        "text": "",
    },
    # The GitHub icon in the navigation bar.
    "github_url": "https://github.com/sherzod-juraev/pyml",
    # The links to the previous and the next page, at the end of a page.
    "show_prev_next": True,
    # The arrow keys switch to the previous and the next page.
    "navigation_with_keys": True,
    # The sections of the left navigation are collapsed, and open when they are clicked.
    "collapse_navigation": True,
    # No level of the left navigation is open at the start.
    "show_nav_level": False,
    # The right end of the navigation bar: the light/dark switch and the icon links (GitHub).
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# --- CSS Files --- #
# Folders whose files are copied to _static: the stylesheets and the branding images.
html_static_path = [
    "_static",
    "../../assets/branding",
]
# The stylesheets of pyml, one per part of the page (the name tells which).
html_css_files = [
    "css/fonts.css",
    "css/header.css",
    "css/cards.css",
    "css/tabs.css",
    "css/left_sidebar.css",
    "css/right_sidebar.css",
    "css/breadcrumb.css",
    "css/footer_nav.css",
    "css/code_blocks.css",
    "css/gallery.css",
    "css/about.css",
]

# --- LaTeX / PDF Output --- #
# The program that builds the PDF. The settings of the page are in _latex/pyml.sty, see "PDF" in
# docs/README.md.
latex_engine = "xelatex"

latex_elements = {
    # A4 paper, 11 pt text.
    "papersize": "a4paper",
    "pointsize": "11pt",
    # A chapter may start on any page, and the pages are one-sided.
    "extraclassoptions": "openany,oneside",
    # A figure stays exactly where it is written.
    "figure_align": "H",
    # No font package from Sphinx.
    "fontpkg": "",
    # The tagline for the title page, and the style file of pyml.
    "preamble": r"\newcommand{\pymltagline}{" + _HOME_DESCRIPTION + r"}" + r"\usepackage{pyml}",
    # The table of contents lists two levels.
    "tableofcontents": r"\setcounter{tocdepth}{2}\sphinxtableofcontents",
    # No fncychap style for the chapter headings.
    "fncychap": "",
}
# (start page, file name, title, author, document class): one manual that starts at the index.
latex_documents = [
    (
        "index",
        "pyml.tex",
        "pyml Documentation",
        author,
        "manual",
    ),
]
# Files that are copied next to the .tex file.
latex_additional_files = [
    "_latex/pyml.sty",
]
# No generated index pages of the modules.
latex_domain_indices = False
# Tables with the rules of the booktabs package and coloured rows.
latex_table_style = ["booktabs", "colorrows"]

# --- Autodoc Base Filtering --- #
# Base classes that pyml_hide_bases leaves out of the "Bases:" lines, see _ext/pyml_hide_bases.py.
# The private bases (_LinearModelBase and the like) are hidden too.
pyml_hidden_bases = {"BaseEstimator", "DataValidatorMixin"}

# --- Sphinx-gallery --- #
# The example scripts live in <repo>/examples/<family>/ and are plain Python files. Every
# family that has at least one script becomes a gallery in docs/source/examples/<family>/
# (generated, ignored by git). A new family also needs its card and its toctree entry in
# docs/source/examples/index.rst, otherwise the build warns.
_EXAMPLES = Path(__file__).parents[2] / "examples"
_FAMILIES = (
    sorted(p.name for p in _EXAMPLES.iterdir() if any(p.glob("*.py")))
    if _EXAMPLES.is_dir()
    else []
)
# Groups of the example cards on the index page of a family, see _ext/pyml_gallery_groups.py.
# (heading, one sentence, scripts in the order of the cards). A family without an entry keeps
# one grid. Every script of a grouped family must be in a group, otherwise the build warns.
pyml_gallery_groups = {
    "linear-model": [
        (
            "Regression",
            "Predict a number. Start with Linear Regression, then add a penalty on the weights.",
            ["linear-regression", "ridge", "lasso", "elastic-net"],
        ),
        (
            "Classification",
            "Predict one of two classes. The same four models, with the sigmoid on top.",
            [
                "logistic-regression",
                "ridge-classifier",
                "lasso-classifier",
                "elastic-net-classifier",
            ],
        ),
    ],
}

sphinx_gallery_conf = {
    # Order of the examples of a family, see EXAMPLE_ORDER in _ext/pyml_gallery_scraper.py.
    "within_subsection_order": "pyml_gallery_scraper.ExampleOrder",
    "examples_dirs": [f"../../examples/{name}" for name in _FAMILIES],
    "gallery_dirs": [f"examples/{name}" for name in _FAMILIES],
    # Every script runs, except the ones that start with an underscore (helpers).
    # filename_pattern is matched against the full path, which has "\\" on Windows and "/"
    # elsewhere, so both separators are allowed. ignore_pattern sees the bare file name.
    "filename_pattern": r"[\\/][^_\\/][^\\/]*\.py$",
    "ignore_pattern": r"^_",
    # An example that raises stops the build, so the docs never show a broken example.
    "abort_on_example_error": True,
    # Light and dark images, see _ext/pyml_gallery_scraper.py.
    "image_scrapers": ("pyml_gallery_scraper.themed_scraper",),
    "reset_modules": ("matplotlib", "pyml_gallery_scraper.reset_matplotlib"),
    # Links from the API reference to the examples that use a class (.. minigallery::).
    "doc_module": ("pyml",),
    "backreferences_dir": "examples/backreferences",
    # The "# sphinx_gallery_..." comments are not shown in the code of an example.
    "remove_config_comments": True,
    # No zip of all the examples to download.
    "download_all_examples": False,
    # No "Generated by Sphinx-Gallery" line under an example.
    "show_signature": False,
    # No page with the run time of each example.
    "write_computation_times": False,
}
