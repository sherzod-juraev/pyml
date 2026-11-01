"""Configuration file for the Sphinx documentation builder."""

import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyml

sys.path.append(str(Path(__file__).parent / "_ext"))

# --- Project Configurations --- #
project = "pyml"
year = 2026
current_year = datetime.now(UTC).year
year_str = str(year) if current_year == year else f"{year}-{current_year}"
author = pyml.__author__
copyright = f"{year_str}, {author}"
release = pyml.__version__
version = release
source_suffix = {
    ".rst": "restructuredtext",
}

# --- RST Substitutions --- #
_HOME_DESCRIPTION = (
    "A machine learning library built from scratch on NumPy and SciPy — "
    "no TensorFlow, no PyTorch, just the math."
)

rst_prolog = f"""
.. |home_description| replace:: {_HOME_DESCRIPTION}
.. |nuu_link| replace:: National University of Uzbekistan
.. _nuu_link: https://nuu.uz/en/
"""

# --- Canonical URL --- #
_READTHEDOCS_CANONICAL_URL = os.environ.get("READTHEDOCS_CANONICAL_URL")
_CANONICAL_URL = _READTHEDOCS_CANONICAL_URL or "https://pyml-edu.readthedocs.io/en/latest/"

# --- Extensions --- #
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "notfound.extension",
    "matplotlib.sphinxext.plot_directive",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "autodocsumm",
    "sphinx_gallery.gen_gallery",
    # custom extensions
    "pyml_hide_bases",
    "pyml_themed_plot",
    "pyml_empty_sidebar",
    "pyml_dataset_info",
    "pyml_gallery_scraper",
    "pyml_gallery_groups",
]

# --- Nitpicky --- #
nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy.typing.NDArray"),
]

# --- Autodoc --- #
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "private-members": False,
    "exclude-members": "_abc_impl,__init__",
    "inherited-members": "object",
    "autosummary": True,
}
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
autosummary_generate = False

# --- Napoleon --- #
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_use_ivar = False
napoleon_custom_sections = [
    ("Attributes", "params_style"),
]

# --- Code Syntax & Styling --- #
toc_object_entries_show_parents = "hide"
pygments_style = "friendly"
default_role = "literal"
add_module_names = False

# --- Sphinx Autodoc Typehints --- #
typehints_fully_qualified = False
typehints_use_signature = True
typehints_use_signature_return = True
typehints_defaults = "comma"

# --- Intersphinx --- #
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# --- CopyButton --- #
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# --- Matplotlib --- #
plot_include_source = True
plot_html_show_source_link = False
plot_formats = [
    ("png", 100),
    "pdf",
]

phi = (math.sqrt(5) + 1) / 2
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
ogp_site_url = _CANONICAL_URL
ogp_image = "_static/branding/og-image.png"
ogp_description_length = 200
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
]

# --- Sphinx Sitemap --- #
html_baseurl = _CANONICAL_URL
sitemap_url_scheme = "{link}"
sitemap_excludes = [
    "search.html",
    "genindex.html",
]

# --- Options For Html Output --- #
templates_path = [
    "_templates",
]
html_extra_path = [
    "_extra",
]
exclude_patterns = []

# --- Theme & Branding --- #
html_theme = "pydata_sphinx_theme"
html_title = "pyml"
html_context = {
    "default_mode": "light",
}
html_domain_indices = False
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = False

# --- Theme Customizations --- #
html_theme_options = {
    "logo": {
        "image_light": "_static/branding/logo.svg",
        "image_dark": "_static/branding/logo-dark.svg",
        "text": "",
    },
    "github_url": "https://github.com/sherzod-juraev/pyml",
    "show_prev_next": True,
    "navigation_with_keys": True,
    "collapse_navigation": True,
    "show_nav_level": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# --- CSS Files --- #
html_static_path = [
    "_static",
]
html_css_files = [
    "css/fonts.css",
    "css/header.css",
    "css/cards.css",
    "css/left_sidebar.css",
    "css/right_sidebar.css",
    "css/breadcrumb.css",
    "css/footer_nav.css",
    "css/code_blocks.css",
    "css/gallery.css",
    "css/about.css",
]

# --- LaTeX / PDF Output --- #
latex_engine = "xelatex"

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "extraclassoptions": "openany,oneside",
    "figure_align": "H",
    "fontpkg": "",
    "preamble": r"\newcommand{\pymltagline}{" + _HOME_DESCRIPTION + r"}" + r"\usepackage{pyml}",
    "tableofcontents": r"\setcounter{tocdepth}{2}\sphinxtableofcontents",
    "fncychap": "",
}
latex_documents = [
    (
        "index",
        "pyml.tex",
        "pyml Documentation",
        author,
        "manual",
    ),
]
latex_additional_files = [
    "_latex/pyml.sty",
]
latex_show_urls = "no"
latex_domain_indices = False
latex_use_xindy = False
latex_table_style = ["booktabs"]

# --- Autodoc Base Filtering --- #
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
    "abort_on_example_error": True,
    # Light and dark images, see _ext/pyml_gallery_scraper.py.
    "image_scrapers": ("pyml_gallery_scraper.themed_scraper",),
    "reset_modules": ("matplotlib", "pyml_gallery_scraper.reset_matplotlib"),
    # Links from the API reference to the examples that use a class (.. minigallery::).
    "doc_module": ("pyml",),
    "backreferences_dir": "examples/backreferences",
    "remove_config_comments": True,
    "download_all_examples": False,
    "show_signature": False,
    "show_memory": False,
    "write_computation_times": False,
}
