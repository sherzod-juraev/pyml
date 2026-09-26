"""Configuration file for the Sphinx documentation builder."""

import math
import os
from datetime import datetime

import pyml

# ================================================================================= #
# ----------------------------- Project Configurations ---------------------------- #
# ================================================================================= #
project = "pyml"
year = 2026
current_year = datetime.now().year
year_str = str(year) if current_year == year else f"{year}-{current_year}"
copyright = f"{year_str}, Sherzod Juraev"
author = "Sherzod Juraev"
release = pyml.__version__
version = release
source_suffix = {
    ".rst": "restructuredtext",
}

# ================================================================================= #
# --------------------------- RST Substitutions ----------------------------------- #
# ================================================================================= #
_HOME_DESCRIPTION = (
    "A machine learning library built from scratch on NumPy and SciPy — "
    "no TensorFlow, no PyTorch, just the math."
)

rst_prolog = f"""
.. |home_description| replace:: {_HOME_DESCRIPTION}
.. |nuu_link| replace:: National University of Uzbekistan
.. _nuu_link: https://nuu.uz/en/
"""

# ================================================================================= #
# ------------------------------ Canonical URL ------------------------------------ #
# ================================================================================= #
_READTHEDOCS_CANONICAL_URL = os.environ.get("READTHEDOCS_CANONICAL_URL")
_CANONICAL_URL = _READTHEDOCS_CANONICAL_URL or "https://pyml-edu.readthedocs.io/en/latest/"

# ================================================================================= #
# ----------------------------------- Extensions ---------------------------------- #
# ================================================================================= #
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
    "sphinxcontrib.programoutput",
]

# ================================================================================= #
# ------------------------------------ Autodoc ------------------------------------ #
# ================================================================================= #
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "private-members": False,
    "exclude-members": "_abc_impl,__init__",
    "inherited-members": "object",
    "autosummary": True,
}
autodoc_typehints_format = "short"
autosummary_generate = False

# ================================================================================= #
# ------------------------------------ Napoleon ----------------------------------- #
# ================================================================================= #
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False
napoleon_use_ivar = False
napoleon_custom_sections = [
    ("Attributes", "params_style"),
]

# ================================================================================= #
# ----------------------------- Code Syntax & Styling ----------------------------- #
# ================================================================================= #
toc_object_entries_show_parents = "hide"
pygments_style = "friendly"
default_role = "literal"
add_module_names = False

# ================================================================================= #
# -------------------------- Sphinx Autodoc Typehints ----------------------------- #
# ================================================================================= #
typehints_fully_qualified = False
typehints_use_signature = True
typehints_use_signature_return = True
typehints_defaults = "comma"

# ================================================================================= #
# -------------------------------- Intersphinx ------------------------------------ #
# ================================================================================= #
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ================================================================================= #
# --------------------------------- CopyButton ------------------------------------ #
# ================================================================================= #
copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# ================================================================================= #
# --------------------------------- Not Found ------------------------------------- #
# ================================================================================= #
notfound_urls_prefix = "/en/latest/"

# ================================================================================= #
# --------------------------------- Matplotlib ------------------------------------ #
# ================================================================================= #
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

# ================================================================================= #
# ------------------------------- OpenGraph --------------------------------------- #
# ================================================================================= #
ogp_site_url = _CANONICAL_URL
ogp_image = "_static/branding/og-image.png"
ogp_description_length = 200
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
]

# ================================================================================= #
# ------------------------------- Sphinx Sitemap ---------------------------------- #
# ================================================================================= #
html_baseurl = _CANONICAL_URL
sitemap_url_scheme = "{link}"
sitemap_excludes = [
    "search.html",
    "genindex.html",
]

# ================================================================================= #
# -------------------------- Options For Html Output ------------------------------ #
# ================================================================================= #
templates_path = [
    "_templates",
]
html_extra_path = [
    "_extra",
]
exclude_patterns = []

# ================================================================================= #
# ---------------------------- Theme & Branding ----------------------------------- #
# ================================================================================= #
html_theme = "pydata_sphinx_theme"
html_title = "pyml"
html_context = {
    "default_mode": "light",
}
html_domain_indices = False
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = False

# ================================================================================= #
# --------------------------- Theme Customizations -------------------------------- #
# ================================================================================= #
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

# ================================================================================= #
# ------------------------------------- CSS Files --------------------------------- #
# ================================================================================= #
html_static_path = [
    "_static",
]
html_css_files = [
    "css/header.css",
    "css/cards.css",
    "css/left_sidebar.css",
    "css/right_sidebar.css",
    "css/breadcrumb.css",
    "css/footer_nav.css",
    "pygments.css",
    "css/code_blocks.css",
]

# ================================================================================= #
# ------------------------------ LaTeX / PDF Output ------------------------------- #
# ================================================================================= #
latex_engine = "xelatex"

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \setlength{\headheight}{14pt}
    """,
    "tableofcontents": r"\setcounter{tocdepth}{2}",
    "sphinxsetup": "noteBorderColor={rgb}{0.2,0.4,0.8}, noteBgColor={rgb}{0.93,0.95,1}",
    "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
}
latex_documents = [
    (
        "index",
        "pyml.tex",
        "pyml Documentation",
        "Sherzod Juraev",
        "manual",
    ),
]
latex_show_urls = "no"
latex_domain_indices = False
