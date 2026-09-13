"""Configuration file for the Sphinx documentation builder."""

import math
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
]

# ================================================================================= #
# ------------------------------------ Autodoc ------------------------------------ #
# ================================================================================= #
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "private-members": False,
    "exclude-members": "_abc_impl,__init__",
}
autodoc_typehints_format = "short"

# ================================================================================= #
# ------------------------------------ Napoleon ----------------------------------- #
# ================================================================================= #
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

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
ogp_site_url = "https://pyml-edu.readthedocs.io"
ogp_description_length = 200
ogp_type = "website"

# ================================================================================= #
# -------------------------- Options For Html Output ------------------------------ #
# ================================================================================= #
templates_path = [
    "_templates",
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
    "github_url": "https://github.com/sherzod-juraev/pyml",
    "show_prev_next": True,
    "navigation_with_keys": True,
    "collapse_navigation": True,
    "show_nav_level": False,
    "navbar_end": ["navbar-icon-links"],
}

# ================================================================================= #
# ------------------------------------- CSS Files --------------------------------- #
# ================================================================================= #
html_static_path = ["_static"]
html_css_files = [
    "header.css",
    "cards.css",
    "left_sidebar.css",
    "right_sidebar.css",
    "breadcrumb.css",
    "footer_nav.css",
    "pygments.css",
    "code_blocks.css",
    "tab_set.css",
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
