"""Configuration file for the Sphinx documentation builder."""

import pyml

# ================================================================================= #
# ----------------------------- Project Configurations ---------------------------- #
# ================================================================================= #
project = "pyml"
copyright = "2026, Sherzod Juraev"
author = "Sherzod Juraev"
release = pyml.__version__
version = release

# ================================================================================= #
# ----------------------------------- Extensions ---------------------------------- #
# ================================================================================= #
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "notfound.extension",
    "matplotlib.sphinxext.plot_directive",
]

# ================================================================================= #
# ------------------------------------ Autodoc ------------------------------------ #
# ================================================================================= #
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "private-members": False,
    "exclude-members": "_abc_impl",
}
autodoc_typehints_format = "short"

# ================================================================================= #
# ------------------------------------ Napoleon ----------------------------------- #
# ================================================================================= #
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
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
    "preamble": "",
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
