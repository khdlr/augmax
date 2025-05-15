import sys
import os

project = "Augmax"
copyright = "2025, Konrad Heidler. Jax, NumPy and SciPy documentation are copyright to the respective authors"
author = "Konrad Heidler"

version = ""
release = ""

needs_sphinx = "5.0.0"

os.environ["JAX_PLATFORM_NAME"] = "cpu"

sys.path.insert(0, os.path.abspath("../src/"))
sys.path.append(os.path.abspath("sphinx_autoaug"))
extensions = [
  "myst_nb",
  "sphinx.ext.autodoc",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "sphinx.ext.napoleon",
  "sphinx.ext.viewcode",
  "sphinx_autoaug",
  "IPython.sphinxext.ipython_console_highlighting",
  "sphinx_rtd_theme",
]

intersphinx_mapping = {
  "python": ("https://docs.python.org/3/", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
  "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
  "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

source_suffix = [".rst", ".ipynb", ".md"]

# Myst-NB
nb_execution_mode = "force"
nb_execution_timeout = 100
nb_execution_allow_errors = True

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

main_doc = "index"

html_theme = "sphinx_rtd_theme"

pygments_style = "abap"

html_theme_options = {
  "collapse_navigation": False,
  "sticky_navigation": True,
  "navigation_depth": 4,
  "includehidden": True,
  "titles_only": False,
}

latex_documents = [
  (main_doc, "Augmax.tex", "Augmax Documentation", author, "manual"),
]

texinfo_documents = [
  (
    main_doc,
    "Augmax",
    "Augmax Documentation",
    author,
    "Augmax",
    "Efficiently composable data augmentation on the GPU.",
    "Miscellaneous",
  ),
]

man_pages = [(main_doc, "augmax", "Augmax Documentation", [author], 1)]

epub_title = project

autodoc_default_options = {"member-order": "bysource"}
