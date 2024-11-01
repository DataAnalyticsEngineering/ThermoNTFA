# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "ThermoNTFA"
copyright = "2024, Felix Fritzen, Julius Herb, Shadi Sharba"
author = "Felix Fritzen, Julius Herb, Shadi Sharba"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "sphinx_gallery.load_style",
]
myst_enable_extensions = ["dollarmath", "amsmath"]
nbsphinx_allow_errors = True
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "DataAnalyticsEngineering",  # Username
    "github_repo": "ThermoNTFA",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    if not os.path.exists("examples"):
        os.symlink(os.path.join("..", "examples"), "examples")

    app.connect("autodoc-skip-member", skip)
