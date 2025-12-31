# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from pathlib import Path


#<project_root>/boat_torch, <project_root>/docs/source/conf.py
CUR = Path(__file__).resolve()
PROJECT_ROOT = CUR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
#  sys.path
sys.path.insert(0, os.path.abspath("../../"))



autodoc_mock_imports = ["mindspore","matplotlib", "matplotlib.pyplot"]
autodoc_typehints = "none"
html_logo = "_static/logo.jpg"

project = "BOAT-MS"
copyright = "2024, Yaohua Liu"
author = "Yaohua Liu"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
# html_theme = 'alabaster'

html_context = {
    "extrahead": '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">',
}