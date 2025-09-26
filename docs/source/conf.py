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

# 将项目的根目录添加到 sys.path
sys.path.insert(0, os.path.abspath("../../"))

html_logo = "_static/logo.jpg"

project = "BOAT-Torch"
copyright = "2024, Yaohua Liu"
author = "Yaohua Liu"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx 配置
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # 支持 Google 和 NumPy 风格的 docstring
    "sphinx.ext.viewcode",  # 在文档中生成代码链接
    "myst_parser",  # 支持 Markdown (可选)
]

autodoc_mock_imports = [
    "numpy", "scipy", "sklearn",
    "torch", "torchvision", "torchmeta", "higher",
    "matplotlib"
]

autodoc_typehints = "none"


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_book_theme"

# html_theme = "sphinx_material"
html_theme = "sphinx_rtd_theme"
# html_theme = "furo"
# html_theme = "pydata_sphinx_theme"


# html_theme = 'alabaster'
html_static_path = ["_static"]
html_css_files = [
    "custom.css",  # 引入自定义 CSS
]


html_context = {
    "extrahead": '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">',
}
