# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../boat'))

project = 'BOAT'
copyright = '2024, Yaohua Liu'
author = 'Yaohua Liu'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Sphinx 配置
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # 支持 Google 和 NumPy 风格的 docstring
    'sphinx.ext.viewcode', # 在文档中生成代码链接
    'myst_parser',         # 支持 Markdown (可选)
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# html_theme = 'alabaster'
html_static_path = ['_static']


source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}