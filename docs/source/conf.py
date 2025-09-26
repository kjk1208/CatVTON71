# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CatVTON'
copyright = '2025, kjk'
author = 'kjk'
release = '1.0'


import os, sys
# repo 루트가 docs/ 상위이면 ..
ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))  # ~/CatVTON
sys.path.insert(0, ROOT)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
autodoc_mock_imports = ["torch","torchvision","diffusers","transformers","safetensors","PIL","tqdm","numpy"]

templates_path = ['_templates']
exclude_patterns = []

language = 'ko'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
napoleon_google_docstring = True
napoleon_numpy_docstring = True