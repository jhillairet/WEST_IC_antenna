# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WEST IC Antenna'
copyright = '2024, Julien Hillairet'
author = 'Julien Hillairet'

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(1500)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']
# exclude some notebooks not yet ready to be converted into html
exclude_files = [
    #'chart_strap_reactance.ipynb',
    #'chart_manual_matching.ipynb',
    'coupling_to_ideal_load.ipynb',
    #'coupling_to_plasma_from_COMSOL.ipynb',
    'coupling_to_plasma_from_TOPICA.ipynb',
    'coupling_to_water_tank.ipynb',
    'digital_twin.ipynb',
    #'introduction.ipynb',
    'phase_scan.ipynb',
    'Plasma Startup.ipynb',
    #'tutorial_matching_automatic.ipynb',
    #'tutorial_matching_manual.ipynb'
]
for file in exclude_files:
    exclude_patterns.append(file)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

nbsphinx_allow_errors = True