
import os
import sys
sys.path.insert(0, os.path.abspath('../scmags'))
from pathlib import Path

HERE = Path(__file__).parents[1]
sys.path[:0] = [str(HERE.parent), str(HERE / 'extensions')]

# -- Project information -----------------------------------------------------

project = 'scmags'
copyright = '2021, Yusuf Baran'
author = 'Yusuf Baran'
release = '0.1.5'


# -- General configuration ---------------------------------------------------

extensions = [
  	"sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax", 
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints", 
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive', 
    'nbsphinx' 
 
]

napoleon_use_rtype = True  
napoleon_use_param = True
add_module_names = False
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_custom_sections = [("Params", "Parameters")]
autoclass_content = "class"
api_dir = HERE / 'api'  


templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


html_static_path = ['_static']

plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_working_directory = HERE.parent  # Project root

