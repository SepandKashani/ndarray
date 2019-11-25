# #############################################################################
# conf.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

import datetime

project = 'ndarray'
copyright = f'{datetime.date.today().year}, Sepand KASHANI'
author = 'Sepand KASHANI [kashani.sepand@gmail.com]'

templates_path = ['_templates']
master_doc = 'index'
exclude_patterns = []
pygments_style = 'sphinx'
add_module_names = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_path = ["_themes",]
html_theme_options = {
    'navigation_depth': -1,
    'titles_only': True}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
