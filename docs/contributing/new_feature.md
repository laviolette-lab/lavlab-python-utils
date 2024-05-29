# Adding a New Feature Target

## Overview

This guide covers how to add a new feature target in LavLab Python Utils.

## Steps
A new feature is an extension of a new module, this is an extension of the process defined in with a few clarifications
1. **Define the feature**: Clearly define the new feature and its purpose.
2. **Register the feature**: Add the new dependencies to an install target.
3. **Implement the feature**: Write the code to implement the new feature.

## Example

1. **Define the feature**:

We'll be following along with the lavlab/jupyter module. This module is designed to handle all our jupyter-based gui interactivity. We have matplotlib in our base package, but it can be a bit slow for interactive setups. So we will use Dash for our interactive viewer. That gives us the following new dependencies.
* jupyter
* dash-slicer
* dash-bootstrap-components

2. **Register the feature**:

We add those dependencies to our pyproject.toml 
```toml
[project.optional-dependencies]
jupyter = [
  "jupyter",
  "dash-slicer",
  "dash-bootstrap-components"
]
```

3. **Implement the feature**:

This is identical to implementing a module, but since most multi-file modules will be feature targets, we'll talk a bit about that here. Every multi-file module needs an \_\_init\_\_.py, it is what is being referenced when you do `import lavlab.jupyter`. If you import a submodule, like `import lavlab.jupyter.viewers` \_\_init\_\_.py will automatically be sourced. So it is a great place to put module level configs, like so:
```python
# lavlab/jupyter/__init__.py
"""Jupyter Utilities, like interactive inputs or visuals"""
from IPython import get_ipython
import lavlab

get_ipython().run_line_magic('matplotlib', 'widget')
LOGGER = lavlab.LOGGER.getChild("jupyter")

```
Now we gotta actually implement our module, big thing to consider here is that now we have a new parent module, so we will be using the logger from the jupyter's  \_\_init\_\_.py instead like so:
```python
import lavlab.jupyter
LOGGER = lavlab.jupyter.LOGGER.getChild("viewers")
```
Beyond that, a feature is a module, and from here it's all the same as [implementing a module](new_module.md)