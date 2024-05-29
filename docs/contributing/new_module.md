# Adding a New Module

## Overview

This guide covers how to add a new module to LavLab Python Utils.

## Steps

1. **Identify the scope**: Determine the purpose and functionality of the new module. Also determine whether this is needs to be larger than a single file. If larger than a single file, look into the new dependencies you are adding, this may be worth adding as a [new feature](new_feature.md). Feel free to open an issue on [Github](https://github.com/laviolette-lab/lavlab-python-utils) to discuss what the file structure might look like.
2. **Implement the module**: Write the code for the new module.
3. **Document the module**: Update the documentation to include information about the new module.
4. **Write tests**: Ensure that you write tests for the new module.

## Example

Here is an example of adding a new module:

1. **Identify Scope**:

For our example we'll follow along with our lavlab/imsuite.py module. 

2. **Implement the module**:

The actual implementation details are not too important and are, of course, very functionality specific. but there are a few things to keep in mind.  
```python
# lavlab/tissuemask.py

import lavlab
# get a child logger from the parent module.
LOGGER = lavlab.LOGGER.getChild("imsuite")

# if the context is required, DO NOT DO from lavlab import ctx
# importing the context that way will cause this module's context to diverge from the rest of the package
# if you need to use the context always do
lavlab.ctx...

#here's a function from insuite as an example
# be sure to type your variables and returns as well as using numpy docstrings
def imwrite(
    img: Union[np.ndarray, pv.Image], path: os.PathLike, **kwargs
) -> os.PathLike:
    """Writes an image to path. kwargs are passthrough to wrapped function.

    Parameters
    ----------
    img : Union[np.ndarray, pv.Image]
        Numpy array or PyVips image.
    path : os.PathLike
        Path to desired file.

    Returns
    -------
    os.PathLike
        Path of newly created file.
    """
    if not isinstance(img, pv.Image):
        assert isinstance(img, np.ndarray)
        img = pv.Image.new_from_array(img)
    assert isinstance(img, pv.Image)
    # need to figure out default compression settings some day
    settings = (
        kwargs if kwargs else lavlab.ctx.histology.get_compression_settings()
    )  
    img.write_to_file(path, **settings)
    return path
```

3. **Document the module**:

When you use [numpy formatted docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) you will automatically be documenting your code. MkDocs will use those docstrings when generating its documents. All you need to do is add a new file like docs/api/new_module.md:
```
# new_module.py

::: lavlab.new_module
```
Add usage examples and details in the relevant documentation sections. Adding new pages where necessary. Markdown is pretty simple, there's a whole website explaining how to use it, but [this cheatsheet](https://www.markdownguide.org/cheat-sheet/) is probably all you need. (VSCode has a [builtin markdown viewer](https://code.visualstudio.com/docs/languages/markdown))

4. **Writing tests**

Ensure that you write tests for your new module and run them to verify that everything works correctly. Pytest handles a lot of the work, all you need to do is create a test_...py file and if necessary a conftest.py file, see [pytest documentation](https://docs.pytest.org/en/latest/getting-started.html) for more info.

```python
# test/test_imsuite.py

# add pylint and type ignoring because tests aren't linted anyway.

# pylint: skip-file
# type: ignore
def test_imwrite():
    img = np.random.rand(100, 100)
    path = "test_output.png"
    imwrite(img, path)

    # use assert
    assert os.path.exists(path), "The output file should exist."
    os.remove(path)
```