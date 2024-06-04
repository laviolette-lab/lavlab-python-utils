# Installation

## Using latest wheel

You can install the latest release from GitHub, once installed you can get the optional dependencies:

```sh
python3 -m pip install https://github.com/laviolette-lab/lavlab-python-utils/releases/latest/download/lavlab_python_utils-latest-py3-none-any.whl
# optional install targets, must install wheel from github using command above first!
python3 -m pip install 'lavlab-python-utils[all]'
```
We support the following install targets:
* omero
  * installs omero-py for omero api access
* jupyter
  * installs dash for performant image viewing

## Using versioned wheel

You can install a specific version from Github:

```sh
VERSION=v1.3.0 python3 -m pip install https://github.com/laviolette-lab/lavlab-python-utils/releases/$VERSION/download/lavlab_python_utils-$VERSION-py3-none-any.whl
```

## From Source

Install using pip+git:

```sh
python3 -m pip install git+https://github.com/laviolette-lab/lavlab-python-utils.git
```
