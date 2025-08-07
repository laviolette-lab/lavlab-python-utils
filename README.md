# LavLab Python Utils

[![Documentation Status](https://readthedocs.org/projects/lavlab-python-utils/badge/?version=stable)](https://lavlab-python-utils.readthedocs.io/stable)
[![GitHub release](https://img.shields.io/github/v/release/laviolette-lab/lavlab-python-utils.svg)](https://github.com/laviolette-lab/lavlab-python-utils/releases)
[![Build Status](https://github.com/laviolette-lab/lavlab-python-utils/actions/workflows/pylint.yml/badge.svg)](https://github.com/laviolette-lab/lavlab-python-utils/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/laviolette-lab/lavlab-python-utils.svg)](https://codecov.io/gh/laviolette-lab/lavlab-python-utils)

Welcome to the **LavLab Python Utils** repository! This library is a collection of Python utilities developed by the LaViolette Lab to support our research in RadioPathomics and digital pathology.

<img src="logo.webp" width="256">

## Overview

LavLab Python Utils is designed to streamline common tasks in our research workflow, including data preprocessing, analysis, and visualization. Our goal is to provide a set of tools that are easy to use, well-documented, and performant.

## Features

- **Data Processing:** Functions to handle various data formats used in our research, including DICOM, NIfTI, and more.
- **Visualization:** Tools to create publication-quality plots and visualizations.
- **Integration:** Seamless integration with other tools and services used in our lab.
- **OMERO Support:** Connect to and work with OMERO servers for image data management.
- **XNAT Support:** Connect to and work with XNAT servers for neuroimaging data management.

## Installation

You can install LavLab Python Utils via pip:

```bash
python3 -m pip install https://github.com/laviolette-lab/lavlab-python-utils/releases/latest/download/lavlab_python_utils-latest-py3-none-any.whl
# optional install targets, must install wheel from github using command above first!
python3 -m pip install 'lavlab-python-utils[all]'

# Or install specific features:
python3 -m pip install 'lavlab-python-utils[omero]'  # For OMERO support
python3 -m pip install 'lavlab-python-utils[xnat]'   # For XNAT support
python3 -m pip install 'lavlab-python-utils[jupyter]' # For Jupyter tools
```

## Documentation

Comprehensive documentation is available on [Read the Docs](https://lavlab-python-utils.readthedocs.io/stable/). This includes detailed API references, usage examples, and tutorials.

## Contributing

We welcome contributions from the community! If you are interested in contributing, please read our [contributing guide]([CONTRIBUTING.md](https://lavlab-python-utils.readthedocs.io/stable/contributing/)) and check out our [open issues](https://github.com/laviolette-lab/lavlab-python-utils/issues).

---

Thank you for using LavLab Python Utils! We hope you find it useful in your research.
