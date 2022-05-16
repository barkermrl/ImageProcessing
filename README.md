# Part IIA Project: SF2: Image Processing

[![codecov](https://codecov.io/gh/sigproc/cued_sf2_lab/branch/master/graph/badge.svg)](https://codecov.io/gh/sigproc/cued_sf2_lab)
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/from-referrer)

This repository contains the Python package and Jupyter Notebooks for the SF2 lab project in the Cambridge University Engineering department.

***Note that the python version of this lab is experimental and only runs up to the end of section 7. Notebooks for sections 8 through 11 are provided but are incomplete and not fully tested.***

***It is likely that the notebooks will be updated midway through the course.***

To get started, you should:

* **If using the DPO computers**:
  * It is advisible to boot into Linux, as `git` will already be installed there.
  * You should use the "Anaconda terminal" not the usual terminal, as this will have a more recent version of Python.
* Have a recent version of python + Jupyter installed.
  Check that `python --version` emits what you expect it to.
* `git clone` this repository (recommended). If you do not have git installed, you can download and extract the zip from the top of the github page; but this will make it harder for you to get updated versions.
* Open a command prompt in the folder you downloaded the code to, and run `python -m pip install -e . --user --upgrade`.
  This will install various dependencies, and a `cued_sf2_lab` python package containing a collection of helper functions.
* Open the notebooks (`ipynb` files) in the root of this repository.

## FAQ

1. **Why aren't my matplotlib color bar plots showing up?**  
   Likely you are not using the latest matplotlib.
   You can find out the version with `import matplotlib; matplotlib.__version__` inside Jupyter.
   To update, try `pip install --upgrade --user matplotlib`.
   
2. **Why aren't the image plots showing up in VSCode**?
   The notebooks are written with Jupyter Notebook in mind; we have not thoroughly tested VSCode or JupyterLab.
   Most likely, you will need to swap `%matplotlib nbagg` for `%matplotlib widget` or `%matplotlib inline`.

## Note for demonstrators

This software consists of two repos, which share this README.

* https://github.com/sigproc/cued_sf2_lab
* https://github.com/sigproc/cued_sf2_lab-answers

If you are a student, you will only have access to the former!
If you are a demonstrator, you should request access to the latter.
The answers repository generates the other repository automatically.

More information for demonstrators can be found in [the demonstrator readme](https://github.com/sigproc/cued_sf2_lab-answers/blob/main/README-demonstrators.md).
