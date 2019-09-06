.. _install:

**********
Installation
**********

The package is has been written using Python 3.6.8, though all elements should run on any version from 3.5.4 (limitation due to requirment of pymc3).  SQuEELS uses elements from the following packages:

* numpy
* scipy
* matplotlib
* hyperspy (tested using version 1.5.1, but others should work)
* lmfit
* pymc3

Currently, SQuEELS is only available for installation from its GitLab repository.  There are no current plans to change this.

Installation from GitLab:

```bash
pip3 install --user git+https://gitlab.com/rwebster/SQuEELS.git
```

The package can be removed with:

```bash
pip3 uninstall SQuEELS
```

