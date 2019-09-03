SQuEELS v0.1.3
====

About
----

SQuEELS (**S**tandard-based **Qu**antification of **EELS** data) is a package for in-depth analysis of elemental quantification of EELS core-loss spectra.

The SQuEELS package is designed to provide code for straightforward implementation of compositional quantification using reference spectra.  The code is intended to be easily inspected and understood, removing the ambiguity of black-box style operations found in Gatan DigitalMicrograph and Hyperspy.  SQuEELS is geared towards not only providing accurate quantification, but also enabling in-depth analysis of the errors and uncertainties of fitting procedures, to engender better confidence in results.

Installation
----

The package is has been written using Python 3.6.8, though all elements should run on any version from 3.5.4 (limitation due to requirment of pymc3).

Installation from GitLab:

```bash
pip3 install --user git+https://gitlab.com/rwebster/SQuEELS.git
```

The package can be removed with:

```bash
pip3 uninstall SQuEELS
```

Usage
----

In python or ipython, load package as:

```python
import SQuEELS as sq
```

Standards can be loaded using the `io` module:

```python
refs = sq.io.Standards()
```

If no argument is provided to `Standards`, a file browser opens to ask the user to select a folder where reference standards are stored.  Otherwise, the argument `fp` can be provided to specify a folder without use of a GUI.
