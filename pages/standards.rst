.. _standards:

**********
Loading a Standards Library
**********

Here we discuss how to use the io module of SQuEELS to load a library of standard spectra.

To use this method, your reference spectra should all be contained within a single folder to point your python script towards.

A library can be assigned without declaring any arguments to open a file browser to use to locate the directory which contains the reference spectra;

.. code-block:: python
    
    import SQuEELS as sq
    
    refs = sq.io.Standards()


Otherwise, a path to the folder containing the standards can be specified like so;

.. code-block:: python
    
    refs = sq.io.Standards(fp='c/data/references')


