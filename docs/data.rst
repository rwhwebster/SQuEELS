.. _data:

**********
Loading an EELS dataset
**********

Here we discuss how to use the io module of SQuEELS to load an EELS dataset which we wish to analyse.

Data can be assigned without declaring any arguments to open a file browser to use to locate the data-file;

.. code-block:: python
    
    import SQuEELS as sq
    
    refs = sq.io.Data()


Otherwise, a path can be specified like so;

.. code-block:: python
    
    refs = sq.io.Data(fp='c/data/EELS_SI.dm3')


Note that the code is designed to work with dm3 file format, and does not have support for other types.


