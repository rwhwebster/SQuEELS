.. _mlls:

**********
MLLS modelling of data
**********

Here, we introduce the ``MLLSmodel`` class, which handles the core standard-based quantification functionality of SQuEELS.



To use this method, your reference spectra should all be contained within a single folder to point your python script towards.

A library can be assigned without declaring any arguments to open a file browser to use to locate the directory which contains the reference spectra;

.. code-block:: python
    
    import SQuEELS as sq
    
    refs = sq.io.Standards()



