.. _viz:

**********
Visualising results of MLLS fitting
**********

Introducing ``SQuEELS.viz`` and the ``FitInspector`` class.





To use this method, your reference spectra should all be contained within a single folder to point your python script towards.

A library can be assigned without declaring any arguments to open a file browser to use to locate the directory which contains the reference spectra;

.. code-block:: python
    
    import SQuEELS as sq
    
    refs = sq.io.Standards()



