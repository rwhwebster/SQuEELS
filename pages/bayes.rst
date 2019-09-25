.. _bayes:

**********
Creating a Bayesian Model Instance
**********

To create a Bayesian model to quantify a spectrum, we first need to have references loaded, as per the instructions on the Standards page, as well as having loaded some data to model.

.. code-block:: python
    
    import hyperspy.api as hs

    import SQuEELS as sq

    refs = sq.io.Standards()

    LL = hs.load('low_loss.dm3')
    HL = hs.load('high_loss.dm3')

With this data loaded, we can go on to initialise a model object, using the class ``BayesModel`` in SQuEELS:

.. code-block:: python
    
    components = ['Ti', 'O', 'TiO'] # A list of materials from out Standards library we want to use in the fit.
    model_range = (400.0, 700.0) # The energy-loss range over which the model will be fitted.

    our_model = sq.bayes.BayesModel(HL, refs, components, model_range) # The model-object initialisation


Executing the model
======================

Once the model instance has been created, it can be executed using the ``BayesModel.simple_multimodel`` method.  This can be called in such a way as to allow fitting of specific spectra in a spectrum image (SI), randomly sample spectra in the SI, or model the full SI.  A typical call to ``simple_multimodel`` may look like:

.. code-block:: python
    to_sample = np.array([[1,0],[2,1]])
    guesses = (10.0, 1.0, 5.0)
    chain_params = {'init':'advi+adapt_diag', 'cores':4,'chains':8, 'progressbar':False}

    df = our_model.simple_multimodel(nSamples=to_sample, prior_means=guesses, nDraws=1000, chain_params=chain_params)

The output of ``simple_multimodel`` is a pandas dataframe which contains the MCMC outputs and some summary statistics.  Inspection of these can be done manually using visualisation methods available through matplotlib, pandas and pymc3.  Future versions of SQuEELS will provide handles to some common calls.