.. _bayes:

**********
Creating a Bayesian Model
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

Additionally, if we wish to convolve/deconvolve our core-loss data using a low-loss spectrum, this can be done by adjusting our call to the model to include the low-loss data like so:

.. code-block:: python

    our_model = sq.bayes.BayesModel(high_loss, refs, components, model_range, low_loss=LL)

Initialising the model
======================

Once the model object has been created, it is necessary to further intialise the model parameters using the ``BayesModel.init_model`` method.  This can be called with or without arguments as the situation requires.  Here are a few use cases:

.. code-block:: python
    
    # For when a single spectrum has been provided, and no guesses about the results can be made.
    our_model.init_model()

    # If a spectrum image has been provided, the coordinates 
    # of the specific spectrum to be fitted are required
    our_model.init_model(nav=[1,1])

    # If there is prior knowledge about the quantities to be fitted,
    # guesses can be provided to improve the priors.
    our_model.init_model(mu_0=(5.0, 5.0, 10))
    # Note the order of the guesses mu_0 matches the order we provided the components earlier.

For other arguments when calling ``BayesModel.init_model`` please refer to the docstrings.

Executing the fit
=================

After initialising the model for the current spectrum, a monte carlo estimation of the parameters can be run by using the method ``BayesModel.start_chains``.  
