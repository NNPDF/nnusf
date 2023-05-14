Fit & report
============

Preparing the run card
----------------------

In order to run a fit, we first need to prepare a run card. An example of
such a run card can be found in the following `folder <https://github.com/NNPDF/nnusf/tree/main/runcards>`_.
Essentially, there are two main important keys in the run card which specify
the data sets to be included and the fitting parameters.

The specifications regarding the data sets are stored in the :mod:`experiment` key
in which one needs to specify the data set name and the training fraction as
follows:

.. code-block:: yaml

   experiments:
   - {dataset: BEBCWA59_F2, frac: 0.75}
   - {dataset: BEBCWA59_F3, frac: 0.75}

The details regarding the fit instead are stored in the :mod:`fit_parameters` key.
Generally, it has the following structure:

.. code-block:: yaml

   fit_parameters:
     epochs: 100000
     stopping_patience: 20000
     units_per_layer: [70, 55, 40, 20, 20]
     activation_per_layer: [tanh, tanh, tanh, tanh, selu]
     optimizer_parameters:
       optimizer: Adam
       clipnorm: 0.00001
       learning_rate: 0.05
     val_chi2_threshold: 4
     small_x_exponent:
         f2nu  : [0.25, 2.0]
         flnu  : [0.25, 2.0]
         xf3nu : [0.25, 2.0]
         f2nub : [0.25, 2.0]
         flnub : [0.25, 2.0]
         xf3nub: [0.25, 2.0]


Perform a fit
-------------

To run a fit, one can simplify type the following commands:

.. code-block:: bash

   nnu fit run ${PATH_RUNCARD} ${REPLICA_ID} [-d ${OUTPUT_PATH}]

An example of a runcard to perform a fit is 
`runcards/fit_runcard.yml <https://github.com/NNPDF/nnusf/blob/main/runcards/fit_runcard.yml>`_.

This will generate inside a folder :mod:`RUNCARD_NAME` folders called
:mod:`replica_${REPLICA_ID}` which in turn contain tensorflow models
that can be used to generate predictions. In general, one needs to run the
above command for :mod:`REPLICA_ID={1, ..., n}`.

.. note::
   A pre-trained model is available in the following
   `link <https://data.nnpdf.science/NNUSF/>`_ if the user wants to skip the
   training part and directly wants to generate predictions.


Post-fit selection
------------------

If needed, one can perform a post-selection on the replicas generated
from the fit. For instance, one can only select replicas whose :math:`\chi^2`
values are below some thresholds. Below is an example in which
we only select replicas with :math:`\chi^2_{\rm tr}` and :math:`\chi^2_{\rm vl}`
below :mod:`3`:

.. code-block:: bash

   nnu fit postfit ${RUNCARD_NAME} -t '{"tr_max": 3, "vl_max": 3}'

This will generate inside :mod:`RUNCARD_NAME` a folder called :mod:`postfit`
which contains the replicas that satisfy the selection criteria.


Generate a fit report
---------------------

Using the trained model, we can generate a report containing the summary
of the :math:`\chi^2` values and the comparisons between the experimental
data sets and the NNSFν predictions. To generate the report
just run the following command:

.. code-block:: bash

   nnu report generate ${RUNCARD_NAME}/postfit -t "<Title>" -a "<author>" -k "<keyword>"

This will generate a folder called :mod:`output` inside :mod:`RUNCARD_NAME`
which contains an :mod:`index.html` summarizing the results. The :mod:`.html`
file can then be opened on a browser. If :mod:`postfit` was not run in the
previous step, simply remove :mod:`/postfit` in the command above.


Store NνSF predictions as LHAPDF
--------------------------------

For future convenience, the NNSFν predictions can be stored as LHAPDF
grids. The structure functions have the following LHAPDF IDs:

.. list-table:: LHAPDF ID
   :widths: 20 20 20 20 20 20 20 20 20 20
   :header-rows: 1

   * - SFs
     - :math:`F_2^{\nu}`
     - :math:`F_L^{\nu}`
     - :math:`xF_3^{\nu}`
     - :math:`F_2^{\bar{\nu}}`
     - :math:`F_L^{\bar{\nu}}`
     - :math:`xF_3^{\bar{\nu}}`
     - :math:`\langle F_2^{\bar{\nu}} \rangle`
     - :math:`\langle F_L^{\bar{\nu}} \rangle`
     - :math:`\langle xF_3^{\bar{\nu}} \rangle`
   * - LHAID
     - 1001
     - 1002
     - 1003
     - 2001
     - 2002
     - 2003
     - 3001
     - 3002
     - 3003


The LHAPDF set can be generated using the following command:

.. code-block:: bash

   nnu fit dump_grids ${RUNCARD_NAME}/postfit -a ${A_VALUE} -o ${SET_NAME} [-q '{"min": 1e-3, "max": 500, "num": 100}]'

.. note:: 

   As before, the user can choose the ranges of :math:`x` and :math:`Q^2` from which
   the predictions will be generated. By default, the :math:`Q^2` range is defined to
   be between :math:`[10^{-3}, 500]`. This range is chosen based on the
   :math:`Q^2` values included in the fit through both the experimental and Yadism
   pseudo data.
