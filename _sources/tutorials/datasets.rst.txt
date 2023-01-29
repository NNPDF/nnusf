Input datasets
==============

The following is not mandatory in order to run a fit given that all the
necessary inputs in order to reproduce the published NNSFν have already
been generated. The following is more useful in case the user wants to
include new datasets. NNSFν receives three main types of inputs: the
experimental data sets, the coefficients of the structure functions in
order to compute a given observable, and the matching predictions from
Yadism.


Adding a new dataset
--------------------

The implementation of a new data set starts by downloading the
`hepdata <https://www.hepdata.net/>`_ tables and store them in
:mod:`commondata/rawdata` using the experiment as the folder name.
Then, the user needs to create a filter that extracts the experimental
values and the treatment of the statistical and/or systematic
uncertainties. For some examples on how to create a filter, refer to
the following `folder <https://github.com/NNPDF/nnusf/tree/main/commondata/filters>`_.
Once this is done run the following command:

.. code-block:: bash

   nnu data filter ./commondata/rawdata/*

This will dump the pandas tables containing the input kinematics,
central values, and uncertainties in :mod:`commondata/kinematics`,
:mod:`commondata/data`, and :mod:`commondata/uncertainties`
respectively. In addition, this will also generate inside
:mod:`commondata/info` various information concerning a given
experiment such as the (nuclear) target.


Building the coefficients
-------------------------

Now that we have the input datasets we need to generate the corresponding
coefficients that connect the structure function bases to the desired observable.
To do so, just run the following:

.. code-block:: bash

   nnu data coefficients ./commondata/data/*


Yadism pseudo-data
------------------

To generate the Yadism pseudo-data, we first need to generate the Yadism theory
card, and in order to generate the run card we need to generate define the
kinematic grid :math:`\left(x, Q^2, y \right)` on which the predictions will
be computed. Ideally, the kinematic grids should match a specific dataset for
:math:`\left(x, y \right)` and only the :math:`Q^2` values are different since
they have to be generate at medium-:math:`Q^2`. In order
to generate the input kinematics, just run the following command:

.. code-block:: bash

   nnu data matching_grids_empty ./commondata/data/DATA_${EXPERIMENT}_${OBSERVABLE}.csv

This will generate inside :mod:`commondata/kinematics` a table named
:mod:`KIN_${EXPERIMENT}_${OBSERVABLE}_MATCHING.csv` containing the input
kinematic values which will be used later to generate the theory card.

The free proton :math:`A=1` Yadism pseudo-data needs a special treatment
when generating the input kinematics, that is one needs to run the following
command:

.. code-block:: bash

   nnu data proton_bc_empty

This will read the grid specifications from :mod:`commondata/matching-grids.yml`
and as before will generate the table inside :mod:`commondata/kinematics`.

We can now generate the grids containing the predictions using the following:

.. code-block:: bash

   nnu grids # for general matching
    # or
   nnu grids # for proton boundary condition

In order to generate the central values and uncertainties for the matching data sets
we need to convolute the grids with the corresponding nuclear PDFs (nPDFs). To do
so, run the following command for a given dataset:

.. code-block:: bash

   nnu data matching_grids ./grids/grids-${EXPERIMENT}_${OBSERVABLE}.csv ${NUCLEAR_PDF_NAME}

In the same way as before, the free-proton used as the boundary condition needs a
special treatment in that they have to be generated at the same time in the following
way:

.. code-block:: bash

   nu data proton_bc ./grids-PROTONBC_*_MATCHING.tar.gz ${PDF_NAME}

Once these are done the remaining thing to do is to generate the corresponding
coefficients in the same way as for the real experimental data. For this we
just need to run the same command as previously:

.. code-block:: bash

   nnu data coefficients ./commondata/data/*
