Yadism predictions
==================

Generating Yadism theory cards
------------------------------

An alternative way to generate Yadism theory cards that will be used to
generate theory predictions is to use the :mod:`nnusf.theory.runcards` module.
This is in particular convenient in order to generate standalone predictions
that will be dumped into grids using the LHAPDF format.

To generate the theory card for a given atomic mass number :math:`A`, just run the following:

.. code-block:: bash

   nnu theory runcards yadknots -a ${A_VALUE} [--q2_grids '{"min": 2, "max": 1.96e8, "num": 200}']

The command above will dump the Yadism card as a compressed :mod:`.tar` file
inside a directory called :mod:`theory` unless otherwise specified. Notice that
as illustrated above one can optionally specify the :math:`Q^2` range from which
the theory predictions will be computed.

.. note::

   By default, the range is taken to be between :math:`[2, 10^{11}]~\mathrm{GeV}^2`
   as this is relevant range for the (anti-)neutrino predictions and for which the
   Yadism calculations are valid. However, this can be changed easily by replacing
   the numbers in the input dictionary.


Generating PineAPPL grids
-------------------------

The theory card can now be passed to Yadism to calculate coefficients to be stored in PineAPPL
grids. This can be done in the following way:

.. code-block:: bash

   nnu theory grids theory/runcards-yadknots_A${A_VALUE}.tar

As before, unless otherwise specified, the command above will store the PineAPPL
grid inside the :mod:`theory` folder with the name :mod:`grids-runcards.tar`.


Dumping predictions as LHAPDF
-----------------------------

In order to compute the final predictions and dump the results into LHAPDF grids,
we need to convolute the PineAPPL grid with the corresponding (nuclear) PDFs. To
do so, just run the following command:

.. code-block:: bash

   nnu theory predictions theory/grids-runcards.tar ${PDFSET_NAME} --err pdf --no-compare_to_by

This will generate in the current directory a LHAPDF set called :mod:`YADISM_${A_VALUE}`
which for use should be place in the LHAPDF data directory :mod:`lhapdf-config --datadir`.

