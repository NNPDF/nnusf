Installation
============

User installation
-----------------

The easiest way to install the latest stable version of NNSFν is
via the `Python Package Index <https://pypi.org/>`_ using the following command:

.. code-block:: bash

   pip install nnusf

To check that the package has been installed correctly, just run the following
which will print out all the available subcommands:

.. code-block:: bash

   nnu --help

Development installation
------------------------

In order to develop on the codes it is required to clone the github repository and
install the package using `Poetry <https://python-poetry.org/>`_. To install :mod:`poetry`
just follow the instructions `here <https://python-poetry.org/docs/#installation>`_.
Once this is done, first clone the repository and enter into the directory:

.. code-block:: bash

   git clone https://github.com/NNPDF/nnusf.git --depth 1
   cd nnusf

Then, to install the NNSF:math:`\nu` package just type:

.. code-block:: bash

   poetry install
