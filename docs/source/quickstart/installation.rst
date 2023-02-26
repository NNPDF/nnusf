Installation
============

User installation
-----------------

The easiest way to install the latest stable version of NNSFν is
via the `Python Package Index <https://pypi.org/>`_ using the following command:

.. code-block:: bash

   pip install nnusf

.. note::

   In order to use NNSFν one needs to download the the :mod:`commondata`
   and :mod:`theory` files and store them into the user directory
   (which is platform/system dependent).

   NNSFν provides an easy way to download and install these input data files
   by simply running the following commands:

   .. code-block:: console

      nns get theory

   To see where the files have been installed, type the following commands:

   .. code-block:: console

      nns get print_userdir_path

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

Then, to install the NNSFν package just type:

.. code-block:: bash

   poetry install

.. note::

   Note that when installing using :mod:`poetry` one has the choice of doing so in
   a :mod:`poetry` virtual environment or not. To install the package in the current
   active environment, before :mod:`poetry install` type the following command:

   .. code-block:: console

      poetry config virtualenvs.create false --local

   If instead you choose to install the package in a clean environment, first you
   need to save the path to the environment into an environment variable:

   .. code-block:: console

      export PREFIX=$(realpath $(poetry env --path))

   Then download the scripts from
   `N3PDF/workflows/packages/lhapdf <https://github.com/N3PDF/workflows/tree/v2/packages/lhapdf>`_
   and install :mod:`LHAPDF`:

   .. code-block:: console

      sh install.sh

.. tip::

   If NNSFv was instead installed in a :mod:`poetry` virtual environment then
   it may be useful to enter in a :mod:`poetry` shell by invoking the following
   command:

   .. code-block:: console

      poetry shell

   otherwise prepending all the commands by :mod:`poetry` will be required.
