Welcome to NNSFν
================

.. include:: badges.rst

NNSFν povides predictions for neutrino inelastic structure functions
valid for the complete range of energies relevant for phenomenology
involving :math:`\nu / \bar{\nu}`-experiments, from oscillation measurements
carried out with reactors, accelerators, and atmospheric neutrinos to astroparticle
physics at ultra-high-energy (UHE) neutrino telescopes such as IceCube
and KM3NET.

|

.. grid:: 2

   .. grid-item-card::

       ⚡ Getting Started
       ^^^^^^^^^^^^^^^^^^

       Description of the package and instructions
       for the installation.

       +++

       .. button-ref:: quickstart/description
           :expand:
           :color: secondary
           :click-parent:

           To quick start

   .. grid-item-card::

       🔥 Tutorials
       ^^^^^^^^^^^^

       Various tutorials on using the codes and
       generate predictions.

       +++

       .. button-ref:: tutorials/datasets
           :expand:
           :color: secondary
           :click-parent:

           To tutorials

Citing NNSFν
============

To reference NNSFν in a scientific publication please use the following:

.. code-block:: latex

   @article {reference_id,
      author = {Alessandro Candido, Alfonso Garcia, Giacomo Magni, Tanjona Rabemananjara, Juan Rojo, Roy Stegeman},
      title = {Neutrino Structure Functions from GeV to EeV Energies},
      year = {2023},
      doi = {10.1101/2020.07.15.204701},
      eprint = {https://arxiv.org/list/hep-ph/},
      journal = {aRxiv}
   }

If NNSFν proves to be useful in your work, please also reference the codes:

.. code-block:: latex

   @article {reference_id,
      author = {Alessandro Candido, Alfonso Garcia, Giacomo Magni, Tanjona Rabemananjara, Juan Rojo, Roy Stegeman},
      title = {Neutrino Structure Functions from GeV to EeV Energies},
      year = {2023},
      doi = {10.1101/2020.07.15.204701},
   }


.. toctree::
   :maxdepth: 2
   :caption: 📑 Quick Start
   :hidden:

   quickstart/description
   quickstart/installation

.. toctree::
   :maxdepth: 2
   :caption: 🚀 Tutorials
   :hidden:

   tutorials/datasets
   tutorials/yadism
   tutorials/fitting

.. toctree::
   :maxdepth: 2
   :caption: 📦 Delivery
   :hidden:

   delivery/lhapdf

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
