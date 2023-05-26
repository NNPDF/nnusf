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

   @article{Candido:2023utz,
       author = "Candido, Alessandro and Garcia, Alfonso and Magni, Giacomo and Rabemananjara, Tanjona and Rojo, Juan and Stegeman, Roy",
       title = "{Neutrino Structure Functions from GeV to EeV Energies}",
       eprint = "2302.08527",
       archivePrefix = "arXiv",
       primaryClass = "hep-ph",
       reportNumber = "Nikhef 2022-014, Edinburgh 2022/27, TIF-UNIMI-2023-5",
       month = "2",
       year = "2023"
   }

If NNSFν proves to be useful in your work, please also reference the codes:

.. code-block:: latex

   @misc{https://doi.org/10.5281/zenodo.7657132,
     doi = {10.5281/ZENODO.7657132},
     url = {https://zenodo.org/record/7657132},
     author = "Candido, Alessandro and Garcia, Alfonso and Magni, Giacomo and Rabemananjara, Tanjona and Rojo, Juan and Stegeman, Roy",
     title = "{Neutrino Structure Functions from GeV to EeV Energies}",
     publisher = {Zenodo},
     year = {2023},
     copyright = {Open Access}
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
