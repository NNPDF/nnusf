Grids & Usage
=============

LHAPDF grids
------------

The NNSFν structure function grids are available in the LHAPDF format
for various nuclear targets. For a given nuclear target, the grids is
split into small- and large-:math:`Q`, which could be combined according
to a prescription described below.

.. list-table:: LHAPDF GRIDS
   :widths: 30 60 60
   :header-rows: 1

   * - :math:`(Z, A)` [target]
     - Low-:math:`Q` Grid
     - High-:math:`Q` Grid
   * - :math:`(1, 2)`
     - `NNSFnu\_D\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_D_lowQ.tar.gz>`_
     - `NNSFnu\_D\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_D_highQ.tar.gz>`_
   * - :math:`(2, 4)`
     - `NNSFnu\_He\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_He_lowQ.tar.gz>`_
     - `NNSFnu\_He\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_He_highQ.tar.gz>`_
   * - :math:`(3, 6)`
     - `NNSFnu\_Li\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Li_lowQ.tar.gz>`_
     - `NNSFnu\_Li\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Li_highQ.tar.gz>`_
   * - :math:`(4, 9)`
     - `NNSFnu\_Be\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Be_lowQ.tar.gz>`_
     - `NNSFnu\_Be\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Be_highQ.tar.gz>`_
   * - :math:`(6, 12)`
     - `NNSFnu\_C\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_C_lowQ.tar.gz>`_
     - `NNSFnu\_C\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_C_highQ.tar.gz>`_
   * - :math:`(7, 14)`
     - `NNSFnu\_N\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_N_lowQ.tar.gz>`_
     - `NNSFnu\_N\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_N_highQ.tar.gz>`_
   * - :math:`(8, 16)`
     - `NNSFnu\_O\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_O_lowQ.tar.gz>`_
     - `NNSFnu\_O\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_O_highQ.tar.gz>`_
   * - :math:`(13, 27)`
     - `NNSFnu\_Al\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Al_lowQ.tar.gz>`_
     - `NNSFnu\_Al\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Al_highQ.tar.gz>`_
   * - :math:`(15, 31)`
     - `NNSFnu\_Ea\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ea_lowQ.tar.gz>`_
     - `NNSFnu\_Ea\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ea_highQ.tar.gz>`_
   * - :math:`(18, 40)`
     - `NNSFnu\_Ar\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ar_lowQ.tar.gz>`_
     - `NNSFnu\_Ar\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ar_highQ.tar.gz>`_
   * - :math:`(20, 40)`
     - `NNSFnu\_Ca\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ca_lowQ.tar.gz>`_
     - `NNSFnu\_Ca\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ca_highQ.tar.gz>`_
   * - :math:`(26, 56)`
     - `NNSFnu\_Fe\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Fe_lowQ.tar.gz>`_
     - `NNSFnu\_Fe\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Fe_highQ.tar.gz>`_
   * - :math:`(29, 64)`
     - `NNSFnu\_Cu\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Cu_lowQ.tar.gz>`_
     - `NNSFnu\_Cu\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Cu_highQ.tar.gz>`_
   * - :math:`(47, 108)`
     - `NNSFnu\_Ag\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ag_lowQ.tar.gz>`_
     - `NNSFnu\_Ag\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Ag_highQ.tar.gz>`_
   * - :math:`(50, 119)`
     - `NNSFnu\_Sn\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Sn_lowQ.tar.gz>`_
     - `NNSFnu\_Sn\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Sn_highQ.tar.gz>`_
   * - :math:`(54, 131)`
     - `NNSFnu\_Xe\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Xe_lowQ.tar.gz>`_
     - `NNSFnu\_Xe\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Xe_highQ.tar.gz>`_
   * - :math:`(74, 184)`
     - `NNSFnu\_W\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_W_lowQ.tar.gz>`_
     - `NNSFnu\_W\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_W_highQ.tar.gz>`_
   * - :math:`(79, 197)`
     - `NNSFnu\_Au\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Au_lowQ.tar.gz>`_
     - `NNSFnu\_Au\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Au_highQ.tar.gz>`_
   * - :math:`(82, 208)`
     - `NNSFnu\_Pb\_lowQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Pb_lowQ.tar.gz>`_
     - `NNSFnu\_Pb\_highQ <https://data.nnpdf.science/NNSFnu/NNSFnu_Pb_highQ.tar.gz>`_

The structure function grids can then be used to compute the
double differential cross-sections using the following command:

.. code-block:: bash

   nnu extra compute_xsecs ${SET_NAME} [-q '{"min": 1e-3, "max": 400, "num": 100}]' [-q ${type}]

where :mod:`type` can be either :mod:`neutrino` or :mod:`antineutrino`, if not
specified the default value is chosen to be :mod:`neutrino`.

One can also from a given structure function set compute the integrated
cross-sections via the command:

.. code-block:: bash

   nnu extra integrated_xsecs ${SET_NAME} [-q '{"min": 1e-3, "max": 400, "num": 100}]' [-q ${type}]


NNSFν pre-trained model
-----------------------

We also make publicly available the pre-trained NNSFν model used to generate
the predictions published in the paper. The model can be downloaded at the
following `link <https://data.nnpdf.science/NNUSF/Models/nnsfnu.tar.gz>`_.
Such a model can be used for various purposes but as explained in the
tutorial part it can be mainly used to generate structure function grids:

.. code-block:: bash

   nnu fit dump_grids nnsfnu/postfit -a ${A_VALUE} -o ${SET_NAME} [-q '{"min": 1e-3, "max": 500, "num": 100}]'


.. note:: 

   If one encouters the issue that the git version does not match, one just
   needs to checkout to the commit from which the fit was generated.

