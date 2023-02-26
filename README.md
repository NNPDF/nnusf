<h1 align="center">NNSFν</h1>
<p align="center">
  <a href="https://zenodo.org/account/settings/github/repository/NNPDF/nnusf#"><img alt="Zenodo" src="https://zenodo.org/badge/DOI/10.5281/zenodo.7657132 .svg"></a>
  <a href="https://arxiv.org/abs/2302.08527"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2302.08527-b31b1b?labelColor=222222"></a>
  <img alt="Docs" src="https://assets.readthedocs.org/static/projects/badges/passing-flat.svg">
  <a href="https://pypi.org/project/nnusf/"><img alt="PyPI" src="https://img.shields.io/pypi/v/nnusf"/></a>
  <img alt="Status" src="https://www.repostatus.org/badges/latest/active.svg">
  <img alt="License" src="https://img.shields.io/badge/License-GPL3-blue.svg">
</p>

<p align="justify">
  <b>NNSFν</b> is a python module that provides predictions for neutrino structure functions. 
  It relies on <a href="https://github.com/N3PDF/yadism">YADISM</a> for the large-Q region 
  while the low-Q regime is modelled in terms of a Neural Network (NN). The NNSFν 
  determination is also made available in terms of fast interpolation
  <a href="https://lhapdf.hepforge.org/">LHAPDF</a> grids that can be accessed through an independent
  driver code and directly interfaced to the <a href="http://www.genie-mc.org/">GENIE</a> Monte Carlo
  neutrino event generators.
</p>


# Quick links

- [Installation instructions](https://nnpdf.github.io/nnusf/quickstart/installation.html)
- [Tutorials](https://nnpdf.github.io/nnusf/tutorials/datasets.html)
- [Delivery & Usage](https://nnpdf.github.io/nnusf/delivery/lhapdf.html)


# Citation

To refer to NNSFν in a scientific publication, please use the following:
```bibtex
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
```
And if NNSFν proved to be useful in your work, consider also to reference the codes:
```bibtex
@misc{https://doi.org/10.5281/zenodo.7657132,
  doi = {10.5281/ZENODO.7657132},
  url = {https://zenodo.org/record/7657132},
  author = "Candido, Alessandro and Garcia, Alfonso and Magni, Giacomo and Rabemananjara, Tanjona and Rojo, Juan and Stegeman, Roy",
   title = "{Neutrino Structure Functions from GeV to EeV Energies}",
  publisher = {Zenodo},
  year = {2023},
  copyright = {Open Access}
}
```
