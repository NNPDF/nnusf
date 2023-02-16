<h1 align="center">NNSFν</h1>
<p align="center">
  <img alt="Zenodo" src="https://zenodo.org/badge/DOI/10.1101/2023.02.15.204701.svg">
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2223.04638-b31b1b?labelColor=222222">
  <img alt="Docs" src="https://assets.readthedocs.org/static/projects/badges/passing-flat.svg">
  <img alt="Status" src="https://www.repostatus.org/badges/latest/active.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

NNSFν is a python module that provides predictions for neutrino structure functions. 
It relies on [Yadism](https://github.com/N3PDF/yadism) for the large $Q^2$ region 
while the low $Q^2$ regime is modelled in terms of a Neural Network (NN). The NNSFν 
determination is also made available in terms of fast interpolation
[LHAPDF](https://lhapdf.hepforge.org/) grids that can be accessed through an independent
driver code and directly interfaced to the [GENIE](http://www.genie-mc.org/) neutrino 
event generators.

# Quick links

- [Installation instructions](https://nnpdf.github.io/nnusf/quickstart/installation.html)
- [Tutorials](https://nnpdf.github.io/nnusf/tutorials/datasets.html)
- [Delivery & Usage](https://nnpdf.github.io/nnusf/delivery/lhapdf.html)

# Citation

To refer to NNSFν in a scientific publication, please use the following:
```bibtex
@article {reference_id,
   author = {Alessandro Candido, Alfonso Garcia, Giacomo Magni, Tanjona Rabemananjara, Juan Rojo, Roy Stegeman},
   title = {Neutrino Structure Functions from GeV to EeV Energies},
   year = {2023},
   doi = {10.1101/2020.07.15.204701},
   eprint = {https://arxiv.org/list/hep-ph/},
   journal = {aRxiv}
}
```
And if NNSFν proved to be useful in your work, consider also to reference the codes:
```bibtex
@article {reference_id,
   author = {Alessandro Candido, Alfonso Garcia, Giacomo Magni, Tanjona Rabemananjara, Juan Rojo, Roy Stegeman},
   title = {Neutrino Structure Functions from GeV to EeV Energies},
   year = {2023},
   doi = {10.1101/2020.07.15.204701},
}
```
