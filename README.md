<table style="width:100%">
  <tr>
    <td rowspan="2" style="width:50px">
      <b>NNSFν</b> 
    </td>
    <td>
      <img alt="Zenodo" src="https://zenodo.org/badge/DOI/10.1101/2023.02.15.204701.svg">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2223.04638-b31b1b?labelColor=222222">
      <img alt="Docs" src="https://assets.readthedocs.org/static/projects/badges/passing-flat.svg">
      <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </td>
  </tr>
  <tr>
    <td> 
      <p align="justify">
        <b>NNSFν</b> is a python module that provides predictions for neutrino structure functions. 
        It relies on <a href="https://github.com/N3PDF/yadism">YADISM</a> for the large-Q region 
        while the low-Q regime is modelled in terms of a Neural Network (NN). The NNSFν 
        determination is also made available in terms of fast interpolation
        <a href="https://lhapdf.hepforge.org/">LHAPDF</a> grids that can be accessed through an independent
        driver code and directly interfaced to the <a href="http://www.genie-mc.org/">GENIE</a> Monte Carlo
        neutrino event generators.
      </p>
    </td>
  </tr>
</table>


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
