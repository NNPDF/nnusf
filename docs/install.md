## User installation :id=installation

The easiest way to install `NνSF` is via the [Python Package Index](https://pypi.org/)
via the following command:
```bash
pip install nnusf
```
To check that the package has been installed correctly, just type the following:
```bash
nnu --version
```

## Development installation

In order to develop on the codes it is required to clone the github repository and
install the package using [Poetry](https://python-poetry.org/). To install `poetry`
just follow the instructions [here](https://python-poetry.org/docs/#installation).
Once this is done, first clone the repository and enter into the directory:

```bash
git clone https://github.com/NNPDF/nnusf.git --depth 1
cd nnusf
```
Then, to install the package just type:
```bash
poetry install
```
