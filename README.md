<h1 align="center">NνSF</h1>

NνSF is a python module that provides predictions for neutrino structure functions. It relies on [Yadism](https://github.com/N3PDF/yadism) for the large-$Q^2$ region while the low-$Q^2$ regime is modelled in terms of a Neural Network (NN).

## Installation & Development

The package can be installed from source using the following commands:

```bash
git clone https://github.com/NNPDF/nnusf.git
cd nnusf
poetry install
```

This also provides the user the ability to develop on the codes.

## Usage

NνSF provides an extensive Command Line Interface (CLI) that permits the user to perform various classes of tasks. To know more about all the available options, just run `nnu --help`. For convenience, below we provide details on how the fitting part of the codes can be run.

#### Perform a fit

To run a fit, one can simplify type the following commands:

```bash
nnu fit run <runcard> <replica> -d <output_path>
```
An example of a runcard to perform a fit is [./runcards/fit_runcard.yml](./runcards/fit_runcard.yml).

This will generate inside the folder `<output_path>` a folder named `replica_<replica>` which in turn contains a tensorflow model that can be used to generate predictions. In general, one needs to run the above command for `replica={1, ..., n}`.

#### Perform a Postfit

If needed, one can perform a post-selection on the replicas generated from the fit. For instance, one can only select the replicas whose $\chi^2$ values are below some thresholds. Below is an example in which we only select replicas with $\chi^2_{\rm tr}$ and $\chi^2_{\rm vl}$ below `10`:
```bash
nnu fit postfit <output_path> -t '{"tr_max": 10, "vl_max": 10}'
```
This will generate inside `<output_path>` a folder named `postfit` which contains the replicas that satisfy the selection criteria.

#### Generate a report & Predictions

To generate a report from a fit, one can simply run:
```bash
nnu report generate <output_path>/postfit -t "<Title>" -a "<author>" -k "<keyword>"
```
This will generate a folder called `output` inside `<output_path>` which contains an `index.html` summarizing the results. If `postfit` was not run in the previous step, simply remove `/postfit` in the command above.

Finally, to generate predictions using the trained models, just run the following commands:

```bash
nnu plot fit <output_path> <runcard>
```
An example of such a runcard is [./runcards/generate_predictions.yml](./runcards/generate_predictions.yml).

This will generate a `.txt` file containing the NN predictions for the different structure functions.
