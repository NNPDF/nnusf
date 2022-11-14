## Preparing the run card

In order to run a fit, we first need to prepare a run card. An example of
such a run card can be found in the following [folder](https://github.com/NNPDF/nnusf/tree/main/runcards).
Essentially, there are two main important keys in the run card which specify
the data sets to be included and the fitting parameters.

The specifications regarding the data sets are stored in the `experiment` key
in which one needs to specify the data set name and the training fraction as
follows:
```yaml
experiments:
- {dataset: BEBCWA59_F2, frac: 0.75}
- {dataset: BEBCWA59_F3, frac: 0.75}
```

The details regarding the fit instead are stored in the `fit_parameters` key.
Generally, it has the following structure:
```yaml
fit_parameters:
  epochs: 100000
  stopping_patience: 20000
  units_per_layer: [70, 55, 40, 20, 20]
  activation_per_layer: [tanh, tanh, tanh, tanh, selu]
  optimizer_parameters:
    optimizer: Adam
    clipnorm: 0.00001
    learning_rate: 0.05
  val_chi2_threshold: 4
  small_x_exponent:
      f2nu  : [0.25, 2.0]
      flnu  : [0.25, 2.0]
      xf3nu : [0.25, 2.0]
      f2nub : [0.25, 2.0]
      flnub : [0.25, 2.0]
      xf3nub: [0.25, 2.0]
```


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
