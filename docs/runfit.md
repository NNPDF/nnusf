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
nnu fit run ${PATH_RUNCARD} ${REPLICA_ID} [-d ${OUTPUT_PATH}]
```
An example of a runcard to perform a fit is [runcards/fit_runcard.yml](../runcards/fit_runcard.yml).

This will generate inside a folder `RUNCARD_NAME` folders called `replica_${REPLICA_ID}` which in turn
contain tensorflow models that can be used to generate predictions. In general, one needs to run the
above command for `REPLICA_ID={1, ..., n}`.

!> A pre-trained model is available in the following [link](https://data.nnpdf.science/NNUSF/) if the
user wants to skip the training part and directly wants to generate predictions.


#### Post-fit selection

If needed, one can perform a post-selection on the replicas generated from the fit. For instance, one
can only select replicas whose $\chi^2$ values are below some thresholds. Below is an example in which
we only select replicas with $\chi^2_{\rm tr}$ and $\chi^2_{\rm vl}$ below `2.8`:
```bash
nnu fit postfit ${RUNCARD_NAME} -t '{"tr_max": 2.8, "vl_max": 2.8}'
```
This will generate inside `RUNCARD_NAME` a folder called `postfit` which contains the replicas that satisfy
the selection criteria.


#### Generate a fit report

Using the trained model, we can generate a report containing the summary of the $\chi^2$ values and
the comparisons between the experimental data sets and the N$\nu$SF predictions. To generate the report
just run the following command:
```bash
nnu report generate ${RUNCARD_NAME}/postfit -t "<Title>" -a "<author>" -k "<keyword>"
```
This will generate a folder called `output` inside `RUNCARD_NAME` which contains an `index.html` summarizing
the results. The `.html` file can then be opened on a browser. If `postfit` was not run in the previous step,
simply remove `/postfit` in the command above.


#### Store N$\nu$SF predictions as LHAPDF

For future convenience, the N$\nu$SF predictions can be stored as LHAPDF grids. The structure functions
have the following LHAPDF IDs:

| Structure Functions  | $F_2^{\nu}$   | $F_L^{\nu}$  | $x F_3^{\nu}$  |  $F_2^{\bar{\nu}}$ |  $F_L^{\bar{\nu}}$ |  $x F_3^{\bar{\nu}}$ | $\langle F_2 \rangle$  |  $\langle F_L \rangle$ |  $\langle x F_3 \rangle$ |
|---|---|---|---|---|---|---|---|---|---|
| LHAPDF ID | 1001  | 1002  | 1003  | 2001  | 2002  | 2003 |  3001  | 3002  | 3003  |

The LHAPDF set can be generated using the following command:
```bash
nnu fit dump_grids ${RUNCARD_NAME}/postfit -a ${A_VALUE} -o ${SET_NAME} [-q '{"min": 1e-3, "max": 400, "num": 100}]'
```
!> As before, the user can choose the ranges of $x$ and $Q^2$ from which the predictions will be
generated. By default, the $Q^2$ range is defined to be between $[10^{-3}, 4 \cdot 10^2]$. This
range is chosen based on the $Q^2$ values included in the fit through both the experimental and
Yadism pseudo data.
