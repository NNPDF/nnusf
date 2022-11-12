## Generating Yadism cards

An alternative way to generate Yadism theory cards that will be used to
generate theory predictions is to use the `nnusf.theory.runcards` module.
This is in particular convenient in order to generate standalone predictions
that will be dumped into grids using the LHAPDF format.

To generate the theory card for a given $A$ value, just run the following:
```bash
nnu theory runcards yadknots -a ${A_VALUE} [--q2_grids '{"min": 9e2, "max": 1.96e8, "num": 200}']
```
The command above will dump the Yadism card as a compressed `.tar` file
inside a directory called `theory` unless otherwise specified. Notice that
as illustrated above one can optionally specify the $Q^2$ range. By default,
the range is taken to be between $[9 \cdot 10^2, 1.96 \cdot 10^8]~\mathrm{GeV}^2$.
As will be explained later, this range is chosen in order to match the high-$Q^2$
yadism predictions with the N$\nu$SF predictions.


## Generating PineAPPL grids

The theory card can now be passed to Yadism in order to generate the PineAPPL
grids. This can be done in the following way:
```bash
nnu theory grids theory/runcards-yadknots_A${A_VALUE}.tar
```
As before, unless otherwise specified, the command above will store the PineAPPL
grid inside the `theory` folder with the name `grids-runcards.tar`.


## Dumping predictions as LHAPDF

In order to compute the final predictions and dump the results into LHAPDF grids,
we need to convolute the PineAPPL grid with the corresponding (nuclear) PDFs. To
do so, just run the following command:
```bash
nnu theory predictions theory/grids-runcards.tar ${PDFSET_NAME} --err pdf --no-compare_to_by
```
This will generate in the current directory a LHAPDF set called `YADISM_${A_VALUE}`
which for use should be place in the LHAPDF data directory `lhapdf-config --datadir`.
