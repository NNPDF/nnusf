# Bodek-Yang comparison

Running `yadism` for:

- LO - nubar

  ```
  poetry run nnu theory runcards by -u '{"PTO": 0, "FNS": "ZM-VFNS", "IC": 0}' -o '{"ProjectileDIS": "antineutrino"}'
  poetry run nnu theory grids theory/runcards.tar
  poetry run nnu theory predictions theory/grids.tar NNPDF40_nnlo_as_01180 --err pdf -x 23
  ```

- NNLO - nubar

  ```
  poetry run nnu theory runcards by -u '{"PTO": 0, "FNS": "ZM-VFNS", "IC": 0}' -o '{"ProjectileDIS": "antineutrino"}'
  poetry run nnu theory grids theory/runcards.tar
  poetry run nnu theory predictions theory/grids.tar NNPDF40_nnlo_as_01180 --err pdf -x 23
  ```

Note: in order to suffer as least as possible, the following strategy is
suggested to deal with LHAPDF:

- install in the environment the user-wide library (i.e. activate the
  environment, but don't set `$PREFIX`, use `N3PDF/external` installer)
- export the environment variable `LHAPDF_DATA_PATH` to point at your favorite
  PDF storage folder

### Convert numpy arrays to txt

A script is provided in `docs/sf_to_txt.py`, to be used in the following way:

```sh
python docs/sf_to_txt.py theory/predictions.tar
```

And text file will be appended to the archive.
