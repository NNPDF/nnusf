# -*- coding: utf-8 -*-
"""File management utilities."""
import sys
import logging
import pathlib
import tarfile
from typing import Optional, Union

import lhapdf
import numpy as np
import pygit2
import yaml

# Package location, wherever it is installed
pkg = pathlib.Path(__file__).parent.absolute()

_logger = logging.getLogger(__name__)

ROUNDING = 6


def add_git_info(runcard_dict: dict):
    """Add the git version from which the fit was generated to the
    run card. This could later on be compared to the git version
    from which the report will be generated in order to make everything
    fully consistent.

    Parameters:
    -----------
    runcard_dict: dict
        dictionary containing information on the run card
    """
    try:
        import nnusf

        repo = pygit2.Repository(nnusf.__path__[0])
        # repo = pygit2.Repository(pathlib.Path().cwd())
        commit = repo[repo.head.target]
        runcard_dict["git_info"] = str(commit.id)
    except pygit2._pygit2.GitError as msg:
        runcard_dict["git_info"] = ""
        _logger.warning(f"Git version could not be retrieved! {msg}")


def compare_git_versions(runcard_dict: dict) -> None:
    """Compare the git versions from a fit card and the current
    local. If they are not the same then raises an error. This is
    relevant when it comes to generating report. This ensures that
    the report and subsequent LHAPDF grids are generated consistently.

    Parameters:
    -----------
    runcard_dict: dict
        dictionary containing information on the run card
    """
    try:
        import nnusf

        fit_version = runcard_dict["git_info"]
        repo = pygit2.Repository(nnusf.__path__[0])

        # Check if there are unstaged files in repository
        if repo.diff().stats.files_changed > 0:
            _logger.warning("There are unstaged files in the repository.")
        # repo = pygit2.Repository(pathlib.Path().cwd())
        commit = str(repo[repo.head.target].id)
        _logger.info("Git versions checked successfully.")

        if fit_version != commit:
            _logger.error(
                f"The git version '{fit_version}' from which the fit was produced"
                f" and the current git version '{commit}' from which the report is"
                f" about to be generated are different. Please switch to the branch"
                f" from the fit was generated."
            )
            sys.exit()
    except pygit2._pygit2.GitError as msg:
        _logger.warning(f"Git version could not be retrieved! {msg}")


def read(path: pathlib.Path, what=None) -> dict:
    """Read a file, the suitable way."""
    # autodetect
    if what is None:
        what = path.suffix[1:]

    if what == "yaml":
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"Format to be read undetected (attempted for '{what}')"
        )


def write(content: dict, path: pathlib.Path, what=None):
    """Write a file, the suitable way."""
    # autodetect
    if what is None:
        what = path.suffix[1:]

    if what == "yaml":
        path.write_text(yaml.dump(content), encoding="utf-8")
    else:
        raise ValueError(
            f"Format to be read undetected (attempted for '{what}')"
        )


def extract_tar(
    path: pathlib.Path, dest: pathlib.Path, subdirs: Optional[int] = None
):
    """Extract a tar archive to given directory."""
    with tarfile.open(path) as tar:
        tar.extractall(dest)

    if subdirs is not None:
        count = len(list(dest.iterdir()))
        if count == subdirs:
            return

        expected = (
            f"{subdirs} folders are" if subdirs > 1 else "A single folder is"
        )
        found = f"{count} files" if count > 1 else "a single file"
        raise ValueError(
            f"{expected} supposed to be contained by the tar file,"
            f" but more {found} have been detected"
        )


def mkdest(destination: pathlib.Path):
    """Make sure destination exists.

    Create it if does not exist, else make sure it is a directory.

    Parameters
    ----------
    destination: pathlib.Path
        path to check

    Raises
    ------
    NotADirectoryError
        if it exists but it is not a directory

    """
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(
                f"The given destination exists, but is not a"
                f" directory - '{destination}'"
            )
    else:
        destination.mkdir(parents=True)


def split_data_path(ds: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Extract dataset name, and commondata folder.

    Parameters
    ----------
    ds: pathlib.Path
        path to dataset

    Returns
    -------
    str
        dataset name
    pathlib.Path
        commondata base folder

    """
    name = ds.stem.strip("DATA_")

    return name, ds.parents[1]


def compute_lhapdf(
    pdfname: str,
    xgrid: Optional[Union[list, np.ndarray]] = None,
    q2grid: Optional[Union[list, np.ndarray]] = None,
    pid: Optional[list] = None,
):
    """Compute the high-Q2 Yadism predictions from a LHAPDF set.

    Returns:
    --------
        tuple(np.ndarray, dict)
            a tuple containing the predictions that has a shape
            (nrep, n_q2, n_x, n_pid) and a dictionary containing
            some metadata.
    """

    _logger.info("Computing the LHAPDF predictions.")
    input_pdfset = lhapdf.mkPDFs(pdfname)
    pdfinfo, n_rep = input_pdfset[0], len(input_pdfset)

    if xgrid is None:
        xgrid = np.geomspace(pdfinfo.xMin, pdfinfo.xMax, 100)

    if q2grid is None:
        q2grid = np.geomspace(pdfinfo.q2Min, pdfinfo.q2Max, 200)

    pid = pdfinfo.flavors() if pid is None else pid

    xmesh, q2mesh = np.meshgrid(xgrid, q2grid)
    store_replicas = []
    for replica in input_pdfset:
        pred = replica.xfxQ2(pid, xmesh.flatten(), q2mesh.flatten())
        pred = np.asarray(pred).reshape((*xmesh.shape, len(pid)))
        store_replicas.append(pred)

    if not isinstance(xgrid, list):
        xgrid = xgrid.tolist()

    if not isinstance(q2grid, list):
        q2grid = q2grid.tolist()

    q2grid = [round(q, ROUNDING) for q in q2grid]

    metadata = dict(x_grids=xgrid, q2_grids=q2grid, pid=pid, nrep=n_rep)
    return store_replicas, metadata
