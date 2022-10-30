# -*- coding: utf-8 -*-
"""File management utilities."""
import logging
import pathlib
import tarfile
from typing import Optional

import pygit2
import yaml

# Package location, wherever it is installed
pkg = pathlib.Path(__file__).parent.absolute()

_logger = logging.getLogger(__name__)


class GitVerionsNonMatchError(Exception):
    pass


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
        if repo.diff().stats.files_changed >= 0:
            _logger.warning("There are unstaged files in the repository.")
        # repo = pygit2.Repository(pathlib.Path().cwd())
        commit = str(repo[repo.head.target].id)
        _logger.info("Git versions checked successfully.")

        if fit_version != commit:
            raise GitVerionsNonMatchError(
                f"The git version '{fit_version}' from which the fit was produced"
                f" and the current git version '{commit}' from which the report is"
                f" about to be generated are different. Please switch to the branch"
                f" from the fit was generated."
            )
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
