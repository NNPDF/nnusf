# -*- coding: utf-8 -*-
"""File management utilities."""
import pathlib
import tarfile
from typing import Optional

import yaml

pkg = pathlib.Path(__file__).parent.absolute()
"""Package location, wherever it is installed."""


def read(path: pathlib.Path, what=None) -> dict:
    """Read a file, the suitable way."""
    # autodetect
    if what is None:
        what = path.suffix[1:]

    if what == "yaml":
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Format to be read undetected (attempted for '{what}')")


def write(content: dict, path: pathlib.Path, what=None):
    """Write a file, the suitable way."""
    # autodetect
    if what is None:
        what = path.suffix[1:]

    if what == "yaml":
        path.write_text(yaml.dump(content), encoding="utf-8")
    else:
        raise ValueError(f"Format to be read undetected (attempted for '{what}')")


def extract_tar(path: pathlib.Path, dest: pathlib.Path, subdirs: Optional[int] = None):
    """Extract a tar archive to given directory."""
    with tarfile.open(path) as tar:
        tar.extractall(dest)

    if subdirs is not None:
        count = len(list(dest.iterdir()))
        if count == subdirs:
            return

        expected = f"{subdirs} folders are" if subdirs > 1 else "A single folder is"
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
                f"The given destination exists, but is not a directory - '{destination}'"
            )
    else:
        destination.mkdir(parents=True)
