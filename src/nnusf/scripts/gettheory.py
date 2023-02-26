# -*- coding: utf-8 -*-
import logging
import requests
import tarfile
import tempfile

from appdirs import AppDirs
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)


def setup_directory(dirname: str) -> Path:
    """Set up the user data directories.
    
    Parameters:
    -----------
    dirname: str
        Name of the App directory
    
    Returns:
    --------
    pathlib.Path:
        Path to the user data folder
    """
    dirs = AppDirs(dirname, "nnsfnu")
    return Path(dirs.user_data_dir)


def extract_tar(tempath: Path, userpath: Path) -> None:
    """Extract the .tar.gz file from the temporary directory
    and copy the resulting file to the user directory.

    Parameters:
    ----------
    tempath: Path
        Path to the temporary folder containing the compressed
        file
    userpath: Path
        Path to the user data folder
    """
    with tarfile.open(tempath, "r") as targz:
        targz.extractall(path=userpath)


def download_targz(response, tempdir: Path, filename: str) -> None:
    """Download the .tar.gz file and store the file in the
    specified temporary directory.

    Parameters:
    ----------
    tempdir: Path
        Path to the temporary directory
    filename: str
        Name of the .tar.gz file
    """
    with open(tempdir.joinpath(filename), 'wb') as ofile:
        ofile.write(response.raw.read())
    _logger.info(f"Theory downloaded succesfully in '{tempdir}'")


def get_theory(userdir: Path) -> None:
    """Download the theory in a temporary directory and
    extract it to the user data folder.
    
    Parameters:
    -----------
    userdir: Path
        Path to the user data folder
    """
    name = "nnusf_data.tar.gz"
    url = f"https://data.nnpdf.science/NNSFnu/data/{name}"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        _logger.error(err)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        download_targz(response, tmpdir, name)

        with tarfile.open(tmpdir.joinpath(name), "r") as trf:
            trf.extractall(path=userdir)

    _logger.info(f"Theory extracted succesfully in '{userdir}'")


def main():
    user_dir = setup_directory(dirname="nnusf")
    get_theory(userdir=user_dir)
