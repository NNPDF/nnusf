import itertools
import pathlib
import tarfile
from typing import Optional

import yaml

pkg = pathlib.Path(__file__).parent.absolute()


def read(path: pathlib.Path, what=None) -> dict:
    # autodetect
    if what is None:
        what = path.suffix[1:]

    if what == "yaml":
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Format to be read undetected (attempted for '{what}')")


def write(content: dict, path: pathlib.Path, what=None):
    # autodetect
    if what is None:
        what = path.suffix[1:]

    if what == "yaml":
        path.write_text(yaml.dump(content), encoding="utf-8")
    else:
        raise ValueError(f"Format to be read undetected (attempted for '{what}')")


def extract_tar(path: pathlib.Path, dest: pathlib.Path, subdirs: Optional[int] = None):
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


three_points = [0.5, 1.0, 2.0]
"Three points prescription for scale variations."
nine_points = list(itertools.product(three_points, three_points))
"""Nine points prescription for scale variations (as couples, referred to ``(fact,
ren)`` scales)."""
