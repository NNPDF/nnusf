import pathlib
import tarfile

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


def extract_tar(path: pathlib.Path, dest: pathlib.Path):
    with tarfile.open(path) as tar:
        tar.extractall(dest)

    content = iter(dest.iterdir())
    cards_path = next(content)
    try:
        next(content)
        raise ValueError(
            "A single folder is supposed to be contained by the tar file,"
            " but more files have been detected"
        )
    except StopIteration:
        pass
