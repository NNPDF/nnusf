import yaml

from .. import utils


def theory() -> dict:
    runcard = yaml.safe_load(
        (utils.pkg / "theory_200.yaml").read_text(encoding="utf-8")
    )
    return runcard
