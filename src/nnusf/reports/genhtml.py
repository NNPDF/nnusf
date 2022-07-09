# -*- coding: utf-8 -*-
import logging
import pathlib
from textwrap import dedent

from .genfiles import chi2_tables, data_vs_predictions, summary_table

_logger = logging.getLogger(__name__)


class FolderNotFound(Exception):
    pass


def check_arguments(func):
    def wrapper(folder):
        if not folder.exists():
            raise FolderNotFound("The folder does not exist.")
        result = func(folder)
        return result

    return wrapper


def main(fitfolder: pathlib.Path) -> None:
    # summary_table(fitfolder=fitfolder)
    # chi2_tables(fitfolder=fitfolder)
    # data_vs_predictions(fitfolder=fitfolder)

    # Generate
    output_folder = fitfolder.absolute().parents[0]
    figures = output_folder.joinpath("output/figures")
    data_comparison_html(figures)


@check_arguments
def data_comparison_html(figures: pathlib.Path) -> None:
    index_path = figures.absolute().parents[0]
    index = open(f"{index_path}/data_comparison.html", "w")
    header = f"""\
        <h1 id="plot-pdfs">Plot PDFs</h1>
        <div class="figiterwrapper">
    """
    footer = f"""
        </div>
    """
    index.write(dedent(header))

    plots = figures.glob("**/prediction_data_comparison_*.png")

    for plot in plots:
        name = str(plot).split("/")[-1][:-4]
        path = plot.relative_to(index_path)
        comparison_plot = f"""
            <div>
            <figure>
            <img src="{path}" id="{name}"
            alt=".png" />
            <figcaption aria-hidden="true"><a
            href="{path}">.png</a></figcaption>
            </figure>
            </div>
        """
        index.write(dedent(comparison_plot))

    index.write(dedent(footer))
    index.close()
    _logger.info(
        f"Data-Prediction comparisons HTML stored in "
        f"'{index_path.relative_to(pathlib.Path.cwd())}'"
    )
