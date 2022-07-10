# -*- coding: utf-8 -*-
import logging
import pathlib
from textwrap import dedent

import pandas as pd

from .genfiles import chi2_tables, data_vs_predictions, summary_table

_logger = logging.getLogger(__name__)


class FolderNotFound(Exception):
    pass


def check_arguments(func):
    def wrapper(folder, *args):
        if not folder.exists():
            raise FolderNotFound("The folder does not exist.")
        result = func(folder, *args)
        return result

    return wrapper


def main(fitfolder: pathlib.Path) -> None:
    # Generate the various tables and predictions
    data_vs_predictions(fitfolder=fitfolder)
    summtable = summary_table(fitfolder=fitfolder)
    chi2table = chi2_tables(fitfolder=fitfolder)

    # Construct the paths to the corresponding folders
    output_folder = fitfolder.absolute().parents[0]
    figures = output_folder.joinpath("output/figures")
    tables = output_folder.joinpath("output/tables")

    # Generate the different html files & store them
    chi2s_html = map(
        dump_table_html,
        [tables, tables],
        [summtable, chi2table],
        ["summary", "chi2s"],
    )
    _ = list(chi2s_html)
    data_comparison_html(figures)


@check_arguments
def dump_table_html(
    folder: pathlib.Path, table: pd.DataFrame, name: str
) -> None:
    index_path = folder.absolute().parents[0]
    index = open(f"{index_path}/{name}_table.html", "w")
    header = f"""<h2 id="summary">Summary</h2>"""

    # Dump the Panda table into the HTML  file
    index.write(dedent(header))
    index.write(dedent(f"""{table.to_html()}"""))
    index.close()
    _logger.info(
        f"{name}_table HTML file stored in folder: "
        f"'{index_path.relative_to(pathlib.Path.cwd())}'"
    )


@check_arguments
def data_comparison_html(figures: pathlib.Path) -> None:
    index_path = figures.absolute().parents[0]
    index = open(f"{index_path}/data_comparison.html", "w")
    header = f"""\
        <h1 id="plot-pdfs">Plot PDFs</h1>
        <div class="figiterwrapper">
    """
    footer = f"</div>"
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
