# -*- coding: utf-8 -*-
import logging
import pathlib
import shutil
from fileinput import FileInput
from textwrap import dedent

import pandas as pd
import yaml

from .genfiles import chi2_tables, data_vs_predictions, summary_table

_logger = logging.getLogger(__name__)


CURRENT_PATH = pathlib.Path(__file__)


class FolderNotFound(Exception):
    pass


def check_arguments(func):
    def wrapper(folder, **kwargs):
        if not folder.exists():
            raise FolderNotFound("The folder does not exist.")
        result = func(folder, **kwargs)
        return result

    return wrapper


def generate_metadata(
    folder: pathlib.Path,
    report_title: str,
    author_name: str,
    keyword_notes: str,
) -> None:
    index_path = folder.absolute().parents[0].joinpath("output")
    info = {
        "title": report_title,
        "author": author_name,
        "keywords": keyword_notes,
    }
    with open(f"{index_path}/meta.yaml", "w") as ostream:
        yaml.dump(info, ostream, sort_keys=False)
    _logger.info(
        f"The meda.yaml file stored in folder: "
        f"'{index_path.relative_to(pathlib.Path.cwd())}'"
    )


def dump_table_html(table: pd.DataFrame, name: str) -> str:
    """Build the HTML containing the tables"""
    tables = f"""\n<h2 id="{name}">{name} table</h2>"""
    return tables + f"""\n{table.to_html(border=0)}"""


@check_arguments
def build_html(htmlfile: pathlib.Path, **replacement_rules):
    for key, replace_with in replacement_rules.items():
        for line in FileInput(str(htmlfile), inplace=True):
            print(line.replace(key, replace_with), end="")


@check_arguments
def data_comparison_html(figures: pathlib.Path) -> str:
    index_path = figures.absolute().parent
    html_entry = f"""
    <h1 id="compare-data">Comparisons to Data</h1>
    <div class="figiterwrapper">
    """
    html_entry = dedent(html_entry)
    plots = figures.glob("**/prediction_data_comparison_*.png")

    for plot in plots:
        name = str(plot).split("/")[-1][:-4]
        path = plot.relative_to(index_path)
        html_entry += f"""
    <div>
    <figure>
    <img src="{path}" id="{name}"
    alt=".png" />
    <figcaption aria-hidden="true"><a
    href="{path}">.png</a></figcaption>
    </figure>
    </div>
        """
    return html_entry + f"\n</div>"


def main(fitfolder: pathlib.Path, **metadata) -> None:
    # Generate the various tables and predictions
    data_vs_predictions(fitfolder=fitfolder)
    summtable = summary_table(fitfolder=fitfolder)
    chi2table = chi2_tables(fitfolder=fitfolder)
    generate_metadata(fitfolder, **metadata)

    # Construct the paths to the corresponding folders
    output_folder = fitfolder.absolute().parent
    figures = output_folder.joinpath("output/figures")

    # Generate the different html files & store them
    chi2s_html = map(
        dump_table_html,
        [summtable, chi2table],
        ["summary", "chi2s"],
    )
    summary_html, chi2s_html = list(chi2s_html)
    comparison_data_html = data_comparison_html(figures)

    # Combine all the resulted HTMLs into one
    combined = summary_html + chi2s_html + comparison_data_html
    index = CURRENT_PATH.parent.joinpath("assets/index.html")
    index_store = output_folder.joinpath("output/index.html")
    shutil.copyfile(index, index_store)
    metadata["contents_html"] = combined
    build_html(index_store, **metadata)

    # Copy the report CSS file into the output directory
    report = CURRENT_PATH.parent.joinpath("assets/report.css")
    shutil.copyfile(report, f"{output_folder}/output/report.css")
