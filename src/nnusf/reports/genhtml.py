# -*- coding: utf-8 -*-
import pathlib
import shutil
from fileinput import FileInput
from textwrap import dedent

import pandas as pd
import yaml

from .genfiles import (
    addinfo_yaml,
    additional_plots,
    build_data_models,
    chi2_tables,
    compute_totchi2,
    data_vs_predictions,
    summary_table,
)
from ..sffit.scaling import extract_extreme_values

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
    number_plots = len([plot for plot in plots])

    for i in range(number_plots):
        fpath = f"figures/prediction_data_comparison_{i}.png"
        plot_path = index_path.joinpath(fpath)
        name = str(plot_path).split("/")[-1][:-4]
        path = plot_path.relative_to(index_path)
        pdf = path.with_suffix(".pdf")
        html_entry += f"""
    <div>
    <figure>
    <img src="{path}" id="{name}"
    alt=".png" />
    <figcaption aria-hidden="true"><a
    href="{path}">.png</a><a href="{pdf}">
    .pdf</a></figcaption>
    </figure>
    </div>
        """
    return html_entry + f"\n</div>"


@check_arguments
def split_trvl_html(figures: pathlib.Path) -> str:
    index_path = figures.absolute().parent
    html_entry = f"""
    <h1 id="trvl-split">Training details</h1>
    <div class="figiterwrapper">
    """
    html_entry = dedent(html_entry)
    plots = [
        figures.joinpath("chi2_split.png"),
        figures.joinpath("distr_epochs.png"),
    ]

    for plot in plots:
        name = str(plot).split("/")[-1][:-4]
        path = plot.relative_to(index_path)
        pdf = path.with_suffix(".pdf")
        html_entry += f"""
    <div>
    <figure>
    <img src="{path}" id="{name}"
    alt=".png" />
    <figcaption aria-hidden="true"><a
    href="{path}">.png</a><a href="{pdf}">
    .pdf</a></figcaption>
    </figure>
    </div>
    """
    return html_entry + f"\n</div>"


def main(fitfolder: pathlib.Path, **metadata) -> None:
    # Generate the various tables and predictions
    data_vs_predictions(fitfolder=fitfolder)
    additional_plots(fitfolder=fitfolder)

    # Compute the total Chi2 from the trained model
    cardrun = addinfo_yaml(fitfolder=fitfolder)
    models, raw_data, scaled_data = build_data_models(**cardrun)

    # Compute the total Chi2 for the real datasets
    totchi2, normalized_chi2 = compute_totchi2(models, scaled_data)

    # Construct the Chi2 tables & generate metadata
    summtable = summary_table(fitfolder=fitfolder, chi2tot=totchi2)
    chi2table = chi2_tables(fitfolder=fitfolder, chi2_datasets=normalized_chi2)
    generate_metadata(fitfolder, **metadata)

    # Define the list of tables to be constructed in the report
    tabvalues = [summtable, chi2table]
    tablabels = ["summary", "chi2"]

    if cardrun.get("check_chi2_experiments", None) is not None:
        from .genfiles import nonfitted_chi2

        # Compute the extremum from the raw datasets
        max_kins = extract_extreme_values(raw_data)
        # And use the information to Compute the Chi2 of nonfitted
        xchi2 = nonfitted_chi2(models, max_kins, fitfolder, **cardrun)

        # Append the tables to the lists above
        tabvalues += [xchi2]
        tablabels += ["postchi2"]

    # Construct the paths to the corresponding folders
    output_folder = fitfolder.absolute().parent
    figures = output_folder.joinpath("output/figures")
    others = output_folder.joinpath("output/others")

    # Generate the different html files & store them
    chi2_html = map(dump_table_html, tabvalues, tablabels)
    comparison_data = data_comparison_html(figures)
    trvl_split_html = split_trvl_html(others)

    # Combine all the resulted HTMLs into one
    combined = "".join(list(chi2_html)) + comparison_data + trvl_split_html

    index = CURRENT_PATH.parent.joinpath("assets/index.html")
    index_store = output_folder.joinpath("output/index.html")
    shutil.copyfile(index, index_store)
    metadata["contents_html"] = combined
    build_html(index_store, **metadata)

    # Copy the report CSS file into the output directory
    report = CURRENT_PATH.parent.joinpath("assets/report.css")
    shutil.copyfile(report, f"{output_folder}/output/report.css")
