# -*- coding: utf-8 -*-
import logging
import pathlib

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

OBS_TYPE = ["F2", "F3", "FW", "DXDYNUU", "DXDYNUB", "QBAR"]

MAP_EXP_YADISM = {"NUTEV": "XSNUTEVCC", "CHORUS": "XSCHORUSCC", "CDHSW": "XSCHORUSCC"}


class ObsTypeError(Exception):
    pass


class Loader:
    """Load a dataset given the name and build the covariance matrix

    Parameters
    ----------
    path_to_commondata:
        path to commondata folder
    path_to_theory:
        path to theory folder
    data_name:
        dataset name
    data_type: str
        data type: F2,F3, DXDYNUU, DXDYNUB

    """

    def __init__(
        self,
        path_to_commondata: pathlib.Path,
        path_to_theory: pathlib.Path,
        data_name: str,
    ) -> None:

        self.data_name = data_name
        self.data_type = data_name.split("_")[1]
        if self.data_type not in OBS_TYPE:
            raise ObsTypeError("Observable not implemented or Wrong!")

        self.commondata_path = path_to_commondata
        self.theory_path = path_to_theory
        self.fulltables = self.load()
        self.covariance_matrix = self.build_covariance_matrix(self.fulltables)

        _logger.info(f"Loaded '{data_name}' dataset")

    def load(self) -> pd.DataFrame:
        """Load the dataset information

        Returns
        -------
        table with loaded data

        """
        # info file
        exp_name = self.data_name.split("_")[0]
        info_df = pd.read_csv(f"{self.commondata_path}/info/{exp_name}.csv")

        # Extract values from the kinematic tables
        kin_file = self.commondata_path.joinpath(f"kinematics/KIN_{self.data_name}.csv")
        if kin_file.exists():
            kin_df = pd.read_csv(kin_file).iloc[1:, 1:4].reset_index(drop=True)
        elif self.data_type in ["F2", "F3"]:
            kin_df = pd.read_csv(
                f"{self.commondata_path}/kinematics/KIN_{exp_name}_F2F3.csv"
            ).iloc[1:, 1:4].reset_index(drop=True)
        elif self.data_type in ["DXDYNUU", "DXDYNUB"]:
            kin_df = pd.read_csv(
                f"{self.commondata_path}/kinematics/KIN_{exp_name}_DXDY.csv"
            ).iloc[1:, 1:4].reset_index(drop=True)


        # Extract values from the central and uncertainties
        data_df = pd.read_csv(
            f"{self.commondata_path}/data/DATA_{self.data_name}.csv",
            header=0,
            na_values=["-", " "],
        ).iloc[:, 1:].reset_index(drop=True)
        unc_df = pd.read_csv(
            f"{self.commondata_path}/uncertainties/UNC_{self.data_name}.csv",
            na_values=["-", " "],
        ).iloc[2:, 1:].reset_index(drop=True)

        # Concatenate enverything into one single big table
        new_df = pd.concat([kin_df, data_df, unc_df], axis=1)
        new_df = new_df.dropna().astype(float)

        # drop data with 0 total uncertainty:
        new_df = new_df[new_df["stat"] + new_df["syst"] != 0.0]

        number_datapoints = new_df.shape[0]

        # Extract the information on the cross section
        # FW is a different case
        if self.data_type == "FW":
            data_spec = "FW"
        else:
            data_spec = MAP_EXP_YADISM.get(exp_name, None)

        # Append all the columns to the `kin_df` table
        new_df["A"] = np.full(number_datapoints, info_df["target"][0])
        new_df["xsec"] = np.full(number_datapoints, data_spec)
        new_df["Obs"] = np.full(number_datapoints, self.data_type)
        new_df["projectile"] = np.full(number_datapoints, info_df.loc[info_df["type"] == self.data_type, "projectile"])

        return new_df

    @property
    def kinematics(self) -> np.ndarray:
        """Returns the kinematics variables"""
        return self.fulltables[["x", "Q2", "A"]].values

    @property
    def n_data(self):
        """Returns the number of datapoints"""
        return self.fulltables.shape[0]

    @property
    def central_values(self) -> np.ndarray:
        """Returns the dataset central values"""
        return self.fulltables["data"].values

    @property
    def covmat(self) -> np.ndarray:
        """Returns the covariance matrix"""
        return self.covariance_matrix

    @property
    def coefficients(self) -> np.ndarray:
        """Returns the coefficients prediction"""
        return self.coeff_array

    @staticmethod
    def build_covariance_matrix(unc_df: pd.DataFrame) -> np.ndarray:
        """Build the covariance matrix given the statistical and systematics uncertainties

        Parameters
        ----------
        unc_df:
            uncertainties table
        Returns
        -------
        covariance matrix

        """
        diagonal = np.power(unc_df["stat"], 2)
        corr_sys = unc_df["syst"]
        return np.diag(diagonal) + np.outer(corr_sys, corr_sys)
