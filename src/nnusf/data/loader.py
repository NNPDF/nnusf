# -*- coding: utf-8 -*-
"""Provide a Loader class to retrieve data information."""
import logging
import pathlib
from typing import Optional
from webbrowser import Elinks

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

OBS_TYPE = ["F2", "F3", "FW", "DXDYNUU", "DXDYNUB", "QBAR"]

MAP_EXP_YADISM = {
    "NUTEV": "XSNUTEVNU",
    "CHORUS": "XSCHORUSCC",
    "CDHSW": "XSCHORUSCC",
    # for the proton boundary condition the xs
    # definition is arbitrary
    "PROTONBC": "XSCHORUSCC",
}


class ObsTypeError(Exception):
    """Raised when observable is not recognized."""


class Loader:
    """Load a dataset given the name.

    This includes:

        - loading the central values
        - building the covariance matrix
        - (on demand) loading the coefficients (if the path is available)

    """

    def __init__(
        self,
        name: str,
        path_to_commondata: pathlib.Path,
        path_to_coefficients: Optional[pathlib.Path] = None,
        include_syst: Optional[bool] = True,
        w2min: Optional[float] = None,
    ):
        """Initialize object.

        Parameters
        ----------
        name: str
            dataset name
        path_to_commondata: os.PathLike
            path to commondata folder
        path_to_coefficients: os.PathLike or None
            path to theory folder
        include_syst:
            if True include syst
        w2min;
            if True cut all datapoints below `w2min`
        """
        self.name = name
        if self.obs not in OBS_TYPE:
            raise ObsTypeError(
                f"Observable '{self.obs}' not implemented or Wrong!"
            )

        self.commondata_path = path_to_commondata
        self.coefficients_path = path_to_coefficients
        self.table, self.leftindex = self._load(w2min)
        self.tr_frac = None
        self.covariance_matrix = self.build_covariance_matrix(
            self.table, include_syst
        )
        _logger.info(f"Loaded '{name}' dataset")

    def _load(self, w2min: float) -> tuple[pd.DataFrame, pd.Index]:
        """Load the dataset information.

        Returns
        -------
        table with loaded data

        """
        # info file
        exp_name = self.name.split("_")[0]
        if "_MATCHING-" in exp_name:
            exp_name = exp_name.strip("_MATCHING")
        info_df = pd.read_csv(f"{self.commondata_path}/info/{exp_name}.csv")

        # Extract values from the kinematic tables
        kin_file = self.commondata_path.joinpath(
            f"kinematics/KIN_{self.name}.csv"
        )
        if kin_file.exists():
            kin_df = pd.read_csv(kin_file).iloc[1:, 1:4].reset_index(drop=True)
        elif "_MATCHING" in self.name:
            if "FW" in self.name or "DXDY" in self.name:
                file = (
                    f"{self.commondata_path}/kinematics/KIN_MATCHING_XSEC.csv"
                )
            else:
                file = f"{self.commondata_path}/kinematics/KIN_MATCHING_FX.csv"
            kin_df = pd.read_csv(file).iloc[1:, 1:4].reset_index(drop=True)
        elif self.obs in ["F2", "F3"]:
            file = f"{self.commondata_path}/kinematics/KIN_{exp_name}_F2F3.csv"
            kin_df = pd.read_csv(file).iloc[1:, 1:4].reset_index(drop=True)
        elif self.obs in ["DXDYNUU", "DXDYNUB"]:
            file = f"{self.commondata_path}/kinematics/KIN_{exp_name}_DXDY.csv"
            kin_df = pd.read_csv(file).iloc[1:, 1:4].reset_index(drop=True)
        else:
            raise ObsTypeError("{self.obs} is not recognised as an Observable.")

        # Extract values from the central data
        dat_name = f"{self.commondata_path}/data/DATA_{self.name}.csv"
        data_df = pd.read_csv(dat_name, header=0, na_values=["-", " "])
        data_df = data_df.iloc[:, 1:].reset_index(drop=True)
        # Extract values from the uncertainties
        unc_name = f"{self.commondata_path}/uncertainties/UNC_{self.name}.csv"
        unc_df = pd.read_csv(unc_name, na_values=["-", " "])
        unc_df = unc_df.iloc[2:, 1:].reset_index(drop=True)

        # Add a column to `kin_df` that stores the W
        q2 = kin_df["Q2"].astype(float, errors="raise")  # Object -> float
        xx = kin_df["x"].astype(float, errors="raise")  # Object -> float
        kin_df["W2"] = q2 * (1 - xx) / xx + info_df["m_nucleon"][0]

        # Concatenate enverything into one single big table
        new_df = pd.concat([kin_df, data_df, unc_df], axis=1)
        new_df = new_df.dropna().astype(float)

        # drop data with 0 total uncertainty:
        if "_MATCHING" not in self.name:
            new_df = new_df[new_df["stat"] + new_df["syst"] != 0.0]

        # Restore index before implementing the W cut
        new_df.reset_index(drop=True, inplace=True)
        # Only now we can perform the cuts on W
        new_df = new_df[new_df["W2"] >= w2min] if w2min else new_df

        number_datapoints = new_df.shape[0]

        # Extract the information on the cross section (FW is a special case)
        if self.obs == "FW":
            data_spec = "FW"
        else:
            data_spec = MAP_EXP_YADISM.get(exp_name, None)

        # Append all the columns to the `kin_df` table
        new_df["A"] = np.full(number_datapoints, info_df["target"][0])
        new_df["xsec"] = np.full(number_datapoints, data_spec)
        new_df["Obs"] = np.full(number_datapoints, self.obs)
        new_df["projectile"] = np.full(
            number_datapoints,
            info_df.loc[info_df["type"] == self.obs, "projectile"],
        )
        new_df["m_nucleon"] = np.full(
            number_datapoints,
            info_df["m_nucleon"][0],
        )

        return new_df, new_df.index

    @property
    def exp(self) -> str:
        """Return the name of the experiment."""
        return self.name.split("_")[0]

    @property
    def obs(self) -> str:
        """Return the observable name."""
        return self.name.split("_")[1]

    @property
    def kinematics(self) -> np.ndarray:
        """Return the kinematics variables."""
        return self.table[["x", "Q2", "A"]].values

    @property
    def n_data(self):
        """Return the number of datapoints."""
        return self.table.shape[0]

    @property
    def central_values(self) -> np.ndarray:
        """Return the dataset central values."""
        return self.table["data"].values

    @property
    def covmat(self) -> np.ndarray:
        """Return the covariance matrix."""
        return self.covariance_matrix

    @property
    def coefficients(self) -> np.ndarray:
        """Return the coefficients prediction."""
        if self.coefficients_path is None:
            raise ValueError(
                f"No path available to load coefficients for '{self.name}'"
            )

        coeffs = np.load(
            (self.coefficients_path / self.name).with_suffix(".npy")
        )
        return coeffs[self.leftindex]

    @staticmethod
    def build_covariance_matrix(
        unc_df: pd.DataFrame, include_syst: bool
    ) -> np.ndarray:
        """Build the covariance matrix.

        It consumes as input the statistical and systematics uncertainties.

        Parameters
        ----------
        unc_df:
            uncertainties table
        include_syst:
            if True include syst

        Returns
        -------
        covariance matrix

        """
        diagonal = np.power(unc_df["stat"], 2)
        if include_syst:
            corr_sys = unc_df["syst"]
            return np.diag(diagonal) + np.outer(corr_sys, corr_sys)
        return np.diag(diagonal)
