import pathlib

import numpy as np
import pandas as pd


OBS_TYPE = ["F2", "F3", "DXDYNUU", "DXDYNUB"]

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
        data_type: str,
    ) -> None:
        if data_type not in OBS_TYPE:
            raise ObsTypeError("Observable not implemented or Wrong!")

        self.commondata_path = path_to_commondata
        self.theory_path = path_to_theory
        self.data_type = data_type
        self.data_name = f"{data_name}_{data_type}"
        self.fulltables = self.load()
        # self.covariance_matrix = self.build_covariance_matrix(self.fulltables)

    def load(self) -> pd.DataFrame:
        """Load the dataset information

        Returns
        -------
        kin_df:
            kinematic variables
        data_df:
            central values
        unc_df:
            uncertainties

        """
        # info file
        exp_name = self.data_name.split("_")[0]
        info_df = pd.read_csv(f"{self.commondata_path}/info/{exp_name}.csv")

        # Extract values from the kinematic tables
        kin_file = self.commondata_path.joinpath(f"kinematics/KIN_{self.data_name}.csv")
        if kin_file.exists():
            kin_df = pd.read_csv(kin_file).iloc[1:, :2]
        else:
            kin_df = pd.read_csv(
                f"{self.commondata_path}/kinematics/KIN_{exp_name}_F2F3.csv"
            ).iloc[1:, :2]

        # Extract values from the central and uncertainties
        data_df = pd.read_csv(
            f"{self.commondata_path}/data/DATA_{self.data_name}.csv",
            header=0,
            na_values=["-", " "],
        )
        unc_df = pd.read_csv(
            f"{self.commondata_path}/uncertainties/UNC_{self.data_name}.csv",
            na_values=["-", " "],
        )[2:]

        # Concatenate everything into one single big table
        new_df = pd.concat([kin_df, data_df, unc_df], axis=1)
        new_df = new_df.dropna().astype(float)
        number_datapoints = new_df.shape[0]

        # Extract the information on the cross section
        data_spec = MAP_EXP_YADISM.get(exp_name, None)

        # Append all the columns to the `kin_df` table
        new_df["A"] = np.full(number_datapoints, info_df["target"][0])
        new_df["xsec"] = np.full(number_datapoints, data_spec)
        new_df["Obs"] = np.full(number_datapoints, self.data_type)
        new_df["projectile"] = np.full(number_datapoints, info_df["projectile"][0])

        return new_df

    @property
    def kinematics(self) -> tuple:
        """Returns the kinematics variables"""
        return (
            self.fulltables["x"].values,
            self.fulltables["Q2"].values,
            self.fulltables["A"].values,
        )

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
        covmat:
            covariance matrix

        """
        diagonal = np.power(unc_df["stat"], 2)
        corr_sys = unc_df["syst"]
        return np.diag(diagonal) + np.outer(corr_sys, corr_sys)
