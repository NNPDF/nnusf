import imp
import pathlib

import numpy as np
import pandas as pd


class Loader:
    """Load a dataset given the name and build the covariance matrix

    Parameters
    ----------
    path_to_commondata:
        path to commondata folder
    data_name:
        dataset name
    data_type: str
        data type: F2,F3, DXDYNUU, DXDYNUB

    """

    def __init__(
        self, path_to_commondata: pathlib.Path, data_name: str, data_type: str
    ) -> None:
        self.path = path_to_commondata.joinpath("commondata")
        self.data_name = f"{data_name}_{data_type}"

        self.kin_df, self.data_df, unc_df = self.load()
        self.covariance_matrix = self.build_covariance_matrix(unc_df)

    def load(self) -> tuple:
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
        info_df = pd.read_csv(f"{self.path}/info/{exp_name}.csv")

        # kinematic file
        kin_file = self.path.joinpath(f"kinematics/KIN_{self.data_name}.csv")
        if kin_file.exists():
            kin_df = pd.read_csv(kin_file)[1:].astype(float)
        else:
            kin_df = pd.read_csv(f"{self.path}/kinematics/KIN_{exp_name}_F2F3.csv")[
                1:
            ].astype(float)
        kin_df["A"] = np.full(kin_df.shape[0], info_df["target"][0])

        # central values and uncertainties
        data_df = pd.read_csv(
            f"{self.path}/data/DATA_{self.data_name}.csv", header=0, dtype=float
        )
        unc_df = pd.read_csv(f"{self.path}/uncertainties/UNC_{self.data_name}.csv")[
            2:
        ].astype(float)
        return kin_df, data_df, unc_df

    @property
    def kinematics(self) -> tuple:
        """Returns the kinematics variables"""
        return self.kin_df["x"], self.kin_df["Q2"], self.kin_df["A"]

    @property
    def central_values(self) -> np.ndarray:
        """Returns the dataset central values"""
        return self.data_df["data"]

    @property
    def covmat(self) -> np.ndarray:
        """Returns the covariance matrix"""
        return self.covariance_matrix

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
        return np.diag(diagonal) + corr_sys @ corr_sys.T
