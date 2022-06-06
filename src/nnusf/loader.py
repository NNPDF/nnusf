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

    """

    def __init__(self, path_to_commondata: pathlib.Path, data_name: str) -> None:
        self.path = path_to_commondata.joinpath("commondata")
        self.data_name = data_name

        self.kin_df, self.data_df, unc_df = self.load()
        self.covariance_matrix = self.build_covariance_matrix(unc_df)

    def load(self) -> tuple(pd.DataFrame):
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
        info_df = pd.read_csv(f"{self.path}/info/{self.data_name}.csv")
        data_type = info_df["type"]

        # kinematic file
        kin_file = self.path.joinpath(
            f"kinematics/KIN_{self.data_name}_{data_type}.csv"
        )
        if kin_file.exist():
            kin_df = pd.read_csv(
                f"{self.path}/kinematics/KIN_{self.data_name}_{data_type}.csv"
            )
        else:
            kin_df = pd.read_csv(
                f"{self.path}/kinematics/KIN_{self.data_name}_F2F3.csv"
            )
        kin_df["A"] = info_df["target"]

        # central values and uncertainties
        data_df = pd.read_csv(f"{self.path}/data/DATA_{self.data_name}_{data_type}.csv")
        unc_df = pd.read_csv(
            f"{self.path}/uncertainties/UNC_{self.data_name}_{data_type}.csv"
        )

        return kin_df, data_df, unc_df

    @property
    def kinematics(self) -> tuple(np.ndarray):
        """Returns the kinematics variables"""
        return self.kin_df["x"], self.kin_df["q2"], self.kin_df["A"]

    @property
    def central_values(self) -> np.ndarray:
        """Returns the dataset central values"""
        return self.data_df["data"]

    @property
    def covmat(self) -> np.ndarray:
        """Returns the covariance matrix"""
        return self.covariance_matrix()

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
        diagonal = unc_df["stat"] ** 2
        corr_sys = unc_df["syst"]
        return np.diag(diagonal) + corr_sys @ corr_sys.T
