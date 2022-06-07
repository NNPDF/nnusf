import pathlib

import numpy as np
import pandas as pd


OBS_TYPE = ["F2", "F3", "DXDYNUU", "DXDYNUB"]

MAP_EXP_YADISM = {
    "NUTEV": "XSNUTEVCC",
    "CHORUS": "XSCHORUSCC",
    "CDHSW": "XSCHORUSCC"
}


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
        self, path_to_commondata: pathlib.Path, path_to_theory: pathlib.Path, data_name: str, data_type: str
    ) -> None:
        if data_type not in OBS_TYPE:
            raise ObsTypeError("Observable not implemented or Wrong!")

        self.commondata_path = path_to_commondata
        self.theory_path = path_to_theory
        self.data_type = data_type
        self.data_name = f"{data_name}_{data_type}"

        self.kin_df, self.data_df, unc_df, self.coeff_array = self.load()
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
        info_df = pd.read_csv(f"{self.commondata_path}/info/{exp_name}.csv")

        # kinematic file
        kin_file = self.commondata_path.joinpath(f"kinematics/KIN_{self.data_name}.csv")
        if kin_file.exists():
            kin_df = pd.read_csv(kin_file)[1:].astype(float)
        else:
            kin_df = pd.read_csv(f"{self.commondata_path}/kinematics/KIN_{exp_name}_F2F3.csv")[
                1:
            ].astype(float)
        kin_df["A"] = np.full(kin_df.shape[0], info_df["target"][0])
        kin_df["Obs"] = kin_df.shape[0] * [self.data_type]
        kin_df["projectile"] = np.full(kin_df.shape[0], info_df["projectile"][0])
        if self.data_type in ["DXDYNUU", "DXDYNUB"]:
            data_spec = MAP_EXP_YADISM[self.data_name.split("_")[0]]
        else:
            data_spec = None
        kin_df["xsec"] = kin_df.shape[0] * [data_spec]

        # central values and uncertainties
        data_df = pd.read_csv(
            f"{self.commondata_path}/data/DATA_{self.data_name}.csv", header=0, dtype=float
        )
        unc_df = pd.read_csv(f"{self.commondata_path}/uncertainties/UNC_{self.data_name}.csv")[
            2:
        ].astype(float)

        # coeff_array = np.load(f"{self.theory_path}/coefficients/{self.data_name}.npy")
        coeff_array = None
        return kin_df, data_df, unc_df, coeff_array

    @property
    def kinematics(self) -> tuple:
        """Returns the kinematics variables"""
        return self.kin_df["x"].values, self.kin_df["Q2"].values, self.kin_df["A"].values

    @property
    def central_values(self) -> np.ndarray:
        """Returns the dataset central values"""
        return self.data_df["data"].values

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
        return np.diag(diagonal) + corr_sys @ corr_sys.T
