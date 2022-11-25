# -*- coding: utf-8 -*-
"""Provide a Loader class to retrieve data information."""
import logging
import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    ObsTypeError,
    add_w2_table,
    append_target_info,
    apply_cuts,
    combine_tables,
    parse_central_values,
    parse_input_kinematics,
    parse_uncertainties,
)

_logger = logging.getLogger(__name__)

OBS_TYPE = ["F2", "F3", "FW", "DXDYNUU", "DXDYNUB", "QBAR"]


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
        kincuts: dict = {},
        path_to_coefficients: Optional[pathlib.Path] = None,
        include_syst: Optional[bool] = True,
        verbose: bool = True,
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
        # In case of post-fit related operations, we do not need to
        # print out the noisy log outputs
        if not verbose:
            _logger.info("Loading the datasets...")
            _logger.setLevel(logging.ERROR)

        self.name = name
        if self.obs not in OBS_TYPE:
            msg = f"Observable '{self.obs}' not implemented or Wrong!"
            raise ObsTypeError(msg)

        self.commondata_path = path_to_commondata
        self.coefficients_path = path_to_coefficients
        self.table, self.leftindex = self._load(kincuts)
        self.tr_frac = None
        self.covariance_matrix = self.build_covariance_matrix(
            self.commondata_path,
            self.table,
            self.name,
            include_syst,
            self.leftindex,
        )
        _logger.info(f"'[{name:<25}]' loaded successfully")

    def _load(self, kincuts: dict) -> tuple[pd.DataFrame, pd.Index]:
        """Load the dataset information.

        Returns
        -------
        table with loaded data

        """
        # Extract the information from the INFO files
        exp_name = self.name.split("_")[0]
        if "_MATCHING" in exp_name:
            exp_name = exp_name.strip("_MATCHING")

        info_name = exp_name if "PROTONBC" not in self.name else "PROTONBC"
        info_df = pd.read_csv(f"{self.commondata_path}/info/{info_name}.csv")

        # Extrac the values from the kinematics
        kin_df = parse_input_kinematics(
            mainpath=self.commondata_path,
            name=self.name,
            info_name=info_name,
            exp_name=exp_name,
            obs=self.obs,
        )

        # Extract values from the central data
        data_df = parse_central_values(self.commondata_path, self.name)

        # Extract values from the uncertainties
        unc_df = parse_uncertainties(self.commondata_path, self.name)

        # Add a column to `kin_df` that stores the W2 values
        kin_df = add_w2_table(kin_df, m_nucleus=info_df["m_nucleon"][0])

        # Combine the tables and restore the indices
        new_df = combine_tables(kin_df, data_df, unc_df, self.name)

        # Apply kinematic cuts on the (pseudo-)datasets
        new_df = apply_cuts(new_df, self.name, kincuts=kincuts)

        # Append the info regarding the nucleon/nucleus to the table
        new_df = append_target_info(new_df, info_df, exp_name, self.obs)

        _logger.info(
            f"'[{self.name:<25}]' Q2min={new_df['Q2'].min():.3f}"
            f", Q2max={new_df['Q2'].max():.3f}"
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
        """Return the kinematic variables."""
        return self.table[["x", "Q2", "A"]].values

    @kinematics.setter
    def kinematics(self, new_kinematics):
        """Replace the kinematic variables."""
        self.table[["x", "Q2", "A"]] = new_kinematics

    @property
    def n_data(self):
        """Return the number of datapoints."""
        return self.table.shape[0]

    @property
    def central_values(self) -> np.ndarray:
        """Return the dataset central values."""
        if self.name.endswith("_MATCHING"):
            cholesky = np.linalg.cholesky(self.covariance_matrix)
            np_rng_state = np.random.get_state()
            np.random.seed(pow(self.table.shape[0], 3))
            random_samples = np.random.randn(self.table.shape[0])
            np.random.set_state(np_rng_state)
            shift_data = cholesky @ random_samples
            return self.table["data"].values + shift_data
        else:
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
        commondata_path: pathlib.Path,
        unc_df: pd.DataFrame,
        dataset_name: str,
        include_syst: Union[bool, None],
        mask_predictions: Union[pd.Index, None],
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
        if "_MATCHING" in dataset_name:
            sv_variations = []
            for variation in pathlib.Path(
                f"{commondata_path}/matching/"
            ).iterdir():
                if dataset_name in variation.stem:
                    # central scale
                    if "xif1_xir1" in variation.stem:
                        nrep_predictions = np.load(variation)
                    else:
                        sv_variations.append(np.load(variation))
            # build th shift
            th_shift = (sv_variations - nrep_predictions[:, 0]).T
            # build covaraince
            pdf_covmat = np.cov(nrep_predictions[mask_predictions])
            th_covamt = np.cov(th_shift[mask_predictions])
            covmat = np.sqrt(th_covamt**2 + pdf_covmat**2)
            return clip_covmat(covmat, dataset_name)
        else:
            diagonal = np.power(unc_df["stat"], 2)
            if include_syst:
                corr_sys = unc_df["syst"]
                return np.diag(diagonal) + np.outer(corr_sys, corr_sys)
            return np.diag(diagonal)


def clip_covmat(covmat, dataset_name):
    """Given a covariance matrix, performs a regularization by cutting
    negative values.
    """
    # eigh gives eigenvals in ascending order
    e_val, e_vec = np.linalg.eigh(covmat)
    # if eigenvalues are close to zero, can be negative
    if e_val[0] < 0:
        _logger.warning(
            f"'[{dataset_name:<25}]' Negative eigenvalue encountered."
            " Clipping values to 1e-5."
        )
    else:
        return covmat
    # set negative eigenvalues to 1e-5
    new_e_val = np.clip(e_val, a_min=1e-5, a_max=None)
    return (e_vec * new_e_val) @ e_vec.T
