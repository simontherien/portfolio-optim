import numpy as np
import pandas as pd
from sklearn.covariance import *


class Risks:
    def __init__(self, returns_data: pd.DataFrame):
        """
        Initializes the Risks class instance with asset returns data

        Parameters
        ----------
        returns_data : pd.DataFrame
            Price data of assets
        """
        self.return_matrix = returns_data.values.T
        self.assets = returns_data.columns

    def semi_cov(
        self,
        return_matrix: np.ndarray,
        bm_return: float = 0.0001,
        assume_zero: bool = False,
    ) -> np.ndarray:
        """
        Computes the semi-covariance matrix

        Parameters
        ----------
        return_matrix : np.ndarray
            Array of returns
        bm_return : float, optional
            Ignores all individual asset returns above the bm_return when calculating covariance, by default 0.0001
        assume_zero : bool, optional
            Long term daily mean return for an individual asset is sometimes assumed to be 0, by default False

        Returns
        -------
        np.ndarray
            Matrix of returns to use in semi-covariance estimation
        """
        return_matrix_copy = return_matrix.copy()

        def adjust_return_vector(return_vector: np.ndarray, bm_return: float):
            return_vector[return_vector >= bm_return] = np.mean(
                return_vector[return_vector < bm_return]
            )

            return return_vector

        return_matrix_copy = (
            np.fmin(return_matrix_copy - bm_return, 0)
            if assume_zero
            else np.apply_along_axis(
                adjust_return_vector,
                axis=1,
                arr=return_matrix_copy,
                bm_return=bm_return,
            )
        )

        return return_matrix_copy

    def construct_weights(self, return_matrix: np.ndarray) -> np.ndarray:
        """
        Customized weight array

        Parameters
        ----------
        return_matrix : np.ndarray
            Array of returns

        Returns
        -------
        np.ndarray
            Returns array of weights for each asset (can be not all equal, in progress)
        """

        return np.repeat(
            np.divide(1, return_matrix).shape[1], repeats=return_matrix.shape[1]
        )

    @staticmethod
    def find_cov(
        return_matrix: np.ndarray, weight_factor: np.ndarray, builtin: bool
    ) -> np.ndarray:
        """
        Estimate a covariance matrix, given data and weights

        Parameters
        ----------
        return_matrix : np.ndarray
            Array of returns
        weight_factor : np.ndarray
            Array of observation vector weights. These relative weights are typically large for observations considered “important” and smaller for observations considered less “important”
        builtin : bool
            If True then calls np.cov() to calculate, otherwise use matrix calculation method written in the class

        Returns
        -------
        np.ndarray
            The covariance matrix of the asset returns
        """
        return np.cov(return_matrix, aweights=weight_factor)

    def sample_cov(
        self,
        return_matrix: np.ndarray,
        unit_time: int = 1,
        weights: np.ndarray = None,
        builtin: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Sample covariance computation

        Parameters
        ----------
        return_matrix : np.ndarray
            Array of 
        unit_time : int
            Frequency of returns, by default 1 (daily)
        weights : np.ndarray, optional
            Array of observation weights, by default None
        builtin : bool, optional
            If True then calls np.cov() to calculate covariance, otherwise use matrix calculation method written in the class, by default False

        Returns
        -------
        np.ndarray
            Array of sample covariance values
        """
        weights = self.construct_weights(return_matrix)

        return Risks.find_cov(return_matrix, weights, builtin) * unit_time

    def scikit_cov_technique(
        self,
        return_matrix: np.ndarray,
        technique: str,
        time_scaling: int = 252,
        **kwargs
    ) -> np.ndarray:
        """
        Using sklearn.covariance methods to construct covariance matrix

        Parameters
        ----------
        return_matrix : np.ndarray
            Array of returns
        technique : str
            Options to select sklearn.covariance methods
        time_scaling : int, optional
            Annualize covariance matrix (assuming daily input), by default 252

        Returns
        -------
        np.ndarray
            Returns covariance matrix in it's raw form
        """
        technique_dict = {
            "EmpiricalCovariance": EmpiricalCovariance,
            "EllipticEnvelope": EllipticEnvelope,
            "GraphicalLasso": GraphicalLasso,
            "GraphicalLassoCV": GraphicalLassoCV,
            "LedoitWolf": LedoitWolf,
            "MinCovDet": MinCovDet,
            "OAS": OAS,
            "ShrunkCovariance": ShrunkCovariance,
        }

        try:
            return (
                technique_dict[technique](**kwargs).fit(return_matrix.T).covariance_
                * time_scaling
            )
        except KeyError:
            print(
                "What is going on? Invalid technique! Valid inputs: EmpiricalCovariance, EllipticEnvelope, GraphicalLasso, GraphicalLassoCV, LedoitWolf, MinCovDet, OAS, ShrunkCovariance"
            )

    def compute_cov_matrix(
        self,
        technique: str = "sample",
        semi: bool = False,
        time_scaling: int = 252,
        builtin: bool = False,
        weights: np.ndarray = None,
        bm_return: float = 0.00025,
        assume_zero: bool = False,
        normalize: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculates covariance matrix given the return data input

        Parameters
        ----------
        technique : str, optional
            additional_options: ["EmpiricalCovariance", "EllipticEnvelope", "GraphicalLasso", "GraphicalLassoCV",
                                    "LedoitWolf", "MinCovDet", "OAS", "ShrunkCovariance"]
            Specifies the calculation technique for the covariance matrix, by default "sample"
        semi : bool, optional
            If True, returns a semivariance matrix that emphasizes on downside portfolio variance, by default False
        time_scaling : int, optional
            Default annualizes the covariance matrix (assuming daily return is the input), by default 252
        builtin : bool, optional
            If True then calls np.cov() to calculate, otherwise use matrix calculation method written in the class, by default False
        weights : np.ndarray, optional
            Array of observation vector weights, by default None
        bm_return : float, optional
            additional parameter for calculating semivariance matrix, by default 0.00025
        assume_zero : bool, optional
            Long term daily mean return for an individual asset is sometimes assumed to be 0, by default False
        normalize : bool, optional
            To normalize the covariance matrix. In the specific case for covariance matrix, a normalized covariance
            matrix is a correlation matrix, by default False

        Returns
        -------
        pd.DataFrame
            Returns the covariance matrix as a pandas DataFrame
        """
        return_matrix = self.return_matrix

        if semi:
            return_matrix = self.semi_cov(
                return_matrix, bm_return=bm_return, assume_zero=assume_zero
            )

        if technique == "sample":
            cov_matrix = self.sample_cov(
                return_matrix, time_scaling, builtin=builtin, weights=weights, **kwargs
            )
        else:
            cov_matrix = self.scikit_cov_technique(
                return_matrix, technique, time_scaling, **kwargs
            )

        if normalize:
            cov_mat = cov_mat * np.dot(
                ((np.diag(cov_mat)) ** -0.5).reshape(-1, 1),
                ((np.diag(cov_mat)) ** -0.5).reshape(1, -1),
            )

        return pd.DataFrame(cov_matrix, index=self.assets, columns=self.assets)
