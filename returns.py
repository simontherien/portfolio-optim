import numpy as np
import pandas as pd


class Returns:
    def __init__(self, price_data: pd.DataFrame):
        """
        Initializes the Returns class instance with asset price data

        Parameters
        ----------
        price_data : pd.DataFrame
            Price data of assets
        """
        self.price_matrix = price_data.values.T
        self.assets = price_data.columns
        self.index = price_data.index

    @staticmethod
    def return_formula(
        price_matrix: np.ndarray,
        index: pd.Index,
        roll: bool = True,
        window: int = 1,
        log: bool = False,
    ) -> tuple:
        """"
        Convert price data to return data

        Parameters
        ----------
        price_matrix : np.ndarray
            Matrix of prices
        index : pd.Index
            Dates time scale
        roll : bool, optional
            Rolling return, by default True
        int : int, optional
            Time interval, by default 1
        log : bool, optional
            Continuous or discrete, by default False

        Returns
        -------
        tuple
            Returns a tuple of (array of returns, dates)
        """
        step = 1 if roll else window
        shift = window

        return (
            (
                np.log(
                    ((price_matrix / np.roll(price_matrix, shift=shift, axis=1)) - 1)
                )[:, shift::step],
                index[shift::step],
            )
            if log
            else (
                ((price_matrix / np.roll(price_matrix, shift=shift, axis=1)) - 1)[
                    :, shift::step
                ],
                index[shift::step],
            )
        )

    def compute_returns(self, method: str, **kwargs) -> pd.DataFrame:
        """
        Calculates asset returns based on defined method and parameters

        Parameters
        ----------
        method: str
            Options = ["daily", "rolling", "collapse"]
            daily: calculates daily percentage change
            rolling: calculates rolling percentage change based on window, user passes in a parameter window=?
            collapse: calculates percentage change based on window, user passes in a parameter window=?
                e.g.: if window=22, output is the return between each 22 day interval from the beginning
            Aditional option: calculates continuous return by passing in log=True, or discrete otherwise
        **kwargs: arguments passed into return_formula()

        Returns
        -------
        pd.DataFrame
            Returns a pandas DataFrame of asset returns
        """
        price_matrix = self.price_matrix
        index = self.index

        if method == "daily":
            return_matrix, return_idx = Returns.return_formula(
                price_matrix, index, window=1, roll=True, **kwargs
            )
        elif method == "rolling":
            return_matrix, return_idx = Returns.return_formula(
                price_matrix, index, roll=True, **kwargs
            )
        elif method == "collapse":
            return_matrix, return_idx = Returns.return_formula(
                price_matrix, index, roll=False, **kwargs
            )
        else:
            print(
                "What is going on? Invalid method! Valid Inputs: daily, rolling, collapse"
            )

        return pd.DataFrame(return_matrix.T, columns=self.assets, index=return_idx)

    def compute_mean_return(
        self, method: str, time_scaling: int = 252, **kwargs
    ) -> pd.Series:
        """
        Calculates mean historical asset returns to be used in mean-variance optimizer

        Parameters
        ----------
        method: str
            Options = ["arithmetic", "geometric"]
            arithmetic: Calculates the arithmetic mean of return, all paramters in compute_returns() can be passed in as additional arguments
            geometric: Calculates the geometric mean from first to last observation

        time_scaling: int, optional
            Annualizes daily mean return, by default 252

        **kwargs: additional arguments if using arithmetic

        Returns
        -------
        pd.Series
            Returns a pandas Series of asset mean returns
        """
        price_matrix = self.price_matrix
        index = self.index

        return_matrix, return_idx = Returns.return_formula(
            price_matrix, index, **kwargs
        )

        if method == "arithmetic":
            mean_return = np.mean(return_matrix, axis=1) * time_scaling
        elif method == "geometric":
            mean_return = (
                (price_matrix[:, -1] / price_matrix[:, 0])
                ** (time_scaling / price_matrix.shape[1])
            ) - 1
        else:
            print(
                "What is going on? Invalid method! Valid Inputs: arithmetic, geometric"
            )

        return pd.Series(mean_return, index=self.assets)
