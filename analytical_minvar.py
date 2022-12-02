import numpy as np
import pandas as pd

from returns import Returns
from risks import Risks

class AnalyticalMinVar:
    def __call__(self, df_prices: pd.DataFrame) -> dict:
        """
        Computes weights of assets in minimum variance portfolio

        Parameters
        ----------
        df_prices : pd.DataFrame
            DataFrame of asset prices

        Returns
        -------
        dict
            Dict of ticker and weight pair
        """
        returns_generator = Returns(df_prices)
        df_returns = returns_generator.compute_returns(method="daily")
        mu_return_geom = returns_generator.compute_mean_return(
            method="geometric", time_scaling=20
        )
        cov_generator = Risks(df_returns)
        cov_matrix = cov_generator.compute_cov_matrix(time_scaling=20)

        inverse_cov_df = np.linalg.pinv(cov_matrix)

        n_assets = len(cov_matrix)

        numerator = np.matmul(np.ones(n_assets).T, inverse_cov_df)
        denominator = np.matmul(
            np.ones(n_assets).T, (np.matmul(inverse_cov_df, np.ones(n_assets)))
        )

        return dict(zip(cov_matrix.columns, numerator/denominator))
