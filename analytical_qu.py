import pandas as pd
import numpy as np

from returns import Returns
from risks import Risks

class AnalyticalQU:
    def __call__(self, df_prices:pd.DataFrame) -> dict:
        returns_generator = Returns(df_prices)
        df_returns = returns_generator.compute_returns(method="daily")
        mu_return_geom = returns_generator.compute_mean_return(
            method="geometric", time_scaling=20
        )
        cov_generator = Risks(df_returns)
        cov_matrix = cov_generator.compute_cov_matrix(time_scaling=20)


        inverse_cov_df = np.linalg.pinv(cov_matrix)

        n_assets = len(cov_matrix)

        aversion = 3
        numerator_mvp = np.matmul(np.ones(n_assets).T, inverse_cov_df)
        denominator_mvp = np.matmul(
            np.ones(n_assets).T, (np.matmul(inverse_cov_df, np.ones(n_assets)))
        )
        mvp_w = numerator_mvp / denominator_mvp
        numerator_tangency = np.matmul(mu_return_geom.T, inverse_cov_df)
        denominator_tangency = np.matmul(
            np.ones(n_assets).T, (np.matmul(inverse_cov_df, mu_return_geom))
        )
        tangency_w = numerator_tangency / denominator_tangency
        quadratic_w = mvp_w + (1 / aversion) * denominator_tangency * (tangency_w - mvp_w)

        return dict(zip(cov_matrix.columns, quadratic_w))