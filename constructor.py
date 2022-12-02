import pandas as pd

from returns import Returns
from risks import Risks
from optimizer import Optimizer


class Constructor:
    def __call__(
        self,
        df_prices: pd.DataFrame,
        rfr: float,
        freq: str = "monthly",
        objective: str = "quadratic_utility",
        constraints: list = [],
        objective_kwargs: dict = {},
        constraint_kwargs: list = [],
    ) -> tuple:
        """
        Global function to solve portfolio problem

        Parameters
        ----------
        df_prices : pd.DataFrame
            DataFrame containing asset prices
        rfr : float
            Risk free rate of return
        freq : str, optional
            Frequency of portfolio rebalancing, by default "monthly"
        objective : str, optional
            Objective of investor, by default "quadratic_utility"
        constraint : list, optional
            Constraints of investor

        Returns
        -------
        tuple
            Tuple of dicts: weights and portfolio metrics
        """
        # Convert to same time scale
        if freq == "monthly":
            t = 12
            ts = 20
        elif freq == "yearly":
            t = 1
            ts = 252
        elif freq == "daily":
            t = 252
            ts = 1

        # Input data
        returns_generator = Returns(df_prices)
        df_returns = returns_generator.compute_returns(method="daily")
        mu_return_geom = returns_generator.compute_mean_return(
            method="geometric", time_scaling=ts
        )
        cov_generator = Risks(df_returns)
        cov_matrix = cov_generator.compute_cov_matrix(time_scaling=ts)

        # Construct optimizer
        portfolio_problem = Optimizer(
            mu_return_geom, cov_matrix, asset_names=list(mu_return_geom.index)
        )

        portfolio_problem.clear_problem()
        portfolio_problem.add_objective(objective, **objective_kwargs)
        for c, kwarg in zip(constraints, constraint_kwargs):
            portfolio_problem.add_constraint(c, **kwarg)

        # Solve problem
        portfolio_problem.solve()

        return portfolio_problem.summary(rfr=rfr)
