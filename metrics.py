import numpy as np


class Metrics:
    def __init__(
        self,
        return_vector: np.ndarray,
        moment_matrix: np.ndarray,
        assets: list,
        moment: int = 2,
    ):
        """
        Initializes a Metrics instance with parameters to compute portfolio metrics

        Parameters
        ----------
        return_vector : np.ndarray
            Vector of mean returns
        moment_matrix : np.ndarray
            Covariance matrix
        assets : list[str]
            List of asset names
        moment : int, optional
            The order of moment matrix, by default 2
        """
        self.return_vector = return_vector
        self.moment_matrix = moment_matrix
        self.moment = moment
        self.assets = assets

        self.method_dict = {
            "sum": self.sum,
            "num_assets": self.num_assets,
            "concentration": self.concentration,
            "correlation": self.correlation,
            "diversification": self.diversification,
            "volatility": self.volatility,
            "risk_parity": self.risk_parity,
            "expected_return": self.expected_return,
            "sharpe": self.sharpe,
        }

    # Weight related portfolio metrics

    def sum(self, w: np.ndarray) -> float:
        """
        Computes the leverage of the portfolio based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Total leverage of portfolio
        """
        return np.sum(w)

    def num_assets(self, w: np.ndarray) -> int:
        """
        Computes the number of assets in the portfolio based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        int
            Number of assets
        """
        return len(w[np.round(w, 4) != 0])

    def concentration(self, w: np.ndarray, top_holdings: int) -> float:
        """
        Computes the % concentration of the portfolio based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights
        top_holdings : int
            _description_

        Returns
        -------
        float
            The percentage of the portfolio formed by the top_holdings number of assets
        """
        return -np.sum(
            np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings]
        ) / np.sum(np.sqrt(np.square(w)))

    # Risk portfolio metrics

    def correlation(self, w: np.ndarray) -> float:
        """
        Computes the portfolio correlation coefficient based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Correlation coefficient
        """
        corr_matrix = self.moment_matrix * np.dot(
            ((np.diag(self.moment_matrix)) ** -0.5).reshape(-1, 1),
            ((np.diag(self.moment_matrix)) ** -0.5).reshape(1, -1),
        )

        return w @ corr_matrix @ w.T

    def diversification(self, w: np.ndarray) -> float:
        """
        Computes the portfolio diversification based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Ratio of the weighted average of volatilities divided by the portfolio volatility
        """
        std_arr = np.diag(self.moment_matrix) ** 0.5

        return (w @ std_arr) / np.sqrt(w @ self.moment_matrix @ w.T)

    def volatility(self, w: np.ndarray) -> float:
        """
        Computes the portfolio volatility based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Standard deviation of portfolio returns
        """
        return np.sqrt(w @ self.moment_matrix @ w.T)

    def variance(self, w: np.ndarray) -> float:
        """
        Computes the portfolio variance based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Variance of portfolio returns
        """
        return w @ self.moment_matrix @ w.T

    def risk_parity(self, w: np.ndarray) -> float:
        """
        Computes the portfolio risk-parity based on weights

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Risk-parity of portfolio
        """
        return 0.5 * w @ self.moment_matrix @ w.T - np.sum(np.log(w)) / len(self.assets)

    # Risk-reward portfolio metrics

    def expected_return(self, w: np.ndarray) -> float:
        """
        Compute the portfolio expected return based on weights

        Parameters
        ----------
        w : np.ndarray
            

        Returns
        -------
        float
            _description_
        """
        return w @ self.return_vector

    def sharpe(self, w: np.ndarray, rfr: float = 0.04) -> float:
        """
        Calculates the Sharpe ratio of the portfolio
        *If the covariance matrix is semivariance matrix, then it calculates the Sortino ratio

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights
        rfr : float, optional
            Constant risk free rate of return, by default 0.04

        Returns
        -------
        float
            Sharpe ratio of portfolio
        """
        assert rfr > 0, "Risk free rate must be greater than 0."

        return (self.expected_return(w) - rfr) / self.volatility(w)

    # Constant portfolio weights

    def inverse_volatility(self, leverage: float) -> np.ndarray:
        """
        Weights of the portfolio are based on the inverse volatility of the individual assets

        Parameters
        ----------
        leverage : float
            Leverage coefficient

        Returns
        -------
        np.ndarray
            Array of portfolio weights
        """
        std_arr = np.diag(self.moment_matrix) ** 0.5

        return (1 / std_arr) / np.sum(1 / std_arr) * leverage

    def inverse_variance(self, leverage: float) -> np.ndarray:
        """
        Weights of the portfolio are based on the inverse variance of the individual assets

        Parameters
        ----------
        leverage : float
            Leverage coefficient

        Returns
        -------
        np.ndarray
            Array of portfolio weights
        """
        var_arr = np.diag(self.moment_matrix)

        return (1 / var_arr) / np.sum(1 / var_arr) * leverage

    def equal_weight(self, leverage: float) -> np.ndarray:
        """
        Equal weight portfolio

        Parameters
        ----------
        leverage : float
            Leverage coefficient

        Returns
        -------
        np.ndarray
            Array of portfolio weights
        """
        return np.repeat(leverage / len(self.assets), len(self.assets))

