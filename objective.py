import numpy as np
from typing import Callable
from metrics import Metrics


class Objective(Metrics):
    def __init__(
        self,
        return_vector: np.ndarray,
        moment_matrix: np.ndarray,
        assets: list,
        moment: int = 2,
    ):
        # Same parameters as Metrics()
        super().__init__(return_vector, moment_matrix, assets, moment)

        self.method_dict = {
            "quadratic_utility": self.efficient_frontier,
            "power_utility": self.power_utility,
            "equal_risk_parity": self.equal_risk_parity,
            "min_correlation": self.min_correlation,
            "min_volatility": self.min_volatility,
            "min_variance": self.min_variance,
            "max_return": self.max_return,
            "max_diversification": self.max_diversification,
            "max_sharpe": self.max_sharpe,
            "inverse_volatility": self.inverse_volatility,
            "inverse_variance": self.inverse_variance,
            "equal_weight": self.equal_weight,
        }

    # Risk related objective functions

    def equal_risk_parity(self, w: np.ndarray) -> float:
        """
        Individual assets contribute equal amounts of risk to the portfolio

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Risk parity global value
        """
        return self.risk_parity(w)

    def min_correlation(self, w: np.ndarray) -> float:
        """
        Minimize portfolio correlation factor

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Correlation factor global value
        """
        return self.correlation(w)

    def min_volatility(self, w: np.ndarray) -> float:
        """
        Minimize portfolio volatility

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Volatility global value
        """
        return self.volatility(w)

    def min_variance(self, w: np.ndarray) -> float:
        """
        Minimize portfolio variance

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Variance global value
        """
        return self.variance(w)

    def max_diversification(self, w: np.ndarray) -> float:
        """
        Maximize portfolio diversification

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Diversification factor global value
        """
        return -self.diversification(w)

    # Risk-reward related objective functions

    def efficient_frontier(self, w: np.ndarray, aversion: float) -> float:
        """
        Maximize return with lowest variance (quadratic utility)

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights
        aversion : float
            Risk aversion parameter

        Returns
        -------
        float
            Quadratic utility global value
        """
        return -(self.expected_return(w) - 0.5 * aversion * self.variance(w))

    def power_utility(self, w: np.ndarray, aversion: float) -> float:
        """
        Maximize power utility of expected return

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights
        aversion : float
            Risk aversion parameter

        Returns
        -------
        float
            Power utility global value
        """
        return -(
            np.log(self.expected_return(w))
            + 0.5 * (1 - aversion) * (self.variance(w) / self.expected_return(w) ** 2)
        )

    def max_return(self, w: np.ndarray) -> float:
        """
        Maximize return regardless of risk

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights

        Returns
        -------
        float
            Return global value
        """
        return -self.expected_return(w)

    def max_sharpe(self, w: np.ndarray, rfr: float) -> float:
        """
        Maximize Sharpe ratio

        Parameters
        ----------
        w : np.ndarray
            Array of portfolio weights
        rfr : float
            Risk free rate of return

        Returns
        -------
        float
            Sharpe ratio global value
        """
        return -self.sharpe(w, rfr)

    def create_objective(self, objective_type: str, **kwargs) -> Callable:
        """
        Function to construct objective function

        Parameters
        ----------
        objective_type : str
            String to specify the type of objective function

        Returns
        -------
        function
            If weight function is not numerical (array of weights) return objective function
        """
        if objective_type in {"equal_weight", "inverse_volatility", "inverse_variance"}:
            return self.method_dict["objective_type"](**kwargs)
        else:
            return self.method_dict[objective_type]
