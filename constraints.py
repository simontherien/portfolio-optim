import numpy as np
from typing import Callable
from metrics import Metrics


# TODO: beta constraint, treynor, alpha
class Constraints(Metrics):
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
            "weight": self.weight_const,
            "num_assets": self.num_assets_const,
            "concentration": self.concentration_const,
            "expected_return": self.expected_return_const,
            "sharpe": self.sharpe_const,
            "volatility": self.volatility_const,
            "variance": self.moment_const,
        }

    @staticmethod
    def construct_weight_bound(
        size: int, init_bound: tuple, weight_bound: np.ndarray
    ) -> list:
        """
        Construct portfolio weight bounds

        Parameters
        ----------
        size : int
            Number of assets
        init_bound : tuple
            Initial bound (0,1)
        weight_bound : np.ndarray or tuple
            Predetermined constraints on individual weights

        Returns
        -------
        List[tuple]
            List of tuples of lower and upper bounds
        """
        individual_bound = init_bound

        if isinstance(weight_bound[0], (float, int)):
            individual_bound = list(
                zip(np.repeat(weight_bound[0], size), np.repeat(weight_bound[1], size))
            )
        elif isinstance(weight_bound, np.ndarray):
            if weight_bound.ndim == 1:
                individual_bound = list(
                    zip(
                        np.repeat(weight_bound[0], size),
                        np.repeat(weight_bound[1], size),
                    )
                )
            elif weight_bound.ndim == 2:
                individual_bound = list(zip(weight_bound[:, 0], weight_bound[:, 1]))

        return individual_bound

    @staticmethod
    def construct_const_bound(
        bound: float, minimum: bool, opposite_value: float
    ) -> tuple:
        """
        Construct constraint bound based on input

        Parameters
        ----------
        bound : float
            Bound value
        minimum : bool
            Indicate whether passed in bound is upper or lower
        opposite_value : float
            The opposite value of the bound

        Returns
        -------
        tuple
            Constraint bound
        """
        if isinstance(bound, (int, float)):
            print(
                f"Only one bound is given, will set the {'maximum' if minimum else 'minimum'} value to be {opposite_value}"
            )
            bound = (bound, opposite_value) if minimum else (opposite_value, bound)
        return bound

    @staticmethod
    def generate_random_weights(size: int, bound: list, leverage: float) -> np.ndarray:
        """
        Generate random portfolio weights

        Parameters
        ----------
        size : int
            Number of assets
        bound : list
            If bounds are identical for every single asset than use dirichilet distribution to generate random portfolios
            This has the advantage of creating highly concentrated/diversified portfolios that proxy real world portfolio allocations
            Note that bound constraints are perfectly adhered to if leverage=1 and setting an extremely high leverage value may cause violations on bound constraints
            If bounds are not identical then generate with normal distribution Note that randomness deteriorates with more assets
        leverage : float
            Total leverage

        Returns
        -------
        np.ndarray
            Array of random weights
        """
        if all(bound[0][0] == low for low, high in bound) and all(
            bound[0][1] == high for low, high in bound
        ):
            rand_weight = np.random.dirichlet(np.arange(1, size + 1))
            if bound[0][0] < 0:
                neg_idx = np.random.choice(
                    rand_weight.shape[0], np.random.choice(size + 1), replace=False
                )
                rand_weight[neg_idx] = -rand_weight[neg_idx]
                temp = rand_weight * (bound[0][1] - bound[0][0]) / 2 + (
                    bound[0][0] + (bound[0][1] - bound[0][0]) / 2
                )
            else:
                temp = rand_weight * (bound[0][1] - bound[0][0]) + bound[0][0]
        else:
            temp = np.zeros(shape=size)
            for idx, interval in enumerate(bound):
                val = np.random.randn(1)[0]
                std = (interval[1] - interval[0]) / 2
                mu = (interval[1] + interval[0]) / 2
                temp[idx] = val * std + mu

        temp = temp / np.abs(temp).sum() * leverage  # Two Standard Deviation
        return temp

    def create_constraint(self, constraint_type: str, **kwargs) -> Callable:
        """
        Universal method for creating a constraint

        Parameters
        ----------
        constraint_type : str
            Constraint type, options are in Constraints.method_dict
        **kwargs: arguments to be passed to construct constraints

        Returns
        -------
        Callable
            Constraint function
        """
        return self.method_dict[constraint_type](**kwargs)

    # Portfolio composition constraints

    def weight_const(self, weight_bound: np.ndarray, sum: float) -> tuple:
        """
        Construction of individual portfolio weight bound and total leverage

        Parameters
        ----------
        weight_bound : np.ndarray
            Specify each indivual asset's weight bound or weight for all assets
        leverage : float
            The total leverage constraint for the portfolio

        Returns
        -------
        tuple
            Individual bound and total leverage for the portfolio
        """
        init_bound = (0, 1)

        individual_bound = Constraints.construct_weight_bound(
            self.return_vector.shape[0], init_bound, weight_bound
        )

        total_leverage = [{"type": "eq", "fun": lambda w: -self.sum(w) + sum}]

        return individual_bound, total_leverage

    def num_assets_const(self, num_assets: int) -> list:
        """
        Constraint on the number of assets that can be held

        Parameters
        ----------
        num_assets : int
            Maximum number of assets

        Returns
        -------
        list
            Dictionnary to specify the type of constraint and the callable constraint         
        """
        if self.return_vector.shape[0] <= num_assets:
            print(
                "What is going on? The number of assets to hold exceeds the number of assets available, default to a 1 asset only scenario."
            )
            num_assets = self.return_vector.shape[0] - 1
        non_holdings = self.return_vector.shape[0] - num_assets

        return [
            {
                "type": "eq",
                "fun": lambda w: np.sum(
                    np.partition(np.sqrt(np.square(w)), non_holdings)[:non_holdings]
                ),
            }
        ]

    def concentration_const(self, top_holdings: int, top_concentration: float) -> list:
        """
        Constraint on the concentration of the portfolio in the most heavily weighted assets

        Parameters
        ----------
        top_holdings : int
            Number of top holdings to compute concentration
        top_concentration : float
            Maximum % concentration of the top holdings

        Returns
        -------
        list
            Dictionnary to specify the type of constraint and the callable constraint
        """
        if self.return_vector.shape[0] <= top_holdings:
            print(
                "What is going on? Number of top holdings exceeds total available assets. Will default top_holdings to be number of holdings available."
            )
            top_holdings = self.return_vector.shape[0]

        return [
            {
                "type": "ineq",
                "fun": lambda w: np.sum(
                    np.partition(-np.sqrt(np.square(w)), top_holdings)[:top_holdings]
                )
                / np.sum(np.sqrt(np.square(w)))
                + top_concentration,
            }
        ]

    # Risk only constraints

    def volatility_const(self, bound: tuple) -> list:
        """
        Constraint on portfolio vol.

        Parameters
        ----------
        bound : tuple/float
            If passed in tuple, then construct lower bound and upper bound
            Otherwise, assume passed in an upper bound

        Returns
        -------
        list
            Dictionnary to specify the type of constraint and the callable constraint
        """
        bound = Constraints.construct_const_bound(bound, False, 10)

        return [
            {"type": "ineq", "fun": lambda w: self.volatility(w) + bound[0]},
            {"type": "ineq", "fun": lambda w: -self.volatility(w) + bound[1]},
        ]

    # Risk reward constraints

    def expected_return_const(self, bound: tuple) -> list:
        """
        Constraint on portfolio expected return

        Parameters
        ----------
        bound : tuple/float
            If passed in tuple, then construct lower bound and upper bound
            Otherwise, assume passed in an upper bound

        Returns
        -------
        list
            Dictionnary to specify the type of constraint and the callable constraint
        """
        bound = Constraints.construct_const_bound(bound, True, 10)

        return [
            {"type": "ineq", "fun": lambda w: self.expected_return(w) - bound[0]},
            {"type": "ineq", "fun": lambda w: -self.expected_return(w) + bound[1]},
        ]

    def sharpe_const(self, bound: tuple, rfr: float) -> list:
        """
        Constraint on portfolio Sharpe

        Parameters
        ----------
        bound : tuple/float
            If passed in tuple, then construct lower bound and upper bound
            Otherwise, assume passed in an upper bound
        rfr : float
            Risk free rate for Sharpe computation

        Returns
        -------
        list
            Dictionnary to specify the type of constraint and the callable constraint
        """
        bound = Constraints.construct_const_bound(bound, True, 10)

        return [
            {"type": "ineq", "fun": lambda w: self.sharpe(w, rfr) - bound[0]},
            {"type": "ineq", "fun": lambda w: -self.sharpe(w, rfr) + bound[1]},
        ]

    def moment_const(self, bound: tuple) -> list:
        """
        Constraint on portfolio variance

        Parameters
        ----------
        bound : tuple/float
            If passed in tuple, then construct lower bound and upper bound
            Otherwise, assume passed in an upper bound

        Returns
        -------
        list
            Dictionnary to specify the type of constraint and the callable constraint
        """
        bound = Constraints.construct_const_bound(bound, False, 0)
        return [
            {"type": "ineq", "fun": lambda w: self.variance(w) + bound[0]},
            {"type": "ineq", "fun": lambda w: -self.variance(w) + bound[1]},
        ]

