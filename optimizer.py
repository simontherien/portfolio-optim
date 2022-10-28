import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
import inspect
from objective import Objective
from constraints import Constraints
from metrics import Metrics


class Optimizer:
    def __init__(
        self,
        return_data: pd.Series,
        moment_data: pd.DataFrame,
        asset_names: list = None,
    ):
        """
        Initializes an Optimizer instance with data

        Parameters
        ----------
        return_data : pd.Series
            Mean return data
        moment_data : pd.DataFrame
            Covariance data
        asset_names : list, optional
            List of asset names, by default None
        """
        (
            self.return_vector,
            self.moment_matrix,
            self.assets,
            self.moment,
        ) = Optimizer.init_checker(return_data, moment_data, asset_names)

        self.weight_sols = None

        self.objective = None
        self.objective_sol = None
        self.objective_args = None

        self.objective_creator = Objective(
            self.return_vector, self.moment_matrix, self.assets, self.moment
        )
        self.constraint_creator = Constraints(
            self.return_vector, self.moment_matrix, self.assets, self.moment
        )
        self.metric_creator = Metrics(
            self.return_vector, self.moment_matrix, self.assets, self.moment
        )

        self.bounds, self.constraints = self.constraint_creator.create_constraint(
            "weight", weight_bound=(-1, 1), sum=1
        )
        self.sum=1

    @staticmethod
    def init_checker(
        return_data: pd.Series, moment_data: pd.DataFrame, asset_names: list
    ) -> tuple:
        """
        Dimensionality check when initializing Optimizer

        Parameters
        ----------
        return_data : pd.Series
            Mean return data
        moment_data : pd.DataFrame
            Covariance data
        asset_names : list
            List of asset names

        Returns
        -------
        tuple
            Returns class member data to instantiate Optimizer class
        """
        asset_candidates = None
        if isinstance(return_data, pd.Series):
            return_vec = return_data.values
            asset_candidates = list(return_data.index)
        else:
            print("Return Vector must be a pd.Series!")

        if isinstance(moment_data, pd.DataFrame):
            moment_mat = moment_data.values
            asset_candidates = list(moment_data.index)
        else:
            print("Moment Matrix must be a pd.DataFrame!")

        moment = math.log(moment_mat.shape[1], moment_mat.shape[0]) + 1

        if asset_names:
            assets = asset_names
        elif asset_candidates:
            assets = asset_candidates
        else:
            assets = [f"ASSET_{x}" for x in range(moment_mat.shape[0])]

        if return_vec.shape[0] != moment_mat.shape[0]:
            print("Inconsistent shape between return vector and the moment matrix!")
        elif int(moment) != moment:
            print("Incorrect dimension of the moment matrix!")

        return return_vec, moment_mat, assets, int(moment)

    @staticmethod
    def list_method_options(method_dict):
        """
        List all method options
        """
        return {method: inspect.signature(method_dict[method]) for method in method_dict}

    def objective_options(self):
        return Optimizer.list_method_options(self.objective_creator.method_dict)

    def constraint_options(self):
        return Optimizer.list_method_options(self.constraint_creator.method_dict)

    def metric_options(self):
        return Optimizer.list_method_options(self.metric_creator.method_dict)



    def add_objective(self, objective_type: str, **kwargs) -> None:
        """
        Add objective to portfolio problem
        Call objective_options() to check all available options
        Can input a customized objective by setting objective_type="custom".
        The custom_func should follow the parameter structure of custom_func(w, **kwargs)
        custom_func cannot call the moment matrix/return matrix that are passed into Metrics

        Parameters
        ----------
        objective_type : str
            Objective name
        """
        if objective_type != "custom":
            self.objective_args = tuple(kwargs.values())
            self.objective = self.objective_creator.create_objective(
                objective_type, **kwargs
            )
        else:
            self.objective_args = tuple(kwargs.values())[1:]
            self.objective = tuple(kwargs.values())[0]

    def add_constraint(self, constraint_type: str, **kwargs) -> None:
        """
        Add constraint to portfolio problem
        Call constraint_options() to check all available options
        Can input a customized constraint by setting constraint_type="custom".
        The custom_func should follow the parameter structure of custom_func(w, **kwargs)
        custom_func cannot call the moment matrix/return matrix that are passed into Metrics

        Parameters
        ----------
        constraint_type : str
            Constraint name
        """
        if constraint_type == "custom":
            self.constraints += tuple(kwargs.values())[0]
        elif constraint_type == "weight":
            bound, sum = self.constraint_creator.create_constraint(
                constraint_type, **kwargs
            )
            self.bounds = bound
            self.sum = kwargs["sum"]
            self.constraints[0] = sum[0]  # total leverage is always first constraint
        else:
            self.constraints += self.constraint_creator.create_constraint(
                constraint_type, **kwargs
            )

    def clear_problem(
        self, clear_objective: bool = True, clear_constraints: bool = True
    ) -> None:
        """
        Clear protfolio problem

        Parameters
        ----------
        clear_objective : bool, optional
            Clear the objective functions, by default True
        clear_constraints : bool, optional
            Clear the constraints
            Weight and sum will be defaulted to (-1,1) and sum of 1 after clearance, by default True
        """
        if clear_constraints:
            self.constraints = []
            self.bounds, self.constraints = self.constraint_creator.create_constraint(
                "weight", weight_bound=(-1, 1), sum=1
            )
        if clear_objective:
            self.objective = None

    def solve(self, x0: np.ndarray = None, round_digit: int = 4, **kwargs) -> None:
        """
        Solve the portfolio problem

        Parameters
        ----------
        x0 : np.ndarray, optional
            can pass in an initial guess to avoid SciPy from running into local minima, by default None
        round_digit : int, optional
            Round weight, by default 4
        """
        if type(self.objective) != np.ndarray:
            result = minimize(
                self.objective,
                x0=Constraints.generate_random_weights(
                    self.return_vector.shape[0], self.bounds, self.sum
                )
                if x0 is None
                else x0,
                options={"maxiter": 10000},
                constraints=self.constraints,
                bounds=self.bounds,
                args=self.objective_args,
            )
            if not result.success:
                self.clear_problem(**kwargs)
                print(
                    f"Optimization has failed. Error message: {result.message}. Please adjust constraints/objectives or input an initial guess!"
                )

            self.clear_problem(**kwargs)
            self.weight_sols = np.round(result.x, round_digit) + 0
        else:
            print(
                "The problem formulated is not an optimization problem (constant) and is calculated numerically!"
            )

            self.weight_sols = np.round(self.objective, round_digit) + 0
            self.clear(**kwargs)

    def summary(
        self, rfr: float = None, top_holdings: int = None, round_digit: int = 4
    ) -> tuple:
        """
        Summary of solved portfolio problem

        Parameters
        ----------
        rfr : float, optional
            Pass in a float to compute Sharpe, by default None
        top_holdings : int, optional
            Pass in an int to compute portfolio concentration, by default None
        round_digit : int, optional
            Round the metrics, by default 4

        Returns
        -------
        tuple
            Tuple of dicts : weights and portfolio metrics
        """
        weight_dict = dict(zip(self.assets, self.weight_sols))
        metric_dict = {
            "Expected return": self.metric_creator.expected_return(self.weight_sols),
            "Sum": self.metric_creator.sum(self.weight_sols),
            "Num. holdings": self.metric_creator.num_assets(self.weight_sols),
        }

        if top_holdings is not None:
            metric_dict[
                f"Top {top_holdings} Holdings concentrations"
            ] = self.metric_creator.concentration(self.weight_sols, top_holdings)

        if self.moment == 2:
            metric_dict["Volatility"] = self.metric_creator.volatility(self.weight_sols)
            metric_dict["Correlation"] = self.metric_creator.correlation(
                self.weight_sols
            )

        if rfr is not None:
            metric_dict["Sharpe Ratio"] = self.metric_creator.sharpe(
                self.weight_sols, rfr
            )

        for item in metric_dict:
            metric_dict[item] = np.round(metric_dict[item], round_digit)

        weight_dict = {k: v for k, v in weight_dict.items() if v}

        return weight_dict, metric_dict
