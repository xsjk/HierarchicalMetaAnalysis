from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Literal, get_args, get_origin, get_type_hints
from argparse import ArgumentParser, Namespace


@dataclass
class Analyzer(ABC):

    # Parameters for prior μ ~ N(η, κ^2)
    eta: float = 0
    kappa: float = 1

    # Τype of prior τ,
    #   if "inv_gamma"  then τ^2 ~ Inv-Gamma(α_τ, β_τ)
    #   if "uniform"    then τ ~ Uniform(0, inf) (inf approximated by τ_max)
    tau_prior_type: Literal["inv_gamma", "uniform"] = "uniform"

    # Parameters for prior τ^2 ~ Inv-Gamma(α_τ, β_τ) (if tau_prior_type == "inv_gamma")
    alpha_tau: float = 1
    beta_tau: float = 1

    # Parameters for prior τ ~ Uniform(0, τ_max) (if tau_prior_type == "uniform")
    tau_max: float = 10

    def __post_init__(self):
        self.eta2 = self.eta**2
        self.kappa2 = self.kappa**2

    def analyze(
        self,
        *,
        data: np.ndarray = None,
        y: np.ndarray = None,
        sigma2: np.ndarray = None,
        calculate_ci: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Analyze the data using the analytical method.

        Parameters
        ----------
        data : np.ndarray, optional
            A 2D numpy array with 4 columns: y0, n0, y1, n1. If None, y and σ² must be provided.
        y : np.ndarray, optional
            Array of y values. If None, it is calculated from the data.
        sigma2 : np.ndarray, optional
            Array of σ² values. If None, it is calculated from the data.
        calculate_ci : bool, optional
            Whether to calculate the credible intervals. Default is True.
        **kwargs
            Additional keyword arguments for the calculation.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the summary statistics.
        """
        if data is not None:
            y, sigma2 = self._extract_y_sigma2(data)

        assert isinstance(y, np.ndarray), "y must be a numpy array."
        assert isinstance(sigma2, np.ndarray), "sigma2 must be a numpy array."
        assert y.ndim == 1, "y must be 1D."
        assert sigma2.ndim == 1, "sigma2 must be 1D."
        assert y.shape == sigma2.shape, "y and sigma2 must have the same shape."

        return self._do_analysis(y, sigma2, calculate_ci, **kwargs)

    @staticmethod
    def _extract_y_sigma2(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(data, np.ndarray), "Data must be a numpy array."
        assert data.ndim == 2, "Data must be 2D."
        assert data.shape[1] == 4, "Data must have 4 columns."

        y0, n0, y1, n1 = data.T
        y = np.log(y0 / n0) - np.log(y1 / n1)
        σ2 = 1 / y0 + 1 / n0 + 1 / y1 + 1 / n1
        return y, σ2

    @abstractmethod
    def _do_analysis(
        self, y: np.ndarray, sigma2: np.ndarray, calculate_ci: bool, **kwargs
    ) -> pd.DataFrame:
        pass

    def __call__(self, *args, **kwds):
        if len(args) == 0:
            return self.analyze(**kwds)
        elif len(args) == 1:
            return self.analyze(data=args[0], **kwds)
        elif len(args) == 2:
            return self.analyze(y=args[0], sigma2=args[1], **kwds)
        else:
            raise ValueError("Invalid number of arguments.")

    @classmethod
    def config_parser(cls, parser: ArgumentParser) -> None:
        for k, v in get_type_hints(cls).items():
            default = getattr(cls, k, None)
            if origin := get_origin(v):
                assert origin is Literal, f"Only Literal types are supported for now, got {v}"
                choices = get_args(v)
                parser.add_argument(f"--{k}", type=type(choices[0]), choices=choices, default=default)
            else:
                parser.add_argument(f"--{k}", type=v, default=default)

    @classmethod
    def extract_kwargs(cls, namespace: Namespace) -> dict:
        return {k: getattr(namespace, k) for k in get_type_hints(cls)}


from .analytical import AnalyticalAnalyzer
from .mcmc import MCMCAnalyzer

__all__ = ["Analyzer", "AnalyticalAnalyzer", "MCMCAnalyzer"]
