from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

@dataclass
class Analyzer(ABC):

    eta: float = 0
    kappa: float = 1
    alpha_tau: float = 1/1000
    beta_tau: float = 1/1000

    def __post_init__(self):
        self.eta2 = self.eta ** 2
        self.kappa2 = self.kappa ** 2

    @abstractmethod
    def analyze(self, data: np.ndarray, calculate_ci: bool = True, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.analyze(*args, **kwargs)

from .analytical import AnalyticalAnalyzer
from .mcmc import MCMCAnalyzer
