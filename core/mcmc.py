import re
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc.math
from pytensor.tensor.variable import TensorVariable

from .base import Analyzer


def get_samples(idata: az.InferenceData):
    post = idata["posterior"]
    samples = {}
    for var in post:
        values = post[var].values
        n_chains, n_draws, *shape = values.shape
        shape = tuple(shape)
        values = values.reshape(n_chains * n_draws, *shape)
        if len(shape) == 0:
            samples[var] = values
        else:
            for idx in np.ndindex(shape):
                name = str(var) + ",".join(map(lambda j: "".join(chr(0x2080 + ord(c) - ord("0")) for c in f"{j[0]:0{len(str(j[1] - 1))}d}"), zip(idx, shape)))
                samples[name] = values[:, *idx]
    return samples


def get_summary(
    samples: dict[str, np.ndarray],
    ci_prob: float | None = None,
) -> pd.DataFrame:
    attrs: dict[str, Callable[[np.ndarray], np.floating]] = {"mean": np.mean, "sd": np.std}
    if ci_prob is not None:
        for var, sample in samples.items():
            samples[var] = np.sort(sample)
        for prob in ((1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2):
            attrs[f"ci_{prob:.1%}"] = (lambda prob: lambda samples: np.percentile(samples, 100 * prob))(prob)

    df = pd.DataFrame({attr: {var: func(samples) for var, samples in samples.items()} for attr, func in attrs.items()})

    df.index = df.index.map(lambda x: re.sub(r"_([0-9])", lambda m: chr(0x2080 + int(m.group(1))), x).replace("^2", "\u00b2"))

    return df.sort_index()


class MCMCAnalyzer(Analyzer):
    def _do_analysis(self, y: np.ndarray, sigma2: np.ndarray, calculate_ci: bool, save_path: str = None, **kwargs) -> pd.DataFrame:
        σ2 = sigma2
        σ = np.sqrt(σ2)
        N = y.shape[0]

        V: dict[str, TensorVariable] = {}

        model = pm.Model()

        with model:
            V["μ"] = pm.Normal("μ", mu=self.eta, sigma=self.kappa)
            if self.tau_prior_type == "uniform":
                V["τ"] = pm.Uniform("τ", lower=0, upper=self.tau_max)
                V["τ^2"] = pm.Deterministic("τ\u00b2", V["τ"] ** 2)
            elif self.tau_prior_type == "sqrt_inv_gamma":
                V["τ^2"] = pm.InverseGamma("τ\u00b2", alpha=self.alpha_tau, beta=self.beta_tau)
                V["τ"] = pm.Deterministic("τ", pymc.math.sqrt(V["τ^2"]))
            elif self.tau_prior_type == "half_cauthy":
                V["τ"] = pm.HalfCauchy("τ", beta=self.gamma_tau)
                V["τ^2"] = pm.Deterministic("τ\u00b2", V["τ"] ** 2)
            else:
                raise ValueError(f"Invalid tau_prior_type '{self.tau_prior_type}', must in ['uniform', 'sqrt_inv_gamma', 'half_cauthy']")

            V["θ"] = pm.Normal("θ", mu=V["μ"], sigma=V["τ"], shape=N)
            V["y"] = pm.Normal("y", mu=V["θ"], sigma=σ, observed=y)

            idata = pm.sample(**kwargs)

        if save_path is not None:
            idata.to_netcdf(save_path)

        samples = get_samples(idata)
        samples["RR"] = np.exp(samples["μ"])
        for j in range(N):
            samples[f"RR{chr(0x2080 + j)}"] = np.exp(samples[f"θ{chr(0x2080 + j)}"])
        summary = get_summary(samples, ci_prob=0.95 if calculate_ci else None)

        return summary.sort_index()


__all__ = ["MCMCAnalyzer"]
