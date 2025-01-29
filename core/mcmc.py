import re
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from argparse import ArgumentParser, Namespace
from pymc.distributions import Distribution
from typing import Callable
from . import Analyzer


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
                name = var + ','.join(map(lambda j: ''.join(chr(0x2080+ord(c)-ord('0')) for c in f'{j[0]:0{len(str(j[1]-1))}d}'), zip(idx, shape)))
                samples[name] = values[:, *idx]
    return samples


def get_summary(
    samples: dict[str, np.ndarray],
    ci_prob: float | None = None,
):

    attrs: dict[str | tuple[str], Callable] = {"mean": np.mean, "sd": np.std}
    if ci_prob is not None:
        for var, sample in samples.items():
            samples[var] = np.sort(sample)
        for prob in ((1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2):
            attrs[f"ci_{prob:.1%}"] = (lambda prob: lambda samples: np.percentile(samples, 100 * prob))(prob)

    data: dict[str, dict[str, float]] = {
        attr: {var: func(samples) for var, samples in samples.items()}
        for attr, func in attrs.items()
    }

    df = pd.DataFrame(data)

    df.index = df.index.map(lambda x: re.sub(
        r"_([0-9])", lambda m: chr(0x2080 + int(m.group(1))), x).replace("^2", "\u00B2"))

    return df.sort_index()


class MCMCAnalyzer(Analyzer):

    def _do_analysis(
        self, y: np.ndarray, sigma2: np.ndarray, calculate_ci: bool, save_path: str = None, **kwargs
    ) -> pd.DataFrame:

        σ2 = sigma2
        σ = np.sqrt(σ2)
        N = y.shape[0]

        V: dict[str, Distribution] = {}

        model = pm.Model()

        with model:

            V["μ"] = pm.Normal("μ", mu=self.eta, sigma=self.kappa)
            if self.tau_prior_type == "uniform":
                V["τ"] = pm.Uniform("τ", lower=0, upper=self.tau_max)
                V["τ^2"] = pm.Deterministic("τ\u00B2", V["τ"] ** 2)
            elif self.tau_prior_type == "inv_gamma":
                V["τ^2"] = pm.InverseGamma("τ\u00B2", alpha=self.alpha_tau, beta=self.beta_tau)
                V["τ"] = pm.Deterministic("τ", pm.math.sqrt(V["τ^2"]))
            elif self.tau_prior_type == "half_cauthy":
                V["τ"] = pm.HalfCauchy("τ", beta=self.gamma_tau)
                V["τ^2"] = pm.Deterministic("τ\u00B2", V["τ"] ** 2)
            else:
                raise ValueError(f"Invalid tau_prior_type '{self.tau_prior_type}', must in ['uniform', 'inv_gamma', 'half_cauthy']")

            V["θ"] = pm.Normal("θ", mu=V["μ"], sigma=V["τ"], shape=N)
            V["y"] = pm.Normal("y", mu=V["θ"], sigma=σ, observed=y)

            idata = pm.sample(**kwargs)

        if save_path is not None:
            idata.to_netcdf(save_path)

        samples = get_samples(idata)
        samples["RR"] = np.exp(samples["μ"])
        for j in range(N):
            samples[f"RR{chr(0x2080+j)}"] = np.exp(samples[f"θ{chr(0x2080+j)}"])
        summary = get_summary(samples, ci_prob=0.95 if calculate_ci else None)

        return summary.sort_index()

    @classmethod
    def config_parser(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--nuts_sampler", choices=["pymc", "nutpie", "numpyro", "blackjax"], default="pymc")
        parser.add_argument("--draws", type=int, default=5000)
        parser.add_argument("--tune", type=int, default=5000)
        parser.add_argument("--chains", type=int, default=1)
        parser.add_argument("--random_seed", type=int, default=None)
        parser.add_argument("--target_accept", type=float, default=0.99)
        parser.add_argument("--save_path", type=str, default=None)
        parser.add_argument("--progressbar", action="store_true")

    @classmethod
    def extract_kwargs(cls, namespace: Namespace) -> dict:
        return {k: getattr(namespace, k) for k in ("draws", "tune", "chains", "random_seed", "target_accept", "save_path", "progressbar", "nuts_sampler")}

__all__ = ["MCMCAnalyzer"]
