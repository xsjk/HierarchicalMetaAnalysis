import re
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pymc.distributions import Distribution
from typing import Callable
from . import Analyzer


def get_samples(idata: az.InferenceData):
    post = idata["posterior"]
    return {var: post[var].values.flatten() for var in post}


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

    def analyze(self, data: np.ndarray, calculate_ci: bool = True, save_path: str = None, **kwargs) -> pd.DataFrame:
        '''
        Analyze the data using MCMC.

        Parameters
        ----------
        data : np.ndarray
            A 2D numpy array with 4 columns: y0, n0, y1, n1.
        calculate_ci : bool
            Whether to calculate the credible intervals.
        save_path : str
            The path to save the samples. If None, the samples are not saved.
        **kwargs
            Additional keyword arguments for the MCMC sampler.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the summary statistics.
        '''

        V: dict[str, Distribution] = {}

        N = data.shape[0]
        y0, n0, y1, n1 = data.T

        y = np.log(y0 / n0) - np.log(y1 / n1)
        σ2 = 1 / y0 + 1 / n0 + 1 / y1 + 1 / n1

        σ = np.sqrt(σ2)

        model = pm.Model()

        with model:

            V["μ"] = pm.Normal("μ", mu=self.eta, sigma=self.kappa)
            V["RR"] = pm.Deterministic("RR", pm.math.exp(V["μ"]))
            V["τ^2"] = pm.InverseGamma("τ\u00B2", alpha=self.alpha_tau, beta=self.beta_tau)
            V["τ"] = pm.Deterministic("τ", pm.math.sqrt(V["τ^2"]))

            for j in range(N):
                V[f"θ_{j}"] = pm.Normal(f"θ{chr(0x2080+j)}", mu=V["μ"], sigma=V["τ"])
                V[f"RR_{j}"] = pm.Deterministic(f"RR{chr(0x2080+j)}", pm.math.exp(V[f"θ_{j}"]))

            for j in range(N):
                V[f"y_{j}"] = pm.Normal(f"y{chr(0x2080+j)}", mu=V[f"θ_{j}"], sigma=σ[j], observed=y[j])

            idata = pm.sample(**kwargs)

        if save_path is not None:
            idata.to_netcdf(save_path)

        samples = get_samples(idata)
        summary = get_summary(samples, ci_prob=0.95 if calculate_ci else None)

        return summary.sort_index()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, required=True)
    parser.add_argument("--draws", type=int, default=9000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--target_accept", type=float, default=0.99)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--calculate_ci", action="store_true")
    parser.add_argument("--progressbar", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.data_path, index_col=0).to_numpy()

    analyzer = MCMCAnalyzer(eta=0, kappa=1, alpha_tau=1/1000, beta_tau=1/1000)
    summary = analyzer(data, calculate_ci=args.calculate_ci, save_path=args.save_path, random_seed=args.random_seed, draws=args.draws, tune=args.tune, chains=args.chains, progressbar=args.progressbar, target_accept=args.target_accept)
    print(summary)
