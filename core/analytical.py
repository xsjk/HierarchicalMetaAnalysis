import numpy as np
import pandas as pd
import scipy.special
import sympy as sp
from argparse import ArgumentParser, Namespace
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import rv_continuous
from sympy import Expr, IndexedBase, Idx, symbols
from sympy.stats import Normal, GammaInverse
from sympy.stats.crv import ContinuousDistribution
from sympy.stats.rv_interface import density
from typing import Callable
from . import Analyzer


def solve_ub(f, y, lower_bound=0, upper_bound=10, epsrel=1e-6, epsabs=1e-6):
    '''
    solve the equation F(x) = y, where F(x) = \int_{lower_bound}^{x} f(x) dx
    '''
    last_x: float = lower_bound + 1e-6
    integral_value, abserr = quad(f, lower_bound, last_x, epsrel=epsrel, epsabs=epsabs)

    def F(x_new):
        nonlocal last_x
        nonlocal integral_value
        nonlocal abserr
        result, err = quad(f, last_x, x_new, epsrel=epsrel, epsabs=epsabs)
        abserr += err

        integral_value += result
        last_x = x_new
        return integral_value

    return brentq(lambda x: F(x) - y, lower_bound, upper_bound)


def make_rv(pdf=None, cdf=None, epsrel=1e-6, epsabs=1e-6, **kwargs) -> rv_continuous:
    class D(rv_continuous):
        def _pdf(self, x):
            return pdf(x)

        def _ppf(self, y):
            if isinstance(y, np.ndarray):
                return np.array([self._ppf(xi) for xi in y])
            if cdf is None:
                return solve_ub(pdf, y, self.a, self.b, epsrel=epsrel, epsabs=epsabs)
            else:
                return brentq(lambda x: self._cdf(x) - y, self.a, self.b, xtol=epsabs, rtol=epsrel)

        def _cdf(self, x):
            if cdf is None:
                return super()._cdf(x)
            return cdf(x)

    return D(**kwargs)


def normal_cdf(x, μ, σ):
    return scipy.special.ndtr((x - μ) / σ)



class AnalyticalAnalyzer(Analyzer):

    S: dict[str, Expr] = {}

    S["η"], S["μ"], S["α"], S["β"], S["τ"], S["τ^2"], S["κ"] = symbols("η μ α β τ τ^2 κ", real=True)
    S["i"], S["j"], S["N"] = map(Idx, 'ijN')
    S["θ"], S["y"], S["σ"] = symbols("θ y σ", real=True, cls=IndexedBase)

    P: dict[str, Expr]
    RV: dict[str, ContinuousDistribution]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        S = self.S
        P = self.P = {}
        RV = self.RV = {}
        κ2 = self.kappa2
        η = self.eta

        # set prior for τ
        RV["μ"] = Normal("μ", self.eta, self.kappa)
        RV["θ_j|μ,τ"] = Normal("θ_j", S["μ"], S["τ"])
        RV["y_j|σ_j,μ,τ"] = Normal("y_j", S["μ"], sp.sqrt(S["τ^2"] + S["σ"][S["j"]] ** 2))

        P["μ"] = density(RV["μ"])(S["μ"])
        if self.tau_prior_type == "uniform":
            P["τ^2"] = 1 / sp.sqrt(S["τ^2"])
            P["τ"] = 1
        elif self.tau_prior_type == "inv_gamma":
            RV["τ^2"] = GammaInverse(S["τ^2"], self.alpha_tau, self.beta_tau)
            P["τ^2"] = density(RV["τ^2"])(S["τ^2"])
            P["τ"] = density(RV["τ^2"])(S["τ"] ** 2) * 2 * S["τ"]
        else:
            raise ValueError(f"Invalid tau_prior_type '{self.tau_prior_type}', must in ['uniform', 'inv_gamma']")

        P["y|σ,τ"] = (lambda: (
            a := sp.Sum(1 / (S["σ"][S["i"]] ** 2 + S["τ^2"]), (S["i"], 0, S["N"]-1)) + 1 / κ2,
            b := sp.Sum(S["y"][S["i"]] / (S["σ"][S["i"]] ** 2 + S["τ^2"]), (S["i"], 0, S["N"]-1)) + η / κ2,
            c := sp.Sum(S["y"][S["i"]] ** 2 / (S["σ"][S["i"]] ** 2 + S["τ^2"]), (S["i"], 0, S["N"]-1)) + η ** 2 / κ2,
            1 / sp.sqrt(sp.exp((c - b ** 2 / a)) * a * sp.Product(S["σ"][S["i"]] ** 2 + S["τ^2"], (S["i"], 0, S["N"]-1))),
        )[-1])()

        P["y,τ^2|σ"] = P["y|σ,τ"] * P["τ^2"]


    def _do_analysis(self, y: np.ndarray, sigma2: np.ndarray, calculate_ci: bool, **kwargs) -> pd.DataFrame:

        S = self.S
        P = self.P
        RV = self.RV
        κ2 = self.kappa2
        η = self.eta
        η2 = self.eta2

        σ2 = sigma2
        σ = np.sqrt(σ2)

        N = y.shape[0]

        # Analyze the data

        E: dict[str, float] = {}
        Var: dict[str, float] = {}
        PDF: dict[str, Callable] = {}
        CDF: dict[str, Callable] = {}
        CI95: dict[str, tuple[float, float]] = {}

        ## Define the constants as a condition

        y_cond = {S["y"][j]: y[j] for j in range(N)}
        σ_cond = {S["σ"][j]: σ[j] for j in range(N)}
        D = y_cond | σ_cond

        ## Define the PDFs

        ν0 = lambda τ2: sum(1 / (σ2[i] + τ2) for i in range(N)) + 1 / κ2
        ν1 = lambda τ2: sum(y[i] / (σ2[i] + τ2) for i in range(N)) + η / κ2
        ν2 = lambda τ2: sum(y[i] ** 2 / (σ2[i] + τ2) for i in range(N)) + η2 / κ2
        Π = lambda τ2: np.prod(σ2 + τ2)

        PDF["τ^2|y,σ"] = (lambda: (
            f := sp.lambdify([S["τ^2"]], P["y,τ^2|σ"].subs(S["N"], N).doit().subs(D).evalf()),
            s := quad(f, 0, np.inf, **kwargs)[0],
            lambda τ2: f(τ2) / s
        )[-1])()

        PDF["τ|y,σ"] = lambda τ: PDF["τ^2|y,σ"](τ ** 2) * (2 * τ)

        ## Calculate the expectation and variance

        E["τ^2|y,σ"] = quad(lambda τ2: τ2 * PDF["τ^2|y,σ"](τ2), 0, np.inf, **kwargs)[0]
        E["τ|y,σ"] = quad(lambda τ2: np.sqrt(τ2) * PDF["τ^2|y,σ"](τ2), 0, np.inf, **kwargs)[0]
        Var["τ^2|y,σ"] = quad(lambda τ2: τ2 ** 2 * PDF["τ^2|y,σ"](τ2), 0, np.inf, **kwargs)[0] - E["τ^2|y,σ"] ** 2
        Var["τ|y,σ"] = quad(lambda τ2: τ2 * PDF["τ^2|y,σ"](τ2), 0, np.inf, **kwargs)[0] - E["τ|y,σ"] ** 2


        E["μ|y,σ"], Var["μ|y,σ"], E["exp(μ)|y,σ"], Var["exp(μ)|y,σ"] = (lambda: (
            G := lambda τ2: PDF["τ^2|y,σ"](τ2) * ν1(τ2) / ν0(τ2),
            H := lambda τ2: PDF["τ^2|y,σ"](τ2) * ((ν1(τ2) / ν0(τ2)) ** 2 + 1 / ν0(τ2)),
            I := lambda τ2: PDF["τ^2|y,σ"](τ2) * np.exp((ν1(τ2) + 0.5) / ν0(τ2)),
            J := lambda τ2: PDF["τ^2|y,σ"](τ2) * np.exp((2 * ν1(τ2) + 2) / ν0(τ2)),
            G := quad(G, 0, np.inf, **kwargs)[0],
            H := quad(H, 0, np.inf, **kwargs)[0],
            I := quad(I, 0, np.inf, **kwargs)[0],
            J := quad(J, 0, np.inf, **kwargs)[0],
            (G, H - G ** 2, I, J - I ** 2)
        )[-1])()

        for j, r in enumerate(((lambda j: (
            ν0j := lambda τ2: ν0(τ2) - 1 / (σ2[j] + τ2),
            ν1j := lambda τ2: ν1(τ2) - y[j] / (σ2[j] + τ2),
            o1 := lambda τ2: y[j] / σ2[j] + ν1j(τ2) / (τ2 * ν0j(τ2) + 1),
            o0 := lambda τ2: 1 / σ2[j] + ν0j(τ2) / (τ2 * ν0j(τ2) + 1),
            G := lambda τ2: PDF["τ^2|y,σ"](τ2) * o1(τ2) / o0(τ2),
            H := lambda τ2: PDF["τ^2|y,σ"](τ2) * ((o1(τ2) / o0(τ2)) ** 2 + 1 / o0(τ2)),
            I := lambda τ2: PDF["τ^2|y,σ"](τ2) * np.exp((o1(τ2) + 0.5) / o0(τ2)),
            J := lambda τ2: PDF["τ^2|y,σ"](τ2) * np.exp((2 * o1(τ2) + 2) / o0(τ2)),
            G := quad(G, 0, np.inf, **kwargs)[0],
            H := quad(H, 0, np.inf, **kwargs)[0],
            I := quad(I, 0, np.inf, **kwargs)[0],
            J := quad(J, 0, np.inf, **kwargs)[0],
            (G, H - G ** 2, I, J - I ** 2)
        )[-1])(j) for j in range(N))):
            E[f"θ_{j}|y,σ"], Var[f"θ_{j}|y,σ"], E[f"exp(θ_{j})|y,σ"], Var[f"exp(θ_{j})|y,σ"] = r

        summary = pd.DataFrame(index=[f"θ{chr(0x2080+j)}" for j in range(N)] + ["μ", "τ", "τ\u00B2"] + [f"RR{chr(0x2080+j)}" for j in range(N)] + ["RR"])
        summary["mean"] = [E[f"θ_{j}|y,σ"] for j in range(N)] + [E["μ|y,σ"], E["τ|y,σ"], E["τ^2|y,σ"]] + [E[f"exp(θ_{j})|y,σ"] for j in range(N)] + [E["exp(μ)|y,σ"]]
        summary["sd"] = [Var[f"θ_{j}|y,σ"] ** .5 for j in range(N)] + [Var["μ|y,σ"] ** .5, Var["τ|y,σ"] ** .5, Var["τ^2|y,σ"] ** .5] + [Var[f"exp(θ_{j})|y,σ"] ** .5 for j in range(N)] + [Var["exp(μ)|y,σ"] ** .5]

        if calculate_ci:
            ## Define the CDFs
            CDF["μ|y,σ"] = lambda μ: quad(lambda τ2: normal_cdf(μ, ν1(τ2) / ν0(τ2), ν0(τ2) ** -0.5) * PDF["τ^2|y,σ"](τ2), 0, np.inf, **kwargs)[0]

            for j in range(N):
                CDF[f"θ_{j}|y,σ"] = (lambda j:(
                    fτ2 := PDF[f"τ^2|y,σ"],
                    ν1 := lambda τ2: sum(y[i] / (σ2[i] + τ2) for i in range(N) if i != j) + η / κ2,
                    ν0 := lambda τ2: sum(1 / (σ2[i] + τ2) for i in range(N) if i != j) + 1 / κ2,
                    ω1 := lambda τ2: y[j] / σ2[j] + ν1(τ2) / (τ2 * ν0(τ2) + 1),
                    ω0 := lambda τ2: 1 / σ2[j] + ν0(τ2) / (τ2 * ν0(τ2) + 1),
                    lambda μ: quad(lambda τ2: normal_cdf(μ, ω1(τ2) / ω0(τ2), ω0(τ2) ** -0.5) * fτ2(τ2), 0, np.inf, **kwargs)[0]
                )[-1])(j)


            ## Create the random variables
            RV["μ|y,σ"] = make_rv(cdf=CDF["μ|y,σ"], a=-5, b=5, **kwargs)
            RV["τ|y,σ"] = make_rv(pdf=PDF["τ|y,σ"], a=0, b=self.tau_max, **kwargs)
            RV["τ^2|y,σ"] = make_rv(pdf=PDF["τ^2|y,σ"], a=0, b=self.tau_max**2, **kwargs)
            for j in range(N):
                RV[f"θ_{j}|y,σ"] = make_rv(cdf=CDF[f"θ_{j}|y,σ"], a=-5, b=5, **kwargs)

            ## Calculate the 95% credible interval
            var_names = ["μ", "τ", "τ^2"] + [f"θ_{j}" for j in range(N)]
            for v in var_names:
                try:
                    CI95[f"{v}|y,σ"] = RV[f"{v}|y,σ"].ppf([0.025, 0.975])
                except Exception as e:
                    print(f"Failed to calculate 95% credible interval for {v}: {e}")
                    CI95[f"{v}|y,σ"] = (np.nan, np.nan)


            summary["ci_2.5%"] = [CI95[f"θ_{j}|y,σ"][0] for j in range(N)] + [CI95["μ|y,σ"][0], CI95["τ|y,σ"][0], CI95["τ^2|y,σ"][0]] + [np.exp(CI95[f"θ_{j}|y,σ"][0]) for j in range(N)] + [np.exp(CI95["μ|y,σ"][0])]
            summary["ci_97.5%"] = [CI95[f"θ_{j}|y,σ"][1] for j in range(N)] + [CI95["μ|y,σ"][1], CI95["τ|y,σ"][1], CI95["τ^2|y,σ"][1]] + [np.exp(CI95[f"θ_{j}|y,σ"][1]) for j in range(N)] + [np.exp(CI95["μ|y,σ"][1])]

        return summary.sort_index()

    @classmethod
    def config_parser(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--epsabs", type=float, default=1e-3)
        parser.add_argument("--epsrel", type=float, default=1e-2)

    @classmethod
    def extract_kwargs(cls, namespace: Namespace) -> dict:
        return {k: getattr(namespace, k) for k in ("epsabs", "epsrel")}

__all__ = ["AnalyticalAnalyzer"]
