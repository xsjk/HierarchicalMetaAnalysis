import os
import tomllib
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray
from tqdm import tqdm

from core.analytical import AnalyticalAnalyzer


def run(config: dict):
    seed = config.get("seed", None)
    M = config["n_samples"]

    output_dir = config["output_dir"]

    output_name_no_ext = f"simulation_[seed={seed}]"

    analyzer = AnalyticalAnalyzer.from_config(config["prior"])

    match (c := config["sample"]["mu"])["type"]:
        case "fixed":
            μs = np.broadcast_to(c["value"], shape=(M, 1))
            output_name_no_ext += f"_[mu=fixed({c['value']})]"
        case "normal":
            μs = stats.norm(c["mean"], c["sd"]).rvs(size=(M, 1), random_state=seed)
            output_name_no_ext += f"_[mu=normal({c['mean']}, {c['sd']})]"
        case _:
            raise ValueError(f"Invalid sample type '{c['type']}' for mu")

    match (c := config["sample"]["tau"])["type"]:
        case "fixed":
            τs = np.broadcast_to(c["value"], shape=(M, 1))
            output_name_no_ext += f"_[tau=fixed({c['value']})]"
        case "uniform":
            τs = stats.uniform(0, c["max"]).rvs(size=(M, 1), random_state=seed)
            output_name_no_ext += f"_[tau=uniform(0, {c['max']})]"
        case _:
            raise ValueError(f"Invalid sample type '{c['type']}' for tau")

    match (c := config["sample"]["sigma"])["type"]:
        case "fixed":
            σ = np.array(c["values"])
            σ2 = σ**2
        case _:
            raise ValueError(f"Invalid sample type '{c['type']}' for sigma")

    N = len(σ)

    θs = stats.norm(μs, τs).rvs(size=(M, N), random_state=seed)
    ys = stats.norm(θs, σ).rvs(size=(M, N), random_state=seed)

    global analyze_sample  # Make the function global to be picklable by the ProcessPoolExecutor

    def analyze_sample(sample):
        return analyzer(sample, σ2, calculate_ci=True, **config["analytical"])

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_sample, ys), total=M))

    sample_indices = np.arange(M)

    results = xarray.Dataset({
        "samples": xarray.DataArray(
            np.hstack([τs, τs**2, μs, θs, np.exp(μs), np.exp(θs)]),
            dims=["sample", "variable"],
            coords={"sample": sample_indices, "variable": ["τ", "τ²", "μ"] + [f"θ{chr(0x2080 + j)}" for j in range(N)] + ["RR"] + [f"RR{chr(0x2080 + j)}" for j in range(N)]},
        ),
        "inference": xarray.DataArray(
            results,
            dims=["sample", "variable", "attribute"],
            coords={
                "sample": np.arange(M),
                "variable": results[0].index,
                "attribute": results[0].columns,
            },
        ),
    })

    inference = results["inference"]

    result_analysis = pd.DataFrame(
        {
            "coverage rate": ((results["samples"] > inference.sel(attribute="ci_2.5%")) & (results["samples"] < inference.sel(attribute="ci_97.5%"))).mean(dim="sample"),
            "bias": (inference.sel(attribute="mean") - results["samples"]).mean(dim="sample"),
            "mse": (mse := ((inference.sel(attribute="mean") - results["samples"]) ** 2).mean(dim="sample")),
            "rmse": np.sqrt(mse),
        },
        index=inference.coords["variable"],
    )
    print(result_analysis)
    output_dir = config["output_dir"]

    result_analysis.to_csv(os.path.join(output_dir, f"{output_name_no_ext}.csv"))
    output_file = os.path.join(output_dir, f"{output_name_no_ext}.nc")
    os.makedirs(output_dir, exist_ok=True)
    results.to_netcdf(output_file)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the inference.")
    parser.add_argument("config", type=str, help="Path to the config file")

    args = parser.parse_args()
    config = tomllib.load(open(args.config, "rb"))

    run(config)
