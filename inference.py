import os
import tomllib
from argparse import ArgumentParser

import pandas as pd

from core import AnalyticalAnalyzer, MCMCAnalyzer


def run(config):
    match method := config["method"]:
        case "analytical":
            cls = AnalyticalAnalyzer
        case "mcmc":
            cls = MCMCAnalyzer
        case _:
            raise ValueError(f"Invalid method '{method}', must be in ['analytical', 'mcmc']")

    analyzer = cls.from_config(config["prior"])

    input_file = config["input_file"]
    input_file_name_no_ext = os.path.splitext(os.path.basename(input_file))[0]
    data = pd.read_csv(config["input_file"], index_col=0).to_numpy()
    summary = analyzer(data=data, calculate_ci=config["calculate_ci"], **config[method])

    if config["output_dir"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        output_file = os.path.join(config["output_dir"], f"{input_file_name_no_ext}_inference_summary.csv")
        if os.path.exists(output_file):
            print(f"Warning: {output_file} already exists. Overwriting.")
        summary.to_csv(output_file)
        print(f"Summary saved to {output_file}")
    else:
        print(summary)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run the inference.")
    parser.add_argument("config", type=str, help="Path to the config file")

    args = parser.parse_args()
    config = tomllib.load(open(args.config, "rb"))

    run(config)
