import pandas as pd
from argparse import ArgumentParser
from . import Analyzer, AnalyticalAnalyzer, MCMCAnalyzer

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, default=None, required=True)
parser.add_argument("--calculate_ci", action="store_true")

subparsers = parser.add_subparsers(dest="method", required=True)

Analyzer.config_parser(parser)
AnalyticalAnalyzer.config_parser(subparsers.add_parser("analytical"))
MCMCAnalyzer.config_parser(subparsers.add_parser("mcmc"))

namespace = parser.parse_args()

if namespace.method == "analytical":
    cls = AnalyticalAnalyzer
elif namespace.method == "mcmc":
    cls = MCMCAnalyzer

analyzer = cls(**Analyzer.extract_kwargs(namespace))

summary = analyzer(
    data = pd.read_csv(namespace.data_path, index_col=0).to_numpy(),
    calculate_ci = namespace.calculate_ci,
    **cls.extract_kwargs(namespace)
)

print(summary)
