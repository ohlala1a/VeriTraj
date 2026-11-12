"""Run online group-relative training from a YAML configuration."""
import argparse
from veritraj.runtime import load_yaml
from veritraj.train.trainer import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_training(load_yaml(args.config))
