"""Train a policy on action spans exported by the curation command."""

import argparse
import json

from veritraj.runtime import load_yaml
from veritraj.train.sft import run_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_sft(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
