"""Collect, verify, train and evaluate the resulting checkpoint."""

import argparse
from pathlib import Path

from collect_rollouts import main as collect_rollouts
from prepare_data import prepare_data
from veritraj.evaluation.aggregation import aggregate_runs, latex_table
from veritraj.evaluation.online import run_evaluation
from veritraj.runtime import load_yaml, write_json
from veritraj.train.trainer import run_training


def run_pipeline(collection, training, evaluation, output_dir):
    output = Path(output_dir)
    collection_dir = collect_rollouts(collection)
    prepare_data(collection_dir, collection["components"], output / "curated")
    training_dir = run_training(training)
    checkpoints = list(training_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise ValueError(f"Training produced no checkpoint in {training_dir}")
    checkpoint = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
    config = {**evaluation, "model": str(checkpoint.resolve()), "output_dir": str(output / "evaluation")}
    config.setdefault("model_name", training["model"])
    run_evaluation(config)
    rows = aggregate_runs([output / "evaluation" / "metrics.json"])
    (output / "results.tex").write_text(latex_table(rows), encoding="utf-8")
    write_json(output / "results.sources.json", rows)
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_pipeline(load_yaml(args.collection), load_yaml(args.training), load_yaml(args.evaluation), args.out)


if __name__ == "__main__":
    main()
