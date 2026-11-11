"""Shared fixtures: components built from the default config and the example trajectories."""

from pathlib import Path

import pytest

from veritraj.config import build_components
from veritraj.records import read_trajectories

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "trajectories" / "household_examples.yaml"


@pytest.fixture(scope="session")
def comps():
    return build_components(ROOT / "configs" / "alfworld.yaml")


@pytest.fixture(scope="session")
def trajs():
    return {t.traj_id: t for t in read_trajectories(EXAMPLES)}
