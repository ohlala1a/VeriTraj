"""Wall-clock measurements on trajectory prefixes and observed states."""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

from ..pipeline import parse_executed
from ..records import Trajectory


def benchmark_prefixes(trajectories: list[Trajectory], components, lengths: list[int], repeats: int = 3) -> list[dict]:
    if repeats < 1 or any(length < 1 for length in lengths):
        raise ValueError("Lengths and repeats must be positive")
    output = []
    for length in lengths:
        selected = [trajectory for trajectory in trajectories if len(trajectory.records) >= length]
        if not selected:
            continue
        timings = {"incremental": [], "recompute": []}
        for _ in range(repeats):
            for trajectory in selected:
                final = (trajectory.records[length].observation if length < len(trajectory.records)
                         else trajectory.final_observation)
                prefix = replace(trajectory, records=trajectory.records[:length], final_observation=final)
                for mode in timings:
                    worker = components.worker() if mode == "incremental" else components.recompute_worker()
                    start = time.perf_counter_ns()
                    worker.process(prefix)
                    timings[mode].append((time.perf_counter_ns() - start) / 1e6 / length)
        output.append({"length": length, "trajectories": len(selected), "repeats": repeats,
                       **{f"{mode}_ms_per_record": statistics.median(values) for mode, values in timings.items()}})
    return output


def benchmark_states(trajectories: list[Trajectory], components, repeats: int = 10) -> list[dict]:
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    output = []
    for trajectory in trajectories:
        if not trajectory.records:
            continue
        worker = components.worker()
        worker.reset(trajectory.traj_id, components.goal(trajectory), trajectory.records[0].observation)
        previous = None
        for index, record in enumerate(trajectory.records):
            if index:
                worker.maintain(previous, record.observation)
            parsed = components.interface.parse(record.generation, worker.store)
            start = time.perf_counter_ns()
            for _ in range(repeats):
                worker.verifier.check(worker.store, parsed)
            verify_us = (time.perf_counter_ns() - start) / 1000 / repeats
            cold_ms = cached_us = None
            if components.progress_factory and components.goal(trajectory):
                annotator = components.progress_factory(components.goal(trajectory))
                start = time.perf_counter_ns()
                annotator.h(worker.store)
                cold_ms = (time.perf_counter_ns() - start) / 1e6
                start = time.perf_counter_ns()
                for _ in range(repeats):
                    annotator.h(worker.store)
                cached_us = (time.perf_counter_ns() - start) / 1000 / repeats
            output.append({"traj_id": trajectory.traj_id, "t": record.t, "facts": len(worker.store),
                           "objects": len(worker.store.types), "verify_us": verify_us,
                           "heuristic_cold_ms": cold_ms, "heuristic_cached_us": cached_us})
            previous = parse_executed(components.domain, record.executed)
    return output
