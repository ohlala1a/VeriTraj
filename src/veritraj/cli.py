"""Command-line interface: `veritraj <command> ...` (run `veritraj -h`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _load(path: str):
    from .records import read_trajectories

    return list(read_trajectories(path))


def _verified_map(path: str):
    from .records import read_verified

    out: dict = {}
    for r in read_verified(path):
        out.setdefault(r.traj_id, []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r.t)
    return out


def _table(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    cols = list(rows[0])
    width = {c: max(len(str(c)), *(len(str(r.get(c))) for r in rows)) for c in cols}
    line = "  ".join(str(c).ljust(width[c]) for c in cols)
    sep = "  ".join("-" * width[c] for c in cols)
    body = ["  ".join(str(r.get(c)).ljust(width[c]) for c in cols) for r in rows]
    return "\n".join([line, sep, *body])


# -- commands -------------------------------------------------------------------
def cmd_check_domain(a) -> int:
    from .pddl import check_domain, check_problem, load_domain, load_problem
    from .planner import gbfs_plan

    dom = load_domain(a.domain)
    issues = check_domain(dom)
    print(f"domain {dom.name}: {len(dom.actions)} actions, {len(dom.predicates)} predicates, "
          f"{len(issues)} issues")
    for i in issues:
        print("  -", i)
    bad = bool(issues)
    for p in a.problems:
        prob = load_problem(p)
        pi = check_problem(dom, prob)
        plan = gbfs_plan(dom, prob) if not pi and a.plan else None
        status = "ok" if not pi else f"{len(pi)} issues"
        extra = f", plan length {len(plan)}" if plan else (", no plan found" if a.plan and not pi else "")
        print(f"problem {prob.name}: {status}{extra}")
        for i in pi:
            print("  -", i)
        bad |= bool(pi) or (a.plan and plan is None)
    return 1 if bad else 0


def cmd_plan(a) -> int:
    from .config import build_components
    from .pddl import load_problem
    from .planner import gbfs_plan

    comps = build_components(a.config, progress=False)
    plan = gbfs_plan(comps.domain, load_problem(a.problem))
    if plan is None:
        print("no plan found")
        return 1
    for i, step in enumerate(plan):
        print(f"{i:3d}  {step}    | {comps.interface.render(step)}")
    return 0


def cmd_check_trajectories(a) -> int:
    from .config import build_components
    from .latent import replay

    comps = build_components(a.config, progress=False)
    n_issues = 0
    for traj in _load(a.trajectories):
        rep = replay(comps.domain, comps.problem(traj), comps.interface, traj)
        n_issues += len(rep.issues)
        print(f"{traj.traj_id:28s} records={len(traj):3d} transitions={len(rep.transitions):3d} "
              f"goal={rep.goal_reached!s:5s} issues={len(rep.issues)}")
        for i in rep.issues:
            print("   -", i)
    return 1 if n_issues else 0


def cmd_verify(a) -> int:
    from .ops import verify_corpus
    from .records import write_jsonl

    trajs = _load(a.trajectories)
    verified, stats = verify_corpus(trajs, a.config, policy=a.policy, heuristic=a.heuristic,
                                    progress=not a.no_progress, workers=a.workers, recompute=a.recompute)
    if a.out:
        n = write_jsonl(a.out, (r.to_dict() for t in trajs for r in verified[t.traj_id]))
        print(f"wrote {n} verified records to {a.out}")
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def cmd_show(a) -> int:
    from .config import build_components

    comps = build_components(a.config)
    for traj in _load(a.trajectories):
        if a.traj and traj.traj_id != a.traj:
            continue
        w = comps.worker()
        rows = w.process(traj, comps.goal(traj))
        print(f"== {traj.traj_id}  task={traj.task_id}  success={traj.success}")
        for rec, vr in zip(traj.records, rows):
            mark = "PARSE" if vr.p else ("ok   " if vr.v else "FLAG ")
            h = "inf" if vr.h is None or vr.h == float("inf") else f"{vr.h:.0f}"
            span = rec.generation[vr.char_span[0]:vr.char_span[1]].replace("\n", " ")
            print(f"  t={vr.t:<3d} {mark} {vr.action:42s} h={h:>3s} r_prog={vr.r_prog:+.2f}  span='{span}'")
            if vr.witness:
                print(f"        witness: {{{', '.join(vr.witness)}}}")
        if a.store:
            print("  final store:", json.dumps(w.store.snapshot()["true"]))
    return 0


def cmd_curate(a) -> int:
    from .curation import STRATEGIES, curate
    from .records import write_jsonl

    trajs = _load(a.trajectories)
    verified = _verified_map(a.verified)
    strategies = STRATEGIES if a.strategy == "all" else [a.strategy]
    rows = []
    for s in strategies:
        res = curate(trajs, verified, s, omega=a.omega, window=a.window)
        rows.append(res.summary())
        if a.out:
            write_jsonl(Path(a.out) / f"sft_{s}.jsonl", res.examples)
    print(_table(rows))
    if a.out:
        print(f"\nSFT files written to {a.out}/")
    return 0


def cmd_advantages(a) -> int:
    from .advantage import AdvantageConfig, group_advantages, group_normalize
    from .config import load_config
    from .records import discounted_return

    cfg = AdvantageConfig.from_dict(load_config(a.config).get("advantage"))
    trajs = _load(a.trajectories)
    verified = _verified_map(a.verified)
    groups: dict[str, list] = {}
    for t in trajs:
        if t.group_id:
            groups.setdefault(t.group_id, []).append(t)
    if not groups:
        print("no trajectories with group_id")
        return 1
    for gid, group in groups.items():
        adv = group_advantages(group, verified, cfg)
        a_bar = group_normalize([discounted_return(t, cfg.gamma) for t in group], cfg.eps_norm)
        print(f"== group {gid} (G={len(group)})")
        for t, ab in zip(group, a_bar):
            vals = " ".join(f"{x:+.2f}" for x in adv[t.traj_id])
            print(f"  {t.traj_id:22s} R={discounted_return(t, cfg.gamma):.3f} A_bar={ab:+.2f}  A_t: {vals}")
    return 0


def cmd_eval_labels(a) -> int:
    from .config import build_components
    from .ops import evaluate_labels, verify_corpus

    trajs = _load(a.trajectories)
    comps = build_components(a.config, policy=a.policy, progress=False)
    verified, stats = verify_corpus(trajs, a.config, policy=a.policy, progress=False)
    recomputed, rstats = verify_corpus(trajs, a.config, policy=a.policy, recompute=True)
    out = evaluate_labels(trajs, verified, comps, recomputed)
    out["ms_per_record"] = {"incremental": stats.as_dict()["ms_per_record"],
                            "recompute": rstats.as_dict()["ms_per_record"]}
    print(json.dumps(out, indent=2))
    return 0


def cmd_acquire(a) -> int:
    from .acquisition import FileBuilder, LLMBuilder, LLMChallenger, acquire
    from .config import build_components
    from .ops import acquisition_setup, make_adjudicators

    comps = build_components(a.config, progress=False)
    setup = acquisition_setup(comps, _load(a.trajectories))
    dev, acc, challenger = make_adjudicators(setup)
    llm_ch = None
    if a.drafts:
        builder = FileBuilder(a.drafts)
    else:
        from .llm import OpenAICompatibleClient

        client = OpenAICompatibleClient.from_env()
        docs = Path(a.docs).read_text()
        iface = "\n".join(f"- {n}({', '.join(t for _, t in s.params)})" for n, s in comps.domain.actions.items())
        builder = LLMBuilder(client, docs, iface, setup.transitions)
        if a.llm_challenger:
            llm_ch = LLMChallenger(OpenAICompatibleClient.from_env("VERITRAJ_CHALLENGER"), setup.env, setup.objects)
    print(f"development transitions: {len(setup.transitions)}, dev problems: {len(setup.dev_problems)}, "
          f"acceptance problems: {len(setup.acceptance_problems)}")
    res = acquire(builder, challenger, dev, setup.transitions, acc, a.rounds, llm_ch)
    for r in res.rounds:
        print(f"round {r['round']}: parse={r['parse']} trace={r['trace']}% probes={r['probes']}% "
              f"plan={r['plan']}% execute={r['execute']}% mismatches={r['mismatches']}")
        for f in r["feedback"]:
            print("    ", f)
    print("accepted" if res.accepted else "not accepted")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "report.json").write_text(json.dumps(res.__dict__, indent=2, default=str))
        if res.domain_text:
            (Path(a.out) / "domain.pddl").write_text(res.domain_text)
        print(f"report written to {a.out}/")
    return 0 if res.accepted else 2


def cmd_bench(a) -> int:
    from .ops import bench_length, bench_state

    if a.kind == "length":
        traj = next(t for t in _load(a.trajectories) if not a.traj or t.traj_id == a.traj)
        rows = bench_length(traj, [int(x) for x in a.lengths.split(",")], a.config, a.recompute_max)
        print("per-record cost vs trajectory length (records tiled from", traj.traj_id + ")")
    else:
        rows = bench_state([int(x) for x in a.sizes.split(",")], a.config)
        print("verification / heuristic cost vs state size")
    print(_table(rows))
    return 0


def cmd_make_tasks(a) -> int:
    from .config import build_components
    from .tasks import FAMILIES, generate_suite

    comps = build_components(a.config, progress=False)
    fams = a.families.split(",") if a.families else FAMILIES
    specs = generate_suite(comps.domain, a.out, a.per_family, a.seed, fams)
    print(f"wrote {len(specs)} problems to {a.out}")
    return 0


# -- parser ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="veritraj", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", default=None, help="config YAML (default: configs/alfworld.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check-domain", help="parse and statically check a domain and problems")
    s.add_argument("domain")
    s.add_argument("problems", nargs="*")
    s.add_argument("--plan", action="store_true", help="also check that each problem is solvable")
    s.set_defaults(fn=cmd_check_domain)

    s = sub.add_parser("plan", help="plan for a problem with the configured domain")
    s.add_argument("problem")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("check-trajectories", help="replay logged trajectories against their problems")
    s.add_argument("trajectories")
    s.set_defaults(fn=cmd_check_trajectories)

    s = sub.add_parser("verify", help="build the verified trajectory table")
    s.add_argument("trajectories")
    s.add_argument("--out")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--recompute", action="store_true", help="use full recomputation (reference)")
    s.add_argument("--policy", choices=["conservative", "rule_norm", "llm_impute"])
    s.add_argument("--heuristic", choices=["hff", "hadd", "hmax", "goalcount"])
    s.add_argument("--no-progress", action="store_true")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("show", help="print labels, spans and witnesses per record")
    s.add_argument("trajectories")
    s.add_argument("--traj")
    s.add_argument("--store", action="store_true", help="print the final fact store")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("curate", help="build SFT sets with the curation strategies")
    s.add_argument("trajectories")
    s.add_argument("--verified", required=True)
    s.add_argument("--strategy", default="all")
    s.add_argument("--omega", type=float, default=0.5)
    s.add_argument("--window", type=int, default=6)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_curate)

    s = sub.add_parser("advantages", help="span-level advantages per rollout group")
    s.add_argument("trajectories")
    s.add_argument("--verified", required=True)
    s.set_defaults(fn=cmd_advantages)

    s = sub.add_parser("eval-labels", help="label quality vs latent applicability, IPR, recompute agreement")
    s.add_argument("trajectories")
    s.add_argument("--policy", choices=["conservative", "rule_norm"])
    s.set_defaults(fn=cmd_eval_labels)

    s = sub.add_parser("acquire", help="Builder-Challenger constraint acquisition")
    s.add_argument("trajectories", help="logged trajectories providing development transitions")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--drafts", help="directory with round*.pddl drafts (offline builder)")
    g.add_argument("--llm", action="store_true", help="LLM builder (VERITRAJ_LLM_* env vars)")
    s.add_argument("--llm-challenger", action="store_true", help="add LLM probes (VERITRAJ_CHALLENGER_* env vars)")
    s.add_argument("--docs", default="examples/acquisition/task_documentation.md")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_acquire)

    s = sub.add_parser("bench", help="cost benchmarks")
    s.add_argument("kind", choices=["length", "state"])
    s.add_argument("trajectories", nargs="?", default="examples/trajectories/household_examples.yaml")
    s.add_argument("--traj", default="heat-mug-01/b")
    s.add_argument("--lengths", default="50,200,500,1000,2000")
    s.add_argument("--recompute-max", type=int, default=2000)
    s.add_argument("--sizes", default="25,50,100,200")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("make-tasks", help="generate household PDDL problems")
    s.add_argument("--out", required=True)
    s.add_argument("--per-family", type=int, default=5)
    s.add_argument("--families")
    s.add_argument("--seed", type=int, default=1000)
    s.set_defaults(fn=cmd_make_tasks)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
