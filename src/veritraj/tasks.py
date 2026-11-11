"""Task generator for ALFWorld-style household problems.

Produces PDDL problems for the six ALFWorld task families (pick, examine,
clean, heat, cool, pick_two) on randomly laid-out kitchens and living rooms.
Every generated problem is checked for solvability with the GBFS planner
before it is written.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .pddl.model import Atom, Domain, Literal, Problem
from .planner import gbfs_plan

FAMILIES = ("pick", "examine", "clean", "heat", "cool", "pick_two")

KITCHEN = {"cabinet": (3, 6), "drawer": (1, 3), "countertop": (1, 2), "fridge": (1, 1),
           "microwave": (1, 1), "sinkbasin": (1, 1), "stoveburner": (1, 2), "diningtable": (0, 1),
           "garbagecan": (1, 1), "coffeemachine": (0, 1), "toaster": (0, 1), "shelf": (0, 2)}
LIVING = {"sidetable": (1, 2), "desk": (0, 1), "dresser": (0, 1), "drawer": (2, 4), "shelf": (1, 3),
          "sofa": (0, 1), "armchair": (0, 1), "coffeetable": (0, 1), "garbagecan": (1, 1),
          "safe": (0, 1), "cabinet": (0, 2), "tvstand": (0, 1), "bed": (0, 1)}
OPENABLE = {"cabinet", "drawer", "fridge", "microwave", "safe"}
CAPS = {"fridge": "can-cool", "microwave": "can-heat", "sinkbasin": "can-clean"}

KITCHEN_ITEMS = ["apple", "bread", "butterknife", "cup", "dishsponge", "egg", "fork", "knife", "ladle",
                 "lettuce", "mug", "pan", "plate", "pot", "potato", "saltshaker", "peppershaker",
                 "spatula", "spoon", "tomato", "winebottle", "bowl", "kettle", "glassbottle", "soapbottle"]
LIVING_ITEMS = ["alarmclock", "book", "box", "cd", "cellphone", "creditcard", "keychain", "laptop",
                "newspaper", "pen", "pencil", "pillow", "remotecontrol", "statue", "vase", "watch",
                "candle", "tissuebox", "mug", "bowl"]

TARGETS = {
    "heat": (["apple", "bread", "cup", "egg", "mug", "plate", "potato", "tomato"],
             ["countertop", "diningtable", "cabinet", "fridge", "garbagecan", "shelf"]),
    "cool": (["apple", "bread", "cup", "egg", "lettuce", "mug", "pan", "plate", "pot", "potato",
              "tomato", "winebottle"], ["countertop", "diningtable", "cabinet", "microwave", "shelf"]),
    "clean": (["apple", "butterknife", "cup", "dishsponge", "egg", "fork", "knife", "ladle", "lettuce",
               "mug", "pan", "plate", "pot", "potato", "spatula", "spoon", "tomato"],
              ["countertop", "diningtable", "cabinet", "drawer", "shelf", "fridge"]),
    "examine": (["alarmclock", "book", "bowl", "cd", "cellphone", "creditcard", "keychain", "laptop",
                 "mug", "pen", "pencil", "pillow", "remotecontrol", "statue", "vase", "watch"], []),
    "pick_two": (["book", "cd", "cellphone", "creditcard", "keychain", "pen", "pencil", "pillow",
                  "remotecontrol", "vase", "watch", "candle"],
                 ["sidetable", "desk", "dresser", "drawer", "shelf", "sofa", "armchair", "coffeetable",
                  "safe", "cabinet", "bed"]),
}


@dataclass
class TaskSpec:
    name: str
    family: str
    text: str
    problem: Problem
    seed: int
    meta: dict = field(default_factory=dict)


def _receptacles(rng: random.Random, layout: dict[str, tuple[int, int]]) -> list[str]:
    out = []
    for cls, (lo, hi) in layout.items():
        for i in range(1, rng.randint(lo, hi) + 1):
            out.append(f"{cls}_{i}")
    return out


def _cls(obj: str) -> str:
    return obj.rsplit("_", 1)[0]


def _mention(obj: str) -> str:
    c, i = obj.rsplit("_", 1)
    return f"{c} {i}"


def generate_task(family: str, seed: int) -> TaskSpec:
    if family not in FAMILIES:
        raise ValueError(f"unknown family '{family}'")
    rng = random.Random(seed)
    kitchen = family in {"heat", "cool", "clean"} or (family == "pick" and rng.random() < 0.5)
    recs = _receptacles(rng, KITCHEN if kitchen else LIVING)
    pool = KITCHEN_ITEMS if kitchen else LIVING_ITEMS
    surfaces = [r for r in recs if _cls(r) not in OPENABLE]
    if family == "examine" and not any(_cls(r) in {"sidetable", "desk", "dresser"} for r in recs):
        recs.append("sidetable_1")
        surfaces.append("sidetable_1")

    objects: dict[str, str] = {"room_1": "room"}
    objects.update({r: "receptacle" for r in recs})
    init: set[Atom] = {Atom("at", ("agent", "room_1")), Atom("handempty", ())}
    for r in recs:
        init.add(Atom("visible", (r,)))
        if _cls(r) in OPENABLE:
            init.add(Atom("openable", (r,)))
            init.add(Atom("closed" if rng.random() < 0.8 else "open", (r,)))
        else:
            init.add(Atom("open", (r,)))
        if _cls(r) in CAPS:
            init.add(Atom(CAPS[_cls(r)], (r,)))

    counts: dict[str, int] = {}

    def new_item(cls: str) -> str:
        counts[cls] = counts.get(cls, 0) + 1
        name = f"{cls}_{counts[cls]}"
        objects[name] = "item"
        return name

    def place(item: str, avoid: set[str]) -> None:
        cands = [r for r in recs if r not in avoid and _cls(r) not in {"garbagecan"} | set(CAPS)]
        init.add(Atom("in", (item, rng.choice(cands or recs))))

    goal: list[Literal] = []
    if family == "pick":
        cls = rng.choice(pool)
        target = rng.choice([r for r in recs if _cls(r) not in {"sinkbasin", "microwave", "garbagecan"}])
        item = new_item(cls)
        place(item, {target})
        goal = [Literal(Atom("in", (item, target)))]
        text = f"put a {cls} in {_mention(target)}."
    elif family == "pick_two":
        cls = rng.choice(TARGETS["pick_two"][0])
        tgt_classes = [r for r in recs if _cls(r) in TARGETS["pick_two"][1]]
        target = rng.choice(tgt_classes or recs)
        a, b = new_item(cls), new_item(cls)
        place(a, {target})
        place(b, {target})
        goal = [Literal(Atom("in", (a, target))), Literal(Atom("in", (b, target)))]
        text = f"put two {cls} in {_mention(target)}."
    elif family == "examine":
        cls = rng.choice(TARGETS["examine"][0])
        lamp_host = rng.choice([r for r in recs if _cls(r) in {"sidetable", "desk", "dresser"}])
        objects["desklamp_1"] = "lamp"
        init.add(Atom("lamp-at", ("desklamp_1", lamp_host)))
        item = new_item(cls)
        place(item, {lamp_host})
        goal = [Literal(Atom("examined", (item,)))]
        text = f"look at {cls} under the desklamp."
    else:
        items, tgt = TARGETS[family]
        cls = rng.choice(items)
        target = rng.choice([r for r in recs if _cls(r) in tgt] or surfaces)
        item = new_item(cls)
        place(item, {target})
        pred = {"heat": "heated", "cool": "cooled", "clean": "cleaned"}[family]
        goal = [Literal(Atom(pred, (item,))), Literal(Atom("in", (item, target)))]
        adj = {"heat": "hot", "cool": "cool", "clean": "clean"}[family]
        text = f"put a {adj} {cls} in {_mention(target)}."

    for _ in range(rng.randint(6, 14)):
        place(new_item(rng.choice(pool)), set())

    name = f"{family}-{seed:05d}"
    problem = Problem(name=name, domain_name="veritraj-alfworld", objects=objects, init=init,
                      goal=goal, comment=f"family: {family}; task: {text}; seed: {seed}")
    return TaskSpec(name, family, text, problem, seed, {"room": "kitchen" if kitchen else "living"})


def problem_to_pddl(spec: TaskSpec) -> str:
    p = spec.problem
    by_type: dict[str, list[str]] = {}
    for o, t in sorted(p.objects.items()):
        by_type.setdefault(t, []).append(o)
    objs = "\n    ".join(f"{' '.join(v)} - {t}" for t, v in sorted(by_type.items()))
    init = "\n    ".join(a.to_pddl() for a in sorted(p.init, key=lambda a: (a.pred, a.args)))
    goal = " ".join(l.to_pddl() for l in p.goal)
    return (f"; family: {spec.family}\n; task: {spec.text}\n; seed: {spec.seed}\n"
            f"(define (problem {p.name})\n  (:domain {p.domain_name})\n"
            f"  (:objects\n    {objs})\n  (:init\n    {init})\n  (:goal (and {goal})))\n")


def task_text(problem: Problem) -> str:
    for part in problem.comment.split(";"):
        part = part.strip()
        if part.startswith("task:"):
            return part[5:].strip()
    return ""


def family_of(problem: Problem) -> str:
    for part in problem.comment.split(";"):
        part = part.strip()
        if part.startswith("family:"):
            return part[7:].strip()
    return problem.name.split("-")[0]


def generate_suite(domain: Domain, out_dir: str | Path, per_family: int, seed0: int = 1000,
                   families=FAMILIES) -> list[TaskSpec]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = []
    for fi, fam in enumerate(families):
        k, seed = 0, seed0 + 1000 * fi
        while k < per_family:
            spec = generate_task(fam, seed)
            seed += 1
            if gbfs_plan(domain, spec.problem, max_expansions=5000) is None:
                continue
            (out / f"{spec.name}.pddl").write_text(problem_to_pddl(spec))
            specs.append(spec)
            k += 1
    return specs


def scaled_problem(n_receptacles: int, items_per_receptacle: int = 3, seed: int = 0) -> Problem:
    """Large single-room problem for state-size benchmarks (not a benchmark task)."""
    rng = random.Random(seed)
    classes = list(KITCHEN) + [c for c in LIVING if c not in KITCHEN]
    objects: dict[str, str] = {"room_1": "room"}
    init: set[Atom] = {Atom("at", ("agent", "room_1")), Atom("handempty", ())}
    counts: dict[str, int] = {}
    recs = []
    for i in range(n_receptacles):
        cls = classes[i % len(classes)]
        counts[cls] = counts.get(cls, 0) + 1
        r = f"{cls}_{counts[cls]}"
        recs.append(r)
        objects[r] = "receptacle"
        init.add(Atom("visible", (r,)))
        if cls in OPENABLE:
            init.add(Atom("openable", (r,)))
            init.add(Atom("closed", (r,)))
        else:
            init.add(Atom("open", (r,)))
        if cls in CAPS:
            init.add(Atom(CAPS[cls], (r,)))
    pool = KITCHEN_ITEMS + LIVING_ITEMS
    for r in recs:
        for _ in range(items_per_receptacle):
            cls = rng.choice(pool)
            counts[cls] = counts.get(cls, 0) + 1
            o = f"{cls}_{counts[cls]}"
            objects[o] = "item"
            init.add(Atom("in", (o, r)))
    target_item = next(o for o, t in objects.items() if t == "item")
    goal = [Literal(Atom("heated", (target_item,))), Literal(Atom("in", (target_item, recs[-1])))]
    return Problem(f"scaled-{n_receptacles}", "veritraj-alfworld", objects, init, goal,
                   comment=f"family: heat; task: scaled benchmark problem; seed: {seed}")
