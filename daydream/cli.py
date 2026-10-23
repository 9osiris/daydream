"""cli: run agents, solve maps, benchmark planners, generate maps."""

import argparse
import sys
import time

from . import maps
from .env import GridEnv, NAMES
from .world import WorldModel
from .learn import TabularModel, TransitionLog
from .agent import run_agent
from . import planners

PLANNERS = {
    "bfs": planners.bfs,
    "greedy": planners.greedy,
    "mcts": planners.mcts,
}


def _planner_kwargs(args):
    kw = {}
    if args.planner == "mcts":
        kw["iterations"] = args.iterations
        kw["seed"] = args.seed
    elif args.planner == "bfs":
        kw["max_nodes"] = args.max_nodes
    return kw


def _load(args):
    env = GridEnv(maps.get(args.map))
    if getattr(args, "model", "hand") == "learned":
        if not args.model_file:
            raise SystemExit("--model learned needs --model-file PATH")
        fallback = None if args.no_fallback else WorldModel(env)
        model = TabularModel.load(args.model_file, fallback=fallback)
    else:
        model = WorldModel(env, fake_walls=args.fake_wall, fake_open=args.fake_open)
    return env, model


def cmd_run(args):
    env, model = _load(args)
    planner = PLANNERS[args.planner]
    log = TransitionLog() if args.record else None
    t0 = time.time()
    res = run_agent(env, model, planner, max_steps=args.max_steps, mpc=args.mpc,
                    replan=not args.no_replan, learn=not args.no_learn,
                    on_step=(lambda s, a, t: log.record(s, a, t.next_state, t.reward, t.done)
                             if log is not None else None),
                    **_planner_kwargs(args))
    dt = time.time() - t0
    state = env.start_state()
    if args.trace:
        print(env.render(state))
        print()
    for entry in res.trace:
        kind = entry[0]
        if kind == "plan":
            print("plan: " + entry[1])
        elif kind == "replan":
            print("replan: " + entry[1])
        elif kind == "learn":
            print("learn: " + entry[1])
        elif kind == "stuck":
            print("stuck: " + entry[1])
        elif kind == "step" and args.trace:
            _, name, imagined, real, r_img, r_real = entry
            mark = "" if imagined == real else "  <- surprise"
            print("  %-5s imagined=%s got=%s%s" % (name, imagined[:2], real[:2], mark))
            print(env.render(real))
            print()
    print(res.summary())
    print("wall time %.2fs" % dt)
    if log is not None:
        log.save(args.record)
        print("recorded %d transitions to %s" % (len(log), args.record))
    return 0 if res.succeeded else 1


def cmd_solve(args):
    env, model = _load(args)
    plan = PLANNERS[args.planner](model, env.start_state(), **_planner_kwargs(args))
    print(plan.describe())
    return 0 if plan.actions else 1


def cmd_learn(args):
    model = TabularModel.load(args.log)
    stats = model.stats()
    print("transitions: %d" % stats["transitions"])
    print("states:      %d" % stats["states"])
    if args.out:
        model.save(args.out)
        print("wrote deduped model to %s" % args.out)
    return 0


def cmd_bench(args):
    print("%-8s %-6s %-6s %-10s %-10s %-8s %-8s" % (
        "planner", "wins", "runs", "avg_steps", "avg_reward", "surpris", "time_s"))
    for name in PLANNERS:
        wins = 0
        steps = 0
        reward = 0.0
        surprises = 0
        t0 = time.time()
        kw = {}
        if name == "mcts":
            kw = {"iterations": args.iterations, "seed": args.seed}
        elif name == "bfs":
            kw = {"max_nodes": args.max_nodes}
        for i in range(args.seeds):
            env = GridEnv(maps.get(args.map))
            if args.model == "learned":
                if not args.model_file:
                    raise SystemExit("--model learned needs --model-file PATH")
                fallback = None if args.no_fallback else WorldModel(env)
                model = TabularModel.load(args.model_file, fallback=fallback)
            else:
                model = WorldModel(env)
            res = run_agent(env, model, PLANNERS[name], max_steps=args.max_steps, **kw)
            wins += 1 if res.succeeded else 0
            steps += res.steps
            reward += res.reward
            surprises += res.surprises
        dt = time.time() - t0
        n = args.seeds
        print("%-8s %-6d %-6d %-10.1f %-10.1f %-8.1f %-8.2f" % (
            name, wins, n, steps / n, reward / n, surprises / n, dt))
    return 0


def cmd_gen(args):
    print(maps.generate(width=args.width, height=args.height, seed=args.seed))
    return 0


def cmd_maps(args):
    for name in maps.BUNDLED:
        print(name)
    return 0


def _pos(text):
    # "3,1" on the cli -> (3, 1)
    x, y = text.split(",")
    return (int(x), int(y))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="daydream",
                                 description="world models for agents that think before they act")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--map", default="tiny", help="bundled map name or path to a map file")
        p.add_argument("--planner", choices=sorted(PLANNERS), default="bfs")
        p.add_argument("--seed", type=int, default=0, help="seed for mcts")
        p.add_argument("--iterations", type=int, default=800, help="mcts iterations")
        p.add_argument("--max-nodes", type=int, default=50000, help="bfs node budget")
        p.add_argument("--max-steps", type=int, default=200)
        p.add_argument("--fake-wall", type=_pos, action="append", default=[],
                       help="tile the model wrongly believes is a wall, e.g. 4,1")
        p.add_argument("--fake-open", type=_pos, action="append", default=[],
                       help="wall tile the model wrongly believes is open")
        p.add_argument("--model", choices=["hand", "learned"], default="hand",
                       help="which world model the planners see")
        p.add_argument("--model-file", default=None,
                       help="transition log to load for --model learned")
        p.add_argument("--no-fallback", action="store_true",
                       help="learned model with no fallback: unseen pairs are bumps")

    r = sub.add_parser("run", help="run the agent in the env")
    add_common(r)
    r.add_argument("--trace", action="store_true", help="render the map every step")
    r.add_argument("--mpc", action="store_true", help="replan every step")
    r.add_argument("--no-replan", action="store_true", help="never replan on surprise")
    r.add_argument("--no-learn", action="store_true", help="never fix wall beliefs")
    r.add_argument("--record", default=None, metavar="PATH",
                   help="append this run's transitions to a jsonl log")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("solve", help="plan only, print the action list")
    add_common(s)
    s.set_defaults(func=cmd_solve)

    b = sub.add_parser("bench", help="compare planners across seeds")
    add_common(b)
    b.add_argument("--seeds", type=int, default=5)
    b.set_defaults(func=cmd_bench)

    g = sub.add_parser("gen", help="print a random solvable map")
    g.add_argument("--width", type=int, default=15)
    g.add_argument("--height", type=int, default=11)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gen)

    m = sub.add_parser("maps", help="list bundled maps")
    m.set_defaults(func=cmd_maps)

    l = sub.add_parser("learn", help="fit a tabular model from a transition log")
    l.add_argument("--log", required=True, help="jsonl transition log to fit")
    l.add_argument("--out", default=None, help="write the deduped model to PATH")
    l.set_defaults(func=cmd_learn)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
