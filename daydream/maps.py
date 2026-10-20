"""bundled maps and a random map generator.

map legend:
  #  wall        S  start       G  goal
  .  floor       k  key         D  door (costs one key)
  T  trap (ends the run, -100)
"""

import random

TINY = """\
#####
#S.G#
#####
"""

KEYS = """\
#######
#S.kDG#
#...#.#
#######
"""

POCKET = """\
#######
#S#...#
#.#.#.#
#...#G#
#######
"""

DETOUR = """\
#########
#S..#..G#
#...#...#
#...#...#
#...#...#
#.......#
#########
"""

MAZE = """\
#############
#S....#.....#
#.##..#..#..#
#..##...T#..#
##...###.#..#
#..T.#...#..#
#.##.#.##.#.#
#....#..k.#.#
#.####.##.#G#
#..D......#.#
#############
"""

BUNDLED = {
    "tiny": TINY,
    "keys": KEYS,
    "pocket": POCKET,
    "detour": DETOUR,
    "maze": MAZE,
}


def get(name):
    # bundled map by name, or a path to a map file
    if name in BUNDLED:
        return BUNDLED[name]
    with open(name) as f:
        return f.read()


def generate(width=15, height=11, seed=0, wall_prob=0.22, trap_prob=0.04):
    # random walls and traps, retried until a solver can reach the goal
    from .env import GridEnv
    from .planners import bfs
    from .world import WorldModel

    rng = random.Random(seed)
    for _ in range(300):
        rows = []
        for y in range(height):
            row = []
            for x in range(width):
                if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                    row.append("#")
                elif (x, y) == (1, 1):
                    row.append("S")
                elif (x, y) == (width - 2, height - 2):
                    row.append("G")
                else:
                    r = rng.random()
                    if r < trap_prob:
                        row.append("T")
                    elif r < trap_prob + wall_prob:
                        row.append("#")
                    else:
                        row.append(".")
            rows.append("".join(row))
        text = "\n".join(rows)
        try:
            env = GridEnv(text)
        except ValueError:
            continue
        plan = bfs(WorldModel(env), env.start_state())
        if plan.actions:
            return text
    raise RuntimeError("could not generate a solvable map, try another seed")
