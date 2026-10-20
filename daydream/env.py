"""gridworld environment. deterministic, fully observable, tiny on purpose."""

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4
ACTIONS = (UP, DOWN, LEFT, RIGHT, WAIT)
NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right", WAIT: "wait"}
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0), WAIT: (0, 0)}

STEP_COST = -1
GOAL_REWARD = 100
TRAP_REWARD = -100
KEY_REWARD = 2


class Transition:
    # one step of dynamics, shared by the env and the world model
    __slots__ = ("next_state", "reward", "done", "info")

    def __init__(self, next_state, reward, done, info=None):
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.info = info or {}


# state is a plain tuple: (x, y, keys_held, collected_keys, opened_doors)
# collected_keys and opened_doors are frozensets of (x, y), so states hash


class GridEnv:
    def __init__(self, text):
        self.walls = set()
        self.keys = set()
        self.doors = set()
        self.traps = set()
        self.goals = []
        self.start = None
        rows = [line.rstrip("\n") for line in text.strip("\n").split("\n")]
        if not rows:
            raise ValueError("empty map")
        width = len(rows[0])
        for y, row in enumerate(rows):
            if len(row) != width:
                raise ValueError("map rows must all be the same length")
            for x, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((x, y))
                elif ch == "S":
                    if self.start is not None:
                        raise ValueError("map needs exactly one S")
                    self.start = (x, y)
                elif ch == "G":
                    self.goals.append((x, y))
                elif ch == "k":
                    self.keys.add((x, y))
                elif ch == "D":
                    self.doors.add((x, y))
                elif ch == "T":
                    self.traps.add((x, y))
                elif ch in (".", " "):
                    pass
                else:
                    raise ValueError("unknown map char %r" % ch)
        if self.start is None:
            raise ValueError("map needs an S")
        if not self.goals:
            raise ValueError("map needs at least one G")
        self.width = width
        self.height = len(rows)

    def start_state(self):
        x, y = self.start
        return (x, y, 0, frozenset(), frozenset())

    def blocked(self, x, y, opened, keys=0):
        # walls stop you. a shut door stops you unless you hold a key.
        # everything else is fine.
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        if (x, y) in self.walls:
            return True
        if (x, y) in self.doors and (x, y) not in opened and keys == 0:
            return True
        return False

    def step(self, state, action):
        x, y, keys, collected, opened = state
        dx, dy = DELTAS[action]
        nx, ny = x + dx, y + dy
        info = {}
        if self.blocked(nx, ny, opened, keys):
            # bumping a wall or a locked door costs a step and teaches nothing else
            info["bump"] = True
            return Transition(state, STEP_COST, False, info)
        nkeys = keys
        ncollected = collected
        nopened = opened
        reward = STEP_COST
        if (nx, ny) in self.doors and (nx, ny) not in opened:
            # reachable only when blocked() said no, so we must hold a key
            nkeys = keys - 1
            nopened = opened | {(nx, ny)}
            info["opened_door"] = (nx, ny)
        if (nx, ny) in self.keys and (nx, ny) not in collected:
            nkeys = nkeys + 1
            ncollected = collected | {(nx, ny)}
            reward += KEY_REWARD
            info["picked_key"] = (nx, ny)
        if (nx, ny) in self.traps:
            info["outcome"] = "trap"
            return Transition((nx, ny, nkeys, ncollected, nopened), TRAP_REWARD, True, info)
        if (nx, ny) in self.goals:
            info["outcome"] = "goal"
            return Transition((nx, ny, nkeys, ncollected, nopened), GOAL_REWARD, True, info)
        return Transition((nx, ny, nkeys, ncollected, nopened), reward, False, info)

    def render(self, state):
        x, y, _, collected, opened = state
        out = []
        for j in range(self.height):
            row = []
            for i in range(self.width):
                p = (i, j)
                if p == (x, y):
                    row.append("@")
                elif p in self.walls:
                    row.append("#")
                elif p in self.doors and p not in opened:
                    row.append("D")
                elif p in self.traps:
                    row.append("T")
                elif p in self.keys and p not in collected:
                    row.append("k")
                elif p in self.goals:
                    row.append("G")
                elif p == self.start:
                    row.append("S")
                else:
                    row.append(".")
            out.append("".join(row))
        return "\n".join(out)
