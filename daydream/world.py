"""the world model: what the agent believes the env looks like.

a perfect model predicts exactly what the env will do. an imperfect one
has wrong beliefs about some tiles (fake walls, phantom openings), so the
agent plans through a world that is not quite real. observe() lets the
agent fix its beliefs from experience, which is the whole point: plan in
your head, act in the world, update the head when it lies to you.
"""

from .env import GridEnv


class WorldModel:
    def __init__(self, env, fake_walls=(), fake_open=()):
        # fake_walls: open tiles the model wrongly believes are walls
        # fake_open: wall tiles the model wrongly believes are open
        self.env = env
        self.believed = _copy_layout(env)
        for p in fake_walls:
            self._check_believable(p)
            self.believed.walls.add(p)
        for p in fake_open:
            self._check_believable(p)
            self.believed.walls.discard(p)

    def _check_believable(self, p):
        # the model can be wrong about walls, not about the furniture
        if p in self.env.keys or p in self.env.doors or p in self.env.traps:
            raise ValueError("cant corrupt a key/door/trap tile: %r" % (p,))
        if p in self.env.goals or p == self.env.start:
            raise ValueError("cant corrupt start/goal tiles: %r" % (p,))

    def predict(self, state, action):
        # what the agent imagines will happen. pure function of state.
        return self.believed.step(state, action)

    def observe(self, state, action, actual):
        # compare what we imagined against what really happened, and fix
        # the belief if the mismatch is about a wall. returns True when a
        # belief was corrected.
        predicted = self.predict(state, action)
        if predicted.next_state == actual.next_state:
            return False
        dx, dy = _delta(action)
        x, y = state[0], state[1]
        target = (x + dx, y + dy)
        # we only correct wall beliefs, and only for plain moves
        if target in self.env.keys or target in self.env.doors or target in self.env.traps:
            return False
        if target in self.env.goals or target == self.env.start:
            return False
        really_blocked = target in self.env.walls
        believed_blocked = target in self.believed.walls
        if really_blocked == believed_blocked:
            return False
        if really_blocked:
            self.believed.walls.add(target)
        else:
            self.believed.walls.discard(target)
        return True

    def surprise(self, state, action, actual):
        # how wrong the prediction was, as a single number
        predicted = self.predict(state, action)
        if predicted.next_state == actual.next_state and predicted.reward == actual.reward:
            return 0.0
        return abs(predicted.reward - actual.reward) + 1.0


def _copy_layout(env):
    # a second GridEnv with the same layout, built without re-parsing
    believed = GridEnv.__new__(GridEnv)
    believed.walls = set(env.walls)
    believed.keys = set(env.keys)
    believed.doors = set(env.doors)
    believed.traps = set(env.traps)
    believed.goals = list(env.goals)
    believed.start = env.start
    believed.width = env.width
    believed.height = env.height
    return believed


def _delta(action):
    from .env import DELTAS
    return DELTAS[action]
