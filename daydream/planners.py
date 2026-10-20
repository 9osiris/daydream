"""planners. they only ever touch the world model, never the real env.

that is the entire idea: the agent imagines futures inside its head
(the model) and only acts out the plan it likes. bfs is exact on small
maps, greedy is myopic and honest about it, mcts samples rollouts when
the state space gets big.
"""

import math
import random
from collections import deque

from .env import ACTIONS, NAMES


class Plan:
    def __init__(self, name, actions, value, expanded):
        self.name = name
        self.actions = list(actions)
        self.value = value
        self.expanded = expanded

    def describe(self):
        steps = " ".join(NAMES[a] for a in self.actions)
        return "%s: %d steps, value %.1f, %d nodes (%s)" % (
            self.name, len(self.actions), self.value, self.expanded, steps)


def bfs(model, state, max_nodes=50000):
    # shortest path to the goal inside the model. exact, but the state
    # space blows up on big maps, which is why the others exist.
    start = state
    prev = {start: None}
    todo = deque([start])
    expanded = 0
    while todo and expanded < max_nodes:
        s = todo.popleft()
        expanded += 1
        for a in ACTIONS:
            t = model.predict(s, a)
            ns = t.next_state
            if ns in prev:
                continue
            prev[ns] = (s, a)
            if t.done and t.reward > 0:
                return Plan("bfs", _reconstruct(prev, ns), _value(model, start, _reconstruct(prev, ns)), expanded)
            todo.append(ns)
    return Plan("bfs", [], 0.0, expanded)


def _reconstruct(prev, state):
    actions = []
    s = state
    while prev[s] is not None:
        p, a = prev[s]
        actions.append(a)
        s = p
    actions.reverse()
    return actions


def _value(model, state, actions):
    # imagined total reward of a plan, for reporting
    total = 0.0
    s = state
    for a in actions:
        t = model.predict(s, a)
        total += t.reward
        s = t.next_state
        if t.done:
            break
    return total


def greedy(model, state, horizon=200):
    # one step lookahead, ties broken toward the goal. no memory, so it
    # walks straight into dead ends and stays there. kept because every
    # planner zoo needs its cautionary tale.
    goals = model.env.goals
    actions = []
    s = state
    for _ in range(horizon):
        best = None
        best_key = None
        for a in ACTIONS:
            t = model.predict(s, a)
            nx, ny = t.next_state[0], t.next_state[1]
            dist = min(abs(nx - gx) + abs(ny - gy) for gx, gy in goals)
            key = (t.reward, -dist)
            if best_key is None or key > best_key:
                best_key = key
                best = (a, t)
        a, t = best
        actions.append(a)
        s = t.next_state
        if t.done:
            break
    if not _reaches_goal(model, state, actions):
        return Plan("greedy", [], _value(model, state, actions), len(actions))
    return Plan("greedy", actions, _value(model, state, actions), len(actions))


def _reaches_goal(model, state, actions):
    s = state
    for a in actions:
        t = model.predict(s, a)
        s = t.next_state
        if t.done:
            return t.reward > 0
    return False


class _Node:
    __slots__ = ("state", "done", "visits", "value", "children", "untried", "incoming")

    def __init__(self, state, model, done=False, incoming=0.0):
        self.state = state
        self.done = done
        self.visits = 0
        self.value = 0.0
        self.children = {}
        self.incoming = incoming
        # skip moves that leave the state unchanged (bumps, wait). every
        # step costs -1 and nothing here is time-dependent, so a no-op can
        # never be part of an optimal plan. without this the tree fills up
        # with self-loop children that all share the parent's state and
        # uct burns its whole budget re-exploring the same position.
        self.untried = [a for a in ACTIONS
                        if model.predict(state, a).next_state != state]


def mcts(model, state, iterations=800, horizon=80, c=1.41, seed=0):
    # monte carlo tree search with uct. rolls out imagined futures inside
    # the model, backs up the rewards, then follows the best child by
    # average value. seeded, so the same seed always dreams the same dream.
    rng = random.Random(seed)
    root = _Node(state, model)
    nodes = 1
    for _ in range(iterations):
        node = root
        path = [node]
        depth = 0
        while not node.untried and node.children and not node.done and depth < horizon:
            node = _uct_pick(node, c)
            path.append(node)
            depth += 1
        if node.untried and not node.done and depth < horizon:
            a = node.untried.pop(rng.randrange(len(node.untried)))
            t = model.predict(node.state, a)
            child = _Node(t.next_state, model, t.done, t.reward)
            node.children[a] = child
            path.append(child)
            nodes += 1
            reward = child.incoming + _rollout(model, t.next_state, t.done, rng, horizon - depth - 1)
        else:
            # selection landed on an existing node (terminal, boxed in, or
            # at the depth cap). back up from here, counting the reward of
            # the move that got us here, so revisiting a terminal node
            # reinforces it instead of backing up a bare 0.
            reward = node.incoming + _rollout(model, node.state, node.done, rng, horizon - depth)
        for n in path:
            n.visits += 1
            n.value += reward
    actions = []
    node = root
    value = 0.0
    for _ in range(horizon):
        if not node.children:
            break
        # follow the best child by average imagined value, not the most
        # visited one: with sparse rewards the visit counts stay noisy
        # long after the values have figured out where the goal is
        a, child = max(node.children.items(),
                       key=lambda kv: kv[1].value / kv[1].visits if kv[1].visits else 0.0)
        actions.append(a)
        value = child.value / child.visits if child.visits else 0.0
        node = child
        if node.done:
            break
    if not _reaches_goal(model, state, actions):
        return Plan("mcts", [], value, nodes)
    return Plan("mcts", actions, value, nodes)


def _uct_pick(node, c):
    log_n = math.log(node.visits)
    best = None
    best_score = None
    for child in node.children.values():
        score = child.value / child.visits + c * math.sqrt(log_n / child.visits)
        if best_score is None or score > best_score:
            best_score = score
            best = child
    return best


def _rollout(model, state, done, rng, depth):
    # heavy playouts: mostly drift toward the goal, sometimes wander.
    # pure random rollouts almost never stumble onto a sparse +100, which
    # leaves every branch valued near zero and the tree learns nothing.
    # the model knows where the traps are, so playouts simply do not
    # step on them: suicide teaches the tree nothing either.
    total = 0.0
    s = state
    d = done
    for _ in range(depth):
        if d:
            break
        safe = _safe_actions(model, s)
        if rng.random() < 0.4:
            a = rng.choice(safe)
        else:
            a = _toward_goal(model, s, rng, safe)
        t = model.predict(s, a)
        total += t.reward
        s = t.next_state
        d = t.done
    return total


def _safe_actions(model, s):
    # moves that do not end the run badly right now. if every move kills
    # you, you are already dead, so allow them all and let it happen.
    safe = []
    for a in ACTIONS:
        t = model.predict(s, a)
        if not (t.done and t.reward < 0):
            safe.append(a)
    return safe or list(ACTIONS)


def _toward_goal(model, s, rng, safe):
    # a move that shrinks the manhattan distance to the nearest goal,
    # ties broken randomly. still entirely inside the model.
    goals = model.env.goals
    best = []
    best_d = None
    for a in safe:
        t = model.predict(s, a)
        dist = min(abs(t.next_state[0] - gx) + abs(t.next_state[1] - gy) for gx, gy in goals)
        if best_d is None or dist < best_d:
            best_d = dist
            best = [a]
        elif dist == best_d:
            best.append(a)
    return rng.choice(best)
