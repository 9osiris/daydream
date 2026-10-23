"""learned world models: fit the dynamics from transition logs.

WorldModel.predict hand-writes a mirror of the env. TabularModel instead
records real transitions (state, action -> next_state, reward, done) and
looks them up. wrong beliefs arise on their own from sparse or biased
data, which is the point: this is what happens when your model of the
world comes from experience instead of the source code.

a model file is just the transition log (jsonl). the learn cli command
dedupes and prints coverage stats, run --model learned loads it for the
planners. states are plain tuples, so everything json-serializes.
"""

import json

from .env import Transition, STEP_COST


def encode_state(state):
    # (x, y, keys, frozenset, frozenset) -> json-safe nested list
    x, y, keys, collected, opened = state
    return [x, y, keys, sorted(collected), sorted(opened)]


def decode_state(raw):
    x, y, keys, collected, opened = raw
    return (x, y, keys,
            frozenset(tuple(p) for p in collected),
            frozenset(tuple(p) for p in opened))


class TransitionLog:
    def __init__(self):
        self.rows = []

    def record(self, state, action, next_state, reward, done):
        self.rows.append({
            "state": encode_state(state),
            "action": action,
            "next_state": encode_state(next_state),
            "reward": reward,
            "done": bool(done),
        })

    def __len__(self):
        return len(self.rows)

    def save(self, path):
        with open(path, "a") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    def write(self, path):
        # fresh write, not append. used by the learn command.
        with open(path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")

    @classmethod
    def load(cls, path):
        log = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    log.rows.append(json.loads(line))
        return log


class TabularModel:
    def __init__(self, fallback=None):
        # fallback answers predict() for pairs the log never saw. pass a
        # WorldModel for full coverage, or leave None and get a bump.
        self.table = {}
        self.fallback = fallback

    def record(self, state, action, next_state, reward, done):
        key = (state, action)
        new = key not in self.table
        self.table[key] = (next_state, reward, done)
        return new

    def fit(self, log):
        # merge a TransitionLog (or a path to one) into the table
        if isinstance(log, str):
            log = TransitionLog.load(log)
        for row in log.rows:
            state = decode_state(row["state"])
            next_state = decode_state(row["next_state"])
            self.record(state, row["action"], next_state, row["reward"], row["done"])
        return self

    def predict(self, state, action):
        hit = self.table.get((state, action))
        if hit is not None:
            next_state, reward, done = hit
            return Transition(next_state, reward, done, {"learned": True})
        if self.fallback is not None:
            return self.fallback.predict(state, action)
        # no data and no fallback: assume a bump, stay put, pay the step
        return Transition(state, STEP_COST, False, {"unseen": True})

    def observe(self, state, action, actual):
        # same hook the agent calls on surprises. for a learned model,
        # every real transition is training data.
        return self.record(state, action, actual.next_state, actual.reward, actual.done)

    def surprise(self, state, action, actual):
        predicted = self.predict(state, action)
        if predicted.next_state == actual.next_state and predicted.reward == actual.reward:
            return 0.0
        return abs(predicted.reward - actual.reward) + 1.0

    def stats(self):
        states = set(s for (s, a) in self.table)
        return {
            "transitions": len(self.table),
            "states": len(states),
        }

    def to_log(self):
        log = TransitionLog()
        for (state, action), (next_state, reward, done) in self.table.items():
            log.record(state, action, next_state, reward, done)
        return log

    def save(self, path):
        self.to_log().write(path)

    @classmethod
    def load(cls, path, fallback=None):
        return cls(fallback=fallback).fit(TransitionLog.load(path))
