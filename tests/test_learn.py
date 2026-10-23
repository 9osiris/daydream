"""tests for the learned tabular world model."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from daydream.env import GridEnv, UP, DOWN, LEFT, RIGHT, WAIT, STEP_COST
from daydream.world import WorldModel
from daydream.learn import TabularModel, TransitionLog, encode_state, decode_state
from daydream import planners, maps
from daydream.agent import run_agent
from daydream.cli import main as cli_main


def env_of(name):
    return GridEnv(maps.get(name))


class TestStateCodec(unittest.TestCase):
    def test_roundtrip_with_frozensets(self):
        env = env_of("keys")
        s = env.start_state()
        s2 = (s[0], s[1], 2, frozenset([(3, 4)]), frozenset([(5, 6)]))
        raw = json.dumps(encode_state(s2))
        self.assertEqual(decode_state(json.loads(raw)), s2)

    def test_codec_is_hashable_after_decode(self):
        env = env_of("tiny")
        s = decode_state(encode_state(env.start_state()))
        self.assertEqual(s, env.start_state())


class TestTransitionLog(unittest.TestCase):
    def test_save_load_roundtrip(self):
        env = env_of("tiny")
        s = env.start_state()
        t = env.step(s, RIGHT)
        log = TransitionLog()
        log.record(s, RIGHT, t.next_state, t.reward, t.done)
        self.assertEqual(len(log), 1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.jsonl")
            log.save(p)
            back = TransitionLog.load(p)
        self.assertEqual(len(back), 1)
        row = back.rows[0]
        self.assertEqual(decode_state(row["state"]), s)
        self.assertEqual(decode_state(row["next_state"]), t.next_state)
        self.assertEqual(row["action"], RIGHT)
        self.assertEqual(row["reward"], t.reward)
        self.assertEqual(row["done"], t.done)

    def test_save_appends(self):
        log = TransitionLog()
        env = env_of("tiny")
        s = env.start_state()
        log.record(s, RIGHT, s, -1, False)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.jsonl")
            log.save(p)
            log2 = TransitionLog()
            log2.record(s, LEFT, s, -1, False)
            log2.save(p)
            back = TransitionLog.load(p)
        self.assertEqual(len(back), 2)


class TestTabularModel(unittest.TestCase):
    def test_predict_returns_recorded_transition(self):
        env = env_of("tiny")
        m = TabularModel()
        s = env.start_state()
        actual = env.step(s, RIGHT)
        m.record(s, RIGHT, actual.next_state, actual.reward, actual.done)
        p = m.predict(s, RIGHT)
        self.assertEqual(p.next_state, actual.next_state)
        self.assertEqual(p.reward, actual.reward)
        self.assertEqual(p.done, actual.done)

    def test_unseen_pair_is_a_bump_without_fallback(self):
        env = env_of("tiny")
        m = TabularModel()
        s = env.start_state()
        p = m.predict(s, RIGHT)
        self.assertEqual(p.next_state, s)
        self.assertEqual(p.reward, STEP_COST)
        self.assertFalse(p.done)
        self.assertTrue(p.info.get("unseen"))

    def test_unseen_pair_uses_fallback(self):
        env = env_of("tiny")
        m = TabularModel(fallback=WorldModel(env))
        s = env.start_state()
        p = m.predict(s, RIGHT)
        expected = env.step(s, RIGHT)
        self.assertEqual(p.next_state, expected.next_state)
        self.assertEqual(p.reward, expected.reward)

    def test_fit_from_log(self):
        env = env_of("tiny")
        log = TransitionLog()
        s = env.start_state()
        t = env.step(s, RIGHT)
        log.record(s, RIGHT, t.next_state, t.reward, t.done)
        m = TabularModel().fit(log)
        self.assertEqual(m.stats()["transitions"], 1)
        p = m.predict(s, RIGHT)
        self.assertEqual(p.next_state, t.next_state)

    def test_fit_dedupes_repeats(self):
        env = env_of("tiny")
        s = env.start_state()
        t = env.step(s, RIGHT)
        log = TransitionLog()
        for _ in range(3):
            log.record(s, RIGHT, t.next_state, t.reward, t.done)
        m = TabularModel().fit(log)
        self.assertEqual(m.stats()["transitions"], 1)

    def test_surprise_matches_hand_model_formula(self):
        env = env_of("tiny")
        m = TabularModel()
        s = env.start_state()
        actual = env.step(s, RIGHT)
        m.record(s, RIGHT, actual.next_state, actual.reward, actual.done)
        self.assertEqual(m.surprise(s, RIGHT, actual), 0.0)
        wrong = env.step(s, LEFT)
        self.assertGreater(m.surprise(s, RIGHT, wrong), 0.0)

    def test_stats_counts_states(self):
        env = env_of("tiny")
        m = TabularModel()
        s = env.start_state()
        t = env.step(s, RIGHT)
        m.record(s, RIGHT, t.next_state, t.reward, t.done)
        m.record(s, LEFT, s, STEP_COST, False)
        stats = m.stats()
        self.assertEqual(stats["transitions"], 2)
        self.assertEqual(stats["states"], 1)


class TestLearnedModelEndToEnd(unittest.TestCase):
    def test_recorded_run_trains_a_model_that_solves(self):
        env = env_of("tiny")
        log = TransitionLog()
        run_agent(env, WorldModel(env), planners.bfs, max_steps=50,
                  on_step=lambda s, a, t: log.record(s, a, t.next_state, t.reward, t.done))
        self.assertGreater(len(log), 0)
        m = TabularModel(fallback=WorldModel(env)).fit(log)
        res = run_agent(env, m, planners.bfs, max_steps=50)
        self.assertTrue(res.succeeded)

    def test_sparse_log_gives_naturally_wrong_beliefs(self):
        # trained on one transition only: everything else is a bump
        env = env_of("tiny")
        m = TabularModel()  # no fallback
        s = env.start_state()
        m.record(s, RIGHT, env.step(s, RIGHT).next_state, -1, False)
        res = run_agent(env, m, planners.bfs, max_steps=50)
        self.assertFalse(res.succeeded)

    def test_observe_plugs_into_agent_learn_path(self):
        # fit the model, then corrupt one entry: the agent should bump
        # into the lie, observe the truth, and correct the table itself
        env = env_of("tiny")
        log = TransitionLog()
        run_agent(env, WorldModel(env), planners.bfs, max_steps=50,
                  on_step=lambda s, a, t: log.record(s, a, t.next_state, t.reward, t.done))
        m = TabularModel().fit(log)
        s0 = env.start_state()
        real = env.step(s0, RIGHT)
        m.table[(s0, RIGHT)] = (real.next_state, 100, False)  # the lie
        res = run_agent(env, m, planners.bfs, max_steps=50, learn=True)
        self.assertTrue(res.succeeded)
        self.assertGreater(res.surprises, 0)
        fixed = m.predict(s0, RIGHT)
        self.assertEqual(fixed.next_state, real.next_state)
        self.assertEqual(fixed.reward, real.reward)


class TestLearnCli(unittest.TestCase):
    def run_cli(self, argv):
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            code = cli_main(argv)
            return code, sys.stdout.getvalue()
        finally:
            sys.stdout = old

    def test_run_record_then_learn_then_solve(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.jsonl")
            code, out = self.run_cli(["run", "--map", "tiny", "--record", p])
            self.assertEqual(code, 0)
            self.assertIn("recorded", out)
            self.assertTrue(os.path.exists(p))
            code, out = self.run_cli(["learn", "--log", p])
            self.assertEqual(code, 0)
            self.assertIn("transitions:", out)
            code, out = self.run_cli(["run", "--map", "tiny", "--model", "learned",
                                      "--model-file", p])
            self.assertEqual(code, 0)
            self.assertIn("succeeded=True", out)

    def test_learn_out_writes_deduped_model(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.jsonl")
            out = os.path.join(d, "model.jsonl")
            self.run_cli(["run", "--map", "tiny", "--record", p])
            code, text = self.run_cli(["learn", "--log", p, "--out", out])
            self.assertEqual(code, 0)
            m = TabularModel.load(out)
            raw = TransitionLog.load(p)
            self.assertLessEqual(m.stats()["transitions"], len(raw))

    def test_learned_requires_model_file(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["run", "--map", "tiny", "--model", "learned"])


if __name__ == "__main__":
    unittest.main()
