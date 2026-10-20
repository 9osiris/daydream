"""tests for daydream. run with: python -m unittest discover -s tests"""

import io
import sys
import unittest

sys.path.insert(0, ".")

from daydream.env import GridEnv, UP, DOWN, LEFT, RIGHT, WAIT
from daydream.world import WorldModel
from daydream import planners, maps
from daydream.agent import run_agent
from daydream.cli import main as cli_main


def env_of(name):
    return GridEnv(maps.get(name))


class TestEnv(unittest.TestCase):
    def test_parse_tiny(self):
        env = env_of("tiny")
        self.assertEqual((env.width, env.height), (5, 3))
        self.assertEqual(env.start, (1, 1))
        self.assertEqual(env.goals, [(3, 1)])

    def test_bad_maps_rejected(self):
        with self.assertRaises(ValueError):
            GridEnv("###\n#S#\n###")  # no goal
        with self.assertRaises(ValueError):
            GridEnv("#####\n#S.G#\n####\n")  # ragged rows
        with self.assertRaises(ValueError):
            GridEnv("#####\n#S.X#\n#####")  # unknown char, also no goal

    def test_wall_bump_stays(self):
        env = env_of("tiny")
        s = env.start_state()
        t = env.step(s, LEFT)
        self.assertEqual(t.next_state, s)
        self.assertTrue(t.info["bump"])
        self.assertFalse(t.done)
        self.assertEqual(t.reward, -1)

    def test_walk_to_goal(self):
        env = env_of("tiny")
        s = env.start_state()
        t = env.step(s, RIGHT)
        self.assertFalse(t.done)
        t = env.step(t.next_state, RIGHT)
        self.assertTrue(t.done)
        self.assertEqual(t.reward, 100)
        self.assertEqual(t.info["outcome"], "goal")

    def test_key_and_door(self):
        env = env_of("keys")
        s = env.start_state()  # (1,1)
        s = env.step(s, RIGHT).next_state  # (2,1)
        t = env.step(s, RIGHT)  # (3,1) key
        self.assertEqual(t.next_state[2], 1)
        self.assertIn("picked_key", t.info)
        s = t.next_state
        # door without key would block; we have the key
        t = env.step(s, RIGHT)  # (4,1) door
        self.assertIn("opened_door", t.info)
        self.assertEqual(t.next_state[2], 0)  # key consumed
        self.assertIn((4, 1), t.next_state[4])
        t = env.step(t.next_state, RIGHT)  # (5,1) goal
        self.assertTrue(t.done)
        self.assertEqual(t.info["outcome"], "goal")

    def test_door_blocks_without_key(self):
        env = env_of("keys")
        # teleport past the key to the door tile approach
        s = (3, 1, 0, frozenset(), frozenset())
        t = env.step(s, RIGHT)
        self.assertEqual(t.next_state, s)
        self.assertTrue(t.info["bump"])

    def test_trap_kills(self):
        env = GridEnv("######\n#S.TG#\n######")
        s = env.start_state()
        s = env.step(s, RIGHT).next_state
        t = env.step(s, RIGHT)
        self.assertTrue(t.done)
        self.assertEqual(t.reward, -100)
        self.assertEqual(t.info["outcome"], "trap")

    def test_render_marks_agent(self):
        env = env_of("tiny")
        out = env.render(env.start_state())
        self.assertIn("@", out)
        self.assertNotIn("S", out)  # agent stands on the start tile


class TestWorld(unittest.TestCase):
    def test_perfect_model_matches_env(self):
        env = env_of("maze")
        model = WorldModel(env)
        import random
        rng = random.Random(7)
        s = env.start_state()
        for _ in range(60):
            a = rng.choice([UP, DOWN, LEFT, RIGHT])
            p = model.predict(s, a)
            r = env.step(s, a)
            self.assertEqual(p.next_state, r.next_state)
            self.assertEqual(p.reward, r.reward)
            self.assertEqual(p.done, r.done)
            s = r.next_state
            if r.done:
                break

    def test_fake_wall_changes_prediction(self):
        env = env_of("tiny")
        model = WorldModel(env, fake_walls=[(2, 1)])
        s = env.start_state()
        t = model.predict(s, RIGHT)
        self.assertEqual(t.next_state, s)  # model thinks (2,1) is a wall
        self.assertTrue(env.step(s, RIGHT).next_state != s)  # but it is not

    def test_fake_open_changes_prediction(self):
        env = env_of("detour")
        model = WorldModel(env, fake_open=[(4, 1)])
        s = (3, 1, 0, frozenset(), frozenset())
        t = model.predict(s, RIGHT)
        self.assertEqual(t.next_state[:2], (4, 1))  # model walks through the wall
        real = env.step(s, RIGHT)
        self.assertEqual(real.next_state, s)  # reality bumps

    def test_cant_corrupt_furniture(self):
        env = env_of("keys")
        with self.assertRaises(ValueError):
            WorldModel(env, fake_walls=[(3, 1)])  # key tile
        with self.assertRaises(ValueError):
            WorldModel(env, fake_open=[(4, 1)])  # door tile
        with self.assertRaises(ValueError):
            WorldModel(env, fake_walls=[(1, 1)])  # start

    def test_observe_corrects_belief(self):
        env = env_of("detour")
        model = WorldModel(env, fake_open=[(4, 1)])
        s = (3, 1, 0, frozenset(), frozenset())
        actual = env.step(s, RIGHT)  # bump, stayed at (3,1)
        self.assertTrue(model.observe(s, RIGHT, actual))
        t = model.predict(s, RIGHT)
        self.assertEqual(t.next_state, s)  # model learned the wall

    def test_observe_noop_when_right(self):
        env = env_of("tiny")
        model = WorldModel(env)
        s = env.start_state()
        actual = env.step(s, RIGHT)
        self.assertFalse(model.observe(s, RIGHT, actual))


class TestPlanners(unittest.TestCase):
    def test_bfs_solves_tiny(self):
        env = env_of("tiny")
        plan = planners.bfs(WorldModel(env), env.start_state())
        self.assertEqual(len(plan.actions), 2)
        self.assertEqual(plan.value, 99)  # -1 then +100

    def test_bfs_solves_keys(self):
        env = env_of("keys")
        model = WorldModel(env)
        plan = planners.bfs(model, env.start_state())
        self.assertTrue(plan.actions)
        # simulate the plan in the model, it must end at the goal
        s = env.start_state()
        for a in plan.actions:
            t = model.predict(s, a)
            s = t.next_state
        self.assertTrue(t.done and t.reward > 0)

    def test_bfs_empty_when_impossible(self):
        env = GridEnv("#####\n#S#G#\n#####")
        plan = planners.bfs(WorldModel(env), env.start_state())
        self.assertEqual(plan.actions, [])

    def test_greedy_solves_tiny(self):
        env = env_of("tiny")
        plan = planners.greedy(WorldModel(env), env.start_state())
        self.assertTrue(plan.actions)

    def test_greedy_stuck_in_pocket(self):
        # greedy has no memory: it walks into the dead end at (3,3),
        # manhattan stops improving, and it bumps forever
        env = env_of("pocket")
        plan = planners.greedy(WorldModel(env), env.start_state())
        self.assertEqual(plan.actions, [])

    def test_bfs_beats_pocket(self):
        env = env_of("pocket")
        plan = planners.bfs(WorldModel(env), env.start_state())
        self.assertTrue(plan.actions)

    def test_mcts_solves_maze(self):
        env = env_of("maze")
        plan = planners.mcts(WorldModel(env), env.start_state(), iterations=1200, seed=3)
        self.assertTrue(plan.actions)

    def test_mcts_deterministic(self):
        env = env_of("maze")
        m = WorldModel(env)
        a = planners.mcts(m, env.start_state(), iterations=400, seed=11)
        b = planners.mcts(m, env.start_state(), iterations=400, seed=11)
        self.assertEqual(a.actions, b.actions)

    def test_mcts_empty_when_impossible(self):
        env = GridEnv("#####\n#S#G#\n#####")
        plan = planners.mcts(WorldModel(env), env.start_state(), iterations=100, seed=1)
        self.assertEqual(plan.actions, [])


class TestAgent(unittest.TestCase):
    def test_open_loop_perfect_model(self):
        env = env_of("keys")
        res = run_agent(env, WorldModel(env), planners.bfs)
        self.assertTrue(res.succeeded)
        self.assertEqual(res.surprises, 0)
        self.assertEqual(res.replans, 0)
        self.assertEqual(res.outcome, "goal")

    def test_wrong_model_replans_and_learns(self):
        # model believes the wall at (4,1) is open, plans straight
        # through it, bumps, fixes the belief, replans the long way
        env = env_of("detour")
        model = WorldModel(env, fake_open=[(4, 1)])
        res = run_agent(env, model, planners.bfs)
        self.assertTrue(res.succeeded)
        self.assertGreaterEqual(res.surprises, 1)
        self.assertGreaterEqual(res.replans, 1)
        # and the belief actually got fixed
        self.assertIn((4, 1), model.believed.walls)

    def test_wrong_model_no_learn_no_replan_fails(self):
        env = env_of("detour")
        model = WorldModel(env, fake_open=[(4, 1)])
        res = run_agent(env, model, planners.bfs, replan=False, learn=False, max_steps=60)
        self.assertFalse(res.succeeded)

    def test_mpc_every_step(self):
        env = env_of("maze")
        res = run_agent(env, WorldModel(env), planners.bfs, mpc=True, max_steps=120)
        self.assertTrue(res.succeeded)
        self.assertGreater(res.replans, 5)

    def test_no_plan_reports_stuck(self):
        env = GridEnv("#####\n#S#G#\n#####")
        res = run_agent(env, WorldModel(env), planners.bfs)
        self.assertFalse(res.succeeded)
        self.assertTrue(any(e[0] == "stuck" for e in res.trace))

    def test_greedy_fails_honestly_on_pocket(self):
        # greedy has no memory, so on the pocket map it finds no plan.
        # the agent reports being stuck instead of crashing.
        env = env_of("pocket")
        res = run_agent(env, WorldModel(env), planners.greedy)
        self.assertFalse(res.succeeded)
        self.assertTrue(any(e[0] == "stuck" for e in res.trace))

    def test_bfs_routes_around_traps(self):
        env = env_of("maze")
        res = run_agent(env, WorldModel(env), planners.bfs, max_steps=150)
        self.assertTrue(res.succeeded)
        self.assertEqual(res.outcome, "goal")


class TestMaps(unittest.TestCase):
    def test_generate_solvable(self):
        for seed in range(5):
            text = maps.generate(seed=seed)
            env = GridEnv(text)
            plan = planners.bfs(WorldModel(env), env.start_state())
            self.assertTrue(plan.actions, "seed %d unsolvable" % seed)

    def test_bundled_maps_all_solvable(self):
        for name in maps.BUNDLED:
            env = GridEnv(maps.get(name))
            plan = planners.bfs(WorldModel(env), env.start_state())
            self.assertTrue(plan.actions, name)


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = cli_main(argv)
        finally:
            sys.stdout = old
        return code, buf.getvalue()

    def test_solve(self):
        code, out = self.run_cli(["solve", "--map", "tiny", "--planner", "bfs"])
        self.assertEqual(code, 0)
        self.assertIn("right right", out)

    def test_run(self):
        code, out = self.run_cli(["run", "--map", "keys", "--planner", "bfs"])
        self.assertEqual(code, 0)
        self.assertIn("succeeded=True", out)

    def test_run_greedy_tiny(self):
        code, out = self.run_cli(["solve", "--map", "tiny", "--planner", "greedy"])
        self.assertEqual(code, 0)  # greedy solves tiny fine

    def test_bench(self):
        code, out = self.run_cli(["bench", "--map", "tiny", "--seeds", "2"])
        self.assertEqual(code, 0)
        self.assertIn("bfs", out)
        self.assertIn("mcts", out)

    def test_maps(self):
        code, out = self.run_cli(["maps"])
        self.assertEqual(code, 0)
        self.assertIn("maze", out)

    def test_gen(self):
        code, out = self.run_cli(["gen", "--seed", "42", "--width", "9", "--height", "7"])
        self.assertEqual(code, 0)
        env = GridEnv(out)  # the printed map must parse
        self.assertEqual((env.width, env.height), (9, 7))


if __name__ == "__main__":
    unittest.main()
