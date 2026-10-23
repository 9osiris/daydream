daydream - world models for agents that think before they act [educational project]

disclaimer:
this is a toy, not a framework. i built it to mess with the plan-act-replan
loop that sits inside every agent i write. the gridworld is tiny on purpose,
the planners are textbook stuff (bfs, greedy, mcts), and none of this is
meant to run anything real. educational, for fun.

why i made this:
every agent i build does the same thing: imagine a future, act, get
surprised, replan. cognix does it with beliefs and goals, mule does it with
tool calls, but nobody had the "imagine" part as its own piece. so this is
that piece, standalone. the world model is just a believed copy of the map
that can be wrong on purpose (fake walls, phantom openings), and the agent
fixes its beliefs when reality disagrees. that plan-in-your-head, act-in-the-
world, update-the-head loop is the whole project. i also wanted a planner zoo
in one place because greedy getting stuck in a dead end forever is funny and
instructive. stdlib only, no deps, because these little things should run
anywhere.

the guide:

```bash
python -m daydream maps                        # list the bundled maps
python -m daydream run --map keys               # watch the agent solve it
python -m daydream run --map maze --planner mcts --trace
python -m daydream solve --map pocket --planner bfs
python -m daydream bench --map maze --seeds 5   # planners head to head
python -m daydream gen --seed 7                 # random solvable map
```

maps are ascii: `#` wall, `S` start, `G` goal, `k` key, `D` door (costs one
key), `T` trap. each step costs 1, the goal pays 100, traps pay -100.

the interesting flags:

```bash
# lie to the model: it believes the wall at (4,1) is open
python -m daydream run --map detour --fake-open 4,1

# same, but the agent never fixes its beliefs and never replans (it fails)
python -m daydream run --map detour --fake-open 4,1 --no-learn --no-replan

# replan every single step instead of only on surprise
python -m daydream run --map maze --planner mcts --mpc
```

as a library:

```python
from daydream import GridEnv, WorldModel, run_agent, planners, maps

env = GridEnv(maps.get("keys"))
model = WorldModel(env, fake_open=[(4, 1)])  # optional wrong beliefs
result = run_agent(env, model, planners.mcts, iterations=800, seed=0)
print(result.summary())
# succeeded=True outcome=goal steps=9 reward=93 replans=0 surprises=0 imagined=93
```

planners only ever see the model, never the real env. that separation is
the point. `WorldModel.observe()` patches wall beliefs from experience, so
an agent with a wrong model can still converge by bumping into reality.

learned models: instead of hand-writing the dynamics, record transitions
and fit a table. sparse logs make the model wrong on its own, no fake
walls needed.

```bash
python -m daydream run --map keys --record run.jsonl   # record the episode
python -m daydream learn --log run.jsonl               # coverage stats
python -m daydream run --map keys --model learned --model-file run.jsonl
# unseen pairs fall back to the hand model; --no-fallback turns them into bumps
```

as a library:

```python
from daydream.learn import TabularModel, TransitionLog

log = TransitionLog()
# ... record real transitions: log.record(state, action, next_state, reward, done)
model = TabularModel(fallback=WorldModel(env)).fit(log)
result = run_agent(env, model, planners.bfs)
```

`TabularModel.observe()` records every real transition, so it plugs into
the agent's learn path: run with a half-trained model and it patches its
own table wherever reality disagrees.

tests: `python -m unittest discover -s tests` (55 tests, no network, no keys).
