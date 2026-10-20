"""the agent: plan in the model, act in the world, fix the model on surprise.

open loop by default: it commits to the whole plan and only replans when
reality disagrees with imagination. --mpc replans every single step,
which is safer under a wrong model and slower everywhere else.
"""

from .env import NAMES


class RunResult:
    def __init__(self):
        self.succeeded = False
        self.outcome = None
        self.steps = 0
        self.reward = 0.0
        self.replans = 0
        self.surprises = 0
        self.imagined = 0.0
        self.trace = []

    def summary(self):
        return "succeeded=%s outcome=%s steps=%d reward=%.0f replans=%d surprises=%d imagined=%.0f" % (
            self.succeeded, self.outcome, self.steps, self.reward,
            self.replans, self.surprises, self.imagined)


def run_agent(env, model, planner, max_steps=200, mpc=False, replan=True, learn=True, **planner_kw):
    # planner is a callable: planner(model, state, **planner_kw) -> Plan
    res = RunResult()
    state = env.start_state()
    plan = planner(model, state, **planner_kw)
    res.imagined = plan.value
    res.trace.append(("plan", plan.describe()))
    if not plan.actions:
        res.trace.append(("stuck", "planner found no plan from the start"))
        return res
    while res.steps < max_steps:
        action = plan.actions.pop(0)
        before = state
        predicted = model.predict(before, action)
        actual = env.step(before, action)
        res.steps += 1
        res.reward += actual.reward
        res.trace.append(("step", NAMES[action], predicted.next_state, actual.next_state,
                          predicted.reward, actual.reward))
        state = actual.next_state
        if actual.done:
            res.outcome = actual.info.get("outcome")
            res.succeeded = res.outcome == "goal"
            break
        surprised = (predicted.next_state != actual.next_state
                     or predicted.reward != actual.reward)
        if surprised:
            res.surprises += 1
            if learn:
                corrected = model.observe(before, action, actual)
                res.trace.append(("learn", "corrected a wall belief" if corrected else "no belief to correct"))
        if mpc or (surprised and replan) or not plan.actions:
            plan = planner(model, state, **planner_kw)
            res.imagined = plan.value
            if mpc or surprised:
                res.replans += 1
            res.trace.append(("replan", plan.describe()))
            if not plan.actions:
                res.trace.append(("stuck", "planner found no plan anymore"))
                break
    return res
