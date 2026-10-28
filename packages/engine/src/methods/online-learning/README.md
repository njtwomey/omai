# Online learning (part 5, learning paradigms)

Learning as data arrive: online convex optimisation (online gradient descent, regret), prediction with expert advice (weighted majority, Hedge, follow the leader), streaming and incremental learners (online SGD, passive-aggressive, the perceptron, Hoeffding trees), online active learning, and continual learning (catastrophic forgetting, replay, regularisation), as sessions over `optim`, `learn` and the models of `supervised-learning` with a stream as the data slot.

## Capabilities

- The stream base: data arriving one point or one batch per step, with drift and recurrence planted (`shift-and-drift`); every learner's cumulative loss and regret on the tape.
- Online convex optimisation: the regret against the best fixed point, the step-size rule as the knob.
- Experts: the weights over experts updating, the regret bound drawn.
- Incremental learners: the perceptron's mistakes, passive-aggressive's margin updates, the Hoeffding tree splitting when the bound allows.
- Continual learning: tasks in sequence, accuracy on earlier tasks falling and recovered by replay or EWC.

## Examples to make

- The perceptron on a stream: each mistake and the boundary's move.
- Hedge's weights over experts as one expert turns bad, the regret curve against its bound.
- A Hoeffding tree splitting on a stream, the bound and the gap at each candidate.
- Catastrophic forgetting on two tasks, with replay as the knob.

## Rendering

- Curves of cumulative loss and regret, decision surfaces at the viewed step, weight bars over experts, trees growing.
- Missing: a stream strip (the points so far, drift marked) as a panel.

## Shared

- `shared/`: the stream base (shared with `shift-and-drift`, `time-series`, `online-experimentation`), the regret record.

## Problems

- New chapter (the old hub had `online/`); `active-learning` keeps its own chapter rather than nesting under online learning as the encyclopedia does.
