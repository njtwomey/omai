# Extensibility: what adding something costs

The test the design has to pass: "I decide later that I want the number of margin violations; adding it is trivial."
This file walks that case through, then tables every extension point with its cost, then says honestly what is not
trivial.

## 1. The worked case: margin violations on the SVM

A margin violation is a training row with `y_i f(x_i) < 1`. It is a function of the solver's state (the duals, from
which `f` follows) and the data. Three ways to want it, in increasing cost.

**As something to see over time, or to ask about at any step.** One line in the method's contract:

```ts
'smo/margin-violations': derived({
  description: 'training rows inside the margin',
  compute: (s) => data.x.filter((x, i) => y[i] * decision(s, x) < 1).length,
}),
```

That is the whole change. From then on:
- `logs: { 'smo/margin-violations': Log() }` records it every step; `Log({ every: 10 })` every tenth;
  `Log({ every: 'smo/converged' })` once at the end;
- `session.probe(t, 'smo/margin-violations')` answers for any recorded step, including steps nobody logged;
- the generic viewer draws it as a curve under the `smo` group, because it is a scalar path;
- `logs: 'all'` picks it up without being told; `'off'` never computes it;
- nothing in the base, the solver, the records, the views or the hub changed.

**As something only visible inside a step** (say, the violation of the _pair being optimised_ before the update, which
the state does not keep). One `ctx.log` where it happens and one contract line:

```ts
// in the solver's step, where the pair is chosen:
ctx.log('smo/pair-violation', pick.violation)
// in the contract:
'smo/pair-violation': scalar({ description: 'the KKT violation of the pair before its update' }),
```

`ctx.log` is a no-op unless subscribed, so the solver pays nothing for the line when nobody asks.

**As something to react to** (stop when violations fall below ten). One `event` line, one `ctx.emit`, and the caller
hooks it:

```ts
'smo/violations-below': event({ context: { count: 'the count' } }),       // contract
if (count < 10) ctx.emit('smo/violations-below', { count })               // in the step
logs: { 'smo/violations-below': On((h) => h.stop('few violations')) }     // the caller
```

Or, with no change to the method at all, the caller hooks `smo/iteration/end` (a base event) and computes the count
from `h.run` and a probe. The method need not anticipate the question.

## 2. Every extension point, with its cost

| I want to add…                                | what changes                                                                     | where                                  | cost                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------- |
| a quantity computable from the state          | one `derived(...)` line                                                          | the method's contract                  | trivial                                     |
| a quantity that exists only mid-step          | one `ctx.log` + one contract line                                                | the algorithm's step, the contract     | trivial                                     |
| a new lifecycle event                         | one `event(...)` line + one `ctx.emit`                                           | the algorithm, the contract            | trivial                                     |
| a hook (stop early, change a rate)            | `On(...)` in the subscription; to cross a worker, a named hook in `method.hooks` | the caller, or one entry in the method | trivial                                     |
| the same quantity on every method of a kind   | one line in the kind add-on's `probes`                                           | `kinds.<kind>.probes`                  | trivial, and it is one line for all of them |
| the same event on every method of a family    | one line in the base                                                             | the base                               | small                                       |
| a new schedule (every nth _epoch_, say)       | nothing: `every: ['training/epoch/end', n]`                                      | the subscription                       | nothing                                     |
| a new default view for a path kind            | one case in the generic viewer                                                   | render                                 | small                                       |
| a new structural record                       | a constructor in `lib/exposition` and a view in render                           | engine and render                      | the same as today                           |
| a new family (a sampler, an inference engine) | a base: the loop's events, serialise, standard probes                            | one file in the engine                 | medium, once                                |
| a new method in an existing family            | a particular: 10–40 lines                                                        | one file in methods                    | small                                       |

## 3. What is not trivial, and why

- **A quantity that needs something the state discarded.** If the solver throws away what you want (the candidate
  pairs it did _not_ choose, say), the state must keep it or the step must log it. That is a change to the solver,
  as it would be in any design; the contract makes it one line once the value exists.
- **Full tensors at every step on big models.** Logging `model/params/**` at `detail: 'full'` every step is the tape
  growing by the model size per step. The policy's byte budget stops it, but the honest answer is to subscribe at a
  schedule. The design makes the cost visible, not free.
- **Probing far from a checkpoint.** `probe(t, ...)` re-runs from the nearest checkpoint; with checkpoints every one
  percent of the budget that is cheap, but a method whose step is expensive (a path tracer) wants a denser
  checkpoint interval, which is a session option, not a method change.
- **States that do not serialise.** A search tree with parent links, a class instance, a `Map`: `stateToJson` refuses
  them, so such a method gets no `seek` and no `fork` until it offers its own `serialise`. Search is the case we have
  decided not to care about.
- **A hook that changes the computation** (a learning-rate schedule by hook) makes replay depend on the hook having
  run. Replay from checkpoints is still exact because the state carries the rate; but a _probe_ that re-runs from a
  checkpoint without the hook installed would diverge. The trainer base therefore keeps rate changes in the state, as
  `lrScale` is today.
