# eval-overhead: does a compiled evaluation plan pay?

The question (PLAN.md section 5.8 and WP9): the log joint of a bound spec is evaluated by `potentialEnergy(toProgram(bound))`, which runs the
closure through the handler stack, builds the distributions, records a trace and differentiates the result by reverse-mode autodiff.
A compiled plan would pre-resolve shapes and indices and evaluate the same density with fewer allocations, still through the
library's tensor ops and autodiff (no code generation, which is a non-goal). It is justified only if the closure path is the bottleneck.

Measured on 2026-11-10 with `docs/probprog/eval-timings.ts` (Node 24, one machine, median of five blocks; compare the ratios, not the
microseconds; **not measured in a browser**, where the same code runs 1.5 to 3 times slower by the plan's estimate, so the ratios carry over
and the absolute numbers grow):

| model | interpretive log density (us) | interpretive value+grad (us) | plan ceiling value+grad (us) | hand-written value+grad (us) | interpretive / ceiling | interpretive / hand |
| --- | --- | --- | --- | --- | --- | --- |
| logistic regression (50 rows, 3 features) | 42.9 | 115.5 | 79.8 | 1.0 | 1.45x | 112x |
| eight schools, centred (10 coordinates) | 40.0 | 78.5 | 58.9 | 0.3 | 1.33x | 276x |
| radon (6 counties x 8 houses, 10 coordinates) | 57.1 | 127.3 | 102.2 | 0.4 | 1.25x | 319x |

The three columns that matter:

- **interpretive** is what every engine runs today.
- **plan ceiling** is the same density written by hand with the library's own distributions and tensor ops under `target` (the values agree to
  the last digit), with no spec evaluation, no handler stack and no trace. A plan that stays on the library's tensor ops and autodiff cannot be
  faster than this, and a generic plan would be slower than a hand-tuned function.
- **hand-written** is plain TypeScript with an analytic gradient on `number`s, for scale.

As the rows grow (logistic regression, three features), the interpretive overhead disappears into the tensor work:

| logistic rows | interpretive value+grad (us) | plan ceiling value+grad (us) | interpretive / ceiling |
| --- | --- | --- | --- |
| 50 | 92.9 | 73.5 | 1.26x |
| 500 | 291.8 | 269.4 | 1.08x |
| 5000 | 2307.9 | 2419.8 | 0.95x |

## Decision: not built

The closure-interpretive path costs 1.25 to 1.45 times what the best plan on the same tensor ops could, and nothing at 5000 rows. The 100 to 300
times to plain numbers is the cost of the tensor library and autodiff themselves (an allocation per op, a tape per call), which a plan on the
same ops keeps; only code generation or specialised kernels would remove it, and that is out of scope. A CPU profile of the radon gradient
agrees: the time is spread over tensor allocation, `bind`, autodiff `backward`/`topological` and broadcasting, with the spec's own evaluation
(`alignTo`, `expandDistribution`) under 1 percent each. So `probprog/model/compile-plan.ts` is not written: a second evaluator, with its own
equality test and the maintenance of 35 operators and 22 families, for at most a 1.3 to 1.5 times gain on small models.

What does move browser budgets, from the same numbers: fewer gradient evaluations per effective sample (collapse the conjugate block:
[structure-collapse-efficiency.md](structure-collapse-efficiency.md) gets 40 times the effective sample size of the centred model per
gradient), minibatches on exchangeable plates (`structureReport(...).minibatchAxes`) and message passing. The time and iteration budget and the
anytime snapshots stay the user's parameters. The decision should be revisited only if a browser measurement shows an overhead the Node numbers
do not (for example allocation pressure from the trace), and the first thing to try then is to cheapen the trace, not a second evaluator.
