# Gates

What each merge was tested with, and whether it reached CI. The full local gate: every engine and other test in CI mode
(all cores, shared module graph, 30 s timeout), every `@example`, `make barrels-check`, `make agent-index-check`,
`make layers`, `make api-surface-check`, `make lint`, and `tsc` for engine, render, meta, site, pocs and the root. A
fixing agent runs only its own checks; the gate is run by the main session after merging. CI is the GitHub run.

| date       | commit     | what                              | local gate                                                                     | pushed | CI                                       | notes                                                              |
| ---------- | ---------- | --------------------------------- | ------------------------------------------------------------------------------ | ------ | ---------------------------------------- | ------------------------------------------------------------------ |
| 2026-11-03 | `1038fd26` | release 0.1.2                     | 26,076 tests, 9,567 examples, static and tsc clean (before the version commit) | yes    | green, run 41642601394; pack 41643666479 | released as `mdml-engine-v0.1.2`, `mdml-render-v0.1.2`             |
| 2026-11-07 | `44f5bd50` | P3.1 the type unification, merged | pending: the gate was started after the merge                                  | no     |                                          | the agent's scoped runs: 125 test files, 2,282 tests, 603 examples |
| 2026-11-07 | `b1c32597` | P3.2 the clearer pytree error     | scoped: 10 tests in two files, 8 examples, lint                                | no     |                                          |                                                                    |
