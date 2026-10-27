# Progress

The methods port (`docs/methods_port/`), phase by phase. Maintained by hand at each transition; times are local.

| phase                                                                                | status        | started          | ended            | notes                                                                                                                                                                         |
| ------------------------------------------------------------------------------------ | ------------- | ---------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pre: log contract, sessions, PoC                                                     | done          | 2026-10-26 09:00 | 2026-10-26 17:54 | commit 27a7a00b                                                                                                                                                               |
| 0: retire and keep                                                                   | done          | 2026-10-26 17:55 | 2026-10-26 18:09 | commit 9c6bb451; 18,966 tests, 6,542 examples green                                                                                                                           |
| 1: the tracker final + iterative base                                                | done          | 2026-10-26 18:09 | 2026-10-26 18:24 | commit 5a7d794e; 13,083 tests, 6,549 examples green                                                                                                                           |
| 2: algorithms home (linear, cluster first)                                           | done          | 2026-10-26 18:09 | 2026-10-26 18:27 | cluster d8a4e058 and linear 503925c3 merged (000ed2f7); rest moves as needed                                                                                                  |
| 3: four particulars + minimal hub piece                                              | done          | 2026-10-26 18:24 | 2026-10-26 18:38 | k-means, logistic regression, ridge, lasso path on `iterative`; hub = entries as data + one host; PoC removed (c24ef36f, 5fed91ec, merged)                                    |
| 3b: Gaussian mixture (asked 2026-10-26)                                              | done          | 2026-10-26 19:09 | 2026-10-26 19:09 | EM home to lib/mixture; `gmm` on the base; density probe; commit 043e615f                                                                                                     |
| 3c: hub playback fixes (asked 2026-10-26)                                            | done          | 2026-10-26 19:09 | 2026-10-26 19:54 | off-logs sessions keep checkpoints; rows always drawn; empty projections skipped; rate honoured; no flashing; GMM ellipses (b20e1bf3 merged 3ef22f32)                         |
| 3d: layouts as data, minimal defaults (asked 2026-10-27)                             | done          | 2026-10-27 08:57 | 2026-10-27 09:06 | entries declare `layout`, `mode` (live), `logs`; host draws the layout only; whole-tape view is a toggle (1fe98095)                                                           |
| 3e: click a curve to scrub every panel (asked 2026-10-27)                            | done          | 2026-10-27 09:46 | 2026-10-27 09:51 | click anywhere on a curves chart → nearest step; marker; projections follow the step (ca50f919)                                                                               |
| 3f: session UI into mdml-render, dataset-free entries, new panels (asked 2026-10-27) | done          | 2026-10-27 22:20 | 2026-10-27 23:05 | `react/session/` in render (one step cursor, `series`/`trajectory`/`tensor` panels, `band` overlay); entries may hold a plain `Method`; tracer stores nested arrays row-major |
| 4: trainer, agent and sweep bases                                                    | not requested |                  |                  |                                                                                                                                                                               |
| 5: procedure base and the viewer                                                     | not requested |                  |                  |                                                                                                                                                                               |
| 6: breadth (asked 2026-10-27)                                                        | in progress   | 2026-10-27 22:05 |                  | see the table below                                                                                                                                                           |

## 6. Breadth

Piece by piece, not deep: each entry is a thin particular or a plain method over the library, a few panels, one smoke test.

| piece                                                             | who          | status      | notes                                                                                     |
| ----------------------------------------------------------------- | ------------ | ----------- | ----------------------------------------------------------------------------------------- |
| `lib/vision`: classical computer vision, OpenCV-shaped            | agent (Opus) | in progress | worktree `agent-afa77cd434e278c8a`; built on `signal/image`; oracle group `oracle-vision` |
| Bayesian: sequential linear regression, variational mixture       | agent        | started     |                                                                                           |
| sequential: Kalman filter tracking, HMM by Baum-Welch             | agent        | started     |                                                                                           |
| search and tuning: A\* on grids, annealing, Bayesian optimisation | agent        | started     |                                                                                           |
| mixture of experts, ODE systems                                   | agent        | started     | `lib/mixture/experts.ts` new                                                              |
| probability: MCMC on a target, change of variables                | me           | started     |                                                                                           |
| linear algebra: power and QR iteration                            | me           | started     |                                                                                           |
| stochastic processes: diffusion paths                             | me           | started     |                                                                                           |
| CV entries in the hub (after the library merges)                  | later        |             |                                                                                           |

Last updated: 2026-10-27 23:05
