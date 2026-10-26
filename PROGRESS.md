# Progress

The methods port (`docs/methods_port/`), phase by phase. Maintained by hand at each transition; times are local.

| phase                                      | status        | started          | ended            | notes                                                                                                                                      |
| ------------------------------------------ | ------------- | ---------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| pre: log contract, sessions, PoC           | done          | 2026-10-26 09:00 | 2026-10-26 17:54 | commit 27a7a00b                                                                                                                            |
| 0: retire and keep                         | done          | 2026-10-26 17:55 | 2026-10-26 18:09 | commit 9c6bb451; 18,966 tests, 6,542 examples green                                                                                        |
| 1: the tracker final + iterative base      | done          | 2026-10-26 18:09 | 2026-10-26 18:24 | commit 5a7d794e; 13,083 tests, 6,549 examples green                                                                                        |
| 2: algorithms home (linear, cluster first) | done          | 2026-10-26 18:09 | 2026-10-26 18:27 | cluster d8a4e058 and linear 503925c3 merged (000ed2f7); rest moves as needed                                                               |
| 3: four particulars + minimal hub piece    | done          | 2026-10-26 18:24 | 2026-10-26 18:38 | k-means, logistic regression, ridge, lasso path on `iterative`; hub = entries as data + one host; PoC removed (c24ef36f, 5fed91ec, merged) |
| 3b: Gaussian mixture (asked 2026-10-26)    | done          | 2026-10-26 19:09 | 2026-10-26 19:09 | EM home to lib/mixture; `gmm` on the base; density probe; commit 043e615f                                                                  |
| 3c: hub playback fixes (asked 2026-10-26)  | in progress   | 2026-10-26 19:09 |                  | off-logs past, data always drawn, empty projections, playback rate, surface flashing; Sonnet agent                                         |
| 4: trainer, agent and sweep bases          | not requested |                  |                  |                                                                                                                                            |
| 5: procedure base and the viewer           | not requested |                  |                  |                                                                                                                                            |

Last updated: 2026-10-26 19:09
