# Algorithm gaps

A catalogue of the algorithms in `packages/engine/src/lib` that are iterative, sequential, online, sampling or search-based and are exposed only as a function returning a finished result: no exported `*Algorithm`, no estimator on a base. The reference case is Bayesian online changepoint detection: `bocpd` (`dynamics/filtering/changepoint.ts:219`) builds its Algorithm inline, runs it and returns the result, so nothing can step it on the tape or vary its slots. The capability is there; the Algorithm is not. Another agent is fixing BOCPD; it is row DY4 and not part of this work.

Audit of 2026-11-01 on branch `agent-c3d2e8-audit` (from `main` at `6638020e`). Read-only: no source was changed. The method, with every count and how it was obtained, is at the end.

How to read the rows. Locations are `file:line` under `packages/engine/src/lib/`. Log paths are suggestions in the `area/name` style of `lib/trace/contract.ts`. Priority: H = classic, widely taught, the step-by-step behaviour is the whole point; M = useful; L = minor. Effort: S = lift an existing loop (or export a private Algorithm) into an exported Algorithm; M = also a new estimator on a base, an exposition record or tests; L = needs new library pieces. The evidence column starts with `[lit]` when an `init`/`step` Algorithm literal already exists privately in the function (so the lift is an export plus a wrapper) or `[loop]` when only a plain loop exists, then says what was checked.

## Summary

| domain | files | exported Algorithm declarations | estimators on `iterative` | gap rows | high-priority rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| agents | 38 | 10 | 0 | 1 | 0 |
| anomaly | 3 | 5 | 4 | 0 | 0 |
| autodiff | 10 | 0 | 0 | 0 | 0 |
| base | 8 | 0 | 0 | 0 | 0 |
| causal | 15 | 4 | 0 | 0 | 0 |
| cluster | 11 | 16 | 13 | 1 | 0 |
| context | 3 | 0 | 0 | 0 | 0 |
| covariance | 3 | 4 | 4 | 0 | 0 |
| data | 29 | 1 | 0 | 0 | 0 |
| datasets | 66 | 0 | 0 | 0 | 0 |
| decomposition | 8 | 9 | 7 | 0 | 0 |
| distances | 3 | 0 | 0 | 0 | 0 |
| dists | 39 | 0 | 0 | 0 | 0 |
| dynamics | 80 | 36 | 0 | 10 | 4 |
| embedding | 5 | 2 | 0 | 0 | 0 |
| ensembles | 10 | 3 | 2 | 1 | 0 |
| environments | 17 | 0 | 0 | 0 | 0 |
| explain | 23 | 4 | 0 | 4 | 1 |
| exposition | 53 | 0 | 0 | 0 | 0 |
| fairness | 14 | 0 | 0 | 1 (+1 slot-rule) | 0 |
| foundations | 15 | 0 | 0 | 0 | 0 |
| gam | 16 | 6 | 2 | 0 | 0 |
| games | 23 | 9 | 0 | 3 | 1 |
| gp | 7 | 2 | 4 | 0 (+1 slot-rule) | 0 |
| graph | 24 | 26 | 0 | 8 | 3 |
| graphics | 16 | 3 | 0 | 0 | 0 |
| information | 24 | 14 | 0 | 2 | 0 |
| kernel-methods | 13 | 8 | 8 | 0 | 0 |
| kernels | 8 | 0 | 0 | 0 | 0 |
| learn | 62 | 44 | 26 | 4 (+1 slot-rule) | 1 |
| linear | 23 | 19 | 18 | 1 | 0 |
| logic | 31 | 26 | 0 | 4 | 0 |
| losses | 18 | 0 | 0 | 0 | 0 |
| manifold | 12 | 7 | 6 | 0 (+1 slot-rule) | 0 |
| metrics | 31 | 0 | 0 | 0 | 0 |
| mixture | 3 | 3 | 0 | 0 | 0 |
| nn | 62 | 12 | 0 | 3 | 1 |
| numerics | 90 | 17 | 0 | 8 | 3 |
| optim | 85 | 86 | 0 | 1 | 0 |
| patterns | 4 | 7 | 0 | 0 | 0 |
| pgm | 51 | 29 | 0 | 5 | 1 |
| problems | 8 | 0 | 0 | 0 | 0 |
| probprog | 30 | 5 | 0 | 1 | 0 |
| random | 11 | 0 | 0 | 0 | 0 |
| rl | 19 | 8 | 0 | 1 | 0 |
| rules | 7 | 6 | 5 | 0 | 0 |
| sampling | 23 | 4 | 0 | 3 | 1 |
| scene | 8 | 0 | 0 | 0 | 0 |
| search | 24 | 39 | 0 | 0 | 0 |
| signal | 63 | 4 | 0 | 3 | 0 |
| sketches | 8 | 7 | 0 | 1 | 0 |
| space | 6 | 0 | 0 | 0 | 0 |
| stats | 46 | 6 | 0 | 1 (+1 slot-rule) | 0 |
| stochastic | 22 | 15 | 0 | 2 | 0 |
| survival | 8 | 1 | 0 | 0 | 0 |
| svm | 5 | 0 | 2 | 0 (+1 slot-rule) | 0 |
| symbolic | 33 | 0 | 0 | 0 | 0 |
| systems | 18 | 1 | 0 | 0 | 0 |
| tensor | 26 | 0 | 0 | 0 | 0 |
| text | 42 | 7 | 0 | 3 | 1 |
| timeseries | 37 | 22 | 0 | 1 | 0 |
| trace | 17 | 0 | 0 | 0 | 0 |
| trees | 7 | 5 | 2 | 0 | 0 |
| tune | 16 | 1 | 0 | 0 | 0 |
| vision | 37 | 17 | 0 | 4 | 1 |
| worlds | 13 | 2 | 0 | 0 | 0 |
| total | 1590 | 562 | 103 | 77 (+6 slot-rule) | 18 |

The 77 gap rows cover 197 functions (the functions named in the first column of each row). Slot-rule rows (B1 to B6) and the base-adoption note (B0) are counted separately. Domains with no row were either covered (every iterative routine has an exported Algorithm) or contain no iterative routine of this kind; see "Not examined" for the ones read only by script.

What exists, for scale: 539 exported names end in `Algorithm` and 562 exported declarations return a type naming `Algorithm`; 611 object literals have both `init` and `step`. 103 exported declarations call `iterative` (118 call sites outside `learn/bases`), 44 call `defineEstimator`. `defineTrainer`, `defineAgentMethod` and `sweepMethod` have no call site anywhere outside `lib/learn/bases` (B0).

## Do these first

Twenty-five rows, chosen by priority and then effort. Every H row except DY4 is here.

1. **NU1** the quintessential step-by-step iteration, and not one root finder is an Algorithm; one mode table covers eight functions.
2. **GR1** Dijkstra, Bellman-Ford, Floyd-Warshall, A*: the most taught graph algorithms, function-only.
3. **DY1** EKF, UKF, EnKF: the Algorithm already exists privately in `runFilter`.
4. **DY2** particle filter: Algorithm exists privately in `runPf`; ESS and resampling are the whole lesson.
5. **DY5** heat, wave and advection steppers: Algorithm exists privately in `runSteps`; fields over time.
6. **NN1** adversarial attacks step by step; PGD is already exported, three siblings are not.
7. **EX1** counterfactual search by gradient steps; Algorithm exists privately.
8. **SA1** rejection and adaptive rejection sampling: the classic accept/reject picture.
9. **GA1** replicator dynamics: Algorithm exists privately; the projected sibling is exported.
10. **GR2** Kruskal and Prim: one edge at a time.
11. **NU3** power and inverse iteration: two short loops.
12. **NU4** adaptive Gauss-Kronrod quadrature: interval list and error as it refines.
13. **PG1** variable elimination: one variable and one new factor per step.
14. **VI1** seam carving: the seam and the shrinking image.
15. **LE1** fold-by-fold cross-validation, learning curves and searches: Algorithms exist privately; the sweep base has no user.
16. **GA2** iterated elimination of dominated strategies: round by round.
17. **TX1** CYK and Earley charts: the table filling is the point.
18. **GR3** max-flow and min-cost flow: augmenting paths and the residual graph.
19. **SO1** point-process simulators: six functions over one private Algorithm; thinning accept/reject.
20. **DY9** PID, relay autotune and MPC loops: Algorithms exist privately.
21. **EX3** anchors: beam search with precision bounds; Algorithm exists privately.
22. **LE3** pool-adjacent-violators: block merging.
23. **SG1** empirical mode decomposition: sifting; VMD is exported, EMD is not.
24. **CL1** HDBSCAN: tree, condense, select; DBSCAN and OPTICS are exported.
25. **B1** SVMs are fused to SMO; slot rule.

## Gaps by domain

### agents

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| AG1 | `runBandit` agents/bandits.ts:791 | one pull: arm, reward, agent state, cumulative regret | environment, agent (UCB, Thompson, epsilon-greedy), pulls, seed | `bandit/arm`, `bandit/reward`, `bandit/regret`, `bandit/estimate` | M | S | [loop] hand-written act/step/learn loop (read in full); `trainingAlgorithm` (rl) may already serve it, unchecked |

### cluster

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| CL1 | `hdbscan` cluster/density.ts:676 (with `condenseTree`:494, `selectClusters`:568) | edges of the mutual-reachability MST merged into a single-linkage tree, then condensed and scored: tree, condensed rows, stabilities | minClusterSize, minSamples, metric, selection (eom or leaf), alpha | `tree/merge`, `tree/condense`, `select/stability`, `labels/strength` | M | M | [loop] five private steps called in sequence, no runAlgorithm; `dbscanAlgorithm` and `opticsAlgorithm` are exported beside it |

### dynamics

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| DY1 | `extendedKalmanFilter`:113, `unscentedKalmanFilter`:274, `ensembleKalmanFilter`:342 (filtering/nonlinear.ts, private `runFilter`:59) | one observation per step: mean, covariance, log-likelihood (EnKF: the ensemble) | dynamics f and observation h, Q and R, linearisation (Jacobian by autodiff), sigma-point scheme and alpha/beta/kappa, ensemble size and inflation | `filter/mean`, `filter/cov`, `filter/innovation`, `filter/logLikelihood`, `filter/sigmaPoints`, `filter/ensemble` | H | S | [lit] `runFilter` builds `Algorithm<S>` inline, runs it, returns `{means, covariances}`. Pattern: `kalmanFilterAlgorithm` (kalman.ts:214) is exported |
| DY2 | `bootstrapParticleFilter`:288, `auxiliaryParticleFilter`:321, `particleSmoother`:354 (filtering/particle.ts, private `runPf`:168) | one observation per step: particles, log-weights, ESS, evidence, resample flag, ancestry | model (initial, transition, log-likelihood, predict), particle count, resampling scheme, ESS threshold, proposal (bootstrap or auxiliary) | `pf/ess`, `pf/logEvidence`, `pf/resampled`, `pf/weights`, `pf/particles` | H | S | [lit] `runPf` builds `Algorithm<PfState>` inline. Also see duplicate D2 |
| DY3 | `pelt`:358, `binarySegmentation`:418 (filtering/changepoint.ts) | PELT: end index b, F[b], admissible set, pruned set. Binary segmentation: segment list and the best split of each | cost (l2 and others), penalty or number of changepoints, minSize, jump | `pelt/F`, `pelt/admissible`, `pelt/pruned`, `binseg/split`, `binseg/gain` | M | M | [loop] exact loops read in full; no Algorithm. Needs a changepoint exposition record (series with segments) if none exists. Sits beside BOCPD (DY4) |
| DY4 | `bocpd` filtering/changepoint.ts:219 (reference example, being fixed by another agent) | run-length posterior per observation | data, hazard, conjugate model parameters | `bocpd/runLength`, `bocpd/map`, `bocpd/changeProbability` | H | S | [lit] inline Algorithm inside `bocpd`; `exactBocpd` datasets/changepoints.ts:152 is the twin (D1) |
| DY5 | `heat`:159, `heat1d`:218, `wave1d`:264, `advection1d`:325 (pde/finite-difference.ts, private `runSteps`:70) | one time step: field u(t), previous field (leapfrog), finite check | scheme, dt, grid, boundary conditions, initial condition, CFL/diffusion number | `pde/t`, `pde/u`, `pde/norm`, `pde/max`, `pde/failed` | H | S | [lit] `runSteps` builds `Algorithm<TimeStepState>` inline (stops with status `diverged`); results returned as a whole `PdeSolution` |
| DY6 | `poisson`:410 and private `multigridPoisson`:555 (pde/finite-difference.ts) | relaxation sweeps or V-cycles: iterate, residual | smoother, cycles, levels, tolerance | `mg/residual`, `mg/level`, `mg/iterate`, `mg/cycle` | M | S | [lit] inline Algorithms in both. `multigridAlgorithm` exists (numerics/sparse/multigrid.ts:382), so this may be a second implementation (check before lifting) |
| DY7 | `heatFem` pde/finite-element.ts:354, `conservationLaw` pde/finite-volume.ts:171, `fokkerPlanck` pde/fokker-planck.ts:161, `spectralEtdrk4` pde/spectral.ts:343 | one time step of the semi-discrete system: field or density, time | mesh and element order, flux and limiter, integrator, modes (spectral), dt | `pde/t`, `pde/u`, `pde/mass`, `pde/entropy`, `pde/error` | M | S | [lit] each builds its own inline Algorithm and runs it |
| DY8 | `solveBvp` ode/boundary.ts:98 (with `solveNewton` ode/_bvp.ts:213), `shooting`:306, `multipleShooting`:349 | collocation Newton iterations and mesh refinements; shooting: root-find on the unknown initial slope | tolerance, initial mesh, root solver, damping | `bvp/residual`, `bvp/meshSize`, `bvp/damping`, `shoot/miss`, `shoot/slope` | M | M | [loop] loops only; no Algorithm in boundary.ts or _bvp.ts |
| DY9 | `simulatePid` control/pid.ts:126, `relayAutotune`:298, `recedingHorizon` control/predictive.ts:221 | one control period: plant state, controller state, output, input, cost | plant (LTI), gains and anti-windup, setpoint and disturbance functions, planner (linear or nonlinear MPC), horizon, constraints, relay amplitude | `pid/error`, `pid/u`, `pid/integral`, `mpc/cost`, `mpc/plan`, `plant/x` | M | S | [lit] each builds an inline Algorithm (read in full for `simulatePid`, `recedingHorizon`) |
| DY10 | `place` control/placement.ts:281 (Kautsky-Nichols pole placement) | eigenvector conditioning sweeps over the poles | method (YT or KNV0), maxiter, tolerance | `place/conditioning`, `place/sweep`, `place/poles` | L | M | [loop] loops read via head only |

### ensembles

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| EN1 | `cyclicBoosting` ensembles/cyclic.ts:175 and `explainableBoosting`:360 | round-robin boosting of one feature at a time: per-feature shape scores, loss | learning rate, rounds, outer bags, binning, interactions, loss and link | `ebm/round`, `ebm/feature`, `ebm/loss`, `ebm/shape` | M | S | [lit] 160-line inline Algorithm in `cyclicBoosting`; the exports return `CyclicBooster` and `GlassBoxModel`, not an estimator on a base. Exposition: additive shape functions at round t |

### explain

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| EX1 | `wachterCounterfactual` explain/counterfactuals.ts:127, `diverseCounterfactuals`:248 | gradient steps on the counterfactual: point, prediction loss, distance, lambda | distance (L1/MAD), target class or probability, lambda schedule, optimiser, actionability | `cf/x`, `cf/loss`, `cf/validity`, `cf/distance`, `cf/lambda` | H | S | [lit] inline Algorithm in both. Exposition: the trail of the point through feature space |
| EX2 | `growingSpheres`:644, `faceCounterfactual`:387, `prototypeCounterfactual`:954 (explain/counterfactuals.ts) | shrinking radius and layer growth; Dijkstra over a density graph; prototype-guided search | radius schedule, k, density estimate, prototype count | `gs/radius`, `gs/layer`, `face/path`, `proto/distance` | M | M | [loop] loops read via head; no Algorithm |
| EX3 | `anchors` explain/surrogates.ts:370 | beam search over predicate sets: current rule, precision and coverage bounds, samples drawn | precision threshold, delta, beam width, perturbation sampler | `anchor/rule`, `anchor/precision`, `anchor/coverage`, `anchor/samples`, `anchor/beam` | M | S | [lit] 172-line function with an inline Algorithm, runs it |
| EX4 | `samplingShapley` explain/shapley.ts:303, `sage`:1015, `dataShapley` explain/valuation.ts:197, `permutationImportance` explain/inspection.ts:68 | one permutation or repeat: running mean of marginal contributions and standard error | value function or model, permutations, antithetic pairs, truncation tolerance (TMC), loss | `shapley/phi`, `shapley/stderr`, `shapley/permutation`, `shapley/marginal` | M | S | [lit] inline Algorithms. `sampledShapley` in games (GA3) is the same shape |

### fairness

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| FA1 | `exponentiatedGradientReduction` fairness/in-processing.ts:274, `gridReduction`:528 | game rounds: multipliers, learner best response, ensemble of classifiers, violation (grid: one multiplier vector per fit) | constraint moment (parity, equalised odds), eps, eta, base estimator, maxIter | `eg/violation`, `eg/lambda`, `eg/error`, `eg/classifier` | M | M | [loop] 181-line loop `for t < maxIter`; no Algorithm |

### games

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| GA1 | `replicatorDynamics` games/evolutionary.ts:138, `asymmetricReplicatorDynamics`:166 (private `integration`:84) | one time point: population state x, time, path | payoff matrix, mutation matrix, x0, time points, maxStep, integrator (fixed RK4 here, could be a solveIvp stepper) | `replicator/x`, `replicator/time`, `replicator/fitness`, `replicator/meanFitness` | H | S | [lit] `integration` is an Algorithm built privately (read in full); the sibling `projectedReplicatorDynamicsAlgorithm`:227 is exported |
| GA2 | `iteratedElimination` games/normal-form/dominance.ts:189 | one round: dominated actions removed, game restricted | strict or weak dominance, pure or mixed dominating strategies | `elim/round`, `elim/removed`, `elim/remaining` | M | S | [loop] `for round` loop; no Algorithm |
| GA3 | `sampledShapley` games/cooperative.ts:321 | one random permutation: marginal contributions, running mean | characteristic function, permutations, antithetic | `shapley/phi`, `shapley/stderr`, `shapley/permutation` | L | S | [loop] same shape as EX4 |

### graph

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| GR1 | `dijkstra` graph/paths.ts:92, `bellmanFord`:118, `aStar`:176, `floydWarshall`:231, `johnson`:288 | Dijkstra/A*: one settled node (frontier, distances, predecessors); Bellman-Ford: one relaxation round; Floyd-Warshall: one intermediate k | weights, source and target, heuristic (A*), negative-edge handling | `sp/settled`, `sp/relaxed`, `sp/frontier`, `sp/distance`, `sp/negativeCycle` | H | S | [loop] read `bellmanFord` loop; only `bfsAlgorithm` and `dfsAlgorithm` are exported in graph/traversal.ts. `aStar` duplicates `aStarSearchAlgorithm` (D5) |
| GR2 | `minimumSpanningTree` graph/trees.ts:112 (Kruskal and Prim) | one edge considered: forest, components, chosen or rejected edge | algorithm (Kruskal or Prim), weights, root | `mst/edge`, `mst/accepted`, `mst/weight`, `mst/components` | H | S | [loop] no Algorithm in trees.ts |
| GR3 | `maxFlow` graph/flows.ts:192, `minCut`:228, `minCostFlow`:283, `maxFlowMinCost`:399 | one augmentation: residual graph, flow, augmenting path, cut | method (Edmonds-Karp, Dinic), capacities, costs, demands | `flow/augment`, `flow/value`, `flow/residual`, `flow/path`, `flow/cut` | H | M | [loop] 97-line `minCostFlow` read via head; no Algorithm in flows.ts |
| GR4 | `bipartiteMatching` graph/flows.ts:442 (Hopcroft-Karp) | one phase: BFS layers, augmenting paths, matching size | left set, graph | `match/phase`, `match/size`, `match/augmenting`, `match/layers` | M | M | [loop] two `while` loops; `hungarianAlgorithm` and `galeShapleyAlgorithm` exist in optim |
| GR5 | `stronglyConnectedComponents` graph/components.ts:190, `articulationPoints`:342, `bridges`:360, `biconnectedComponents`:385 | DFS: stack, discovery and low-link arrays, finished components | algorithm (Tarjan or Kosaraju), graph | `dfs/discover`, `dfs/low`, `dfs/component`, `dfs/stack` | M | M | [loop] no Algorithm; `dfsAlgorithm` exists and is the shape to follow |
| GR6 | `topologicalSort` graph/traversal.ts:333, `greedyColouring` graph/colouring.ts:99, `maximalCliques`:260 (Bron-Kerbosch), `maximalIndependentSet`:217 | one node ordered or coloured; one Bron-Kerbosch call with its R, P and X sets | strategy (largest first, DSATUR), pivot rule, graph | `order/node`, `colour/assign`, `clique/R`, `clique/P`, `clique/X` | M | S | [loop] no Algorithm |
| GR7 | `betweennessCentrality` graph/centrality.ts:158, `edgeBetweennessCentrality`:196 (Brandes) | one source: BFS, dependency accumulation, running scores | normalisation, weights, sampled sources | `between/source`, `between/score`, `between/delta` | L | S | [loop] `girvanNewmanAlgorithm` is exported |
| GR8 | `kShortestPaths` graph/paths.ts:399 (Yen), `vf2` graph/isomorphism.ts:203 | Yen: one spur path; VF2: partial mapping and backtrack | k, weights; induced or monomorphism | `yen/spur`, `yen/candidates`, `vf2/mapping`, `vf2/backtrack` | L | M | [loop] no Algorithm |

### information

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| IN1 | `lz77Encode` information/dictionary-coding.ts:40, `lempelZivComplexity` information/complexity.ts:114 | one parse position: longest match, emitted triple or phrase | window, max length, lookahead | `lz/position`, `lz/match`, `lz/phrase`, `lz/ratio` | L | S | [loop] while loops; `huffmanAlgorithm`, `arithmeticEncoderAlgorithm` and `prefixEncoderAlgorithm` are exported beside |
| IN2 | `rsDecode` information/reed-solomon.ts:230 | Berlekamp-Massey step: error-locator polynomial, discrepancy | code (n, k), erasures, field | `rs/syndrome`, `rs/locator`, `rs/discrepancy`, `rs/errors` | L | M | [loop] `for step` loop inside a 78-line function |

### learn

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| LE1 | `crossValidate` learn/model-selection/cross-validation.ts:178, `crossValPredict`:289, `learningCurve`:370, `permutationTestScore`:532, `evaluateCandidates` model-selection/search.ts:164, `nestedCrossValidation`:411, `bootstrap632` model-selection/criteria.ts:257 | one fold, size, permutation or candidate: scores so far | estimator, splitter, scorers, candidate space and sampler, repeats, train sizes | `cv/fold`, `cv/trainScore`, `cv/testScore`, `search/candidate`, `search/best`, `curve/size` | H | S | [lit] each builds an inline Algorithm (read in full for `crossValidate`). `sweep` and `race` in learn/bases/sweep.ts have no users, so these are the natural first ones |
| LE2 | `streamActiveLearning` learn/active-learning.ts:1136, `adaptiveConformalInference` learn/conformal.ts:665, `crossConformalRegressor`:260 | one stream item: query decision, labelled set, model; online miscoverage level alpha_t | query strategy, budget, learner, gamma (ACI), coverage target | `stream/queried`, `stream/accuracy`, `aci/alpha`, `aci/covered`, `aci/width` | M | S | [lit] inline Algorithms; `activeLearningAlgorithm` exists for the pool case |
| LE3 | `poolAdjacentViolators` learn/calibration.ts:77, `isotonicCalibration`:185 | one pooling merge: block stack, pooled means | weights, increasing or decreasing | `pava/blocks`, `pava/merge`, `pava/violations` | M | S | [loop] while loops over a block stack |
| LE4 | `rfe`:150, `rfecv`:208 learn/feature-selection-wrappers.ts (private `eliminate`:88) | one elimination round: ranking, features left, score | estimator, step, number of features, cross-validation | `rfe/ranking`, `rfe/remaining`, `rfe/score` | M | S | [lit] `eliminate` builds an inline Algorithm; the estimators are `defineEstimator`, not on the iterative base |

### linear

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| LI1 | `maximiseEvidence` linear/bayesian.ts:343 | MacKay fixed-point iteration: alpha, beta, effective parameters, evidence | initial alpha and beta, tolerance | `evidence/alpha`, `evidence/beta`, `evidence/gamma`, `evidence/logEvidence` | M | S | [loop] `while (iterations < maxIter)`; `bayesianLinearAlgorithm`:265 takes fixed alpha and beta |

### logic

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| LO1 | `primeImplicants` logic/propositional.ts:797, `minimalDnf`:864, `minimalCnf`:928 | Quine-McCluskey round: cubes merged, primes found; then cover selection | formula, cover method | `qm/round`, `qm/cubes`, `qm/primes`, `qm/cover` | L | S | [loop] `while (cubes.size > 0)` |
| LO2 | `sampleSat` logic/markov-logic.ts:693 | one flip (WalkSAT move or simulated-annealing move): assignment, unsatisfied clauses | noise, temperature, number of samples | `sat/flips`, `sat/unsat`, `sat/move`, `sat/sample` | M | M | [loop] 107-line function; `walksatAlgorithm` is exported |
| LO3 | `findS` logic/ilp/version-spaces.ts:82, `candidateElimination`:259 | one example: specific and general boundaries | hypothesis language, examples | `vs/specific`, `vs/general`, `vs/example` | M | S | [loop] `for (const e of examples)`; classic textbook trace |
| LO4 | `unify` logic/unify.ts:189, `mgu`:286 | one decompose/eliminate step: equation set, substitution | occurs check, terms | `unify/equations`, `unify/substitution`, `unify/rule` | L | S | [loop] stepwise rewriting inside a function |

### nn

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| NN1 | `carliniWagnerL2` nn/adversarial.ts:408, `deepFool`:502, `squareAttack`:642 | one attack step: perturbation, margin, norm, success flag | model, norm, epsilon, step size, optimiser (CW uses Adam), steps, targeted, c | `attack/delta`, `attack/loss`, `attack/margin`, `attack/norm`, `attack/success` | H | S | [loop] `pgdAlgorithm`:309 is exported (and `pgd`:241 wraps it); these three are not |
| NN2 | `iterativeMagnitudePruning` nn/pruning.ts:312 | one prune-rewind-retrain round: masks, sparsity, loss | final sparsity, rounds, global or layerwise, rewind step, train callback | `prune/round`, `prune/sparsity`, `prune/loss`, `prune/mask` | M | M | [loop] `for r <= rounds`; takes a training function, so it would nest the trainer |
| NN3 | `gptq` nn/quantise.ts:303, `awq`:418 | one column quantised with error compensation: weight matrix, error | bits, group size, damping, calibration data | `quant/column`, `quant/error`, `quant/weights` | L | S | [loop] `for j < cols` |

### numerics

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| NU1 | `bisect` numerics/roots/scalar.ts:72, `brentq`:112, `ridder`:186, `toms748`:240, `newton`:353, `secant`:412, `halley`:436, `fixedPoint` roots/systems.ts:208 | one iterate: bracket or point, f(x), step | method (a mode table, as `glmSolvers`), bracket or start, tolerance, derivatives | `root/x`, `root/fx`, `root/bracketWidth`, `root/step`, `root/method` | H | S | [loop] no Algorithm anywhere in numerics/roots. Minimisation has `brentAlgorithm`, `goldenAlgorithm` in optim |
| NU2 | `fsolve` numerics/roots/systems.ts:74, `broyden`:129, `root`:298 | one Newton or Broyden step: x, residual norm, Jacobian estimate | method, line search, Jacobian (autodiff or Broyden update) | `root/x`, `root/residual`, `root/jacobianAge`, `root/lineSearch` | M | S | [loop] `for i` plus backtracking `while` |
| NU3 | `powerIteration` numerics/linalg/eig.ts:176, `inverseIteration`:209 | one iterate: vector, Rayleigh estimate, residual | matrix, shift, start vector, tolerance | `eig/vector`, `eig/value`, `eig/residual`, `eig/angle` | H | S | [loop] loop headers and calls read; `powerPcaAlgorithm` (decomposition/pca.ts:267) is the PCA-specific power method |
| NU4 | `quad` numerics/quadrature/adaptive.ts:103 (globally adaptive Gauss-Kronrod) | one bisection: interval list with estimates and errors, total error | rule (7-15), tolerances, limit, integrand | `quad/estimate`, `quad/error`, `quad/intervals`, `quad/split` | H | S | [loop] `while (!converged && intervals.length < limit)` |
| NU5 | `remez` numerics/approximation/best.ts:55, `chebfun` numerics/approximation/chebyshev.ts:220 | one exchange: reference points, levelled error; chebfun: degree doubling | degree (m, n), interval, tolerance | `remez/reference`, `remez/error`, `remez/alternation`, `chebfun/degree` | L | M | [loop] see also duplicate D7 |
| NU6 | `arnoldi` numerics/sparse/eigen.ts:117, `eigsh`:194, `svds`:304 | Arnoldi step or restart: basis, Ritz values, residuals | k, which, block size, tolerance | `krylov/basis`, `krylov/ritz`, `krylov/residual`, `krylov/restart` | M | S | [loop] `lanczosAlgorithm`, `gmresAlgorithm` are exported; these are not |
| NU7 | `hnswIndex` numerics/spatial/graphs.ts:73 | one insertion: layer, entry point, neighbour lists | M, efConstruction, level multiplier, distance | `hnsw/insert`, `hnsw/layer`, `hnsw/neighbours` | L | M | [loop] `nnDescentAlgorithm` exists for the other graph index |
| NU8 | `solveDare` numerics/linalg/equations.ts:185, `univariateSpline` numerics/interpolate/smoothing.ts:174 | doubling iteration; knot insertion until residual below s | tolerance; smoothing factor s | `dare/residual`, `dare/iterate`, `spline/knots`, `spline/residual` | L | S | [loop] |

### optim

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| OP1 | `backtracking` optim/line-search.ts:110, `hagerZhang` optim/line-search-interval.ts:266, `bracket` optim/scalar.ts:54 | one trial step: alpha, phi(alpha), Armijo and curvature flags | c1, c2, interpolation, initial step | `ls/alpha`, `ls/phi`, `ls/armijo`, `ls/curvature`, `ls/bracket` | M | S | [loop] used by the optimisers; a nested child run (`trace/nesting.ts`) would show it inside a step |

### pgm

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| PG1 | `variableElimination` pgm/inference/elimination.ts:142, `maxProductElimination`:212 | one variable eliminated: factors multiplied, summed or maximised out, new factor size | query and evidence, elimination order (given or min-fill or min-neighbours), semiring | `ve/eliminated`, `ve/factorSize`, `ve/width`, `ve/factors` | H | S | [loop] `eliminateAll` loop; `junctionTreeAlgorithm`, `beliefPropagationAlgorithm` are exported |
| PG2 | `eliminationOrder` pgm/triangulation.ts:89, `maximumCardinalitySearch`:181, `triangulate`:144, `treewidth`:270 | one vertex chosen: fill-in edges, clique sizes | heuristic (min-fill, min-degree, MCS), graph | `order/vertex`, `order/fill`, `order/width` | L | S | [loop] `while (remaining.size > 0)` |
| PG3 | `rejectionSample` pgm/inference/sampling.ts:37, `likelihoodWeighting`:68 | one sample: accepted or weight, effective sample size | evidence, number of samples, proposal order | `is/weight`, `is/ess`, `is/accepted`, `is/estimate` | M | S | [loop] `gibbsSamplingAlgorithm` is exported next to them |
| PG4 | `chowLiu` pgm/learning/structure.ts:80, `directLingam` pgm/causal/discovery.ts:328 | Chow-Liu: one edge added to the max-MI tree; DirectLiNGAM: one variable placed in causal order, residualised | MI estimator, root; independence measure | `cl/edge`, `cl/mi`, `lingam/exogenous`, `lingam/order` | M | S | [loop] `while (remaining.length > 0)`; `hillClimbAlgorithm`, `gesAlgorithm`, `pcAlgorithm` exist |
| PG5 | `fci` pgm/causal/fci.ts:353, `meekRules` pgm/learning/equivalence.ts:210, `pdagToDag`:313, `cpdag`:284 | FCI: orientation rule application after the PC skeleton; Meek: rule firings to a fixed point | independence test, alpha, max conditioning set, rules enabled | `fci/rule`, `fci/oriented`, `fci/sepset`, `meek/rule` | M | M | [loop] `fci` calls `runAlgorithm(pcAlgorithm)` for the skeleton and runs the rest in plain loops |

### probprog

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| PP1 | `mcmc` probprog/engines/nuts.ts:146, `inferDiscrete`:48, `findInitialParams` probprog/density.ts:621 | chains by draws: kernel state, adaptation, divergences | kernel (NUTS, HMC, ...), chains, warm-up, draws, target acceptance, initial strategy | `mcmc/chain`, `mcmc/stepSize`, `mcmc/accept`, `mcmc/divergent`, `mcmc/energy` | M | S | [loop] the driver loops chains in a function; `samplerAlgorithm` (sampling) exists and `gibbsAlgorithm`, `sviAlgorithm`, `particleFilterAlgorithm` are exported beside |

### rl

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| RL1 | `policyEvaluation` rl/mdp.ts:341, `rolloutPlanner` rl/planning.ts:399 | one Bellman sweep; one rollout batch | policy, gamma, tolerance; rollouts, horizon | `eval/sweep`, `eval/delta`, `plan/rollout`, `plan/value` | L | S | [loop] `valueIterationAlgorithm` and `policyIterationAlgorithm` are exported |

### sampling

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| SA1 | `rejectionSample` sampling/rejection.ts:57, `adaptiveRejectionSample`:136, `ratioOfUniforms`:313 | one proposal: x, u, accept or reject; ARS: envelope abscissae | proposal and bound M, target density, envelope refinement, grid | `reject/proposal`, `reject/accepted`, `reject/rate`, `reject/envelope` | H | S | [loop] `while (accepted < n && proposals < budget)`; MCMC kernels run through `samplerAlgorithm`, exact samplers do not |
| SA2 | `annealedImportanceSampling` sampling/smc.ts:602 | one rung of the temperature ladder: particles, log-weights, ESS | ladder, transition kernel, particle count | `ais/beta`, `ais/ess`, `ais/logWeight`, `ais/logZ` | M | S | [loop] `smcAlgorithm` is exported |
| SA3 | `findReasonableStepSize` sampling/hamiltonian/adaptation.ts:148, `windowAdaptation`:387, `randomWalkAdaptation` sampling/metropolis.ts:355 | doubling and halving of the step size; warm-up windows | target acceptance, window schedule, regularisation | `adapt/stepSize`, `adapt/window`, `adapt/accept` | L | S | [lit] adaptations are inline objects consumed by the sampler |

### signal

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| SG1 | `emd` signal/decompositions.ts:142, `eemd`:181, `ceemdan`:220 | one sift: candidate IMF, envelopes, stopping statistic, residue | stopping rule (SD, S-number), boundary mirroring, spline type, ensemble size and noise | `emd/imf`, `emd/sift`, `emd/sd`, `emd/residue`, `emd/envelope` | M | S | [loop] `vmdAlgorithm`:339 is exported; EMD is not |
| SG2 | `dtw` signal/similarity.ts:63 | one row of the cost matrix: accumulated cost, warping path | step pattern, window, distance | `dtw/row`, `dtw/cost`, `dtw/path` | L | M | [loop] `for i, for j` table fill |
| SG3 | `felzenszwalb` signal/image/segmentation.ts:442 | one edge merged in weight order: components, thresholds | scale, sigma, min size | `seg/merge`, `seg/components`, `seg/threshold` | L | S | [loop] `watershedAlgorithm` (signal/image/segmentation.ts:343) is exported in the same file |

### sketches

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| SK1 | `gkSummary` sketches/quantiles.ts:77, `kllSketch`:496, `p2Push` stats/running.ts:226 | one item inserted: tuples or compactors, markers, error bound | epsilon or k, stream | `sketch/size`, `sketch/compress`, `sketch/error`, `sketch/quantile` | L | S | [loop] seven other sketches (HyperLogLog, count-min, space-saving, Bloom, cuckoo, reservoir, skip list) have Algorithms |

### stats

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| ST1 | `mmdTest` stats/tests/kernel.ts:200, `hsicTest`:325 (private `permutationAlgorithm`:153), `monteCarloTest` stats/resampling.ts:597, `steinGoodnessOfFit` kernel-methods/mean-embedding.ts:516 (private `bootstrapAlgorithm`:464), `simulatePower` stats/power.ts:258 | one permutation, bootstrap or simulated dataset: null statistic, running p-value | statistic, kernel and bandwidth, permutations, alternative | `null/statistic`, `null/pvalue`, `null/count` | L | S | [lit] private Algorithms in four of them; `permutationTestAlgorithm`, `bootstrapAlgorithm` (stats) exist and should be reused |

### stochastic

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| SO1 | `poissonProcess` stochastic/point-processes.ts:116, `inhomogeneousPoisson`:146, `renewalProcess`:212, `hawkesProcess`:309, `coxProcess`:603, `markedPoissonProcess`:670 (private `eventAlgorithm`:56) | one candidate event: time, thinning accept or reject, intensity, count | process family and parameters, horizon, thinning bound, seed | `pp/time`, `pp/intensity`, `pp/accepted`, `pp/count` | M | S | [lit] `eventAlgorithm` is private; only `hawkesEmAlgorithm`:422 is exported |
| SO2 | `sampleDpp` stochastic/dpp.ts:251, `sampleKDpp`:276, `greedyMap`:313 | phase 1: eigenvectors kept; phase 2: one item chosen, basis projected | L-ensemble or kernel, k, eigendecomposition | `dpp/kept`, `dpp/item`, `dpp/probability`, `dpp/basis` | M | S | [loop] |

### text

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| TX1 | `cyk` text/parsing.ts:163, `earley`:239 | CYK: one span length of the chart; Earley: one predict, scan or complete | grammar, start symbol, CNF conversion, tree enumeration limit | `chart/span`, `chart/cell`, `earley/predict`, `earley/scan`, `earley/complete` | H | M | [loop] `for len` and `for changed` loops; `pcfgEmAlgorithm` exists for the trained-grammar case |
| TX2 | `editCounts` text/similarity.ts:103, `levenshtein`:42, `longestCommonSubsequence`:391, `sellers` text/approximate-search.ts:321 | one row of the DP table, then the traceback | costs, transpositions, free start (Sellers) | `dp/row`, `dp/cell`, `dp/path` | L | M | [loop] table fills |
| TX3 | `trainPunkt` text/sentences.ts:236, `bpeTokenize` text/subword/bpe.ts:521, `wordPieceTokenize` text/subword/wordpiece.ts:75 | abbreviation scoring pass; merges applied in rank order; greedy longest match | training text, merge table, vocabulary | `tok/merge`, `tok/piece`, `punkt/score` | L | S | [loop] `bpeTrainingAlgorithm` and `unigramTrainingAlgorithm` exist for training |

### timeseries

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| TS1 | `selectArOrder` timeseries/arma-fit.ts:108, `selectCointegrationRank` timeseries/cointegration.ts:284 | one order or one rank tested: criterion, statistic | criterion (AIC, BIC), max order, test level | `select/order`, `select/criterion`, `select/statistic` | L | S | [loop] sequential model selection as sweeps |

### vision

| id | algorithm and location | iterates | slots | log paths | pri | eff | evidence and dependencies |
| --- | --- | --- | --- | --- | :-: | :-: | --- |
| VI1 | `seamCarve` vision/seams.ts:141 (with `findVerticalSeam`:59) | one seam removed: energy map, cumulative map, seam, image width | energy function, direction, target size, protected or removed masks | `seam/energy`, `seam/cost`, `seam/path`, `seam/width` | H | S | [loop] `while (r.w > tw)` in a function; no Algorithm |
| VI2 | `camShift` vision/tracking.ts:300 (and the mean-shift window it calls) | one window update: centre, size, orientation, moments | histogram back-projection, termination criteria | `track/window`, `track/moment`, `track/shift` | M | S | [loop] `kltAlgorithm`, `runningAverageAlgorithm` exist in vision |
| VI3 | `softNms` vision/boxes.ts:263, `houghLinePeaks` vision/edges.ts:276, `regionGrowing` vision/segmentation.ts:265, `floodFill` vision/components.ts:130 | one box kept, one peak removed, one pixel absorbed | IoU threshold and decay, accumulator threshold, similarity bound, connectivity | `greedy/pick`, `greedy/suppress`, `region/pixel`, `region/mean` | L | S | [loop] queue and greedy loops |
| VI4 | `minEnclosingCircle` vision/fitting.ts:228 (Welzl), `approxPolyDP` vision/contours.ts:378 (Douglas-Peucker) | incremental circle; recursive split | epsilon, closed or open curve | `geom/circle`, `geom/split`, `geom/points` | L | S | [loop] |

## Near-duplicate implementations (a)

One algorithm in two places. The cost is that a fix or an Algorithm lands in one copy only. Found by name; for each pair the signature, return type and size of both sides were compared, and the bodies were read only for D3 (the heads) and D14.

| id | where | note |
| --- | --- | --- |
| D1 | `bocpd` dynamics/filtering/changepoint.ts:219 and `exactBocpd` datasets/changepoints.ts:152 | Two BOCPDs, as named in the brief. The aim is one implementation, exported as one Algorithm (DY4) |
| D2 | `bootstrapParticleFilter` dynamics/filtering/particle.ts:288 and `particleFilterAlgorithm` probprog/engines/smc.ts (exported) | Particle filter twice, one as an exported Algorithm |
| D3 | `alphaBeta` agents/planning.ts:407 (56-line hand loop) and `alphaBeta`/`alphaBetaAlgorithm` search/adversarial.ts:440 | Game-tree alpha-beta twice |
| D4 | `kalmanSmoother` probprog/engines/gaussian.ts:291 (109 lines) and the Kalman/RTS code in dynamics/filtering/kalman.ts; `filterForm` timeseries/state-space.ts:192 | Several Kalman recursions (`filterForm` is the ARMA state-space filter; its body was not read) |
| D5 | `aStar` graph/paths.ts:176 and `aStarSearchAlgorithm` search/informed.ts:133 | A* on a graph and on a problem; the second is an Algorithm |
| D6 | `minHash` and `lshIndex` in numerics/spatial/hashing.ts:180,331 and sketches/similarity.ts:82,210 | MinHash-LSH twice with different return types |
| D7 | `remez` numerics/approximation/best.ts:55 and signal/filters/fir.ts:261; `secant` and `fminbound` in signal/filters/_numeric.ts:163,87 against numerics/roots and `fminboundAlgorithm` (optim) | Filter design carries private copies of root and minimiser code |
| D8 | `poissonProcess`, `inhomogeneousPoisson`, `hawkesProcess` in datasets/events.ts:33,79,237 and stochastic/point-processes.ts:116,146,309 | Point-process simulators twice |
| D9 | `generateMaze` and `deadEnds` in environments/procedural.ts:242,213 and worlds/generators.ts:199 / worlds/analysis.ts:228; `bridges` worlds/analysis.ts:313 against graph/components.ts:360 | Maze generation (worlds has `mazeAlgorithm`) and bridge finding twice |
| D10 | `maximalCliques` pgm/triangulation.ts:229 and graph/colouring.ts:260; `convexHull` vision/hull.ts:103 and numerics/geometry/polygons.ts:44 | Same function name and purpose in two domains |
| D11 | `huffmanCoding` losses/large-output.ts:61 and `huffman`, `huffmanAlgorithm` information/source-coding.ts | Huffman tree for hierarchical softmax built separately from the coder |
| D12 | `sinkhorn` losses/divergence.ts:322 and `sinkhornAlgorithm`, `sinkhorn` optim/transport.ts:479 | Differentiable Sinkhorn loss and the optimisation-side one |
| D13 | `policyEvaluation` datasets/trajectories.ts:152 (linear solve) and rl/mdp.ts:341 | Two policy evaluations |
| D14 | `multigridPoisson` dynamics/pde/finite-difference.ts:555 and `multigridAlgorithm` numerics/sparse | `multigridAlgorithm` is numerics/sparse/multigrid.ts:382; whether `multigridPoisson` is a second implementation was not checked beyond the heads |

## Slot-rule violations and base adoption (b)

| id | domain | where | what was found | fix |
| --- | --- | --- | --- | --- |
| B1 | svm | `svm/classifier.ts`, `svm/regressor.ts`, `svm/one-class.ts` | classifier.ts solves with the library SMO only (`SvmPath` is `smo/alphaChanged`, :85); none of svm/*.ts refers to `smoothSolvers` or a `solver` hyperparameter (grep count 0; regressor.ts and one-class.ts mechanism not read). CLAUDE.md: kernel and linear classifiers take `solver`, SMO an extra mode. `linearSvmDualCdAlgorithm` (optim) is the other mode | Expose the (squared-)hinge objective as a `SmoothProblem`; keep SMO as a mode. M, plus tests |
| B2 | gp | `gp/_search.ts:52` `lbfgsbSearch`, used by classification.ts:384, regression.ts, sparse.ts | Hyperparameters are fitted by the marginal likelihood (smooth) with L-BFGS-B only; `optimise` is a boolean, not a solver choice | Take a bounded solver from a table; L-BFGS-B stays default. M |
| B3 | learn | `ncaAlgorithm` learn/metric-learning.ts:249 (L-BFGS, :273), `lmnnAlgorithm` (own learning rate, maxIter) | NCA and LMNN minimise smooth objectives with a hard-wired L-BFGS or hand-rolled gradient steps | `solver` from `smoothSolvers`. M |
| B4 | fairness | `learnFairRepresentation` fairness/preprocessing.ts:440 (`lbfgs` call at :502) | A smooth prototype-representation loss minimised by one L-BFGS call inside a function; no Algorithm, no solver slot | Algorithm plus `solver`. M |
| B5 | manifold | `tsne` manifold/tsne.ts:458 (momentum and gains), `umap` manifold/umap.ts:173 (SGD with negative sampling) | Optimisers are fixed because they follow sklearn and umap-learn. Both are exported Algorithms | Low: keep as the default named modes, add the generic optimiser as another. L |
| B6 | stats | `newtonMle` stats/fit.ts:191, `fitDistribution`:477 | MLE of a smooth likelihood fixed to Newton | Low: solver slot from `smoothSolvers`. S |
| B0 | learn | `defineTrainer`, `defineAgentMethod`, `sweepMethod` (learn/bases) | Not one call site of any of the three outside learn/bases (Method, step 2). The trainer base has no network behind it, the agent base no agent, the sweep base no study. `iterative` has 118 call sites outside the bases | Adoption, not a violation: first users are `fitAlgorithm` (nn/training/fit.ts:101), `trainingAlgorithm` (rl) and LE1 |

## Absences (c)

Classic algorithms with no version in the library. Each was searched by name and by common alias with `grep -rliE` over `packages/engine/src/lib` (tests excluded); a name that only appears in docs or comments does not count as present. Searches that found an unrelated hit (for example `lda` matching linear discriminant analysis) were not turned into claims: latent Dirichlet allocation, for one, is not claimed because a probabilistic program could express it.

| id | algorithm | search |
| --- | --- | --- |
| C1 | Federated averaging (FedAvg, FedProx, SCAFFOLD) | grep for fedavg, federated, clientUpdate, fedprox, scaffold: no hit in lib. docs/structure.md lists federated learning as a planned area |
| C2 | Population-based training | grep for pbt, populationBased over lib: no hit. docs/structure.md says PBT is a sampler/scheduler in `tune` |
| C3 | Parallel tempering or replica exchange MCMC | grep for parallelTempering, replicaExchange over lib: no hit |
| C4 | Reversible-jump MCMC; pseudo-marginal or particle MCMC (PMMH) | grep for reversibleJump, pseudoMarginal, particleMarginal, pmmh: no hit |
| C5 | Model-agnostic meta-learning (MAML, Reptile) | grep for maml, reptile, metaLearn: no hit (autodiff supports second derivatives) |
| C6 | Interacting multiple model (IMM) filter | grep for imm, interactingMultipleModel over lib: no hit |
| C7 | Global min-cut (Karger, Stoer-Wagner); balanced graph partitioning (Kernighan-Lin, Fiduccia-Mattheyses); Infomap | grep for karger, stoerWagner, kernighanLin, fiduccia, infomap: no hit (the only `kernighan` hit is the Kernighan-Church-Gale spelling channel in text/spelling.ts). `minCut` is the s-t cut |
| C8 | Gittins index | grep for gittins over lib: no hit |
| C9 | Newton-Schulz iteration, CORDIC, divide-and-conquer eigensolver | grep: no hit. Only worth adding as Algorithms with NU3 |

## Method

The audit can be rerun with the steps below; the scripts themselves are not committed.

1. File set. `find packages/engine/src/lib -name "*.ts" ! -name "*.test.ts" ! -name "*.test-d.ts" ! -name index.ts` gives 1598 files in 66 domain folders. Every file was parsed with the TypeScript compiler API (TypeScript 5.6.1-rc from `node_modules/.pnpm`; the repository's TS 7 has no JavaScript API). For each top-level function, arrow or function-valued const the pass records: exported or not, declared return type text, line, line count, number of `for`/`while`/`do`/`for-of`/`for-in` loops (and how many are `while` or `do`), the names it calls, the number of `Algorithm<...>` type references, the number of object literals having both an `init` and a `step` member ("Algorithm literals"), and calls to `runAlgorithm`, `iterate`, `runFor`, `iterative`, `defineTrainer`, `defineAgentMethod`, `sweepMethod`, `sweep`, `race`, `defineEstimator`, `fittedSession`. Result: 16498 declarations in 1590 files (8 files have no top-level function or const).
2. Inventory. "Exported Algorithm declarations": exported declarations whose declared return type text contains the word `Algorithm` (562). Cross-check by grep: `grep -hE "^export (function|const) [A-Za-z0-9_]*Algorithm\b"` over the same files gives 539 names. "Algorithm literals": 611. "Estimators on `iterative`": exported declarations that call `iterative` (103); `grep -E "\biterative<"` outside `lib/learn/bases` gives 118 call sites. `defineEstimator` call sites: 44 (grep). `grep -rF "defineTrainer("`, `"defineAgentMethod("`, `"sweepMethod("` outside `lib/learn/bases`, tests included: 0, 0, 0.
3. Candidate lists, each built from the records above and each by name-level call graph inside one file (calls to another file are not followed, which is why a few cross-file wrappers were found only by reading):
   - S1, exported inline: exported, return type does not name `Algorithm` or an estimator kind, contains an Algorithm literal, and calls no exported `*Algorithm` factory. 59 (67 before dropping 8 wrappers such as `subgradientMethod` that call an exported factory).
   - P, private and uncovered: not exported, contains an Algorithm literal or returns `Algorithm`, and not reachable by calls from any exported Algorithm factory of the same file. 29.
   - R, runs external: exported, calls `runAlgorithm`, has no literal, and reaches no Algorithm literal in its file. 34.
   - L, loops: exported function, not returning an Algorithm or estimator, with at least one loop and no Algorithm reachable in its file, a name matching `fit|run|sample|simulate|train|search|grow|boost|em|gibbs|anneal|evolve|propagate|filter|smooth|track|solve|iterate|update|learn|infer|estimate|cluster|mine|find|extract|build|induce|prune|select|walk|sweep|stream|online|adapt|bandit|policy|rollout|message|mcmc|newton|bfgs|lanczos|arnoldi|power|jacobi|krylov` (abridged; case-insensitive prefix match) or a `while`/`do` loop, and at least 12 lines. 410.
   - D, delegating: as L, but the loop is only in functions reachable by calls within the file, restricted to a `while`/`do` loop in the reach, at least 40 lines reached, and excluding the domains `dists foundations symbolic tensor autodiff exposition datasets data losses kernels metrics scene graphics space base environments problems random trace worlds systems signal text tune`. 2232 before the restrictions, 148 after (signal, text and tune were then read through L only).
   Listings examined in total: 67 + 29 + 34 + 410 + 148 = 688 (the lists overlap and were not de-duplicated). Most functions named in a row appeared in one of these lists. A few were found by reading the exports of a domain instead, because their loop sits in a helper in another file: the graph classics `dijkstra`, `bellmanFord`, `floydWarshall`, `minimumSpanningTree`, `topologicalSort`, and the scalar root finders.
4. Triage. For every L and D entry an AST pass printed the doc summary, the first loop headers and the calls; entries that were one-shot (closed-form, a parser, a tokenizer, a sort, a kernel function, a metric, a design-matrix builder, a codec) were dropped. Wrappers of an exported `*Algorithm` (`dpll`, `cdcl`, `walksat`, `arimaFit`, `etsFit`, `trainGlove`, `trainWord2vec`, `fitGlm`, `fitHawkes`, `ldpcDecode`, `fit`, ...) and entries whose Algorithm is delivered through a mode table (`fitGlm`/`glmSolvers`, `trainKg`/`kgSolvers`, `choiceAlgorithm`, `irtAlgorithm`) were dropped as covered; estimators built by `iterative` around a private Algorithm (`aode`, `adaBoostClassifier`, `noveltyEstimator`, stacking) were dropped as covered. MCMC kernels (`hmcKernel`, `nutsKernel`, `metropolisHastings`, ODE steppers) are consumed by `samplerAlgorithm` and `solveIvpAlgorithm` and were dropped as slots.
5. Verification depth. Read in full: `runFilter`, `runPf`, `runSteps`, `simulatePid`, `recedingHorizon`, `relayAutotune` (opening), `pelt`, `binarySegmentation`, `bellmanFord`, `crossValidate`, `runBandit`, `integration` (games), `subgradientMethod`, `euclid`, `dpll`, and the head of `mcmc`. The remaining rows were confirmed from the AST extract of the function (doc summary, loop headers, calls, line count) plus the exported-Algorithm list of the domain, not from a full read of each body. A row claims only what that extract shows. Slot-rule rows were checked by grep for `smoothSolvers`, `solver` and the optimiser call in the named files.
6. Duplicates (a): exported function names present in two or more files, filtered by eye, plus name searches for a list of algorithm names (`viterbi`, `kalman`, `changepoint|bocpd|cusum|pelt`, `convexHull`, `dijkstra|aStar`, `huffman`, `kmeans|lloyd`, `ransac`, `sinkhorn`, `bfs|dfs`, `lsh|minhash`, `levenshtein`, `mcmc|metropolis`, `topologicalSort`, ...). For each pair the signatures, return types and sizes of both sides were compared from the records of step 1; heads were opened for D3 and D14 only.
7. Absences (c): `grep -rliE "<name|alias>" packages/engine/src/lib --include="*.ts"` excluding tests, for each name in the table above and other names that turned out to exist (Dinic, Edmonds-Karp, Kosaraju, Tarjan, Hopcroft-Karp, Gale-Shapley, t-digest, Hyperband, PPO, DQN, SARSA, BiCGStab, GMRES, ILU, Arnoldi, LARS, ...).

## Not examined

Read only through the L and D lists and by name searches, not row by row: `tensor`, `autodiff`, `symbolic`, `foundations` (its step-by-step functions such as `euclid` return a `steps` array instead of an Algorithm: by design, "step-recorded versions of direct methods", and not listed), `dists`, `losses`, `metrics`, `kernels`, `distances`, `space`, `random`, `context`, `base`, `exposition`, `scene`, `graphics` (the three renderers have Algorithms), `environments`, `problems`, `datasets` (only for duplicates), `data`, `systems`, `worlds`, `trace` and `tune` beyond the study Algorithm. These are mostly one-shot computations, containers and generators, so the omission is a judgement of value, not a measured zero.

Not covered by this method at all: iterative code that is not a top-level function or const (class methods, methods of returned objects), and routines whose loop sits in another file than the exported function that hides it, unless the exported function itself calls `runAlgorithm`. Estimator classes (`EstimatorClass`) were not scanned for hard-wired solvers beyond the files named in section (b).

