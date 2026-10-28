# Inventory: what is parked, and what in it belongs in the library

Written at the end of part 0 (2026-10-26). Everything under `_parked/` keeps its original path below that folder, so
restoring a file is `git mv` back. Nothing in `_parked/` is built, linted, typed or tested.

**Update, 2026-10-28.** `_parked/` was deleted from the tree, with the hub. Every path below is in git history at
933f0dd8: `git show 933f0dd8:_parked/<path>` reads a file, `git checkout 933f0dd8 -- _parked/<path>` brings it back.

## 1. Parked

| what                                                                           | where it is now                                                                                      | lines        |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ------------ |
| the methods layer (all areas but the keepers below)                            | `_parked/packages/engine/src/methods/`                                                               | about 43,000 |
| the hub package (220 entries, the entry contract, the worker host)             | `_parked/packages/hub/`                                                                              | about 20,000 |
| the site's hub section (55 standalone pages, the index plugins, the redirects) | `_parked/apps/site/src/hub/`, `_parked/apps/site/plugins/`, `_parked/apps/site/src/gallery/moved.ts` | about 8,000  |
| two gallery recipes that only showed methods                                   | `_parked/apps/site/src/gallery/recipes/{charts/05-any-record,controls/07-method-controls}.tsx`       | small        |
| two render tests that fed views from methods                                   | `_parked/packages/render/src/views/{agents,epicycles}.test.ts`                                       | small        |
| the fairness in-processing test (its subject is logistic regression, a method) | `_parked/packages/engine/src/lib/fairness/in-processing.test.ts`                                     | small        |

Kept and moved into the library in part 0: `methods/rl/environments` → `lib/environments` (41 environments),
`methods/rl/agents` → `lib/agents` (76 agents), `methods/gofai/{maps,puzzles,planning-domains}.ts` and `gridProblem`
→ `lib/problems`, `methods/graphics/rendering.ts` → `lib/graphics`. Their golden fixtures and oracle cases stay at
`fixtures/methods/rl/...` and `oracle/cases/methods/rl/...` (the tests name them by string); moving them is a Python
job for whenever the oracle is next touched.

## 2. Algorithms inside parked methods that belong in `lib/` (part 2)

By the rule "a method is a light wrapper; if what it needs is not in the library, that is the thing to fix". Each is an
`Algorithm` factory today; it moves with its helpers, tests and goldens into the domain named, and the particular that
wraps it in part 3 imports it from there. None duplicates something already in `lib/` (checked by name;
`studyAlgorithm` exists in both and the library's wins).

| algorithm                                                                                                                             | parked file                            | target domain                                                   | needed by the part 3 examples         |
| ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------- | ------------------------------------- |
| `lloydAlgorithm`, `miniBatchKMeansAlgorithm`                                                                                          | `cluster/kmeans.ts`                    | `lib/cluster` (new)                                             | **yes** (clusterer)                   |
| `dbscanAlgorithm`, `meanShiftAlgorithm`                                                                                               | `cluster/density.ts`                   | `lib/cluster`                                                   |                                       |
| `agglomerationAlgorithm`                                                                                                              | `cluster/hierarchical.ts`              | `lib/cluster`                                                   |                                       |
| `affinityPropagationAlgorithm`, `fuzzyCMeansAlgorithm`, `pamSwapAlgorithm`, `copKMeansAlgorithm`                                      | `cluster/*.ts`                         | `lib/cluster`                                                   |                                       |
| `irlsAlgorithm` (and the coordinate-descent solver of `solvers.ts`, `paths.ts`)                                                       | `linear/solvers.ts`, `linear/paths.ts` | `lib/linear` (new; the design matrix, GLM families and solvers) | **yes** (classifier, regressor, path) |
| `plainSgdAlgorithm`                                                                                                                   | `linear/online.ts`                     | `lib/linear`                                                    |                                       |
| `growTreeAlgorithm`                                                                                                                   | `trees/grow.ts`                        | `lib/trees` (new)                                               |                                       |
| `isolationForestAlgorithm`                                                                                                            | `anomaly/isolation.ts`                 | `lib/trees` or `lib/anomaly`                                    |                                       |
| `gaussianEmAlgorithm`, `variationalMixtureAlgorithm`                                                                                  | `mixture/*.ts`                         | `lib/mixture` (new) or `lib/dists`                              |                                       |
| `baumWelchAlgorithm`                                                                                                                  | `models/hmm.ts`                        | moved: `lib/pgm/learning/baum-welch.ts` (`methods/sequence`)    |                                       |
| `ldaGibbsAlgorithm`                                                                                                                   | `models/topic.ts`                      | `lib/pgm` or `lib/models` (new)                                 |                                       |
| `fastIcaAlgorithm`, `nmfMultiplicativeAlgorithm`, `incrementalPcaAlgorithm`, `factorAnalysisAlgorithm`                                | `decomposition/*.ts`                   | `lib/decomposition` (new)                                       |                                       |
| `smacofAlgorithm`, `somAlgorithm`, `tsneAlgorithm`, `umapLayoutAlgorithm`                                                             | `manifold/*.ts`                        | `lib/manifold` (new)                                            |                                       |
| `graphicalLassoAlgorithm`, `cStepAlgorithm`                                                                                           | `covariance/*.ts`                      | `lib/stats` or `lib/covariance` (new)                           |                                       |
| `graphLabelAlgorithm`                                                                                                                 | `semi-supervised/graph-based.ts`       | `lib/graph`                                                     |                                       |
| matrix-profile family (`stump`, `stomp`, `stamp`, `scrump`, `stumpi`, `mstump`, `stimp`, `floss`)                                     | `timeseries/matrix-profile/*.ts`       | `lib/signal` or `lib/timeseries` (new)                          |                                       |
| `daggerAlgorithm`, `maxEntIrlAlgorithm`, `selfPlayAlgorithm`, `fictitiousSelfPlayAlgorithm`, `offlineAlgorithm`, `alphaZeroAlgorithm` | already moved with `lib/agents`        | —                                                               |                                       |

Not algorithms but library-grade pieces inside parked methods, to move when a particular needs them: the SVM kernel
and dual plumbing (`svm/_dual.ts`, over `lib/optim`'s SMO), the neural `_nets.ts` builders (`buildMlp`, `lenet`),
the GP marginal-likelihood fit (`gp/`), the ensembles' staging (`ensemble/`), explain's attribution kernels
(`explain/`: Shapley sampling, LIME, integrated gradients; `lib/explain` has most of them already, check for
duplicates), the fairness mitigations (`fairness/`; `lib/fairness` has the measures).

The 23 placeholder areas (`algorithms`, `causal`, `dynamics`, `embedding`, `gam`, `survival`, `scientific`, ...) hold a
six-line `index.ts` each and nothing else; nothing to move.

## 3. What part 3's four examples need from part 2

- **classifier**: logistic regression — the IRLS solver and the design matrix from `linear/`.
- **regressor**: ridge or the elastic net — the coordinate-descent solver from `linear/solvers.ts`.
- **path**: the lasso path over a penalty grid — `linear/paths.ts` (`pathOn`, `alphaGrid`), as a sweep.
- **clusterer**: k-means — `lloydAlgorithm` and `initialCentres` from `cluster/kmeans.ts`.

So part 2 can start with `linear` and `cluster` alone; the rest moves when something asks.
