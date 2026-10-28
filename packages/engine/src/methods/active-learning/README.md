# Active learning (part 5, learning paradigms; a chapter the encyclopedia folds into online learning and data-centric learning)

Choosing what to label: the query loop over a pool (fit, score the pool, pick, label, repeat) with the strategy as the mode (uncertainty, margin, entropy, random), query by committee over a committee slot, the lookahead strategies (expected error reduction, expected model change), the Bayesian ones (BALD, BatchBALD), the ones that look at the pool's geometry (density weighting, core sets, BADGE), selective sampling on a stream with a budget, and the learning curve by labels bought against random sampling, as sessions over `learn/active-learning` (the loop and the strategies exist) and the models of `supervised-learning`, with a labelled set that grows step by step and the hidden labels as the truth.

## Structure

Active learning chooses rows, so it is a selection study in the sense of section 3 of `methods/README.md` (`data/`: "preparation, shift and selection studies") and lives under `data/selection/active/`, one folder per family of strategy; the strategy slot table is the chapter's shared piece; the learning curve by budget and the failure cases are studies that live beside the loop because every strategy's page ends in them. The alternative, `fit/classify/active/` (the loop is a classifier that asks), is recorded under Problems.

```
data/selection/active/
  pool/               the query loop over a pool with the uncertainty strategies as modes (the lib loop exists)
  committee/          query by committee over a committee slot: vote entropy, consensus entropy, maximum disagreement
  expected/           expected error reduction and expected model change, with their cost
  bayesian/           BALD and BatchBALD over a Bayesian model slot (MC dropout, an ensemble, a GP)
  diversity/          density-weighted uncertainty, core sets by k-centre greedy, BADGE
  stream/             selective sampling on a stream with a budget
  studies/            the learning curve by labels bought against random, the failure cases
shared/               the strategy slot table, the pool and stream data slots, the query record (below)
```

Nothing is built as a method; the loop and six strategies are in `learn/active-learning.ts` as a traced `Algorithm`, so the pool row is a method the moment the shared pieces exist.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of every method is one query round (one row or one batch bought); the library loop logs `labeled` and `score` as metrics and `query` as a message, which the base exposes as `query/labeled` s, `query/score` s and `query/query` m; the inner model's probes (`model/decision-surface` p at the viewed round) come from the model slot. The status is `built <path>`, `partly` or empty; library readiness is in the needs column (a function named there exists; `TODO:` names what is missing and its proposed home).

### data/selection/active/pool/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `pool-query-loop` | The loop as the algorithm, one round per step with the batch size as the knob and the strategy as the mode (least confidence, margin, entropy, random), any classifier with probabilities as the model slot: the rows chosen marked in order on the plane, the strategy's score over the pool as a surface at the viewed round, the held-out score climbing, the learning curve against random sampling (the gain, and the rounds where uncertainty chases the boundary's noise); the initial labels (how many, stratified) as the cold-start knob. | `query/labeled` s, `query/score` s, `query/query` m, `query/strategyScores` p (over the pool), `rows/order` p (the labelled set in query order), `model/decision-surface` p, `study/learningCurve` proj | `learn/active-learning.ts`: `activeLearning`, `uncertaintySampling`, `marginSampling`, `entropySampling`, `randomSampling`, `leastConfidence`, `classMargin`, `predictiveEntropy`, `appendRows`; `data/modifiers/shift.ts`: `semiSupervisedSplit` (the initial labels); `exposition/searches.ts`: `learningCurve` | |

### data/selection/active/committee/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `query-by-committee` | The committee as a slot (bootstrap replicas of one model, several models of the registry, MC-dropout passes of one network), the disagreement as the mode (vote entropy, consensus entropy, maximum disagreement by KL to the mean): the committee's boundaries drawn, the disagreement surface and the point chosen on it; the version space shrinking on a linear model (the boundaries converging round by round). | as `pool-query-loop`, plus `committee/size` s, `committee/disagreement` p, `model/committeeSurfaces` p | `learn/active-learning.ts`: `voteEntropySampling`, `consensusEntropySampling`, `maxDisagreementSampling`; `random`: `bootstrapIndices`; `nn/functional/regularisation.ts`: `dropout` (the MC-dropout committee); `learn/estimator.ts`: `cloneEstimator` | |

### data/selection/active/expected/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `expected-error-reduction` | For each candidate and each label, refit and measure the expected loss over the pool (the cost: candidates times labels refits per round, on the tape; the candidate subsample as the knob), expected model change (the gradient norm a candidate would induce under each label) as the cheaper mode; against uncertainty on a dataset where the uncertain rows are uninformative (a planted noisy region). | as `pool-query-loop`, plus `query/innerFits` s, `query/expectedLoss` p (per candidate), `query/modelChange` p | `learn/active-learning.ts`: `expectedErrorReduction`, `rankedBatchSampling`; `autodiff`: `grad` (the model change); TODO `expectedModelChange` in `learn/active-learning.ts` | |

### data/selection/active/bayesian/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `bald` | The mutual information between a candidate's label and the parameters (the predictive entropy minus the expected entropy over posterior samples) over a Bayesian model slot (MC dropout, a deep ensemble, a GP classifier), BatchBALD for batches (the joint information, avoiding near-duplicates): BALD ignoring the aleatoric noise that entropy chases (the planted noisy region again), the two surfaces side by side. | as `pool-query-loop`, plus `query/epistemic` p, `query/aleatoric` p, `query/batchInformation` s | `information/discrete.ts`: `entropy`; `nn/functional/regularisation.ts`: `dropout`; `tune/surrogate.ts`: `gpPosterior` (the GP slot's posterior); TODO `baldSampling`, `batchBald` in `learn/active-learning.ts` | |

### data/selection/active/diversity/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `diversity-strategies` | Density-weighted uncertainty (the information density: a candidate's mean similarity to the pool, β as the knob; the outliers uncertainty chases avoided), core sets (k-centre greedy on the embedding, one centre per step, the covering radius falling), BADGE (k-means++ on gradient embeddings, uncertainty and diversity at once) as modes; the batch chosen drawn with its coverage of the pool; against uncertainty on a pool with planted outliers and duplicates. | as `pool-query-loop`, plus `query/coveringRadius` s (core sets), `query/density` p, `query/batchCoverage` p | `numerics/spatial/brute.ts`: `kNearest`; `distances/distances.ts`: `distances`; `stats/density.ts`: `gaussianKde`; `cluster/kmeans.ts`: `initialCentres` (k-means++ seeding for BADGE); `autodiff`: `grad`; TODO `densityWeightedSampling`, `coreSetSampling`, `badgeSampling` in `learn/active-learning.ts` | |

### data/selection/active/stream/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `stream-selective-sampling` | A row arrives, the strategy decides to ask (uncertainty above a threshold, with a budget: labels bought against rows seen), the model updated online (any learner of `online-learning` as the model slot); the threshold and the budget as knobs, the labels bought against the rows seen on the tape, the accuracy against pool-based querying with the same budget. | `stream/t` s, `stream/uncertainty` s, `stream/labeled` s, `stream/asked` e, `model/decision-surface` p | `learn/active-learning.ts`: `streamActiveLearning`; `methods/shared/streams.ts` (the stream base, to build) | |

### data/selection/active/studies/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `budget-curve` | Every strategy against random on one dataset, the learning curve by labels bought with a band over seeds, the area under the curve as the summary; the model as a slot (does the ranking of strategies change with the model?), the dataset's shape as the other (where random is hard to beat). | `trial/labels` s, `trial/score` s, `study/curves` proj (a `learningCurve` per strategy), `study/areas` proj | `learn/model-selection/cross-validation.ts`: `learningCurve`; `exposition/searches.ts`: `learningCurve`; a `study/sweep` over the rows above | |
| `failure-cases` | Planted scenarios on one dataset: sampling bias (the labelled set's class balance drifting from the pool's, on the tape), outliers (uncertainty buying noise), the cold start (too few labels for the model to be informative; the length of the random phase as the knob), near-duplicates in the pool (batches wasted without diversity); what each strategy does in each. | `query/labelBalance` t, `query/outliersBought` s, `query/duplicatesBought` s, `study/scenarios` proj | `data/modifiers/features.ts`: `withOutliers`; `data/modifiers/labels.ts`: `withClassImbalance`; the strategies above | |
| `data-valuation` | Which rows are worth labelling by their value to the model (data Shapley, kNN Shapley, influence). | | `explain/valuation.ts`: `dataShapley`, `knnShapley`, `influenceFunctions` | see `data-centric-learning` |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `strategies.ts` | The strategy slot table over `learn/active-learning.ts`: every strategy with its knobs (the measure, the batch size, the subsample, β, the committee size), what it needs of the model (probabilities, a committee, posterior samples, an embedding, gradients) as its validity rule, and the pool scores it exposes as a probe | every group |
| `pool.ts` | The pool and stream data slots: a labelled generator with its labels hidden (the shared slot of `semi-supervised-learning`), the initial labelled set (how many, stratified), the held-out set the score is on, the pool as a stream for `stream/`; the truth kept for every probe | every group |
| `loop.ts` | The query loop base over `activeLearning`: the inner model warm-started or refitted (the budget of inner fits on the tape), the oracle that reveals hidden labels, the batch, the stopping rule (budget, score plateau) | `pool/`, `committee/`, `expected/`, `bayesian/`, `diversity/` |
| `records.ts` | The query record: the labelled set in query order with the strategy's score at the time, the strategy's score surface, the learning curve against random with its band | every group |

## Examples to make

- Uncertainty sampling against random on two moons: the chosen points marked in order, the two learning curves.
- Query by committee: the committee's disagreement surface and the point chosen, the boundaries converging.
- Density-weighted uncertainty avoiding the outliers that plain uncertainty chases.
- BALD against entropy on a planted noisy region: epistemic against aleatoric.

## Rendering

- Rows with the labelled set growing (order as colour), decision surfaces, learning curves (`learningCurve`), the strategy's score surface as a probe.
- Missing: a query-order overlay on the rows panel; a budget-curve panel with bands and the area; a two-surface panel (the strategy's score beside the model's surface under one cursor).

## Problems

- Keeps its name. Each query round refits a model: the inner fit is budgeted or warm-started (`shared/loop.ts`), and the expected-error strategies multiply it by candidates and labels; the subsample knob is the cost control, and the tape says what it cost.
- The home: `data/selection/active/` reads the loop as a selection of rows; `fit/classify/active/` would read it as a classifier that asks. The first is chosen because the strategy, not the model, is the method, and the model is a slot; the data-valuation strategies of `data-centric-learning` sit beside it either way.
- The library has the loop and six strategies; BALD, the diversity strategies and expected model change are additions to `learn/active-learning.ts`, each over pieces that exist.
- The stream row waits on `methods/shared/streams.ts` with `online-learning`.
