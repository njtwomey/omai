# Where omai goes next: a directional plan

This is the short version of the completeness survey ([README.md](README.md) has the numbers and the full top-30 list; the family
files have the detail). It says where the value is, in what order it would make sense to build, and what needs a decision from
the owner. It is a direction, not a task list: sizes and homes are in the survey files and will want checking when something is picked
up. Everything rests on name-level matching, so treat each item as "worth a closer look", not as a verified spec.

## What omai is, and so what "next level" means

omai is an observable machine-learning library: methods run as traced, steppable sessions, with datasets that carry their truth, and
figures that can be played. It is not trying to be scikit-learn, PyTorch or PyMC at scale. So the useful question about any gap is
not "does a famous library have it" but two questions:
1. Would a learner or practitioner reasonably look for it here, and be let down by its absence?
2. Would a traced, steppable version teach something a one-shot function cannot?

The survey finds omai already strong where it has put its depth: linear models with the solver as a slot, trees and boosting, kernels
and GPs, model selection and calibration, scipy.stats and optimisation, classical time series, samplers and their diagnostics as
numbers, distributions, graphical models (a superset of pgmpy, and CRFs in the CRFsuite and CRF++ forms), tabular and deep RL, the
torch layer, loss and optimiser surface, and explainability, fairness, causal and vision. What follows is where the picture is thin.

## The big gaps, in order of value

**1. Seeing what the sampler did (posterior and MCMC diagnostic charts).** omai computes R-hat, ESS, divergences and rank statistics, and
nothing draws them: no trace, rank, autocorrelation, forest, pair or energy plots, no posterior-predictive checks, no PIT or LOO-PIT. This is the
cheapest large win. The numbers and the samplers exist, the charts are render capabilities (so they belong in the gallery by the
repository's own rule), and every Bayesian user expects them. It also makes the sampling examples much better. Items 1, 19 and 20 of the
list.

**2. Time series beyond the classical toolkit.** This is the largest cluster of absences in scipy, statsmodels and sktime space (about 600
absent names, though much of it is a long tail). The ARIMA/ETS/VAR/GARCH side is covered; what is missing is the modern forecasting
workflow and the state-space family built on code omai already has: structural time series on the Kalman filter, Markov-switching
models on the HMM code, a forecasting framework (windowed-regression forecasters, pipelines, backtests, Theta, a Prophet-style trend),
and time-series classification and clustering (DTW k-NN, ROCKET, soft-DTW, barycentre averaging). Items 2 to 5. A learner doing
forecasting today hits the edge of omai quickly; this is where the most users would notice.

**3. Learning paradigms that only make sense when you can watch them run.** Meta-learning (MAML, Reptile, prototypical networks),
continual learning (forgetting curves, the accuracy matrix), federated learning (FedAvg), self-supervised trainers (SimCLR, BYOL with
the embedding rank on the tape, MoCo, masked autoencoders) and batch-label metric learning (miners, multi-similarity, lifted-structure,
proxy losses, with a retrieval report). Items 21, 25, 26, 27. These are the best fit for omai's identity: the interesting thing about
each is a curve or a dynamic (collapse, forgetting, drift of the inner loop) that a final number hides. They are also where the survey
found the largest absences against the libraries a learner would meet next (pytorch-metric-learning, lightly, learn2learn, Avalanche).
omai already has the pieces (contrastive losses, NCA and LMNN, the trainer bases), so the work is orchestration and the tape.

**4. Graph learning as a pipeline, not just layers.** omai has message passing, GCN-style layers, graph algorithms and structure
drawing. Missing is what turns them into a working example: graph mini-batching, edge and node splits with negative sampling,
link-prediction heuristics, a few real graph datasets (Cora, CiteSeer, MUTAG), more layers (GATv2, GINE, Chebyshev), pooling
(DiffPool, Set2Set) and embeddings (Node2Vec, GAE). Items 6 to 8. The survey's 625 "absent" graph names are mostly NetworkX's long tail
of graph generators; the real gap is this short pipeline list.

**5. The Bayesian modelling workflow.** Reparametrisation handlers in probprog (the eight-schools funnel is the standard lesson), Bayesian
neural network layers, topic models (LDA by Gibbs and by variational Bayes), parallel tempering and ensemble samplers, and model-checking
tools. Items 10, 11, 12, 19, 20. omai's samplers and diagnostics are good; the surrounding workflow is thinner than in NumPyro or PyMC.

**6. Finishing what omai is already good at.** Two places where omai leads and a few additions would make it the obvious reference:
structured prediction (CRFs exist and are strong; the ring around them is missing: a CRF layer on a neural encoder, constrained and
n-best decoding, a max-margin structured SVM, a MEMM to teach label bias, semi-Markov CRFs; item 28) and Bayesian networks (classic
benchmark networks such as Asia, Sachs and Alarm, a BIF reader, noisy-OR, K2 and BDeu scores; item 9).

**7. Real data and the audit step.** The datasets with known truth are a strength, but several well-known real ones are missing: small real
graph sets, a handwritten-digit subsample, a colour image set, survival sets such as Rossi and lung (items 7, 16, 17), and a data-audit
report (drift per column, duplicates, conflicting labels, leakage, label issues; item 30). They are cheap to add and appear in every
tutorial; the cost is licence and repository size, which is an owner decision (below).

**8. The sklearn-shaped front door.** A set of small things that make omai feel complete to someone arriving from scikit-learn:
estimator wrappers for things that exist only as algorithms (`kmeans`, `gaussianMixture`), dummy estimators, decision-threshold tuning
(item 14), named small networks as builders with depth and width as knobs (LeNet, VGG-style, ResNet, a small ViT and U-Net, each
describable; item 29), the missing anomaly detectors (item 13) and the regression diagnostics statsmodels users expect (item 15). None
is deep. Together they remove the "it is not quite what I expected" moments.

**Second tier.** Competing risks and survival extras, recurrent and action-masked policies for PPO, classic board games and one more
control task, causal attribution and policy learning, and the missing named distributions (items 17, 18, 22, 23, 24). Worth doing,
each for a smaller audience.

## Two lists, one direction

The AIFN survey ([../aifn/gaps.md](../aifn/gaps.md)) and this one answer different questions: what the existing notes need to run on
omai, and what a good library of this kind should have. They agree on a few points and the combined order should respect both:
- Notes first, where cheap: `Readout` (a labelled value under a figure, one small component that is the only missing piece for over
  900 notes), the teaching dataset generators with an exact truth, and a handful of small numerics. These come from the AIFN list and
  are small.
- The biggest porting cost in the notes is not a missing feature at all: it is the figure layer, key-based randomness (different
  numbers from the same seed) and the estimator wrappers. A migration guide and a few helpers would unlock more notes than any single
  algorithm.
- Items that appear in both lists (survey 28 and the notes' structured-prediction needs; real datasets; graph pipeline pieces) go
  first among the larger items.

## A sensible order

1. **Quick, visible wins** (weeks, not months): the posterior diagnostic charts, the estimator wrappers and named networks, the Bayesian
   network benchmarks, `Readout` and the small numerics for the notes. These need little new design and each shows up immediately.
2. **The time-series framework**: the largest single hole against statsmodels and sktime, and the one most users reach.
3. **Graph learning pipeline**: batching, splits, datasets and the missing layers, one worked example end to end.
4. **The paradigm family** (metric learning with miners, meta-learning, continual and federated, self-supervised): each can ship alone,
   and each should come with the curve or tape that makes it worth watching.
5. **Bayesian workflow and the CRF ring**, then the second tier.

Each step should land the way recent work did: a library module with tests and TSDoc and an `@index` entry first; gallery recipes only
for new render capabilities (the diagnostic charts qualify; a model such as MAML does not); examples under `examples/` where a figure
and a tape make a good front-page story.

## What needs a decision from the owner

- **Vendoring real datasets.** Cora, MNIST and CIFAR subsamples, survival sets and others add weight and carry licences. Options: vendor
  small subsamples (as the existing real datasets are), or keep them remote and pinned by hash (as `electrocardiogram` is). Which, and
  what size limit?
- **How far to go with deep models.** Named networks, self-supervised trainers and a few architectures are in scope as teaching pieces
  at small scale. Where is the line (parameter count, training time in the browser)?
- **AIFN-driven or survey-driven first.** If porting the notes is the priority, step 1 should lean on the AIFN list; if broadening omai
  is, lean on this one. The plan above assumes both for the quick wins and the survey for everything larger.
- **`syntheticEcg` and `syntheticBeat`** from the AIFN engine, which came with the convolutional sparse coding work but were not ported
  (omai has a general `pulseTrain` generator instead).

## What this plan cannot tell you

- The matching is by name. A capability omai offers as a configuration of something else (a loss as a slot of a trainer, say) can look
  absent; one that exists by name can be much thinner than the library's version. Check an item against the source before building it.
- Many reference lists (PyG, Stable-Baselines3, PyMC, lifelines, Lightning, pytorch-metric-learning and about 60 others) were written from
  memory, not from the installed library. They are checklists, not release-accurate inventories.
- Some omai domains had no library to compare against, or a thin one: fields, foundations, graphics, search, problems, worlds, the non-ODE
  half of dynamics, and (thinly) logic, stochastic processes, rules, survival, GAMs, kernel methods, anomaly and agents. The blind-spot
  map ([blind-spots.md](blind-spots.md)) lists textbook-level gaps for some of these (SMA*, AND-OR search, RRT; BDDs and model checking;
  Jackson networks; partial information decomposition; marching cubes; neuron models; POS taggers). Another pass against textbooks, not
  libraries, would measure those.
- A name comparison cannot see what omai is for: whether a method is traced well, steppable, teachable, or has a good example. The
  best next-level work is judged on that, not on the count of absent names.
