# Graphical models

Part of the [completeness survey](README.md). pgmpy is the closest library to omai's `pgm` domain: factors, Bayesian and Markov networks, structure and parameter learning, exact and approximate inference, causal inference. hmmlearn covers HMMs. Both were introspected. pyAgrum and libpgm were not installed and are not listed separately; their scope is the same.

Counts are of names: "covered" means an omai export or a documented configuration does what the reference name does, "partial" means a different shape or fewer options, "absent" means nothing found by name or by concept, and "out of scope" is plumbing, GPU or production features, downloads and the like that omai should not copy. Semantics were not compared.

## Summary

| library      | source              | area               | covered | partial | absent | out of scope |
| ------------ | ------------------- | ------------------ | ------: | ------: | -----: | -----------: |
|              |                     | base               |       3 |       0 |      4 |            0 |
|              |                     | causal_discovery   |       3 |       0 |      2 |            0 |
|              |                     | ci_tests           |      10 |       0 |      6 |            1 |
|              |                     | estimators         |      12 |      11 |      1 |            1 |
|              |                     | factors            |       5 |       0 |      0 |            2 |
|              |                     | factors.continuous |       1 |       0 |      0 |            0 |
|              |                     | factors.discrete   |       3 |       0 |      1 |            1 |
|              |                     | factors.hybrid     |       1 |       0 |      0 |            0 |
|              |                     | inference          |       7 |       0 |      0 |            1 |
|              |                     | metrics            |       1 |       0 |      6 |            2 |
|              |                     | models             |      13 |       2 |      0 |            0 |
|              |                     | readwrite          |       0 |       0 |     14 |            0 |
|              |                     | sampling           |       3 |       0 |      0 |            1 |
|              |                     | utils              |       0 |       0 |      3 |           14 |
| **pgmpy**    | introspection 1.1.2 | **all**            |  **62** |  **13** | **37** |       **23** |
|              |                     | hmm                |       3 |       2 |      0 |            0 |
|              |                     | vhmm               |       0 |       0 |      2 |            0 |
| **hmmlearn** | introspection 0.3.3 | **all**            |   **3** |   **2** |  **2** |        **0** |

## Where omai is already strong

- Exact inference: variable elimination with elimination-order heuristics, junction trees, belief propagation, max-product and MAP, MPLP, treewidth; approximate: Gibbs, likelihood weighting, rejection, mean field, variational message passing, EP, tree-reweighted BP.
- Models beyond pgmpy: CRFs (linear-chain and general), Gaussian MRFs, influence diagrams, plate models, factor graphs, Ising and Potts samplers (Swendsen-Wang, Wolff), HMM variants (HSMM, IO-HMM, factorial, GMM-HMM), linear dynamical systems.
- Structure learning (PC, GES, hill climbing, Chow-Liu, NOTEARS, DirectLiNGAM, FCI), identification (backdoor, front-door, instruments), counterfactuals and twin networks.
- Inference algorithms run stepwise on the tape (messages, eliminations, sweeps).

## Absent and partial items

Priority is high, medium or low by how many learners would expect the thing, how many pages it unlocks and how well it shows off a stepped, traced run; not by recency. Sizes: S a day or less, M a few days, L a week or more. Homes follow `docs/structure.md`. "Nearest in omai" is what a user would reach for today. Items are grouped; the full list of names per group is in `data/match-<lib>.json`.

### pgmpy

| reference (area)                                                                                                                                                                                                             | what it is and why it matters                                                                                                                            | nearest in omai                                               | proposed home                 | size | priority | status  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------- | ---- | -------- | ------- |
| **classic Bayesian network benchmarks**: `get_example_model` (utils)                                                                                                                                                         | the classic benchmark networks (asia, cancer, earthquake, sachs, alarm, insurance, child) that textbooks use; none ships with omai                       | datasets                                                      | datasets/bayes-nets.ts        | M    | high     | absent  |
| **Bayesian network file formats**: `BIFReader`, `BIFWriter`, `NETReader`, `NETWriter`, `PomdpXReader`, `PomdpXWriter`, `UAIReader`, `UAIWriter`, `XBNReader`, `XBNWriter`, `XDSLReader`, `XDSLWriter` and 2 more (readwrite) | BIF, XMLBIF, UAI, NET, XDSL and XBN readers and writers: exchange formats for published Bayesian networks                                                | pgm#toStructure                                               | pgm/io.ts (BIF and UAI first) | M    | medium   | absent  |
| **Bayesian structure scores**: `AICCondGauss`, `AICGauss`, `BDeu`, `BDs`, `BICCondGauss`, `BICGauss`, `K2`, `LogLikelihoodCondGauss`, `LogLikelihoodGauss` (estimators)                                                      | BIC, AIC and likelihood scores exist; K2 and the Bayesian Dirichlet family (BDeu, BDs) are not named                                                     | pgm/learning#structureScore, #discreteScorer, #gaussianScorer | pgm/learning/scores.ts        | S    | medium   | partial |
| **model checking**: `AdjacencyConfusionMatrix`, `CorrelationScore`, `FisherC`, `ImpliedCIs`, `OrientationConfusionMatrix`, `get_metrics` (metrics)                                                                           | testing a model against data (implied conditional independences, Fisher's C) and scoring a learnt graph by adjacency or orientation precision and recall | pgm/causal#structuralHammingDistance                          | pgm/learning/model-checks.ts  | S    | medium   | absent  |
| **structure-learning odds and ends**: `ExpertInLoop`, `ExpertKnowledge`, `llm_pairwise_orient`, `manual_pairwise_orient` (causal_discovery, utils)                                                                           | exhaustive DAG enumeration (small n), MMHC (hybrid), whitelists and blacklists of edges                                                                  | pgm/learning#hillClimb                                        | pgm/learning/hybrid.ts        | M    | medium   | absent  |
| **noisy-OR CPDs**: `NoisyORCPD` (factors.discrete)                                                                                                                                                                           | noisy-OR and noisy-MAX CPDs (compact conditional tables for many parents); a standard teaching device                                                    | pgm#bayesNode                                                 | pgm/noisy-or.ts               | S    | medium   | absent  |
| `GCM`, `GeneralizedCov`, `HotellingLawley`, `PillaiTrace`, `RoysLargestRoot`, `WilksLambda` (ci_tests)                                                                                                                       | CI tests built from multivariate statistics; Hotelling and MANOVA exist in stats                                                                         | stats/tests#hotellingT2, stats/tests#manova                   | pgm/learning/ci-tests.ts      | S    | low      | absent  |
| **latent-variable graphs**: `ADMG`, `AncestralBase`, `MAG`, `SimpleCausalModel` (base)                                                                                                                                       | latent-variable graphs (ADMG, MAG); PAG and FCI exist                                                                                                    | pgm/causal#Pag                                                | pgm/causal                    | M    | low      | absent  |
| **structural equation models**: `IVEstimator`, `SEMEstimator`, `SEMAlg`, `SEMGraph` (estimators, models)                                                                                                                     | linear SEMs appear as linearGaussianBayesNet and structural causal models; no path-diagram SEM with fit indices                                          | pgm/causal#scm, causal#twoStageLeastSquares                   | pgm/sem.ts                    | M    | low      | partial |
| `MirrorDescentEstimator` (estimators)                                                                                                                                                                                        | estimating a model from marginals (private synthetic data); niche                                                                                        | nothing close                                                 | to be placed                  | -    | low      | absent  |

### hmmlearn

| reference (area)                                                                   | what it is and why it matters                                                                                                        | nearest in omai                      | proposed home                   | size | priority | status  |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ | ------------------------------- | ---- | -------- | ------- |
| `GaussianHMM`, `PoissonHMM` (hmm)                                                  | continuous and count emissions are available through the emission slot of the HMM; check that Gaussian and Poisson are named options | pgm/learning#baumWelch, #HmmEmission | to be placed                    | -    | low      | partial |
| **variational HMMs**: `VariationalCategoricalHMM`, `VariationalGaussianHMM` (vhmm) | variational Bayes for HMMs (VB Baum-Welch)                                                                                           | nothing close                        | pgm/learning/hmm-variational.ts | M    | low      | absent  |
