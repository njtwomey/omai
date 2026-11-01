# Exploration and discovery (part 3, data)

A dataset explored before any model is fitted: profiling (per-column summaries, missingness, duplicates, a data-bias audit), views (pairwise plots and correlation maps, slices), dependence between every pair with its tests, pattern mining (frequent itemsets and rules, sequential patterns, subgroup discovery) and causal discovery on data from a known graph (constraint-based, score-based, functional), as sessions over `stats` (correlation, tests, densities), `information` (estimators), `text` (fingerprinting), `fairness` (data bias), `pgm` (structure learning, causal discovery) and the pattern-mining algorithms still to come in `lib/patterns`.

## Structure

The chapter lives under `data/exploration/` (section 3 of `methods/README.md`), one folder per activity. Embedding and clustering for overview are `fit/reduce/` and `fit/cluster/` (the unsupervised chapter's) and appear here only as `see` rows; causal discovery sits here rather than in `causal-inference` because its comparison is with the planted graph, and that chapter carries a `see` row for it.

```
data/exploration/
  profiling/         per-column summaries and histograms, missingness, cardinality, duplicates, a data-bias audit
  views/             pairwise views and correlation maps, slices and subgroups; embeddings for overview are fit/reduce/'s
  dependence/        measures between every pair estimated as samples arrive; independence tests with their nulls
  patterns/          frequent itemsets and association rules, sequential patterns, subgroup discovery
  causal-discovery/  constraint-based (PC, FCI), score-based (hill climbing, GES, Chow-Liu), functional (LiNGAM, NOTEARS), on data from a known graph
  shared/            planted structure (a DAG with its data, transactions with planted patterns, planted dependences), the graph-against-truth and lattice records (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a profile is one column; of an estimator, one batch of samples; of a test, one permutation; of a miner, one level, one prefix or one beam expansion; of a discovery algorithm, one test, one move or one ordered variable.

### profiling/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `profile` | A dataset from the registry profiled one column per step: type, summary statistics, a histogram or the level counts, the missing rate, cardinality, skew; the table of columns as the picture; the same profile on a planted problem shows what the modifiers did. | `column/index` s, `column/summary` m, `column/histogram` t!, `profile/table` p, `profile/missingMap` p | `stats/descriptive.ts`: `describe`, `skewness`, `kurtosis`; `stats/histograms.ts`: `histogram`; `stats/quantiles.ts`: `quantile`, `iqr`; `data/tensors.ts`: `describe`, `table`; `learn/preprocessing/imputation.ts`: `missingIndicator` | |
| `duplicates` | Exact and near duplicates found: shingles and MinHash signatures with the bands and rows as knobs (the collision probability's S-curve as a probe), the candidate pairs against the planted duplicates (precision and recall), and across a split as the leak they cause. | `hashing/bands` s, `hashing/candidates` s, `duplicates/pairs` t!, `duplicates/precision` s, `duplicates/recall` s, `hashing/sCurve` p | `text/fingerprinting.ts`: `nearDuplicates`, `minhash`, `minhashJaccard`, `lshParameters`, `lshCandidates`, `simhash`; `text/shingles.ts`: `shingles`; `numerics/spatial/hashing.ts`: `lshIndex`, `bandingProbability`; `datasets/corpus.ts`: `nearDuplicateCorpus` | |
| `data-bias-audit` | A dataset with a protected attribute audited before any model: the representation of each group, base rates and their gap, label bias against the clean labels the generator gives, proxy features scored by how well they predict the group, the amplification a model would add; the knobs of `biasedCredit` planting each bias in turn. | `audit/representation` t!, `audit/baseRates` t!, `audit/labelBias` s, `audit/proxies` t!, `audit/record` p (the `dataBias` record) | `fairness/data-bias.ts`: `representation`, `baseRates`, `labelBias`, `proxyScores`, `biasAmplification`; `exposition/data-bias.ts`: `dataBias`; `datasets/fairness.ts`: `biasedCredit`, `germanCredit` | |
| `see trustworthy-machine-learning` | Fairness of a fitted model (group reports, mitigation, frontiers) is that chapter's; the audit above is the data side. | | | |

### views/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `pairwise-views` | Every pair of columns as a scatter or a 2-D histogram with a kernel density (the bandwidth as the knob), the diagonal's marginals, the correlation map beside it; the pair the cursor is on enlarged with its measures. | `pair/indices` m, `pair/density` t!, `pair/correlation` s, `views/grid` p, `correlation/matrix` p | `stats/histograms.ts`: `histogram2d`; `stats/density.ts`: `gaussianKde`, `bandwidthScott`, `bandwidthSilverman`; `stats/correlation.ts`: `corrcoef` | |
| `slices` | Rows cut by a column or a pair (quantile bins as the knob): the target's rate or mean per slice with its interval, the slices with the most rows and the most extreme values as a grid; the subgroup record at the viewed step. | `slice/index` s, `slice/size` s, `slice/rate` s, `slice/interval` t, `slices/grid` p | `exposition/subgroups.ts`: `subgroupReport`, `subgroupGrid`; `stats/tests/proportions.ts`: `proportionConfint`; `stats/histograms.ts`: `binnedStatistic` | |
| `see unsupervised-learning` | Embeddings for overview (PCA, t-SNE and UMAP moving step by step) are `fit/reduce/`'s; a view here embeds the registry's dataset through them. | | | |

### dependence/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `dependence-measures` | Planted relations (linear, quadratic, a circle, a sinusoid, independent, with the noise as the knob) and the measure as a mode: Pearson, Spearman, Kendall, distance correlation, mutual information (KSG), HSIC, Hoeffding's D; estimated as samples arrive one batch per step and converging to the population value (known in the linear and Gaussian cases); correlation missing what mutual information finds; the matrix of one measure over all pairs of a real dataset. | `estimate/value` s, `estimate/n` s, `estimate/error` s, `pair/picture` p, `measures/matrix` p (the `dependence` record), `study/convergence` proj | `stats/correlation.ts`: `pearson`, `spearman`, `kendallTau`, `distanceCorrelation`, `partialCorrelation`; `information/neighbour-estimators.ts`: `ksgMutualInformation`; `stats/tests/kernel.ts`: `hsic`, `medianHeuristic`; `stats/tests/independence.ts`: `hoeffdingD`, `mutualInformation`; `exposition/dependence.ts`: `dependence` | |
| `independence-tests` | The test as a mode (chi-square and G on a contingency table, the permutation test of any measure with its null built one permutation per step, the distance-covariance test, HSIC's test, the kernel conditional-independence test, the partial-correlation test): the statistic against its null forming, the p-value settling, power against the planted strength and n; many pairs corrected (Bonferroni, Holm, Benjamini-Hochberg as modes). | `null/statistics` t!, `test/statistic` s, `test/pValue` s, `test/permutation` s, `study/power` proj, `tests/corrected` t! | `stats/tests/contingency.ts`: `chi2Contingency`, `gTest`; `stats/resampling.ts`: `permutationTestAlgorithm`; `stats/tests/independence.ts`: `distanceCovarianceTest`, `mutualInformationTest`, `partialCorrelationTest`, `kernelCITest`, `gTestConditional`; `stats/tests/kernel.ts`: `hsicTest`; `stats/tests/multiple.ts`: `multipleTests`, `benjaminiHochberg`, `holm` | |

### patterns/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `frequent-itemsets` | Transactions with planted patterns; the algorithm as a mode: Apriori one level per step (candidates generated by joining, pruned by the downward closure, counted; the lattice drawn with the frequent sets lit), FP-growth's tree built one transaction per step and mined by conditional trees, Eclat's vertical intersections; the support threshold as the knob and the count of frequent sets against it; the planted patterns recovered. | `level/k` s, `level/candidates` s, `level/frequent` s, `level/pruned` s, `itemsets/lattice` p, `fptree/structure` p, `patterns/recall` s, `study/bySupport` proj | TODO `lib/patterns/itemsets.ts`: `aprioriAlgorithm`, `fpGrowthAlgorithm`, `eclat`, `supportCount`; TODO `datasets/transactions.ts`: `plantedTransactions` (the patterns and their supports as the truth) | |
| `association-rules` | Rules from the frequent sets: confidence, lift, leverage and conviction (the measure as a mode) with their thresholds as knobs, the rules ranked, the trivial ones (high confidence from a frequent consequent) exposed by lift; the rules as a table at the viewed threshold. | `rules/count` s, `rules/table` t!, `rules/thresholds` m, `rules/scatter` p (support against confidence, coloured by lift) | TODO `lib/patterns/rules.ts`: `associationRules`, `ruleMeasures` | |
| `sequential-patterns` | Sequences of itemsets: PrefixSpan growing patterns by projected databases (one prefix extension per step), GSP's level-wise candidates as the comparison; the gap constraint as the knob; the patterns against the planted ones. | `prefix/pattern` m, `prefix/projectedSize` s, `patterns/found` t!, `patterns/recall` s | TODO `lib/patterns/sequences.ts`: `prefixSpanAlgorithm`, `gsp`; planted sequences in `shared/structure.ts` | |
| `subgroup-discovery` | A beam search over descriptions (conjunctions of conditions on columns; the beam width and depth as knobs) scored by a quality measure as a mode (weighted relative accuracy, a binomial test, the mean shift of a numeric target), one beam expansion per step: the best subgroups so far with their size and quality, their rows coloured on the data, redundant descriptions pruned; the planted subgroup recovered. | `beam/depth` s, `beam/candidates` s, `subgroups/best` t!, `subgroups/quality` s, `subgroups/report` p (the `subgroupReport` record), `subgroups/rows` p | `exposition/subgroups.ts`: `subgroupReport`; `explain/rules.ts`: `ruleMask`; TODO `lib/patterns/subgroups.ts`: `subgroupDiscoveryAlgorithm`, `wracc` and the other quality measures | |
| `see time-series` | Motifs, discords and chains in a series are the matrix-profile family's (`process/signals/series/matrix-profile/`). | | | |

### causal-discovery/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `constraint-based` | Data from a planted DAG (its size, density and the sample size as knobs): PC's skeleton phase one conditional-independence test per step (the edge removed, the separating set recorded, the conditioning sets growing), then the orientation phase (v-structures, Meek's rules as events); the test as a mode (chi-square, G², Fisher's z, kernel CI, the oracle from the true graph); the structural Hamming distance and the CPDAG against the truth; FCI as the mode that admits latent confounders and bidirected edges. | `test/pair` m, `test/conditioning` t, `test/pValue` s, `skeleton/edges` t!, `orientation/rule` e, `graph/structure` p (found against true), `graph/shd` s, `tests/count` s | `pgm/learning/structure.ts`: `pcAlgorithm`, `pc`; `pgm/learning/independence.ts`: `chiSquareTest`, `gSquareTest`, `fisherZTest`, `independenceOracle`; `stats/tests/independence.ts`: `kernelCITest`; `pgm/causal/fci.ts`: `fci` (whole; TODO a stepped form); `pgm/learning/equivalence.ts`: `cpdag`, `meekRules`, `structuralHammingDistance`, `isMarkovEquivalent`; `datasets/causal.ts`: `linearScm`; `pgm/models/bayes-net.ts`: `ancestralSample` | |
| `score-based` | Hill climbing over DAGs one move per step (add, remove, reverse; the score's local decomposition as the saving; tabu as the knob), GES's forward and backward phases over equivalence classes, Chow-Liu's maximum-weight spanning tree of mutual informations as the tree mode; the score as a mode (BIC, BDeu with its equivalent sample size, the Gaussian score); the score climbing, the graph against the truth. | `move/kind` m, `move/delta` s, `score/value` s, `graph/structure` p, `graph/shd` s, `ges/phase` e | `pgm/learning/structure.ts`: `hillClimbAlgorithm`, `chowLiu`, `mutualInformationOf`; `pgm/causal/discovery.ts`: `gesAlgorithm`; `pgm/learning/scores.ts`: `discreteScorer`, `gaussianScorer`, `structureScore`, `localScore`, `structurePrior`; `pgm/learning/equivalence.ts`: `pdagToDag` | |
| `functional` | Non-Gaussian noise breaks the equivalence class: DirectLiNGAM ordering the variables one per step by the independence of residuals (the ordering as the output, the mixing matrix against the truth); NOTEARS as the continuous mode, the acyclicity penalty falling over the augmented Lagrangian's iterations, the weighted adjacency thresholded; both against PC on the same data, with Gaussian against non-Gaussian noise as the knob. | `order/variables` t!, `order/step` s, `solver/objective` s, `solver/acyclicity` s, `graph/weights` t!, `graph/structure` p, `graph/shd` s | `pgm/causal/discovery.ts`: `directLingam`, `notearsAlgorithm`, `notears`; `datasets/causal.ts`: `linearScm` (Gaussian or non-Gaussian noise) | |
| `see causal-inference` | Identification (adjustment sets, d-separation) and effect estimation on a known graph are the causal chapter's; discovery ends at the graph. | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `structure.ts` | Planted structure with its truth: a DAG with its data (`linearScm`, the discrete networks of `probabilistic-inference`'s `networks.ts` sampled by `ancestralSample`), transactions and sequences with planted patterns (to add), a planted subgroup, planted dependences of each shape with their population values; shared with `causal-inference` | `dependence/`, `patterns/`, `causal-discovery/` |
| `graphs.ts` | The graph-against-truth record: a `Structure` with every edge marked found, missed or wrong (and its orientation), the SHD and the equivalence class beside it, at the viewed step | `causal-discovery/` |
| `lattice.ts` | The itemset lattice and the FP-tree as records (a level-wise board, a tree with counts), the rule table | `patterns/` |

## Examples to make

- Apriori: candidate itemsets generated and pruned level by level, support as the knob.
- PC on data from a planted DAG: edges removed test by test, then oriented, against the truth; GES on the same data beside it.
- Dependence measures on nonlinear relations as samples arrive: correlation misses what mutual information finds.
- Subgroup discovery: the best subgroups and their quality as the beam advances.
- The data-bias audit of `biasedCredit` with each planted bias turned on in turn.

## Rendering

- Tables, graphs with edges appearing and orienting (structure view), heatmaps of dependence, histograms and 2-D densities.
- Missing: a lattice or level-wise panel for itemsets; the graph-against-truth overlay (true, found and wrong edges); a pair-plot grid panel; a profile table panel.

## Problems

- `patterns/` and the discovery half of `causal/` of the old plan move here. Pattern mining is not in the library: `lib/patterns/` (itemsets, rules, sequences, subgroups) comes first, with a transaction generator in `datasets/`. Everything else named here exists, with `fci` wanting a stepped form.
- Exploration by embedding and by clustering is `fit/reduce/` and `fit/cluster/`; this chapter only points at them, so `views/` is thin on purpose.
- Two kinds of page: one-shot views with a knob (profiling, pairwise views) and iterations watched (PC's tests, Apriori's levels, the beam); the first kind runs as one-step sessions.
- Causal discovery is shared with `causal-inference`; it sits here because its comparison is with the planted graph, and that chapter carries a `see` row for discovery. If the causal chapter would rather own it, the group moves whole.
