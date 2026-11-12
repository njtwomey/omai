# Bayesian trees: one split at a time, and all trees at once

Design note for `packages/engine/src/lib/trees/bayesian/` (exported flat from `omai-engine/trees`) and the showcase story
`apps/showcase/src/stories/trees-one-split-at-a-time/`. Status: first piece, built 2026-11-12.

## 1. The idea, as the owner put it, and as read here

The owner's words: "I have an idea about trying to find posteriors around a decision tree. It's kind of like a Bayesian,
tree-based version of AdaGrad. So not boosting, but the things should work. I think we could even do things either in a
myopic or full manner (i.e. greedily learn it, vs. learning the full posterior over the whole thing)."

The reading, in three parts.

1. **Every leaf keeps a posterior.** A tree cuts the input space into regions (its leaves). Each leaf holds a posterior over a
   small parameter: a mean (with a noise level) for regression, class probabilities for classification. The posterior is
   updated as data arrive. Its precision (the evidence the leaf holds, in observations) only grows. The **step size of a
   leaf is the reciprocal of that precision**, so a rarely visited region keeps large steps and a well visited one takes
   small steps. This is the AdaGrad-like part (section 4 says exactly how far the analogy goes).
2. **One tree, learned, not boosted.** There is no additive sequence of trees. The structure is the unknown.
3. **Two ways to learn the structure.**
   - **Myopic:** grow one split at a time, each time taking the split that most raises the exact marginal likelihood, and
     stop when no split pays for itself under the prior.
   - **Full:** a posterior over whole trees (structure and leaf parameters together), a distribution that predictions are
     averaged over and trees are drawn from.

## 2. The model

**Data.** Rows `(x_i, y_i)`, `x_i` in R^d. A tree `T` is a set of splits `(feature j, threshold t)`; a row goes left when
`x_j <= t`. The leaves partition the rows.

**Leaf models (conjugate, so everything below is exact).** Leaf parameters are integrated out; the tree carries none.

- Regression, Normal-Gamma: `y | mu, tau ~ N(mu, 1/tau)`, `mu | tau ~ N(m0, 1/(kappa0 tau))`, `tau ~ Gamma(alpha0, beta0)`.
  Statistics of a leaf: `n`, `sum z`, `sum z^2` with `z = y - m0`. Posterior: `kappa = kappa0 + n`, `m = m0 + sum z / kappa`,
  `alpha = alpha0 + n/2`, `beta = beta0 + (sum z^2 - (sum z)^2 / kappa) / 2`. Exact evidence of the rows of a leaf:
  `lgamma(alpha) - lgamma(alpha0) + alpha0 log beta0 - alpha log beta + (log kappa0 - log kappa)/2 - (n/2) log 2 pi`.
  The mean function of a leaf has a Student-t posterior (variance `beta / ((alpha-1) kappa)`); a new observation a wider one.
- Classification, Dirichlet-multinomial (Beta-Bernoulli for two classes): statistics are the class counts `n_k`; posterior
  concentration `a_k + n_k`; evidence `lgamma(A) - lgamma(A+n) + sum_k [lgamma(a_k+n_k) - lgamma(a_k)]`.

**Evidence of a tree.** The sum over leaves of the leaf evidences: `log p(y | T) = sum_leaf log p(y_leaf)`.

**Prior over structures** (Chipman, George and McCulloch 1998). A node at depth `d` splits with probability
`p_d = alpha (1+d)^(-beta)`; a split's feature and threshold are uniform among the rules available at that node (those that
leave at least `minLeaf` rows on each side); a node with no available rule, or at the depth limit, is a leaf for certain.
`p(T) = prod_internal p_d / R_node * prod_leaf (1 - p_d)` with `R_node` the number of available rules. Defaults:
`alpha = 0.5` (Chipman et al.'s 0.95 suits ensembles of small trees and splits pure noise in a single tree), `beta = 2`,
`maxDepth = 3`, `minLeaf = 2`. In the exact and sampled modes the candidate rules are a finite set (a few quantile
thresholds per feature); the myopic mode can search every threshold.

## 3. The two learning modes

**Myopic (`growBayesTree`, `bayesGrowAlgorithm`).** Every leaf holds its best split, found over all features and all
thresholds at once. The rows of the node are sorted per feature (one `argsort`), their per-row statistics gathered in that
order and accumulated (one `cumsum`), so the left and right statistics at every threshold are one tensor, and the exact
log Bayes factor of every split against no split is two leaf evidences minus the parent's, evaluated on the whole tensor.
The log posterior odds of splitting leaf `l` by its best rule are

`log p_d - log(1 - p_d) - log R + 2 log(1 - p_{d+1}) + log BF`,

the prior odds, one over the number of rules chosen from, the children's chance of staying leaves, and the Bayes factor.
Each step splits the leaf with the largest odds; the run stops when none is above zero. Every decision (node, rule, gain,
odds, evidence of the tree after it) is recorded, so a story can replay the growth. This criterion judges the best
single tree against the tree without the split. Judging "some split at this node" by the average Bayes factor over the
rules is a different, more generous stopping rule (not implemented). The myopic rule cannot see a split that only pays
through the splits below it (the XOR case); that is its price.

**Streaming (`streamUpdate`, `streamAlgorithm`).** Rows arrive one at a time or in batches. Each is routed to its leaf and
the leaf's statistics are updated **by addition**; so the leaf posterior after any prefix of the stream equals the batch
posterior of the rows seen, to rounding (tested). After `checkEvery` new rows a leaf asks the myopic question on the rows it
holds and splits when the odds are above zero; its children ask in turn. The leaves keep their rows (the stream is held in
memory, per leaf); a constant-memory version, keeping per-candidate statistics per leaf, is possible but the children of
a split would then start without the past, as in the Hoeffding tree, so it is not what is built here.

**Full: exact (`enumerateTrees`).** For depth at most two over a finite rule set `C`, every tree is a root rule `r` and, for
each child, a leaf or a rule: `C (1+C)^2` trees plus the single leaf. Given `r` the two subtrees are independent, so the log
weights (prior times evidence) are a sum of three vectors, `[C, 1+C, 1+C]` in one tensor; the normaliser is one
`logsumexp`; the posterior of the root rule, the use of each rule, the probability of 0 to 3 splits and the exact
posterior-predictive average over all trees are tensor sums over region summaries, not loops over trees. Regions (the rows
reaching a node, as a 0/1 indicator, remembered under the path of rules that leads there) are computed once, with one
matrix product over all rules.

**Full: sampled (`sampleTrees`).** Metropolis-Hastings over structures with grow, prune and change moves (Chipman et al.'s
swap is left out). Leaf parameters are integrated out, so the chain moves on structures and the acceptance ratio is exact:
`posterior(T')/posterior(T) * q(T|T')/q(T'|T)`, where the proposal probabilities depend on the move and on the numbers of growable
leaves, prunable nodes, internal nodes and available rules (the single-leaf tree can only grow). Several chains from the
single leaf; acceptance rate per move kind, split R-hat and ESS of the log posterior and the leaf count are reported.
Works for deeper trees where enumeration does not. The sampler is checked against the enumeration (section 6).

**Prediction.** A leaf's posterior predictive is Student-t (regression) or the Dirichlet mean (classification). The summary
of a leaf (mean, second moments, class probabilities) is linear, so the posterior prediction of the full model is the
probability-weighted average of the summaries over trees; the mean function's variance is read from the averaged second
moment, so it includes both the within-leaf uncertainty and the disagreement between trees.

## 4. The AdaGrad analogy, stated precisely

For the mean of a leaf with the noise precision known, the posterior after `t` rows has precision `kappa_t = kappa_0 + t`
and mean `m_t = m_{t-1} + (y_t - m_{t-1}) / kappa_t`. That is a gradient step on the squared error with step size
`1 / kappa_t`, where `kappa_t` is the **accumulated Fisher information** of the leaf (a sum of ones, one per row). The
Dirichlet leaf has the same form: step `1 / (A + n)`. What accumulates in a leaf, and the rates it gives:

- the count (precision of the mean): `kappa = kappa0 + n`, step size `1/kappa`;
- the sum of squared residuals: `beta = beta0 + (sum z^2 - (sum z)^2/kappa)/2`, which sets the noise scale `beta/(alpha-1)`
  that the predictive spread uses. This is the part that looks like AdaGrad's accumulated squared gradients (the residual
  is the gradient of the Gaussian loss); it scales the uncertainty, not the step.

AdaGrad divides by the square root of accumulated squared gradients, per coordinate, to get a per-coordinate rate. Here
the divisor is the accumulated information itself (power one, not one half), per region instead of per coordinate. The
Bayesian recursion is the exact version of what AdaGrad approximates: for a conjugate leaf it is a Kalman filter / online
Newton step with the true curvature, and it is the case where the natural-gradient step on the exact posterior is exact
(Jones, Chang and Murphy 2024, BONG). So "per-region adaptive step size = posterior precision" is an exact and
well-defined statement for a leaf; it is the AdaGrad idea with the square root and the gradient proxy replaced by the real
information.

## 5. What is known, and what, if anything, is new

Known, and not claimed as new:

- **Bayesian trees over structure.** Bayesian CART: Chipman, George and McCulloch (1998) and Denison, Mallick and Smith
  (1998), with this prior, these conjugate leaves, and Metropolis-Hastings over grow, prune, change and swap. BART (2010)
  is a sum of such trees with a shrinking prior (a boosting-like model; not this). Bayesian treed models: Chipman et al.
  (2002), treed Gaussian processes (Gramacy and Lee 2008).
- **Exact averaging over tree structures.** Buntine (1992) averaged over trees; context-tree weighting (Willems, Shtarkov and
  Tjalkens 1995) computes the exact Bayesian mixture over all pruned trees of a fixed tree by recursion. The exact depth-two
  posterior here is a small instance of the same idea (a posterior over structures from region evidences).
- **Online and streaming Bayesian trees.** Dynamic trees (Taddy, Gramacy and Polson 2011): sequential Monte Carlo over tree
  structures with conjugate leaves updated as data arrive, the closest relative of the streaming mode. Mondrian trees and
  forests (Roy and Teh 2008; Lakshminarayanan et al. 2014): a Bayesian nonparametric prior over partitions that grows
  online, with a hierarchical leaf posterior. Hoeffding trees (Domingos and Hulten 2000) split online with a frequentist
  bound.
- **Adaptive step sizes as Bayesian updates.** AdaGrad, RMSprop and Adam as approximations of natural-gradient or
  Bayesian-learning-rule updates (Khan et al. 2018; Khan and Rue 2023); online Bayesian updates as Kalman filters (BONG,
  Jones et al. 2024).

What the combination adds, stated carefully: nothing at the level of a new algorithm or theorem. A search for "Bayesian
AdaGrad" and for a per-region adaptive rate in Bayesian trees found no paper that states it. What is a framing and a
construction, not a result:

1. The reading of a conjugate Bayesian treed model as a **per-region adaptive-rate learner**: the leaf's precision is its
   AdaGrad-like accumulator and its step size is the reciprocal, shown on the tape per leaf, so the regions with little
   evidence are visibly the ones that move.
2. **Both modes on one scoring function**: the myopic grower, the exact posterior over all trees and the sampler share one
   prior and one evidence, so the single greedy tree can be placed inside the posterior it came from (its posterior
   probability, how far the cut could be moved) and the sampler can be checked against an exact enumeration.
3. **Tensor form of the search**: all features and all thresholds of a node, and the whole depth-two posterior, in a
   handful of tensor operations.

It is a worked, tested combination of known pieces. It should not be called a new method.

## 6. What is exact, what is sampled, what is tested

Exact: leaf posteriors and evidence; the evidence and prior of any tree; the myopic search over every threshold; the
streaming leaf posterior; the depth-two posterior over trees and its predictive average. Sampled: the posterior over
trees of any depth (`sampleTrees`). Tests: leaf evidence against the sequential-predictive product, a numerical integral
over the mean and precision, and the Beta-Bernoulli and Dirichlet closed forms; the tree prior against hand products; the
enumeration weights against a brute-force sum over all trees, summing to one, equal to the score of each listed tree;
the exact predictive against a brute-force average; the greedy choosing the known best split, both steps of a staircase,
no split on noise, each decision's gain equal to the rise of the tree evidence; the all-threshold search equal to the
candidate-table route; streaming equal to batch for any batching; the sampler against the enumeration on the root-rule
posterior and the predictive curve (loose tolerance, fixed seed).

## 7. What remains open

- Exact enumeration beyond depth two (the factorisation continues, the number of regions grows as `C^depth`).
- Swap moves, tempering, or a better kernel for the slow mixing between the two modes of a staircase (about 5 percent
  acceptance for grow and prune in the test problem).
- Learning `alpha`, `beta`, the leaf hyperparameters and the noise (here fixed, with data-scale defaults).
- Constant-memory streaming with inherited child posteriors; forgetting (a discount on the statistics) for drift.
- Sums of trees (BART) and the comparison with boosting. Multi-feature and categorical splits in the full mode (the
  machinery is feature-generic; only 1-D and two-feature problems are exercised).
- A comparison against the `lib/trees` CART and against BART on a benchmark; none was run.
