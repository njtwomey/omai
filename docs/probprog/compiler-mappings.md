# What the compiler maps

Generated from the mapping tables of `packages/engine/src/lib/probprog/model/factor-graph/_table.ts` by `make ep-compiler-doc` (a test checks that it is
current): edit the tables, not this page. A compilation runs in passes: the events and scans are unrolled, the constants are found (`_known.ts`),
the layout plan is made (`_plan.ts`), the nodes are walked in spec order through the tables below, the dead `apply` factors are pruned, and the
opaque nodes are reported. In a table the first entry whose pattern the node is maps it; an entry that can only tell by trying answers `NOT_MINE`,
and the graph is then restored to what it was (`_mapping.ts`). The operators of an entry are the ones it emits itself; every mapping can also
cause the operators of the value algebra (`apply`, `apply-product`, `difference`, `dot-constant`, `equal`, `gamma-product`, `linear`, `max`, `min`, `plus`, `product`, `select`, `sum`, `times-constant`).

## Nodes

What the walk does with each kind of node of the spec, in order. The latents and the observed nodes go to the tables below.

| # | mapping | families | what it maps | operators | order |
| --- | --- | --- | --- | --- | --- |
| 1 | `data` |  | bound data is known from the start (the constants pass): nothing to map |  |  |
| 2 | `non-centred-absorbed` |  | a standard normal vector that a non-centred derived node absorbs: the derived node compiles it, only if the dense factor does not apply |  | before the latent mappings: the vector has no mapping of its own |
| 3 | `latent` |  | a latent node: the first mapping of the latent table that takes it |  |  |
| 4 | `derived` |  | a derived node: its expression evaluated in the value algebra (a known constant, a symbolic value, a recorded variable), or a non-centred vector z / S(x) as one Gaussian vector with a gaussian-mv-function-scale factor (one shared scale) or assembled from its scalar products by a gaussian-mv-stack factor (a scale each) | `gaussian-mv-function-scale`, `gaussian-mv-stack` |  |
| 5 | `observed` |  | an observed node: the first mapping of the observed tables that takes it |  |  |
| 6 | `generated` |  | a generated node (a prediction that is not part of the model): nothing to map |  |  |
| 7 | `scan` |  | a scan, which the driver has unrolled into one variable and factor per step before the walk: nothing to map |  |  |
| 8 | `param-refused` |  | a param (a learnable value, not a random variable) has no factor: refused |  |  |
| 9 | `factor-refused` |  | an arbitrary log-density term has no message operator: refused |  | last: takes every kind of node that is left |

## Latent nodes

The first entry that takes a latent node maps it. The refusals sit where a later entry could otherwise take the node.

| # | mapping | families | what it maps | operators | order |
| --- | --- | --- | --- | --- | --- |
| 1 | `constraint-refused` |  | a constraint beyond the family support (a simplex, a positive-definite matrix) is refused, but for a truncated normal |  | first: a constrained node is never mapped by the family entries below |
| 2 | `projected-prior` | halfNormal, halfCauchy, logNormal, uniform, truncatedNormal | a half-normal, half-Cauchy, log-normal, uniform or truncated normal prior, projected onto a Gaussian (on the log of a positive scale) | `prior`, `prior-site` | before the family lookup: these families have no message family of their own |
| 3 | `wishart` | wishart, inverseWishart | a Wishart or inverse-Wishart matrix: one variable of the Wishart family, read only as the precision or covariance of a multivariate normal | `prior` |  |
| 4 | `correlation` | correlation | a correlation matrix of order two, as the beta variable of (rho + 1) / 2 | `prior` |  |
| 5 | `bivariate-normal-refused` | bivariateNormal | a latent bivariate normal is refused: only observed vectors are mapped |  |  |
| 6 | `mvn-wishart` | multivariateNormal | a multivariate normal whose precision (or covariance) is a Wishart latent: an mvn-wishart factor per vector for a known mean, one mvn-wishart-group factor for all the vectors when the mean is a Gaussian vector variable | `mvn-wishart`, `mvn-wishart-group`, `prior` | before the plain multivariate normal below |
| 7 | `normal-gamma-mean` |  | the Gaussian mean of an activated normal-gamma pair: it has no variable of its own, the pair is the gamma precision variable |  | before the gamma entry: the pair is activated by the precision, which the walk reaches first |
| 8 | `normal-gamma-precision` | gamma, inverseGamma | a gamma or inverse-gamma precision (or variance) that a Gaussian mean and its observations share: one normal-gamma variable with a gaussian-normal-gamma factor per observation | `gaussian-normal-gamma`, `prior` | before the family lookup (the pair replaces the gamma and the mean); NOT_MINE when the plan found no pair, then the gamma is an ordinary one |
| 9 | `family-refused` |  | a family with no message family and no projection is refused, naming the engine families |  | after the entries of the families that have no message family of their own, before the message families |
| 10 | `normal` | normal | a normal: a Gaussian variable per element, with a known or random mean and scale (a precision, a random scale, a switch), grouped or packed where the plan says; a truncation constrains it | `gate`, `gaussian`, `gaussian-gamma-precision`, `gaussian-mv-precision-scale`, `greater-than`, `prior` |  |
| 11 | `multivariate-normal` | multivariateNormal | a multivariate normal with a known or a random mean vector: one Gaussian vector variable | `gaussian-transition`, `prior` |  |
| 12 | `spike-and-slab` | normalSpikeSlab | a spike-and-slab normal: a Gaussian that a Bernoulli inclusion switches between a narrow and a wide component | `bernoulli`, `gate`, `gaussian-gamma-precision`, `mixture-row`, `mixture-site`, `prior` |  |
| 13 | `discrete` | bernoulli, categorical | a Bernoulli or categorical: a discrete variable with a prior, a table over its discrete parents, a gate for a switch, or Dirichlet rows | `bernoulli`, `discrete`, `gate`, `prior`, `stick-breaking`, `table` |  |
| 14 | `dirichlet-random-concentration` | dirichlet | a Dirichlet whose concentration is random (a symmetric strength with a gamma or log-normal prior): a dirichlet-concentration factor | `dirichlet-concentration`, `prior` | before the Dirichlet with a known concentration |
| 15 | `dirichlet` | dirichlet | a Dirichlet with a known concentration: one Dirichlet variable per element with its prior | `prior` |  |
| 16 | `beta-random-shapes` | beta | a beta with a random shape (a log-normal or gamma strength): a beta-two-shapes factor | `beta-two-shapes`, `dirichlet-concentration`, `prior` | before the beta with known shapes in the closed-form entry at the end |
| 17 | `gamma-random-rate` | gamma, inverseGamma | a gamma or inverse gamma whose rate (or scale) is a known multiple of another gamma latent: a gamma-rate factor | `gamma-rate`, `prior` | before the gamma with known parameters in the closed-form entry at the end |
| 18 | `known-prior` | beta, gamma, inverseGamma, exponential | a beta, gamma, inverse-gamma or exponential with known parameters: one variable per element with its prior | `prior` | last: what is left of the message families once every random-parameter form has been tried |

## Observed nodes, as a whole

The forms that take the whole node (a vector with a Wishart precision, a bivariate normal, a value moved to the mean); the last entry maps the rows.

| # | mapping | families | what it maps | operators | order |
| --- | --- | --- | --- | --- | --- |
| 1 | `mvn-wishart-observed` | multivariateNormal | an observed multivariate normal whose precision (or covariance) is a Wishart latent: an mvn-wishart factor on the observed vector | `mvn-wishart` |  |
| 2 | `bivariate-normal-observed` | bivariateNormal | an observed bivariate normal with a Gaussian mean and a correlation latent: an mvn-sd-corr factor | `mvn-sd-corr` |  |
| 3 | `value-moved-to-mean` | normal | a Gaussian observation of a value that is a known function of a discrete variable: the value moves to the mean, a gate over the shifted forms |  | before every other observation: the node is rewritten and mapped again from the top of this table |
| 4 | `rows` |  | every other observed node, row by row: the values are read, then the table of rows and of elements maps them |  |  |

## Observed nodes, by rows

Once the values are read (missing values and a batch are refused first): a normal-gamma pair, one plated factor, a table over discrete parents, else element by element.

| # | mapping | families | what it maps | operators | order |
| --- | --- | --- | --- | --- | --- |
| 1 | `normal-gamma-observed` | normal | an observed Gaussian of an activated normal-gamma pair: a gaussian-normal-gamma factor on the pair variable | `gaussian-normal-gamma` | before the plate: the pair variable is not a weight vector |
| 2 | `plate` | normal, bernoulli | rows that are i.i.d. draws of one regression on a weight vector (a Gaussian with one known scale, a probit or logistic Bernoulli): ONE plated factor holding the rows | `linear-regression-plate`, `logistic-plate`, `probit-plate` | before the discrete-parent table and the elements: the plated factor replaces the per-row factors (NOT_MINE: the rows are expanded) |
| 3 | `bernoulli-table` | bernoulli | a Bernoulli whose probability depends on discrete latents: a table factor over the parents and the node | `table` |  |
| 4 | `elements` |  | every other observed node, element by element, by the table of its family |  |  |

## Observed nodes, element by element

The likelihoods, by family (the order across families does not matter, the order within a family does); the last entry refuses a family with no mapping.

| # | mapping | families | what it maps | operators | order |
| --- | --- | --- | --- | --- | --- |
| 1 | `normal-switch-log-scale` | normal | a Gaussian whose log scale (and mean) is chosen by one discrete variable among random forms: one gate of gaussian-scale branches | `gate`, `gaussian-scale` | first for the normal: it takes only a scale exp(s) with s a lookup of a discrete variable; NOT_MINE sends the rest down the chain |
| 2 | `normal-switched` | normal | a Gaussian whose mean or scale is a lookup by a discrete variable: a table over it (known means) or a gate of Gaussian likelihoods | `gate`, `gaussian`, `gaussian-gamma-precision`, `table` | before the random-scale forms: a scale chosen by a discrete variable is not a random scale |
| 3 | `normal-log-scale` | normal | a Gaussian whose scale is exp of an affine form of Gaussian variables (stochastic volatility): a gaussian-log-scale factor, the mean known | `gaussian-log-scale` | before the precision idiom and the random scale: NOT_MINE (a random mean, or a log that is not a Gaussian form) falls to them |
| 4 | `normal-random-scale` | normal | a Gaussian whose scale is a positive random value that is not the precision idiom (exp of a Gaussian, a half-normal sd): a gaussian-scale factor over the log scale | `gaussian-scale` | before the precision idiom: it takes the scales that are not 1 / sqrt(tau) |
| 5 | `normal-random-precision` | normal | a Gaussian whose scale is the precision idiom 1 / sqrt(tau) of a gamma latent: a gaussian-gamma-precision factor (the mean a variable or known) | `gaussian-gamma-precision` |  |
| 6 | `normal` | normal | a Gaussian with a known scale: a gaussian factor on the mean (a regression row on a weight vector, or a variable) | `gaussian`, `linear-regression-row` | last for the normal: the known-scale form |
| 7 | `heavy-tailed` | studentT, laplace, normalOutliers | a Student-t, Laplace or outlier-mixture observation: the scale-mixture factors of the engine on the mean; a random scale, random degrees of freedom or both: a likelihood-pair or likelihood-triple factor | `bernoulli`, `gate`, `gaussian`, `gaussian-gamma-precision`, `laplace-row`, `laplace-site`, `likelihood-pair`, `likelihood-triple`, `mixture-row`, `mixture-site`, `prior`, `student-t-row`, `student-t-site` |  |
| 8 | `beta` | beta | a beta observation with random shapes: a likelihood-pair factor | `likelihood-pair` |  |
| 9 | `ordered-numeric` | orderedProbit, orderedLogistic, asymmetricLaplace | an ordered probit or logistic, or an asymmetric Laplace (quantile regression): a likelihood-site factor on the predictor, and for random cut points a likelihood-triple factor on the predictor and the two cuts of the observed category | `likelihood-pair`, `likelihood-row`, `likelihood-site`, `likelihood-triple` |  |
| 10 | `ordered-unmapped` | orderedProbit, orderedLogistic, asymmetricLaplace | the parameters of an ordered or quantile likelihood that are not a form with a message rule are refused |  |  |
| 11 | `log-normal` | logNormal | a log-normal observation: log y is Gaussian around the predictor, the exact Gaussian rule on the log of the outcome | `gaussian`, `linear-regression-row` |  |
| 12 | `bernoulli` | bernoulli | a Bernoulli observation: a probit or logistic link of an affine form, or a beta probability | `bernoulli`, `greater-than`, `logistic`, `probit` |  |
| 13 | `binomial-poisson-numeric` | binomial, poisson | a Poisson with a log link, a binomial with a logit or probit link: the numeric likelihood of the registry on the predictor | `gate`, `likelihood-row`, `likelihood-site`, `table` | before the conjugate rule: the numeric form is tried first, NOT_MINE when the rate or probability is a conjugate latent |
| 14 | `binomial-poisson-conjugate` | binomial, poisson | a Poisson with a gamma rate, a binomial with a beta probability: the conjugate observation | `binomial`, `poisson` |  |
| 15 | `multiclass-logits` | categorical | a categorical with logits: a multiclass-site factor on the predictors of the classes | `multiclass-site` | before the conjugate categorical: logits are the multiclass form |
| 16 | `categorical-conjugate` | categorical | a categorical with Dirichlet probabilities: a discrete factor on the Dirichlet variable, or a gate over rows chosen by a discrete latent | `discrete`, `gate` |  |
| 17 | `categorical-probit` | categoricalProbit | a categorical with probit scores: a multiclass-site factor | `multiclass-site` |  |
| 18 | `zero-truncated-poisson-numeric` | zeroTruncatedPoisson | a zero-truncated Poisson with a log link: the numeric likelihood of the registry | `likelihood-row`, `likelihood-site` |  |
| 19 | `zero-truncated-poisson-unmapped` | zeroTruncatedPoisson | a zero-truncated Poisson whose rate is not exposure * exp(affine) is refused |  |  |
| 20 | `hurdle-poisson` | hurdlePoisson | a hurdle Poisson: a Bernoulli for zero and a zero-truncated Poisson for the rest | `bernoulli`, `likelihood-row`, `likelihood-site` |  |
| 21 | `gate-families` | gate(...) | a likelihood chosen by an index among branches of different families (d.gate): one gate whose branch k is the factor its family maps to alone, a known branch a weight of its state | `gate`, `table` | a family of its own (gate(poisson,negativeBinomial)): each branch goes back through the whole table, so the entries above and below are the branches' mappings |
| 22 | `mixture-weight` | normalPair | a pair of Gaussian densities weighted by a beta latent (the clutter problem): a mixture-weight factor | `mixture-weight` |  |
| 23 | `gamma-rate` | exponential, gamma | an exponential or gamma observation whose rate is a gamma latent: a gamma-observed factor | `gamma-observed` |  |
| 24 | `geometric` | geometric | a geometric observation with a beta probability: a negative-binomial-beta factor (the failures before a success) | `negative-binomial-beta` |  |
| 25 | `negative-binomial-weibull-numeric` | negativeBinomial, weibull | a negative binomial with a log link (a known or random dispersion), a Weibull with a log-linear scale: the numeric likelihood of the registry | `likelihood-pair`, `likelihood-row`, `likelihood-site`, `negative-binomial-dispersion` | before the beta-conjugate failures of the negative binomial |
| 26 | `negative-binomial-failures` | negativeBinomial | a negative binomial with a beta probability: a negative-binomial-beta factor | `negative-binomial-beta` |  |
| 27 | `weibull-unmapped` | weibull | the parameters of a Weibull that are not a form with a message rule are refused |  |  |
| 28 | `family-unmapped` |  | a family with no likelihood mapping is refused, listing the likelihoods mapped |  | last: takes every element that no entry above took |
