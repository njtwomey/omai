# structure/benchmarks

The structure report of every benchmark model against the findings written down for it.

Run on 2026-11-10 at commit aceb09a9; seed 0, budget scale 1 (models 27), 0.0 s.

**Passed**: 27 models: every finding as expected

What `structureReport` finds in each benchmark model. The columns are the findings the campaign checks (`STRUCTURE_EXPECTED` in `probprog/models/structure-campaigns.ts`): the conjugate latents (a hierarchy or an affine map is marked), what the `collapse` handler can integrate out, the Gaussian blocks (`sites | given`), the discrete part, the exchangeable plate axes and the number of independent sub-problems. Below, the full report of each model as `explain()` prints it. The report is advice: nothing is applied.

| model | target | conjugate latents | collapsible by the handler | Gaussian blocks | discrete part | minibatch axes | components |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beta-bernoulli | posterior | p | p | - | none | flip | 1 |
| normal-normal | posterior | mu | mu | mu \| nothing | none | obs | 1 |
| gamma-poisson | posterior | rate | rate | - | none | obs | 1 |
| logistic-regression | posterior | - | - | - | none | row | 1 |
| probit-regression | posterior | - | - | - | none | row | 1 |
| poisson-regression | posterior | - | - | - | none | row | 1 |
| negative-binomial-regression | posterior | - | - | - | none | row | 1 |
| eight-schools-centred | posterior | mu (hierarchical, direct); theta | theta | mu,theta \| tau | none | school | 1 |
| eight-schools-non-centred | posterior | mu (affine); z (affine) | - | mu,z \| tau | none | school | 1 |
| gaussian-mixture-ordered | posterior | mu (index) not exact | - | - | z: width 0, 2 states | row | 1 |
| gaussian-mixture | posterior | mu (index) | - | mu \| z | z: width 0, 2 states | row | 1 |
| funnel-centred | density | - | - | - | none | - | 1 |
| funnel-non-centred | density | - | - | - | none | - | 2 |
| banana | density | - | - | - | none | - | 1 |
| bimodal | density | - | - | - | z: width 0, 2 states | - | 1 |
| radon | posterior | mu (hierarchical, direct); beta (affine); alpha (affine) | - | mu,beta,alpha \| tau,sigma | none | county; house | 1 |
| trueskill-three-players | posterior | - | - | - | none | game | 1 |
| ar1-state-space | posterior | - | - | x \| nothing | none | - | 1 |
| stochastic-volatility | posterior | - | - | - | none | - | 1 |
| gaussian-hmm | posterior | - | - | - | none | - | 1 |
| gaussian-process-regression | posterior | f | f | f \| ell | none | - | 1 |
| gaussian-process-non-centred | posterior | - | - | - | none | - | 1 |
| lda-lite | posterior | theta (hierarchical, direct); phi (index) | - | - | z: width 0 | doc; token | 1 |
| bayesian-neural-network | posterior | w2 (affine); b2 (affine) | - | w2,b2 \| w1,b1 | none | point | 1 |
| item-response-scoring | posterior | - | - | - | none | person; item | 1 |
| stick-breaking-mixture | posterior | - | - | - | z: width 0 | row | 1 |
| clutter-problem | posterior | - | - | - | z: width 0, 2 states | row | 1 |

### Beta-Bernoulli coin (beta-bernoulli)

```text
Structure of beta-bernoulli: 1 continuous latent, 1 observed; plates flip.
Independent components (1): {p} with `y`.
Conjugacy:
  - p [pair]: p is conjugate (beta-bernoulli probs through direct): its posterior is closed form, with nothing else to condition on, summing over flip. Posterior: Beta(a + sum y, b + n - sum y). Collapse handler: yes.
Minibatch axes: flip (20).
Markov blankets: p: {y}.
Suggestions (advice, nothing is applied):
  - every latent (`p`) is conjugate to its observations with known hyperparameters: the posterior is closed form.
      why: each latent has a conjugate pair with known fixed parameters, known hyperparameters and nothing else depending on it
      use method `conjugate`: closed-form posterior and evidence in one step
  - site `p` is conjugate to `y` (beta-bernoulli probs): collapse can integrate it out and a sampler then moves in no continuous latents.
      why: p is conjugate (beta-bernoulli probs through direct): its posterior is closed form, with nothing else to condition on, summing over flip
      use handler `collapse` {"sites":["p"]}: collapsePlan(bound) lists the sites, applyCollapse(bound, plan) gives the collapsed program; recover draws p back from its closed-form posterior
```

### Normal-normal mean (normal-normal)

```text
Structure of normal-normal: 1 continuous latent, 1 observed; plates obs.
Independent components (1): {mu} with `y`.
Conjugacy:
  - mu [pair]: mu is conjugate (normal-normal loc through direct): its posterior is closed form, with nothing else to condition on, summing over obs. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: yes.
Linear-Gaussian blocks:
  - {mu} (the whole model), explaining `y`: every node that touches mu is Gaussian with a mean affine in them: jointly Gaussian, exactly marginalisable.
Minibatch axes: obs (12).
Markov blankets: mu: {y}.
Suggestions (advice, nothing is applied):
  - every latent (`mu`) is conjugate to its observations with known hyperparameters: the posterior is closed form.
      why: each latent has a conjugate pair with known fixed parameters, known hyperparameters and nothing else depending on it
      use method `conjugate`: closed-form posterior and evidence in one step
  - site `mu` is conjugate to `y` (normal-normal loc): collapse can integrate it out and a sampler then moves in no continuous latents.
      why: mu is conjugate (normal-normal loc through direct): its posterior is closed form, with nothing else to condition on, summing over obs
      use handler `collapse` {"sites":["mu"]}: collapsePlan(bound) lists the sites, applyCollapse(bound, plan) gives the collapsed program; recover draws mu back from its closed-form posterior
  - the whole model is linear-Gaussian (`mu`, 1 variable): the posterior is a Gaussian, exact.
      why: every node that touches mu is Gaussian with a mean affine in them: jointly Gaussian, exactly marginalisable
      use method `gaussian`: exact Gaussian posterior (joint solve, Kalman or belief propagation)
```

### Gamma-Poisson rate (gamma-poisson)

```text
Structure of gamma-poisson: 1 continuous latent, 1 observed; plates obs.
Independent components (1): {rate} with `k`.
Conjugacy:
  - rate [pair]: rate is conjugate (gamma-poisson rate through direct): its posterior is closed form, with nothing else to condition on, summing over obs. Posterior: Gamma(shape + sum y, rate + n). Collapse handler: yes.
Minibatch axes: obs (15).
Markov blankets: rate: {k}.
Suggestions (advice, nothing is applied):
  - every latent (`rate`) is conjugate to its observations with known hyperparameters: the posterior is closed form.
      why: each latent has a conjugate pair with known fixed parameters, known hyperparameters and nothing else depending on it
      use method `conjugate`: closed-form posterior and evidence in one step
  - site `rate` is conjugate to `k` (gamma-poisson rate): collapse can integrate it out and a sampler then moves in no continuous latents.
      why: rate is conjugate (gamma-poisson rate through direct): its posterior is closed form, with nothing else to condition on, summing over obs
      use handler `collapse` {"sites":["rate"]}: collapsePlan(bound) lists the sites, applyCollapse(bound, plan) gives the collapsed program; recover draws rate back from its closed-form posterior
```

### Bayesian logistic regression (logistic-regression)

```text
Structure of logistic-regression: 1 continuous latent, 1 observed; plates row, feature.
Independent components (1): {w} with `y`.
Conjugacy: none found.
Minibatch axes: row (50).
Markov blankets: w: {y}.
Suggestions: none; nothing in the structure is specific enough to exploit.
```

### Bayesian probit regression (probit-regression)

```text
Structure of probit-regression: 1 continuous latent, 1 observed; plates row, feature.
Independent components (1): {w} with `y`.
Conjugacy: none found.
Minibatch axes: row (50).
Markov blankets: w: {y}.
Suggestions: none; nothing in the structure is specific enough to exploit.
```

### Poisson regression (poisson-regression)

```text
Structure of poisson-regression: 2 continuous latents, 1 observed; plates row.
Independent components (1): {a, b} with `k`.
Conjugacy: none found.
Minibatch axes: row (80).
Markov blankets: a: {k, b}; b: {k, a}.
Suggestions (advice, nothing is applied):
  - dimension `row` (80 elements) is exchangeable (nothing reduces over it or looks it up by index): a minibatch of it gives an unbiased density estimate.
      why: `k` are over row and no expression reduces over it
      use method `svi` {}: bind the model with `{ batch: { dim: 'row', size: 20 } }`: log densities over row are rescaled by size over batch size
```

### Negative binomial regression (negative-binomial-regression)

```text
Structure of negative-binomial-regression: 3 continuous latents, 1 observed; plates row.
Independent components (1): {a, alpha, b} with `k`.
Conjugacy: none found.
Minibatch axes: row (100).
Markov blankets: a: {k, alpha, b}; b: {k, alpha, a}; alpha: {k, a, b}.
Suggestions (advice, nothing is applied):
  - dimension `row` (100 elements) is exchangeable (nothing reduces over it or looks it up by index): a minibatch of it gives an unbiased density estimate.
      why: `k` are over row and no expression reduces over it
      use method `svi` {}: bind the model with `{ batch: { dim: 'row', size: 25 } }`: log densities over row are rescaled by size over batch size
```

### Eight schools, centred (eight-schools-centred)

```text
Structure of eight-schools-centred: 3 continuous latents, 1 observed; plates school.
Independent components (1): {mu, tau, theta} with `y`.
Conjugacy:
  - mu [hierarchical]: mu is conjugate (normal-normal loc through direct): its posterior is closed form given tau, summing over school. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (some dependents are latent: collapse integrates a site into observed children, so collapse those first).
  - theta [pair]: theta is conjugate (normal-normal loc through direct): its posterior is closed form given mu, tau. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: yes.
Linear-Gaussian blocks:
  - {mu, theta} given {tau}, explaining `y`: given tau, every node that touches mu, theta is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable.
Minibatch axes: school (8).
Markov blankets: mu: {theta, tau}; tau: {theta, mu}; theta: {mu, tau, y} (per element of school).
Suggestions (advice, nothing is applied):
  - site `mu` has a closed-form conditional (normal-normal loc given `tau`), with latent children: a Gibbs step can draw it exactly; the collapse handler does not take it (some dependents are latent: collapse integrates a site into observed children, so collapse those first).
      why: mu is conjugate (normal-normal loc through direct): its posterior is closed form given tau, summing over school
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - site `theta` is conjugate to `y` (normal-normal loc) given `mu`, `tau`: collapse can integrate it out and a sampler then moves in `mu`, `tau` alone.
      why: theta is conjugate (normal-normal loc through direct): its posterior is closed form given mu, tau
      use handler `collapse` {"sites":["theta"]}: collapsePlan(bound) lists the sites, applyCollapse(bound, plan) gives the collapsed program; recover draws theta back from its closed-form posterior
  - sites {mu, theta} form a Gaussian block conditional on {tau} (9 variables): given those, the block's posterior is Gaussian and its marginal likelihood closed form.
      why: given tau, every node that touches mu, theta is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable
      use method `nuts`: run NUTS on {tau} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names
  - site `theta` is centred (its scale is the latent `tau`): the funnel geometry slows NUTS when the data for each element are weak; sampling the standardised value avoids it.
      why: theta is normal over school with scale depending on tau
      use handler `reparam` {"theta":"locScale(0)"}: reparam(program, { theta: locScale(0) }) makes the program sample a standard value and transform it
  - nodes `tau` are opaque (a custom function or a family with no exponential-family form): message passing and the closed-form methods do not apply, the sampling methods do.
      why: message passing needs every factor to be a registered operator and an exponential-family distribution
      use method `nuts`: sampling methods run on any differentiable density
```

### Eight schools, non-centred (eight-schools-non-centred)

```text
Structure of eight-schools-non-centred: 3 continuous latents, 1 observed; plates school.
Independent components (1): {mu, tau, z} with `y`.
Conjugacy:
  - mu [pair]: mu is conjugate (normal-normal loc through affine): its posterior is closed form given tau, z, summing over school. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not affine).
  - z [pair]: z is conjugate (normal-normal loc through affine): its posterior is closed form given mu, tau. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not affine).
Linear-Gaussian blocks:
  - {mu, z} given {tau}, explaining `y`: given tau, every node that touches mu, z is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable.
Minibatch axes: school (8).
Markov blankets: mu: {y, tau, z}; tau: {y, mu, z}; z: {y, mu, tau} (per element of school).
Suggestions (advice, nothing is applied):
  - site `mu` has a closed-form conditional (normal-normal loc given `tau`, `z`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not affine).
      why: mu is conjugate (normal-normal loc through affine): its posterior is closed form given tau, z, summing over school
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - site `z` has a closed-form conditional (normal-normal loc given `mu`, `tau`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not affine).
      why: z is conjugate (normal-normal loc through affine): its posterior is closed form given mu, tau
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - sites {mu, z} form a Gaussian block conditional on {tau} (9 variables): given those, the block's posterior is Gaussian and its marginal likelihood closed form.
      why: given tau, every node that touches mu, z is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable
      use method `nuts`: run NUTS on {tau} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names
  - nodes `tau` are opaque (a custom function or a family with no exponential-family form): message passing and the closed-form methods do not apply, the sampling methods do.
      why: message passing needs every factor to be a registered operator and an exponential-family distribution
      use method `nuts`: sampling methods run on any differentiable density
```

### Gaussian mixture, ordered means (gaussian-mixture-ordered)

```text
Structure of gaussian-mixture-ordered: 1 continuous latent, 1 finite discrete, 1 observed; plates row, component.
Independent components (1): {mu, z} with `y`.
Conjugacy:
  - mu [pair, not exact]: mu would be conjugate (normal-normal loc through index) but the constraint 'ordered' on 'mu' truncates the prior, so the closed form is not its posterior. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the site is not conjugate with everything that depends on it).
Discrete part: `z`; induced width 0; elimination order {z}.
Minibatch axes: row (40).
Markov blankets: mu: {y, z}; z: {y, mu} (per element of row).
Suggestions (advice, nothing is applied):
  - site `z` (finite discrete over row) can be summed out: NUTS and VI can enumerate it (2 joint states per element, 40 independent elements; width 0).
      why: the discrete part has 1 site in this component, elimination order {z} gives induced width 0
      use method `nuts` {"inferDiscrete":true}: NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw
```

### Gaussian mixture, label switching (gaussian-mixture)

```text
Structure of gaussian-mixture: 1 continuous latent, 1 finite discrete, 1 observed; plates row, component.
Independent components (1): {mu, z} with `y`.
Conjugacy:
  - mu [pair]: mu is conjugate (normal-normal loc through index): its posterior is closed form given z, summing over row. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not index).
Linear-Gaussian blocks:
  - {mu} given {z}, explaining `y`: given z, every node that touches mu is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable.
Discrete part: `z`; induced width 0; elimination order {z}.
Minibatch axes: row (40).
Markov blankets: mu: {y, z}; z: {y, mu} (per element of row).
Suggestions (advice, nothing is applied):
  - site `z` (finite discrete over row) can be summed out: NUTS and VI can enumerate it (2 joint states per element, 40 independent elements; width 0).
      why: the discrete part has 1 site in this component, elimination order {z} gives induced width 0
      use method `nuts` {"inferDiscrete":true}: NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw
  - site `mu` has a closed-form conditional (normal-normal loc given `z`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not index).
      why: mu is conjugate (normal-normal loc through index): its posterior is closed form given z, summing over row
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - sites {mu} form a Gaussian block conditional on {z} (2 variables): given those, the block's posterior is Gaussian and its marginal likelihood closed form.
      why: given z, every node that touches mu is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable
      use method `nuts`: run NUTS on {z} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names
```

### Neal's funnel, centred (funnel-centred)

```text
Structure of funnel-centred: 2 continuous latents, 0 observed; plates coord.
Independent components (1): {v, x} with no observations.
Conjugacy: none found.
Markov blankets: v: {x}; x: {v} (per element of coord).
Suggestions (advice, nothing is applied):
  - site `x` is centred (its scale is the latent `v`): the funnel geometry slows NUTS when the data for each element are weak; sampling the standardised value avoids it.
      why: x is normal over coord with scale depending on v
      use handler `reparam` {"x":"locScale(0)"}: reparam(program, { x: locScale(0) }) makes the program sample a standard value and transform it
```

### Neal's funnel, non-centred (funnel-non-centred)

```text
Structure of funnel-non-centred: 2 continuous latents, 0 observed; plates coord.
Independent components (2): {v} with no observations; {z} with no observations.
Conjugacy: none found.
Suggestions (advice, nothing is applied):
  - the model splits into 2 independent sub-problems ({v}, {z}): each can be inferred on its own.
      why: no density term connects them: the posterior is a product, and the costs add instead of multiplying
      use method `nuts`: one model per component (conditionSpec or a model with only that component), one run each
```

### Haario's banana (banana)

```text
Structure of banana: 2 continuous latents, 0 observed; plates none.
Independent components (1): {x0, x1} with no observations.
Conjugacy: none found.
Markov blankets: x0: {x1}; x1: {x0}.
Suggestions: none; nothing in the structure is specific enough to exploit.
```

### Two Gaussian modes (bimodal)

```text
Structure of bimodal: 1 continuous latent, 1 finite discrete, 0 observed; plates coord.
Independent components (1): {x, z} with no observations.
Conjugacy: none found.
Discrete part: `z`; induced width 0; elimination order {z}.
Markov blankets: z: {x}; x: {z} (per element of coord).
Suggestions (advice, nothing is applied):
  - site `z` (finite discrete) can be summed out: NUTS and VI can enumerate it (2 joint states per element, 1 independent element; width 0).
      why: the discrete part has 1 site in this component, elimination order {z} gives induced width 0
      use method `nuts` {"inferDiscrete":true}: NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw
```

### Hierarchical radon regression (radon)

```text
Structure of radon: 5 continuous latents, 1 observed; plates county, house.
Independent components (1): {alpha, beta, mu, sigma, tau} with `y`.
Conjugacy:
  - mu [hierarchical]: mu is conjugate (normal-normal loc through direct): its posterior is closed form given tau, summing over county. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (some dependents are latent: collapse integrates a site into observed children, so collapse those first).
  - beta [pair]: beta is conjugate (normal-normal loc through affine): its posterior is closed form given alpha, sigma, summing over county, house. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not affine).
  - alpha [pair]: alpha is conjugate (normal-normal loc through affine): its posterior is closed form given beta, sigma, mu, tau, summing over house. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not affine).
Linear-Gaussian blocks:
  - {mu, beta, alpha} given {tau, sigma}, explaining `y`: given tau, sigma, every node that touches mu, beta, alpha is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable.
Minibatch axes: county (6), house (8).
Markov blankets: mu: {alpha, tau}; beta: {y, alpha, sigma}; tau: {alpha, mu}; sigma: {y, alpha, beta}; alpha: {mu, tau, y, beta, sigma} (per element of county).
Suggestions (advice, nothing is applied):
  - site `mu` has a closed-form conditional (normal-normal loc given `tau`), with latent children: a Gibbs step can draw it exactly; the collapse handler does not take it (some dependents are latent: collapse integrates a site into observed children, so collapse those first).
      why: mu is conjugate (normal-normal loc through direct): its posterior is closed form given tau, summing over county
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - site `beta` has a closed-form conditional (normal-normal loc given `alpha`, `sigma`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not affine).
      why: beta is conjugate (normal-normal loc through affine): its posterior is closed form given alpha, sigma, summing over county, house
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - site `alpha` has a closed-form conditional (normal-normal loc given `beta`, `sigma`, `mu`, `tau`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not affine).
      why: alpha is conjugate (normal-normal loc through affine): its posterior is closed form given beta, sigma, mu, tau, summing over house
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - sites {mu, beta, alpha} form a Gaussian block conditional on {tau, sigma} (8 variables): given those, the block's posterior is Gaussian and its marginal likelihood closed form.
      why: given tau, sigma, every node that touches mu, beta, alpha is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable
      use method `nuts`: run NUTS on {tau, sigma} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names
  - site `alpha` is centred (its scale is the latent `tau`): the funnel geometry slows NUTS when the data for each element are weak; sampling the standardised value avoids it.
      why: alpha is normal over county with scale depending on tau
      use handler `reparam` {"alpha":"locScale(0)"}: reparam(program, { alpha: locScale(0) }) makes the program sample a standard value and transform it
```

### TrueSkill, three players (trueskill-three-players)

```text
Structure of trueskill-three-players: 1 continuous latent, 1 observed; plates player, game.
Independent components (1): {skill} with `win`.
Conjugacy: none found.
Minibatch axes: game (24).
Markov blankets: skill: {win}.
Suggestions: none; nothing in the structure is specific enough to exploit.
```

### AR(1) state space (ar1-state-space)

```text
Structure of ar1-state-space: 1 continuous latent, 1 observed; plates time.
Independent components (1): {x} with `y`.
Conjugacy: none found.
Linear-Gaussian blocks:
  - {x} (the whole model), explaining `y`: a Gaussian chain: 1 Gaussian emission(s) with means affine in the state, linear steps: jointly Gaussian over the series, exactly marginalisable by a Kalman filter.
Markov blankets: x: {y} (per element of time).
Suggestions (advice, nothing is applied):
  - the whole model is linear-Gaussian (`x`, 12 variables): the posterior is a Gaussian, exact.
      why: a Gaussian chain: 1 Gaussian emission(s) with means affine in the state, linear steps: jointly Gaussian over the series, exactly marginalisable by a Kalman filter
      use method `gaussian`: exact Gaussian posterior (joint solve, Kalman or belief propagation)
```

### Stochastic volatility (stochastic-volatility)

```text
Structure of stochastic-volatility: 1 continuous latent, 1 observed; plates time.
Independent components (1): {h} with `y`.
Conjugacy: none found.
Markov blankets: h: {y} (per element of time).
Suggestions (advice, nothing is applied):
  - `h` is a state-space model of 30 steps that is not a Gaussian chain (the scale of emission 'y' depends on a state or a latent): the bootstrap particle filter follows it step by step, and gives an unbiased evidence estimate.
      why: the states are the only latents, so the filter needs nothing else fixed
      use method `particle-filter` {"numParticles":1000}: one time step per session step, filtering marginals at any step
```

### Gaussian hidden Markov model (gaussian-hmm)

```text
Structure of gaussian-hmm: 1 finite discrete, 1 observed; plates time, from, state.
Independent components (1): {z} with `y`.
Conjugacy: none found.
Markov blankets: z: {y} (per element of time).
Suggestions (advice, nothing is applied):
  - `z` is a hidden Markov chain of 16 steps (categorical states, nothing else random): forward-backward gives the exact marginals and the evidence.
      why: one scan with finite discrete states is the only latent: the chain engine reads its initial, transition and emission potentials
      use method `chain`: forward-backward, FFBS draws and the Viterbi path in one step
```

### Gaussian process regression (gaussian-process-regression)

```text
Structure of gaussian-process-regression: 2 continuous latents, 1 observed; plates point.
Independent components (1): {ell, f} with `y`.
Conjugacy:
  - f [pair]: f is conjugate (multivariateNormal-multivariateNormal loc through direct): its posterior is closed form given ell. Posterior: Normal with precision L_0 + n S^-1 and mean solving that precision against L_0 m_0 + S^-1 sum y. Collapse handler: yes.
Linear-Gaussian blocks:
  - {f} given {ell}, explaining `y`: given ell, every node that touches f is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable.
Markov blankets: ell: {f}; f: {ell, y}.
Suggestions (advice, nothing is applied):
  - site `f` is conjugate to `y` (multivariateNormal-multivariateNormal loc) given `ell`: collapse can integrate it out and a sampler then moves in `ell` alone.
      why: f is conjugate (multivariateNormal-multivariateNormal loc through direct): its posterior is closed form given ell
      use handler `collapse` {"sites":["f"]}: collapsePlan(bound) lists the sites, applyCollapse(bound, plan) gives the collapsed program; recover draws f back from its closed-form posterior
  - sites {f} form a Gaussian block conditional on {ell}: given those, the block's posterior is Gaussian and its marginal likelihood closed form.
      why: given ell, every node that touches f is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable
      use method `nuts`: run NUTS on {ell} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names
  - nodes `f` are opaque (a custom function or a family with no exponential-family form): message passing and the closed-form methods do not apply, the sampling methods do.
      why: message passing needs every factor to be a registered operator and an exponential-family distribution
      use method `nuts`: sampling methods run on any differentiable density
```

### Gaussian process regression, non-centred (gaussian-process-non-centred)

```text
Structure of gaussian-process-non-centred: 2 continuous latents, 1 observed; plates point.
Independent components (1): {ell, z} with `y`.
Conjugacy: none found.
Markov blankets: ell: {y, z}; z: {y, ell} (per element of point).
Suggestions (advice, nothing is applied):
  - nodes `f` are opaque (a custom function or a family with no exponential-family form): message passing and the closed-form methods do not apply, the sampling methods do.
      why: message passing needs every factor to be a registered operator and an exponential-family distribution
      use method `nuts`: sampling methods run on any differentiable density
```

### LDA on a tiny corpus (lda-lite)

```text
Structure of lda-lite: 2 continuous latents, 1 finite discrete, 1 observed; plates doc, token, topic, word.
Independent components (1): {phi, theta, z} with `w`.
Conjugacy:
  - theta [hierarchical]: theta is conjugate (dirichlet-categorical probs through direct): its posterior is closed form, with nothing else to condition on, summing over token. Posterior: Dirichlet(alpha + counts). Collapse handler: no (some dependents are latent: collapse integrates a site into observed children, so collapse those first).
  - phi [pair]: phi is conjugate (dirichlet-categorical probs through index): its posterior is closed form given z, summing over doc, token. Posterior: Dirichlet(alpha + counts). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not index).
Discrete part: `z`; induced width 0; elimination order {z}.
Minibatch axes: doc (3), token (5).
Markov blankets: theta: {z} (per element of doc); phi: {w, z}; z: {theta, w, phi} (per element of doc, token).
Suggestions (advice, nothing is applied):
  - site `z` (finite discrete over doc x token) can be summed out: NUTS and VI can enumerate it (a number of states that is not known from the spec, 15 independent elements; width 0).
      why: the discrete part has 1 site in this component, elimination order {z} gives induced width 0
      use method `nuts` {"inferDiscrete":true}: NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw
  - site `theta` has a closed-form conditional (dirichlet-categorical probs), with latent children: a Gibbs step can draw it exactly; the collapse handler does not take it (some dependents are latent: collapse integrates a site into observed children, so collapse those first).
      why: theta is conjugate (dirichlet-categorical probs through direct): its posterior is closed form, with nothing else to condition on, summing over token
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - site `phi` has a closed-form conditional (dirichlet-categorical probs given `z`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not index).
      why: phi is conjugate (dirichlet-categorical probs through index): its posterior is closed form given z, summing over doc, token
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
```

### Bayesian neural network (bayesian-neural-network)

```text
Structure of bayesian-neural-network: 4 continuous latents, 1 observed; plates point, hidden, test.
Independent components (1): {b1, b2, w1, w2} with `y`.
Conjugacy:
  - w2 [pair]: w2 is conjugate (normal-normal loc through affine): its posterior is closed form given w1, b1, b2, summing over point. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not affine).
  - b2 [pair]: b2 is conjugate (normal-normal loc through affine): its posterior is closed form given w1, b1, w2, summing over point. Posterior: Normal(m_n, s_n^2) with 1/s_n^2 = 1/s_0^2 + n/sigma^2 and m_n = s_n^2 (m_0/s_0^2 + sum y/sigma^2). Collapse handler: no (the handler takes the parameter itself (or the variance behind a scale), not affine).
Linear-Gaussian blocks:
  - {w2, b2} given {w1, b1}, explaining `y`: given w1, b1, every node that touches w2, b2 is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable.
Minibatch axes: point (16).
Markov blankets: w1: {y, b1, w2, b2}; b1: {y, w1, w2, b2}; w2: {y, w1, b1, b2}; b2: {y, w1, b1, w2}.
Suggestions (advice, nothing is applied):
  - site `w2` has a closed-form conditional (normal-normal loc given `w1`, `b1`, `b2`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not affine).
      why: w2 is conjugate (normal-normal loc through affine): its posterior is closed form given w1, b1, b2, summing over point
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - site `b2` has a closed-form conditional (normal-normal loc given `w1`, `b1`, `w2`): a Gibbs step can draw it exactly; the collapse handler does not take it (the handler takes the parameter itself (or the variance behind a scale), not affine).
      why: b2 is conjugate (normal-normal loc through affine): its posterior is closed form given w1, b1, w2, summing over point
      use method `gibbs`: conjugate conditionals are drawn exactly inside a Gibbs sweep
  - sites {w2, b2} form a Gaussian block conditional on {w1, b1} (4 variables): given those, the block's posterior is Gaussian and its marginal likelihood closed form.
      why: given w1, b1, every node that touches w2, b2 is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable
      use method `nuts`: run NUTS on {w1, b1} alone with the block marginalised: no handler integrates a multi-site Gaussian block today (collapse takes one conjugate site at a time), so this is a gap the report names
```

### Item response theory (2PL scoring) (item-response-scoring)

```text
Structure of item-response-scoring: 1 continuous latent, 1 observed; plates person, item.
Independent components (1): {theta} with `y`.
Conjugacy: none found.
Minibatch axes: person (12), item (8).
Markov blankets: theta: {y} (per element of person).
Suggestions: none; nothing in the structure is specific enough to exploit.
```

### Stick-breaking mixture (truncated Dirichlet process) (stick-breaking-mixture)

```text
Structure of stick-breaking-mixture: 1 continuous latent, 1 finite discrete, 1 observed; plates row, stick, component.
Independent components (1): {v, z} with `y`.
Conjugacy: none found.
Discrete part: `z`; induced width 0; elimination order {z}.
Minibatch axes: row (30).
Markov blankets: v: {z}; z: {v, y} (per element of row).
Suggestions (advice, nothing is applied):
  - site `z` (finite discrete over row) can be summed out: NUTS and VI can enumerate it (a number of states that is not known from the spec, 30 independent elements; width 0).
      why: the discrete part has 1 site in this component, elimination order {z} gives induced width 0
      use method `nuts` {"inferDiscrete":true}: NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw
  - nodes `pi` are opaque (a custom function or a family with no exponential-family form): message passing and the closed-form methods do not apply, the sampling methods do.
      why: message passing needs every factor to be a registered operator and an exponential-family distribution
      use method `nuts`: sampling methods run on any differentiable density
```

### The clutter problem (clutter-problem)

```text
Structure of clutter-problem: 1 continuous latent, 1 finite discrete, 1 observed; plates row.
Independent components (1): {theta, z} with `y`.
Conjugacy: none found.
Discrete part: `z`; induced width 0; elimination order {z}.
Minibatch axes: row (12).
Markov blankets: theta: {y, z}; z: {y, theta} (per element of row).
Suggestions (advice, nothing is applied):
  - site `z` (finite discrete over row) can be summed out: NUTS and VI can enumerate it (2 joint states per element, 12 independent elements; width 0).
      why: the discrete part has 1 site in this component, elimination order {z} gives induced width 0
      use method `nuts` {"inferDiscrete":true}: NUTS sums the discrete sites out of the density; `inferDiscrete: true` draws them back for each draw
```

