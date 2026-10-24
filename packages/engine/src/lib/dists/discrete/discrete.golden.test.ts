import { describeDistributions, type Family } from '../../../testing/distributions.ts'
import type { Tensor } from '../../tensor/index.ts'
import * as D from '../index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const FAMILIES: Readonly<Record<string, Family>> = {
  bernoulli: (p) => D.bernoulli({ probs: t(p, 0) }),
  binomial: (p) => D.binomial(t(p, 0), t(p, 1)),
  betaBinomial: (p) => D.betaBinomial(t(p, 0), t(p, 1), t(p, 2)),
  categorical: (p) => D.categorical({ probs: t(p, 0) }),
  oneHotCategorical: (p) => D.oneHotCategorical({ probs: t(p, 0) }),
  discreteUniform: (p) => D.discreteUniform(t(p, 0), t(p, 1)),
  orderedLogistic: (p) => D.orderedLogistic(t(p, 0), t(p, 1)),
  orderedProbit: (p) => D.orderedProbit(t(p, 0), t(p, 1)),
  poisson: (p) => D.poisson(t(p, 0)),
  geometric: (p) => D.geometric(t(p, 0)),
  negativeBinomial: (p) => D.negativeBinomial(t(p, 0), t(p, 1)),
  hypergeometric: (p) => D.hypergeometric(t(p, 0), t(p, 1), t(p, 2)),
  zeroInflatedPoisson: (p) => D.zeroInflatedPoisson(t(p, 0), t(p, 1)),
  zeroInflatedNegativeBinomial: (p) => D.zeroInflatedNegativeBinomial(t(p, 0), t(p, 1), t(p, 2)),
}

for (const file of ['binary', 'categorical', 'counts']) describeDistributions(`dists/discrete/${file}`, FAMILIES)
