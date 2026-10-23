import { describeDistributions, type Family } from '../../../testing/distributions.ts'
import type { Tensor } from '../../tensor/index.ts'
import * as D from '../index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const FAMILIES: Readonly<Record<string, Family>> = {
  normal: (p) => D.normal(t(p, 0), t(p, 1)),
  logNormal: (p) => D.logNormal(t(p, 0), t(p, 1)),
  halfNormal: (p) => D.halfNormal(t(p, 0)),
  truncatedNormal: (p) => D.truncatedNormal(t(p, 0), t(p, 1), t(p, 2), t(p, 3)),
  skewNormal: (p) => D.skewNormal(t(p, 0), t(p, 1), t(p, 2)),
  studentT: (p) => D.studentT(t(p, 0), t(p, 1), t(p, 2)),
  cauchy: (p) => D.cauchy(t(p, 0), t(p, 1)),
  halfCauchy: (p) => D.halfCauchy(t(p, 0)),
  laplace: (p) => D.laplace(t(p, 0), t(p, 1)),
  logistic: (p) => D.logistic(t(p, 0), t(p, 1)),
  pareto: (p) => D.pareto(t(p, 0), t(p, 1)),
  johnsonSU: (p) => D.johnsonSU(t(p, 0), t(p, 1), t(p, 2), t(p, 3)),
  gamma: (p) => D.gamma(t(p, 0), t(p, 1)),
  erlang: (p) => D.erlang(t(p, 0), t(p, 1)),
  exponential: (p) => D.exponential(t(p, 0)),
  inverseGamma: (p) => D.inverseGamma(t(p, 0), t(p, 1)),
  chi2: (p) => D.chi2(t(p, 0)),
  weibull: (p) => D.weibull(t(p, 0), t(p, 1)),
  inverseGaussian: (p) => D.inverseGaussian(t(p, 0), t(p, 1)),
  uniform: (p) => D.uniform(t(p, 0), t(p, 1)),
  beta: (p) => D.beta(t(p, 0), t(p, 1)),
  kumaraswamy: (p) => D.kumaraswamy(t(p, 0), t(p, 1)),
  triangular: (p) => D.triangular(t(p, 0), t(p, 1), t(p, 2)),
  gumbel: (p) => D.gumbel(t(p, 0), t(p, 1)),
  generalisedExtremeValue: (p) => D.generalisedExtremeValue(t(p, 0), t(p, 1), t(p, 2)),
  generalisedPareto: (p) => D.generalisedPareto(t(p, 0), t(p, 1), t(p, 2)),
  frechet: (p) => D.frechet(t(p, 0), t(p, 1), t(p, 2)),
  fDistribution: (p) => D.fDistribution(t(p, 0), t(p, 1)),
  noncentralChi2: (p) => D.noncentralChi2(t(p, 0), t(p, 1)),
  noncentralT: (p) => D.noncentralT(t(p, 0), t(p, 1)),
  studentisedRange: (p) => D.studentisedRange(t(p, 0), t(p, 1)),
}

for (const file of ['normal', 'heavy_tailed', 'gamma', 'bounded', 'extreme', 'sampling_distributions'])
  describeDistributions(`dists/continuous/${file}`, FAMILIES)
