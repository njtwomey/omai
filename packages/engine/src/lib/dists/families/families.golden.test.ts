import { type Custom, describeDistributions } from '../../../testing/distributions.ts'
import type { Tensor } from '../../tensor/index.ts'
import * as F from './index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const LINKS: Readonly<Record<string, () => F.Link>> = {
  identity: F.identity,
  log: F.log,
  logit: F.logit,
  probit: F.probit,
  cauchit: F.cauchit,
  cloglog: F.cloglog,
  loglog: F.loglog,
  inverse: F.inverse,
  inverseSquared: F.inverseSquared,
  sqrt: F.sqrt,
  power: () => F.power(1.7),
  negativeBinomial: () => F.negativeBinomialLink(0.6),
}
const FAMILIES: Readonly<Record<string, () => F.DispersionFamily>> = {
  gaussian: () => F.gaussian(),
  binomial: () => F.binomial(),
  poisson: () => F.poisson(),
  gamma: () => F.gamma(),
  inverseGaussian: () => F.inverseGaussian(),
  negativeBinomial: () => F.negativeBinomial(0.7),
  tweedie: () => F.tweedie(1.5),
}

const linkCases: Record<string, Custom> = {}
for (const [name, make] of Object.entries(LINKS))
  for (const m of ['link', 'inverse', 'derivative', 'inverseDerivative', 'secondDerivative'] as const)
    linkCases[`${name}.${m}`] = (xs) => make()[m](t(xs, 0))

const familyCases: Record<string, Custom> = {
  'tweedieDistribution.logProb': (xs, c) =>
    F.tweedieDistribution(t(xs, 0), t(xs, 1), c.params['power'] as number).logProb(t(xs, 2)),
  'tweedieDistribution.cdf': (xs, c) =>
    F.tweedieDistribution(t(xs, 0), t(xs, 1), c.params['power'] as number).cdf(t(xs, 2)),
}
for (const [name, make] of Object.entries(FAMILIES)) {
  familyCases[`${name}.variance`] = (xs) => make().variance(t(xs, 0))
  familyCases[`${name}.varianceDerivative`] = (xs) => make().varianceDerivative(t(xs, 0))
  familyCases[`${name}.deviance`] = (xs) => make().deviance(t(xs, 0), t(xs, 1), t(xs, 2))
  familyCases[`${name}.devianceResiduals`] = (xs) => make().devianceResiduals(t(xs, 0), t(xs, 1), t(xs, 2))
  familyCases[`${name}.logLikelihoodObs`] = (xs) => make().logLikelihoodObs(t(xs, 0), t(xs, 1), t(xs, 2), t(xs, 3))
  familyCases[`${name}.startingMu`] = (xs) => make().startingMu(t(xs, 0))
}

const GAMLSS: Readonly<Record<string, () => F.DistributionalFamily>> = {
  normal: F.gamlssNormal,
  studentT: F.gamlssStudentT,
  skewNormal: F.gamlssSkewNormal,
  gamma: F.gamlssGamma,
  logNormal: F.gamlssLogNormal,
  weibull: F.gamlssWeibull,
  poisson: F.gamlssPoisson,
  negativeBinomial: F.gamlssNegativeBinomial,
  beta: F.gamlssBeta,
  zeroInflatedPoisson: F.gamlssZeroInflatedPoisson,
}
const distributionalCases: Record<string, Custom> = {}
for (const [name, make] of Object.entries(GAMLSS))
  distributionalCases[`gamlss.${name}`] = (xs) => make().logLikelihood(t(xs, 0), xs.slice(1))
for (const link of ['logit', 'probit', 'cloglog', 'loglog', 'cauchit'] as const)
  distributionalCases[`ordinal.${link}`] = (xs) => F.ordinalFamily(link).logLikelihood(t(xs, 0), t(xs, 1), t(xs, 2))

describeDistributions('dists/families/links', {}, linkCases)
describeDistributions('dists/families/distributional', {}, distributionalCases)
describeDistributions('dists/families/dispersion', {}, familyCases)
