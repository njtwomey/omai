import { type Custom, describeDistributions } from '../../../testing/distributions.ts'
import { squeeze, type Tensor, take, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor
const dim = (c: { readonly params: Readonly<Record<string, unknown>> }): number => c.params['dim'] as number
const archimedean: Readonly<Record<string, (theta: Tensor, d: number) => D.Copula>> = {
  clayton: D.claytonCopula,
  gumbel: D.gumbelCopula,
  frank: D.frankCopula,
}

const custom: Record<string, Custom> = {
  'gaussian.logDensity': (xs) => D.gaussianCopula(t(xs, 0)).logDensity(t(xs, 1)),
  'gaussian.cdf': (xs) => D.gaussianCopula(t(xs, 0)).cdf(t(xs, 1)),
  'studentT.logDensity': (xs, c) => D.studentTCopula(t(xs, 0), c.params['df'] as number).logDensity(t(xs, 1)),
  'fit.gaussian': (xs): Value =>
    squeeze(take(take(D.fitCopula('gaussian', t(xs, 0)).params['correlation'] as Value, [0], 0), [1], 1)),
}
for (const [name, make] of Object.entries(archimedean)) {
  custom[`${name}.logDensity`] = (xs, c) => make(t(xs, 0), dim(c)).logDensity(t(xs, 1))
  custom[`${name}.cdf`] = (xs, c) => make(t(xs, 0), dim(c)).cdf(t(xs, 1))
  custom[`${name}.kendallTau`] = (xs, c) => make(t(xs, 0), dim(c)).kendallTau()
  custom[`fit.${name}`] = (xs) => D.fitCopula(name as D.CopulaFamily, t(xs, 0)).params['theta'] as Value
}

for (const file of ['copulas', 'archimedean_copulas', 'copula_fitting'])
  describeDistributions(`dists/multivariate/${file}`, {}, custom)
