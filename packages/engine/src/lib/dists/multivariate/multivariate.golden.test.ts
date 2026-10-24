import { type Custom, describeDistributions, type Family } from '../../../testing/distributions.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import * as D from '../index.ts'

const t = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor
type Moment = 'mean' | 'variance' | 'entropy'
const n = (c: { readonly params: Readonly<Record<string, unknown>> }): number => c.params['n'] as number

const mvn = (xs: readonly Tensor[]): D.MultivariateNormal => D.multivariateNormal(t(xs, 0), { covariance: t(xs, 1) })
const lowRank = (xs: readonly Tensor[]): D.LowRankMultivariateNormal =>
  D.lowRankMultivariateNormal(t(xs, 0), t(xs, 1), t(xs, 2))
const matrix = (xs: readonly Tensor[]): D.Distribution => D.matrixNormal(t(xs, 0), t(xs, 1), t(xs, 2))

const CUSTOM: Readonly<Record<string, Custom>> = {
  'multivariateNormal.logProb': (xs) => mvn(xs).logProb(t(xs, 2)),
  'multivariateNormal.entropy': (xs) => mvn(xs).entropy(),
  'multivariateNormal.variance': (xs) => mvn(xs).variance(),
  'multivariateNormal.conditionMean': (xs) => mvn(xs).condition([2], [1.5]).loc,
  'multivariateNormal.conditionCov': (xs) => mvn(xs).condition([2], [1.5]).covariance(),
  'lowRankMultivariateNormal.logProb': (xs) => lowRank(xs).logProb(t(xs, 3)),
  'lowRankMultivariateNormal.entropy': (xs) => lowRank(xs).entropy(),
  'lowRankMultivariateNormal.variance': (xs) => lowRank(xs).variance(),
  'matrixNormal.logProb': (xs) => matrix(xs).logProb(t(xs, 3)),
  'matrixNormal.entropy': (xs) => matrix(xs).entropy(),
  'dirichlet.logProb': (xs) => D.dirichlet(t(xs, 0)).logProb(t(xs, 1)),
  'multinomial.logProb': (xs, c) => D.multinomial(n(c), t(xs, 0)).logProb(t(xs, 1)),
  'multinomial.entropy': (xs, c) => D.multinomial(n(c), t(xs, 0)).entropy(),
  'multinomial.variance': (xs, c) => D.multinomial(n(c), t(xs, 0)).variance(),
  'dirichletMultinomial.logProb': (xs, c) => D.dirichletMultinomial(n(c), t(xs, 0)).logProb(t(xs, 1)),
  'dirichletMultinomial.mean': (xs, c) => D.dirichletMultinomial(n(c), t(xs, 0)).mean(),
  'dirichletMultinomial.variance': (xs, c) => D.dirichletMultinomial(n(c), t(xs, 0)).variance(),
}
const moment =
  (m: Moment): Custom =>
  (xs): Value =>
    D.dirichlet(t(xs, 0))[m]()
const wishart = (xs: readonly Tensor[]): D.Distribution => D.wishart(t(xs, 0), { covariance: t(xs, 1) })
const invWishart = (xs: readonly Tensor[]): D.Distribution => D.inverseWishart(t(xs, 0), t(xs, 1))
type Method = 'mean' | 'variance' | 'mode' | 'entropy'
const MATRIX: Record<string, Custom> = {
  'wishart.logProb': (xs) => wishart(xs).logProb(t(xs, 2)),
  'inverseWishart.logProb': (xs) => invWishart(xs).logProb(t(xs, 2)),
  'lkjCholesky.logProb': (xs, c) => D.lkjCholesky(c.params['dim'] as number, t(xs, 0)).logProb(t(xs, 1)),
  multigammaln: (xs, c) => D.multigammaln(t(xs, 0), c.params['p'] as number),
}
for (const m of ['mean', 'variance', 'mode', 'entropy'] as const satisfies readonly Method[]) {
  MATRIX[`wishart.${m}`] = (xs) => wishart(xs)[m]()
  MATRIX[`inverseWishart.${m}`] = (xs) => invWishart(xs)[m]()
}
const ALL: Readonly<Record<string, Custom>> = {
  ...CUSTOM,
  ...MATRIX,
  'dirichlet.mean': moment('mean'),
  'dirichlet.variance': moment('variance'),
  'dirichlet.entropy': moment('entropy'),
}

const FAMILIES: Readonly<Record<string, Family>> = {
  vonMises: (p) => D.vonMises(t(p, 0), t(p, 1)),
  vonMisesFisher: (p) => D.vonMisesFisher(t(p, 0), t(p, 1)),
}

for (const file of ['normal', 'simplex', 'matrix', 'directional'])
  describeDistributions(`dists/multivariate/${file}`, FAMILIES, ALL)
