// Distributional checks of the samplers in continuous.ts and discrete.ts against scipy.stats: each fixture case holds
// quantiles x (scipy's ppf at fixed levels) and the reference CDF F(x); the empirical CDF of many draws must agree
// within five standard errors of a proportion.
import { describe, expect, it } from 'vitest'
import { type GoldenCase, loadFixture } from '../../testing/golden.ts'
import { decodeNumber } from '../base/index.ts'
import { type TensorJson, tensorFromJson, toFlat } from '../tensor/index.ts'
import {
  aliasSample,
  aliasTable,
  bernoulli,
  beta,
  binomial,
  categorical,
  cauchy,
  chisquare,
  dirichlet,
  exponential,
  gamma,
  geometric,
  gumbel,
  type Key,
  keyFromPath,
  laplace,
  logistic,
  multinomial,
  multivariateNormal,
  normal,
  poisson,
  randint,
  studentT,
  truncatedNormal,
  uniform,
  vonMises,
} from './index.ts'

const N = 20000

function expectCdfMatches(samples: ArrayLike<number>, expected: unknown, label: string): void {
  const ref = tensorFromJson(expected as TensorJson)
  const m = ref.shape[1] ?? 0
  for (let c = 0; c < m; c++) {
    const x = ref.data[c] ?? Number.NaN
    const f = ref.data[m + c] ?? Number.NaN
    let below = 0
    for (let i = 0; i < samples.length; i++) if ((samples[i] ?? Number.NaN) <= x) below++
    const empirical = below / samples.length
    const band = 5 * Math.sqrt((f * (1 - f)) / samples.length) + 1e-12
    expect(Math.abs(empirical - f), `${label}: P(X <= ${x}) = ${empirical}, scipy ${f}`).toBeLessThanOrEqual(band)
  }
}

const num = (c: GoldenCase, name: string): number => decodeNumber(c.params[name] as number | string)
const nums = (c: GoldenCase, name: string): number[] => (c.params[name] as number[]).map(Number)

function column(flat: ArrayLike<number>, width: number, j: number): number[] {
  const out: number[] = []
  for (let i = j; i < flat.length; i += width) out.push(flat[i] ?? Number.NaN)
  return out
}

type Draw = (k: Key, c: GoldenCase) => ArrayLike<number>

const CONTINUOUS: Readonly<Record<string, Draw>> = {
  uniform: (k, c) => uniform(k, [N], { low: num(c, 'low'), high: num(c, 'high') }).data,
  normal: (k, c) => normal(k, [N], { loc: num(c, 'loc'), scale: num(c, 'scale') }).data,
  exponential: (k, c) => exponential(k, [N], num(c, 'rate')).data,
  laplace: (k, c) => laplace(k, [N], num(c, 'loc'), num(c, 'scale')).data,
  logistic: (k, c) => logistic(k, [N], num(c, 'loc'), num(c, 'scale')).data,
  gumbel: (k, c) => gumbel(k, [N], num(c, 'loc'), num(c, 'scale')).data,
  cauchy: (k, c) => cauchy(k, [N], num(c, 'loc'), num(c, 'scale')).data,
  chisquare: (k, c) => chisquare(k, [N], num(c, 'df')).data,
  studentT: (k, c) => studentT(k, [N], num(c, 'df')).data,
  truncatedNormal: (k, c) => truncatedNormal(k, [N], num(c, 'lower'), num(c, 'upper')).data,
  gamma: (k, c) => gamma(k, [N], num(c, 'alpha')).data,
  beta: (k, c) => beta(k, [N], num(c, 'a'), num(c, 'b')).data,
  vonMises: (k, c) => vonMises(k, [N], num(c, 'mu'), num(c, 'kappa')).data,
  dirichletMarginal: (k, c) => {
    const alpha = nums(c, 'alpha')
    return column(dirichlet(k, alpha, N).data, alpha.length, num(c, 'component'))
  },
  multivariateNormalMarginal: (k, c) =>
    column(
      toFlat(
        multivariateNormal(
          k,
          [0, 1],
          [
            [1, 0],
            [0.5, 1],
          ],
          N,
        ),
      ),
      2,
      num(c, 'component'),
    ),
}

const DISCRETE: Readonly<Record<string, Draw>> = {
  randint: (k, c) => randint(k, [N], num(c, 'low'), num(c, 'high')).data,
  bernoulli: (k, c) => bernoulli(k, num(c, 'p'), [N]).data,
  categorical: (k, c) => categorical(k, nums(c, 'weights'), [N]).data,
  categoricalLogits: (k, c) => categorical(k, nums(c, 'logits'), [N], { logits: true }).data,
  aliasSample: (k, c) => aliasSample(k, aliasTable(nums(c, 'weights')), [N]).data,
  poisson: (k, c) => poisson(k, num(c, 'rate'), [N]).data,
  binomial: (k, c) => binomial(k, num(c, 'n'), num(c, 'p'), [N]).data,
  geometric: (k, c) => geometric(k, num(c, 'p'), [N]).data,
  multinomialMarginal: (k, c) => {
    const probs = nums(c, 'probs')
    return column(multinomial(k, num(c, 'n'), probs, N).data, probs.length, num(c, 'component'))
  },
}

for (const [file, table] of [
  ['random/continuous', CONTINUOUS],
  ['random/discrete', DISCRETE],
] as const) {
  describe(`${file} against scipy.stats`, () => {
    for (const c of loadFixture(file).cases) {
      it(c.name, () => {
        const draw = table[c.fn]
        if (draw === undefined) throw new Error(`no sampler for ${c.fn}`)
        expectCdfMatches(draw(keyFromPath(0, c.name), c), c.output, c.name)
      })
    }
  })
}
