import { describe, expect, it } from 'vitest'
import * as R from '../random/index.ts'
import { spearman } from '../stats/index.ts'
import { chisquare, kstest } from '../stats/tests/index.ts'
import { item, toFlat, type Value } from '../tensor/index.ts'
import * as D from './index.ts'

// Goodness of fit of the samplers: a seeded sample of 20,000 draws against the exact distribution function of the same law
// (one-sample Kolmogorov-Smirnov for a continuous law, Pearson's chi-square on the counts for a discrete one). The seeds are
// fixed, so the outcome is deterministic; the level (a p-value of at least 1e-3) is generous enough that a correct sampler
// passes at any seed, and a sampler off by one percent of probability anywhere fails (the last test shows the power).
const N = 20000
const SEED = 7
const LEVEL = 1e-3
const flat = (v: Value): number[] => Array.from(toFlat(v))

// `n` is the sample size where it is not `N`: the distribution functions of the skew normal and the noncentral t are slow.
const wrapped = (x: number[], mu: number): number[] => x.map((t) => Math.atan2(Math.sin(t - mu), Math.cos(t - mu)))

type Row = readonly [name: string, draw: (k: R.Key, n: number) => number[], law: D.Distribution, n?: number]

// The samplers of `random`, each against the `dists` distribution of the same law.
const randomRows: readonly Row[] = [
  ['uniform(2, 5)', (k, n) => flat(R.uniform(k, [n], { low: 2, high: 5 })), D.uniform(2, 5)],
  ['normal(1, 2)', (k, n) => flat(R.normal(k, [n], { loc: 1, scale: 2 })), D.normal(1, 2)],
  ['truncatedNormal(-1, 2)', (k, n) => flat(R.truncatedNormal(k, [n], -1, 2)), D.truncatedNormal(0, 1, -1, 2)],
  [
    'truncatedNormal in the far tail (3, 5)',
    (k, n) => flat(R.truncatedNormal(k, [n], 3, 5)),
    D.truncatedNormal(0, 1, 3, 5),
  ],
  ['exponential(2)', (k, n) => flat(R.exponential(k, [n], 2)), D.exponential(2)],
  ['laplace(1, 2)', (k, n) => flat(R.laplace(k, [n], 1, 2)), D.laplace(1, 2)],
  ['logistic(1, 2)', (k, n) => flat(R.logistic(k, [n], 1, 2)), D.logistic(1, 2)],
  ['gumbel(1, 2)', (k, n) => flat(R.gumbel(k, [n], 1, 2)), D.gumbel(1, 2)],
  ['cauchy(1, 2)', (k, n) => flat(R.cauchy(k, [n], 1, 2)), D.cauchy(1, 2)],
  ['gamma(0.5)', (k, n) => flat(R.gamma(k, [n], 0.5)), D.gamma(0.5, 1)],
  ['gamma(2)', (k, n) => flat(R.gamma(k, [n], 2)), D.gamma(2, 1)],
  ['gamma(20)', (k, n) => flat(R.gamma(k, [n], 20)), D.gamma(20, 1)],
  ['beta(0.5, 0.5)', (k, n) => flat(R.beta(k, [n], 0.5, 0.5)), D.beta(0.5, 0.5)],
  ['beta(2, 3)', (k, n) => flat(R.beta(k, [n], 2, 3)), D.beta(2, 3)],
  ['beta(30, 40)', (k, n) => flat(R.beta(k, [n], 30, 40)), D.beta(30, 40)],
  ['chisquare(4)', (k, n) => flat(R.chisquare(k, [n], 4)), D.chi2(4)],
  ['studentT(3)', (k, n) => flat(R.studentT(k, [n], 3)), D.studentT(3, 0, 1)],
  ['studentT(30)', (k, n) => flat(R.studentT(k, [n], 30)), D.studentT(30, 0, 1)],
  // `random.vonMises` returns angles in (-pi, pi]; the distribution's support is [mu - pi, mu + pi]: wrap the offset from mu.
  ['vonMises(0.5, 0.1)', (k, n) => wrapped(flat(R.vonMises(k, [n], 0.5, 0.1)), 0.5), D.vonMises(0, 0.1)],
  ['vonMises(1, 5)', (k, n) => wrapped(flat(R.vonMises(k, [n], 1, 5)), 1), D.vonMises(0, 5)],
  ['vonMises(0, 50)', (k, n) => wrapped(flat(R.vonMises(k, [n], 0, 50)), 0), D.vonMises(0, 50)],
  // A projection of a multivariate normal: w . x has variance w' Sigma w for Sigma = L L'.
  [
    'multivariateNormal projected',
    (k, n) => {
      const x = flat(
        R.multivariateNormal(
          k,
          [1, -1],
          [
            [2, 0],
            [1, 0.5],
          ],
          n,
        ),
      )
      return Array.from({ length: n }, (_, i) => 0.6 * (x[2 * i] ?? 0) + 0.8 * (x[2 * i + 1] ?? 0))
    },
    // x = mean + L z with L = [[2, 0], [1, 0.5]]: w . x = -0.2 + (w'L) z with w'L = [2, 0.4], so the variance is 4 + 0.16.
    D.normal(-0.2, Math.sqrt(4.16)),
  ],
  // The first component of a Dirichlet is beta(a_1, a_2 + a_3).
  ['dirichlet component', (k, n) => flat(R.dirichlet(k, [2, 3, 5], n)).filter((_, i) => i % 3 === 0), D.beta(2, 8)],
]

// The sampling methods of the distributions (`sample` of the family), whose own code differs from `random`'s.
const familyRows: readonly Row[] = (
  [
    ['halfNormal(2)', D.halfNormal(2)],
    ['logNormal(0, 0.5)', D.logNormal(0, 0.5)],
    ['skewNormal(0, 1, 3)', D.skewNormal(0, 1, 3)],
    ['kumaraswamy(2, 3)', D.kumaraswamy(2, 3)],
    ['logitNormal(0, 1)', D.logitNormal(0, 1)],
    ['triangular(0, 1, 3)', D.triangular(0, 1, 3)],
    ['halfCauchy(1)', D.halfCauchy(1)],
    ['pareto(3, 1)', D.pareto(3, 1)],
    ['johnsonSU(0.5, 2, 0, 1)', D.johnsonSU(0.5, 2, 0, 1)],
    ['sinhArcsinh(0, 1, 0.3, 1.2)', D.sinhArcsinh(0, 1, 0.3, 1.2)],
    ['gamma(0.3, 1)', D.gamma(0.3, 1)],
    ['gamma(30, 2)', D.gamma(30, 2)],
    ['inverseGamma(3, 2)', D.inverseGamma(3, 2)],
    ['weibull(1.5, 2)', D.weibull(1.5, 2)],
    ['inverseGaussian(1, 2)', D.inverseGaussian(1, 2)],
    ['inverseGaussian(1, 0.2)', D.inverseGaussian(1, 0.2)],
    ['fDistribution(5, 7)', D.fDistribution(5, 7)],
    ['noncentralChi2(3, 2)', D.noncentralChi2(3, 2)],
    ['noncentralT(5, 1)', D.noncentralT(5, 1)],
    ['generalisedExtremeValue(0, 1, 0.2)', D.generalisedExtremeValue(0, 1, 0.2)],
    ['generalisedPareto(0, 1, 0.2)', D.generalisedPareto(0, 1, 0.2)],
    ['frechet(3, 0, 1)', D.frechet(3, 0, 1)],
    ['vonMises(0, 2)', D.vonMises(0, 2)],
    ['beta(0.4, 0.7)', D.beta(0.4, 0.7)],
  ] as const
).map(
  ([name, law]) =>
    [
      `${name}.sample`,
      (k: R.Key, n: number) => flat(law.sample(k, [n])),
      law,
      name.startsWith('skewNormal') || name.startsWith('noncentralT') ? 3000 : undefined,
    ] as const,
)

describe('the samplers draw from the distribution they claim', () => {
  it('Kolmogorov-Smirnov: 20,000 draws of every continuous sampler match the distribution function, and the same test rejects a shifted, rescaled or mis-shaped law', () => {
    const wrong: readonly (readonly [string, number[], D.Distribution])[] = [
      ['normal against a mean off by 0.05', flat(R.normal(R.key(1), [N])), D.normal(0.05, 1)],
      ['normal against a scale off by 10%', flat(R.normal(R.key(1), [N])), D.normal(0, 1.1)],
      ['gamma(2) against gamma(2.2)', flat(R.gamma(R.key(1), [N], 2)), D.gamma(2.2, 1)],
      ['uniform against a quarter of the range', flat(R.uniform(R.key(1), [N])), D.uniform(0.02, 1)],
    ]
    for (const [name, x, law] of wrong) expect(kstest(x, law, { method: 'asymp' }).pValue, name).toBeLessThan(1e-6)
    for (const [name, draw, law, n = N] of [...randomRows, ...familyRows]) {
      const r = kstest(draw(R.key(SEED), n), law, { method: 'asymp' })
      expect(r.pValue, `${name}: D = ${r.statistic}`).toBeGreaterThan(LEVEL)
    }
  })

  const atoms = (law: D.Distribution): { values: number[]; probs: number[] } => {
    const lo = item(law.ppf(1e-9))
    const hi = item(law.ppf(1 - 1e-9))
    const values = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
    return { values, probs: flat(law.prob(values)) }
  }
  /** Counts of the draws on the atoms, with every atom of expected count below 5 pooled into the tails. */
  function countsAgainst(draws: number[], law: D.Distribution): { observed: number[]; expected: number[] } {
    const { values, probs } = atoms(law)
    const lowest = values[0] ?? 0
    const observed = new Array<number>(values.length).fill(0)
    for (const v of draws) {
      const at = Math.min(values.length - 1, Math.max(0, v - lowest))
      observed[at] = (observed[at] ?? 0) + 1
    }
    const bins = values.map((_, i) => [observed[i] ?? 0, (probs[i] ?? 0) * draws.length] as [number, number])
    // Pool from the left and from the right until every bin expects at least 5 draws.
    while (bins.length > 2 && (bins[0]?.[1] ?? 0) < 5) {
      const [a, b] = [bins.shift() ?? [0, 0], bins.shift() ?? [0, 0]]
      bins.unshift([a[0] + b[0], a[1] + b[1]])
    }
    while (bins.length > 2 && (bins[bins.length - 1]?.[1] ?? 0) < 5) {
      const [a, b] = [bins.pop() ?? [0, 0], bins.pop() ?? [0, 0]]
      bins.push([a[0] + b[0], a[1] + b[1]])
    }
    const total = bins.reduce((s, b) => s + b[1], 0)
    // The mass beyond the atoms (below 2e-9) is added to the expected counts so that both totals agree.
    return { observed: bins.map((b) => b[0]), expected: bins.map((b) => (b[1] * draws.length) / total) }
  }

  const discreteRows: readonly Row[] = [
    ['randint(0, 7)', (k, n) => flat(R.randint(k, [n], 0, 7)), D.discreteUniform(0, 6)],
    ['bernoulli(0.3)', (k, n) => flat(R.bernoulli(k, 0.3, [n])), D.bernoulli({ probs: 0.3 })],
    ['categorical', (k, n) => flat(R.categorical(k, [0.2, 0.5, 0.3], [n])), D.categorical({ probs: [0.2, 0.5, 0.3] })],
    [
      'categorical (logits)',
      (k, n) => flat(R.categorical(k, [0, -1, 2], [n], { logits: true })),
      D.categorical({ logits: [0, -1, 2] }),
    ],
    [
      'aliasSample',
      (k, n) => flat(R.aliasSample(k, R.aliasTable([1, 4, 2, 3, 0.5, 6]), [n])),
      D.categorical({ probs: [1 / 16.5, 4 / 16.5, 2 / 16.5, 3 / 16.5, 0.5 / 16.5, 6 / 16.5] }),
    ],
    ['poisson(0.5)', (k, n) => flat(R.poisson(k, 0.5, [n])), D.poisson(0.5)],
    ['poisson(4)', (k, n) => flat(R.poisson(k, 4, [n])), D.poisson(4)],
    ['poisson(60)', (k, n) => flat(R.poisson(k, 60, [n])), D.poisson(60)],
    ['binomial(10, 0.3)', (k, n) => flat(R.binomial(k, 10, 0.3, [n])), D.binomial(10, 0.3)],
    ['binomial(100, 0.6)', (k, n) => flat(R.binomial(k, 100, 0.6, [n])), D.binomial(100, 0.6)],
    ['binomial(1000, 0.02)', (k, n) => flat(R.binomial(k, 1000, 0.02, [n])), D.binomial(1000, 0.02)],
    ['geometric(0.2)', (k, n) => flat(R.geometric(k, 0.2, [n])), D.geometric(0.2)],
    ...(
      [
        ['poisson(0.5)', D.poisson(0.5)],
        ['poisson(60)', D.poisson(60)],
        ['binomial(10, 0.3)', D.binomial(10, 0.3)],
        ['binomial(100, 0.6)', D.binomial(100, 0.6)],
        ['geometric(0.7)', D.geometric(0.7)],
        ['negativeBinomial(3, 0.4)', D.negativeBinomial(3, 0.4)],
        ['negativeBinomial(40, 0.9)', D.negativeBinomial(40, 0.9)],
        ['hypergeometric(50, 20, 10)', D.hypergeometric(50, 20, 10)],
        ['betaBinomial(10, 2, 3)', D.betaBinomial(10, 2, 3)],
        ['discreteUniform(2, 7)', D.discreteUniform(2, 7)],
        ['zeroInflatedPoisson(3, 0.2)', D.zeroInflatedPoisson(3, 0.2)],
      ] as const
    ).map(([name, law]) => [`${name}.sample`, (k: R.Key, n: number) => flat(law.sample(k, [n])), law] as const),
  ]

  it("Pearson's chi-square: 20,000 draws of every discrete sampler match the probability mass function", () => {
    for (const [name, draw, law] of discreteRows) {
      const { observed, expected } = countsAgainst(draw(R.key(SEED), N), law)
      const r = chisquare(observed, { expected })
      expect(r.pValue, `${name}: chi-square ${r.statistic} on ${observed.length} bins`).toBeGreaterThan(LEVEL)
    }
  })
  it('draws a stream that depends on the key alone: the same key repeats, other keys are uncorrelated with it', () => {
    const n = 2000
    for (const [name, draw] of [...randomRows, ...familyRows, ...discreteRows]) {
      const a = draw(R.key(5), n)
      expect(draw(R.key(5), n), `${name}: the same key twice`).toEqual(a)
      // Rank correlation with the streams of another seed, of the two children of the key, and of a folded key: its standard
      // deviation is 1 / sqrt(2000), so 6 standard deviations is passed by chance in 2e-9 of the cases.
      for (const other of [R.key(6), ...R.split(R.key(5)), R.foldIn(R.key(5), 1)]) {
        const b = draw(other, n)
        expect(b, `${name}: another key`).not.toEqual(a)
        expect(Math.abs(spearman(a, b).statistic), `${name}: correlation with another key`).toBeLessThan(
          6 / Math.sqrt(n),
        )
      }
    }
  })
})
