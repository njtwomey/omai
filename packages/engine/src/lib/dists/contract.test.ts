import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { gaussLegendre } from '../numerics/quadrature/index.ts'
import { key } from '../random/index.ts'
import { item, toFlat, type Value } from '../tensor/index.ts'
import * as D from './index.ts'

// Contract checks over one table of (family, parameters) rows. Each check is a single test that loops over the rows and
// names the row in its failure message, so a new family costs one line here and no test.
const families: readonly (readonly [string, () => D.Distribution])[] = [
  ['normal', () => D.normal(1, 2)],
  ['halfNormal', () => D.halfNormal(2)],
  ['logNormal', () => D.logNormal(0, 1)],
  ['truncatedNormal', () => D.truncatedNormal(0, 1, -1, 2)],
  ['skewNormal', () => D.skewNormal(0, 1, 3)],
  ['uniform', () => D.uniform(-1, 3)],
  ['beta', () => D.beta(2, 3)],
  ['beta (a > 1 >= b)', () => D.beta(3, 0.5)],
  ['kumaraswamy', () => D.kumaraswamy(2, 3)],
  ['logitNormal', () => D.logitNormal(0, 1)],
  ['triangular', () => D.triangular(0, 1, 3)],
  ['studentT', () => D.studentT(3, 0, 1)],
  ['cauchy', () => D.cauchy(0, 1)],
  ['halfCauchy', () => D.halfCauchy(1)],
  ['laplace', () => D.laplace(0, 1)],
  ['logistic', () => D.logistic(0, 1)],
  ['pareto', () => D.pareto(3, 1)],
  ['johnsonSU', () => D.johnsonSU(0.5, 2, 0, 1)],
  ['gamma', () => D.gamma(2, 1.5)],
  ['gamma (shape below 1)', () => D.gamma(0.3, 1)],
  ['exponential', () => D.exponential(2)],
  ['inverseGamma', () => D.inverseGamma(3, 2)],
  ['chi2', () => D.chi2(4)],
  ['weibull', () => D.weibull(1.5, 2)],
  ['inverseGaussian', () => D.inverseGaussian(1, 2)],
  ['fDistribution', () => D.fDistribution(5, 7)],
  ['fDistribution (d1 below 2)', () => D.fDistribution(1, 7)],
  ['noncentralChi2', () => D.noncentralChi2(3, 2)],
  ['noncentralT', () => D.noncentralT(5, 1)],
  ['gumbel', () => D.gumbel(0, 1)],
  ['generalisedExtremeValue', () => D.generalisedExtremeValue(0, 1, 0.2)],
  ['generalisedPareto', () => D.generalisedPareto(0, 1, 0.2)],
  ['frechet', () => D.frechet(3, 0, 1)],
  ['vonMises', () => D.vonMises(0, 2)],
  ['vonMises (loc outside the support)', () => D.vonMises(3, 0.5)],
  ['poisson', () => D.poisson(3)],
  ['binomial', () => D.binomial(20, 0.3)],
  ['binomial (p = 1)', () => D.binomial(20, 1)],
  ['geometric', () => D.geometric(0.2)],
  ['negativeBinomial', () => D.negativeBinomial(3, 0.4)],
  ['hypergeometric', () => D.hypergeometric(50, 20, 10)],
  ['betaBinomial', () => D.betaBinomial(10, 2, 3)],
  ['discreteUniform', () => D.discreteUniform(2, 7)],
  ['categorical', () => D.categorical({ probs: [0.2, 0.5, 0.3] })],
  ['categorical (one class)', () => D.categorical({ probs: [1] })],
  ['categorical (masked)', () => D.categorical({ logits: [0, Number.NEGATIVE_INFINITY, 0] })],
  ['sinhArcsinh', () => D.sinhArcsinh(0, 1, 0.3, 1.2)],
  ['erlang', () => D.erlang(3, 2)],
  ['bernoulli', () => D.bernoulli({ probs: 0.3 })],
  ['zeroInflatedPoisson', () => D.zeroInflatedPoisson(3, 0.2)],
]

const methods: readonly (readonly [string, (d: D.Distribution, x: number) => number])[] = [
  ['logProb', (d, x) => item(d.logProb(x))],
  ['prob', (d, x) => item(d.prob(x))],
  ['cdf', (d, x) => item(d.cdf(x))],
  ['sf', (d, x) => item(d.sf(x))],
  ['logCdf', (d, x) => item(d.logCdf(x))],
  ['logSf', (d, x) => item(d.logSf(x))],
  ['ppf', (d, x) => item(d.ppf(x))],
  ['isf', (d, x) => item(d.isf(x))],
]

describe('every family follows the Distribution contract', () => {
  it('answers NaN for a NaN argument in every density, distribution and quantile function', () => {
    for (const [name, make] of families) {
      const d = make()
      for (const [method, call] of methods) expect(call(d, Number.NaN), `${name}.${method}(NaN)`).toBeNaN()
    }
  })

  it('has the limits of a density and a distribution function at an infinite argument', () => {
    const inf = Number.POSITIVE_INFINITY
    for (const [name, make] of families) {
      const d = make()
      expect([item(d.logProb(-inf)), item(d.logProb(inf)), item(d.prob(inf))], `${name}: density`).toEqual([
        -inf,
        -inf,
        0,
      ])
      expect([item(d.cdf(-inf)), item(d.cdf(inf)), item(d.sf(-inf)), item(d.sf(inf))], `${name}: cdf and sf`).toEqual([
        0, 1, 1, 0,
      ])
      expect(
        [d.logCdf(-inf), d.logCdf(inf), d.logSf(-inf), d.logSf(inf)].map((v) => item(v) + 0),
        `${name}: log`,
      ).toEqual([-inf, 0, 0, -inf])
    }
  })

  it('has its mode in the support (or a point of infinite or finite density) and every sample inside it', () => {
    for (const [name, make] of families) {
      const d = make()
      const inside =
        item(D.inSupport(d.support, d.mode())) === 1 || item(d.logProb(d.mode())) > Number.NEGATIVE_INFINITY
      expect(inside, `${name}: mode ${item(d.mode())}`).toBe(true)
      expect(toFlat(D.inSupport(d.support, d.sample(key(5), [200]))).every(Boolean), `${name}: samples`).toBe(true)
    }
  })

  it('has a CDF that inverts its quantile function (exactly when continuous, from above when discrete)', () => {
    for (const [name, make] of families) {
      const d = make()
      const discrete = d.support.kind === 'integerInterval' || d.support.kind === 'boolean'
      for (const p of [0.001, 0.3, 0.9, 0.999]) {
        const c = item(d.cdf(d.ppf(p)))
        expect(c, `${name}: cdf(ppf(${p}))`).toBeGreaterThanOrEqual(p - 1e-9)
        if (!discrete) expect(c, `${name}: cdf(ppf(${p}))`).toBeLessThanOrEqual(p + 1e-9)
      }
    }
  })
})

describe('closed forms at the edge of the parameter range', () => {
  const lp0 = (d: D.Distribution): number => item(d.logProb(0))
  const rows: readonly (readonly [string, () => number, number])[] = [
    ['beta(3, 0.5) mode is 1', () => item(D.beta(3, 0.5).mode()), 1],
    ['beta(0.5, 3) mode is 0', () => item(D.beta(0.5, 3).mode()), 0],
    ['beta(0.5, 0.5) is bimodal: its mode is undefined', () => item(D.beta(0.5, 0.5).mode()), Number.NaN],
    ['beta(2, 3) mode is 1/3', () => item(D.beta(2, 3).mode()), 1 / 3],
    ['dirichlet([0.5, 3]) mode is [0, 1]', () => toFlat(D.dirichlet([0.5, 3]).mode())[0] ?? Number.NaN, 0],
    ['binomial(20, 1) mode is 20', () => item(D.binomial(20, 1).mode()), 20],
    ['binomial(20, 0.3) mode is 6', () => item(D.binomial(20, 0.3).mode()), 6],
    ['F(1, 7) density at 0 is infinite', () => lp0(D.fDistribution(1, 7)), Number.POSITIVE_INFINITY],
    ['F(3, 7) density at 0 is zero', () => lp0(D.fDistribution(3, 7)), Number.NEGATIVE_INFINITY],
    ['vonMises(3, 0.5) cdf at loc - pi is 0', () => item(D.vonMises(3, 0.5).cdf(3 - Math.PI)), 0],
    ['vonMises(3, 0.5) cdf at loc + pi is 1', () => item(D.vonMises(3, 0.5).cdf(3 + Math.PI)), 1],
    ['vonMises(3, 0.5) support starts at loc - pi', () => (D.vonMises(3, 0.5).support.kind === 'interval' ? 1 : 0), 1],
    [
      'masked categorical entropy is log 2',
      () => item(D.categorical({ logits: [0, -Infinity, 0] }).entropy()),
      Math.LN2,
    ],
    ['kl(categorical [1, 0], itself) is 0', () => klCategorical([1, 0], [1, 0]), 0],
    ['kl(categorical [1, 0], [0.5, 0.5]) is log 2', () => klCategorical([1, 0], [0.5, 0.5]), Math.LN2],
    [
      'kl(categorical [0.5, 0.5], [1, 0]) is infinite',
      () => klCategorical([0.5, 0.5], [1, 0]),
      Number.POSITIVE_INFINITY,
    ],
    [
      'kl(bernoulli 0, bernoulli 0.5) is log 2',
      () => item(D.klDivergence(D.bernoulli({ probs: 0 }), D.bernoulli({ probs: 0.5 }))),
      Math.LN2,
    ],
    ['categorical with one class samples 0', () => item(D.categorical({ probs: [1] }).sample(key(1), [])), 0],
  ]
  function klCategorical(p: number[], q: number[]): number {
    return item(D.klDivergence(D.categorical({ probs: p }), D.categorical({ probs: q })))
  }

  it('hold', () => {
    for (const [name, value, expected] of rows) {
      if (Number.isFinite(expected)) expect(value(), name).toBeCloseTo(expected, 12)
      else if (Number.isNaN(expected)) expect(value(), name).toBeNaN()
      else expect(value(), name).toBe(expected)
    }
  })
})

describe('parameter validation', () => {
  const invalid: readonly (readonly [string, () => unknown])[] = [
    ['uniform(0, -1)', () => D.uniform(0, -1)],
    ['uniform(0, 0)', () => D.uniform(0, 0)],
    ['triangular with the mode below the range', () => D.triangular(0, -1, 1)],
    ['triangular with the mode above the range', () => D.triangular(0, 2, 1)],
    ['triangular with low above high', () => D.triangular(2, 1, 0)],
    ['truncatedNormal with low above high', () => D.truncatedNormal(0, 1, 2, 1)],
    ['studentisedRange with zero degrees of freedom', () => D.studentisedRange(3, 0)],
    ['normal with an infinite scale', () => D.normal(0, Number.POSITIVE_INFINITY)],
    ['gamma with an infinite concentration', () => D.gamma(Number.POSITIVE_INFINITY, 1)],
    ['studentT with infinite degrees of freedom', () => D.studentT(Number.POSITIVE_INFINITY, 0, 1)],
    ['poisson with an infinite rate', () => D.poisson(Number.POSITIVE_INFINITY)],
    ['geometric(0)', () => D.geometric(0)],
    ['negativeBinomial(3, 0)', () => D.negativeBinomial(3, 0)],
    ['hypergeometric with more successes than the total', () => D.hypergeometric(5, 7, 2)],
    ['hypergeometric with more draws than the total', () => D.hypergeometric(5, 3, 9)],
    [
      'multivariateNormal with an asymmetric covariance',
      () =>
        D.multivariateNormal([0, 0], {
          covariance: [
            [1, 0.5],
            [0, 1],
          ],
        }),
    ],
  ]
  it('rejects the invalid parameters of the register with value', () => {
    for (const [name, make] of invalid) expect(make, name).toThrow(expect.objectContaining({ code: 'value' }))
    expect(new OmaiError('value', 'x').code).toBe('value')
  })

  it('accepts the boundary cases', () => {
    expect(() => D.triangular(0, 1, 1)).not.toThrow()
    expect(() => D.triangular(0, 0, 1)).not.toThrow()
    expect(() => D.geometric(1)).not.toThrow()
    expect(() => D.negativeBinomial(3, 1)).not.toThrow()
    expect(() => D.poisson(0)).not.toThrow()
    expect(() => D.studentisedRange(3, Number.POSITIVE_INFINITY)).not.toThrow()
  })
})

// Generic consistency checks over the same table. Each one has an observable oracle that needs no formula from the family:
// the density against the distribution function, the moments and entropy against quadrature of the quantile function (the
// mean of X is the integral of ppf over (0, 1)), and the survival function against 1 - cdf.
const rule16 = gaussLegendre(16)
const nodes16 = Array.from(toFlat(rule16.nodes))
const weights16 = Array.from(toFlat(rule16.weights))

/** Gauss-Legendre of `f` (called once with all the abscissae of the panel) over `[a, b]`. */
function panel(f: (xs: number[]) => ArrayLike<number>, a: number, b: number): number {
  const half = (b - a) / 2
  const values = f(nodes16.map((t) => a + half * (t + 1)))
  return weights16.reduce((acc, w, i) => acc + w * (values[i] ?? Number.NaN), 0) * half
}

const flat = (v: Value): number[] => Array.from(toFlat(v))
const isDiscrete = (d: D.Distribution): boolean => d.support.kind === 'integerInterval' || d.support.kind === 'boolean'

/** Integers from the `1e-12` quantile to the `1 - 1e-12` quantile of a discrete distribution (at most 3000 of them). */
function atoms(d: D.Distribution): number[] {
  const lo = item(d.ppf(1e-12))
  const hi = Math.min(item(d.ppf(1 - 1e-12)), lo + 3000)
  return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
}

/** Edges of the panels of an integral over the probability scale: decades into both tails, then steps of 0.2. */
function probabilityEdges(): number[] {
  const decades = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05]
  const middle = Array.from({ length: 4 }, (_, i) => (i + 1) / 5)
  return [...decades, ...middle, ...decades.map((u) => 1 - u).reverse()]
}

/**
 * The integrals over the probability scale `u` in (0, 1) of `x = ppf(u)`, `x^2` and `-logProb(x)`: the mean, the second
 * moment and the entropy of a continuous law. A node where the log density is infinite (the quantile is rounded onto an
 * edge of the support, `1 - 1e-10` of beta(3, 0.5) is 1) counts as zero: it carries at most 1e-10 of the mass.
 */
function quantileIntegrals(d: D.Distribution): { mean: number; second: number; entropy: number } {
  const kink = [item(d.cdf(d.mode()))].filter((u) => u > 1e-10 && u < 1 - 1e-10)
  const edges = [...probabilityEdges(), ...kink].sort((a, b) => a - b)
  const out = { mean: 0, second: 0, entropy: 0 }
  for (let i = 1; i < edges.length; i++) {
    const [a, b] = [edges[i - 1] ?? 0, edges[i] ?? 1]
    const half = (b - a) / 2
    const x = flat(d.ppf(nodes16.map((t) => a + half * (t + 1))))
    const lp = flat(d.logProb(x))
    weights16.forEach((w, j) => {
      out.mean += w * half * (x[j] ?? 0)
      out.second += w * half * (x[j] ?? 0) ** 2
      out.entropy -= Number.isFinite(lp[j]) ? w * half * (lp[j] ?? 0) : 0
    })
  }
  return out
}

describe('every family is consistent with itself', () => {
  it('has a density that integrates to the probability, and a CDF that is its running integral', () => {
    for (const [name, make] of families) {
      const d = make()
      if (isDiscrete(d)) {
        const ks = atoms(d)
        const pmf = flat(d.prob(ks))
        expect(
          pmf.reduce((a, b) => a + b, 0),
          `${name}: the pmf sums to 1`,
        ).toBeCloseTo(1, 9)
        const cdf = flat(d.cdf(ks))
        let running = flat(d.cdf((ks[0] ?? 0) - 1))[0] ?? 0
        ks.forEach((k, i) => {
          running += pmf[i] ?? 0
          expect(cdf[i], `${name}: cdf(${k}) is the sum of the pmf`).toBeCloseTo(running, 10)
        })
        continue
      }
      const us = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
      // The mode is an edge of a panel, where a density such as the triangular one has its kink.
      const mode = item(d.mode())
      const xs = flat(d.ppf(us))
      if (mode > (xs[0] ?? 0) && mode < (xs[xs.length - 1] ?? 0))
        xs.splice(
          xs.findIndex((x) => x > mode),
          0,
          mode,
        )
      const cdf = flat(d.cdf(xs))
      let total = 0
      for (let i = 1; i < xs.length; i++) {
        const [a, b] = [xs[i - 1] ?? 0, xs[i] ?? 0]
        // Geometric sub-panels next to a singular edge (the density of gamma(0.3) at 0), equal ones elsewhere.
        const cuts = Array.from({ length: 5 }, (_, j) =>
          a > 0 && b / a > 10 ? a * (b / a) ** (j / 4) : a + ((b - a) * j) / 4,
        )
        let mass = 0
        for (let j = 1; j < cuts.length; j++) mass += panel((x) => flat(d.prob(x)), cuts[j - 1] ?? a, cuts[j] ?? b)
        total += mass
        expect(mass, `${name}: integral of the density over [${a}, ${b}]`).toBeCloseTo(
          (cdf[i] ?? 0) - (cdf[i - 1] ?? 0),
          6,
        )
      }
      expect(total, `${name}: mass between the 0.001 and 0.999 quantiles`).toBeCloseTo(0.998, 6)
    }
  })

  it('has moments and an entropy equal to the integrals of its quantile function and log density', () => {
    // Relative to the scale of the quantity. The quantile integral stops at 1e-10 of the mass, which an inverse gamma's
    // tail (second moment: 4e-4 of it) does not allow for, nor a Pareto, Frechet, F or Student's t. The variance of a von
    // Mises law is its circular variance 1 - I1 / I0, not the second moment of the angle, so only the mean is compared.
    const loose: Readonly<Record<string, number>> = {
      inverseGamma: 3e-3,
      studentT: 3e-3,
      pareto: 3e-3,
      fDistribution: 3e-3,
      frechet: 3e-3,
    }
    for (const [name, make] of families) {
      const d = make()
      const tol = (loose[name.split(' ')[0] ?? ''] ?? 2e-4) as number
      const mean = item(d.mean())
      const variance = item(d.variance())
      const entropy = item(d.entropy())
      let m: number
      let v: number
      let h: number
      if (isDiscrete(d)) {
        const ks = atoms(d)
        const pmf = flat(d.prob(ks))
        m = ks.reduce((a, k, i) => a + k * (pmf[i] ?? 0), 0)
        v = ks.reduce((a, k, i) => a + (k - m) ** 2 * (pmf[i] ?? 0), 0)
        h = -pmf.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0)
      } else {
        const q = quantileIntegrals(d)
        m = q.mean
        v = q.second - q.mean ** 2
        h = q.entropy
      }
      if (Number.isFinite(mean))
        expect
          .soft(Math.abs(m - mean), `${name}: mean ${mean}, quadrature ${m}`)
          .toBeLessThan(tol * (1 + Math.abs(mean)))
      if (Number.isFinite(variance) && Number.isFinite(mean) && !name.startsWith('vonMises'))
        expect
          .soft(Math.abs(v - variance), `${name}: variance ${variance}, quadrature ${v}`)
          .toBeLessThan(tol * (1 + variance))
      if (Number.isFinite(entropy))
        expect
          .soft(Math.abs(h - entropy), `${name}: entropy ${entropy}, quadrature ${h}`)
          .toBeLessThan(tol * (1 + Math.abs(entropy)))
    }
  })

  it('has log, survival and inverse-survival functions that agree with density, cdf and quantiles', () => {
    for (const [name, make] of families) {
      const d = make()
      const discrete = isDiscrete(d)
      const us = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]
      const xs = flat(d.ppf(us))
      const [lp, pr, cdf, sf, lc, ls] = [d.logProb(xs), d.prob(xs), d.cdf(xs), d.sf(xs), d.logCdf(xs), d.logSf(xs)].map(
        flat,
      ) as number[][]
      xs.forEach((x, i) => {
        const at = `${name} at ${x}`
        if ((pr?.[i] ?? 0) > 0 && Number.isFinite(lp?.[i]))
          expect(lp?.[i], `${at}: logProb is log(prob)`).toBeCloseTo(Math.log(pr?.[i] ?? 0), 9)
        expect((cdf?.[i] ?? 0) + (sf?.[i] ?? 0), `${at}: cdf + sf`).toBeCloseTo(1, 12)
        expect(lc?.[i], `${at}: logCdf`).toBeCloseTo(Math.log(cdf?.[i] ?? 0), 9)
        expect(ls?.[i], `${at}: logSf`).toBeCloseTo(Math.log(sf?.[i] ?? 0), 9)
      })
      // isf inverts sf: exactly for a continuous law, and for an atomic one as the smallest atom whose sf is at most q.
      const qs = [0.02, 0.1, 0.4, 0.6, 0.9, 0.98]
      const inv = flat(d.isf(qs))
      qs.forEach((q, i) => {
        const x = inv[i] ?? Number.NaN
        if (discrete) {
          expect(item(d.sf(x)), `${name}: sf(isf(${q}))`).toBeLessThanOrEqual(q + 1e-12)
          expect(item(d.sf(x - 1)), `${name}: sf(isf(${q}) - 1)`).toBeGreaterThan(q - 1e-12)
        } else {
          expect(item(d.sf(x)), `${name}: sf(isf(${q}))`).toBeCloseTo(q, 8)
          expect(x, `${name}: isf(${q}) = ppf(1 - ${q})`).toBeCloseTo(item(d.ppf(1 - q)), 6)
        }
      })
      if (!discrete)
        expect(
          flat(d.ppf(cdf as number[]))
            .map((x, i) => Math.abs(x - (xs[i] ?? 0)))
            .every((e) => e < 1e-6 * (1 + Math.abs(xs[0] ?? 0)) + 1e-6 * Math.max(...xs.map(Math.abs))),
          `${name}: ppf inverts cdf`,
        ).toBe(true)
    }
  })
})

// scipy.stats reference values (scipy 1.x, a one-off snippet): the density or mass at x, the distribution function at x, the
// 0.3 quantile, the entropy, the mean and the variance of the same law (for the von Mises law the circular variance
// 1 - I1(2) / I0(2) of scipy.special, which is what `variance` returns).
describe('reference values and divergences', () => {
  const rows: readonly (readonly [
    string,
    () => D.Distribution,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ])[] = [
    [
      'normal(1, 2)',
      () => D.normal(1, 2),
      2.5,
      0.15056871607740221,
      0.7733726476231317,
      -0.04880102541608178,
      2.112085713764618,
      1.0,
      4.0,
    ],
    [
      'gamma(2, 1.5)',
      () => D.gamma(2, 1.5),
      1.2,
      0.4463069981982837,
      0.5371631129795577,
      0.7315661404689944,
      1.1717505567933684,
      1.3333333333333333,
      0.8888888888888888,
    ],
    [
      'beta(2, 3)',
      () => D.beta(2, 3),
      0.4,
      1.7280000000000006,
      0.5247999999999999,
      0.27238394207510536,
      -0.2349066497880008,
      0.4,
      0.04,
    ],
    [
      'studentT(3, 0, 1)',
      () => D.studentT(3, 0, 1),
      0.7,
      0.2715883590882466,
      0.7328365008476182,
      -0.5843897274398188,
      1.7734775718632907,
      0.0,
      3.0,
    ],
    [
      'logNormal(0, 1)',
      () => D.logNormal(0, 1),
      1.8,
      0.18647244853890815,
      0.7216622520650846,
      0.5919101006095541,
      1.4189385332046727,
      1.6487212707001282,
      4.670774270471604,
    ],
    [
      'weibull(1.5, 2)',
      () => D.weibull(1.5, 2),
      1.7,
      0.3158136937291174,
      0.5432692722262781,
      1.005877429831437,
      1.4800872940856251,
      1.805490585901867,
      1.5027611392557279,
    ],
    [
      'gumbel(0, 1)',
      () => D.gumbel(0, 1),
      0.9,
      0.2707472203216076,
      0.6659307054401221,
      -0.18562675886236574,
      1.5772156649015328,
      0.5772156649015329,
      1.6449340668482264,
    ],
    [
      'laplace(0, 1)',
      () => D.laplace(0, 1),
      -0.6,
      0.2744058180470132,
      0.2744058180470132,
      -0.5108256237659907,
      1.6931471805599454,
      0.0,
      2.0,
    ],
    [
      'logistic(0, 1)',
      () => D.logistic(0, 1),
      1.1,
      0.18736987954752057,
      0.7502601055951177,
      -0.8472978603872037,
      2.0,
      0.0,
      3.289868133696453,
    ],
    [
      'pareto(3, 1)',
      () => D.pareto(3, 1),
      1.6,
      0.4577636718749999,
      0.755859375,
      1.1262478804436062,
      0.23472104466522348,
      1.5,
      0.75,
    ],
    [
      'inverseGamma(3, 2)',
      () => D.inverseGamma(3, 2),
      0.9,
      0.6606799159999748,
      0.6167612185838767,
      0.5531634821501717,
      0.695157020726022,
      1.0,
      1.0,
    ],
    [
      'fDistribution(5, 7)',
      () => D.fDistribution(5, 7),
      1.3,
      0.3374033624257489,
      0.6377672305521428,
      0.6050766341882883,
      1.252403133378504,
      1.4,
      2.6133333333333333,
    ],
    [
      'chi2(4)',
      () => D.chi2(4),
      3.1,
      0.16449217971572588,
      0.45876766674180514,
      2.1946984214069833,
      2.270362845461478,
      4.0,
      8.0,
    ],
    [
      'noncentralChi2(3, 2)',
      () => D.noncentralChi2(3, 2),
      4.0,
      0.11839464506363488,
      0.48388135838807433,
      2.573566309241688,
      2.5375984422967317,
      5.0,
      14.0,
    ],
    [
      'noncentralT(5, 1)',
      () => D.noncentralT(5, 1),
      1.4,
      0.3185723323716759,
      0.6185758260650649,
      0.49366404521216084,
      1.6661551066788047,
      1.1894160774351803,
      1.9186227280720427,
    ],
    [
      'skewNormal(0, 1, 3)',
      () => D.skewNormal(0, 1, 3),
      0.5,
      0.6570896552387413,
      0.38929437512197634,
      0.36349057919482386,
      0.952834969040814,
      0.7569397566060481,
      0.42704220486917677,
    ],
    [
      'vonMises(0, 2)',
      () => D.vonMises(0, 2),
      0.8,
      0.2812652262368242,
      0.8410725934916646,
      -0.40925923646296974,
      1.2663212919642852,
      0,
      0.3022253420359917,
    ],
    [
      'generalisedExtremeValue(0, 1, 0.2)',
      () => D.generalisedExtremeValue(0, 1, 0.2),
      0.7,
      0.27102661873865275,
      0.5948960084892825,
      -0.182223277958826,
      1.6926587978818395,
      0.8211485686265171,
      3.3440355622978766,
    ],
    [
      'generalisedPareto(0, 1, 0.2)',
      () => D.generalisedPareto(0, 1, 0.2),
      1.3,
      0.2499060265013569,
      0.6851184066082903,
      0.3697046189288967,
      1.2,
      1.25,
      2.6041666666666665,
    ],
    [
      'frechet(3, 0, 1)',
      () => D.frechet(3, 0, 1),
      1.4,
      0.5424235394008758,
      0.6945914229874681,
      0.93999982838481,
      0.6710085978672673,
      1.3541179394264005,
      0.8453031408313467,
    ],
    [
      'johnsonSU(0.5, 2, 0, 1)',
      () => D.johnsonSU(0.5, 2, 0, 1),
      -0.3,
      0.7610530175264203,
      0.4636087962227625,
      -0.5348917546544242,
      0.8643698704599944,
      -0.28624725601680295,
      0.34763296553761214,
    ],
    [
      'inverseGaussian(1, 2)',
      () => D.inverseGaussian(1, 2),
      0.8,
      0.750024342768835,
      0.4967548784810746,
      0.5703999337363391,
      0.7628464680731164,
      1.0,
      0.5,
    ],
    ['poisson(3)', () => D.poisson(3), 4, 0.16803135574154085, 0.8152632445237722, 2.0, 1.9314701981485691, 3.0, 3.0],
    [
      'binomial(20, 0.3)',
      () => D.binomial(20, 0.3),
      7,
      0.16426198521723667,
      0.7722717974181608,
      5.0,
      2.132538641661489,
      6.0,
      4.2,
    ],
    [
      'geometric(0.2)',
      () => D.geometric(0.2),
      3,
      0.12800000000000003,
      0.48800000000000004,
      2.0,
      2.5020121176909393,
      5.0,
      20.0,
    ],
    [
      'negativeBinomial(3, 0.4)',
      () => D.negativeBinomial(3, 0.4),
      4,
      0.12441600000000012,
      0.5800960000000001,
      2.0,
      2.4880372470519614,
      4.499999999999999,
      11.249999999999996,
    ],
    [
      'hypergeometric(50, 20, 10)',
      () => D.hypergeometric(50, 20, 10),
      4,
      0.28005860310537134,
      0.645026889882208,
      3.0,
      1.753823879258456,
      4.0,
      1.9591836734693877,
    ],
    [
      'betaBinomial(10, 2, 3)',
      () => D.betaBinomial(10, 2, 3),
      3,
      0.14385614385614395,
      0.4545454545454546,
      2.0,
      2.257723212496241,
      4.0,
      6.0,
    ],
  ]
  const pairs: readonly (readonly [string, D.Distribution, D.Distribution])[] = [
    ['normal', D.normal(0, 1), D.normal(1, 2)],
    ['logNormal', D.logNormal(0, 1), D.logNormal(0.5, 1.5)],
    ['halfNormal', D.halfNormal(1), D.halfNormal(2)],
    ['gamma', D.gamma(2, 1.5), D.gamma(3, 1)],
    ['inverseGamma', D.inverseGamma(3, 2), D.inverseGamma(2, 1)],
    ['chi2', D.chi2(4), D.chi2(6)],
    ['exponential', D.exponential(2), D.exponential(1)],
    ['beta', D.beta(2, 3), D.beta(1.5, 2)],
    ['laplace', D.laplace(0, 1), D.laplace(0.5, 2)],
    ['gumbel', D.gumbel(0, 1), D.gumbel(1, 2)],
    ['uniform (inside)', D.uniform(0, 1), D.uniform(-1, 2)],
    ['pareto', D.pareto(3, 1), D.pareto(2, 0.5)],
    ['weibull', D.weibull(1.5, 2), D.weibull(2, 1)],
    ['vonMises', D.vonMises(0, 2), D.vonMises(0.5, 1)],
    ['bernoulli', D.bernoulli({ probs: 0.3 }), D.bernoulli({ probs: 0.6 })],
    ['binomial', D.binomial(20, 0.3), D.binomial(20, 0.5)],
    ['poisson', D.poisson(3), D.poisson(5)],
    ['geometric', D.geometric(0.2), D.geometric(0.4)],
    ['categorical', D.categorical({ probs: [0.2, 0.5, 0.3] }), D.categorical({ probs: [0.4, 0.4, 0.2] })],
  ]
  it('matches scipy.stats and, for the Kullback-Leibler divergence, is zero for a law against itself, positive between two, and equal to the integral of log p - log q', () => {
    for (const [name, make, x, dens, cdf, ppf, entropy, mean, variance] of rows) {
      const d = make()
      const near = (what: string, got: number, want: number): void =>
        expect
          .soft(Math.abs(got - want), `${name}: ${what} ${got}, scipy ${want}`)
          .toBeLessThan(1e-8 * (1 + Math.abs(want)))
      near('density', item(d.prob(x)), dens)
      near('cdf', item(d.cdf(x)), cdf)
      near('ppf(0.3)', item(d.ppf(0.3)), ppf)
      near('entropy', item(d.entropy()), entropy)
      near('mean', item(d.mean()), mean)
      near('variance', item(d.variance()), variance)
    }
    for (const [name, p, q] of pairs) {
      expect(item(D.klDivergence(p, p)), `${name}: KL(p, p)`).toBeCloseTo(0, 10)
      const kl = item(D.klDivergence(p, q))
      expect(kl, `${name}: KL(p, q)`).toBeGreaterThan(0)
      let integral: number
      if (isDiscrete(p)) {
        const ks = atoms(p)
        const [lp, lq] = [flat(p.logProb(ks)), flat(q.logProb(ks))]
        integral = ks.reduce((a, _, i) => a + Math.exp(lp[i] ?? 0) * ((lp[i] ?? 0) - (lq[i] ?? 0)), 0)
      } else {
        const edges = probabilityEdges()
        integral = 0
        for (let i = 1; i < edges.length; i++)
          integral += panel(
            (us) => {
              const x = flat(p.ppf(us))
              const [lp, lq] = [flat(p.logProb(x)), flat(q.logProb(x))]
              return lp.map((v, j) => v - (lq[j] ?? 0))
            },
            edges[i - 1] ?? 0,
            edges[i] ?? 1,
          )
      }
      expect(Math.abs(integral - kl), `${name}: KL ${kl}, quadrature ${integral}`).toBeLessThan(1e-4 * (1 + kl))
    }
  })
})
