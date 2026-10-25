/**
 * The sampling distributions of test statistics: Snedecor's F and the noncentral chi-square, F and t distributions
 * (scipy.stats f, ncx2, ncf, nct), whose noncentral forms give the power of chi-square, ANOVA and t tests.
 */

import { maxOf, minOf } from '../../base/index.ts'
import {
  betainc,
  betaincinv,
  chdtr,
  chdtrc,
  digamma,
  fdtr,
  fdtrc,
  lbeta,
  lgamma,
  ndtr,
} from '../../numerics/special/index.ts'
import { foldIn, gammaVariates, type Key, normal as normalNoise, poisson as poissonDraw } from '../../random/index.ts'
import {
  abs,
  add,
  arange,
  div,
  exp,
  fromStorage,
  greater,
  less,
  log,
  log1p,
  logsumexp,
  max,
  maximum,
  mul,
  neg,
  primalOf,
  promoteFloat,
  reshape,
  sign,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericEntropy, numericMode, numericPpf } from '../_numeric.ts'
import { gammaSample, implicitReparam } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

/**
 * The indices j of a Poisson(m)-weighted series that carry all but about 1e-17 of the weight, on a new leading axis
 * (shape `[J, 1, ..., 1]` with `rank` trailing ones), with the log weights -m + j log m - log j!.
 */
function poissonTerms(m: Value, rank: number): { readonly j: Value; readonly logW: Value } {
  const values = toFlat(primalOf(m))
  const hi = maxOf(values)
  const lo = minOf(values)
  const j0 = Math.max(0, Math.floor(lo - 10 * Math.sqrt(lo) - 15))
  const j1 = Math.ceil(hi + 10 * Math.sqrt(hi) + 30)
  const j = reshape(arange(j0, j1 + 1), [j1 - j0 + 1, ...new Array<number>(rank).fill(1)])
  // j log m, with m^0 = 1 and 0^j = 0 for j > 0.
  const jLogM = where(
    greater(j, 0),
    where(greater(m, 0), mul(j, log(where(greater(m, 0), m, 1))), Number.NEGATIVE_INFINITY),
    0,
  )
  return { j, logW: sub(sub(jLogM, m), lgamma(add(j, 1))) }
}

/**
 * Snedecor's F distribution with d1 and d2 degrees of freedom (scipy.stats.f, torch.distributions.FisherSnedecor): the
 * ratio of independent chi-squares, each divided by its degrees of freedom.
 *
 * @param d1 - Numerator degrees of freedom (positive).
 * @param d2 - Denominator degrees of freedom (positive).
 * @returns The distribution; `rsample` is (V1 / d1) / (V2 / d2) for implicitly reparameterised chi-squares, so samples
 *   differentiate in both degrees of freedom.
 * @example
 * ```ts
 * import { fDistribution } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(fDistribution(4, 4).cdf(1)) - 0.5) < 1e-14 // => true
 * item(fDistribution(3, 6).mean()) // => 1.5
 * ```
 */
export function fDistribution(d1: TensorLike, d2: TensorLike): Distribution {
  const [n, m] = promoteFloat(d1, d2) as [Value, Value]
  validateParams('fDistribution', { d1: [n, 'positive'], d2: [m, 'positive'] })
  const batchShape = batchShapeOf(n, m)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const hn = mul(n, 0.5)
  const hm = mul(m, 0.5)
  const positive = (x: Value): Value => greater(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  return defineDistribution({
    name: 'fDistribution',
    params: { d1: n, d2: m },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => {
      const v = safe(x)
      const lp = sub(
        sub(add(mul(hn, log(div(n, m))), mul(sub(hn, 1), log(v))), mul(add(hn, hm), log1p(div(mul(n, v), m)))),
        lbeta(hn, hm),
      )
      return where(positive(x), lp, Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => where(positive(x), fdtr(n, m, safe(x)), 0),
    sf: (x) => where(positive(x), fdtrc(n, m, safe(x)), 1),
    ppf: (q) => {
      const v = betaincinv(hn, hm, q)
      return div(mul(m, v), mul(n, sub(1, v)))
    },
    isf: (q) => {
      const w = betaincinv(hm, hn, q)
      return div(mul(m, sub(1, w)), mul(n, w))
    },
    rsample: (key, shape) => {
      const v1 = gammaSample(foldIn(key, 0), b(hn), shape)
      const v2 = gammaSample(foldIn(key, 1), b(hm), shape)
      return div(div(v1, n), div(v2, m))
    },
    mean: () => b(where(greater(m, 2), div(m, sub(m, 2)), Number.POSITIVE_INFINITY)),
    variance: () =>
      b(
        where(
          greater(m, 4),
          div(mul(mul(square(m), 2), sub(add(n, m), 2)), mul(mul(n, square(sub(m, 2))), sub(m, 4))),
          where(greater(m, 2), Number.POSITIVE_INFINITY, Number.NaN),
        ),
      ),
    mode: () => b(where(greater(n, 2), mul(div(sub(n, 2), n), div(m, add(m, 2))), 0)),
    entropy: () => {
      const hs = add(hn, hm)
      return b(
        add(
          add(add(sub(log(m), log(n)), lbeta(hn, hm)), sub(mul(sub(1, hn), digamma(hn)), mul(add(1, hm), digamma(hm)))),
          mul(hs, digamma(hs)),
        ),
      )
    },
  })
}

/**
 * The noncentral chi-square distribution with `df` degrees of freedom and noncentrality `nc` (scipy.stats.ncx2): the
 * sum of df squared normals with means whose squares sum to nc. Computed as the Poisson(nc / 2) mixture of central
 * chi-squares with df + 2j degrees of freedom, summed over the terms that carry the weight.
 *
 * @param df - Degrees of freedom (positive).
 * @param nc - Noncentrality (non-negative).
 * @returns The distribution: density and CDF by the mixture series (differentiable in both parameters), quantiles,
 *   mode and entropy numerically, and `rsample` by implicit reparameterisation of the mixture sampler.
 * @example
 * ```ts
 * import { noncentralChi2 } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(noncentralChi2(3, 2).mean()) // => 5
 * Math.abs(item(noncentralChi2(2, 0).cdf(2)) - (1 - Math.exp(-1))) < 1e-15 // => true
 * ```
 */
export function noncentralChi2(df: TensorLike, nc: TensorLike): Distribution {
  const [k, lambda] = promoteFloat(df, nc) as [Value, Value]
  validateParams('noncentralChi2', { df: [k, 'positive'], nc: [lambda, 'nonNegative'] })
  const batchShape = batchShapeOf(k, lambda)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const half = mul(lambda, 0.5)
  const positive = (x: Value): Value => greater(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  const rankOf = (x: Value): number => batchShapeOf(x, k, lambda).length
  const chi2LogPdf = (nu: Value, x: Value): Value =>
    sub(sub(sub(mul(sub(mul(nu, 0.5), 1), log(x)), mul(x, 0.5)), mul(nu, 0.5 * Math.LN2)), lgamma(mul(nu, 0.5)))
  const mixture = (x: Value, term: (nu: Value) => Value): Value => {
    const { j, logW } = poissonTerms(half, rankOf(x))
    return sum(mul(exp(logW), term(add(k, mul(j, 2)))), 0)
  }
  const twin = concreteTwin([k, lambda], (a, c) => noncentralChi2(a as Tensor, c as Tensor))
  // Concrete draws of the Poisson mixture: chi-square with df + 2N degrees of freedom, N ~ Poisson(nc / 2).
  const draw = (key: Key, shape: readonly number[]): Value => {
    const c = twin()
    const kk = toFlat(broadcastBatch(c.params['df'] as Value, batchShape))
    const ll = toFlat(broadcastBatch(c.params['nc'] as Value, batchShape))
    const full = [...shape, ...batchShape]
    const size = full.reduce((p, d) => p * d, 1)
    const out = new Float64Array(size)
    for (let i = 0; i < size; i++) {
      const bi = i % Math.max(1, kk.length)
      const kn = foldIn(key, i)
      const count = poissonDraw(foldIn(kn, 0), (ll[bi] ?? 0) / 2, []).data[0] ?? 0
      out[i] = 2 * (gammaVariates(foldIn(kn, 1), [(kk[bi] ?? 1) / 2 + count])[0] ?? 0)
    }
    return fromStorage(out, full, 'float64')
  }
  const mean = add(k, lambda)
  const variance = mul(add(k, mul(lambda, 2)), 2)
  const self: Distribution = defineDistribution({
    name: 'noncentralChi2',
    params: { df: k, nc: lambda },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => {
      const v = safe(x)
      const { j, logW } = poissonTerms(half, rankOf(x))
      return where(positive(x), logsumexp(add(logW, chi2LogPdf(add(k, mul(j, 2)), v)), 0), Number.NEGATIVE_INFINITY)
    },
    cdf: (x) =>
      where(
        positive(x),
        mixture(safe(x), (nu) => chdtr(nu, safe(x))),
        0,
      ),
    sf: (x) =>
      where(
        positive(x),
        mixture(safe(x), (nu) => chdtrc(nu, safe(x))),
        1,
      ),
    ppf: (q) => numericPpf(self, twin, q),
    sample: (key, shape) => draw(key, shape),
    rsample: (key, shape) => implicitReparam(draw(key, shape), self.cdf, self.prob),
    mean: () => b(mean),
    variance: () => b(variance),
    mode: () => {
      const sd = sqrt(variance)
      return numericMode(self, twin, 1e-12, b(add(mean, mul(sd, 2))))
    },
    entropy: () => numericEntropy(self, { kind: 'halfLine', low: 0, scale: b(div(variance, mean)) }),
  })
  return self
}

/**
 * The noncentral F distribution with d1 and d2 degrees of freedom and noncentrality `nc` (scipy.stats.ncf): (V1 / d1)
 * / (V2 / d2) for a noncentral chi-square V1 and a central V2. With u = d1 x / (d1 x + d2), the CDF is the Poisson(nc /
 * 2) mixture of regularised incomplete beta functions I_u(d1 / 2 + j, d2 / 2) and the density the matching mixture of
 * beta densities; this is the distribution of the ANOVA F statistic under an alternative, used for power and effect-size
 * intervals.
 *
 * @param d1 - Numerator degrees of freedom (positive).
 * @param d2 - Denominator degrees of freedom (positive).
 * @param nc - Noncentrality (non-negative).
 * @returns The distribution: series density, CDF and survival function (differentiable), closed-form moments, numerical
 *   quantiles, mode and entropy, and `rsample` by implicit reparameterisation of the ratio sampler.
 * @example
 * ```ts
 * import { fDistribution, noncentralF } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(noncentralF(4, 10, 2).mean()) // => 1.875
 * Math.abs(item(noncentralF(4, 10, 0).cdf(1.5)) - item(fDistribution(4, 10).cdf(1.5))) < 1e-14 // => true
 * ```
 */
export function noncentralF(d1: TensorLike, d2: TensorLike, nc: TensorLike): Distribution {
  const [n, m, lambda] = promoteFloat(d1, d2, nc) as [Value, Value, Value]
  validateParams('noncentralF', { d1: [n, 'positive'], d2: [m, 'positive'], nc: [lambda, 'nonNegative'] })
  const batchShape = batchShapeOf(n, m, lambda)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const half = mul(lambda, 0.5)
  const hn = mul(n, 0.5)
  const hm = mul(m, 0.5)
  const positive = (x: Value): Value => greater(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  const rankOf = (x: Value): number => batchShapeOf(x, n, m, lambda).length
  // u = d1 x / (d1 x + d2) and its complement, from logs for accuracy at both ends.
  const parts = (x: Value): { readonly u: Value; readonly v: Value; readonly logU: Value; readonly logV: Value } => {
    const nx = mul(n, x)
    const den = add(nx, m)
    return { u: div(nx, den), v: div(m, den), logU: sub(log(nx), log(den)), logV: sub(log(m), log(den)) }
  }
  const mixture = (x: Value, term: (a: Value) => Value): Value => {
    const { j, logW } = poissonTerms(half, rankOf(x))
    return sum(mul(exp(logW), term(add(hn, j))), 0)
  }
  const twin = concreteTwin([n, m, lambda], (a, c, e) => noncentralF(a as Tensor, c as Tensor, e as Tensor))
  // Concrete draws: (chi2(d1 + 2N) / d1) / (chi2(d2) / d2), N ~ Poisson(nc / 2).
  const draw = (key: Key, shape: readonly number[]): Value => {
    const c = twin()
    const nn = toFlat(broadcastBatch(c.params['d1'] as Value, batchShape))
    const mm = toFlat(broadcastBatch(c.params['d2'] as Value, batchShape))
    const ll = toFlat(broadcastBatch(c.params['nc'] as Value, batchShape))
    const full = [...shape, ...batchShape]
    const size = full.reduce((p, d) => p * d, 1)
    const out = new Float64Array(size)
    for (let i = 0; i < size; i++) {
      const bi = i % Math.max(1, nn.length)
      const kn = foldIn(key, i)
      const count = poissonDraw(foldIn(kn, 0), (ll[bi] ?? 0) / 2, []).data[0] ?? 0
      const a = 2 * (gammaVariates(foldIn(kn, 1), [(nn[bi] ?? 1) / 2 + count])[0] ?? 0)
      const d = 2 * (gammaVariates(foldIn(kn, 2), [(mm[bi] ?? 1) / 2])[0] ?? 1)
      out[i] = a / (nn[bi] ?? 1) / (d / (mm[bi] ?? 1))
    }
    return fromStorage(out, full, 'float64')
  }
  const mean = where(greater(m, 2), div(mul(m, add(n, lambda)), mul(n, sub(m, 2))), Number.POSITIVE_INFINITY)
  const variance = where(
    greater(m, 4),
    div(
      mul(mul(square(div(m, n)), 2), add(square(add(n, lambda)), mul(add(n, mul(lambda, 2)), sub(m, 2)))),
      mul(square(sub(m, 2)), sub(m, 4)),
    ),
    where(greater(m, 2), Number.POSITIVE_INFINITY, Number.NaN),
  )
  const self: Distribution = defineDistribution({
    name: 'noncentralF',
    params: { d1: n, d2: m, nc: lambda },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => {
      const s = safe(x)
      const { logU, logV } = parts(s)
      const { j, logW } = poissonTerms(half, rankOf(x))
      const a = add(hn, j)
      const terms = add(logW, sub(add(mul(sub(a, 1), logU), mul(sub(hm, 1), logV)), lbeta(a, hm)))
      const jacobian = sub(log(mul(n, m)), mul(log(add(mul(n, s), m)), 2))
      return where(positive(x), add(logsumexp(terms, 0), jacobian), Number.NEGATIVE_INFINITY)
    },
    cdf: (x) => {
      const { u } = parts(safe(x))
      return where(
        positive(x),
        mixture(x, (a) => betainc(a, hm, u)),
        0,
      )
    },
    sf: (x) => {
      const { v } = parts(safe(x))
      return where(
        positive(x),
        mixture(x, (a) => betainc(hm, a, v)),
        1,
      )
    },
    ppf: (q) => numericPpf(self, twin, q),
    sample: (key, shape) => draw(key, shape),
    rsample: (key, shape) => implicitReparam(draw(key, shape), self.cdf, self.prob),
    mean: () => b(mean),
    variance: () => b(variance),
    mode: () => numericMode(self, twin, 1e-12, b(add(mul(div(add(n, lambda), n), 3), 2))),
    entropy: () => numericEntropy(self, { kind: 'halfLine', low: 0, scale: b(div(add(n, lambda), n)) }),
  })
  return self
}

/**
 * The noncentral t distribution with `df` degrees of freedom and noncentrality `nc` (scipy.stats.nct): (Z + nc) /
 * sqrt(V / df) for standard normal Z and chi-square V. The CDF is Lenth's series (AS 243) in incomplete beta functions
 * and the density its derivative, both summed over the Poisson(nc^2 / 2) terms that carry the weight, using
 * F(t; df, nc) = 1 - F(-t; df, -nc) for t < 0.
 *
 * @param df - Degrees of freedom (positive).
 * @param nc - Noncentrality (real).
 * @returns The distribution with series density and CDF (differentiable), closed-form moments, `rsample` as
 *   (eps + nc) / sqrt(V / df) with an implicitly reparameterised chi-square, and numerical quantiles, mode and entropy.
 *   Like the reference implementations, the series loses relative accuracy in the far tail opposite to nc.
 * @example
 * ```ts
 * import { noncentralT } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(noncentralT(5, 0).cdf(0)) - 0.5) < 1e-15 // => true
 * ```
 */
export function noncentralT(df: TensorLike, nc: TensorLike): Distribution {
  const [nu, delta] = promoteFloat(df, nc) as [Value, Value]
  validateParams('noncentralT', { df: [nu, 'positive'], nc: [delta, 'finite'] })
  const batchShape = batchShapeOf(nu, delta)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const halfNu = mul(nu, 0.5)
  const m = mul(square(delta), 0.5)
  const rankOf = (x: Value): number => batchShapeOf(x, nu, delta).length
  // F(s; nu, d) for s >= 0: Phi(-d) + (1/2) sum_j [p_j I_x(j + 1/2, nu/2) + q_j I_x(j + 1, nu/2)].
  const upperHalfCdf = (s: Value, d: Value): Value => {
    const { j, logW } = poissonTerms(m, rankOf(s))
    const x = div(square(s), add(square(s), nu))
    const p = exp(logW)
    // q_j = d e^{-m} m^j / (sqrt 2 Gamma(j + 3/2)) = sign(d) exp(logW + log j! + log|d| - log sqrt 2 - lgamma(j + 3/2)).
    const logQ = sub(
      add(add(logW, lgamma(add(j, 1))), log(where(greater(abs(d), 0), abs(d), 1))),
      add(lgamma(add(j, 1.5)), 0.5 * Math.LN2),
    )
    const q = where(greater(abs(d), 0), mul(sign(d), exp(logQ)), 0)
    const series = sum(add(mul(p, betainc(add(j, 0.5), halfNu, x)), mul(q, betainc(add(j, 1), halfNu, x))), 0)
    return add(ndtr(neg(d)), mul(series, 0.5))
  }
  const cdf = (t: Value): Value => {
    const neg0 = less(t, 0)
    const s = abs(t)
    const d = where(neg0, neg(delta), delta)
    const f = upperHalfCdf(s, d)
    return where(neg0, sub(1, f), f)
  }
  // The density: (1/2) sum_j [p_j g_{j+1/2} + q_j g_{j+1}] dx/ds with g_a = x^{a-1} (1-x)^{nu/2-1} / B(a, nu/2),
  // assembled in logs with x^{a-1} dx/ds = 2 nu |s|^{2a-1} / (s^2 + nu)^{a+1} so that it is finite at s = 0.
  const logPdf = (t: Value): Value => {
    const s = abs(t)
    const d = where(less(t, 0), neg(delta), delta)
    const { j, logW } = poissonTerms(m, rankOf(t))
    const r = add(square(s), nu)
    const logS = log(where(greater(s, 0), s, 1))
    const term = (a: Value): Value => {
      const power = sub(mul(a, 2), 1)
      const sPart = where(greater(s, 0), mul(power, logS), where(greater(power, 0), Number.NEGATIVE_INFINITY, 0))
      return sub(
        add(add(sPart, Math.log(2)), log(nu)),
        add(add(mul(add(a, 1), log(r)), lbeta(a, halfNu)), mul(sub(halfNu, 1), sub(log(r), log(nu)))),
      )
    }
    const logQ = sub(
      add(add(logW, lgamma(add(j, 1))), log(where(greater(abs(d), 0), abs(d), 1))),
      add(lgamma(add(j, 1.5)), 0.5 * Math.LN2),
    )
    const lp = add(logW, term(add(j, 0.5)))
    const lq = add(logQ, term(add(j, 1)))
    const scale = max(where(greater(abs(d), 0), maximum(lp, lq), lp), 0, true)
    const total = sum(
      add(exp(sub(lp, scale)), where(greater(abs(d), 0), mul(sign(d), exp(sub(lq, scale))), 0)),
      0,
      true,
    )
    return reshape(sub(add(log(total), scale), Math.LN2), batchShapeOf(t, nu, delta))
  }
  const ratio = exp(sub(lgamma(mul(sub(nu, 1), 0.5)), lgamma(halfNu)))
  const mean = mul(mul(delta, sqrt(halfNu)), ratio)
  const meanValue = where(greater(nu, 1), mean, Number.NaN)
  const variance = where(
    greater(nu, 2),
    sub(div(mul(nu, add(square(delta), 1)), sub(nu, 2)), square(mean)),
    where(greater(nu, 1), Number.POSITIVE_INFINITY, Number.NaN),
  )
  const twin = concreteTwin([nu, delta], (a, c) => noncentralT(a as Tensor, c as Tensor))
  const spread = (): Value => where(greater(nu, 2), sqrt(variance), add(abs(delta), 3))
  const centre = (): Value => where(greater(nu, 1), mean, delta)
  const self: Distribution = defineDistribution({
    name: 'noncentralT',
    params: { df: nu, nc: delta },
    support: { kind: 'real' },
    batchShape,
    logProb: logPdf,
    cdf,
    sf: (t) => sub(1, cdf(t)),
    ppf: (q) => numericPpf(self, twin, q),
    rsample: (key, shape) => {
      const eps = normalNoise(foldIn(key, 0), [...shape, ...batchShape])
      const v = mul(gammaSample(foldIn(key, 1), b(halfNu), shape), 2)
      return div(add(eps, delta), sqrt(div(v, nu)))
    },
    mean: () => b(meanValue),
    variance: () => b(variance),
    mode: () => {
      const c = centre()
      const w = spread()
      return numericMode(self, twin, b(sub(c, mul(w, 2))), b(add(c, mul(w, 2))))
    },
    entropy: () => numericEntropy(self, { kind: 'real', center: b(centre()), scale: b(spread()) }),
  })
  return self
}
