/**
 * The studentised range distribution (scipy.stats.studentized_range): the range of k standard normals divided by an
 * independent chi-based estimate of their scale, the reference distribution of Tukey's HSD and Nemenyi post-hoc
 * comparisons, computed by numerical quadrature.
 */

import { MdmlError } from '../../base/index.ts'
import { doubleExponentialRule } from '../../numerics/quadrature/index.ts'
import { lgamma, ndtr, normalLogPdf } from '../../numerics/special/index.ts'
import { foldIn, gammaVariates, normal as normalNoise } from '../../random/index.ts'
import {
  add,
  div,
  exp,
  fromStorage,
  greater,
  isfinite,
  isinf,
  log,
  maximum,
  minimum,
  mul,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  sqrt,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { concreteTwin, numericEntropy, numericMode, numericPpf } from '../_numeric.ts'
import { discreteSample } from '../_sampling.ts'
import { batchShapeOf, broadcastBatch, type Distribution, defineDistribution, validateParams } from '../distribution.ts'

// Double-exponential rules (step 1/16) for the inner integral over z and the outer integral over log s.
const INNER = doubleExponentialRule('real', 1 / 16)
const OUTER = doubleExponentialRule('real', 1 / 16)

/** Lift a 1-D rule to axis `axis` of a value of rank `rank` (all other axes of size 1). */
function lift(t: Tensor, axis: number, rank: number): Value {
  const shape = new Array<number>(rank).fill(1)
  shape[axis] = t.shape[0] ?? 0
  return reshape(t, shape)
}

/**
 * The studentised range distribution (scipy.stats.studentized_range(k, df)): the range of k independent standard
 * normals divided by an independent sqrt(V / df) with V chi-square on df degrees of freedom (Tukey's HSD statistic).
 * The CDF and density are double integrals, P(Q <= q) = int f_S(s) G(q s) ds with
 * G(w) = k int phi(z) (Phi(z) - Phi(z - w))^(k-1) dz, evaluated by a product of double-exponential rules (about 1e-9
 * relative accuracy); df = Infinity gives the range of normals itself.
 *
 * @param k - Number of groups (an integer of at least 2).
 * @param df - Degrees of freedom (positive, or Infinity).
 * @returns The distribution with quadrature density and CDF (differentiable in df and q), numerical quantiles, moments
 *   from E[Q^n] = E[R^n] E[S^-n], numerical mode and entropy, and non-reparameterised `sample`.
 * @example
 * ```ts
 * import { studentisedRange } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * item(studentisedRange(3, 10).cdf(0)) // => 0
 * ```
 */
export function studentisedRange(k: TensorLike, df: TensorLike): Distribution {
  const [groups, nu] = promoteFloat(k, df) as [Value, Value]
  validateParams('studentisedRange', { k: [groups, 'positiveInteger'], df: [nu, 'positive'] })
  if (toFlat(primalOf(groups)).some((v) => v < 2))
    throw new MdmlError('value', 'studentisedRange: k must be at least 2')
  const batchShape = batchShapeOf(groups, nu)
  const b = (v: Value): Value => broadcastBatch(v, batchShape)
  const infinite = isinf(nu)
  const finiteNu = where(infinite, 1, nu)
  const halfNu = mul(finiteNu, 0.5)
  const logFs = (s: Value): Value =>
    add(
      sub(add(Math.LN2, mul(halfNu, log(halfNu))), lgamma(halfNu)),
      sub(mul(sub(finiteNu, 1), log(s)), mul(mul(finiteNu, 0.5), mul(s, s))),
    )
  // G(w) (density = false) or its derivative g(w) (density = true) at w of rank r, integrating over z on a new
  // leading axis.
  const rangeIntegral = (w: Value, density: boolean): Value => {
    const r = w.shape.length
    const z = add(minimum(mul(w, 0.5), 1), lift(INNER.nodes, 0, r + 1))
    const weights = lift(INNER.weights, 0, r + 1)
    const span = maximum(sub(ndtr(z), ndtr(sub(z, w))), 0)
    const kk = groups
    const integrand = density
      ? mul(mul(mul(kk, sub(kk, 1)), exp(add(normalLogPdf(z), normalLogPdf(sub(z, w))))), pow(span, sub(kk, 2)))
      : mul(mul(kk, exp(normalLogPdf(z))), pow(span, sub(kk, 1)))
    return sum(mul(weights, where(isfinite(integrand), integrand, 0)), 0)
  }
  // E over S of h(s) = G(q s) or s g(q s), with s = exp(u) and u on a sinh-sinh rule centred at 0.
  const overS = (q: Value, density: boolean): Value => {
    const r = batchShapeOf(q, groups, nu).length
    const scale = div(1, sqrt(mul(finiteNu, 2)))
    const u = mul(lift(OUTER.nodes, 0, r + 1), scale)
    const s = exp(u)
    const weights = mul(lift(OUTER.weights, 0, r + 1), scale)
    const inner = rangeIntegral(mul(q, s), density)
    const term = mul(exp(add(logFs(s), u)), density ? mul(s, inner) : inner)
    const finiteValue = sum(mul(weights, where(isfinite(term), term, 0)), 0)
    return where(infinite, rangeIntegral(q, density), finiteValue)
  }
  const positive = (x: Value): Value => greater(x, 0)
  const safe = (x: Value): Value => where(positive(x), x, 1)
  // Moments of the range R of k normals: E[R] = int (1 - G) dw, E[R^2] = int 2 w (1 - G) dw.
  const rangeMoment = (n: 1 | 2): Value => {
    const rule = doubleExponentialRule('halfLine', 1 / 16)
    const r = batchShape.length
    const w = mul(lift(rule.nodes, 0, r + 1), 3)
    const weights = mul(lift(rule.weights, 0, r + 1), 3)
    const tail = sub(1, rangeIntegral(w, false))
    // P(R > 40) underflows for any practical k; beyond that the quadrature of 1 - G is only rounding.
    const term = where(greater(w, 40), 0, n === 1 ? tail : mul(mul(w, 2), tail))
    return sum(mul(weights, term), 0)
  }
  const inverseS = (n: 1 | 2): Value =>
    n === 1
      ? where(
          infinite,
          1,
          where(
            greater(nu, 1),
            mul(sqrt(halfNu), exp(sub(lgamma(mul(sub(finiteNu, 1), 0.5)), lgamma(halfNu)))),
            Number.POSITIVE_INFINITY,
          ),
        )
      : where(infinite, 1, where(greater(nu, 2), div(finiteNu, sub(finiteNu, 2)), Number.POSITIVE_INFINITY))
  const mean = (): Value => b(mul(rangeMoment(1), inverseS(1)))
  const variance = (): Value => {
    const m = mean()
    return b(sub(mul(rangeMoment(2), inverseS(2)), mul(m, m)))
  }
  const twin = concreteTwin([groups, nu], (a, c) => studentisedRange(a as Tensor, c as Tensor))
  const self: Distribution = defineDistribution({
    name: 'studentisedRange',
    params: { k: groups, df: nu },
    support: { kind: 'positive' },
    batchShape,
    logProb: (x) => where(positive(x), log(overS(safe(x), true)), Number.NEGATIVE_INFINITY),
    cdf: (x) => where(positive(x), overS(safe(x), false), 0),
    sf: (x) => where(positive(x), sub(1, overS(safe(x), false)), 1),
    ppf: (q) => numericPpf(self, twin, q),
    sample: (key, shape) => {
      const full = [...shape, ...batchShape]
      return discreteSample([groups, nu], full, 'float64', ([kt, nt]) => {
        const ks = toFlat(broadcastBatch(kt as Tensor, batchShape))
        const ns = toFlat(broadcastBatch(nt as Tensor, batchShape))
        const size = full.reduce((p, d) => p * d, 1)
        const out = new Float64Array(size)
        for (let i = 0; i < size; i++) {
          const bi = i % Math.max(1, ks.length)
          const kn = foldIn(key, i)
          const draws = normalNoise(foldIn(kn, 0), [ks[bi] ?? 2]).data
          let lo = Number.POSITIVE_INFINITY
          let hi = Number.NEGATIVE_INFINITY
          for (const d of draws) {
            lo = Math.min(lo, d)
            hi = Math.max(hi, d)
          }
          const n = ns[bi] ?? 1
          const v = Number.isFinite(n) ? (2 * (gammaVariates(foldIn(kn, 1), [n / 2])[0] ?? 0)) / n : 1
          out[i] = (hi - lo) / Math.sqrt(v)
        }
        return fromStorage(out, full, 'float64')
      })
    },
    mean,
    variance,
    mode: () => {
      const m = mean()
      return numericMode(self, twin, mul(m, 0.2), mul(m, 2))
    },
    entropy: () => {
      const m = mean()
      return numericEntropy(self, { kind: 'halfLine', low: 0, scale: mul(m, 0.5) })
    },
  })
  return self
}
