/**
 * Distributions built from other distributions: `independent` (batch axes treated as one event), `transformed` (the law
 * of f(X) under a bijector), `truncated` and `censored` versions of a base, the `empirical` distribution of weighted
 * samples, and the `deterministic` point mass, following torch.distributions, NumPyro and TFP.
 */

import { maxOf, minOf, WhiteboxError } from '../../base/index.ts'
import { type Key, uniform as uniformNoise } from '../../random/index.ts'
import {
  add,
  all,
  arange,
  argmax,
  asTensor,
  broadcastTo,
  cast,
  clip,
  cumsum,
  div,
  equal,
  exp,
  expandDims,
  fromStorage,
  greater,
  greaterEqual,
  isfinite,
  less,
  lessEqual,
  log,
  logicalAnd,
  minimum,
  mul,
  neg,
  primalOf,
  promoteFloat,
  reshape,
  square,
  stopGradient,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { discretePpf, integrate, quantileExpectation } from '../_numeric.ts'
import { discreteSample, uniformOpen } from '../_sampling.ts'
import { type Bijector, chain } from '../bijectors/bijector.ts'
import {
  type Distribution,
  type DistributionSpec,
  defineDistribution,
  type Support,
  sumRightmost,
  unavailable,
} from '../distribution.ts'

const isDiscrete = (s: Support): boolean => s.kind === 'integerInterval' || s.kind === 'boolean'

/** A distribution made by {@link independent}, remembering its base. */
export interface Independent extends Distribution {
  readonly base: Distribution
  readonly reinterpretedBatchRank: number
}

/**
 * Reinterpret the last `reinterpretedBatchRank` batch axes of a distribution as event axes
 * (torch.distributions.Independent): log densities and entropies sum over them, so a batch of independent coordinates
 * becomes one multivariate distribution.
 *
 * @param base - The distribution.
 * @param reinterpretedBatchRank - How many trailing batch axes become event axes (default 1).
 * @returns The distribution.
 * @throws WhiteboxError with code `shape` when the base has fewer batch axes.
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { independent } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * const d = independent(normal([0, 0], [1, 1]))
 * d.eventShape // => [2]
 * Math.abs(item(d.logProb([0, 0])) + Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function independent(base: Distribution, reinterpretedBatchRank = 1): Independent {
  const r = reinterpretedBatchRank
  const n = base.batchShape.length
  if (r < 0 || r > n)
    throw new WhiteboxError('shape', `independent: cannot reinterpret ${r} of ${n} batch axes as event axes`)
  const spec: DistributionSpec = {
    name: 'independent',
    params: base.params,
    support: base.support,
    batchShape: base.batchShape.slice(0, n - r),
    eventShape: [...base.batchShape.slice(n - r), ...base.eventShape],
    logProb: (x) => sumRightmost(base.logProb(x), r),
    sample: (key, shape) => base.sample(key, shape),
    mean: () => base.mean(),
    variance: () => base.variance(),
    mode: () => base.mode(),
    entropy: () => sumRightmost(base.entropy(), r),
  }
  const d = defineDistribution(base.hasRsample ? { ...spec, rsample: (key, shape) => base.rsample(key, shape) } : spec)
  return { ...d, base, reinterpretedBatchRank: r }
}

/** A distribution made by {@link transformed}, remembering its base and bijector. */
export interface Transformed extends Distribution {
  readonly base: Distribution
  readonly bijector: Bijector
}

/**
 * The distribution of f(X) for X ~ base and a bijector f (torch.distributions.TransformedDistribution): log p(y) =
 * log p_X(f^{-1}(y)) + log |det d f^{-1}/dy|. A bijector acting on more axes than the base's event reinterprets batch
 * axes as event axes. For a scalar event and a monotone f the CDF and quantiles carry over, and the mean, variance and
 * entropy (H(X) + E[log |f'(X)|]) are computed by quadrature over the base's quantile function.
 *
 * @param base - The base distribution.
 * @param bijector - The bijector, or bijectors to chain (first applied first).
 * @returns The distribution; `rsample` is f applied to the base's `rsample`.
 * @concept normalizing-flows
 * @example
 * ```ts
 * import { bijectors, normal } from 'whitebox-engine/dists'
 * import { transformed } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * // exp of a normal is the log-normal.
 * const d = transformed(normal(0, 1), bijectors.exp())
 * item(d.cdf(1)) // => 0.5
 * ```
 */
export function transformed(base: Distribution, bijector: Bijector | readonly Bijector[]): Transformed {
  const b = 'forward' in bijector ? bijector : chain(bijector)
  const extra = Math.max(0, b.inputRank - base.eventShape.length)
  const inner = extra > 0 ? independent(base, extra) : base
  const eventShape = b.forwardShape(inner.eventShape)
  const rankX = inner.eventShape.length
  const rankY = eventShape.length
  const scalar = rankY === 0 && b.sign !== 0
  const ldjX = (x: Value): Value => sumRightmost(b.forwardLogDetJacobian(x), rankX - b.inputRank)
  const spec: DistributionSpec = {
    name: 'transformed',
    params: inner.params,
    support: b.codomain,
    batchShape: inner.batchShape,
    eventShape,
    logProb: (y) => add(inner.logProb(b.inverse(y)), sumRightmost(b.inverseLogDetJacobian(y), rankY - b.outputRank)),
    sample: (key, shape) => b.forward(inner.sample(key, shape)),
    mean: () => (scalar ? quantileExpectation(self, (y) => y) : unavailable('transformed', 'mean (not a scalar map)')),
    variance: () => {
      if (!scalar) return unavailable('transformed', 'variance (not a scalar map)')
      const m = quantileExpectation(self, (y) => y)
      return quantileExpectation(self, (y) => square(sub(y, m)))
    },
    mode: () => unavailable('transformed', 'mode'),
    entropy: () =>
      scalar
        ? add(inner.entropy(), quantileExpectation(inner, ldjX))
        : unavailable('transformed', 'entropy (not a scalar map; estimate it by Monte Carlo)'),
  }
  const increasing = b.sign > 0
  const withCdf: DistributionSpec = scalar
    ? {
        ...spec,
        cdf: (y) => (increasing ? inner.cdf(b.inverse(y)) : inner.sf(b.inverse(y))),
        sf: (y) => (increasing ? inner.sf(b.inverse(y)) : inner.cdf(b.inverse(y))),
        ppf: (q) => b.forward(increasing ? inner.ppf(q) : inner.isf(q)),
        isf: (q) => b.forward(increasing ? inner.isf(q) : inner.ppf(q)),
      }
    : spec
  const self: Distribution = defineDistribution(
    inner.hasRsample ? { ...withCdf, rsample: (key, shape) => b.forward(inner.rsample(key, shape)) } : withCdf,
  )
  return { ...self, base: inner, bijector: b }
}

/** Bounds for {@link truncated} and {@link censored}; either may be omitted. */
export interface Bounds {
  readonly low?: TensorLike
  readonly high?: TensorLike
}

function boundsOf(bounds: Bounds): [Value, Value] {
  return promoteFloat(bounds.low ?? Number.NEGATIVE_INFINITY, bounds.high ?? Number.POSITIVE_INFINITY) as [Value, Value]
}

function intervalSupport(lo: Value, hi: Value, bounds: Bounds): Support {
  if (bounds.low !== undefined && bounds.high !== undefined) return { kind: 'interval', low: lo, high: hi }
  if (bounds.low !== undefined) return { kind: 'greaterThan', low: lo }
  if (bounds.high !== undefined) return { kind: 'lessThan', high: hi }
  return { kind: 'real' }
}

/**
 * A distribution truncated to [low, high] (NumPyro's TruncatedDistribution, generalised): the base renormalised on the
 * interval. Continuous bases use their CDF (or survival function in the upper tail, for accuracy) and quantile
 * function: `rsample` is the base quantile of uniform noise mapped into the interval, so it differentiates wherever the
 * base's quantile function does, and the moments and entropy are quadratures over the truncated quantile function.
 * Discrete bases truncate to the integers low .. high inclusive, with moments by summation.
 *
 * @param base - A univariate distribution with a CDF.
 * @param bounds - `{ low, high }` (either may be omitted).
 * @returns The truncated distribution. Its mode is the base's mode clipped to the interval (exact for unimodal bases).
 * @example
 * ```ts
 * import { exponential } from 'whitebox-engine/dists'
 * import { truncated } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * const d = truncated(exponential(1), { high: 1 })
 * Math.abs(item(d.cdf(1)) - 1) < 1e-15 // => true
 * Math.abs(item(d.prob(0)) - 1 / (1 - Math.exp(-1))) < 1e-14 // => true
 * ```
 */
export function truncated(base: Distribution, bounds: Bounds): Distribution {
  if (base.eventShape.length > 0) throw new WhiteboxError('shape', 'truncated: the base must be univariate')
  return isDiscrete(base.support) ? truncatedDiscrete(base, bounds) : truncatedContinuous(base, bounds)
}

function truncatedContinuous(base: Distribution, bounds: Bounds): Distribution {
  const [lo, hi] = boundsOf(bounds)
  const loFinite = isfinite(lo)
  const hiFinite = isfinite(hi)
  // Evaluate at finite stand-ins so that unused branches stay finite (and their gradients zero).
  const safeLo = where(loFinite, lo, 0)
  const safeHi = where(hiFinite, hi, 0)
  const fa = where(loFinite, base.cdf(safeLo), 0)
  const fb = where(hiFinite, base.cdf(safeHi), 1)
  const sa = where(loFinite, base.sf(safeLo), 1)
  const sb = where(hiFinite, base.sf(safeHi), 0)
  const upper = greater(fa, 0.5)
  const mass = where(upper, sub(sa, sb), sub(fb, fa))
  const logMass = log(mass)
  const inside = (x: Value): Value => logicalAnd(greaterEqual(x, lo), lessEqual(x, hi))
  const into = (x: Value): Value => clip(x, lo, hi)
  const ppf = (q: Value): Value => where(upper, base.isf(sub(sa, mul(q, mass))), base.ppf(add(fa, mul(q, mass))))
  const batchShape = [...base.batchShape]
  const self: Distribution = defineDistribution({
    name: 'truncated',
    params: { ...base.params, low: lo, high: hi },
    support: intervalSupport(lo, hi, bounds),
    batchShape,
    logProb: (x) => where(inside(x), sub(base.logProb(x), logMass), Number.NEGATIVE_INFINITY),
    cdf: (x) => {
      const v = into(x)
      return clip(where(upper, div(sub(sa, base.sf(v)), mass), div(sub(base.cdf(v), fa), mass)), 0, 1)
    },
    sf: (x) => {
      const v = into(x)
      return clip(where(upper, div(sub(base.sf(v), sb), mass), div(sub(fb, base.cdf(v)), mass)), 0, 1)
    },
    ppf,
    isf: (q) => ppf(sub(1, q)),
    rsample: (key, shape) => ppf(uniformOpen(key, [...shape, ...batchShape])),
    mean: () => quantileExpectation(self, (x) => x),
    variance: () => {
      const m = quantileExpectation(self, (x) => x)
      return quantileExpectation(self, (x) => square(sub(x, m)))
    },
    mode: () => clip(base.mode(), lo, hi),
    entropy: () => neg(quantileExpectation(self, (x) => self.logProb(x))),
  })
  return self
}

function truncatedDiscrete(base: Distribution, bounds: Bounds): Distribution {
  const baseLow = base.support.kind === 'integerInterval' ? base.support.low : 0
  const baseHigh = base.support.kind === 'integerInterval' ? base.support.high : 1
  const [lo, hi] = promoteFloat(bounds.low ?? baseLow, bounds.high ?? baseHigh) as [Value, Value]
  const hiFinite = isfinite(hi)
  const below = sub(lo, 1)
  const fa = base.cdf(below)
  const sa = base.sf(below)
  const fb = where(hiFinite, base.cdf(where(hiFinite, hi, 0)), 1)
  const sb = where(hiFinite, base.sf(where(hiFinite, hi, 0)), 0)
  const upper = greater(fa, 0.5)
  const mass = where(upper, sub(sa, sb), sub(fb, fa))
  const logMass = log(mass)
  const inside = (x: Value): Value => logicalAnd(greaterEqual(x, lo), lessEqual(x, hi))
  const lowest = minOf(toFlat(primalOf(lo)).map(Math.ceil))
  const highest = (): number => {
    const h = maxOf(toFlat(primalOf(hi)))
    if (Number.isFinite(h)) return h
    return maxOf(toFlat(primalOf(base.ppf(1 - 1e-15)))) + 10
  }
  const batchShape = [...base.batchShape]
  // The pmf over lowest .. highest on a new leading axis.
  const table = (): { readonly k: Value; readonly lp: Value } => {
    const top = highest()
    const k = reshape(arange(lowest, top + 1), [top - lowest + 1, ...batchShape.map(() => 1)])
    return { k, lp: self.logProb(k) }
  }
  const support: Support = {
    kind: 'integerInterval',
    low: lowest,
    high: maxOf(toFlat(primalOf(hi))),
  }
  const self: Distribution = defineDistribution({
    name: 'truncated',
    params: { ...base.params, low: lo, high: hi },
    support,
    batchShape,
    logProb: (x) => where(inside(x), sub(base.logProb(x), logMass), Number.NEGATIVE_INFINITY),
    cdf: (x) => {
      const v = minimum(x, where(hiFinite, hi, x))
      return where(
        less(x, lo),
        0,
        clip(where(upper, div(sub(sa, base.sf(v)), mass), div(sub(base.cdf(v), fa), mass)), 0, 1),
      )
    },
    ppf: (q) => discretePpf(self, q, lowest, hi),
    sample: (key: Key, shape) => {
      const u = uniformNoise(key, [...shape, ...batchShape])
      return discretePpf(self, u, lowest, hi)
    },
    mean: () => {
      const { k, lp } = table()
      return sum(mul(k, expSafe(lp)), 0)
    },
    variance: () => {
      const { k, lp } = table()
      const p = expSafe(lp)
      const m = sum(mul(k, p), 0)
      return sum(mul(square(sub(k, m)), p), 0)
    },
    mode: () => {
      const { lp } = table()
      return add(cast(argmax(lp, 0), 'float64'), lowest)
    },
    entropy: () => {
      const { lp } = table()
      const ok = isfinite(lp)
      const safe = where(ok, lp, 0)
      return neg(sum(where(ok, mul(exp(safe), safe), 0), 0))
    },
  })
  return self
}

const expSafe = (lp: Value): Value => {
  const ok = isfinite(lp)
  return where(ok, exp(where(ok, lp, 0)), 0)
}

/**
 * A censored distribution: values beyond [low, high] are recorded at the bound (a Tobit model). The log density is
 * log F(low) at x <= low, log S(high) at x >= high and the base log density between, which is the likelihood of
 * censored observations; draws are the base's draws clipped, so `rsample` keeps the base's pathwise gradients.
 *
 * @param base - A univariate continuous distribution with a CDF and survival function.
 * @param bounds - `{ low, high }` (either may be omitted).
 * @returns The (mixed) distribution; the mean and variance by quadrature, no entropy or mode.
 * @example
 * ```ts
 * import { normal } from 'whitebox-engine/dists'
 * import { censored } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * const d = censored(normal(0, 1), { low: 0 })
 * Math.abs(item(d.prob(0)) - 0.5) < 1e-15 // => true
 * ```
 */
export function censored(base: Distribution, bounds: Bounds): Distribution {
  if (base.eventShape.length > 0) throw new WhiteboxError('shape', 'censored: the base must be univariate')
  const [lo, hi] = boundsOf(bounds)
  const loFinite = isfinite(lo)
  const hiFinite = isfinite(hi)
  const safeLo = where(loFinite, lo, 0)
  const safeHi = where(hiFinite, hi, 0)
  const into = (x: Value): Value => clip(x, lo, hi)
  // E[g(clip(X))]: the masses at the bounds plus the integral of g over the base quantiles between them (split at the
  // kinks, so the quadrature stays smooth).
  const fa = where(loFinite, base.cdf(safeLo), 0)
  const fb = where(hiFinite, base.cdf(safeHi), 1)
  const censoredExpectation = (g: (x: Value) => Value): Value => {
    const atLow = where(loFinite, mul(g(safeLo), fa), 0)
    const atHigh = where(hiFinite, mul(g(safeHi), sub(1, fb)), 0)
    const middle = integrate((u) => g(base.ppf(u)), { kind: 'finite', low: fa, high: fb }, base.batchShape.length)
    return add(add(atLow, atHigh), middle)
  }
  const spec: DistributionSpec = {
    name: 'censored',
    params: { ...base.params, low: lo, high: hi },
    support: intervalSupport(lo, hi, bounds),
    batchShape: base.batchShape,
    logProb: (x) =>
      where(lessEqual(x, lo), base.logCdf(safeLo), where(greaterEqual(x, hi), base.logSf(safeHi), base.logProb(x))),
    cdf: (x) => where(less(x, lo), 0, where(greaterEqual(x, hi), 1, base.cdf(x))),
    sf: (x) => where(less(x, lo), 1, where(greaterEqual(x, hi), 0, base.sf(x))),
    ppf: (q) => into(base.ppf(q)),
    sample: (key, shape) => into(base.sample(key, shape)),
    mean: () => censoredExpectation((x) => x),
    variance: () => {
      const m = censoredExpectation((x) => x)
      return censoredExpectation((x) => square(sub(x, m)))
    },
    mode: () => unavailable('censored', 'mode (a mixed distribution)'),
    entropy: () => unavailable('censored', 'entropy (a mixed distribution has none)'),
  }
  return defineDistribution(
    base.hasRsample ? { ...spec, rsample: (key, shape) => into(base.rsample(key, shape)) } : spec,
  )
}

/**
 * The empirical distribution of weighted samples (TFP's Empirical): atoms at the samples (along the first axis, each
 * of the remaining shape as one event) with probabilities proportional to the weights. `rsample` resamples atoms by
 * index, so draws differentiate with respect to the atom values.
 *
 * @param samples - Atoms `[n, ...eventShape]`.
 * @param weights - Optional non-negative weights `[n]` (default uniform).
 * @returns The distribution: CDF and inverted-CDF quantiles for scalar events, moments, mode and entropy over the
 *   distinct atoms.
 * @example
 * ```ts
 * import { empirical } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * const d = empirical([1, 2, 2, 5])
 * item(d.prob(2)) // => 0.5
 * item(d.mean()) // => 2.5
 * item(d.ppf(0.5)) // => 2
 * ```
 */
export function empirical(samples: TensorLike, weights?: TensorLike): Distribution {
  const [s] = promoteFloat(samples) as [Value]
  if (s.shape.length === 0) throw new WhiteboxError('shape', 'empirical: samples need a leading axis')
  const n = s.shape[0] ?? 0
  const eventShape = s.shape.slice(1)
  const eventRank = eventShape.length
  const raw = weights === undefined ? (promoteFloat(new Array<number>(n).fill(1))[0] as Value) : asTensor(weights)
  const w = div(raw, sum(raw))
  const eventAxes = (x: Value): number[] => Array.from({ length: eventRank }, (_, i) => x.shape.length - 1 - i)
  // For each atom, whether x equals it: [..., n].
  const matches = (x: Value): Value => {
    const xe = expandDims(x, x.shape.length - eventRank)
    const eq = equal(xe, s)
    return eventRank === 0 ? eq : all(eq, eventAxes(eq))
  }
  const weightOf = (x: Value): Value => sum(where(matches(x), w, 0), -1)
  // Distinct atoms and their total weights (concrete).
  const distinct = (): { readonly index: number[]; readonly mass: number[] } => {
    const flat = toFlat(primalOf(s))
    const ws = toFlat(primalOf(w))
    const size = flat.length / Math.max(1, n)
    const seen = new Map<string, number>()
    const index: number[] = []
    const mass: number[] = []
    for (let i = 0; i < n; i++) {
      const id = flat.slice(i * size, (i + 1) * size).join(',')
      const at = seen.get(id)
      if (at === undefined) {
        seen.set(id, index.length)
        index.push(i)
        mass.push(ws[i] ?? 0)
      } else mass[at] = (mass[at] ?? 0) + (ws[i] ?? 0)
    }
    return { index, mass }
  }
  const expand = (v: Value): Value => reshape(v, [n, ...eventShape.map(() => 1)])
  const mean = (): Value => sum(mul(expand(w), s), 0)
  const scalarOnly = (what: string): never => unavailable('empirical', `${what} (only for scalar events)`)
  const sampleIndices = (key: Key, shape: readonly number[]): Value =>
    discreteSample([w], [...shape], 'int32', ([wt]) => {
      const cdf = toFlat(cumsum(wt as Tensor))
      const u = toFlat(uniformNoise(key, [...shape]))
      const out = new Int32Array(u.length)
      for (const [i, v] of u.entries()) {
        let k = cdf.findIndex((c) => c > v)
        if (k < 0) k = n - 1
        out[i] = k
      }
      return fromStorage(out, [...shape], 'int32')
    })
  return defineDistribution({
    name: 'empirical',
    params: { samples: s, weights: w },
    support: { kind: 'real' },
    batchShape: [],
    eventShape,
    logProb: (x) => log(weightOf(x)),
    cdf: (x) => (eventRank === 0 ? sum(where(lessEqual(s, expandDims(x, -1)), w, 0), -1) : scalarOnly('cdf')),
    ppf: (q) => {
      if (eventRank > 0) return scalarOnly('ppf')
      const flat = toFlat(primalOf(s))
      const ws = toFlat(primalOf(w))
      const order = flat.map((_, i) => i).sort((a, b) => (flat[a] ?? 0) - (flat[b] ?? 0))
      let acc = 0
      const cum = order.map((i) => {
        acc += ws[i] ?? 0
        return acc
      })
      const qs = toFlat(primalOf(q))
      const picks = qs.map((v) => {
        const k = cum.findIndex((c) => c >= v - 1e-15)
        return order[k < 0 ? n - 1 : k] ?? 0
      })
      return reshape(take(s, picks, 0), [...q.shape])
    },
    rsample: (key, shape) => take(s, stopGradient(sampleIndices(key, shape)), 0),
    mean,
    variance: () => sum(mul(expand(w), square(sub(s, mean()))), 0),
    mode: () => {
      const { index, mass } = distinct()
      let best = 0
      for (const [i, m] of mass.entries()) if (m > (mass[best] ?? 0)) best = i
      return reshape(take(s, [index[best] ?? 0], 0), eventShape)
    },
    entropy: () => {
      const { mass } = distinct()
      return promoteFloat(-mass.reduce((h, m) => h + (m > 0 ? m * Math.log(m) : 0), 0))[0] as Value
    },
  })
}

/**
 * The point mass at a value (NumPyro's Delta, TFP's Deterministic): log density 0 at the value and -Infinity
 * elsewhere. `rsample` returns the value itself, so gradients flow to it.
 *
 * @param value - The value (with any batch and event axes).
 * @param eventRank - How many trailing axes form one event (default 0).
 * @returns The distribution.
 * @example
 * ```ts
 * import { deterministic } from 'whitebox-engine/dists/compose'
 * import { item } from 'whitebox-engine/tensor'
 * item(deterministic(3).logProb(3)) // => 0
 * item(deterministic(3).cdf(2.5)) // => 0
 * ```
 */
export function deterministic(value: TensorLike, eventRank = 0): Distribution {
  const [v] = promoteFloat(value) as [Value]
  const batchShape = v.shape.slice(0, v.shape.length - eventRank)
  const eventShape = v.shape.slice(v.shape.length - eventRank)
  const at = (x: Value): Value => {
    const eq = equal(x, v)
    return eventRank === 0
      ? eq
      : all(
          eq,
          Array.from({ length: eventRank }, (_, i) => eq.shape.length - 1 - i),
        )
  }
  return defineDistribution({
    name: 'deterministic',
    params: { value: v },
    support: { kind: 'real' },
    batchShape,
    eventShape,
    logProb: (x) => where(at(x), 0, Number.NEGATIVE_INFINITY),
    cdf: (x) => (eventRank === 0 ? where(greaterEqual(x, v), 1, 0) : unavailable('deterministic', 'cdf')),
    ppf: (q) => add(v, mul(q, 0)),
    rsample: (_key, shape) => broadcastTo(v, [...shape, ...v.shape]),
    mean: () => v,
    variance: () => mul(v, 0),
    mode: () => v,
    entropy: () => sumRightmost(mul(v, 0), eventRank),
  })
}
