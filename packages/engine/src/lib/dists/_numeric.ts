// Numerical fallbacks shared by the families: quantiles by safeguarded Newton iteration, modes by search, and moments
// and entropies by double-exponential quadrature over a batch. Searches run on concrete values; the results are made
// differentiable in the parameters by one implicit-function step written with traced operations.
import { jvp } from '../autodiff/index.ts'
import { doubleExponentialRule, type QuadratureRule } from '../numerics/quadrature/index.ts'
import {
  add,
  broadcastShapes,
  broadcastTo,
  div,
  exp,
  type FloatDType,
  fromStorage,
  full,
  greater,
  isfinite,
  isTensor,
  less,
  logicalAnd,
  mul,
  neg,
  primalOf,
  reshape,
  square,
  stopGradient,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import type { Distribution } from './distribution.ts'

/** The concrete twin of a distribution: the same family at the concrete values of traced parameters. */
export function concreteTwin(
  params: readonly Value[],
  make: (...concrete: Tensor[]) => Distribution,
): () => Distribution {
  let cached: Distribution | undefined
  return () => {
    cached ??= make(...params.map((p) => (isTensor(p) ? p : primalOf(p))))
    return cached
  }
}

function flatAt(v: Value, shape: readonly number[]): number[] {
  const t = primalOf(v)
  return toFlat(t.shape.length === shape.length && t.shape.every((d, i) => d === shape[i]) ? t : broadcastTo(t, shape))
}

const at = (xs: readonly number[], shape: readonly number[]): Tensor =>
  fromStorage(Float64Array.from(xs), [...shape], 'float64')

/** Lower and upper ends of a univariate support as concrete arrays (infinite where unbounded). */
function supportEnds(dist: Distribution, shape: readonly number[]): [number[], number[]] {
  const n = shape.reduce((a, b) => a * b, 1)
  const fill = (v: number): number[] => new Array<number>(n).fill(v)
  const s = dist.support
  switch (s.kind) {
    case 'positive':
    case 'nonNegative':
      return [fill(0), fill(Number.POSITIVE_INFINITY)]
    case 'unitInterval':
      return [fill(0), fill(1)]
    case 'interval':
      return [flatAt(s.low, shape), flatAt(s.high, shape)]
    case 'greaterThan':
      return [flatAt(s.low, shape), fill(Number.POSITIVE_INFINITY)]
    case 'lessThan':
      return [fill(Number.NEGATIVE_INFINITY), flatAt(s.high, shape)]
    case 'circular':
      return [fill(-Math.PI), fill(Math.PI)]
    default:
      return [fill(Number.NEGATIVE_INFINITY), fill(Number.POSITIVE_INFINITY)]
  }
}

/**
 * Concrete quantiles of a continuous univariate distribution: a bracket from the support (expanded by doubling where
 * unbounded), then Newton steps on F(x) - q safeguarded by bisection.
 */
function solveQuantile(dist: Distribution, q: Value): { readonly x: number[]; readonly shape: number[] } {
  const shape = broadcastShapes(q.shape, dist.batchShape)
  const qs = flatAt(q, shape)
  const n = qs.length
  const [lo, hi] = supportEnds(dist, shape)
  const cdf = (xs: readonly number[]): number[] => flatAt(dist.cdf(at(xs, shape)), shape)
  const pdf = (xs: readonly number[]): number[] => flatAt(dist.prob(at(xs, shape)), shape)
  const a = lo.map((v, i) => (Number.isFinite(v) ? v : Math.min(-1, (hi[i] ?? 0) - 1)))
  const b = hi.map((v, i) => (Number.isFinite(v) ? v : Math.max(1, (a[i] ?? 0) + 1)))
  // Expand unbounded ends until they bracket q.
  for (let iter = 0; iter < 1100; iter++) {
    const fa = cdf(a)
    const fb = cdf(b)
    let moved = false
    for (let i = 0; i < n; i++) {
      const qi = qs[i] ?? Number.NaN
      if (!Number.isFinite(lo[i] ?? 0) && (fa[i] ?? 0) > qi && Number.isFinite(a[i] ?? 0)) {
        a[i] = (b[i] ?? 0) - 2 * ((b[i] ?? 0) - (a[i] ?? 0))
        moved = true
      }
      if (!Number.isFinite(hi[i] ?? 0) && (fb[i] ?? 1) < qi && Number.isFinite(b[i] ?? 0)) {
        b[i] = (a[i] ?? 0) + 2 * ((b[i] ?? 0) - (a[i] ?? 0))
        moved = true
      }
    }
    if (!moved) break
  }
  const x = a.map((v, i) => v + ((b[i] ?? 0) - v) / 2)
  const done = qs.map((v) => !(v > 0 && v < 1))
  for (let iter = 0; iter < 400 && done.some((d) => !d); iter++) {
    const f = cdf(x)
    const p = pdf(x)
    for (let i = 0; i < n; i++) {
      if (done[i]) continue
      const xi = x[i] ?? 0
      const qi = qs[i] ?? 0
      const fi = f[i] ?? Number.NaN
      if (fi === qi) {
        done[i] = true
        continue
      }
      if (fi < qi) a[i] = xi
      else b[i] = xi
      const ai = a[i] ?? 0
      const bi = b[i] ?? 0
      const newton = xi - (fi - qi) / (p[i] ?? 0)
      const next = Number.isFinite(newton) && newton > ai && newton < bi ? newton : ai + (bi - ai) / 2
      if (Math.abs(next - xi) <= 2e-16 * Math.abs(xi) || next <= ai || next >= bi || bi - ai <= 1e-300) {
        done[i] = true
        if (next > ai && next < bi) x[i] = next
        continue
      }
      x[i] = next
    }
  }
  const out = x.map((v, i) => {
    const qi = qs[i] ?? Number.NaN
    if (qi === 0) return lo[i] ?? Number.NEGATIVE_INFINITY
    if (qi === 1) return hi[i] ?? Number.POSITIVE_INFINITY
    return qi > 0 && qi < 1 ? v : Number.NaN
  })
  return { x: out, shape }
}

/**
 * Quantiles of a continuous univariate distribution by numerical inversion of its CDF, differentiable in the
 * parameters and in q through one implicit Newton step: x = x* - (F(x*) - q) / f(x*), whose derivative at the root is
 * -(dF/dtheta) / f (and 1 / f in q).
 *
 * @param dist - The distribution (parameters possibly traced).
 * @param concrete - Its concrete twin, for the search.
 * @param q - Probabilities.
 * @returns The quantiles.
 */
export function numericPpf(dist: Distribution, concrete: () => Distribution, q: Value): Value {
  const { x, shape } = solveQuantile(concrete(), q)
  const xs = at(x, shape)
  const f = stopGradient(dist.prob(xs))
  const ok = logicalAnd(greater(f, 0), isfinite(f))
  const step = div(sub(dist.cdf(xs), q), where(ok, f, 1))
  return where(ok, sub(xs, step), xs)
}

/**
 * Quantiles of a discrete distribution on the integers: the smallest k in [low, high] with F(k) >= q (scipy's
 * convention, so q = 0 gives low - 1). Piecewise constant, so not differentiated.
 *
 * @param concrete - The concrete distribution.
 * @param q - Probabilities.
 * @param low - Lowest value of the support.
 * @param high - Highest value (may be infinite), a number or a value broadcast against the batch.
 * @returns The quantiles.
 */
export function discretePpf(concrete: Distribution, q: Value, low: number, high: number | Value): Value {
  const shape = broadcastShapes(q.shape, concrete.batchShape)
  const qs = flatAt(q, shape)
  const n = qs.length
  const his = typeof high === 'number' ? new Array<number>(n).fill(high) : flatAt(high, shape)
  const cdf = (xs: readonly number[]): number[] => flatAt(concrete.cdf(at(xs, shape)), shape)
  const a = new Array<number>(n).fill(low - 1)
  const b = his.map((h) => (Number.isFinite(h) ? h : low + 1))
  for (let iter = 0; iter < 1100; iter++) {
    const fb = cdf(b)
    let moved = false
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(his[i] ?? 0) && (fb[i] ?? 1) < (qs[i] ?? 0) && (b[i] ?? 0) < 1e300) {
        a[i] = b[i] ?? 0
        b[i] = low + 2 * ((b[i] ?? 0) - low + 1)
        moved = true
      }
    }
    if (!moved) break
  }
  for (let iter = 0; iter < 1100; iter++) {
    const m = a.map((v, i) => Math.floor((v + (b[i] ?? 0)) / 2))
    if (m.every((v, i) => (b[i] ?? 0) - (a[i] ?? 0) <= 1 || v === a[i])) break
    const fm = cdf(m)
    for (let i = 0; i < n; i++) {
      if ((b[i] ?? 0) - (a[i] ?? 0) <= 1) continue
      if ((fm[i] ?? 0) >= (qs[i] ?? 0)) b[i] = m[i] ?? 0
      else a[i] = m[i] ?? 0
    }
  }
  const out = b.map((v, i) => {
    const qi = qs[i] ?? Number.NaN
    if (qi === 0) return low - 1
    if (qi === 1) return his[i] ?? Number.POSITIVE_INFINITY
    return qi > 0 && qi < 1 ? v : Number.NaN
  })
  return at(out, shape)
}

/** Where the probability mass of a univariate distribution lies, for quadrature. */
export type Bulk =
  | { readonly kind: 'finite'; readonly low: Value | number; readonly high: Value | number }
  | { readonly kind: 'halfLine'; readonly low: Value | number; readonly scale: Value | number }
  | { readonly kind: 'real'; readonly center: Value | number; readonly scale: Value | number }

const RULES = new Map<string, QuadratureRule>()
function ruleFor(kind: Bulk['kind']): QuadratureRule {
  let r = RULES.get(kind)
  if (r === undefined) {
    r = doubleExponentialRule(kind)
    RULES.set(kind, r)
  }
  return r
}

/**
 * The integral of f over a bulk, batched: f receives nodes of shape `[n, 1, ..., 1]` (broadcasting against the batch)
 * and returns values of shape `[n, ...batch]`, summed over the nodes. Non-finite integrand values count as zero (they
 * occur only where nodes round onto singular endpoints). Differentiable in everything f and the bulk close over.
 */
export function integrate(f: (x: Value) => Value, bulk: Bulk, batchRank: number): Value {
  const { nodes, weights } = ruleFor(bulk.kind)
  const n = nodes.shape[0] ?? 0
  const lift = (t: Tensor): Value => reshape(t, [n, ...new Array<number>(batchRank).fill(1)])
  const t = lift(nodes)
  const w = lift(weights)
  let x: Value
  let scale: Value | number
  if (bulk.kind === 'finite') {
    const half = mul(sub(bulk.high, bulk.low), 0.5)
    x = add(mul(add(bulk.high, bulk.low), 0.5), mul(half, t))
    scale = half
  } else if (bulk.kind === 'halfLine') {
    x = add(bulk.low, mul(bulk.scale, t))
    scale = bulk.scale
  } else {
    x = add(bulk.center, mul(bulk.scale, t))
    scale = bulk.scale
  }
  const fx = f(x)
  const ok = isfinite(fx)
  return mul(sum(mul(w, where(ok, fx, 0)), 0), scale)
}

/** E[g(X)] for a univariate continuous distribution by quadrature of g p over its bulk. */
export function expectation(dist: Distribution, g: (x: Value) => Value, bulk: Bulk): Value {
  return integrate(
    (x) => {
      const lp = dist.logProb(x)
      const ok = isfinite(lp)
      return where(ok, mul(g(x), exp(where(ok, lp, 0))), 0)
    },
    bulk,
    dist.batchShape.length,
  )
}

/** The mean and variance of a univariate continuous distribution by quadrature. */
export function numericMoments(dist: Distribution, bulk: Bulk): { readonly mean: Value; readonly variance: Value } {
  const m = expectation(dist, (x) => x, bulk)
  const v = expectation(dist, (x) => square(sub(x, m)), bulk)
  return { mean: m, variance: v }
}

/** The differential entropy -E[log p(X)] of a univariate continuous distribution by quadrature. */
export function numericEntropy(dist: Distribution, bulk: Bulk): Value {
  return neg(
    integrate(
      (x) => {
        const lp = dist.logProb(x)
        const ok = isfinite(lp)
        const safe = where(ok, lp, 0)
        return where(ok, mul(exp(safe), safe), 0)
      },
      bulk,
      dist.batchShape.length,
    ),
  )
}

/**
 * The mode of a univariate continuous distribution: a grid search over [low, high] (concrete), refined by golden
 * sections, then one Newton step on the score d log p / dx written with traced operations, so the result
 * differentiates in the parameters as the implicit function x*(theta) does.
 *
 * @param dist - The distribution.
 * @param concrete - Its concrete twin.
 * @param low - Lower end of the search (broadcast against the batch).
 * @param high - Upper end.
 * @returns The mode.
 */
export function numericMode(
  dist: Distribution,
  concrete: () => Distribution,
  low: Value | number,
  high: Value | number,
): Value {
  const c = concrete()
  const shape = [...c.batchShape]
  const size = shape.reduce((p, d) => p * d, 1)
  const lows = typeof low === 'number' ? new Array<number>(size).fill(low) : flatAt(low, shape)
  const highs = typeof high === 'number' ? new Array<number>(size).fill(high) : flatAt(high, shape)
  const logp = (xs: readonly number[]): number[] => flatAt(c.logProb(at(xs, shape)), shape)
  const grid = 200
  const best = new Array<number>(size).fill(Number.NEGATIVE_INFINITY)
  const arg = [...lows]
  for (let g = 0; g <= grid; g++) {
    const xs = lows.map((l, i) => l + (((highs[i] ?? 0) - l) * g) / grid)
    const v = logp(xs)
    for (let i = 0; i < size; i++)
      if ((v[i] ?? Number.NEGATIVE_INFINITY) > (best[i] ?? 0)) {
        best[i] = v[i] ?? 0
        arg[i] = xs[i] ?? 0
      }
  }
  const step = lows.map((l, i) => ((highs[i] ?? 0) - l) / grid)
  let a = arg.map((x, i) => Math.max(lows[i] ?? 0, x - (step[i] ?? 0)))
  let b = arg.map((x, i) => Math.min(highs[i] ?? 0, x + (step[i] ?? 0)))
  const ratio = (Math.sqrt(5) - 1) / 2
  for (let iter = 0; iter < 80; iter++) {
    const x1 = a.map((v, i) => (b[i] ?? 0) - ratio * ((b[i] ?? 0) - v))
    const x2 = a.map((v, i) => v + ratio * ((b[i] ?? 0) - v))
    const f1 = logp(x1)
    const f2 = logp(x2)
    a = a.map((v, i) => ((f1[i] ?? 0) < (f2[i] ?? 0) ? (x1[i] ?? 0) : v))
    b = b.map((v, i) => ((f1[i] ?? 0) < (f2[i] ?? 0) ? v : (x2[i] ?? 0)))
  }
  const ones = full(shape, 1, { dtype: 'float64' as FloatDType })
  const scoreOf =
    (d: Distribution) =>
    (x: Value): Value =>
      jvp((y: Value) => d.logProb(y), x, ones).tangent
  // Golden sections locate the maximum only to about sqrt(eps); Newton steps on the concrete score finish the job.
  let current = a.map((v, i) => v + ((b[i] ?? 0) - v) / 2)
  const concreteScore = scoreOf(c)
  for (let iter = 0; iter < 4; iter++) {
    const xt = at(current, shape)
    const sc = toFlat(concreteScore(xt))
    const cu = toFlat(jvp(concreteScore, xt, ones).tangent)
    current = current.map((x, i) => {
      const next = x - (sc[i] ?? 0) / (cu[i] ?? -1)
      return (cu[i] ?? 0) < 0 && Number.isFinite(next) && Math.abs(next - x) <= 2 * (step[i] ?? 0) ? next : x
    })
  }
  const xs = at(current, shape)
  const score = scoreOf(dist)
  const s = score(xs)
  const curvature = stopGradient(jvp(score, xs, ones).tangent)
  const ok = logicalAnd(less(curvature, 0), isfinite(curvature))
  return where(ok, sub(xs, div(s, where(ok, curvature, -1))), xs)
}

/**
 * A tail probability by quadrature of the density: P(X <= x) (`lower`) or P(X >= x) (`upper`), integrating
 * p(x -/+ s) over s in [0, inf) with the nodes scaled by `scale` (about the reciprocal decay rate of the density at x).
 * For tails where closed forms cancel catastrophically.
 */
export function tailProbability(dist: Distribution, x: Value, side: 'lower' | 'upper', scale: Value | number): Value {
  const rank = broadcastShapes(x.shape, dist.batchShape).length
  return integrate(
    (s) => {
      const lp = dist.logProb(side === 'lower' ? sub(x, s) : add(x, s))
      const ok = isfinite(lp)
      return where(ok, exp(where(ok, lp, 0)), 0)
    },
    { kind: 'halfLine', low: 0, scale },
    rank,
  )
}
