/**
 * One-dimensional and additive regression curves with known regression functions: sine, staircase, sinc, Donoho and
 * Johnstone's Doppler, heteroscedastic and known-quantile curves for quantile regression and conformal prediction,
 * piecewise-linear functions and additive (GAM) data with known components.
 */

import type { DatasetGenerator, Generated, Knobs, RegressionData, Truth } from '../data/index.ts'
import { xy } from '../data/index.ts'
import { type Distribution, gumbel, laplace, normal as normalDist, studentT } from '../dists/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import {
  abs,
  add,
  cos,
  div,
  exp,
  floor,
  full,
  item,
  less,
  log,
  maximum,
  minimum,
  mul,
  sin,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Value,
  where,
} from '../tensor/index.ts'
import { column, matrix, stamp } from './_util.ts'

/** The features of a 1-D curve: uniform on `[low, high]` (sorted when `sorted`), or an even grid. */
function inputs(k: Key, n: number, low: number, high: number, design: string): Float64Array {
  if (design === 'grid')
    return Float64Array.from({ length: n }, (_, i) => (n === 1 ? low : low + ((high - low) * i) / (n - 1)))
  const u = Float64Array.from(uniform(k, [n], { low, high }).data)
  return design === 'sorted' ? u.sort() : u
}

/** Standardised noise distributions with known quantiles. */
function noiseDistribution(name: string): Distribution {
  switch (name) {
    case 'laplace':
      return laplace(0, 1 / Math.SQRT2)
    case 'student-t':
      return studentT(3, 0, 1 / Math.sqrt(3))
    case 'gumbel':
      // Centred and scaled to unit variance: mean euler * beta, variance (pi beta)^2 / 6.
      return gumbel(-0.5772156649015329 * (Math.sqrt(6) / Math.PI), Math.sqrt(6) / Math.PI)
    default:
      return normalDist(0, 1)
  }
}

/**
 * Build a 1-D regression dataset `y = f(x) + s(x) e` with `e` from a standardised noise distribution, and the exact
 * truth: regression function, conditional quantiles `f(x) + s(x) F^-1(q)`, joint log density (uniform x) and the
 * Bayes error `E[s(x)^2]` (estimated on a fine grid).
 */
function curveData(
  name: string,
  description: string,
  knobs: Knobs,
  k: Key,
  f: (x: Value) => Value,
  scale: (x: Value) => Value,
  range: readonly [number, number],
  noiseName = 'gaussian',
  extra: Readonly<Record<string, unknown>> = {},
): Generated<RegressionData> {
  const n = knobs['n'] as number
  const xs = inputs(foldIn(k, 0), n, range[0], range[1], (knobs['design'] as string | undefined) ?? 'uniform')
  const x = matrix(xs, n, 1)
  const noise = noiseDistribution(noiseName)
  const e = noise.sample(foldIn(k, 1), [n])
  const y = add(f(x), mul(scale(x), e))
  const grid = matrix(
    Float64Array.from({ length: 2001 }, (_, i) => range[0] + ((range[1] - range[0]) * i) / 2000),
    2001,
    1,
  )
  const s2 = Array.from((square(scale(grid)) as { data: Float64Array }).data)
  const bayes = s2.reduce((a, b) => a + b, 0) / s2.length
  const width = range[1] - range[0]
  const truth: Truth = {
    regressionFunction: f,
    quantile: (v, q) => add(f(v), mul(scale(v), item(noise.ppf(q)))),
    logProb: (v, t) => {
      const px = full([v.shape[0] ?? 0], -Math.log(width))
      if (t === undefined) return px
      const s = scale(v)
      return add(px, sub(noise.logProb(div(sub(t, f(v)), s)), log(s)))
    },
    bayesError: Math.round(bayes * 1e10) / 1e10,
    relevantFeatures: [0],
    ...(noiseName === 'gaussian' && extra['constantScale'] !== undefined
      ? { noiseScale: extra['constantScale'] as number }
      : {}),
    extra: { noise: noiseName, ...extra },
  }
  return { data: xy(x, y, { ...stamp(name, description, knobs), featureNames: ['x'] }), truth }
}

const n1 = int(2, 100_000, { description: 'The number of rows.', default: 100, label: 'samples' })
const design = choice(['uniform', 'sorted', 'grid'], {
  description: 'Where x lies: uniform draws, the same sorted, or an even grid.',
  default: 'uniform',
  label: 'design of x',
})
const noiseKnob = real(0, 10, {
  description: 'The standard deviation of the Gaussian noise.',
  default: 0.1,
  label: 'noise standard deviation',
})
const constant =
  (s: number) =>
  (x: Value): Value =>
    full([x.shape[0] ?? 0], s)

/**
 * `f(x) = amplitude sin(2 pi frequency x)` on [0, 1], with Gaussian noise.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { sineCurve } from 'mdml-engine/datasets'
 * generate(sineCurve, { knobs: { noise: 0.5 } }).truth.bayesError // => 0.25
 * ```
 */
export const sineCurve: DatasetGenerator<RegressionData> = {
  name: 'sineCurve',
  kind: 'regression',
  description: 'A sine wave on [0, 1] with Gaussian noise.',
  knobs: dict({
    n: n1,
    noise: noiseKnob,
    frequency: real(0.1, 50, { label: 'frequency', description: 'Cycles over [0, 1].', default: 1 }),
    amplitude: real(0, 100, { label: 'amplitude', description: 'The amplitude of the sine wave.', default: 1 }),
    design,
  }),
  generate: (k, knobs) => {
    const fr = knobs['frequency'] as number
    const amp = knobs['amplitude'] as number
    const s = knobs['noise'] as number
    const f = (x: Value): Value => mul(sin(mul(column(x, 0), 2 * Math.PI * fr)), amp)
    return curveData('sineCurve', sineCurve.description, knobs, k, f, constant(s), [0, 1], 'gaussian', {
      constantScale: s,
    })
  },
}

/**
 * A staircase on [0, 1]: `f(x) = floor((steps + 1) x) / steps` (clipped), a function with jumps for testing
 * smoothers and trees.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { stepCurve } from 'mdml-engine/datasets'
 * generate(stepCurve, { knobs: { steps: 2 } }).truth.extra?.['jumps'] // => [0.3333333333333333, 0.6666666666666666]
 * ```
 */
export const stepCurve: DatasetGenerator<RegressionData> = {
  name: 'stepCurve',
  kind: 'regression',
  description: 'A staircase with equally spaced jumps on [0, 1], with Gaussian noise.',
  knobs: dict({
    n: n1,
    noise: noiseKnob,
    steps: int(1, 50, { description: 'The number of equally spaced jumps.', default: 1, label: 'jumps' }),
    design,
  }),
  generate: (k, knobs) => {
    const steps = knobs['steps'] as number
    const s = knobs['noise'] as number
    const f = (x: Value): Value => div(minimum(floor(mul(column(x, 0), steps + 1)), steps), steps)
    const jumps = Array.from({ length: steps }, (_, i) => (i + 1) / (steps + 1))
    return curveData('stepCurve', stepCurve.description, knobs, k, f, constant(s), [0, 1], 'gaussian', {
      constantScale: s,
      jumps,
    })
  },
}

/**
 * The sinc function `sin(pi x) / (pi x)` on [-5, 5], with Gaussian noise.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { sincCurve } from 'mdml-engine/datasets'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(generate(sincCurve).truth.regressionFunction?.(tensor([[0.5]])) ?? tensor([])) // => [0.6366197723675814]
 * ```
 */
export const sincCurve: DatasetGenerator<RegressionData> = {
  name: 'sincCurve',
  kind: 'regression',
  description: 'The sinc function on [-5, 5] with Gaussian noise.',
  knobs: dict({ n: n1, noise: noiseKnob, design }),
  generate: (k, knobs) => {
    const s = knobs['noise'] as number
    const f = (x: Value): Value => {
      const t = column(x, 0)
      const safe = where(less(abs(t), 1e-12), 1e-12, t)
      return div(sin(mul(safe, Math.PI)), mul(safe, Math.PI))
    }
    return curveData('sincCurve', sincCurve.description, knobs, k, f, constant(s), [-5, 5], 'gaussian', {
      constantScale: s,
    })
  },
}

/**
 * Donoho and Johnstone's Doppler function `sqrt(x (1 - x)) sin(2 pi (1 + e) / (x + e))`, `e = 0.05`, on [0, 1]: a
 * test of spatially adaptive smoothing.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { dopplerCurve } from 'mdml-engine/datasets'
 * generate(dopplerCurve, { knobs: { n: 64, design: 'grid' } }).data.x.shape // => [64, 1]
 * ```
 */
export const dopplerCurve: DatasetGenerator<RegressionData> = {
  name: 'dopplerCurve',
  kind: 'regression',
  description: 'The Doppler function of Donoho and Johnstone on [0, 1] with Gaussian noise.',
  knobs: dict({
    n: n1,
    noise: real(0, 10, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the Gaussian noise.',
      default: 0.05,
    }),
    design,
  }),
  generate: (k, knobs) => {
    const s = knobs['noise'] as number
    const eps = 0.05
    const f = (x: Value): Value => {
      const t = column(x, 0)
      return mul(sqrt(maximum(mul(t, sub(1, t)), 0)), sin(div(2 * Math.PI * (1 + eps), add(t, eps))))
    }
    return curveData('dopplerCurve', dopplerCurve.description, knobs, k, f, constant(s), [0, 1], 'gaussian', {
      constantScale: s,
    })
  },
}

/**
 * Heteroscedastic regression: `f(x) = sin(2 pi x)` with noise standard deviation `base + slope x` on [0, 1], so the
 * conditional quantiles fan out; the truth gives every quantile and `truth.extra.scale` the noise function.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { heteroscedastic } from 'mdml-engine/datasets'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(generate(heteroscedastic).truth.quantile?.(tensor([[0]]), 0.5) ?? tensor([])) // => [0]
 * ```
 */
export const heteroscedastic: DatasetGenerator<RegressionData> = {
  name: 'heteroscedastic',
  kind: 'regression',
  description: 'A sine wave whose noise standard deviation grows linearly with x.',
  knobs: dict({
    n: n1,
    base: real(0.001, 10, {
      description: 'The noise standard deviation at x = 0.',
      default: 0.1,
      label: 'noise at x = 0',
    }),
    slope: real(0, 10, {
      description: 'How much the noise standard deviation grows per unit of x.',
      default: 0.5,
      label: 'noise growth per unit x',
    }),
    design,
  }),
  generate: (k, knobs) => {
    const b = knobs['base'] as number
    const sl = knobs['slope'] as number
    const f = (x: Value): Value => sin(mul(column(x, 0), 2 * Math.PI))
    const scale = (x: Value): Value => add(mul(column(x, 0), sl), b)
    return curveData('heteroscedastic', heteroscedastic.description, knobs, k, f, scale, [0, 1], 'gaussian', {
      scale,
    })
  },
}

/**
 * Curves with known conditional quantiles for quantile regression and conformal prediction: `y = f(x) + s(x) e` with
 * `f(x) = x sin(x)`, `s(x) = 0.2 + 0.1 x` on [0, 10] (scikit-learn's quantile regression example), and `e` Gaussian,
 * Laplace, Student t (3 degrees of freedom) or Gumbel, each standardised to unit variance. `truth.quantile(x, q)` is
 * exact.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { quantileCurves } from 'mdml-engine/datasets'
 * generate(quantileCurves, { knobs: { noise: 'laplace' } }).truth.extra?.['noise'] // => 'laplace'
 * ```
 */
export const quantileCurves: DatasetGenerator<RegressionData> = {
  name: 'quantileCurves',
  kind: 'regression',
  description: 'y = x sin(x) + (0.2 + 0.1 x) e on [0, 10] with Gaussian, Laplace, Student t or Gumbel noise.',
  knobs: dict({
    n: n1,
    noise: choice(['gaussian', 'laplace', 'student-t', 'gumbel'], {
      label: 'noise distribution',
      description: 'The distribution of the noise e: Gaussian, Laplace, Student t (3) or Gumbel, all standardised.',
      default: 'gaussian',
    }),
    design,
  }),
  generate: (k, knobs) => {
    const f = (x: Value): Value => mul(column(x, 0), sin(column(x, 0)))
    const scale = (x: Value): Value => add(mul(column(x, 0), 0.1), 0.2)
    return curveData(
      'quantileCurves',
      quantileCurves.description,
      knobs,
      k,
      f,
      scale,
      [0, 10],
      knobs['noise'] as string,
    )
  },
}

/**
 * Piecewise-linear regression: a continuous function with `knots` breakpoints uniform on [0, 1] and slopes standard
 * normal, plus Gaussian noise; `truth.extra.knots` and `slopes` describe it.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { piecewiseLinear } from 'mdml-engine/datasets'
 * (generate(piecewiseLinear, { knobs: { knots: 3 } }).truth.extra?.['knots'] as number[]).length // => 3
 * ```
 */
export const piecewiseLinear: DatasetGenerator<RegressionData> = {
  name: 'piecewiseLinear',
  kind: 'regression',
  description: 'A continuous piecewise-linear function with random knots and slopes, plus Gaussian noise.',
  knobs: dict({
    n: n1,
    noise: noiseKnob,
    knots: int(0, 50, {
      label: 'knots',
      description: 'The number of interior knots where the slope changes.',
      default: 3,
    }),
    design,
  }),
  generate: (k, knobs) => {
    const s = knobs['noise'] as number
    const m = knobs['knots'] as number
    const knots = Array.from(uniform(foldIn(k, 10), [m]).data).sort((a, b) => a - b)
    const slopes = Array.from(normal(foldIn(k, 11), [m + 1]).data, (v) => 3 * v)
    // f(x) = slope_0 x + sum_j (slope_{j+1} - slope_j) max(x - knot_j, 0): continuous, slope_j between knots.
    const f = (x: Value): Value => {
      const t = column(x, 0)
      let out = mul(t, slopes[0] ?? 0)
      for (const [j, kn] of knots.entries())
        out = add(out, mul(maximum(sub(t, kn), 0), (slopes[j + 1] ?? 0) - (slopes[j] ?? 0)))
      return out
    }
    return curveData('piecewiseLinear', piecewiseLinear.description, knobs, k, f, constant(s), [0, 1], 'gaussian', {
      constantScale: s,
      knots,
      slopes,
    })
  },
}

/** The component functions of {@link additiveModel}, each centred on [0, 1]. */
const COMPONENTS: readonly { readonly name: string; readonly f: (t: Value) => Value }[] = [
  { name: '2 sin(2 pi x)', f: (t) => mul(sin(mul(t, 2 * Math.PI)), 2) },
  { name: 'exp(2x) - (e^2 - 1)/2', f: (t) => sub(exp(mul(t, 2)), (Math.exp(2) - 1) / 2) },
  { name: '4 (x - 1/2)^2 - 1/3', f: (t) => sub(mul(square(sub(t, 0.5)), 4), 1 / 3) },
  { name: '2 cos(4 pi x)', f: (t) => mul(cos(mul(t, 4 * Math.PI)), 2) },
  { name: '3 (x - 1/2)', f: (t) => mul(sub(t, 0.5), 3) },
]

/**
 * Additive (GAM) data: `y = sum_j f_j(x_j) + noise` with known smooth components (sine, exponential, quadratic,
 * cosine and linear shapes, centred on [0, 1], cycling for more features) and optionally pure-noise features. The
 * truth has the regression function and each component (`truth.extra.components`), the targets of a GAM fit.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { additiveModel } from 'mdml-engine/datasets'
 * generate(additiveModel, { knobs: { nFeatures: 4, nInformative: 3 } }).truth.relevantFeatures // => [0, 1, 2]
 * ```
 */
export const additiveModel: DatasetGenerator<RegressionData> = {
  name: 'additiveModel',
  kind: 'regression',
  description: 'y = sum of smooth univariate components of uniform features, plus Gaussian noise (a GAM).',
  knobs: dict({
    n: n1,
    nFeatures: int(1, 100, { label: 'features', description: 'The number of uniform features.', default: 4 }),
    nInformative: int(1, 100, {
      label: 'informative features',
      description: 'How many features have a non-zero component.',
      default: 4,
    }),
    noise: real(0, 10, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the Gaussian noise.',
      default: 0.5,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['nFeatures'] as number
    const inf = Math.min(d, knobs['nInformative'] as number)
    const s = knobs['noise'] as number
    const x = uniform(foldIn(k, 0), [n, d])
    const parts = Array.from(
      { length: inf },
      (_, j) => COMPONENTS[j % COMPONENTS.length] as (typeof COMPONENTS)[number],
    )
    const component =
      (j: number) =>
      (v: Value): Value =>
        (parts[j] as (typeof COMPONENTS)[number]).f(column(v, j))
    const f = (v: Value): Value =>
      sum(
        stack(
          parts.map((_, j) => component(j)(v)),
          1,
        ),
        1,
      )
    const y = add(f(x), mul(normal(foldIn(k, 1), [n]), s))
    const truth: Truth = {
      regressionFunction: f,
      noiseScale: s,
      bayesError: s * s,
      relevantFeatures: Array.from({ length: inf }, (_, j) => j),
      logProb: (v, t) => {
        if (t === undefined) return full([v.shape[0] ?? 0], 0)
        const r = div(sub(t, f(v)), s)
        return mul(add(square(r), Math.log(2 * Math.PI) + 2 * Math.log(s)), -0.5)
      },
      extra: { components: parts.map((p, j) => ({ feature: j, name: p.name, f: component(j) })) },
    }
    return { data: xy(x, y, stamp('additiveModel', additiveModel.description, knobs)), truth }
  },
}
