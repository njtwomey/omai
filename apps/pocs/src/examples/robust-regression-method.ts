/**
 * Robust regression as an example method: a line (or a polynomial) with a known truth and noise, a fraction of the rows
 * replaced by outliers (their size, and where they sit: anywhere, or as leverage points at the ends of the input range), and
 * a model whose LIKELIHOOD is the choice: Gaussian, Student-t (the degrees of freedom fixed or learned with a prior), Laplace,
 * or a two-component Gaussian mixture of inliers and outliers (`normalOutliers`). One inference method of the library with its own knobs and a budget runs it. The Gaussian likelihood is dragged
 * towards the outliers and the heavy-tailed ones are not, which is what the page shows: `robust/fit` (the fitted curve with its
 * band, the data with the outliers marked, the Gaussian conjugate fit to the same rows and the Gaussian fit to the inliers
 * alone, and how far each is from the truth), `robust/df` (the posterior of the degrees of freedom against its prior),
 * `robust/pairs`, `robust/draws` and `robust/check` (the method against the exact normal posterior when the likelihood is
 * Gaussian with the noise known, else against a long NUTS run).
 *
 * @index estimator Robust regression: outliers in a line, Gaussian, Student-t, Laplace or mixture likelihood, one inference method.
 */

import type { Posterior } from 'omai-engine/probprog'
import type { InferenceMethod } from 'omai-engine/probprog/inference'
import { defineModel } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { normal as drawNormal, uniform as drawUniform, foldIn, gammaVariates, key } from 'omai-engine/random'
import { choice, type DictSpace, dict, int, real } from 'omai-engine/space'
import {
  add,
  arange,
  asTensor,
  cast,
  div,
  exp,
  full,
  greater,
  item,
  less,
  logicalAnd,
  logicalNot,
  max,
  mul,
  nonzero,
  pad as padWith,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  take,
  tensor,
  toArray,
  toFlat,
  type Value,
  where,
  zeros,
} from 'omai-engine/tensor'
import { type Config, derived } from 'omai-engine/trace'
import {
  columnMoments,
  defineExample,
  type Example,
  type ExampleProblem,
  exampleNumber,
  exampleString,
} from './example-method.ts'
import { linearTruths } from './linear-regression-method.ts'
import {
  curveBands,
  exactNormalPosterior,
  monomials,
  noiseSdOf,
  polynomialAt,
  type RegressionReference,
  type RegressionShape,
  regressionCheck,
  regressionDiagnostics,
  regressionDraws,
  regressionPairs,
  regressionReference,
} from './regression-views.ts'

/** The coefficients of the true functions (those of `linearTruths` that are polynomials). */
const TRUTH: Readonly<Record<string, readonly number[]>> = {
  line: [1, 2],
  quadratic: [1, 2, -1.5],
  cubic: [0.5, 2, -1.5, 1.2],
}
const RANGE: Readonly<Record<string, readonly [number, number]>> = { centred: [-1, 1], offset: [1, 3] }
/** The likelihoods on offer. */
export const robustLikelihoods: readonly string[] = ['gaussian', 'studentT', 'laplace', 'mixture']
/** Where the outliers sit. */
export const outlierPlacements: readonly string[] = ['random', 'leverage']

/** The problem a robust regression configuration stands for. */
export interface RobustProblem extends ExampleProblem {
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** Which rows are outliers. */
  readonly outlier: readonly boolean[]
  readonly truth: readonly number[]
  readonly noiseSd: number
  readonly learned: boolean
  readonly degree: number
  readonly names: readonly string[]
  readonly extent: readonly [number, number]
  readonly priorScale: number
  readonly design: readonly (readonly number[])[]
  readonly likelihood: string
  /** Whether the degrees of freedom of a Student-t likelihood are learned. */
  readonly dfLearned: boolean
  readonly dfPriorMean: number
  /** The mixture likelihood's outlier probability and width (in inlier sds). */
  readonly contamination: number
  readonly outlierScale: number
}

/**
 * The problem a configuration stands for: `rows` inputs, the true function plus noise, a fraction of the rows made
 * outliers (displaced by a multiple of the noise sd, anywhere or as leverage points at the ends of the range, the first
 * `n` rows of a longer run being the same), and the model with the chosen likelihood.
 *
 * @param config - The knobs (`truth`, `inputs`, `rows`, `noise`, `outliers`, `outlierSize`, `placement`, `dataSeed`, `degree`, `priorScale`, `likelihood` and its own).
 * @returns The problem.
 * @example
 * ```ts
 * import { robustRegressionProblem } from './index.ts'
 * const p = robustRegressionProblem({ truth: 'line', inputs: 'centred', rows: 40, noise: 0.5, outliers: 0.2, outlierSize: 8, placement: 'random', dataSeed: 1, degree: 1, priorScale: 3, likelihood: 'studentT', dfModel: 'learned', df: 4, dfPriorMean: 10, contamination: 0.1, outlierScale: 10, noiseModel: 'learned', sigmaScale: 1 })
 * [p.x.length, p.outlier.some(Boolean)] // => [40, true]
 * ```
 */
export function robustRegressionProblem(config: Config): RobustProblem {
  const n = Math.max(2, Math.round(exampleNumber(config, 'rows')))
  const degree = Math.max(1, Math.round(exampleNumber(config, 'degree')))
  const range = RANGE[exampleString(config, 'inputs')] ?? [-1, 1]
  const generating = TRUTH[exampleString(config, 'truth')] ?? [1, 2]
  const noiseSd = exampleNumber(config, 'noise')
  const fraction = exampleNumber(config, 'outliers')
  const size = exampleNumber(config, 'outlierSize')
  const leverage = exampleString(config, 'placement') === 'leverage'
  const priorScale = exampleNumber(config, 'priorScale')
  const learned = exampleString(config, 'noiseModel') === 'learned'
  const sigmaScale = exampleNumber(config, 'sigmaScale')
  const likelihood = robustLikelihoods.includes(exampleString(config, 'likelihood'))
    ? exampleString(config, 'likelihood')
    : 'studentT'
  const dfLearned = exampleString(config, 'dfModel') !== 'fixed'
  const df = exampleNumber(config, 'df') || 4
  const dfPriorMean = Math.max(1.5, exampleNumber(config, 'dfPriorMean') || 10)
  const contamination = exampleNumber(config, 'contamination') || 0.1
  const outlierScale = exampleNumber(config, 'outlierScale') || 10
  const seed = foldIn(key(exampleNumber(config, 'dataSeed')), 0)
  const u = drawUniform(foldIn(seed, 0), [n])
  const e = drawNormal(foldIn(seed, 1), [n])
  const v = drawUniform(foldIn(seed, 2), [n])
  const side = drawUniform(foldIn(seed, 3), [n])
  const sign = drawUniform(foldIn(seed, 4), [n])
  const outlier = less(v, fraction)
  const span = range[1] - range[0]
  // A leverage point is in the outer tenth of the range, on one side or the other.
  const frac = mul(u, 0.1)
  const ends = where(less(side, 0.5), add(mul(frac, span), range[0]), sub(range[1], mul(frac, span)))
  const x = where(logicalAnd(leverage, outlier), ends, add(mul(u, span), range[0]))
  const centre = (range[0] + range[1]) / 2
  const clean = add(polynomialAt(generating, x), mul(e, noiseSd))
  // Leverage points are pushed against the trend (the slope is positive), the others either way.
  const direction = leverage ? where(greater(x, centre), -1, 1) : where(less(sign, 0.5), -1, 1)
  const y = where(outlier, add(clean, mul(mul(direction, size), noiseSd)), clean)
  const known = generating.length
  const truth = padWith(sliceAxis(tensor(generating), 0, 0, Math.min(known, degree + 1)), [
    [0, Math.max(0, degree + 1 - known)],
  ])
  const design = monomials(x, degree)
  // region robust
  const definition = defineModel('robust-regression', (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const X = m.data('X', { dims: [row, feature], role: 'covariate' })
    const yy = m.data('y', { dims: [row], role: 'response' })
    const w = m.latent('w', d.normal(0, priorScale), {
      dims: [feature],
      doc: 'the coefficient of each power of x',
    })
    const mean = X.matmul(w)
    // endregion
    if (likelihood === 'gaussian') {
      if (learned) {
        // region robust:gaussian-learned
        const sigma = m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the noise scale' })
        m.observe('y', d.normal(mean, sigma), yy)
        // endregion
        return
      }
      // region robust:gaussian-known
      m.observe('y', d.normal(mean, noiseSd), yy)
      // endregion
      return
    }
    if (likelihood === 'laplace') {
      if (learned) {
        // region robust:laplace-learned
        const sigma = m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the noise scale' })
        m.observe('y', d.laplace(mean, sigma.div(Math.SQRT2)), yy)
        // endregion
        return
      }
      // region robust:laplace-known
      m.observe('y', d.laplace(mean, noiseSd / Math.SQRT2), yy)
      // endregion
      return
    }
    if (likelihood === 'mixture') {
      if (learned) {
        // region robust:mixture-learned
        const sigma = m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the noise scale' })
        m.observe('y', d.normalOutliers(mean, sigma, contamination, outlierScale), yy)
        // endregion
        return
      }
      // region robust:mixture-known
      m.observe('y', d.normalOutliers(mean, noiseSd, contamination, outlierScale), yy)
      // endregion
      return
    }
    if (dfLearned) {
      if (learned) {
        // region robust:student-learned-df-learned
        const sigma = m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the noise scale' })
        const excess = m.latent('nuExcess', d.gamma(2, 2 / (dfPriorMean - 1)), {
          doc: 'the degrees of freedom above one',
        })
        const nu = m.derived('nu', excess.add(1), { doc: 'the degrees of freedom of the t likelihood' })
        m.observe('y', d.studentT(nu, mean, sigma), yy)
        // endregion
        return
      }
      // region robust:student-known-df-learned
      const excess = m.latent('nuExcess', d.gamma(2, 2 / (dfPriorMean - 1)), {
        doc: 'the degrees of freedom above one',
      })
      const nu = m.derived('nu', excess.add(1), { doc: 'the degrees of freedom of the t likelihood' })
      m.observe('y', d.studentT(nu, mean, noiseSd), yy)
      // endregion
      return
    }
    if (learned) {
      // region robust:student-learned-df-fixed
      const sigma = m.latent('sigma', d.halfNormal(sigmaScale), { doc: 'the noise scale' })
      m.observe('y', d.studentT(df, mean, sigma), yy)
      // endregion
      return
    }
    // region robust:student-known-df-fixed
    m.observe('y', d.studentT(df, mean, noiseSd), yy)
    // endregion
    // region robust:end
  })
  // endregion
  return {
    x: toFlat(x),
    y: toFlat(y),
    outlier: toFlat(outlier).map(Boolean),
    truth: toFlat(truth),
    noiseSd,
    learned,
    degree,
    names: ['intercept', 'slope', ...Array.from({ length: Math.max(0, degree - 1) }, (_, j) => `x^${j + 2}`)],
    extent: range as readonly [number, number],
    priorScale,
    design: toArray(design) as number[][],
    likelihood,
    dfLearned,
    dfPriorMean,
    contamination,
    outlierScale,
    definition,
    data: { X: design, y },
  }
}

/** The knobs of the data: the truth, the inputs, the rows, the noise, the outliers (how many, how big, where) and the seed. */
export const robustDataKnobs: DictSpace = dict({
  truth: choice(
    linearTruths.filter((t) => t in TRUTH),
    {
      default: 'line',
      label: 'true function',
      descriptions: {
        line: 'a line: intercept 1, slope 2',
        quadratic: 'a parabola: 1 + 2x - 1.5x^2',
        cubic: 'a cubic: 0.5 + 2x - 1.5x^2 + 1.2x^3',
      },
    },
  ),
  inputs: choice(['centred', 'offset'], {
    default: 'centred',
    label: 'inputs',
    descriptions: { centred: 'x in [-1, 1]', offset: 'x in [1, 3], away from zero' },
  }),
  rows: int(5, 400, { default: 40, log: true, label: 'rows' }),
  noise: real(0.02, 3, { default: 0.5, log: true, label: 'noise sd' }),
  outliers: real(0, 0.6, {
    default: 0.15,
    label: 'outlier fraction',
    description: 'the fraction of the rows that are outliers',
  }),
  outlierSize: real(1, 50, {
    default: 8,
    log: true,
    label: 'outlier size',
    description: 'how far an outlier is displaced, in noise standard deviations',
  }),
  placement: choice(outlierPlacements, {
    default: 'random',
    label: 'where the outliers sit',
    descriptions: {
      random: 'anywhere in the range, displaced up or down at random',
      leverage:
        'leverage points: in the outer tenth of the range at either end, displaced against the trend, where they pull a line the most',
    },
  }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the likelihood and its own knobs, the polynomial degree, the prior scale and the noise. */
export const robustModelKnobs: DictSpace = dict({
  likelihood: choice(robustLikelihoods, {
    default: 'studentT',
    label: 'likelihood',
    descriptions: {
      gaussian: 'normal noise: the squared error, which an outlier pulls on without limit',
      studentT: 'Student-t noise: heavy tails, so an outlier is explained as a tail event and pulls little',
      laplace: 'Laplace (double exponential) noise: the absolute error, a median-like fit',
      mixture:
        'a mixture of inliers (the noise scale) and outliers (a wider normal with a fixed probability): the outliers are modelled rather than tolerated',
    },
  }),
  dfModel: choice(['learned', 'fixed'], {
    default: 'learned',
    label: 'degrees of freedom',
    descriptions: {
      learned: 'the degrees of freedom have a prior (a gamma above one) and are inferred with the rest',
      fixed: 'the degrees of freedom are the number below',
    },
  }),
  df: real(1, 100, {
    default: 4,
    log: true,
    label: 'degrees of freedom (fixed)',
    description: 'the degrees of freedom of the t likelihood: 1 is Cauchy, large is Gaussian',
  }),
  dfPriorMean: real(2, 100, {
    default: 10,
    log: true,
    label: 'prior mean of the degrees of freedom',
    description: 'the prior on the degrees of freedom is one plus a gamma(2, .) with this mean',
  }),
  contamination: real(0.01, 0.5, {
    default: 0.1,
    label: 'outlier probability',
    description: 'the probability, in the model, that a row is an outlier (mixture likelihood)',
  }),
  outlierScale: real(2, 100, {
    default: 10,
    log: true,
    label: 'outlier width',
    description: 'the standard deviation of the outliers as a multiple of the inliers (mixture likelihood)',
  }),
  degree: int(1, 6, {
    default: 1,
    label: 'polynomial degree',
    description: 'the model is a polynomial of this degree',
  }),
  priorScale: real(0.1, 30, {
    default: 3,
    log: true,
    label: 'prior scale',
    description: 'standard deviation of the normal prior on each coefficient',
  }),
  noiseModel: choice(['learned', 'known'], {
    default: 'learned',
    label: 'noise scale',
    descriptions: {
      learned: 'the scale is unknown, with a half-normal prior (an outlier makes a Gaussian fit inflate it)',
      known:
        'the scale is the true noise sd of the inliers: with a Gaussian likelihood the posterior is a normal in closed form',
    },
  }),
  sigmaScale: real(0.1, 10, {
    default: 1,
    log: true,
    label: 'noise prior scale',
    description: 'scale of the half-normal prior on the noise scale (when it is learned)',
  }),
})

/** The knobs that make the problem. */
export const robustProblemKnobs: readonly string[] = [
  'truth',
  'inputs',
  'rows',
  'noise',
  'outliers',
  'outlierSize',
  'placement',
  'dataSeed',
  'likelihood',
  'dfModel',
  'df',
  'dfPriorMean',
  'contamination',
  'outlierScale',
  'degree',
  'priorScale',
  'noiseModel',
  'sigmaScale',
]

/** The knobs a configuration does not use (the ones of the likelihoods that are not chosen). */
export function robustHiddenKnobs(config: Config): readonly string[] {
  const likelihood = exampleString(config, 'likelihood')
  const t = likelihood === 'studentT'
  const learnedDf = exampleString(config, 'dfModel') !== 'fixed'
  return [
    ...(t ? [] : ['dfModel', 'df', 'dfPriorMean']),
    ...(t && learnedDf ? ['df'] : []),
    ...(t && !learnedDf ? ['dfPriorMean'] : []),
    ...(likelihood === 'mixture' ? [] : ['contamination', 'outlierScale']),
    ...(exampleString(config, 'noiseModel') === 'known' ? ['sigmaScale'] : []),
  ]
}

const GRID = 60
const FIT_DRAWS = 300
const LINES = 12
const PAIR_DRAWS = 500

/** How the posteriors of a problem are read: the coefficients, the noise scale and the degrees of freedom. */
function shapeOf(problem: RobustProblem): RegressionShape {
  return {
    names: problem.names,
    truth: problem.truth,
    scalars: [
      { site: 'sigma', name: 'sigma', truth: problem.noiseSd },
      { site: 'nu', name: 'nu', truth: null },
    ],
  }
}

/** The gamma(2, rate) density at `x`. */
const gamma2 = (x: Value, rate: number): Value => where(greater(x, 0), mul(mul(rate * rate, x), exp(mul(x, -rate))), 0)

/** The root mean square distance between two curves. */
const rms = (a: TensorLike, b: TensorLike): number => {
  const gap = sub(a, b)
  return item(sqrt(div(sum(square(gap)), Math.max(1, asTensor(a).shape[0] ?? 0))))
}

const robustExample: Example<RobustProblem> = defineExample<RobustProblem>({
  name: 'robust',
  description:
    'Robust regression: a line with outliers fitted with a Gaussian, Student-t, Laplace or mixture likelihood, one inference method checked against the exact or a reference posterior',
  modelName: 'robust',
  dataKnobs: robustDataKnobs,
  modelKnobs: robustModelKnobs,
  problemKnobs: robustProblemKnobs,
  problem: robustRegressionProblem,
  derived: ({ problem, config, kind, posterior: post, fit }) => {
    const n = problem.x.length
    const shape = shapeOf(problem)
    const [lo, hi] = problem.extent
    const pad = 0.08 * (hi - lo)
    const grid = toFlat(add(div(mul(arange(GRID), hi - lo + 2 * pad), GRID - 1), lo - pad))
    const truthCurve = toFlat(polynomialAt(problem.truth, grid))
    const exactApplies = problem.likelihood === 'gaussian' && !problem.learned
    const gaussianCurve = (rows: number[], seen: number): { coefficients: number[]; curve: number[] } => {
      const keep = cast(rows.length === 0 ? arange(seen) : tensor(rows), 'int32')
      const post0 = exactNormalPosterior(
        take(tensor(problem.design as number[][]), keep, 0),
        take(tensor(problem.y as number[]), keep, 0),
        problem.priorScale,
        problem.noiseSd,
      )
      return { coefficients: post0.mean, curve: toFlat(polynomialAt(post0.mean, grid)) }
    }
    return {
      'robust/fit': derived<unknown>({
        cost: 'dear',
        description:
          'the fitted curve after the first `seen` rows (args `{ seen }`): the method refitted on them, its mean curve with the 95% band of the curve and of a new observation under its likelihood, sampled lines, the data with the outliers marked, the true curve, the Gaussian conjugate fit to the same rows and to the inliers alone, and how far each curve is from the truth',
        compute: (state, args) => {
          const asked = Math.round(Number((args as { seen?: number } | null | undefined)?.seen ?? n))
          const seen = Math.max(2, Math.min(n, asked))
          let method: Posterior
          try {
            method = seen >= n ? post(state) : fit({ ...config, rows: seen })
          } catch {
            return null
          }
          const got = regressionDraws(method, shape, FIT_DRAWS)
          if (got === undefined) return null
          const { rows, p } = got
          const drawn = tensor(rows)
          // The degrees of freedom of each draw: learned, or the fixed one.
          const nus = got.names.includes('nu')
            ? take(drawn, got.names.indexOf('nu'), 1)
            : full([rows.length], exampleNumber(config, 'df') || 4)
          const nuAt = (r: readonly number[]): number =>
            got.names.includes('nu') ? (r[got.names.indexOf('nu')] ?? 4) : exampleNumber(config, 'df') || 4
          const z = toFlat(drawNormal(key(7), [rows.length]))
          const u = toFlat(drawUniform(key(8), [rows.length]))
          const g =
            problem.likelihood === 'studentT'
              ? gammaVariates(key(9), toFlat(div(nus, 2)))
              : toFlat(zeros([rows.length]))
          const bands = curveBands(
            rows,
            p,
            grid,
            (r, s, zz) => {
              const sd = noiseSdOf(got.names, r, problem.noiseSd)
              if (problem.likelihood === 'studentT') return (sd * zz) / Math.sqrt((2 * (g[s] ?? 1)) / nuAt(r))
              if (problem.likelihood === 'laplace') {
                const a = (u[s] ?? 0.5) - 0.5
                return -(sd / Math.SQRT2) * Math.sign(a) * Math.log(1 - 2 * Math.abs(a)) || 0
              }
              if (problem.likelihood === 'mixture')
                return sd * zz * ((u[s] ?? 0.5) < problem.contamination ? problem.outlierScale : 1)
              return sd * zz
            },
            LINES,
            z,
          )
          const { mean, sd } = columnMoments(rows, got.names.length)
          const everything = gaussianCurve([], seen)
          const inliers = gaussianCurve(
            toFlat(nonzero(logicalNot(tensor(problem.outlier.slice(0, seen) as boolean[])))[0] as Value),
            seen,
          )
          const fitted = polynomialAt(mean.slice(0, p), grid)
          return {
            seen,
            total: n,
            likelihood: problem.likelihood,
            x: grid,
            ...bands,
            truth: truthCurve,
            gaussian: everything.curve,
            inliers: inliers.curve,
            data: {
              x: problem.x.slice(0, seen),
              y: problem.y.slice(0, seen),
              outlier: problem.outlier.slice(0, seen),
            },
            errors: {
              method: rms(fitted, truthCurve),
              gaussian: rms(everything.curve, truthCurve),
              inliers: rms(inliers.curve, truthCurve),
            },
            coefficients: got.names.map((name, j) => ({
              name,
              truth: got.truth[j] ?? null,
              mean: mean[j] ?? Number.NaN,
              sd: sd[j] ?? Number.NaN,
              gaussian: j < p ? (everything.coefficients[j] ?? null) : null,
              inliers: j < p ? (inliers.coefficients[j] ?? null) : null,
            })),
            noiseSd: problem.noiseSd,
            outliers: problem.outlier.slice(0, seen).filter(Boolean).length,
          } as never
        },
      }),
      'robust/df': derived<unknown>({
        cost: 'dear',
        description:
          'the posterior draws of the degrees of freedom of a Student-t likelihood with its prior density on a grid; null when they are fixed or the likelihood is another',
        compute: (state) => {
          const got = regressionDraws(post(state), shape, 400)
          const j = got?.names.indexOf('nu') ?? -1
          if (got === undefined || j < 0) return null
          const draws = take(tensor(got.rows), j, 1)
          const top = Math.min(100, item(max(draws)) * 1.3)
          const rate = 2 / (problem.dfPriorMean - 1)
          const grid2 = add(div(mul(arange(80), top - 1), 79), 1)
          return { draws: toFlat(draws), grid: toFlat(grid2), prior: toFlat(gamma2(sub(grid2, 1), rate)) } as never
        },
      }),
      'robust/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of the intercept, the slope, the noise scale and the degrees of freedom (a posteriorPairs record)',
        compute: (state) => {
          const got = regressionDraws(post(state), shape, PAIR_DRAWS)
          if (got === undefined || got.rows.length < 3) return null
          const coordinates = [...got.names.slice(0, 2), ...got.names.slice(got.p)]
          return regressionPairs(kind, got, coordinates) as never
        },
      }),
      'robust/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler (a chainDiagnostics record); null for the others',
        compute: (state) => regressionDiagnostics(post(state), shape) as never,
      }),
      'robust/draws': derived<unknown>({
        cost: 'dear',
        description:
          'the draws of the coefficients, the noise scale and the degrees of freedom and the largest R-hat (the reference)',
        compute: (state) => (regressionReference(post(state), shape) ?? null) as never,
      }),
      'robust/check': derived<unknown>({
        cost: 'dear',
        description:
          "the method against the exact normal posterior (Gaussian likelihood, noise known) or against reference draws (args `{ reference }`, the reference's `robust/draws`)",
        compute: (state, args) => {
          const ref = (args as { reference?: RegressionReference | null } | null | undefined)?.reference
          const exact = exactApplies
            ? exactNormalPosterior(problem.design, problem.y, problem.priorScale, problem.noiseSd)
            : undefined
          return regressionCheck(kind, post(state), shape, exact, ref) as never
        },
      }),
    }
  },
})

/**
 * Robust regression as a method, run by name and knobs: see the file's summary. A trace `Method` whose tape is the chosen
 * inference method's own, with the derived paths `robust/fit`, `robust/df`, `robust/pairs`, `robust/draws` and `robust/check`.
 *
 * @concept inference-engines
 * @index estimator Robust regression method: outliers in the data, Gaussian, Student-t, Laplace or mixture likelihood, one inference method.
 * @example
 * ```ts
 * import { robustRegressionMethod, robustRegressionSteps } from './index.ts'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(robustRegressionMethod.config), rows: 20, likelihood: 'gaussian', noiseModel: 'known', inference: { method: 'gaussian' } }
 * const s = session(robustRegressionMethod, { config, steps: robustRegressionSteps(config), mode: 'eager' })
 * const check = s.probe(s.t, 'robust/check') as { maxMeanError: number }
 * check.maxMeanError < 0.01 // => true
 * ```
 */
export const robustRegressionMethod: InferenceMethod<unknown, Config, string> = robustExample.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const robustRegressionSteps: (config: Config) => number = robustExample.steps

export { robustExample }
