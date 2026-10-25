/**
 * Synthetic regression problems from scikit-learn: Friedman's three problems with their regression functions (the first
 * also as a generating program), `makeRegression` for sparse linear problems with optionally low-rank, ill-conditioned
 * designs, and the sparse uncorrelated problem.
 */

import type { DatasetGenerator, GeneratingProgram, RegressionData, Truth } from '../data/index.ts'
import { xy } from '../data/index.ts'
import { normal as normalDistribution, uniform as uniformDistribution } from '../dists/index.ts'
import { generatingProgram, plate, type RowsData, sample } from '../probprog/index.ts'
import { foldIn, type Key, normal, permutation, uniform } from '../random/index.ts'
import { bool, dict, int, real } from '../space/index.ts'
import {
  add,
  arctan,
  div,
  full,
  matmul,
  mul,
  reshape,
  sin,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  take as takeColumns,
  tensor,
  type Value,
} from '../tensor/index.ts'
import { column, stamp, uniformColumns, withNoise } from './_util.ts'
import { makeLowRankMatrix } from './matrices.ts'

const SQRT2PI = Math.log(2 * Math.PI)

/**
 * The joint log density `log p(x) + log N(y; f(x), noise^2)` (summed over target columns); without `y`, `log p(x)`.
 */
function jointLogProb(
  logPx: (v: Value) => Value,
  f: (v: Value) => Value,
  noise: number,
): (v: Value, y?: Value) => Value {
  return (v, y) => {
    const px = logPx(v)
    if (y === undefined) return px
    const r = div(sub(y, f(v)), noise)
    const lp = mul(add(square(r), SQRT2PI + 2 * Math.log(noise)), -0.5)
    return add(px, lp.shape.length === 2 ? sum(lp, 1) : lp)
  }
}

/** `log p(x)` of independent uniform features on boxes (ignoring the support boundary). */
function uniformLogDensity(ranges: readonly (readonly [number, number])[]): (v: Value) => Value {
  const c = -ranges.reduce((s, [lo, hi]) => s + Math.log(hi - lo), 0)
  return (v) => full([v.shape[0] ?? 0], c)
}

/** `log p(x)` of independent standard normal features. */
function standardNormalLogDensity(d: number): (v: Value) => Value {
  return (v) => mul(add(sum(square(v), 1), d * SQRT2PI), -0.5)
}

const knobs = (minFeatures: number) =>
  dict({
    n: int(2, 100_000, { default: 100, label: 'samples', description: 'The number of rows.' }),
    nFeatures: int(minFeatures, 100, {
      default: Math.max(minFeatures, 10),
      label: 'features',
      description: `The number of uniform features (the first ${minFeatures} matter).`,
    }),
    noise: real(0, 10, {
      default: 0,
      label: 'noise standard deviation',
      description: 'The standard deviation of the Gaussian noise on y.',
    }),
  })

/**
 * E[y | x] for Friedman's first problem: 10 sin(pi x0 x1) + 20 (x2 - 1/2)^2 + 10 x3 + 5 x4.
 *
 * @param x - Features `[n, d]` with `d >= 5`.
 * @returns The regression function at each row `[n]`.
 * @example
 * ```ts
 * import { friedman1Function } from 'whitebox-engine/datasets'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(friedman1Function(tensor([[0, 0, 0.5, 1, 1]]))) // => [15]
 * ```
 */
export function friedman1Function(x: Value): Value {
  const c = (j: number) => column(x, j)
  return add(
    add(add(mul(sin(mul(mul(c(0), c(1)), Math.PI)), 10), mul(square(sub(c(2), 0.5)), 20)), mul(c(3), 10)),
    mul(c(4), 5),
  )
}

/**
 * Friedman's problem 1 as a generating program: uniform features on the unit cube and y = f(x) + N(0, noise^2).
 *
 * @param d - The number of features.
 * @param noise - The noise standard deviation (positive).
 * @returns The program.
 * @example
 * ```ts
 * import { friedman1Program } from 'whitebox-engine/datasets'
 * friedman1Program(5, 1).target // => 'y'
 * ```
 */
export function friedman1Program(d: number, noise: number): GeneratingProgram {
  return generatingProgram((data: RowsData) => {
    plate(
      'rows',
      data.n,
      () => {
        const x = sample('x', uniformDistribution(full([d], 0), full([d], 1)), { obs: data.x })
        sample('y', normalDistribution(reshape(friedman1Function(x), [-1, 1]), noise), {
          obs: data.y === undefined ? undefined : reshape(data.y, [-1, 1]),
        })
      },
      { dim: -2 },
    )
  })
}

/**
 * Friedman's first regression problem (sklearn.datasets.make_friedman1): uniform features on [0, 1], of which the
 * first five matter, with the regression function, noise level and Gaussian log density as truth.
 *
 * @concept datasets
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { friedman1 } from 'whitebox-engine/datasets'
 * generate(friedman1, { knobs: { n: 20, noise: 1 } }).truth.bayesError // => 1
 * ```
 */
export const friedman1: DatasetGenerator<RegressionData> = {
  name: 'friedman1',
  kind: 'regression',
  description: 'Friedman (1991) problem 1: y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise.',
  knobs: knobs(5),
  generate: (k, knobValues) => {
    const n = knobValues['n'] as number
    const d = knobValues['nFeatures'] as number
    const noise = knobValues['noise'] as number
    const x = uniformColumns(
      foldIn(k, 0),
      n,
      Array.from({ length: d }, () => [0, 1] as const),
    )
    const y = withNoise(foldIn(k, 1), friedman1Function(x), noise)
    const truth: Truth = {
      ...(noise > 0 ? { program: friedman1Program(d, noise) } : {}),
      regressionFunction: friedman1Function,
      noiseScale: noise,
      bayesError: noise ** 2,
      relevantFeatures: [0, 1, 2, 3, 4],
      ...(noise > 0
        ? {
            logProb: jointLogProb(
              uniformLogDensity(Array.from({ length: d }, () => [0, 1] as const)),
              friedman1Function,
              noise,
            ),
          }
        : {}),
    }
    return { data: xy(x, y, stamp('friedman1', friedman1.description, knobValues)), truth }
  },
}

const FRIEDMAN23_RANGES: readonly (readonly [number, number])[] = [
  [0, 100],
  [40 * Math.PI, 560 * Math.PI],
  [0, 1],
  [1, 11],
]

/**
 * E[y | x] for Friedman's second problem: sqrt(x0^2 + (x1 x2 - 1 / (x1 x3))^2).
 *
 * @param x - Features `[n, 4]`.
 * @returns The regression function `[n]`.
 * @example
 * ```ts
 * import { friedman2Function } from 'whitebox-engine/datasets'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(friedman2Function(tensor([[3, 1, 5, 1]]))) // => [5]
 * ```
 */
export function friedman2Function(x: Value): Value {
  const c = (j: number) => column(x, j)
  return sqrt(add(square(c(0)), square(sub(mul(c(1), c(2)), div(1, mul(c(1), c(3)))))))
}

/**
 * E[y | x] for Friedman's third problem: arctan((x1 x2 - 1 / (x1 x3)) / x0).
 *
 * @param x - Features `[n, 4]`.
 * @returns The regression function `[n]`.
 * @example
 * ```ts
 * import { friedman3Function } from 'whitebox-engine/datasets'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(friedman3Function(tensor([[1, 1, 1, 1]]))) // => [0]
 * ```
 */
export function friedman3Function(x: Value): Value {
  const c = (j: number) => column(x, j)
  return arctan(div(sub(mul(c(1), c(2)), div(1, mul(c(1), c(3)))), c(0)))
}

function friedman23(
  name: 'friedman2' | 'friedman3',
  f: (x: Value) => Value,
  description: string,
): DatasetGenerator<RegressionData> {
  const generator: DatasetGenerator<RegressionData> = {
    name,
    kind: 'regression',
    description,
    knobs: dict({
      n: int(2, 100_000, { description: 'The number of rows.', default: 100, label: 'samples' }),
      noise: real(0, 100, {
        label: 'noise standard deviation',
        description: 'The standard deviation of the Gaussian noise on y.',
        default: 0,
      }),
    }),
    generate: (k, knobValues) => {
      const n = knobValues['n'] as number
      const noise = knobValues['noise'] as number
      const x = uniformColumns(foldIn(k, 0), n, FRIEDMAN23_RANGES)
      const y = withNoise(foldIn(k, 1), f(x), noise)
      return {
        data: xy(x, y, stamp(name, description, knobValues)),
        truth: {
          regressionFunction: f,
          noiseScale: noise,
          bayesError: noise ** 2,
          relevantFeatures: [0, 1, 2, 3],
          ...(noise > 0 ? { logProb: jointLogProb(uniformLogDensity(FRIEDMAN23_RANGES), f, noise) } : {}),
        },
      }
    },
  }
  return generator
}

/**
 * Friedman's second regression problem (sklearn.datasets.make_friedman2).
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { friedman2 } from 'whitebox-engine/datasets'
 * generate(friedman2).data.x.shape // => [100, 4]
 * ```
 */
export const friedman2: DatasetGenerator<RegressionData> = friedman23(
  'friedman2',
  friedman2Function,
  'Friedman (1991) problem 2: the impedance of an AC circuit, y = sqrt(x0^2 + (x1 x2 - 1/(x1 x3))^2) + noise.',
)

/**
 * Friedman's third regression problem (sklearn.datasets.make_friedman3).
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { friedman3 } from 'whitebox-engine/datasets'
 * generate(friedman3).data.featureNames // => ['x0', 'x1', 'x2', 'x3']
 * ```
 */
export const friedman3: DatasetGenerator<RegressionData> = friedman23(
  'friedman3',
  friedman3Function,
  'Friedman (1991) problem 3: the phase of an AC circuit, y = arctan((x1 x2 - 1/(x1 x3)) / x0) + noise.',
)

/** Draw the design of makeRegression: standard normal, or a low-rank matrix with a given effective rank. */
function design(k: Key, n: number, d: number, effectiveRank: number, tailStrength: number): Tensor {
  if (effectiveRank <= 0) return normal(k, [n, d])
  return makeLowRankMatrix(k, { nSamples: n, nFeatures: d, effectiveRank, tailStrength }).x
}

/**
 * A linear regression problem (sklearn.datasets.make_regression): standard normal features (or a low-rank design
 * with a given effective rank and tail strength, for ill-conditioned problems), `nInformative` features with
 * coefficients uniform on [0, 100], several targets, bias and Gaussian noise. Columns stay in order unless
 * `shuffle` is set (then the truth's coefficients and relevant features follow the permutation).
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { makeRegression } from 'whitebox-engine/datasets'
 * generate(makeRegression, { knobs: { nInformative: 3 } }).truth.relevantFeatures // => [0, 1, 2]
 * ```
 */
export const makeRegression: DatasetGenerator<RegressionData> = {
  name: 'makeRegression',
  kind: 'regression',
  description:
    'A random linear model with Gaussian or low-rank features, some informative, plus bias and Gaussian noise.',
  knobs: dict({
    n: int(2, 100_000, { label: 'samples', description: 'The number of rows.', default: 100 }),
    nFeatures: int(1, 1000, { label: 'features', description: 'The number of features.', default: 10 }),
    nInformative: int(0, 1000, {
      label: 'informative features',
      description: 'Features with a non-zero coefficient.',
      default: 5,
    }),
    nTargets: int(1, 100, { label: 'targets', description: 'The number of target columns.', default: 1 }),
    bias: real(-100, 100, { label: 'bias', description: 'The intercept of the linear model.', default: 0 }),
    noise: real(0, 100, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the Gaussian noise on y.',
      default: 0,
    }),
    effectiveRank: int(0, 1000, {
      description: 'The approximate rank of the features (0: independent, well-conditioned features).',
      default: 0,
      label: 'effective rank (0: well conditioned)',
    }),
    tailStrength: real(0, 1, {
      description: 'How heavy the tail of the singular values is (low-rank features).',
      default: 0.5,
      label: 'singular value tail strength',
    }),
    shuffle: bool({
      description: 'Shuffle the feature columns (and rows).',
      default: false,
      label: 'shuffle feature columns',
    }),
  }),
  generate: (k, knobValues) => {
    const n = knobValues['n'] as number
    const d = knobValues['nFeatures'] as number
    const informative = Math.min(d, knobValues['nInformative'] as number)
    const t = knobValues['nTargets'] as number
    const bias = knobValues['bias'] as number
    const noise = knobValues['noise'] as number
    const x0 = design(foldIn(k, 0), n, d, knobValues['effectiveRank'] as number, knobValues['tailStrength'] as number)
    const raw = uniform(foldIn(k, 1), [d, t], { low: 0, high: 100 }).data
    const coefRows = Array.from({ length: d }, (_, j) =>
      Array.from({ length: t }, (_, c) => (j < informative ? (raw[j * t + c] ?? 0) : 0)),
    )
    const order =
      knobValues['shuffle'] === true
        ? Array.from(permutation(foldIn(k, 3), d).data)
        : Array.from({ length: d }, (_, j) => j)
    const x = takeColumns(x0, tensor(order, { dtype: 'int32' }), 1) as Tensor
    const coef = order.map((j) => coefRows[j] ?? [])
    const coefT = tensor(t === 1 ? coef.map((r) => r[0] ?? 0) : coef)
    const f = (v: Value): Value => add(matmul(v, coefT), bias)
    const y = withNoise(foldIn(k, 2), f(x), noise)
    const relevant = order.flatMap((j, newJ) => (j < informative ? [newJ] : [])).sort((a, b) => a - b)
    return {
      data: xy(x, y, stamp('makeRegression', makeRegression.description, knobValues)),
      truth: {
        regressionFunction: f,
        noiseScale: noise,
        bayesError: noise ** 2,
        relevantFeatures: relevant,
        ...(noise > 0 && (knobValues['effectiveRank'] as number) <= 0
          ? { logProb: jointLogProb(standardNormalLogDensity(d), f, noise) }
          : {}),
        extra: { coefficients: t === 1 ? coef.map((r) => r[0] ?? 0) : coef, bias },
      },
    }
  },
}

/**
 * Sparse uncorrelated regression (sklearn.datasets.make_sparse_uncorrelated, Celeux et al.): standard normal
 * features of which four matter, `y = x0 + 2 x1 - 2 x2 - 1.5 x3 + N(0, 1)`.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { sparseUncorrelated } from 'whitebox-engine/datasets'
 * generate(sparseUncorrelated).truth.extra?.['coefficients'] // => [1, 2, -2, -1.5, 0, 0, 0, 0, 0, 0]
 * ```
 */
export const sparseUncorrelated: DatasetGenerator<RegressionData> = {
  name: 'sparseUncorrelated',
  kind: 'regression',
  description: 'Independent Gaussian features, four of them informative: y = x0 + 2 x1 - 2 x2 - 1.5 x3 + N(0, 1).',
  knobs: dict({
    n: int(2, 100_000, { description: 'The number of rows.', default: 100, label: 'samples' }),
    nFeatures: int(4, 1000, {
      description: 'The number of features (the first four matter).',
      default: 10,
      label: 'features',
    }),
  }),
  generate: (k, knobValues) => {
    const n = knobValues['n'] as number
    const d = knobValues['nFeatures'] as number
    const coef = Array.from({ length: d }, (_, j) => [1, 2, -2, -1.5][j] ?? 0)
    const c = tensor(coef)
    const f = (v: Value): Value => matmul(v, c)
    const x = normal(foldIn(k, 0), [n, d])
    const y = withNoise(foldIn(k, 1), f(x), 1)
    return {
      data: xy(x, y, stamp('sparseUncorrelated', sparseUncorrelated.description, knobValues)),
      truth: {
        regressionFunction: f,
        noiseScale: 1,
        bayesError: 1,
        relevantFeatures: [0, 1, 2, 3],
        logProb: jointLogProb(standardNormalLogDensity(d), f, 1),
        extra: { coefficients: coef },
      },
    }
  },
}
