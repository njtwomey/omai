import type { DatasetGenerator, RegressionData, Truth } from '../../lib/data/index.ts'
import { xy } from '../../lib/data/index.ts'
import { foldIn, type Key, normal, uniform } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import {
  add,
  arctan,
  div,
  fromStorage,
  matmul,
  mul,
  sin,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  sub,
  type Tensor,
  tensor,
  type Value,
} from '../../lib/tensor/index.ts'

const column = (x: Value, j: number): Value => squeeze(sliceAxis(x, -1, j, j + 1), -1)

/** Columns of independent uniforms on the given ranges. */
function uniformColumns(k: Key, n: number, ranges: readonly (readonly [number, number])[]): Tensor {
  const cols = ranges.map(([low, high], j) => uniform(foldIn(k, j), [n], { low, high }))
  const data = new Float64Array(n * ranges.length)
  cols.forEach((c, j) => {
    for (let i = 0; i < n; i++) data[i * ranges.length + j] = c.data[i] ?? 0
  })
  return fromStorage(data, [n, ranges.length], 'float64')
}

function withNoise(k: Key, clean: Value, noise: number): Tensor {
  return add(clean, mul(normal(k, clean.shape), noise)) as Tensor
}

const knobs = (minFeatures: number) =>
  dict({
    n: int(2, 100_000, { default: 100, label: 'samples' }),
    nFeatures: int(minFeatures, 100, { default: Math.max(minFeatures, 10), label: 'features' }),
    noise: real(0, 10, { default: 0, label: 'noise standard deviation' }),
  })

/** E[y | x] for Friedman's first problem: 10 sin(pi x0 x1) + 20 (x2 - 1/2)^2 + 10 x3 + 5 x4. */
export function friedman1Function(x: Value): Value {
  const c = (j: number) => column(x, j)
  return add(
    add(add(mul(sin(mul(mul(c(0), c(1)), Math.PI)), 10), mul(square(sub(c(2), 0.5)), 20)), mul(c(3), 10)),
    mul(c(4), 5),
  )
}

/**
 * Friedman's first regression problem (sklearn.datasets.make_friedman1): uniform features on [0, 1], of which the
 * first five matter, with the regression function as truth.
 *
 * @concept datasets
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { friedman1 } from 'whitebox-engine/methods/datasets'
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
      regressionFunction: friedman1Function,
      noiseScale: noise,
      bayesError: noise ** 2,
      relevantFeatures: [0, 1, 2, 3, 4],
    }
    return {
      data: xy(x, y, {
        name: 'friedman1',
        description: friedman1.description,
        recipe: { generator: 'friedman1', knobs: knobValues },
      }),
      truth,
    }
  },
}

const FRIEDMAN23_RANGES: readonly (readonly [number, number])[] = [
  [0, 100],
  [40 * Math.PI, 560 * Math.PI],
  [0, 1],
  [1, 11],
]

/** E[y | x] for Friedman's second problem: sqrt(x0^2 + (x1 x2 - 1 / (x1 x3))^2). */
export function friedman2Function(x: Value): Value {
  const c = (j: number) => column(x, j)
  return sqrt(add(square(c(0)), square(sub(mul(c(1), c(2)), div(1, mul(c(1), c(3)))))))
}

/** E[y | x] for Friedman's third problem: arctan((x1 x2 - 1 / (x1 x3)) / x0). */
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
    knobs: dict({ n: int(2, 100_000, { default: 100, label: 'samples' }), noise: real(0, 100, { default: 0 }) }),
    generate: (k, knobValues) => {
      const n = knobValues['n'] as number
      const noise = knobValues['noise'] as number
      const x = uniformColumns(foldIn(k, 0), n, FRIEDMAN23_RANGES)
      const y = withNoise(foldIn(k, 1), f(x), noise)
      return {
        data: xy(x, y, { name, description, recipe: { generator: name, knobs: knobValues } }),
        truth: { regressionFunction: f, noiseScale: noise, bayesError: noise ** 2, relevantFeatures: [0, 1, 2, 3] },
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
 * import { friedman2 } from 'whitebox-engine/methods/datasets'
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
 * import { friedman3 } from 'whitebox-engine/methods/datasets'
 * generate(friedman3).data.featureNames // => ['x0', 'x1', 'x2', 'x3']
 * ```
 */
export const friedman3: DatasetGenerator<RegressionData> = friedman23(
  'friedman3',
  friedman3Function,
  'Friedman (1991) problem 3: the phase of an AC circuit, y = arctan((x1 x2 - 1/(x1 x3)) / x0) + noise.',
)

/**
 * A linear regression problem with Gaussian features (sklearn.datasets.make_regression, well-conditioned case):
 * `nInformative` features have coefficients uniform on [0, 100], the rest zero.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { makeRegression } from 'whitebox-engine/methods/datasets'
 * generate(makeRegression, { knobs: { nInformative: 3 } }).truth.relevantFeatures // => [0, 1, 2]
 * ```
 */
export const makeRegression: DatasetGenerator<RegressionData> = {
  name: 'makeRegression',
  kind: 'regression',
  description: 'A random linear model with Gaussian features, some informative, plus bias and Gaussian noise.',
  knobs: dict({
    n: int(2, 100_000, { default: 100 }),
    nFeatures: int(1, 1000, { default: 10 }),
    nInformative: int(0, 1000, { default: 5 }),
    bias: real(-100, 100, { default: 0 }),
    noise: real(0, 100, { default: 0 }),
  }),
  generate: (k, knobValues) => {
    const n = knobValues['n'] as number
    const d = knobValues['nFeatures'] as number
    const informative = Math.min(d, knobValues['nInformative'] as number)
    const bias = knobValues['bias'] as number
    const noise = knobValues['noise'] as number
    const x = normal(foldIn(k, 0), [n, d])
    const raw = uniform(foldIn(k, 1), [d], { low: 0, high: 100 }).data
    const coef = tensor(Array.from({ length: d }, (_, j) => (j < informative ? (raw[j] ?? 0) : 0)))
    const f = (v: Value): Value => add(matmul(v, coef), bias)
    const y = withNoise(foldIn(k, 2), f(x), noise)
    return {
      data: xy(x, y, {
        name: 'makeRegression',
        description: makeRegression.description,
        recipe: { generator: 'makeRegression', knobs: knobValues },
      }),
      truth: {
        regressionFunction: f,
        noiseScale: noise,
        bayesError: noise ** 2,
        relevantFeatures: Array.from({ length: informative }, (_, j) => j),
        extra: { coefficients: Array.from(coef.data), bias },
      },
    }
  },
}
