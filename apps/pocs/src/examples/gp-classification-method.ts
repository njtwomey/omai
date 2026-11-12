/**
 * Gaussian process classification as an example method: a latent function with a Gaussian process prior and a Bernoulli
 * observation through a logistic or probit link. The data are binary labels of a 1-D or 2-D input with a known truth (a
 * latent function drawn from a Gaussian process, a sinusoid, a step, a disc, an XOR pattern, or the library's moons and
 * circles, with a fraction of the labels flipped), the model is a kernel (squared exponential, Matern 3/2 or 5/2, periodic)
 * with a length scale and a variance that are either fixed or inferred under log-normal hyperpriors, and one inference
 * method of the library with its own knobs and a budget infers the posterior. The latent function is written in its
 * whitened form `f = sqrt(variance) L(ell) z` with `z` standard normal, so a sampler never sees the funnel between the
 * length scale and the function; with the hyperparameters fixed `L` is data and the model is a probit or logistic
 * regression on `z`, so message passing (EP, VMP) and the Gaussian method apply, with them inferred the Gram matrix is a
 * custom function and only the sampling, variational and optimisation methods do. The derived paths read the run:
 * `gpc/fit` (the predictive probability and the latent function with their bands over the input, or over the plane),
 * `gpc/pairs` (the hyperparameters and the latent function at a few inputs), `gpc/scores`, `gpc/diagnostics`, `gpc/draws`
 * (what a check needs of a reference run) and `gpc/check` (the method against a long NUTS run, with the Laplace and EP fits
 * of the gp domain as extra rows when the hyperparameters are fixed).
 *
 * @index estimator Gaussian process classification: latent GP, Bernoulli link, one inference method, NUTS reference.
 */

import { type ClassificationData, generate } from 'omai-engine/data'
import { circles, moons } from 'omai-engine/datasets'
import { posteriorComparison, posteriorPairs } from 'omai-engine/exposition'
import { gpClassifier } from 'omai-engine/gp'
import {
  crossGram,
  gram,
  type Kernel,
  maternKernel,
  onDimensions,
  periodicKernel,
  productKernel,
  rbfKernel,
  scaleKernel,
} from 'omai-engine/kernels'
import { cholesky, solveTriangular } from 'omai-engine/numerics/linalg'
import { ndtr } from 'omai-engine/numerics/special'
import type { Posterior } from 'omai-engine/probprog'
import type { InferenceMethod } from 'omai-engine/probprog/inference'
import { type DataInput, defineModel, type ModelDefinition, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { type Approximation, agree, type MarginalOracle, weightedOracle } from 'omai-engine/probprog/validation'
import { normal as drawNormal, uniform as drawUniform, foldIn, key } from 'omai-engine/random'
import { choice, type DictSpace, dict, int, real } from 'omai-engine/space'
import { quantile } from 'omai-engine/stats'
import {
  abs,
  add,
  arange,
  argsort,
  cast,
  clip,
  concat,
  div,
  equal,
  exp,
  full,
  greater,
  item,
  less,
  log,
  matmul,
  max,
  maximum,
  min,
  mul,
  neg,
  repeat,
  reshape,
  sin,
  sliceAxis,
  sort,
  sqrt,
  square,
  stack,
  sub,
  sum,
  take,
  tensor,
  tile,
  toArray,
  toFlat,
  transpose,
  type Value,
  where,
  zeros,
} from 'omai-engine/tensor'
import { type Config, derived } from 'omai-engine/trace'
import {
  columnMoments,
  defineExample,
  drawRows,
  type Example,
  type ExampleProblem,
  exampleDiagnostics,
  exampleNumber,
  exampleString,
} from './example-method.ts'

/** The data sets on offer: three of one input and four of two. */
export const gpDatasets = ['gp-draw', 'sinusoid', 'step', 'circle', 'xor', 'moons', 'circles'] as const
/** A data set of {@link gpDatasets}. */
export type GpDataset = (typeof gpDatasets)[number]
/** The kernels on offer. */
export const gpKernels = ['rbf', 'matern52', 'matern32', 'periodic'] as const
/** The links on offer. */
export const gpLinks = ['logistic', 'probit'] as const

const JITTER = 1e-6
const GRID_1D = 100
const GRID_2D = 21
/** The generating length scale and variance of the `gp-draw` data set (an SE kernel). */
const DRAW_ELL = 1
const DRAW_VARIANCE = 4

/** A column of the inputs `[n, dim]`. */
const col = (x: Value, j: number): Value => take(x, j, 1)

/** The truth of a synthetic data set: the latent function `g` of each input (the probability before flipping is `sigmoid(g)`). */
const LATENT: Readonly<Record<string, (x: Value) => Value>> = {
  sinusoid: (x) => mul(3, sin(mul(1.6, col(x, 0)))),
  step: (x) => mul(7, sub(col(x, 0), 0.5)),
  circle: (x) => mul(3, sub(sub(1.6, square(col(x, 0))), square(col(x, 1)))),
  xor: (x) => mul(mul(3, col(x, 0)), col(x, 1)),
}

const sigmoid = (v: Value): Value => div(1, add(1, exp(neg(v))))

/** The problem a Gaussian process classification configuration stands for. */
export interface GpClassificationProblem extends ExampleProblem {
  /** The inputs `[n][dim]`, sorted by the first coordinate, and their labels (0 or 1). */
  readonly x: readonly (readonly number[])[]
  readonly y: readonly number[]
  readonly dim: 1 | 2
  /** The grid the predictive is drawn on (`[m][dim]`; for two inputs `size` by `size`, row-major in the second input). */
  readonly grid: readonly (readonly number[])[]
  readonly gridSize: number
  readonly extent: readonly (readonly [number, number])[]
  /** The probability of a label 1 at each grid point after the flips, when the data set has a known truth. */
  readonly truthProbability: readonly number[] | null
  /** The generating length scale, when the data are a draw from a Gaussian process with one. */
  readonly truthLengthScale: number | null
  readonly kernel: string
  readonly period: number
  readonly lengthScale: number
  readonly variance: number
  readonly inferred: boolean
  readonly link: 'logistic' | 'probit'
  readonly priorSpread: number
  /** The 1/0 labels as flipped, how many were. */
  readonly flipped: number
}

/** The kernel of a name at a length scale (a number, or a tensor in a differentiable program), on `dim` input columns. */
function kernelOf(name: string, lengthScale: Value | number, period: number, dim: number): Kernel {
  switch (name) {
    case 'matern52':
      return maternKernel({ nu: 2.5, lengthScale })
    case 'matern32':
      return maternKernel({ nu: 1.5, lengthScale })
    case 'periodic':
      return dim === 1
        ? periodicKernel({ lengthScale, period })
        : productKernel(
            ...Array.from({ length: dim }, (_, j) => onDimensions(periodicKernel({ lengthScale, period }), [j])),
          )
    default:
      return rbfKernel({ lengthScale })
  }
}

/** The grid of a data set: 1-D points over the range, or a square lattice (first input slowest... the second varies fastest). */
function gridOf(extent: readonly (readonly [number, number])[]): Value {
  const [a, b] = extent
  const axis = (lo: number, hi: number, size: number): Value => add(div(mul(arange(size), hi - lo), size - 1), lo)
  if (a === undefined) return zeros([0, 0])
  if (b === undefined) return reshape(axis(a[0], a[1], GRID_1D), [-1, 1])
  return stack([tile(axis(a[0], a[1], GRID_2D), [GRID_2D]), repeat(axis(b[0], b[1], GRID_2D), GRID_2D)], 1)
}

interface Raw {
  /** The inputs `[n, dim]` and the labels `[n]`. */
  readonly x: Value
  readonly y: Value
  readonly extent: readonly (readonly [number, number])[]
  readonly truth: ((rows: Value) => Value) | null
  readonly ell: number | null
}

const raw1d = (name: GpDataset, n: number, seed: ReturnType<typeof key>): Raw => {
  const x = reshape(sort(add(mul(drawUniform(foldIn(seed, 0), [n]), 6), -3)), [n, 1])
  const extent: [number, number][] = [[-3.3, 3.3]]
  if (name !== 'gp-draw') {
    const latent = LATENT[name] ?? ((v: Value) => col(v, 0))
    const unif = drawUniform(foldIn(seed, 2), [n])
    return {
      x,
      y: cast(less(unif, sigmoid(latent(x))), 'float64'),
      extent,
      truth: (rows) => sigmoid(latent(rows)),
      ell: null,
    }
  }
  // A draw from the squared-exponential prior at the inputs and at the grid, so the truth is known on the grid.
  const all = concat([x, gridOf(extent)], 0)
  const L = cholesky(gram(rbfKernel({ lengthScale: DRAW_ELL }), all, { jitter: JITTER }))
  const f = mul(matmul(L, drawNormal(foldIn(seed, 1), [all.shape[0] ?? 0])), Math.sqrt(DRAW_VARIANCE))
  const unif = drawUniform(foldIn(seed, 2), [n])
  const gridP = sigmoid(sliceAxis(f, 0, n, f.shape[0] ?? n))
  return {
    x,
    y: cast(less(unif, sigmoid(sliceAxis(f, 0, 0, n))), 'float64'),
    extent,
    truth: () => gridP,
    ell: DRAW_ELL,
  }
}

const raw2d = (name: GpDataset, n: number, seed: ReturnType<typeof key>): Raw => {
  const latent = LATENT[name]
  if (latent !== undefined) {
    const x = reshape(add(mul(drawUniform(foldIn(seed, 0), [2 * n]), 4), -2), [n, 2])
    const unif = drawUniform(foldIn(seed, 2), [n])
    return {
      x,
      y: cast(less(unif, sigmoid(latent(x))), 'float64'),
      extent: [
        [-2.3, 2.3],
        [-2.3, 2.3],
      ],
      truth: (rows) => sigmoid(latent(rows)),
      ell: null,
    }
  }
  // The library's two-moons and circles, the inputs standardised so one length scale means the same for both.
  const data: ClassificationData =
    name === 'moons'
      ? generate(moons, { knobs: { n, noise: 0.2 }, key: foldIn(seed, 0) }).data
      : generate(circles, { knobs: { n, noise: 0.1 }, key: foldIn(seed, 0) }).data
  const rows = reshape(data.x, [-1, 2])
  const count = rows.shape[0] ?? 0
  const mean = div(sum(rows, 0), count)
  const sd = sqrt(div(sum(square(sub(rows, mean)), 0), count))
  const x = div(sub(rows, mean), where(equal(sd, 0), 1, sd))
  const range = (j: number): [number, number] => {
    const lo = item(min(col(x, j)))
    const hi = item(max(col(x, j)))
    const pad = 0.15 * (hi - lo || 1)
    return [lo - pad, hi + pad]
  }
  return {
    x,
    y: cast(greater(data.y, 0.5), 'float64'),
    extent: [range(0), range(1)],
    truth: null,
    ell: null,
  }
}

/** Order of the inputs for display (by the first coordinate) and the labels flipped with a probability. */
function finish(r: Raw, flips: number, seed: ReturnType<typeof key>): { x: number[][]; y: number[]; flipped: number } {
  const order = argsort(col(r.x, 0))
  const flip = take(less(drawUniform(foldIn(seed, 3), [r.x.shape[0] ?? 0]), flips), order)
  const y = take(r.y, order)
  return {
    x: toArray(take(r.x, order, 0)) as number[][],
    y: toFlat(where(flip, sub(1, y), y)),
    flipped: item(sum(flip)),
  }
}

/** The unit-scale Cholesky factor `L` of the Gram matrix of the inputs (jitter on the diagonal), `[n, n]`. */
function unitFactor(problem: Pick<GpClassificationProblem, 'x' | 'kernel' | 'period' | 'dim'>, ell: number): Value {
  const k = kernelOf(problem.kernel, ell, problem.period, problem.dim)
  return cholesky(gram(k, problem.x as number[][], { jitter: JITTER }))
}

/** An argument of the custom node: a value. */
type Arg = { readonly value: Value }

/** The model of a configuration: the whitened latent function with a probit or logistic Bernoulli observation. */
function definitionOf(p: {
  readonly kernel: string
  readonly period: number
  readonly dim: number
  readonly inferred: boolean
  readonly link: 'logistic' | 'probit'
  readonly lengthScale: number
  readonly variance: number
  readonly priorSpread: number
}): ModelDefinition {
  if (!p.inferred) {
    // region gp:fixed
    const fixed = defineModel('gp-classification', (m) => {
      const row = m.range('row')
      const latent = m.range('latent')
      const L = m.data('L', { dims: [row, latent], role: 'covariate' })
      const y = m.data('y', { dims: [row], support: 'binary', role: 'response' })
      const z = m.latent('z', d.normal(0, 1), {
        dims: [latent],
        doc: 'the whitened latent function: f = L z',
      })
      const f = L.matmul(z)
      // endregion
      if (p.link === 'probit') {
        // region gp:fixed:probit
        m.observe('y', d.bernoulli({ probs: ops.probit(f) }), y)
        // endregion
      } else {
        // region gp:fixed:logistic
        m.observe('y', d.bernoulli({ logits: f }), y)
        // endregion
      }
      // region gp:fixed:end
    })
    // endregion
    return fixed
  }
  // region gp:function
  // The latent function from the kernel's hyperparameters: f = sqrt(variance) L(ell) z.
  const latentFunction = (args: readonly { readonly value: Value }[]): Value => {
    const [xs, scale, v, white] = args as readonly [Arg, Arg, Arg, Arg]
    const kernel = kernelOf(p.kernel, clip(scale.value, 0.05, 20), p.period, p.dim)
    const L = cholesky(gram(kernel, xs.value, { jitter: JITTER }))
    return mul(matmul(L, white.value), sqrt(clip(v.value, 0.01, 200)))
  }
  // endregion
  // region gp:inferred
  const inferred = defineModel('gp-classification', (m) => {
    const row = m.range('row')
    const dim = m.range('dim')
    const x = m.data('x', { dims: [row, dim], role: 'covariate' })
    const y = m.data('y', { dims: [row], support: 'binary', role: 'response' })
    const ell = m.latent('ell', d.logNormal(Math.log(p.lengthScale), p.priorSpread), {
      doc: 'the length scale of the kernel',
    })
    const variance = m.latent('variance', d.logNormal(Math.log(p.variance), p.priorSpread), {
      doc: 'the variance of the kernel',
    })
    const z = m.latent('z', d.normal(0, 1), {
      dims: [row],
      doc: 'the whitened latent function: f = sqrt(variance) L(ell) z',
    })
    const f = m.custom('gpLatent', latentFunction, [x, ell, variance, z], { dims: [row] })
    // endregion
    if (p.link === 'probit') {
      // region gp:inferred:probit
      m.observe('y', d.bernoulli({ probs: ops.probit(f) }), y)
      // endregion
    } else {
      // region gp:inferred:logistic
      m.observe('y', d.bernoulli({ logits: f }), y)
      // endregion
    }
    // region gp:inferred:end
  })
  // endregion
  return inferred
}

const memo = new Map<string, GpClassificationProblem>()

/**
 * The problem a configuration stands for: the data set (a seed, a size, a fraction of flipped labels), the grid the
 * predictive is drawn on, and the model with the kernel, its hyperparameters (fixed, or inferred with log-normal priors
 * centred on the given values) and the link.
 *
 * @param config - The knobs (`dataset`, `rows`, `flips`, `dataSeed`, `kernel`, `lengthScale`, `variance`, `period`, `hyper`, `link`, `priorSpread`).
 * @returns The problem.
 * @example
 * ```ts
 * import { gpClassificationProblem } from './index.ts'
 * const p = gpClassificationProblem({ dataset: 'sinusoid', rows: 30, flips: 0, dataSeed: 1, kernel: 'rbf', lengthScale: 1, variance: 3, period: 4, hyper: 'fixed', link: 'logistic', priorSpread: 0.6 })
 * [p.x.length, p.dim, p.grid.length] // => [30, 1, 100]
 * ```
 */
export function gpClassificationProblem(config: Config): GpClassificationProblem {
  const id = JSON.stringify(gpProblemKnobs.map((k) => config[k]))
  const hit = memo.get(id)
  if (hit !== undefined) return hit
  const name = exampleString(config, 'dataset') as GpDataset
  const n = Math.max(6, Math.round(exampleNumber(config, 'rows')))
  const seed = key(exampleNumber(config, 'dataSeed'))
  const dim = gpDimension(name)
  const base = dim === 1 ? raw1d(name, n, seed) : raw2d(name, n, seed)
  const flips = exampleNumber(config, 'flips')
  const data = finish(base, flips, seed)
  const gridPoints = gridOf(base.extent)
  const grid = toArray(gridPoints) as number[][]
  const truth = base.truth === null ? null : base.truth(gridPoints)
  const link = exampleString(config, 'link') === 'probit' ? 'probit' : 'logistic'
  const pars = {
    kernel: exampleString(config, 'kernel') || 'rbf',
    period: exampleNumber(config, 'period'),
    dim,
    inferred: exampleString(config, 'hyper') === 'inferred',
    link,
    lengthScale: exampleNumber(config, 'lengthScale'),
    variance: exampleNumber(config, 'variance'),
    priorSpread: exampleNumber(config, 'priorSpread'),
  } as const
  const flipsRate = flips
  const effective = truth === null ? null : toFlat(add(mul(truth, 1 - flipsRate), mul(sub(1, truth), flipsRate)))
  const common = { x: data.x, kernel: pars.kernel, period: pars.period, dim }
  let modelData: DataInput
  if (pars.inferred) modelData = { x: data.x, y: data.y }
  else {
    modelData = { L: mul(unitFactor(common, pars.lengthScale), Math.sqrt(pars.variance)), y: data.y }
  }
  const problem: GpClassificationProblem = {
    ...pars,
    x: data.x,
    y: data.y,
    dim,
    grid,
    gridSize: dim === 1 ? grid.length : GRID_2D,
    extent: base.extent,
    truthProbability: effective,
    truthLengthScale: base.ell,
    flipped: data.flipped,
    definition: definitionOf(pars),
    data: modelData,
  }
  if (memo.size > 24) memo.clear()
  memo.set(id, problem)
  return problem
}

/** The number of inputs of a data set. */
function gpDimension(name: string): 1 | 2 {
  return name === 'gp-draw' || name === 'sinusoid' || name === 'step' ? 1 : 2
}

/** The knobs of the data: the data set, its size, the fraction of labels flipped and the seed. */
export const gpDataKnobs: DictSpace = dict({
  dataset: choice([...gpDatasets], {
    default: 'gp-draw',
    label: 'data set',
    descriptions: {
      'gp-draw':
        'one input: a function drawn from a squared-exponential Gaussian process (length scale 1), labels from its logistic',
      sinusoid: 'one input: the probability of a 1 oscillates, sigmoid(3 sin(1.6 x)) (a periodic kernel fits it)',
      step: 'one input: a sharp boundary at 0.5, sigmoid(7 (x - 0.5))',
      circle: 'two inputs: class 1 inside a disc',
      xor: 'two inputs: class 1 in two opposite quadrants',
      moons: 'two inputs: two interleaved half circles (the library data set, inputs standardised)',
      circles: 'two inputs: one class inside a ring of the other (the library data set, inputs standardised)',
    },
  }),
  rows: int(10, 150, { default: 30, log: true, label: 'rows' }),
  flips: real(0, 0.4, { default: 0.05, step: 0.01, label: 'fraction of labels flipped' }),
  dataSeed: int(0, 9999, { default: 1, label: 'seed' }),
})

/** The knobs of the model: the kernel and its numbers, whether they are inferred, the link. */
export const gpModelKnobs: DictSpace = dict({
  kernel: choice([...gpKernels], {
    default: 'rbf',
    label: 'kernel',
    descriptions: {
      rbf: 'squared exponential: infinitely smooth functions',
      matern52: 'Matern 5/2: twice differentiable functions',
      matern32: 'Matern 3/2: once differentiable functions, rougher',
      periodic: 'periodic: functions that repeat with the period (the period is fixed)',
    },
  }),
  lengthScale: real(0.1, 8, {
    default: 1,
    log: true,
    label: 'length scale',
    description:
      'how far apart two inputs can be and still be related; with the hyperparameters inferred it is the median of the prior',
  }),
  variance: real(0.2, 30, {
    default: 3,
    log: true,
    label: 'variance',
    description:
      'the variance of the latent function (how far from 1/2 the probability can go); with the hyperparameters inferred it is the median of the prior',
  }),
  period: real(0.5, 12, { default: 3.9, log: true, label: 'period', description: 'the period of the periodic kernel' }),
  hyper: choice(['fixed', 'inferred'], {
    default: 'fixed',
    label: 'length scale and variance',
    descriptions: {
      fixed: 'the two numbers are taken as given: message passing (EP, VMP) and the Gaussian method apply',
      inferred:
        'the two numbers get log-normal priors centred on the values above and are inferred with the function: the Gram matrix is then a custom function and only samplers, variational and optimisation methods run (NUTS then needs several seconds: raise the time budget, or use SVI, Laplace or Pathfinder)',
    },
  }),
  link: choice([...gpLinks], {
    default: 'logistic',
    label: 'link',
    descriptions: { logistic: 'the logistic function', probit: 'the normal CDF' },
  }),
  priorSpread: real(0.1, 2, {
    default: 0.6,
    log: true,
    label: 'hyperprior spread',
    description: 'the standard deviation of the log of the length scale and of the variance (when they are inferred)',
  }),
})

/** The knobs that make the problem (what a reference run of it depends on). */
export const gpProblemKnobs: readonly string[] = [
  'dataset',
  'rows',
  'flips',
  'dataSeed',
  'kernel',
  'lengthScale',
  'variance',
  'period',
  'hyper',
  'link',
  'priorSpread',
]

/** The knobs a configuration does not use (the period of a kernel that has none, the hyperprior of fixed numbers). */
export function gpHiddenKnobs(config: Config): readonly string[] {
  return [
    ...(exampleString(config, 'kernel') === 'periodic' ? [] : ['period']),
    ...(exampleString(config, 'hyper') === 'inferred' ? [] : ['priorSpread']),
  ]
}

const FIT_DRAWS = 150
const PAIR_DRAWS = 400
const SCORE_DRAWS = 200
const LATENT_DRAWS = 300

/** What a check against a reference run needs of it: its draws of the hyperparameters and of the latent function at the inputs. */
export interface GpReference {
  readonly names: readonly string[]
  readonly columns: readonly (readonly number[])[]
}

/** The cached solutions for one (length scale, variance): the factor, the map from the whitened `z` to the grid, the conditional sd. */
interface Predictor {
  readonly L: Value
  readonly scale: number
  readonly A: Value | null
  readonly cond: Value | null
}

function predictor(problem: GpClassificationProblem, ell: number, variance: number, onGrid: boolean): Predictor {
  const kernel = kernelOf(problem.kernel, ell, problem.period, problem.dim)
  const L = cholesky(gram(kernel, problem.x as number[][], { jitter: JITTER }))
  const scale = Math.sqrt(variance)
  if (!onGrid) return { L, scale, A: null, cond: null }
  const Ks = crossGram(kernel, problem.x as number[][], problem.grid as number[][])
  const A = solveTriangular(L, Ks, { lower: true })
  const cond = mul(sqrt(maximum(sub(1, sum(square(A), 0)), 0)), scale)
  return { L, scale, A, cond }
}

/** The latent function at the inputs of draws of `z` (a row each): `scale * L z`. */
function latentAtInputs(p: Predictor, z: Value): Value {
  return mul(matmul(z, transpose(p.L)), p.scale)
}

/** The latent function on the grid of draws of `z`: `scale * A^T z` plus the conditional spread times a standard normal `eps`. */
function latentOnGrid(p: Predictor, z: Value, eps: Value): Value {
  return add(mul(matmul(z, p.A as Value), p.scale), mul(p.cond as Value, eps))
}

const probabilityOf = (link: 'logistic' | 'probit', f: Value): Value => (link === 'probit' ? ndtr(f) : sigmoid(f))

/** The draws of a posterior read as a function of the inputs: hyperparameters, the latent function at the inputs and on the grid, and probabilities. */
interface Evaluated {
  readonly count: number
  readonly ell: Value | null
  readonly variance: Value | null
  /** `f[s, i]`: the latent function at input i of draw s. */
  readonly f: Value
  /** `g[s, j]`: the latent function at grid point j (with the conditional spread). */
  readonly g: Value | null
  /** `pg[s, j]`, `pf[s, i]`: the probability of a 1 on the grid and at the inputs. */
  readonly pg: Value | null
  readonly pf: Value
  readonly chain: number[]
  readonly index: number[]
}

function evaluate(
  problem: GpClassificationProblem,
  post: Posterior,
  most: number,
  onGrid: boolean,
): Evaluated | undefined {
  const n = problem.x.length
  const m = problem.grid.length
  const sites = problem.inferred ? ['ell', 'variance', 'z'] : ['z']
  const got = drawRows(post, sites, most)
  if (got === undefined) return undefined
  const lead = problem.inferred ? 2 : 0
  const count = got.rows.length
  const rows = tensor(got.rows)
  const eps = onGrid ? reshape(drawNormal(key(3), [count * m]), [count, m]) : null
  const z = sliceAxis(rows, 1, lead, lead + n)
  const ell = problem.inferred ? clip(take(rows, 0, 1), 0.05, 20) : full([count], problem.lengthScale)
  const variance = problem.inferred ? clip(take(rows, 1, 1), 0.01, 200) : full([count], problem.variance)
  // With fixed hyperparameters one predictor serves every draw; otherwise each draw has its own.
  const draws = problem.inferred
    ? toFlat(ell).map((e, s) => {
        const pr = predictor(problem, e, toFlat(variance)[s] as number, onGrid)
        const zs = take(z, s, 0)
        return {
          f: latentAtInputs(pr, zs),
          g: eps === null ? null : latentOnGrid(pr, zs, take(eps, s, 0)),
        }
      })
    : undefined
  const fixed = problem.inferred ? undefined : predictor(problem, problem.lengthScale, problem.variance, onGrid)
  const f = draws === undefined ? latentAtInputs(fixed as Predictor, z) : stack(draws.map((d) => d.f))
  const g =
    eps === null
      ? null
      : draws === undefined
        ? latentOnGrid(fixed as Predictor, z, eps)
        : stack(draws.map((d) => d.g as Value))
  return {
    count,
    ell: problem.inferred ? ell : null,
    variance: problem.inferred ? variance : null,
    f,
    g,
    pg: g === null ? null : probabilityOf(problem.link, g),
    pf: probabilityOf(problem.link, f),
    chain: got.chain,
    index: got.index,
  }
}

/** The mean, 2.5% and 97.5% of each column of a list of draws (a row per draw). */
function bands(draws: Value, width: number): { mean: number[]; lo: number[]; hi: number[]; sd: number[] } {
  const levels = tensor([0.025, 0.975])
  const q = stack(Array.from({ length: width }, (_, j) => quantile(take(draws, j, 1), levels)))
  return {
    mean: toFlat(div(sum(draws, 0), draws.shape[0] ?? 1)),
    lo: toFlat(take(q, 0, 1)),
    hi: toFlat(take(q, 1, 1)),
    sd: columnMoments(draws, width).sd,
  }
}

/** What the gp domain's Laplace and EP fits say of the latent function at the inputs (fixed hyperparameters). */
interface DomainFit {
  readonly name: string
  readonly mean: number[]
  readonly sd: number[]
  readonly probability: number[]
}

function domainFits(problem: GpClassificationProblem): DomainFit[] {
  const scaled = scaleKernel(
    kernelOf(problem.kernel, problem.lengthScale, problem.period, problem.dim),
    problem.variance,
  )
  const out: DomainFit[] = []
  const likelihood = problem.link === 'probit' ? 'probit' : 'logit'
  const methods: ('laplace' | 'ep')[] = problem.link === 'probit' ? ['laplace', 'ep'] : ['laplace']
  for (const method of methods) {
    try {
      const fitted = gpClassifier
        .with({ kernel: scaled, method, likelihood, optimise: false })
        .fit({ x: problem.x as number[][], y: problem.y as number[] }, { explain: false })
      const { mean, variance } = fitted.latent(problem.x as number[][])
      const proba = fitted.predictProba(problem.x as number[][])
      out.push({
        name: method === 'laplace' ? 'Laplace (gp domain)' : 'EP (gp domain)',
        mean: toFlat(mean),
        sd: toFlat(sqrt(maximum(variance, 0))),
        probability: toFlat(take(reshape(proba, [-1, 2]), 1, 1)),
      })
    } catch {
      // a fit that fails is left out of the rows
    }
  }
  return out
}

const PAIR_INDEXES = [0.2, 0.5, 0.8]

/** The names of the latent function at the inputs: `f(x=-1.2)` for one input, `f_i` for two. */
function latentNames(problem: GpClassificationProblem): string[] {
  return problem.x.map((r, i) => (problem.dim === 1 ? `f(x=${(r[0] ?? 0).toFixed(1)})` : `f_${i}`))
}

export const gpClassificationExample: Example<GpClassificationProblem> = defineExample<GpClassificationProblem>({
  name: 'gpc',
  description:
    'Gaussian process classification (latent GP, Bernoulli observation) with one inference method checked against a long NUTS run',
  modelName: 'gp-classification',
  budgetMs: 4000,
  dataKnobs: gpDataKnobs,
  modelKnobs: gpModelKnobs,
  problemKnobs: gpProblemKnobs,
  problem: gpClassificationProblem,
  derived: ({ problem, kind, posterior: post }) => {
    const n = problem.x.length
    const names = latentNames(problem)
    const columnsOf = (ev: Evaluated): { names: string[]; columns: number[][] } => ({
      names: [...(problem.inferred ? ['ell', 'variance'] : []), ...names],
      columns: [
        ...(ev.ell === null || ev.variance === null ? [] : [toFlat(ev.ell), toFlat(ev.variance)]),
        ...(toArray(transpose(ev.f)) as number[][]),
      ],
    })
    return {
      'gpc/fit': derived<unknown>({
        cost: 'dear',
        description:
          'the predictive probability of a 1 with its 95% band over the input (or the mean and the spread over the plane), the latent function with its band, the data and the truth',
        compute: (state) => {
          const ev = evaluate(problem, post(state), FIT_DRAWS, true)
          if (ev === undefined || ev.pg === null || ev.g === null) return null
          const m = problem.grid.length
          const p = bands(ev.pg, m)
          const latent = bands(ev.g, m)
          return {
            dim: problem.dim,
            size: problem.gridSize,
            x: problem.grid.map((r) => r[0] ?? 0),
            y: problem.grid.map((r) => r[1] ?? 0),
            probability: p,
            latent,
            truth: problem.truthProbability,
            data: { x: problem.x, y: problem.y },
            extent: problem.extent,
            draws: ev.count,
            flipped: problem.flipped,
          } as never
        },
      }),
      'gpc/pairs': derived<unknown>({
        cost: 'dear',
        description:
          'the pair plot of the length scale and the variance (when inferred) and the latent function at three inputs (a posteriorPairs record)',
        compute: (state) => {
          const ev = evaluate(problem, post(state), PAIR_DRAWS, false)
          if (ev === undefined || ev.count < 3) return null
          const pick = PAIR_INDEXES.map((q) => Math.min(n - 1, Math.round(q * (n - 1))))
          const coordinates = [
            ...(problem.inferred ? ['ell', 'variance'] : []),
            ...pick.map((i) => names[i] ?? `f_${i}`),
          ]
          const cols: number[][] = [
            ...(ev.ell === null || ev.variance === null ? [] : [toFlat(ev.ell), toFlat(ev.variance)]),
            ...pick.map((i) => toFlat(take(ev.f, i, 1))),
          ]
          const draws = Object.fromEntries(coordinates.map((c, j) => [c, cols[j] ?? []]))
          const truth =
            problem.inferred && problem.truthLengthScale !== null ? { ell: problem.truthLengthScale } : undefined
          return posteriorPairs([{ name: kind, draws, steps: ev.index, chain: ev.chain }], {
            coordinates,
            maxPoints: 300,
            frames: 2,
            densityDraws: 300,
            ...(truth === undefined ? {} : { truth }),
          }) as never
        },
      }),
      'gpc/scores': derived<unknown>({
        cost: 'dear',
        description:
          'the training accuracy and mean log likelihood of the predictive, the distance from the true probability over the grid when there is one, the hyperparameters, R-hat, ESS and divergences',
        compute: (state) => {
          const posterior = post(state)
          const ev = evaluate(problem, posterior, SCORE_DRAWS, true)
          if (ev === undefined || ev.pg === null) return null
          const pTrain = div(sum(ev.pf, 0), ev.count)
          const labels = tensor(problem.y as number[])
          const accuracy = item(sum(equal(cast(greater(pTrain, 0.5), 'float64'), labels))) / n
          const loglik = item(sum(log(maximum(where(equal(labels, 1), pTrain, sub(1, pTrain)), 1e-12)))) / n
          const pGrid = bands(ev.pg, problem.grid.length).mean
          const truth = problem.truthProbability
          const error =
            truth === null ? null : item(sum(abs(sub(tensor(pGrid), tensor(truth as number[]))))) / pGrid.length
          const rows = posterior.summary()
          const rhat = rows.flatMap((r) => (r.rhat === undefined ? [] : [r.rhat]))
          const ess = rows.flatMap((r) => (r.ess === undefined ? [] : [r.ess]))
          const divergences = posterior.diagnostics['divergences']
          const moments = (v: Value | null): { mean: number; sd: number } | null => {
            if (v === null) return null
            const m = columnMoments(reshape(v, [-1, 1]), 1)
            return { mean: m.mean[0] ?? 0, sd: m.sd[0] ?? 0 }
          }
          return {
            trainAccuracy: accuracy,
            trainLogLikelihood: loglik,
            meanAbsError: error,
            ell: moments(ev.ell),
            variance: moments(ev.variance),
            truthLengthScale: problem.truthLengthScale,
            draws: posterior.chains === undefined ? ev.count : posterior.chains.numChains * posterior.chains.numDraws,
            maxRhat: rhat.length > 0 ? Math.max(...rhat) : null,
            minEss: ess.length > 0 ? Math.min(...ess) : null,
            divergences: typeof divergences === 'number' ? divergences : null,
          }
        },
      }),
      'gpc/diagnostics': derived<unknown>({
        cost: 'dear',
        description:
          'trace, rank, R-hat and ESS of the chains of a sampler for the length scale, the variance and a few whitened coordinates (a chainDiagnostics record); null for the others',
        compute: (state) =>
          exampleDiagnostics(
            post(state),
            problem.inferred ? ['ell', 'variance', 'z'] : ['z'],
            (name) => name,
            5,
          ) as never,
      }),
      'gpc/draws': derived<unknown>({
        cost: 'dear',
        description:
          'the draws of the hyperparameters (when inferred) and of the latent function at every input, one column each (what a check against a reference run needs of the reference)',
        compute: (state) => {
          const ev = evaluate(problem, post(state), LATENT_DRAWS, false)
          return ev === undefined ? null : (columnsOf(ev) as never)
        },
      }),
      'gpc/check': derived<unknown>({
        cost: 'dear',
        description:
          "the method against reference draws (args `{ reference }`, the reference's `gpc/draws`): a comparison record of the hyperparameters and some latent values, the errors over the latent function at every input, and with fixed hyperparameters the Laplace and EP fits of the gp domain beside it",
        compute: (state, args) => {
          const ref = (args as { reference?: GpReference | null } | null | undefined)?.reference
          if (ref === undefined || ref === null) return null
          const ev = evaluate(problem, post(state), LATENT_DRAWS, false)
          if (ev === undefined) return null
          const { names: qn, columns } = columnsOf(ev)
          const approx: Approximation = {
            name: kind,
            draws: Object.fromEntries(qn.map((q, j) => [q, columns[j] ?? []])),
          }
          const oracle: Record<string, MarginalOracle> = {}
          ref.names.forEach((q, j) => {
            const c = ref.columns[j]
            if (c !== undefined && q in approx.draws) oracle[q] = weightedOracle(q, c)
          })
          const quantities = Object.keys(oracle)
          if (quantities.length === 0) return null
          const table = agree([approx], { oracle, quantities })
          const rows = table.rows
          const picked = PAIR_INDEXES.map((q) => names[Math.min(n - 1, Math.round(q * (n - 1)))])
          const shown = quantities.filter((q) => q === 'ell' || q === 'variance' || picked.includes(q))
          // The reference's mean and sd of the latent function at each input.
          const refColumns = tensor(names.map((q) => (ref.columns[ref.names.indexOf(q)] ?? []) as number[]))
          const refCount = refColumns.shape[1] ?? 0
          const refMean = div(sum(refColumns, 1), Math.max(1, refCount))
          const refSd = sqrt(div(sum(square(sub(refColumns, reshape(refMean, [-1, 1]))), 1), Math.max(1, refCount - 1)))
          const latentRows = rows.filter((r) => r.quantity !== 'ell' && r.quantity !== 'variance')
          // The gp domain's fits (fixed hyperparameters): how close each is to the method and to the reference.
          const extras = problem.inferred
            ? []
            : domainFits(problem).map((fit) => {
                const methodMean = div(sum(ev.f, 0), ev.count)
                const methodProb = div(sum(ev.pf, 0), ev.count)
                const fitMean = tensor(fit.mean)
                const vsMethod = item(max(div(abs(sub(fitMean, methodMean)), maximum(tensor(fit.sd), 1e-9))))
                const vsReference = item(max(div(abs(sub(fitMean, refMean)), maximum(refSd, 1e-9))))
                const probMethod = item(max(abs(sub(tensor(fit.probability), methodProb))))
                return { name: fit.name, vsMethod, vsReference, probabilityDifference: probMethod }
              })
          return {
            comparison: posteriorComparison(
              [{ ...approx, draws: Object.fromEntries(shown.map((q) => [q, approx.draws[q] as ArrayLike<number>])) }],
              {
                truth: Object.fromEntries(shown.map((q) => [q, oracle[q] as MarginalOracle])),
                table: { ...table, quantities: shown, rows: rows.filter((r) => shown.includes(r.quantity)) },
                quantities: shown,
                gridPoints: 100,
                oracle: 'a long NUTS run',
              },
            ),
            maxMeanError: Math.max(...rows.map((r) => Math.abs(r.meanError))),
            latentMaxMeanError:
              latentRows.length === 0 ? null : Math.max(...latentRows.map((r) => Math.abs(r.meanError))),
            medianSdRatio: item(take(sort(tensor(rows.map((r) => r.sdRatio))), Math.floor(rows.length / 2))),
            maxWasserstein: Math.max(...rows.map((r) => r.wasserstein)),
            hyper: problem.inferred
              ? ['ell', 'variance'].flatMap((q) => {
                  const r = rows.find((x) => x.quantity === q)
                  return r === undefined ? [] : [{ name: q, meanError: r.meanError, sdRatio: r.sdRatio }]
                })
              : [],
            extras,
            referenceDraws: ref.columns[0]?.length ?? 0,
          } as never
        },
      }),
    }
  },
})

/**
 * Gaussian process classification as a method, run by name and knobs: see the file's summary. A trace `Method` whose tape
 * is the chosen inference method's own, with the derived paths `gpc/fit`, `gpc/pairs`, `gpc/scores`, `gpc/diagnostics`,
 * `gpc/draws` and `gpc/check`.
 *
 * @concept gaussian-process-classification
 * @index estimator Gaussian process classification method: GP prior on the latent function, Bernoulli link, one inference method.
 * @example
 * ```ts
 * import { gpClassificationMethod, gpClassificationSteps } from './index.ts'
 * import { defaults } from 'omai-engine/space'
 * import { session } from 'omai-engine/trace'
 * const config = { ...defaults(gpClassificationMethod.config), rows: 20, inference: { method: 'laplace' } }
 * const s = session(gpClassificationMethod, { config, steps: gpClassificationSteps(config), mode: 'eager' })
 * const fit = s.probe(s.t, 'gpc/fit') as { dim: number }
 * fit.dim // => 1
 * ```
 */
export const gpClassificationMethod: InferenceMethod<unknown, Config, string> = gpClassificationExample.method

/** The steps a configuration's run is expected to take (see `exampleSteps`). */
export const gpClassificationSteps: (config: Config) => number = gpClassificationExample.steps
