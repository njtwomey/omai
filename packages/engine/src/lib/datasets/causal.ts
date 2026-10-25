/**
 * Generators for causal inference with known answers: random linear structural causal models over a random DAG (the
 * causal-discovery benchmark), observational treatment-effect data with confounding and both potential outcomes, and
 * instrumental-variable data with an unobserved confounder.
 */

import { type DatasetGenerator, type RegressionData, type Truth, xy } from '../data/index.ts'
import { foldIn, type Key, laplace, normal, permutation, uniform } from '../random/index.ts'
import { structure } from '../scene/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { add, cos, mul, sigmoid, sin, sliceAxis, square, sub, type Tensor, tanh, type Value } from '../tensor/index.ts'
import { column, matrix, stamp, vector } from './_util.ts'

/** Noise draws for structural equations: Gaussian, uniform or Laplace with unit variance. */
function noise(k: Key, n: number, kind: string): Float64Array {
  if (kind === 'uniform') return Float64Array.from(uniform(k, [n], { low: -Math.sqrt(3), high: Math.sqrt(3) }).data)
  if (kind === 'laplace') return Float64Array.from(laplace(k, [n], 0, 1 / Math.SQRT2).data)
  return Float64Array.from(normal(k, [n]).data)
}

/** `(I - B)^-1 - I` for a strictly lower-triangular (in causal order) weight matrix: the total causal effects. */
function totalEffects(b: readonly (readonly number[])[], order: readonly number[]): number[][] {
  const p = b.length
  // T[i][j] = sum over directed paths j -> ... -> i of the product of weights; accumulate in causal order.
  const t = Array.from({ length: p }, () => new Array<number>(p).fill(0))
  for (const i of order) {
    for (let j = 0; j < p; j++) {
      const direct = b[i]?.[j] ?? 0
      if (direct === 0) continue
      const ti = t[i] as number[]
      ti[j] = (ti[j] ?? 0) + direct
      for (let a = 0; a < p; a++) ti[a] = (ti[a] ?? 0) + direct * (t[j]?.[a] ?? 0)
    }
  }
  return t
}

/**
 * A random linear structural causal model (the standard causal-discovery benchmark): a random DAG over `p` variables
 * (edges with probability `edgeProbability` along a random causal order), weights of magnitude in [0.5, 2] with random
 * signs, and independent Gaussian, uniform or Laplace noise (non-Gaussian noise makes the DAG identifiable, as in
 * LiNGAM). All variables are the features; the target repeats the last variable in causal order. The truth has the
 * DAG (as a `Structure`), the weights, the total effects and interventional means `E[X | do(X_j = v)]`.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { linearScm } from 'mdml-engine/datasets'
 * generate(linearScm, { knobs: { variables: 4 } }).truth.structure?.nodes.length // => 4
 * ```
 */
export const linearScm: DatasetGenerator<RegressionData> = {
  name: 'linearScm',
  kind: 'regression',
  description: 'Samples from a random linear structural causal model over a random DAG.',
  knobs: dict({
    n: int(1, 1_000_000, { label: 'samples', description: 'The number of rows.', default: 500 }),
    variables: int(2, 100, {
      label: 'variables',
      description: 'The number of variables (nodes of the DAG).',
      default: 5,
    }),
    edgeProbability: real(0, 1, {
      label: 'edge probability',
      description: 'The chance of an edge between each ordered pair of variables.',
      default: 0.4,
    }),
    noise: choice(['gaussian', 'uniform', 'laplace'], {
      label: 'noise distribution',
      description: 'The distribution of the exogenous noise (non-Gaussian noise makes the DAG identifiable).',
      default: 'gaussian',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['variables'] as number
    const order = Array.from(permutation(foldIn(k, 0), p).data)
    const u = uniform(foldIn(k, 1), [p * p]).data
    const w = uniform(foldIn(k, 2), [p * p]).data
    const b = Array.from({ length: p }, () => new Array<number>(p).fill(0))
    for (let a = 0; a < p; a++)
      for (let c = 0; c < a; c++) {
        // order[c] precedes order[a]: maybe an edge order[c] -> order[a].
        if ((u[a * p + c] ?? 1) < (knobs['edgeProbability'] as number)) {
          const v = w[a * p + c] ?? 0.5
          const mag = 0.5 + 1.5 * Math.abs(2 * v - 1)
          ;(b[order[a] ?? 0] as number[])[order[c] ?? 0] = v < 0.5 ? -mag : mag
        }
      }
    const e = Array.from({ length: p }, (_, j) => noise(foldIn(foldIn(k, 3), j), n, knobs['noise'] as string))
    const x = new Float64Array(n * p)
    for (let i = 0; i < n; i++)
      for (const j of order) {
        let s = e[j]?.[i] ?? 0
        for (let a = 0; a < p; a++) s += (b[j]?.[a] ?? 0) * (x[i * p + a] ?? 0)
        x[i * p + j] = s
      }
    const sink = order[p - 1] ?? 0
    const total = totalEffects(b, order)
    const names = Array.from({ length: p }, (_, j) => `X${j}`)
    // With zero-mean noise, E[X | do(X_j = v)] = v times the total effects of j (and v itself at j).
    const interventionalMean = (j: number, v: number): number[] =>
      names.map((_, i) => (i === j ? v : v * (total[i]?.[j] ?? 0)))
    const parents = (i: number) => names.flatMap((_, a) => ((b[i]?.[a] ?? 0) !== 0 ? [a] : []))
    const truth: Truth = {
      structure: structure(
        'causal-dag',
        names.map((nm) => ({ id: nm, label: nm, role: 'observed' as const })),
        b.flatMap((row, i) =>
          row.flatMap((v, a) =>
            v !== 0 ? [{ from: names[a] ?? '', to: names[i] ?? '', directed: true, label: v.toFixed(2) }] : [],
          ),
        ),
      ),
      relevantFeatures: parents(sink),
      extra: {
        weights: b,
        totalEffects: total,
        causalOrder: order,
        interventionalMean,
        parents: names.map((_, i) => parents(i)),
      },
    }
    return {
      data: xy(matrix(x, n, p), vector(Array.from({ length: n }, (_, i) => x[i * p + sink] ?? 0)), {
        ...stamp('linearScm', linearScm.description, knobs),
        featureNames: names,
        targetNames: [names[sink] ?? 'X'],
      }),
      truth,
    }
  },
}

/**
 * Observational treatment-effect data with known potential outcomes: covariates `x ~ N(0, I)`, a confounded
 * treatment `T ~ Bernoulli(e(x))` with propensity `e(x) = clip(sigmoid(confounding (x0 + x1) / √2), overlap,
 * 1 - overlap)`, baseline outcome `mu0(x)` (linear or nonlinear), and effect `tau(x) = ate + heterogeneity × x0`
 * (or `× tanh(2 x1)` in the nonlinear setting). Features are the covariates and then the treatment; the target is the
 * observed outcome. The truth has the ATE, ATT, CATE, every unit's individual effect and both potential outcomes.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { treatmentEffects } from 'mdml-engine/datasets'
 * generate(treatmentEffects, { knobs: { ate: 2 } }).truth.effects?.ate // => 2
 * ```
 */
export const treatmentEffects: DatasetGenerator<RegressionData> = {
  name: 'treatmentEffects',
  kind: 'regression',
  description: 'Confounded binary treatment, covariates and outcomes with known heterogeneous effects.',
  knobs: dict({
    n: int(2, 1_000_000, { label: 'samples', description: 'The number of units.', default: 1000 }),
    covariates: int(2, 100, { label: 'covariates', description: 'The number of covariates.', default: 5 }),
    setting: choice(['linear', 'nonlinear'], {
      label: 'setting',
      description: 'Linear or nonlinear outcome and propensity models.',
      default: 'linear',
    }),
    ate: real(-100, 100, { label: 'average effect', description: 'The average treatment effect.', default: 1 }),
    heterogeneity: real(0, 10, {
      label: 'heterogeneity',
      description: 'How much the effect varies with the covariates.',
      default: 0.5,
    }),
    confounding: real(0, 10, {
      description: 'How strongly the covariates drive the treatment propensity.',
      default: 1,
      label: 'strength of confounding in the propensity',
    }),
    overlap: real(0, 0.49, {
      description: 'Propensities are clipped to [overlap, 1 - overlap].',
      default: 0.05,
      label: 'propensity clipping',
    }),
    noise: real(0, 10, {
      label: 'noise standard deviation',
      description: 'The standard deviation of the outcome noise.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['covariates'] as number
    const linear = knobs['setting'] === 'linear'
    const ate = knobs['ate'] as number
    const het = knobs['heterogeneity'] as number
    const conf = knobs['confounding'] as number
    const clipLo = knobs['overlap'] as number
    const x = normal(foldIn(k, 0), [n, d])
    const propensity = (v: Value): Value => {
      const raw = sigmoid(mul(add(column(v, 0), column(v, 1)), conf / Math.SQRT2))
      return add(mul(raw, 1 - 2 * clipLo), clipLo)
    }
    const mu0 = (v: Value): Value =>
      linear
        ? add(add(column(v, 0), mul(column(v, 1), 0.5)), mul(column(v, 2), -0.5))
        : add(
            add(sin(mul(mul(column(v, 0), column(v, 1)), Math.PI)), square(sub(column(v, 2), 0.5))),
            mul(cos(column(v, 1)), 0.5),
          )
    const cate = (v: Value): Value => add(mul(linear ? column(v, 0) : tanh(mul(column(v, 1), 2)), het), ate)
    const e = Float64Array.from((propensity(x) as Tensor).data as ArrayLike<number>)
    const u = uniform(foldIn(k, 1), [n]).data
    const t = Float64Array.from(e, (p, i) => ((u[i] ?? 1) < p ? 1 : 0))
    const eps = normal(foldIn(k, 2), [n]).data
    const base = Float64Array.from((mu0(x) as Tensor).data as ArrayLike<number>)
    const tau = Float64Array.from((cate(x) as Tensor).data as ArrayLike<number>)
    const s = knobs['noise'] as number
    const y0 = Float64Array.from(base, (m, i) => m + s * (eps[i] ?? 0))
    const y1 = Float64Array.from(y0, (v, i) => v + (tau[i] ?? 0))
    const y = Float64Array.from(t, (ti, i) => (ti === 1 ? (y1[i] ?? 0) : (y0[i] ?? 0)))
    const features = new Float64Array(n * (d + 1))
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) features[i * (d + 1) + j] = x.data[i * d + j] ?? 0
      features[i * (d + 1) + d] = t[i] ?? 0
    }
    const treated = Array.from(t).flatMap((ti, i) => (ti === 1 ? [tau[i] ?? 0] : []))
    const regression = (v: Value): Value => {
      const cov = matrixSlice(v, d)
      const tr = column(v, d)
      return add(mu0(cov), mul(tr, cate(cov)))
    }
    return {
      data: xy(matrix(features, n, d + 1), vector(y), {
        ...stamp('treatmentEffects', treatmentEffects.description, knobs),
        featureNames: [...Array.from({ length: d }, (_, j) => `x${j}`), 'treatment'],
        targetNames: ['outcome'],
      }),
      truth: {
        regressionFunction: regression,
        noiseScale: s,
        effects: {
          ate,
          att: treated.reduce((a, b) => a + b, 0) / Math.max(1, treated.length),
          cate: (v) => cate(matrixSlice(v, d)),
          ite: vector(tau),
          potentialOutcomes: { control: vector(y0), treated: vector(y1) },
        },
        extra: {
          propensity: (v: Value) => propensity(matrixSlice(v, d)),
          treatmentColumn: d,
          sampleAte: tau.reduce((a, b) => a + b, 0) / n,
        },
      },
    }
  },
}

/** The first `d` columns of a matrix value (the covariates). */
function matrixSlice(v: Value, d: number): Value {
  return (v.shape[1] ?? d) === d ? v : sliceAxis(v, 1, 0, d)
}

/**
 * Instrumental-variable data with an unobserved confounder: instrument `Z ~ N(0, 1)`, confounder `U ~ N(0, 1)`,
 * treatment `T = strength Z + confounding U + noise`, outcome `Y = effect T + confounding U + noise`. Features are
 * `[Z, T]`; ordinary least squares of Y on T is biased, two-stage least squares is not. The truth gives the effect and
 * the OLS bias.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { instrumentalVariables } from 'mdml-engine/datasets'
 * generate(instrumentalVariables, { knobs: { effect: 1.5 } }).truth.effects?.ate // => 1.5
 * ```
 */
export const instrumentalVariables: DatasetGenerator<RegressionData> = {
  name: 'instrumentalVariables',
  kind: 'regression',
  description: 'An instrument, a confounded treatment and an outcome, with the true effect.',
  knobs: dict({
    n: int(2, 1_000_000, { label: 'samples', description: 'The number of rows.', default: 1000 }),
    effect: real(-100, 100, {
      label: 'treatment effect',
      description: 'The true causal effect of the treatment on the outcome.',
      default: 1,
    }),
    strength: real(0, 10, {
      description: 'How strongly the instrument moves the treatment (weak instruments near 0).',
      default: 1,
      label: 'instrument strength',
    }),
    confounding: real(0, 10, {
      label: 'confounding',
      description: 'The strength of the unobserved confounder of treatment and outcome.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const beta = knobs['effect'] as number
    const pi = knobs['strength'] as number
    const c = knobs['confounding'] as number
    const z = normal(foldIn(k, 0), [n]).data
    const uu = normal(foldIn(k, 1), [n]).data
    const v = normal(foldIn(k, 2), [n]).data
    const e = normal(foldIn(k, 3), [n]).data
    const t = Float64Array.from(z, (zi, i) => pi * zi + c * (uu[i] ?? 0) + (v[i] ?? 0))
    const y = Float64Array.from(t, (ti, i) => beta * ti + c * (uu[i] ?? 0) + (e[i] ?? 0))
    const features = Float64Array.from({ length: 2 * n }, (_, j) =>
      j % 2 === 0 ? (z[j / 2] ?? 0) : (t[(j - 1) / 2] ?? 0),
    )
    // OLS slope of Y on T converges to beta + cov(T, cU) / var(T) = beta + c^2 / (pi^2 + c^2 + 1).
    const bias = (c * c) / (pi * pi + c * c + 1)
    return {
      data: xy(matrix(features, n, 2), vector(y), {
        ...stamp('instrumentalVariables', instrumentalVariables.description, knobs),
        featureNames: ['instrument', 'treatment'],
        targetNames: ['outcome'],
      }),
      truth: {
        effects: { ate: beta },
        latents: { confounder: vector(uu) },
        extra: { olsBias: bias, instrumentColumn: 0, treatmentColumn: 1, firstStage: pi },
      },
    }
  },
}
