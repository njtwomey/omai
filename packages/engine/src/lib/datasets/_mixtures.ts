// Exact truths of Gaussian-mixture class models: component log densities, joint log densities and class posteriors.
// Small models (and traced inputs) use tensor operations, so the truth is differentiable in x; large isotropic
// models on concrete rows use a chunked loop that never materialises an [n, components] array.
import { OmaiError } from '../base/index.ts'
import { bayesErrorEstimate, type GeneratingProgram, type Truth } from '../data/index.ts'
import { categorical, compose, normal } from '../dists/index.ts'
import { generatingProgram, plate, type RowsData, sample } from '../probprog/index.ts'
import type { Key } from '../random/index.ts'
import {
  add,
  fromStorage,
  isTensor,
  logsumexp,
  matmul,
  mul,
  reshape,
  softmax,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  take,
  takeAlongAxis,
  tensor,
  transpose,
  type Value,
} from '../tensor/index.ts'

/** The standard deviation of component `c`: one shared value or one per component. */
function sigmaOf(sigma: number | readonly number[], c: number): number {
  return typeof sigma === 'number' ? sigma : (sigma[c] ?? 1)
}

/** `log N(x; m_c, sigma_c^2 I)` for every row of `x` and every mean `m_c`: `[n, C]`. */
export function isotropicLogDensities(
  means: readonly (readonly number[])[],
  sigma: number | readonly number[],
): (x: Value) => Value {
  const m = tensor(means as number[][])
  const mt = transpose(m)
  const d = means[0]?.length ?? 0
  const mm = tensor(means.map((r) => r.reduce((s, v) => s + v * v, 0)))
  const scale = tensor(means.map((_, c) => -0.5 / sigmaOf(sigma, c) ** 2))
  const constant = tensor(means.map((_, c) => -0.5 * d * Math.log(2 * Math.PI * sigmaOf(sigma, c) ** 2)))
  return (x) => {
    const xx = sum(square(x), 1, true)
    const d2 = add(sub(xx, mul(matmul(x, mt), 2)), mm)
    return add(mul(d2, scale), constant)
  }
}

/** A Gaussian-mixture class model: components with log densities, the class of each and log weights. */
export interface MixtureModel {
  /** Component log densities `[n, C]`. */
  readonly logDensities: (x: Value) => Value
  /** The class of each component. */
  readonly classOf: readonly number[]
  /** The log weight of each component (summing to one over all components). */
  readonly logWeights: readonly number[]
  readonly nClasses: number
  /** For isotropic models: the means and shared standard deviation, enabling the fast path. */
  readonly isotropic?: { readonly means: readonly (readonly number[])[]; readonly sigma: number | readonly number[] }
}

/** `log p(x, y = k)` for every class: `[n, K]`, with tensor operations. */
function classLogJointTensor(model: MixtureModel): (x: Value) => Value {
  const lw = tensor(model.logWeights as number[])
  const groups = Array.from({ length: model.nClasses }, (_, k) =>
    tensor(
      model.classOf.flatMap((c, j) => (c === k ? [j] : [])),
      { dtype: 'int32' },
    ),
  )
  return (x) => {
    const l = add(model.logDensities(x), lw)
    return stack(
      groups.map((g) => logsumexp(take(l, g, 1), 1)),
      1,
    )
  }
}

/** The same joint for concrete rows by a loop (isotropic models only). */
function classLogJointLoop(model: MixtureModel): (x: Tensor) => Tensor {
  const iso = model.isotropic
  if (iso === undefined) throw new OmaiError('value', 'classLogJointLoop needs an isotropic model')
  const means = iso.means
  const c = means.length
  const d = means[0]?.length ?? 0
  const flatMeans = Float64Array.from(means.flat())
  const s2 = means.map((_, j) => sigmaOf(iso.sigma, j) ** 2)
  const constants = s2.map((v) => -0.5 * d * Math.log(2 * Math.PI * v))
  const k = model.nClasses
  return (x) => {
    const n = x.shape[0] ?? 0
    const xs = x.data as ArrayLike<number>
    const out = new Float64Array(n * k)
    const best = new Float64Array(k)
    const acc = new Float64Array(k)
    const terms = new Float64Array(c)
    for (let i = 0; i < n; i++) {
      best.fill(-Infinity)
      for (let j = 0; j < c; j++) {
        let d2 = 0
        for (let a = 0; a < d; a++) d2 += ((xs[i * d + a] ?? 0) - (flatMeans[j * d + a] ?? 0)) ** 2
        const t = (constants[j] ?? 0) - (0.5 * d2) / (s2[j] ?? 1) + (model.logWeights[j] ?? 0)
        terms[j] = t
        const cls = model.classOf[j] ?? 0
        if (t > (best[cls] ?? -Infinity)) best[cls] = t
      }
      acc.fill(0)
      for (let j = 0; j < c; j++) {
        const cls = model.classOf[j] ?? 0
        acc[cls] = (acc[cls] ?? 0) + Math.exp((terms[j] ?? 0) - (best[cls] ?? 0))
      }
      for (let q = 0; q < k; q++) out[i * k + q] = (best[q] ?? -Infinity) + Math.log(acc[q] ?? 0)
    }
    return fromStorage(out, [n, k], 'float64')
  }
}

/** `log p(x, y = k)` for every class: `[n, K]`. */
export function classLogJoint(model: MixtureModel): (x: Value) => Value {
  const viaTensors = classLogJointTensor(model)
  if (model.isotropic === undefined) return viaTensors
  const viaLoop = classLogJointLoop(model)
  const c = model.classOf.length
  return (x) => (isTensor(x) && (x.shape[0] ?? 0) * c > 20_000 ? viaLoop(x) : viaTensors(x))
}

/**
 * The generating program of an isotropic mixture class model: per row, a component from the weights, features from
 * its isotropic normal, and the component's class as the label.
 */
export function mixtureProgram(model: MixtureModel): GeneratingProgram | undefined {
  const iso = model.isotropic
  if (iso === undefined) return undefined
  const c = iso.means.length
  const means = tensor(iso.means as number[][])
  const sigmas = tensor(iso.means.map((_, j) => [sigmaOf(iso.sigma, j)]))
  const weights = model.logWeights.map(Math.exp)
  const onehot = tensor(
    Array.from({ length: c }, (_, j) =>
      Array.from({ length: model.nClasses }, (_, k) => (model.classOf[j] === k ? 1 : 0)),
    ),
  )
  const program = (data: RowsData): void => {
    plate('rows', data.n, () => {
      const j = sample('component', categorical({ probs: weights }))
      sample('x', compose.independent(normal(take(means, j), take(sigmas, j)), 1), { obs: data.x })
      sample('y', categorical({ probs: take(onehot, j) }), { obs: data.y })
    })
  }
  return generatingProgram(program, { classes: model.nClasses })
}

/** The exact log density and posterior of a mixture class model, with `logProb(x)` the marginal when y is absent. */
export function mixtureTruth(model: MixtureModel): {
  readonly logProb: NonNullable<Truth['logProb']>
  readonly posterior: NonNullable<Truth['posterior']>
  readonly program?: GeneratingProgram
} {
  const joint = classLogJoint(model)
  const program = mixtureProgram(model)
  return {
    ...(program === undefined ? {} : { program }),
    logProb: (x, y) => {
      const j = joint(x)
      if (y === undefined) return logsumexp(j, 1)
      return reshape(takeAlongAxis(j, reshape(y, [-1, 1]), 1), [-1])
    },
    posterior: (x) => softmax(joint(x), 1),
  }
}

/**
 * A Monte Carlo Bayes error from fresh draws of the features (rounded to 4 decimals); the number of draws shrinks for
 * models with many components so the cost stays bounded.
 */
export function monteCarloBayesError(
  posterior: (x: Value) => Value,
  draw: (k: Key, n: number) => Value,
  k: Key,
  components = 1,
): number {
  const n = Math.round(Math.min(20_000, Math.max(2_000, 4e6 / components)))
  return Math.round(bayesErrorEstimate(posterior, draw(k, n)).value * 1e4) / 1e4
}

/**
 * A copy of a truth whose Bayes error is computed on first access (and cached), since Monte Carlo estimates cost more
 * than generating the data.
 */
export function withLazyBayesError(truth: Truth, compute: () => number): Truth {
  let cached: number | undefined
  const out: Record<string, unknown> = { ...truth }
  delete out['bayesError']
  Object.defineProperty(out, 'bayesError', {
    enumerable: true,
    get: () => {
      if (cached === undefined) cached = compute()
      return cached
    },
  })
  return out as Truth
}
