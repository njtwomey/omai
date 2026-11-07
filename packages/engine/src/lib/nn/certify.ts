/**
 * Certified robustness: guarantees that no perturbation within a radius changes a prediction, as opposed to the attacks
 * in `adversarial.ts`, which can only show that one does. Randomised smoothing (Cohen, Rosenfeld and Kolter 2019: the
 * vote of a classifier under Gaussian noise, a Clopper-Pearson lower bound on the top class's share, and the l2 radius
 * sigma Phi^-1(p) that follows, as a function and as a stepped algorithm), the closed form for a linear classifier,
 * interval bound propagation through a small network (Gowal et al. 2018), and the Lipschitz bound from the spectral
 * norms of the weights.
 */

import { MdmlError } from '../base/index.ts'
import { svdvals } from '../numerics/linalg/index.ts'
import { ndtr, ndtri } from '../numerics/special/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { proportionConfint } from '../stats/tests/index.ts'
import {
  abs,
  add,
  argmax,
  asTensor,
  fromStorage,
  item,
  matmul,
  mul,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import type { LogitsFunction } from './adversarial.ts'
import { Sequential } from './module/containers.ts'
import { Linear } from './module/layers.ts'
import type { Params } from './module/module.ts'

/** The radius sigma Phi^-1(p) a smoothed classifier is certified to when its top class has probability at least `p` (0 when p <= 1/2). */
function radiusOf(pLower: number, sigma: number): number {
  return pLower > 0.5 ? sigma * item(ndtri(pLower)) : 0
}

/**
 * The certified l2 radius of a smoothed classifier (Cohen et al. 2019, theorem 1 with the runner-up bounded by
 * 1 - p): sigma Phi^-1(pALower), and zero (abstain) when the lower bound on the top class's probability is not above a half.
 *
 * @param pALower - A lower confidence bound on the probability of the top class under the noise.
 * @param sigma - The noise's standard deviation.
 * @returns The radius.
 * @concept randomised-smoothing
 * @example
 * ```ts
 * import { certifiedRadius } from 'mdml-engine/nn'
 * certifiedRadius(0.5, 1) // => 0
 * ```
 */
export function certifiedRadius(pALower: number, sigma: number): number {
  return radiusOf(pALower, sigma)
}

/**
 * Counts of the classes a model predicts for `n` copies of an input under Gaussian noise of standard deviation
 * `sigma`, drawn in chunks of `batchSize`.
 *
 * @param model - Inputs to logits.
 * @param x - One input (a vector or any shape), without a batch axis.
 * @param sigma - Noise standard deviation.
 * @param n - Number of noisy copies.
 * @param key - The noise key.
 * @param options - `batchSize` (default 500) and the number of classes if the model might not predict them all.
 * @returns The counts per class.
 * @concept randomised-smoothing
 * @example
 * ```ts
 * import { smoothedCounts } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1]])
 * smoothedCounts(model, [10], 1, 50, key(0)) // => [50]
 * ```
 */
export function smoothedCounts(
  model: LogitsFunction,
  x: TensorLike,
  sigma: number,
  n: number,
  key: Key,
  options: { readonly batchSize?: number; readonly classes?: number } = {},
): number[] {
  const x0 = asTensor(x) as Tensor
  const batch = options.batchSize ?? 500
  const d = x0.data.length
  const counts: number[] = options.classes === undefined ? [] : new Array<number>(options.classes).fill(0)
  for (let done = 0, chunk = 0; done < n; chunk++) {
    const m = Math.min(batch, n - done)
    const noise = normal(foldIn(key, chunk), [m, ...x0.shape], { scale: sigma })
    const base = toFlat(x0)
    const noisy = Float64Array.from(toFlat(noise), (v, i) => v + (base[i % d] ?? 0))
    const labels = toFlat(argmax(model(fromStorage(noisy, [m, ...x0.shape], 'float64')), -1))
    for (const l of labels) {
      while (counts.length <= l) counts.push(0)
      counts[l] = (counts[l] ?? 0) + 1
    }
    done += m
  }
  return counts
}

/** Options of {@link randomisedSmoothing}. */
export interface SmoothingOptions {
  /** The noise standard deviation. */
  readonly sigma: number
  /** Samples used to pick the top class (default 100). */
  readonly nSelect?: number
  /** Samples used to bound its probability (default 1000). */
  readonly nEstimate?: number
  /** One minus the confidence of the certificate (default 0.001). */
  readonly alpha?: number
  readonly batchSize?: number
  readonly key: Key
}

/** A smoothed classifier's decision at one input. */
export interface SmoothedPrediction {
  /** The class, or -1 when the classifier abstains. */
  readonly prediction: number
  /** The certified l2 radius (0 when abstaining). */
  readonly radius: number
  /** The Clopper-Pearson lower bound on the top class's probability. */
  readonly pLower: number
  /** The votes of the estimation samples per class. */
  readonly counts: readonly number[]
}

function decide(
  selected: number,
  counts: readonly number[],
  n: number,
  sigma: number,
  alpha: number,
): SmoothedPrediction {
  const k = counts[selected] ?? 0
  // One-sided lower bound at confidence 1 - alpha.
  const pLower =
    proportionConfint(k, n, { method: 'clopper-pearson', alternative: 'greater', level: 1 - alpha }).interval?.[0] ?? 0
  const radius = radiusOf(pLower, sigma)
  return radius > 0 ? { prediction: selected, radius, pLower, counts } : { prediction: -1, radius: 0, pLower, counts }
}

const topClass = (counts: readonly number[]): number =>
  counts.reduce((best, c, i) => (c > (counts[best] ?? -1) ? i : best), 0)

/**
 * Certify one input by randomised smoothing (Cohen et al. 2019, CERTIFY): a first batch of noisy samples selects the
 * top class, a second, independent batch counts its votes, and a Clopper-Pearson lower bound on its probability
 * (confidence 1 - alpha) gives the radius sigma Phi^-1(pLower) within which the smoothed classifier's answer cannot
 * change, or abstains when the bound is not above a half.
 *
 * @param model - Inputs to logits (the base classifier).
 * @param x - One input without a batch axis.
 * @param options - sigma, sample counts, alpha and a key.
 * @returns The prediction (or -1), the certified radius, the bound and the votes.
 * @concept randomised-smoothing
 * @example
 * ```ts
 * import { randomisedSmoothing } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1]])
 * randomisedSmoothing(model, [3], { sigma: 0.5, key: key(0) }).prediction // => 0
 * ```
 */
export function randomisedSmoothing(
  model: LogitsFunction,
  x: TensorLike,
  options: SmoothingOptions,
): SmoothedPrediction {
  const n0 = options.nSelect ?? 100
  const n = options.nEstimate ?? 1000
  const batch = options.batchSize === undefined ? {} : { batchSize: options.batchSize }
  const selected = topClass(smoothedCounts(model, x, options.sigma, n0, foldIn(options.key, 0), batch))
  const counts = smoothedCounts(model, x, options.sigma, n, foldIn(options.key, 1), batch)
  return decide(selected, counts, n, options.sigma, options.alpha ?? 0.001)
}

/**
 * Certified accuracy against radius: for each radius, the share of inputs the smoothed classifier gets right with a
 * certified radius at least that large (abstentions count as wrong).
 *
 * @param model - Inputs to logits.
 * @param x - Inputs `[N, ...]`.
 * @param labels - Labels.
 * @param radii - Radii to report.
 * @param options - As {@link randomisedSmoothing}; the key is folded by input.
 * @returns Per-input predictions and the curve.
 * @concept randomised-smoothing
 * @example
 * ```ts
 * import { certifiedAccuracy } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1]])
 * certifiedAccuracy(model, [[5], [-5]], [0, 1], [0, 1], { sigma: 0.5, key: key(0) }).curve.map((c) => c.accuracy) // => [1, 1]
 * ```
 */
export function certifiedAccuracy(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  radii: readonly number[],
  options: SmoothingOptions,
): {
  readonly predictions: readonly SmoothedPrediction[]
  readonly curve: { readonly radius: number; readonly accuracy: number }[]
} {
  const xs = asTensor(x) as Tensor
  const n = xs.shape[0] ?? 0
  const d = n === 0 ? 0 : xs.data.length / n
  const y = toFlat(asTensor(labels))
  const predictions = Array.from({ length: n }, (_, i) =>
    randomisedSmoothing(
      model,
      fromStorage(Float64Array.from(xs.data.slice(i * d, (i + 1) * d)), xs.shape.slice(1), 'float64'),
      {
        ...options,
        key: foldIn(options.key, i),
      },
    ),
  )
  const curve = radii.map((radius) => ({
    radius,
    accuracy: predictions.filter((p, i) => p.prediction === y[i] && p.radius >= radius).length / Math.max(1, n),
  }))
  return { predictions, curve }
}

/**
 * The exact smoothed classifier of a binary linear classifier f(x) = sign(w.x + b): under Gaussian noise of standard
 * deviation sigma the probability of class 1 is Phi((w.x + b) / (sigma |w|)), so the certified radius is exactly the
 * distance to the hyperplane, |w.x + b| / |w|, whatever sigma is. The reference randomised smoothing is tested against.
 *
 * @param w - The weight vector.
 * @param b - The bias.
 * @param x - The input.
 * @param sigma - The noise standard deviation.
 * @returns The probability of the predicted class and the certified radius.
 * @concept randomised-smoothing
 * @example
 * ```ts
 * import { linearSmoothedRadius } from 'mdml-engine/nn'
 * linearSmoothedRadius([3, 4], 0, [1, 0], 1).radius // => 0.6
 * ```
 */
export function linearSmoothedRadius(
  w: readonly number[],
  b: number,
  x: readonly number[],
  sigma: number,
): { readonly probability: number; readonly radius: number } {
  const norm2 = Math.hypot(...w)
  const margin = w.reduce((s, wi, i) => s + wi * (x[i] ?? 0), b)
  const probability = item(ndtr(Math.abs(margin) / (sigma * norm2)))
  return { probability, radius: Math.abs(margin) / norm2 }
}

/** The paths of {@link smoothingAlgorithm}. */
export type SmoothingPath = 'sample/n' | 'vote/counts' | 'vote/bound' | 'certify/radius'

/** What a stepped smoothing certificate logs. */
export const smoothingContract: Contract<SmoothingState, SmoothingPath> = defineContract({
  'sample/n': scalar({ description: 'noisy samples drawn so far' }),
  'vote/counts': tensorSource({ role: 'state', description: 'votes per class' }),
  'vote/bound': scalar({ description: 'the Clopper-Pearson lower bound on the selected class’s probability' }),
  'certify/radius': scalar({ description: 'the certified radius (0 while the bound is not above 1/2)' }),
})

/** The state of a smoothing run: the selected class and the votes so far. */
export interface SmoothingState {
  readonly selected: number
  readonly counts: readonly number[]
  readonly n: number
  readonly pLower: number
  readonly radius: number
}

/**
 * Randomised smoothing as a stepped algorithm: the top class is selected from `nSelect` samples at the start, then each
 * step draws `perStep` more noisy samples and tightens the Clopper-Pearson bound and the radius, which the run logs
 * (the bound approaching the class's true probability from below as the samples come in).
 *
 * @param model - Inputs to logits.
 * @param x - One input.
 * @param options - sigma, `nSelect`, `alpha`, `perStep` (default 100) and `steps` (default 10); the run's key seeds the noise.
 * @returns The algorithm.
 * @concept randomised-smoothing
 * @example
 * ```ts
 * import { smoothingAlgorithm } from 'mdml-engine/nn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1]])
 * runAlgorithm(smoothingAlgorithm(model, [3], { sigma: 1, steps: 3 }), { maxSteps: 10 }).state.n // => 300
 * ```
 */
export function smoothingAlgorithm(
  model: LogitsFunction,
  x: TensorLike,
  options: {
    readonly sigma: number
    readonly nSelect?: number
    readonly alpha?: number
    readonly perStep?: number
    readonly steps?: number
  },
): Algorithm<SmoothingState, SmoothingPath> {
  const perStep = options.perStep ?? 100
  const steps = options.steps ?? 10
  const alpha = options.alpha ?? 0.001
  return {
    name: 'randomisedSmoothing',
    config: { sigma: options.sigma, nSelect: options.nSelect ?? 100, alpha, perStep, steps },
    init: (key) => {
      const selected = topClass(smoothedCounts(model, x, options.sigma, options.nSelect ?? 100, foldIn(key, 1 << 20)))
      return { selected, counts: [], n: 0, pLower: 0, radius: 0 }
    },
    step: (s, ctx) => {
      const more = smoothedCounts(model, x, options.sigma, perStep, ctx.key, {
        classes: Math.max(s.counts.length, s.selected + 1),
      })
      const counts = Array.from(
        { length: Math.max(more.length, s.counts.length) },
        (_, i) => (s.counts[i] ?? 0) + (more[i] ?? 0),
      )
      const n = s.n + perStep
      const d = decide(s.selected, counts, n, options.sigma, alpha)
      ctx.log('sample/n', n)
      ctx.log('vote/counts', counts)
      ctx.log('vote/bound', d.pLower)
      ctx.log('certify/radius', d.radius)
      return { selected: s.selected, counts, n, pLower: d.pLower, radius: d.radius }
    },
    done: (s) => (s.n >= steps * perStep ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SmoothingState,
    observe: (s) => ({ metrics: { pLower: s.pLower, radius: s.radius } }),
  }
}

/** Lower and upper bounds on a tensor, elementwise. */
export interface IntervalBounds {
  readonly lower: Tensor
  readonly upper: Tensor
}

const MONOTONE = new Set([
  'ReLU',
  'ReLU6',
  'LeakyReLU',
  'ELU',
  'CELU',
  'SELU',
  'Softplus',
  'Softsign',
  'Tanh',
  'Sigmoid',
  'Hardtanh',
  'Hardsigmoid',
])
const PASS = new Set(['Identity', 'Flatten', 'Dropout'])

function layersOf(model: Sequential, params: Params): { layer: Sequential['layers'][number]; params: Params }[] {
  return model.layers.map((layer, i) => ({ layer, params: (params[String(i)] ?? {}) as Params }))
}

const toT = (v: Value): Tensor => asTensor(v) as Tensor

/** Interval arithmetic through a linear layer: the centre maps through, the radius through the absolute weights. */
function linearBounds(layer: Linear, p: Params, b: IntervalBounds): IntervalBounds {
  const w = p['weight'] as Value
  const mid = mul(add(b.lower, b.upper), 0.5)
  const rad = mul(sub(b.upper, b.lower), 0.5)
  const outMid = layer.apply(p, mid)
  const outRad = matmul(rad, transpose(abs(w)))
  return { lower: toT(sub(outMid, outRad)), upper: toT(add(outMid, outRad)) }
}

/** The result of {@link intervalBoundPropagation}: bounds on the output and after each layer. */
export interface IbpResult {
  readonly output: IntervalBounds
  /** The bounds on the input of the network, then after each layer. */
  readonly layers: readonly IntervalBounds[]
}

/**
 * Interval bound propagation (Gowal et al. 2018): bounds on every layer's output for all inputs in a box, by interval
 * arithmetic through a `Sequential` of `Linear` layers and monotone activations (ReLU, tanh, sigmoid, leaky ReLU and
 * the like), Flatten, Identity and Dropout. Exact for a single linear layer, loose (and increasingly so with depth)
 * for a deeper network. Every output of the network for an input in the box lies within the bounds.
 *
 * @param model - The network.
 * @param lower - Lower corner of the input box `[N, ...]`.
 * @param upper - Upper corner.
 * @param params - Parameters (default the module's).
 * @returns The output bounds and the bounds after each layer.
 * @throws MdmlError with code `value` for a layer it cannot bound.
 * @concept interval-bound-propagation
 * @example
 * ```ts
 * import { intervalBoundPropagation, Sequential, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const net = new Sequential(new Linear(2, 3, { key: key(0) }))
 * intervalBoundPropagation(net, [[0, 0]], [[0, 0]]).output.lower.shape // => [1, 3]
 * ```
 */
export function intervalBoundPropagation(
  model: Sequential,
  lower: TensorLike,
  upper: TensorLike,
  params: Params = model.params,
): IbpResult {
  let b: IntervalBounds = { lower: toT(asTensor(lower)), upper: toT(asTensor(upper)) }
  const layers: IntervalBounds[] = [b]
  for (const { layer, params: p } of layersOf(model, params)) {
    if (layer instanceof Linear) b = linearBounds(layer, p, b)
    else if (MONOTONE.has(layer.kind))
      b = {
        lower: toT(layer.apply(p, b.lower as never) as Value),
        upper: toT(layer.apply(p, b.upper as never) as Value),
      }
    else if (PASS.has(layer.kind))
      b = {
        lower: toT(layer.apply(p, b.lower as never, { train: false }) as Value),
        upper: toT(layer.apply(p, b.upper as never, { train: false }) as Value),
      }
    else throw new MdmlError('value', `intervalBoundPropagation: cannot bound a ${layer.kind} layer`)
    layers.push(b)
  }
  return { output: b, layers }
}

/** The result of {@link ibpCertify}. */
export interface IbpCertificate {
  /** Whether the label is certified for every input within epsilon (linf) of the original. */
  readonly certified: readonly boolean[]
  /** The lower bound on the smallest margin (label logit minus another logit) over the box, per input. */
  readonly margin: readonly number[]
}

/**
 * Certify an l-infinity ball by interval bound propagation: propagate the box `[x - epsilon, x + epsilon]` to the last
 * layer's input and bound each margin (z_y - z_j) exactly over that box through the last layer (tighter than bounding
 * the logits one by one); the label is certified when every margin's lower bound is positive.
 *
 * @param model - A `Sequential` ending in a `Linear` layer.
 * @param x - Inputs `[N, ...]`.
 * @param labels - Their labels.
 * @param epsilon - The l-infinity radius.
 * @returns Which inputs are certified and the worst margin bound.
 * @concept interval-bound-propagation
 * @example
 * ```ts
 * import { ibpCertify, Sequential, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * ibpCertify(new Sequential(new Linear(2, 2, { key: key(0) })), [[0, 0]], [0], 0).certified.length // => 1
 * ```
 */
export function ibpCertify(model: Sequential, x: TensorLike, labels: TensorLike, epsilon: number): IbpCertificate {
  const x0 = asTensor(x)
  const last = model.layers[model.layers.length - 1]
  if (!(last instanceof Linear)) throw new MdmlError('value', 'ibpCertify: the network must end in a Linear layer')
  const head = new Sequential(...model.layers.slice(0, -1))
  const headParams: Record<string, unknown> = {}
  model.layers.slice(0, -1).forEach((_, i) => {
    headParams[String(i)] = model.params[String(i)] ?? {}
  })
  const lo = sub(x0, epsilon)
  const hi = add(x0, epsilon)
  const h =
    head.layers.length === 0
      ? { lower: toT(lo), upper: toT(hi) }
      : intervalBoundPropagation(head, lo, hi, headParams).output
  const lastParams = model.params[String(model.layers.length - 1)] as Params
  const w = lastParams['weight'] as Value
  const classes = w.shape[0] ?? 0
  const inWidth = w.shape[1] ?? 0
  const wFlat = toFlat(w)
  const bFlat =
    lastParams['bias'] === undefined ? new Array<number>(classes).fill(0) : toFlat(lastParams['bias'] as Value)
  const y = toFlat(asTensor(labels))
  const mid = toFlat(mul(add(h.lower, h.upper), 0.5))
  const rad = toFlat(mul(sub(h.upper, h.lower), 0.5))
  const margin = y.map((label, i) => {
    let worst = Number.POSITIVE_INFINITY
    for (let j = 0; j < classes; j++) {
      if (j === label) continue
      // The margin z_label - z_j is linear in the last layer's input: centre plus or minus the radius through |c|.
      let centre = (bFlat[label] ?? 0) - (bFlat[j] ?? 0)
      let spread = 0
      for (let k = 0; k < inWidth; k++) {
        const c = (wFlat[label * inWidth + k] ?? 0) - (wFlat[j * inWidth + k] ?? 0)
        centre += c * (mid[i * inWidth + k] ?? 0)
        spread += Math.abs(c) * (rad[i * inWidth + k] ?? 0)
      }
      worst = Math.min(worst, centre - spread)
    }
    return worst
  })
  return { certified: margin.map((m) => m > 0), margin }
}

const LIPSCHITZ: Readonly<Record<string, number>> = {
  ReLU: 1,
  ReLU6: 1,
  LeakyReLU: 1,
  Tanh: 1,
  Softsign: 1,
  Hardtanh: 1,
  Sigmoid: 0.25,
  Identity: 1,
  Flatten: 1,
  Dropout: 1,
}

/**
 * An upper bound on a network's l2 Lipschitz constant: the product of the spectral norms of its linear layers' weights
 * (their largest singular values) and of its activations' own constants (1 for ReLU and tanh, 1/4 for sigmoid; a leaky
 * ReLU is taken to have slope at most 1). Loose, but a guarantee.
 *
 * @param model - A `Sequential` of `Linear` layers and Lipschitz activations.
 * @param params - Parameters (default the module's).
 * @returns The bound.
 * @throws MdmlError with code `value` for a layer whose constant it does not know.
 * @concept lipschitz-certificates
 * @example
 * ```ts
 * import { lipschitzBound, Sequential, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * lipschitzBound(new Sequential(new Linear(2, 2, { key: key(0) }))) > 0 // => true
 * ```
 */
export function lipschitzBound(model: Sequential, params: Params = model.params): number {
  let bound = 1
  for (const { layer, params: p } of layersOf(model, params)) {
    if (layer instanceof Linear) bound *= toFlat(svdvals(p['weight'] as Value))[0] ?? 0
    else {
      const c = LIPSCHITZ[layer.kind]
      if (c === undefined) throw new MdmlError('value', `lipschitzBound: no constant for a ${layer.kind} layer`)
      bound *= c
    }
  }
  return bound
}

/**
 * The l2 radius certified by a Lipschitz bound: a logit difference of two classes changes by at most sqrt(2) L |delta|,
 * so a prediction with margin m (top logit minus the runner-up) holds within m / (sqrt(2) L).
 *
 * @param logits - `[N, C]` logits.
 * @param lipschitz - An upper bound on the network's l2 Lipschitz constant.
 * @returns The certified radius per input.
 * @concept lipschitz-certificates
 * @example
 * ```ts
 * import { lipschitzCertifiedRadius } from 'mdml-engine/nn'
 * lipschitzCertifiedRadius([[3, 1]], 1)[0]?.toFixed(4) // => '1.4142'
 * ```
 */
export function lipschitzCertifiedRadius(logits: TensorLike, lipschitz: number): number[] {
  const z = asTensor(logits)
  const c = z.shape[z.shape.length - 1] ?? 0
  const flat = toFlat(z)
  const n = flat.length / Math.max(1, c)
  return Array.from({ length: n }, (_, i) => {
    const row = flat.slice(i * c, (i + 1) * c).sort((a, b) => b - a)
    return ((row[0] ?? 0) - (row[1] ?? 0)) / (Math.SQRT2 * lipschitz)
  })
}
