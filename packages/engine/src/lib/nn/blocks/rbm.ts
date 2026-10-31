/**
 * Restricted Boltzmann machines: the model (binary or Gaussian visible units, binary hidden units) with its free
 * energy, block Gibbs sampling, the exact log partition function and model statistics by enumerating the hidden units
 * of a small machine, the contrastive-divergence surrogate loss, and training by CD-k or persistent CD as a stepped
 * `Algorithm` that puts the reconstruction error and the free energy on the tape.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, normal, subsample, uniform } from '../../random/index.ts'
import {
  add,
  asTensor,
  cast,
  exp,
  less,
  logsumexp,
  matmul,
  mean,
  mul,
  neg,
  reshape,
  sigmoid,
  softplus,
  stopGradient,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { type Algorithm, channels } from '../../trace/index.ts'
import { normalInit } from '../init.ts'
import { initKey, Module, type Params } from '../module/module.ts'

/** The kind of the visible units. */
export type RbmVisible = 'binary' | 'gaussian'

const val = (p: Params, name: string): Value => p[name] as Value

/**
 * The hidden units' activation probabilities given the visible ones, `sigmoid(v W + c)`.
 *
 * @param params - `weight` `[v, h]`, `vbias` `[v]` and `hbias` `[h]`.
 * @param v - The visible units `[B, v]`.
 * @returns The probabilities `[B, h]`.
 * @example
 * ```ts
 * import { rbmHiddenProbabilities } from 'mdml-engine/nn/blocks'
 * import { zeros, toArray } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * toArray(rbmHiddenProbabilities(params, [[1, 0]])) // => [[0.5]]
 * ```
 */
export function rbmHiddenProbabilities(params: Params, v: TensorLike): Value {
  return sigmoid(add(matmul(v, val(params, 'weight')), val(params, 'hbias')))
}

/**
 * The visible units' mean given the hidden ones: the activation probabilities `sigmoid(h W^T + b)` for binary units,
 * the mean `h W^T + b` of the unit-variance Gaussian for Gaussian units.
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param h - The hidden units `[B, h]`.
 * @param visible - The kind of the visible units (default binary).
 * @returns The means `[B, v]`.
 * @example
 * ```ts
 * import { rbmVisibleMean } from 'mdml-engine/nn/blocks'
 * import { zeros, toArray } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * toArray(rbmVisibleMean(params, [[1]], 'gaussian')) // => [[0, 0]]
 * ```
 */
export function rbmVisibleMean(params: Params, h: TensorLike, visible: RbmVisible = 'binary'): Value {
  const a = add(matmul(h, transpose(val(params, 'weight'))), val(params, 'vbias'))
  return visible === 'binary' ? sigmoid(a) : a
}

/**
 * The free energy of visible rows, `F(v) = -log sum_h exp(-E(v, h))`: `-b.v - sum softplus(v W + c)` for binary
 * visible units and `|v - b|^2 / 2 - sum softplus(v W + c)` for Gaussian ones (so `p(v) = exp(-F(v)) / Z`).
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param v - The visible units `[B, v]`.
 * @param visible - The kind of the visible units (default binary).
 * @returns The free energy per row `[B]`.
 * @example
 * ```ts
 * import { rbmFreeEnergy } from 'mdml-engine/nn/blocks'
 * import { zeros, item } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * item(rbmFreeEnergy(params, [[1, 0]])) // => -0.6931471805599453
 * ```
 */
export function rbmFreeEnergy(params: Params, v: TensorLike, visible: RbmVisible = 'binary'): Value {
  const x = asTensor(v)
  const hidden = sum(softplus(add(matmul(x, val(params, 'weight')), val(params, 'hbias'))), -1)
  const centred = sub(x, val(params, 'vbias'))
  const visibleTerm =
    visible === 'binary' ? neg(sum(mul(x, val(params, 'vbias')), -1)) : mul(sum(mul(centred, centred), -1), 0.5)
  return sub(visibleTerm, hidden)
}

/**
 * Sample binary units from their probabilities.
 *
 * @param key - The key.
 * @param probabilities - The probabilities (any shape).
 * @returns Zeros and ones of the same shape.
 * @example
 * ```ts
 * import { sampleBinary } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sampleBinary(key(0), [[1, 0]])) // => [[1, 0]]
 * ```
 */
export function sampleBinary(key: Key, probabilities: TensorLike): Value {
  const p = asTensor(probabilities)
  return cast(less(uniform(key, p.shape), p), 'float64')
}

/** The result of Gibbs sampling: the last samples and the probabilities they were drawn from. */
export interface GibbsResult {
  /** The visible sample `[B, v]` (binary samples, or the Gaussian mean plus unit noise). */
  readonly visible: Value
  /** The visible means (probabilities for binary units) `[B, v]`. */
  readonly visibleMean: Value
  /** The hidden probabilities given the final visible sample `[B, h]`. */
  readonly hiddenProbabilities: Value
}

/**
 * Block Gibbs sampling: `k` rounds of sampling all hidden units given the visible ones and then all visible units
 * given the hidden ones (the layers are conditionally independent given each other).
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param v0 - The starting visible units `[B, v]`.
 * @param k - The number of rounds (at least 1).
 * @param key - The key.
 * @param visible - The kind of the visible units (default binary).
 * @returns The last visible sample, its means and the hidden probabilities given it.
 * @example
 * ```ts
 * import { rbmGibbs } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * import { zeros } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * rbmGibbs(params, [[1, 0]], 3, key(0)).visible.shape // => [1, 2]
 * ```
 */
export function rbmGibbs(
  params: Params,
  v0: TensorLike,
  k: number,
  key: Key,
  visible: RbmVisible = 'binary',
): GibbsResult {
  if (!Number.isInteger(k) || k < 1) throw new MdmlError('value', `rbmGibbs: k must be a positive integer, got ${k}`)
  let v = asTensor(v0)
  let mean_ = v
  for (let i = 0; i < k; i++) {
    const kk = foldIn(key, i)
    const h = sampleBinary(foldIn(kk, 0), rbmHiddenProbabilities(params, v))
    mean_ = rbmVisibleMean(params, h, visible)
    v = visible === 'binary' ? sampleBinary(foldIn(kk, 1), mean_) : add(mean_, normal(foldIn(kk, 1), mean_.shape))
  }
  return { visible: v, visibleMean: mean_, hiddenProbabilities: rbmHiddenProbabilities(params, v) }
}

/**
 * The contrastive-divergence surrogate loss `mean F(v_data) - mean F(v_model)` with the model samples held fixed: its
 * gradient with respect to the parameters is minus the CD update (positive minus negative statistics), so any
 * optimiser can train an RBM with it.
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param data - The data rows `[B, v]`.
 * @param model - The negative samples `[B', v]` (the end of the Gibbs chains).
 * @param visible - The kind of the visible units (default binary).
 * @returns The scalar loss.
 * @example
 * ```ts
 * import { rbmContrastiveLoss } from 'mdml-engine/nn/blocks'
 * import { zeros, item } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * item(rbmContrastiveLoss(params, [[1, 0]], [[1, 0]])) // => 0
 * ```
 */
export function rbmContrastiveLoss(
  params: Params,
  data: TensorLike,
  model: TensorLike,
  visible: RbmVisible = 'binary',
): Value {
  return sub(mean(rbmFreeEnergy(params, data, visible)), mean(rbmFreeEnergy(params, stopGradient(model), visible)))
}

/** The largest number of hidden units whose configurations the exact functions enumerate. */
const MAX_ENUMERATED = 20

function hiddenConfigurations(h: number): Tensor {
  if (h > MAX_ENUMERATED) throw new MdmlError('value', `rbm: ${h} hidden units are too many to enumerate`)
  const rows = 2 ** h
  const out = new Float64Array(rows * h)
  for (let r = 0; r < rows; r++) for (let j = 0; j < h; j++) out[r * h + j] = (r >> j) & 1
  return tensor(Array.from({ length: rows }, (_, r) => Array.from(out.subarray(r * h, (r + 1) * h))))
}

/** `log sum_v exp(-E)` pieces over all hidden configurations: the log weight of each configuration. */
function hiddenLogWeights(params: Params, visible: RbmVisible, configs: Tensor): Value {
  const w = val(params, 'weight')
  const inputs = add(matmul(configs, transpose(w)), val(params, 'vbias'))
  const hb = sum(mul(configs, val(params, 'hbias')), -1)
  if (visible === 'binary') return add(hb, sum(softplus(inputs), -1))
  const wh = matmul(configs, transpose(w))
  return add(hb, add(mul(sum(mul(wh, wh), -1), 0.5), sum(mul(wh, val(params, 'vbias')), -1)))
}

/**
 * The exact log partition function of a small machine, by summing over the `2^h` hidden configurations (the visible
 * units are summed out in closed form). It is differentiable in the parameters, so the exact log-likelihood gradient is
 * available to check CD against.
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param visible - The kind of the visible units (default binary).
 * @returns The scalar `log Z`.
 * @example
 * ```ts
 * import { rbmLogPartition } from 'mdml-engine/nn/blocks'
 * import { zeros, item } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * item(rbmLogPartition(params)) // => 2.0794415416798357
 * ```
 */
export function rbmLogPartition(params: Params, visible: RbmVisible = 'binary'): Value {
  const h = (val(params, 'weight').shape[1] ?? 0) as number
  const z = logsumexp(hiddenLogWeights(params, visible, hiddenConfigurations(h)))
  return visible === 'binary' ? z : add(z, 0.5 * (val(params, 'weight').shape[0] ?? 0) * Math.log(2 * Math.PI))
}

/**
 * The mean log-likelihood of visible rows, `-mean F(v) - log Z`, exact for a small machine.
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param v - The visible rows `[B, v]`.
 * @param visible - The kind of the visible units (default binary).
 * @returns The scalar mean log-likelihood.
 * @example
 * ```ts
 * import { rbmLogLikelihood } from 'mdml-engine/nn/blocks'
 * import { zeros, item } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * item(rbmLogLikelihood(params, [[1, 0]])) // => -1.3862943611198906
 * ```
 */
export function rbmLogLikelihood(params: Params, v: TensorLike, visible: RbmVisible = 'binary'): Value {
  return sub(neg(mean(rbmFreeEnergy(params, v, visible))), rbmLogPartition(params, visible))
}

/** The expectations under the model that the negative phase of CD estimates. */
export interface RbmStatistics {
  /** `E[v h^T]` `[v, h]`. */
  readonly coupling: Value
  /** `E[v]` `[v]`. */
  readonly visible: Value
  /** `E[h]` `[h]`. */
  readonly hidden: Value
}

/**
 * The exact model expectations `E[v h^T]`, `E[v]` and `E[h]` of a small machine (by summing over the hidden
 * configurations). The log-likelihood gradient is the data statistics minus these, which is what CD-k estimates.
 *
 * @param params - The parameters (see {@link rbmHiddenProbabilities}).
 * @param visible - The kind of the visible units (default binary).
 * @returns The three expectations.
 * @example
 * ```ts
 * import { rbmModelStatistics } from 'mdml-engine/nn/blocks'
 * import { zeros, toArray } from 'mdml-engine/tensor'
 * const params = { weight: zeros([2, 1]), vbias: zeros([2]), hbias: zeros([1]) }
 * toArray(rbmModelStatistics(params).hidden) // => [0.5]
 * ```
 */
export function rbmModelStatistics(params: Params, visible: RbmVisible = 'binary'): RbmStatistics {
  const h = (val(params, 'weight').shape[1] ?? 0) as number
  const configs = hiddenConfigurations(h)
  const p = exp(sub(hiddenLogWeights(params, visible, configs), logsumexp(hiddenLogWeights(params, visible, configs))))
  const vMean = rbmVisibleMean(params, configs, visible)
  const weightedV = mul(vMean, column(p))
  return {
    coupling: matmul(transpose(weightedV), configs),
    visible: sum(weightedV, 0),
    hidden: sum(mul(configs, column(p)), 0),
  }
}

const column = (p: Value): Value => reshape(p, [-1, 1])

/** Options of {@link RBM}. */
export interface RbmOptions {
  readonly visible?: RbmVisible
  /** Standard deviation of the initial weights (default 0.01). */
  readonly std?: number
  readonly key?: Key
}

/**
 * A restricted Boltzmann machine: parameters `weight` `[v, h]`, `vbias` and `hbias`, with the weights drawn from a
 * small Gaussian and zero biases. Its forward pass is the encoder (the hidden probabilities); the free energy,
 * sampling, partition function and training are the functions of this file.
 *
 * @example
 * ```ts
 * import { RBM } from 'mdml-engine/nn/blocks'
 * import { key } from 'mdml-engine/random'
 * new RBM(6, 3, { key: key(0) }).forward([[1, 0, 1, 0, 1, 0]]).shape // => [1, 3]
 * ```
 */
export class RBM extends Module {
  readonly kind = 'RBM'
  readonly visibleUnits: number
  readonly hiddenUnits: number
  readonly visible: RbmVisible
  private readonly std: number

  constructor(visibleUnits: number, hiddenUnits: number, options: RbmOptions = {}) {
    super()
    this.visibleUnits = visibleUnits
    this.hiddenUnits = hiddenUnits
    this.visible = options.visible ?? 'binary'
    this.std = options.std ?? 0.01
    this.params = this.initParams(initKey('RBM', options.key))
  }

  override initParams(key: Key): Params {
    return {
      weight: normalInit(foldIn(key, 0), [this.visibleUnits, this.hiddenUnits], 0, this.std),
      vbias: zeros([this.visibleUnits]),
      hbias: zeros([this.hiddenUnits]),
    }
  }

  apply(params: Params, x: Value): Value {
    return rbmHiddenProbabilities(params, x)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.hiddenUnits]
  }

  override extraRepr(): string {
    return `visible=${this.visibleUnits}, hidden=${this.hiddenUnits}, units=${this.visible}`
  }
}

/** How the negative phase of contrastive divergence starts its chains. */
export type CdMode = 'cd' | 'pcd'

/** Options of {@link contrastiveDivergenceAlgorithm}. */
export interface ContrastiveDivergenceOptions {
  /** The number of hidden units. */
  readonly hidden: number
  /** `cd`: chains restart at the data each step; `pcd`: persistent chains continue (default `cd`). */
  readonly mode?: CdMode
  /** The number of Gibbs rounds per step (default 1). */
  readonly k?: number
  readonly learningRate?: number
  readonly momentum?: number
  readonly weightDecay?: number
  readonly batchSize?: number
  readonly visible?: RbmVisible
  /** Held-out rows, for the free-energy gap (held-out minus training; it grows with overfitting). */
  readonly validation?: TensorLike
  /** Record the exact log partition function each step (only for at most 20 hidden units). */
  readonly exactLogPartition?: boolean
  /** The stopping step budget for `done` (default: none, the run's `maxSteps`). */
  readonly steps?: number
  /** The standard deviation of the initial weights (default 0.01). */
  readonly std?: number
}

/** The state of {@link contrastiveDivergenceAlgorithm}. */
export interface CdState {
  readonly weight: Tensor
  readonly vbias: Tensor
  readonly hbias: Tensor
  readonly velocity: { readonly weight: Tensor; readonly vbias: Tensor; readonly hbias: Tensor }
  /** The persistent chains `[batch, v]`. */
  readonly chain: Tensor
  readonly reconstructionError: number
  readonly freeEnergy: number
}

/** The paths of {@link contrastiveDivergenceAlgorithm}. */
export type CdPath =
  | 'train/reconstructionError'
  | 'train/freeEnergy'
  | 'train/freeEnergyGap'
  | 'train/logPartition'
  | 'model/filters'
  | 'chain/samples'

const asParams = (s: CdState): Params => ({ weight: s.weight, vbias: s.vbias, hbias: s.hbias })

/**
 * Train an RBM by contrastive divergence (Hinton 2002) or persistent CD (Tieleman 2008), one minibatch per step: the
 * positive statistics of the batch minus the negative statistics of a Gibbs chain (restarted at the data for CD,
 * continued across steps for PCD), with momentum and weight decay. The reconstruction error, the free energy, the
 * free-energy gap to held-out rows and (for a tiny machine) the exact log partition function are on the tape, with the
 * weights as filters and the chain's samples.
 *
 * @param data - The training rows `[N, v]` (binary, or standardised for Gaussian visible units).
 * @param options - The hidden units, mode, `k`, step sizes, batch size and extra measurements.
 * @returns The algorithm; its state holds the parameters (see {@link rbmParamsOf}).
 * @reference Hinton (2002), Training products of experts by minimizing contrastive divergence.
 * @example
 * ```ts
 * import { contrastiveDivergenceAlgorithm } from 'mdml-engine/nn/blocks'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const rows = Array.from({ length: 20 }, (_, i) => (i % 2 === 0 ? [1, 1, 0, 0] : [0, 0, 1, 1]))
 * const a = contrastiveDivergenceAlgorithm(rows, { hidden: 2, batchSize: 10, learningRate: 0.2 })
 * runAlgorithm(a, { maxSteps: 3, seed: 0 }).steps // => 3
 * ```
 */
export function contrastiveDivergenceAlgorithm(
  data: TensorLike,
  options: ContrastiveDivergenceOptions,
): Algorithm<CdState> {
  const x = asTensor(data) as Tensor
  if (x.shape.length !== 2) throw new MdmlError('shape', 'contrastiveDivergenceAlgorithm: data must be [N, v]')
  const [n, nv] = x.shape as [number, number]
  const nh = options.hidden
  const mode = options.mode ?? 'cd'
  const k = options.k ?? 1
  const lr = options.learningRate ?? 0.1
  const momentum = options.momentum ?? 0
  const decay = options.weightDecay ?? 0
  const batch = Math.min(options.batchSize ?? n, n)
  const visible = options.visible ?? 'binary'
  const validation = options.validation === undefined ? undefined : asTensor(options.validation)
  const exact = options.exactLogPartition ?? false
  const std = options.std ?? 0.01
  return {
    name: 'rbm-contrastive-divergence',
    config: { mode, k, learningRate: lr, momentum, weightDecay: decay, batchSize: batch, hidden: nh, visible },
    schema: channels({
      metrics: {
        'train/reconstructionError': {
          description: 'mean squared reconstruction error of the batch',
          direction: 'lower',
        },
        'train/freeEnergy': { description: 'mean free energy of the batch', direction: 'lower' },
        'train/freeEnergyGap': { description: 'held-out minus training mean free energy' },
        'train/logPartition': { description: 'the exact log partition function (tiny machines)' },
      },
      tensors: {
        'model/filters': { role: 'param', description: 'the weights, one column per hidden unit', shape: [nv, nh] },
        'chain/samples': { role: 'state', description: "the chains' visible samples", shape: ['batch', nv] },
      },
    }),
    init: (key) => ({
      weight: normalInit(foldIn(key, 0), [nv, nh], 0, std),
      vbias: zeros([nv]),
      hbias: zeros([nh]),
      velocity: { weight: zeros([nv, nh]), vbias: zeros([nv]), hbias: zeros([nh]) },
      chain: take(x, subsample(foldIn(key, 1), n, batch)) as Tensor,
      reconstructionError: Number.NaN,
      freeEnergy: Number.NaN,
    }),
    step: (s, ctx) => {
      const params = asParams(s)
      const rows = subsample(foldIn(ctx.key, 0), n, batch)
      const v0 = take(x, rows)
      const start = mode === 'pcd' ? s.chain : (v0 as Tensor)
      const chain = rbmGibbs(params, start, k, foldIn(ctx.key, 1), visible)
      const h0 = rbmHiddenProbabilities(params, v0)
      const hk = chain.hiddenProbabilities
      const scale = (a: Value, m: number): Value => mul(a, 1 / m)
      const bPos = batch
      const bNeg = (start.shape[0] ?? 1) as number
      const gradW = sub(scale(matmul(transpose(v0), h0), bPos), scale(matmul(transpose(chain.visible), hk), bNeg))
      const gradV = sub(mean(v0, 0), mean(chain.visible, 0))
      const gradH = sub(mean(h0, 0), mean(hk, 0))
      const move = (name: 'weight' | 'vbias' | 'hbias', grad: Value, decayed: boolean): [Tensor, Tensor] => {
        const g = decayed ? sub(grad, mul(s[name], decay)) : grad
        const vel = add(mul(s.velocity[name], momentum), mul(g, lr)) as Tensor
        return [add(s[name], vel) as Tensor, vel]
      }
      const [weight, vW] = move('weight', gradW, true)
      const [vbias, vV] = move('vbias', gradV, false)
      const [hbias, vH] = move('hbias', gradH, false)
      const recon = rbmVisibleMean(params, h0, visible)
      const err = mean(mul(sub(v0, recon), sub(v0, recon)))
      return {
        weight,
        vbias,
        hbias,
        velocity: { weight: vW, vbias: vV, hbias: vH },
        chain: chain.visible as Tensor,
        reconstructionError: toFlat(err)[0] as number,
        freeEnergy: toFlat(mean(rbmFreeEnergy(params, v0, visible)))[0] as number,
      }
    },
    ...(options.steps === undefined
      ? {}
      : { done: (_s: CdState, t: number) => (t >= (options.steps ?? 0) ? ('completed' as const) : undefined) }),
    observe: (s) => {
      const params = asParams(s)
      const metrics: Record<string, number> = {
        'train/reconstructionError': s.reconstructionError,
        'train/freeEnergy': s.freeEnergy,
      }
      if (validation !== undefined) {
        const train = toFlat(mean(rbmFreeEnergy(params, x, visible)))[0] as number
        metrics['train/freeEnergyGap'] = (toFlat(mean(rbmFreeEnergy(params, validation, visible)))[0] as number) - train
      }
      if (exact) metrics['train/logPartition'] = toFlat(rbmLogPartition(params, visible))[0] as number
      return {
        metrics,
        tensors: {
          'model/filters': { role: 'param', value: s.weight },
          'chain/samples': { role: 'state', value: s.chain },
        },
      }
    },
    serialise: (s) => ({
      weight: toArray(s.weight) as number[][],
      vbias: toArray(s.vbias) as number[],
      hbias: toArray(s.hbias) as number[],
      velocity: {
        weight: toArray(s.velocity.weight) as number[][],
        vbias: toArray(s.velocity.vbias) as number[],
        hbias: toArray(s.velocity.hbias) as number[],
      },
      chain: toArray(s.chain) as number[][],
      reconstructionError: s.reconstructionError,
      freeEnergy: s.freeEnergy,
    }),
    deserialise: (j) => {
      const o = j as Record<string, unknown>
      const t = (v: unknown): Tensor => tensor(v as number[][])
      const vel = o['velocity'] as Record<string, unknown>
      return {
        weight: t(o['weight']),
        vbias: t(o['vbias']),
        hbias: t(o['hbias']),
        velocity: { weight: t(vel['weight']), vbias: t(vel['vbias']), hbias: t(vel['hbias']) },
        chain: t(o['chain']),
        reconstructionError: o['reconstructionError'] as number,
        freeEnergy: o['freeEnergy'] as number,
      }
    },
  }
}

/**
 * The parameters of an RBM from the state of {@link contrastiveDivergenceAlgorithm}, as the `Params` the functions of
 * this file and the {@link RBM} module take.
 *
 * @param state - The algorithm's state.
 * @returns `weight`, `vbias` and `hbias`.
 * @example
 * ```ts
 * import { rbmParamsOf } from 'mdml-engine/nn/blocks'
 * import { zeros } from 'mdml-engine/tensor'
 * const z = zeros([1])
 * Object.keys(rbmParamsOf({ weight: zeros([1, 1]), vbias: z, hbias: z, velocity: { weight: zeros([1, 1]), vbias: z, hbias: z }, chain: zeros([1, 1]), reconstructionError: 0, freeEnergy: 0 })) // => ['weight', 'vbias', 'hbias']
 * ```
 */
export function rbmParamsOf(state: CdState): Params {
  return asParams(state)
}
