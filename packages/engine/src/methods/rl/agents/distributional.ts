// Distributional value-based agents: C51, QR-DQN, IQN and Rainbow. They share DQN's replay machinery (n-step
// accumulation, uniform or prioritised replay, target networks, epsilon-greedy or noisy exploration); what differs is
// the head, which says how a network represents the return distribution, how Bellman targets are formed from it and
// what loss compares them.

import { customVjp } from '../../../lib/autodiff/index.ts'
import { argmaxOf, type Json, linspaceNumbers, WhiteboxError } from '../../../lib/base/index.ts'
import { ndtr, ndtri } from '../../../lib/numerics/special/index.ts'
import { incrementalUpdate, linearSchedule, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../../lib/random/index.ts'
import {
  type Agent,
  bufferAdd,
  bufferSample,
  nStepAdd,
  nStepBuffer,
  prioritizedAdd,
  prioritizedReplay,
  prioritizedSample,
  prioritizedUpdate,
  replayBuffer,
  type Transition,
} from '../../../lib/rl/index.ts'
import {
  add,
  argmax,
  asTensor,
  broadcastTo,
  cast,
  concat,
  cos,
  exp,
  expandDims,
  fromStorage,
  item,
  logSoftmax,
  matmul,
  mean,
  mul,
  neg,
  oneHot,
  relu,
  repeat,
  reshape,
  softmax,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../../lib/tensor/index.ts'
import {
  actionCount,
  gradientStepWithTerms,
  observationSize,
  optimiserOf,
  rowsTensor,
  type Spaces,
  vectorTensor,
} from './_common.ts'
import {
  type DqnOptions,
  type DqnState,
  type QNetwork,
  type QParams,
  qLayer,
  qLayerInit,
  qNetwork,
} from './value-based.ts'

// ------------------------------------------------------------------------------------- projection and losses

/**
 * The fixed support of a categorical return distribution: `atoms` evenly spaced values from `vMin` to `vMax` (as
 * `numpy.linspace`).
 *
 * @param vMin - The smallest return.
 * @param vMax - The largest return.
 * @param atoms - The number of atoms (at least 2).
 * @returns The support.
 * @throws WhiteboxError with code `value` for fewer than two atoms or an empty range.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { categoricalSupport } from 'whitebox-engine/methods/rl/agents'
 * categoricalSupport(-1, 1, 5) // => [-1, -0.5, 0, 0.5, 1]
 * ```
 */
export function categoricalSupport(vMin: number, vMax: number, atoms: number): number[] {
  if (!(Number.isInteger(atoms) && atoms >= 2)) throw new WhiteboxError('value', 'categoricalSupport: need two atoms')
  if (!(vMax > vMin)) throw new WhiteboxError('value', 'categoricalSupport: need vMin < vMax')
  return linspaceNumbers(vMin, vMax, atoms)
}

/**
 * The categorical projection of C51 (Bellemare, Dabney and Munos 2017, Algorithm 1): each atom `z_j` of the next-state
 * distribution moves to `r + d z_j` (clipped to `[vMin, vMax]`) and its probability is split between the two nearest
 * atoms of the fixed support in proportion to closeness, so the projected distribution lives on the same support.
 * The mean moves to `r + d E[Z]` whenever no atom is clipped.
 *
 * @param probabilities - Next-state distributions, one row per sample, on the support of `vMin`, `vMax`.
 * @param rewards - The rewards (n-step sums for n-step targets).
 * @param discounts - The bootstrap discounts (`gamma^n`, or 0 at terminal states).
 * @param vMin - The smallest return of the support.
 * @param vMax - The largest return of the support.
 * @returns The projected distributions.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { categoricalProjection } from 'whitebox-engine/methods/rl/agents'
 * categoricalProjection([[0, 1, 0]], [0.5], [0.5], -1, 1) // => [[0, 0.5, 0.5]]
 * ```
 */
export function categoricalProjection(
  probabilities: readonly (readonly number[])[],
  rewards: readonly number[],
  discounts: readonly number[],
  vMin: number,
  vMax: number,
): number[][] {
  const n = probabilities[0]?.length ?? 0
  const z = categoricalSupport(vMin, vMax, n)
  const dz = (vMax - vMin) / (n - 1)
  return probabilities.map((p, b) => {
    const m = new Array<number>(n).fill(0)
    const r = rewards[b] ?? 0
    const d = discounts[b] ?? 0
    for (let j = 0; j < n; j++) {
      const tz = Math.min(vMax, Math.max(vMin, r + d * (z[j] ?? 0)))
      const pos = Math.min(n - 1, Math.max(0, (tz - vMin) / dz))
      const l = Math.floor(pos)
      const u = Math.ceil(pos)
      const pj = p[j] ?? 0
      if (l === u) m[l] = (m[l] ?? 0) + pj
      else {
        m[l] = (m[l] ?? 0) + pj * (u - pos)
        m[u] = (m[u] ?? 0) + pj * (pos - l)
      }
    }
    return m
  })
}

/** Options of {@link quantileHuberLoss}. */
export interface QuantileHuberOptions {
  /** The Huber threshold `kappa` (default 1; 0 gives the plain quantile regression loss). */
  readonly kappa?: number
  /** `'mean'` over samples (default), `'sum'`, or `'none'` for one loss per sample. */
  readonly reduction?: 'mean' | 'sum' | 'none'
  /** Per-sample weights `[B]` (importance weights of prioritised replay). */
  readonly weight?: TensorLike
}

/**
 * The quantile Huber loss of QR-DQN and IQN (Dabney et al. 2018, eq. 10; 2018b, eq. 3): with pairwise errors
 * `u_ij = T_j - theta_i` between target samples and predicted quantiles at fractions `tau_i`,
 * `rho(u) = |tau_i - 1{u < 0}| L_kappa(u) / kappa`, summed over `i` and averaged over `j` for each sample.
 *
 * @param quantiles - Predicted quantiles `[B, N]`.
 * @param targets - Target samples `[B, M]` (constants).
 * @param taus - The fractions of the predicted quantiles, `[N]` or `[B, N]`.
 * @param options - `kappa`, the reduction and weights.
 * @returns The loss (or per-sample losses `[B]`), differentiable in the quantiles.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { quantileHuberLoss } from 'whitebox-engine/methods/rl/agents'
 * import { item } from 'whitebox-engine/tensor'
 * item(quantileHuberLoss([[0]], [[2]], [0.25], { kappa: 1 })) // => 0.375
 * ```
 */
export function quantileHuberLoss(
  quantiles: TensorLike,
  targets: TensorLike,
  taus: TensorLike,
  options: QuantileHuberOptions = {},
): Value {
  const theta = asFloat(quantiles)
  const t = asFloat(targets)
  const fractions = broadcastTo(asFloat(taus), theta.shape)
  let perSample = quantileHuberKernel(theta, t, fractions, scalarTensor(options.kappa ?? 1))
  if (options.weight !== undefined) perSample = mul(perSample, options.weight)
  const reduction = options.reduction ?? 'mean'
  return reduction === 'none' ? perSample : reduction === 'sum' ? sum(perSample) : mean(perSample)
}

const asFloat = (x: TensorLike): Value => cast(asTensor(x), 'float64')
const scalarTensor = (x: number): Tensor => fromStorage(Float64Array.of(x), [], 'float64')

// The per-sample quantile Huber loss `[B]` of quantiles `[B, N]`, targets `[B, M]` and fractions `[B, N]` as one
// kernel with its reverse-mode rule (the pairwise `[B, N, M]` terms never become tensors).
function quantileHuberTerms(
  theta: Float64Array,
  target: Float64Array,
  tau: Float64Array,
  B: number,
  N: number,
  M: number,
  kappa: number,
  grads?: { readonly theta: Float64Array; readonly target: Float64Array; readonly tau: Float64Array },
): Float64Array {
  const out = new Float64Array(B)
  for (let b = 0; b < B; b++) {
    let total = 0
    for (let i = 0; i < N; i++) {
      const th = theta[b * N + i] ?? 0
      const ti = tau[b * N + i] ?? 0
      for (let j = 0; j < M; j++) {
        const u = (target[b * M + j] ?? 0) - th
        const a = Math.abs(u)
        const huber = kappa === 0 ? a : a <= kappa ? (0.5 * u * u) / kappa : a - 0.5 * kappa
        const slope = kappa === 0 ? Math.sign(u) : a <= kappa ? u / kappa : Math.sign(u)
        const w = u < 0 ? 1 - ti : ti
        total += w * huber
        if (grads !== undefined) {
          grads.theta[b * N + i] = (grads.theta[b * N + i] ?? 0) - (w * slope) / M
          grads.target[b * M + j] = (grads.target[b * M + j] ?? 0) + (w * slope) / M
          grads.tau[b * N + i] = (grads.tau[b * N + i] ?? 0) + ((u < 0 ? -1 : 1) * huber) / M
        }
      }
    }
    out[b] = total / M
  }
  return out
}

const flat = (x: Value): Float64Array => Float64Array.from(toFlat(x))

const quantileHuberKernel = customVjp({
  name: 'quantileHuberLoss',
  forward: (theta, target, tau, kappa) => {
    const [B = 0, N = 0] = theta.shape
    const M = target.shape[1] ?? 0
    const out = quantileHuberTerms(flat(theta), flat(target), flat(tau), B, N, M, item(kappa))
    return fromStorage(out, [B], 'float64')
  },
  vjp: (g, inputs) => {
    const gb = expandDims(g, -1)
    return [...quantileHuberPartials(inputs).map((p) => mul(gb, p)), null]
  },
  jvp: (inputs, tangents) => {
    let out: Value | null = null
    for (const [i, p] of quantileHuberPartials(inputs).entries()) {
      const t = tangents[i]
      if (t === null || t === undefined) continue
      const term = sum(mul(p, t), -1)
      out = out === null ? term : add(out, term)
    }
    return out
  },
})

// The per-sample partial derivatives `[B, N]`, `[B, M]`, `[B, N]` in quantiles, targets and fractions.
function quantileHuberPartials(inputs: readonly Value[]): Tensor[] {
  const [theta, target, tau, kappa] = inputs as [Value, Value, Value, Value]
  const [B = 0, N = 0] = theta.shape
  const M = target.shape[1] ?? 0
  const grads = { theta: new Float64Array(B * N), target: new Float64Array(B * M), tau: new Float64Array(B * N) }
  quantileHuberTerms(flat(theta), flat(target), flat(tau), B, N, M, item(kappa), grads)
  return [
    fromStorage(grads.theta, [B, N], 'float64'),
    fromStorage(grads.target, [B, M], 'float64'),
    fromStorage(grads.tau, [B, N], 'float64'),
  ]
}

/** A risk distortion of quantile fractions for IQN's risk-sensitive policies (Dabney et al. 2018b, section 3.1). */
export type RiskDistortion = 'neutral' | { readonly kind: 'cvar' | 'wang' | 'cpw' | 'pow'; readonly eta: number }

/**
 * A distortion `beta: [0, 1] -> [0, 1]` of quantile fractions: acting on `E_{tau ~ U}[Z_{beta(tau)}]` gives a
 * distortion risk measure. CVaR (`eta tau`, risk-averse for `eta < 1`), Wang (`Phi(Phi^-1(tau) + eta)`, averse for
 * `eta < 0`), CPW (`tau^eta / (tau^eta + (1 - tau)^eta)^(1/eta)`) and the power family (averse for `eta < 0`).
 *
 * @param distortion - The distortion.
 * @returns The map over arrays of fractions.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { riskDistortion } from 'whitebox-engine/methods/rl/agents'
 * riskDistortion({ kind: 'cvar', eta: 0.25 })([0.5, 1]) // => [0.125, 0.25]
 * ```
 */
export function riskDistortion(distortion: RiskDistortion): (taus: readonly number[]) => number[] {
  if (distortion === 'neutral') return (taus) => [...taus]
  const eta = distortion.eta
  switch (distortion.kind) {
    case 'cvar':
      return (taus) => taus.map((t) => eta * t)
    case 'wang':
      return (taus) => toFlat(ndtr(add(ndtri(vectorTensor(taus)), eta)))
    case 'cpw':
      return (taus) => taus.map((t) => t ** eta / (t ** eta + (1 - t) ** eta) ** (1 / eta))
    default: {
      const e = 1 / (1 + Math.abs(eta))
      return (taus) => taus.map((t) => (eta >= 0 ? t ** e : 1 - (1 - t) ** e))
    }
  }
}

// --------------------------------------------------------------------------------------------- IQN network

/** Options of {@link implicitQuantileNetwork}. */
export interface ImplicitQuantileNetworkOptions {
  /** Hidden widths of the state embedding `psi` (default `[64]`); the last is the embedding width. */
  readonly hidden?: readonly number[]
  /** Cosine basis size `n` of the fraction embedding `phi` (default 64, as in the paper). */
  readonly embedding?: number
}

/**
 * An implicit quantile network as a pure function of its parameters: `apply(params, x [B, d], taus [B, N])` returns the
 * quantiles `Z_tau(x, a)` as `[B, N, nActions]`.
 *
 * @concept distributional-rl
 */
export interface ImplicitQuantileNetwork {
  readonly params: QParams
  readonly nActions: number
  apply(params: QParams, x: Value, taus: TensorLike): Value
}

/**
 * Build the network of IQN (Dabney et al. 2018b): a state embedding `psi(x)` (an MLP with ReLUs), a fraction embedding
 * `phi_j(tau) = ReLU(sum_{i<n} cos(pi i tau) w_ij + b_j)`, their elementwise product, then a hidden layer and one
 * output per action.
 *
 * @param inputSize - The observation size.
 * @param nActions - The number of actions.
 * @param options - Hidden widths and the cosine basis size.
 * @param key - The initialisation key.
 * @returns The network.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { implicitQuantileNetwork } from 'whitebox-engine/methods/rl/agents'
 * import { key } from 'whitebox-engine/random'
 * import { tensor } from 'whitebox-engine/tensor'
 * const net = implicitQuantileNetwork(3, 2, { hidden: [8], embedding: 4 }, key(0))
 * net.apply(net.params, tensor([[1, 0, 0]]), tensor([[0.1, 0.5, 0.9]])).shape // => [1, 3, 2]
 * ```
 */
export function implicitQuantileNetwork(
  inputSize: number,
  nActions: number,
  options: ImplicitQuantileNetworkOptions,
  key: Key,
): ImplicitQuantileNetwork {
  const hidden = options.hidden ?? [64]
  const n = options.embedding ?? 64
  const params: Record<string, Record<string, Tensor>> = {}
  let width = inputSize
  for (const [i, h] of hidden.entries()) {
    params[`h${i}`] = qLayerInit(foldIn(key, i), width, h, false, 0)
    width = h
  }
  params['embedding'] = qLayerInit(foldIn(key, 100), n, width, false, 0)
  params['merge'] = qLayerInit(foldIn(key, 101), width, width, false, 0)
  params['out'] = qLayerInit(foldIn(key, 102), width, nActions, false, 0)
  const basis = rowsTensor([Array.from({ length: n }, (_, i) => Math.PI * i)], n) // [1, n]
  // Rows are flattened to [B N, .] so that every layer is a plain matrix product.
  const apply = (p: QParams, x: Value, taus: TensorLike): Value => {
    const t = asFloat(taus)
    const [B = 1, N = 1] = t.shape
    let psi = x
    for (let i = 0; i < hidden.length; i++) psi = relu(qLayer(p[`h${i}`] ?? {}, psi))
    const features = cos(matmul(reshape(t, [B * N, 1]), basis)) // [B N, n]
    const phi = relu(qLayer(p['embedding'] ?? {}, features)) // [B N, h]
    const merged = mul(repeat(psi, N, 0), phi)
    const z = qLayer(p['out'] ?? {}, relu(qLayer(p['merge'] ?? {}, merged)))
    return reshape(z, [B, N, nActions])
  }
  return { params, nActions, apply }
}

// ------------------------------------------------------------------------------------------- the shared agent

type Obs = readonly number[]

/** A replay minibatch as the heads need it. */
interface Batch {
  readonly obs: Tensor
  readonly next: Tensor
  /** One-hot actions `[B, nA]`. */
  readonly mask: Value
  readonly rewards: readonly number[]
  /** `gamma^k`, or 0 when the episode terminated. */
  readonly discounts: readonly number[]
  readonly size: number
}

/** How an agent represents, targets and fits its return distribution. */
interface Head {
  readonly params: QParams
  /** Whether acting draws randomness from the head (noisy layers, sampled fractions). */
  readonly stochastic: boolean
  /** Action values `[B, nA]`; a key draws the head's randomness, none gives the deterministic readout. */
  q(params: QParams, x: Value, key?: Key): Value
  /** Per-sample losses of a batch (targets computed outside the gradient) and what to report about them. */
  prepare(
    online: QParams,
    target: QParams,
    batch: Batch,
    key: Key,
  ): {
    /** Per-sample losses `[B]` and the chosen actions' values `[B]` (from the same forward pass). */
    readonly terms: (params: QParams) => { readonly loss: Value; readonly q: Value }
    readonly targetMean: readonly number[]
    /** Subtracted from the per-sample losses to give priorities (C51's target entropy, turning CE into KL). */
    readonly priorityOffset?: readonly number[]
  }
}

/** The state of a distributional agent (DQN's state: networks, optimiser state, replay, counters, statistics). */
export type DistributionalState = DqnState

const rowsOf = (x: Value, width: number): number[][] => {
  const flat = toFlat(x)
  return Array.from({ length: flat.length / width }, (_, i) => flat.slice(i * width, (i + 1) * width))
}

// The chosen action's row of `[B, nA, N]` (atoms last) or of `[B, N, nA]` (actions last), for one-hot `mask [B, nA]`.
// (Batched matrix products: the CPU backend's broadcasting elementwise products are much slower.)
const selectAtoms = (x: Value, mask: Value): Value => {
  const [b = 1, , n = 1] = x.shape
  return reshape(matmul(expandDims(mask, 1), x), [b, n])
}
const selectLast = (x: Value, mask: Value): Value => {
  const [b = 1, n = 1] = x.shape
  return reshape(matmul(x, expandDims(mask, -1)), [b, n])
}

function distributionalAgent(
  name: string,
  env: Spaces,
  options: DqnOptions,
  head: Head,
  extraConfig: Readonly<Record<string, Json>>,
): Agent<DistributionalState, Obs, number> {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, name)
  const gamma = options.gamma ?? 0.99
  const batchSize = options.batchSize ?? 32
  const learningStarts = options.learningStarts ?? 100
  const trainFrequency = options.trainFrequency ?? 1
  const gradientSteps = options.gradientSteps ?? 1
  const targetInterval = options.targetUpdateInterval ?? 250
  const tau = options.tau ?? 1
  const opt = optimiserOf(options)
  const epsilon = linearSchedule(
    options.epsilonStart ?? 1,
    options.epsilonEnd ?? 0.05,
    options.explorationSteps ?? 1000,
  )
  const noisy = options.noisy ?? false
  const per =
    options.prioritized === undefined || options.prioritized === false
      ? undefined
      : options.prioritized === true
        ? {}
        : options.prioritized
  const beta0 = per?.beta ?? 0.4
  const betaSteps = per?.betaSteps ?? 10_000
  const n = options.nStep ?? 1
  const capacity = options.bufferSize ?? 10_000
  const qOf = (p: QParams, obs: Obs, k?: Key): number[] => toFlat(head.q(p, rowsTensor([obs], d), k))

  const update = (s: DistributionalState, k: Key): DistributionalState => {
    const beta = Math.min(1, beta0 + (1 - beta0) * (s.steps / betaSteps))
    const sample =
      s.prioritized !== undefined
        ? prioritizedSample(s.prioritized, foldIn(k, 0), batchSize, beta)
        : { ...bufferSample(s.buffer, foldIn(k, 0), batchSize), weights: undefined }
    const items = sample.items
    const batch: Batch = {
      obs: rowsTensor(
        items.map((t) => t.observation),
        d,
      ),
      next: rowsTensor(
        items.map((t) => t.nextObservation),
        d,
      ),
      mask: oneHot(vectorTensor(items.map((t) => t.action)), nA),
      rewards: items.map((t) => t.reward),
      discounts: items.map((t) => (t.terminated ? 0 : t.discount)),
      size: items.length,
    }
    const prep = head.prepare(s.params, s.target, batch, foldIn(k, 1))
    const weights = sample.weights === undefined ? undefined : vectorTensor(sample.weights)
    const lossFn = (p: Tree): readonly [Value, Value] => {
      const { loss, q } = prep.terms(p as QParams)
      return [mean(weights === undefined ? loss : mul(loss, weights)), concat([loss, q])]
    }
    const step = gradientStepWithTerms(lossFn, s.params, opt, s.optState)
    const B = batch.size
    const losses = step.terms.slice(0, B)
    const q = step.terms.slice(B)
    let meanAbs = 0
    let meanQ = 0
    for (const [i, qi] of q.entries()) {
      meanAbs += Math.abs((prep.targetMean[i] ?? 0) - qi) / q.length
      meanQ += qi / q.length
    }
    const priorities = losses.map((t, i) => Math.max(0, t - (prep.priorityOffset?.[i] ?? 0)))
    return {
      ...s,
      params: step.params as QParams,
      optState: step.optState,
      ...(s.prioritized === undefined
        ? {}
        : { prioritized: prioritizedUpdate(s.prioritized, sample.slots, priorities) }),
      updates: s.updates + 1,
      loss: step.loss,
      tdError: meanAbs,
      qValue: meanQ,
    }
  }

  return {
    name,
    config: {
      hidden: [...(options.hidden ?? [64, 64])],
      gamma,
      batchSize,
      learningStarts,
      targetUpdateInterval: targetInterval,
      tau,
      double: options.double ?? false,
      dueling: options.dueling ?? false,
      prioritized: per !== undefined,
      nStep: n,
      noisy,
      ...extraConfig,
    } as Json,
    metrics: {
      epsilon: { description: 'exploration rate' },
      loss: { description: 'distributional loss of the last update', direction: 'lower' },
      tdError: { description: 'mean absolute error between the mean target and Q', direction: 'lower' },
      qValue: { description: 'mean Q value (mean of the return distribution) of the last batch' },
    },
    init: () => ({
      params: head.params,
      target: head.params,
      optState: opt.init(head.params),
      buffer: replayBuffer(capacity),
      ...(per === undefined
        ? {}
        : { prioritized: prioritizedReplay(capacity, per.alpha === undefined ? {} : { alpha: per.alpha }) }),
      nstep: nStepBuffer(n, gamma),
      steps: 0,
      updates: 0,
      loss: 0,
      tdError: 0,
      qValue: 0,
    }),
    act: (s, obs, k) => {
      if (noisy) return argmaxOf(qOf(s.params, obs, foldIn(k, 1)))
      if ((uniform(foldIn(k, 0), []).data[0] ?? 1) < epsilon(s.steps))
        return Math.min(nA - 1, Math.floor((uniform(foldIn(k, 2), []).data[0] ?? 0) * nA))
      return argmaxOf(qOf(s.params, obs, head.stochastic ? foldIn(k, 1) : undefined))
    },
    greedy: (s, obs) => argmaxOf(qOf(s.params, obs)),
    learn: (s, t: Transition<Obs, number>, k) => {
      const acc = nStepAdd(s.nstep, t)
      let buffer = s.buffer
      let prioritized = s.prioritized
      for (const ready of acc.ready) {
        if (prioritized !== undefined) prioritized = prioritizedAdd(prioritized, ready)
        else buffer = bufferAdd(buffer, ready)
      }
      let next: DistributionalState = {
        ...s,
        nstep: acc.buffer,
        buffer,
        ...(prioritized === undefined ? {} : { prioritized }),
        steps: s.steps + 1,
      }
      const size = prioritized?.buffer.size ?? buffer.size
      if (next.steps >= learningStarts && next.steps % trainFrequency === 0 && size >= batchSize) {
        for (let g = 0; g < gradientSteps; g++) next = update(next, foldIn(k, g))
      }
      if (tau < 1) next = { ...next, target: incrementalUpdate(next.params, next.target, tau) as QParams }
      else if (next.steps % targetInterval === 0) next = { ...next, target: next.params }
      return next
    },
    stats: (s) => ({ epsilon: noisy ? 0 : epsilon(s.steps), loss: s.loss, tdError: s.tdError, qValue: s.qValue }),
  }
}

// --------------------------------------------------------------------------------------------------------- C51

/** Options of {@link c51} (and {@link rainbow}). */
export interface C51Options extends DqnOptions {
  /** Atoms of the support (default 51). */
  readonly atoms?: number
  /** The support's range (default -10 to 10). */
  readonly vMin?: number
  readonly vMax?: number
}

function categoricalHead(env: Spaces, options: C51Options): { head: Head; net: QNetwork } {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'c51')
  const atoms = options.atoms ?? 51
  const vMin = options.vMin ?? -10
  const vMax = options.vMax ?? 10
  const support = categoricalSupport(vMin, vMax, atoms)
  const z = vectorTensor(support)
  const net = qNetwork(d, nA, { ...options, atoms }, options.key ?? makeKey(0))
  const noisy = options.noisy ?? false
  const probs = (p: QParams, x: Value, k?: Key): Value => softmax(net.apply(p, x, k), -1)
  const q = (p: QParams, x: Value, k?: Key): Value => matmul(probs(p, x, k), z)
  const head: Head = {
    params: net.params,
    stochastic: noisy,
    q,
    prepare: (online, target, batch, k) => {
      const nk = (j: number): Key | undefined => (noisy ? foldIn(k, j) : undefined)
      const pNext = probs(target, batch.next, nk(0))
      const qNext = options.double === true ? q(online, batch.next, nk(1)) : matmul(pNext, z)
      const best = oneHot(argmax(qNext, -1), nA)
      const m = categoricalProjection(
        rowsOf(selectAtoms(pNext, best), atoms),
        batch.rewards,
        batch.discounts,
        vMin,
        vMax,
      )
      const targetT = rowsTensor(m, atoms)
      const entropy = m.map((row) => {
        let h = 0
        for (const pi of row) if (pi > 0) h -= pi * Math.log(pi)
        return h
      })
      const targetMean = m.map((row) => {
        let s = 0
        for (const [j, pi] of row.entries()) s += pi * (support[j] ?? 0)
        return s
      })
      const noise = nk(2)
      return {
        terms: (p) => {
          const logp = logSoftmax(selectAtoms(net.apply(p, batch.obs, noise), batch.mask), -1)
          return { loss: neg(sum(mul(targetT, logp), -1)), q: matmul(exp(logp), z) }
        },
        targetMean,
        priorityOffset: entropy,
      }
    },
  }
  return { head, net }
}

/**
 * C51 (Bellemare, Dabney and Munos 2017): the network outputs a categorical distribution over `atoms` fixed returns
 * for each action; the target is the {@link categoricalProjection} of `r + gamma^n Z(s', a*)` with
 * `a* = argmax_a E[Z(s', a)]` under the target network (the online one with `double`), and the loss is the
 * cross-entropy to it. Every DQN variant applies (double, duelling, prioritised replay with KL priorities, n-step,
 * noisy networks).
 *
 * @param env - The environment's spaces (vector observations, discrete actions).
 * @param options - DQN's options plus the support.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { c51 } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * c51(cartPole.create(), { atoms: 11, vMin: 0, vMax: 100 }).name // => 'c51'
 * ```
 */
export function c51(env: Spaces, options: C51Options = {}): Agent<DistributionalState, Obs, number> {
  const { head } = categoricalHead(env, options)
  return distributionalAgent('c51', env, options, head, {
    atoms: options.atoms ?? 51,
    vMin: options.vMin ?? -10,
    vMax: options.vMax ?? 10,
  })
}

/** Options of {@link rainbow}: C51's, with Rainbow's components on by default. */
export type RainbowOptions = C51Options

/**
 * Rainbow (Hessel et al. 2018): C51 with double Q-learning, duelling heads, prioritised replay (priorities from the KL
 * loss, exponent 0.5, importance exponent annealed from 0.4 to 1), 3-step returns and noisy networks
 * (`sigma0 = 0.5`) instead of epsilon-greedy exploration. Each component can be switched off for ablations.
 *
 * @param env - The environment's spaces.
 * @param options - C51's options; the Rainbow components default on.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { rainbow } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * rainbow(cartPole.create()).config['nStep'] // => 3
 * ```
 */
export function rainbow(env: Spaces, options: RainbowOptions = {}): Agent<DistributionalState, Obs, number> {
  const full: C51Options = {
    double: true,
    dueling: true,
    noisy: true,
    sigma0: 0.5,
    nStep: 3,
    prioritized: { alpha: 0.5, beta: 0.4 },
    ...options,
  }
  const { head } = categoricalHead(env, full)
  return distributionalAgent('rainbow', env, full, head, {
    atoms: full.atoms ?? 51,
    vMin: full.vMin ?? -10,
    vMax: full.vMax ?? 10,
  })
}

/**
 * The return distributions of a C51 or Rainbow state at an observation (noise-free), for inspection and figures.
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param options - The options the agent was built with.
 * @returns The support and one distribution per action.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { c51, categoricalDistribution } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = cartPole.create()
 * const s = c51(env, { hidden: [8], atoms: 5 }).init(key(0))
 * categoricalDistribution(env, s, [0, 0, 0, 0], { hidden: [8], atoms: 5 }).probabilities.length // => 2
 * ```
 */
export function categoricalDistribution(
  env: Spaces,
  state: DistributionalState,
  observation: Obs,
  options: C51Options = {},
): { readonly support: number[]; readonly probabilities: number[][] } {
  const atoms = options.atoms ?? 51
  const { net } = categoricalHead(env, options)
  const p = softmax(net.apply(state.params, rowsTensor([observation], observationSize(env.observationSpace))), -1)
  return {
    support: categoricalSupport(options.vMin ?? -10, options.vMax ?? 10, atoms),
    probabilities: rowsOf(p, atoms),
  }
}

// ------------------------------------------------------------------------------------------------------ QR-DQN

/** Options of {@link qrDqn}. */
export interface QrDqnOptions extends DqnOptions {
  /** The number of quantiles (default 200, as in the paper). */
  readonly quantiles?: number
  /** The Huber threshold of the quantile loss (default 1). */
  readonly kappa?: number
}

function quantileNet(env: Spaces, options: QrDqnOptions): QNetwork {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'qrDqn')
  return qNetwork(d, nA, { ...options, atoms: options.quantiles ?? 200 }, options.key ?? makeKey(0))
}

/**
 * QR-DQN (Dabney et al. 2018): the network outputs `N` quantiles of the return per action at the fixed fractions
 * `tau_i = (2i + 1) / 2N`; targets are the `N` samples `r + gamma^n theta_j(s', a*)` with `a*` maximising the mean,
 * and the loss is the {@link quantileHuberLoss}. Every DQN variant applies.
 *
 * @param env - The environment's spaces.
 * @param options - DQN's options plus the number of quantiles and `kappa`.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { qrDqn } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * qrDqn(cartPole.create(), { quantiles: 8 }).config['quantiles'] // => 8
 * ```
 */
export function qrDqn(env: Spaces, options: QrDqnOptions = {}): Agent<DistributionalState, Obs, number> {
  const nA = actionCount(env.actionSpace, 'qrDqn')
  const N = options.quantiles ?? 200
  const kappa = options.kappa ?? 1
  const net = quantileNet(env, options)
  const taus = vectorTensor(Array.from({ length: N }, (_, i) => (2 * i + 1) / (2 * N)))
  const noisy = options.noisy ?? false
  const q = (p: QParams, x: Value, k?: Key): Value => mean(net.apply(p, x, k), -1)
  const head: Head = {
    params: net.params,
    stochastic: noisy,
    q,
    prepare: (online, target, batch, k) => {
      const nk = (j: number): Key | undefined => (noisy ? foldIn(k, j) : undefined)
      const zNext = net.apply(target, batch.next, nk(0))
      const qNext = options.double === true ? q(online, batch.next, nk(1)) : mean(zNext, -1)
      const next = rowsOf(selectAtoms(zNext, oneHot(argmax(qNext, -1), nA)), N)
      const targets = next.map((row, b) => row.map((v) => (batch.rewards[b] ?? 0) + (batch.discounts[b] ?? 0) * v))
      const targetT = rowsTensor(targets, N)
      const noise = nk(2)
      return {
        terms: (p) => {
          const chosen = selectAtoms(net.apply(p, batch.obs, noise), batch.mask)
          return { loss: quantileHuberLoss(chosen, targetT, taus, { kappa, reduction: 'none' }), q: mean(chosen, -1) }
        },
        targetMean: targets.map((row) => row.reduce((a, b) => a + b, 0) / N),
      }
    },
  }
  return distributionalAgent('qrDqn', env, options, head, { quantiles: N, kappa })
}

/**
 * The quantiles of a QR-DQN state at an observation, one row of `N` per action.
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param options - The options the agent was built with.
 * @returns The quantiles per action.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { qrDqn, quantileDistribution } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = cartPole.create()
 * const s = qrDqn(env, { hidden: [8], quantiles: 4 }).init(key(0))
 * quantileDistribution(env, s, [0, 0, 0, 0], { hidden: [8], quantiles: 4 })[0]?.length // => 4
 * ```
 */
export function quantileDistribution(
  env: Spaces,
  state: DistributionalState,
  observation: Obs,
  options: QrDqnOptions = {},
): number[][] {
  const net = quantileNet(env, options)
  return rowsOf(
    net.apply(state.params, rowsTensor([observation], observationSize(env.observationSpace))),
    options.quantiles ?? 200,
  )
}

// --------------------------------------------------------------------------------------------------------- IQN

/** Options of {@link iqn}. */
export interface IqnOptions extends Omit<DqnOptions, 'dueling' | 'noisy' | 'sigma0' | 'atoms'> {
  /** Cosine basis size (default 64). */
  readonly embedding?: number
  /** Fractions sampled for the online quantiles, `N` (default 64). */
  readonly quantiles?: number
  /** Fractions sampled for the target quantiles, `N'` (default 64). */
  readonly targetQuantiles?: number
  /** Fractions sampled to estimate action values when acting, `K` (default 32). */
  readonly policySamples?: number
  /** The Huber threshold (default 1). */
  readonly kappa?: number
  /** The risk distortion of the policy (default `'neutral'`). */
  readonly distortion?: RiskDistortion
}

/**
 * IQN, implicit quantile networks (Dabney et al. 2018b): the network maps a state and a fraction `tau ~ U[0, 1]`
 * (through a cosine embedding) to the `tau`-quantile of each action's return. Training samples `N` online and `N'`
 * target fractions and minimises the {@link quantileHuberLoss} over all pairs; actions maximise
 * `mean_k Z_{beta(tau_k)}(s, a)` over `K` sampled fractions under a {@link riskDistortion} `beta` (risk-neutral by
 * default; CVaR and others give risk-sensitive policies). The greedy policy uses `K` evenly spaced fractions.
 *
 * @param env - The environment's spaces.
 * @param options - DQN's replay and target options, the sample sizes, the embedding and the distortion.
 * @returns The agent.
 * @concept distributional-rl
 * @example
 * ```ts
 * import { iqn } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * iqn(cartPole.create(), { distortion: { kind: 'cvar', eta: 0.25 } }).name // => 'iqn'
 * ```
 */
export function iqn(env: Spaces, options: IqnOptions = {}): Agent<DistributionalState, Obs, number> {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'iqn')
  const N = options.quantiles ?? 64
  const N2 = options.targetQuantiles ?? 64
  const K = options.policySamples ?? 32
  const kappa = options.kappa ?? 1
  const distortion = options.distortion ?? 'neutral'
  const beta = riskDistortion(distortion)
  const net = implicitQuantileNetwork(
    d,
    nA,
    {
      hidden: options.hidden ?? [64],
      ...(options.embedding === undefined ? {} : { embedding: options.embedding }),
    },
    options.key ?? makeKey(0),
  )
  const grid = beta(Array.from({ length: K }, (_, k) => (k + 0.5) / K))
  const fractions = (rows: number, k?: Key): Tensor => {
    if (k === undefined)
      return rowsTensor(
        Array.from({ length: rows }, () => grid),
        K,
      )
    const u = Array.from(uniform(k, [rows * K]).data)
    return fromStorage(Float64Array.from(beta(u)), [rows, K], 'float64')
  }
  const q = (p: QParams, x: Value, k?: Key): Value => mean(net.apply(p, x, fractions(x.shape[0] ?? 1, k)), 1)
  const head: Head = {
    params: net.params,
    stochastic: true,
    q,
    prepare: (online, target, batch, k) => {
      const B = batch.size
      const best = oneHot(argmax(q(options.double === true ? online : target, batch.next, foldIn(k, 0)), -1), nA)
      const tauNext = uniform(foldIn(k, 1), [B, N2])
      const next = rowsOf(selectLast(net.apply(target, batch.next, tauNext), best), N2)
      const targets = next.map((row, b) => row.map((v) => (batch.rewards[b] ?? 0) + (batch.discounts[b] ?? 0) * v))
      const targetT = rowsTensor(targets, N2)
      const tau = uniform(foldIn(k, 2), [B, N])
      return {
        terms: (p) => {
          const chosen = selectLast(net.apply(p, batch.obs, tau), batch.mask)
          return { loss: quantileHuberLoss(chosen, targetT, tau, { kappa, reduction: 'none' }), q: mean(chosen, -1) }
        },
        targetMean: targets.map((row) => row.reduce((a, b) => a + b, 0) / N2),
      }
    },
  }
  return distributionalAgent('iqn', env, { ...options, hidden: options.hidden ?? [64] }, head, {
    quantiles: N,
    targetQuantiles: N2,
    policySamples: K,
    kappa,
    distortion: distortion as Json,
  })
}

/**
 * The quantiles of an IQN state at an observation for given fractions, one row per action.
 *
 * @param env - The environment's spaces.
 * @param state - The agent state.
 * @param observation - The observation.
 * @param taus - The fractions.
 * @param options - The options the agent was built with.
 * @returns `Z_tau(s, a)` for each action (rows) and fraction (columns).
 * @concept distributional-rl
 * @example
 * ```ts
 * import { implicitQuantiles, iqn } from 'whitebox-engine/methods/rl/agents'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const env = cartPole.create()
 * const s = iqn(env, { hidden: [8], embedding: 4 }).init(key(0))
 * implicitQuantiles(env, s, [0, 0, 0, 0], [0.1, 0.9], { hidden: [8], embedding: 4 })[1]?.length // => 2
 * ```
 */
export function implicitQuantiles(
  env: Spaces,
  state: DistributionalState,
  observation: Obs,
  taus: readonly number[],
  options: IqnOptions = {},
): number[][] {
  const d = observationSize(env.observationSpace)
  const nA = actionCount(env.actionSpace, 'implicitQuantiles')
  const net = implicitQuantileNetwork(
    d,
    nA,
    { hidden: options.hidden ?? [64], ...(options.embedding === undefined ? {} : { embedding: options.embedding }) },
    makeKey(0),
  )
  const z = net.apply(state.params, rowsTensor([observation], d), rowsTensor([taus], taus.length)) // [1, T, nA]
  const flat = toFlat(z)
  return Array.from({ length: nA }, (_, a) => taus.map((_, t) => flat[t * nA + a] ?? 0))
}
