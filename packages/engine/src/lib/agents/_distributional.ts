// Shared machinery of the distributional agents (private to methods/rl/agents): the replay agent they all are, the
// head contract that distinguishes them, and the quantile Huber kernel with its derivative rules.
import { customVjp } from '../autodiff/index.ts'
import { argmaxOf, type Json } from '../base/index.ts'
import { incrementalUpdate, linearSchedule, type Tree } from '../optim/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
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
} from '../rl/index.ts'
import {
  add,
  asTensor,
  cast,
  concat,
  expandDims,
  fromStorage,
  item,
  matmul,
  mean,
  mul,
  oneHot,
  reshape,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  actionCount,
  gradientStepWithTerms,
  observationSize,
  optimiserOf,
  rowsTensor,
  type Spaces,
  vectorTensor,
} from './_common.ts'
import type { DqnOptions, DqnState, QParams } from './value-based.ts'

export const asFloat = (x: TensorLike): Value => cast(asTensor(x), 'float64')
export const scalarTensor = (x: number): Tensor => fromStorage(Float64Array.of(x), [], 'float64')

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

export const quantileHuberKernel: (...inputs: Value[]) => Value = customVjp({
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

// ------------------------------------------------------------------------------------------- the shared agent

export type Obs = readonly number[]

/** A replay minibatch as the heads need it. */
export interface Batch {
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
export interface Head {
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

export const rowsOf = (x: Value, width: number): number[][] => {
  const flat = toFlat(x)
  return Array.from({ length: flat.length / width }, (_, i) => flat.slice(i * width, (i + 1) * width))
}

// The chosen action's row of `[B, nA, N]` (atoms last) or of `[B, N, nA]` (actions last), for one-hot `mask [B, nA]`.
// (Batched matrix products: the CPU backend's broadcasting elementwise products are much slower.)
export const selectAtoms = (x: Value, mask: Value): Value => {
  const [b = 1, , n = 1] = x.shape
  return reshape(matmul(expandDims(mask, 1), x), [b, n])
}
export const selectLast = (x: Value, mask: Value): Value => {
  const [b = 1, n = 1] = x.shape
  return reshape(matmul(x, expandDims(mask, -1)), [b, n])
}

export function distributionalAgent(
  name: string,
  env: Spaces,
  options: DqnOptions,
  head: Head,
  extraConfig: Readonly<Record<string, Json>>,
): Agent<DqnState, Obs, number> {
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

  const update = (s: DqnState, k: Key): DqnState => {
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
      let next: DqnState = {
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
