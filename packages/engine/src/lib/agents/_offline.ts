// Shared plumbing of the offline agents (private to methods/rl/agents): minibatches of a fixed dataset as tensors,
// small MLPs as pure functions of their parameters, and action encodings for discrete and box action spaces.
import { MdmlError } from '../base/index.ts'
import { mlp } from '../nn/index.ts'
import type { ParamTree } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import type { Transition } from '../rl/index.ts'
import type { Space } from '../space/index.ts'
import { add, mul, oneHot, type Tensor, tanh, type Value } from '../tensor/index.ts'
import { actionBox, rowsTensor, spaceFeatures, vectorTensor } from './_common.ts'

/** A network as a pure function of a parameter tree. */
export interface Net {
  readonly params: ParamTree
  apply(params: ParamTree, x: Value): Value
}

/** An MLP with ReLUs between its layers (lib/nn's `mlp`) as a {@link Net}. */
export function network(widths: readonly number[], key: Key): Net {
  const m = mlp(widths, { key })
  return { params: m.params as ParamTree, apply: (p, x) => m.apply(p as never, x) }
}

/** How a method sees an environment's observations and actions. */
export interface OfflineSpaces {
  readonly obsSize: number
  readonly features: (observation: unknown) => number[]
  readonly discrete: boolean
  /** The number of actions (discrete) or the action dimension (box). */
  readonly actionSize: number
  /** Box bounds as centre and half-width, for tanh-squashed actors. */
  readonly centre: readonly number[]
  readonly scale: readonly number[]
  readonly low: readonly number[]
  readonly high: readonly number[]
}

/** Sizes, encodings and bounds of an environment's spaces. */
export function offlineSpaces(env: { readonly observationSpace: Space; readonly actionSpace: Space }): OfflineSpaces {
  const f = spaceFeatures(env.observationSpace)
  const space = env.actionSpace
  if (space.kind === 'discrete')
    return {
      obsSize: f.size,
      features: f.of,
      discrete: true,
      actionSize: space.n,
      centre: [],
      scale: [],
      low: [],
      high: [],
    }
  if (space.kind !== 'box') throw new MdmlError('value', 'offline agents need discrete or box actions')
  const b = actionBox(space, 'offline')
  return {
    obsSize: f.size,
    features: f.of,
    discrete: false,
    actionSize: b.size,
    centre: b.low.map((l, i) => ((b.high[i] ?? 0) + l) / 2),
    scale: b.low.map((l, i) => ((b.high[i] ?? 0) - l) / 2),
    low: b.low,
    high: b.high,
  }
}

/** A minibatch as tensors. */
export interface OfflineBatch {
  readonly obs: Tensor
  readonly next: Tensor
  /** Discrete: one-hot actions `[B, nA]`; box: actions `[B, m]`. */
  readonly actions: Value
  /** Discrete action indices `[B]` (empty for box actions). */
  readonly labels: Tensor
  readonly rewards: Tensor
  /** `1 - terminated` per transition. */
  readonly notDone: Tensor
  readonly size: number
}

/** Tensors of a minibatch of transitions, with observations through `transform` (normalisation) if given. */
export function offlineBatch<O, A>(
  sp: OfflineSpaces,
  batch: readonly Transition<O, A>[],
  transform: (x: number[]) => number[] = (x) => x,
): OfflineBatch {
  const obs = rowsTensor(
    batch.map((t) => transform(sp.features(t.observation))),
    sp.obsSize,
  )
  const next = rowsTensor(
    batch.map((t) => transform(sp.features(t.nextObservation))),
    sp.obsSize,
  )
  const labels = sp.discrete ? vectorTensor(batch.map((t) => t.action as number)) : vectorTensor([])
  const actions = sp.discrete
    ? oneHot(labels, sp.actionSize)
    : rowsTensor(
        batch.map((t) => t.action as readonly number[]),
        sp.actionSize,
      )
  return {
    obs,
    next,
    actions,
    labels,
    rewards: vectorTensor(batch.map((t) => t.reward)),
    notDone: vectorTensor(batch.map((t) => (t.terminated ? 0 : 1))),
    size: batch.length,
  }
}

/** An action clipped to the box (a coordinate without a bound stays as it is). */
export function clipToBox(sp: OfflineSpaces, a: readonly number[]): number[] {
  return a.map((v, i) => Math.min(sp.high[i] ?? v, Math.max(sp.low[i] ?? v, v)))
}

/** Squash network outputs into the action box: `centre + scale tanh(u)`. */
export function squashToBox(sp: OfflineSpaces, u: Value): Value {
  return add(mul(tanh(u), vectorTensor(sp.scale)), vectorTensor(sp.centre))
}
