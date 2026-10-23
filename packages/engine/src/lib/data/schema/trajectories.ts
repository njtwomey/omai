import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { checkRows, concrete, concreteFloat, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * Logged transitions from an environment (offline RL and bandit data), one row per step, with the behaviour policy's
 * probability of each logged action when it is known.
 */
export interface TrajectoryData {
  readonly kind: 'trajectories'
  /** Observations `[N, ...]`. */
  readonly observations: Tensor
  /** Actions `[N]` (int32, discrete) or `[N, ...]` (float, continuous). */
  readonly actions: Tensor
  readonly rewards: Tensor
  readonly nextObservations?: Tensor
  /** True where the episode ended in a terminal state. */
  readonly terminations: Tensor<'bool'>
  /** True where the episode was cut off (time limit). */
  readonly truncations: Tensor<'bool'>
  /** The episode each step belongs to `[N]` (int32). */
  readonly episodes: Tensor<'int32'>
  /** The behaviour policy's probability (or density) of the logged action `[N]`. */
  readonly propensities?: Tensor
  /** The number of discrete actions, when actions are discrete. */
  readonly nActions?: number
  readonly meta: DatasetMeta
}

/** Arrays for {@link trajectories}. */
export interface TrajectoryArrays {
  readonly observations: TensorLike
  readonly actions: TensorLike
  readonly rewards: TensorLike
  readonly nextObservations?: TensorLike
  readonly terminations?: TensorLike
  readonly truncations?: TensorLike
  /** Default: one episode. */
  readonly episodes?: TensorLike
  readonly propensities?: TensorLike
  readonly nActions?: number
}

/**
 * A trajectory dataset from logged arrays; discrete actions (with `nActions`) are stored as int32.
 *
 * @param arrays - Observations, actions, rewards and the optional fields.
 * @param meta - Name, description and provenance.
 * @returns The dataset.
 * @example
 * ```ts
 * import { trajectories } from 'whitebox-engine/data'
 * trajectories({ observations: [[0], [1]], actions: [1, 0], rewards: [0, 1], nActions: 2 }).actions.dtype // => 'int32'
 * ```
 */
export function trajectories(arrays: TrajectoryArrays, meta: Partial<DatasetMeta> = {}): TrajectoryData {
  const observations = concreteFloat(arrays.observations, 'trajectories')
  const n = observations.shape[0] ?? 0
  const actions =
    arrays.nActions === undefined
      ? concreteFloat(arrays.actions, 'trajectories')
      : concrete(arrays.actions, 'int32', 'trajectories')
  const flags = (x: TensorLike | undefined) =>
    x === undefined ? fromStorage(new Uint8Array(n), [n], 'bool') : concrete(x, 'bool', 'trajectories')
  const out: TrajectoryData = {
    kind: 'trajectories',
    observations,
    actions,
    rewards: concreteFloat(arrays.rewards, 'trajectories'),
    ...(arrays.nextObservations === undefined
      ? {}
      : { nextObservations: concreteFloat(arrays.nextObservations, 'trajectories') }),
    terminations: flags(arrays.terminations),
    truncations: flags(arrays.truncations),
    episodes:
      arrays.episodes === undefined
        ? fromStorage(new Int32Array(n), [n], 'int32')
        : concrete(arrays.episodes, 'int32', 'trajectories'),
    ...(arrays.propensities === undefined ? {} : { propensities: concreteFloat(arrays.propensities, 'trajectories') }),
    ...(arrays.nActions === undefined ? {} : { nActions: arrays.nActions }),
    meta: metaFor('trajectories', meta),
  }
  checkRows('trajectories', n, {
    actions: out.actions,
    rewards: out.rewards,
    nextObservations: out.nextObservations,
    terminations: out.terminations,
    truncations: out.truncations,
    episodes: out.episodes,
    propensities: out.propensities,
  })
  return out
}
