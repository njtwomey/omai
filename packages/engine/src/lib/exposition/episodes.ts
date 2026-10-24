import type { Json } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** A reinforcement-learning run: returns per episode, steps of chosen episodes, and value or policy snapshots. */
export interface EpisodeRecord {
  readonly kind: 'episodes'
  readonly environment: string
  readonly agent: string
  /** The return and length of every episode. */
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  /** Recorded episodes in full: per step the state, action and reward (states as JSON for any environment). */
  readonly recorded?: readonly {
    readonly episode: number
    readonly states: readonly Json[]
    readonly actions: readonly Json[]
    readonly rewards: readonly number[]
  }[]
  /** Snapshots of tabular values or policies at chosen episodes (state-indexed). */
  readonly snapshots?: readonly {
    readonly episode: number
    readonly values?: readonly number[]
    readonly policy?: readonly number[]
  }[]
}

/**
 * An episode record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when returns and lengths differ, or a recorded episode's columns do.
 * @example
 * ```ts
 * import { episodeRecord } from 'whitebox-engine/exposition'
 * episodeRecord({ environment: 'CartPole', agent: 'dqn', returns: [12, 40], lengths: [12, 40] }).returns // => [12, 40]
 * ```
 */
export function episodeRecord(record: Omit<EpisodeRecord, 'kind'>): EpisodeRecord {
  sameLength('episodes', 'lengths', record.lengths.length, record.returns.length)
  for (const r of record.recorded ?? []) {
    sameLength('episodes', `recorded[${r.episode}].actions`, r.actions.length, r.rewards.length)
    sameLength('episodes', `recorded[${r.episode}].states`, r.states.length, r.rewards.length + 1)
  }
  return { kind: 'episodes', ...record }
}
