/**
 * The parts of the agent base that do not depend on a particular: the value and action maps of a tabular agent's state
 * laid out on a grid, and an episode (with the maps at the step it was played) as an `episodes` exposition record.
 */

import { greedyTable, qTable, type TabularState, valueTable } from '../../agents/index.ts'
import { type Json, MdmlError } from '../../base/index.ts'
import { type Exposition, episodeRecord, type ValueGrid } from '../../exposition/index.ts'
import type { Episode } from '../../rl/index.ts'

/** How a tabular agent's states lay out for the maps: `[rows, columns]` (state = row times columns plus column). */
export interface GridLayout {
  readonly shape: readonly [number, number]
  readonly actionNames?: readonly string[]
}

/** A tabular agent's tables on a grid. */
export interface GridMaps {
  readonly shape: readonly [number, number]
  readonly actionNames?: readonly string[]
  /** `max_a Q(s, a)` per cell. */
  readonly value: readonly (readonly number[])[]
  /** One grid per action (its value in each cell), then the greedy action. */
  readonly actions: readonly ValueGrid[]
  readonly snapshot: { readonly values: readonly number[]; readonly policy: readonly number[] }
}

const isTabular = (state: unknown): state is TabularState =>
  typeof state === 'object' &&
  state !== null &&
  (state as { Q?: unknown }).Q instanceof Float64Array &&
  typeof (state as { nStates?: unknown }).nStates === 'number' &&
  typeof (state as { nActions?: unknown }).nActions === 'number'

/**
 * The maps of a tabular agent state, or `undefined` for any other agent.
 *
 * @param state - An agent state.
 * @param layout - The grid the states lay out on (default one row).
 * @returns The maps.
 * @throws MdmlError with code `shape` when the layout does not hold the agent's states.
 */
export function gridMaps(state: unknown, layout?: GridLayout): GridMaps | undefined {
  if (!isTabular(state)) return undefined
  const [rows, cols] = layout?.shape ?? [1, state.nStates]
  if (rows * cols !== state.nStates)
    throw new MdmlError('shape', `agent maps: a ${rows} by ${cols} grid does not hold ${state.nStates} states`)
  const onGrid = (v: readonly number[]): number[][] =>
    Array.from({ length: rows }, (_, r) => v.slice(r * cols, (r + 1) * cols))
  const values = valueTable(state)
  const policy = greedyTable(state)
  const q = qTable(state)
  const names = (a: number): string => layout?.actionNames?.[a] ?? `action ${a}`
  return {
    shape: [rows, cols],
    ...(layout?.actionNames === undefined ? {} : { actionNames: layout.actionNames }),
    value: onGrid(values),
    actions: [
      ...Array.from({ length: state.nActions }, (_, a) => ({
        name: names(a),
        values: onGrid(q.map((row) => row[a] ?? 0)),
      })),
      { name: 'greedy action', values: onGrid(policy) },
    ],
    snapshot: { values, policy },
  }
}

/**
 * An episode as an `episodes` record, with the agent's tables at the step it was played when it has them.
 *
 * @param environment - The environment's id.
 * @param agent - The agent's name.
 * @param trained - Episodes the agent had finished when it was played.
 * @param episode - The episode (its observations stand for its states).
 * @param maps - A tabular agent's maps.
 * @returns The record.
 */
export function episodeRecordOf(
  environment: string,
  agent: string,
  trained: number,
  episode: Episode<unknown, unknown, unknown>,
  maps: GridMaps | undefined,
): Exposition {
  return episodeRecord({
    environment,
    agent,
    returns: [episode.return],
    lengths: [episode.length],
    recorded: [
      {
        episode: 0,
        states: episode.observations as readonly Json[],
        actions: episode.actions as readonly Json[],
        rewards: episode.rewards,
      },
    ],
    ...(maps === undefined
      ? {}
      : {
          snapshotShape: maps.shape,
          ...(maps.actionNames === undefined ? {} : { actionNames: maps.actionNames }),
          snapshots: [{ episode: trained, values: maps.snapshot.values, policy: maps.snapshot.policy }],
        }),
  })
}
