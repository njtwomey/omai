/**
 * The pure logic behind the environment selector: the selection of a registry entry with its default options, making
 * the selected environment, the first frame of an environment of any form (single-agent, turn-based or parallel), and
 * spaces formatted gymnasium-style (`Box(-1, 1, [2])`, `Discrete(4)`) for display.
 */

import { type EnvironmentEntry, type EnvironmentRegistry, findEnvironment } from 'mdml-engine/environments'
import { key } from 'mdml-engine/random'
import type { AecEnv, Env, ParallelEnv } from 'mdml-engine/rl'
import type { Scene } from 'mdml-engine/scene'
import { defaults, type Space, type SpaceValue } from 'mdml-engine/space'

/** What an environment selector has chosen: an entry of its registry and the option values to make it with. */
export interface EnvironmentSelection {
  readonly name: string
  /** The definition's option values (empty when it has none). */
  readonly options: Readonly<Record<string, SpaceValue>>
}

/** An environment of any form. */
export type AnyEnvironment =
  | Env<unknown, unknown, unknown>
  | AecEnv<unknown, unknown, unknown>
  | ParallelEnv<unknown, unknown, unknown>

/**
 * The entry of a selection in a registry.
 *
 * @param registry - The registry.
 * @param name - The entry's name.
 * @returns The entry.
 * @throws Error when the registry has no such entry.
 */
export function entryOf(registry: EnvironmentRegistry, name: string): EnvironmentEntry {
  const entry = findEnvironment(registry, name)
  if (entry === undefined) throw new Error(`the environment registry has no entry '${name}'`)
  return entry
}

/**
 * The selection of one entry with its definition's default options.
 *
 * @param registry - The registry.
 * @param name - The entry's name (default the first entry).
 * @returns The selection.
 * @throws Error when the registry has no such entry.
 */
export function selectEnvironment(registry: EnvironmentRegistry, name?: string): EnvironmentSelection {
  const entry = name === undefined ? registry.entries[0] : entryOf(registry, name)
  if (entry === undefined) throw new Error('the environment registry is empty')
  const options = entry.options === undefined ? {} : (defaults(entry.options) as Record<string, SpaceValue>)
  return { name: entry.name, options }
}

/**
 * Make the selected environment with the selection's options.
 *
 * @param registry - The registry.
 * @param selection - The selection.
 * @returns The environment, of whatever form the entry has.
 * @throws Error when the registry has no such entry.
 */
export function makeSelection(registry: EnvironmentRegistry, selection: EnvironmentSelection): AnyEnvironment {
  return entryOf(registry, selection.name).make(selection.options)
}

/**
 * The first frame of an environment: its render of the state a reset from the seed gives, whatever its form.
 *
 * @param env - The environment.
 * @param seed - The reset seed (default 0).
 * @returns The scene.
 */
export function firstFrame(env: AnyEnvironment, seed = 0): Scene {
  const k = key(seed)
  if (!('possibleAgents' in env)) return env.render(env.reset(k).state)
  if ('currentAgent' in env) return env.render(env.reset(k))
  return env.render(env.reset(k).state)
}

const BIG = Number.MAX_VALUE

function bound(b: number | readonly number[]): string {
  if (typeof b === 'number') return b <= -BIG ? '-inf' : b >= BIG ? 'inf' : String(Number(b.toPrecision(4)))
  if (b.length === 0) return '[]'
  const first = b[0] ?? 0
  if (b.every((v) => v === first)) return bound(first)
  if (b.length <= 4) return `[${b.map(bound).join(', ')}]`
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of b) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return `[${bound(lo)} to ${bound(hi)}]`
}

/**
 * A space as gymnasium prints one: `Discrete(4)`, `Box(-1, 1, [2])` (bounds collapsed when equal, summarised as
 * a range past four entries, infinities as `inf`), `MultiDiscrete([2, 3])`, `MultiBinary(4)`,
 * `Tuple(Discrete(32), Discrete(11))`, `Dict(a: Discrete(2))`; configuration spaces as `Real(0, 1)`, `Int(1, 10)`,
 * `Choice(a | b)` and `Bool`.
 *
 * @param space - The space.
 * @returns Its text.
 */
export function formatSpace(space: Space): string {
  switch (space.kind) {
    case 'discrete':
      return `Discrete(${space.n}${space.start === undefined || space.start === 0 ? '' : `, start ${space.start}`})`
    case 'box':
      return `Box(${bound(space.low)}, ${bound(space.high)}, [${space.shape.join(', ')}])`
    case 'multiDiscrete':
      return `MultiDiscrete([${space.nvec.join(', ')}])`
    case 'multiBinary':
      return `MultiBinary(${space.n})`
    case 'tuple':
      return `Tuple(${space.spaces.map(formatSpace).join(', ')})`
    case 'dict':
      return `Dict(${Object.entries(space.spaces)
        .map(([k, s]) => `${k}: ${formatSpace(s)}`)
        .join(', ')})`
    case 'real':
      return `Real(${bound(space.low)}, ${bound(space.high)}${space.log === true ? ', log' : ''})`
    case 'int':
      return `Int(${space.low}, ${space.high})`
    case 'choice':
      return `Choice(${space.options.map(String).join(' | ')})`
    default:
      return 'Bool'
  }
}
