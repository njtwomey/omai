/**
 * Modes: the options of a slot that each build the algorithm their own way (a solver, an optimiser, a kernel, a
 * search). A mode is one module's worth of declarations: its knobs (a dict space), the factory that makes a library
 * algorithm for a problem, the common fitted parameters its state yields, the paths only it can say, and what it needs
 * of the other slots. A table of modes is one `union` space and one dispatcher, so a method's `algorithm` and `params`
 * become one line each, and the controls, grids and tuners see every mode's knobs through the space. A mode never
 * logs: the method's contract reads its state through `params`, and the mode's `extras` are derived from the state too.
 */

import { type Json, MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import {
  type DictSpace,
  type ModeConfig,
  type ModeTableOptions,
  type SpaceValue,
  type UnionSpace,
  union,
} from '../space/index.ts'
import type { Algorithm } from './algorithm.ts'
import type { Contract } from './contract.ts'

/** The other slots' current values, for the validity check: a primitive, or a union value read by its `kind`. */
export type SlotValues = Readonly<Record<string, SpaceValue>>

/**
 * One mode of a slot: how one option builds the algorithm.
 *
 * @typeParam Problem - What the algorithm is built over (a design matrix, an objective, a search problem).
 * @typeParam S - The algorithm's state, this mode's own.
 * @typeParam P - The fitted parameters every mode of the slot yields.
 */
export interface ModeSpec<Problem, S, P extends Json> {
  readonly description?: string
  /** The mode's own knobs (empty when it has none). */
  readonly space: DictSpace
  /** The library algorithm for a problem under these knobs (library pieces only; nothing solved here). */
  make(problem: Problem, knobs: ModeConfig, key: Key): Algorithm<S>
  /** The common fitted parameters of a state, the one shape the probes and verbs read whatever the mode. */
  params(state: S): P
  /** Paths only this mode can say, derived from its state (a native regularisation path, a line search's trials). */
  readonly extras?: Contract<S>
  /** For each other slot, the option values this mode admits (absent: any); the validity table, kept beside the mode. */
  readonly needs?: Readonly<Record<string, readonly (string | number | boolean)[]>>
}

/** A mode with its name, as the table hands it back. */
export type NamedMode<Problem, P extends Json> = ModeSpec<Problem, unknown, P> & { readonly name: string }

/**
 * A table of modes: one union space over their knobs, and the dispatch from a configuration to the mode it names.
 *
 * @typeParam Problem - What the modes build over.
 * @typeParam P - The fitted parameters every mode yields.
 */
export interface ModeTable<Problem, P extends Json> {
  /** The key of a configuration that names the mode. */
  readonly tag: string
  /** The modes' names, in declaration order. */
  readonly names: readonly string[]
  /** The slot as a space: a union of the modes' knobs, for controls, grids and tuners. */
  readonly space: UnionSpace
  /** The mode a configuration names (the default mode when it names none). */
  modeOf(config: SpaceValue | undefined): NamedMode<Problem, P>
  /** The modes admissible given the other slots' values: those whose `needs` the values satisfy. */
  admissible(others: SlotValues): string[]
  /**
   * The algorithm the configuration's mode builds.
   *
   * @throws MdmlError with code `value` when `others` are given and the mode does not admit them.
   */
  make(problem: Problem, config: SpaceValue, key: Key, others?: SlotValues): Algorithm<unknown>
  /** The fitted parameters of a state under the configuration's mode. */
  params(config: SpaceValue, state: unknown): P
  /** The extra paths of the configuration's mode (empty when it has none). */
  extras(config: SpaceValue): Contract<unknown>
}

/** The option a slot value names: itself when primitive, its `kind` when a union value. */
function optionOf(value: SpaceValue | undefined): string | number | boolean | undefined {
  if (value === undefined) return undefined
  if (typeof value === 'object') {
    if (Array.isArray(value)) return undefined
    const kind = (value as Readonly<Record<string, SpaceValue>>)['kind']
    return typeof kind === 'object' ? undefined : kind
  }
  return value
}

/** Whether a mode admits the other slots' values. */
function admits(mode: ModeSpec<unknown, unknown, Json>, others: SlotValues): boolean {
  if (mode.needs === undefined) return true
  return Object.entries(mode.needs).every(([slot, allowed]) => {
    const option = optionOf(others[slot])
    return option !== undefined && allowed.includes(option)
  })
}

/**
 * A table of modes for one slot: the union space over their knobs, and the dispatch to the mode a configuration
 * names. The knobs of each mode are its own; the tag names the mode; a method keeps one table per slot and writes
 * `algorithm: (problem, h, key) => table.make(problem, h.solver, key)` and `params: (s, h) => table.params(h.solver, s)`.
 *
 * @param table - The modes by name.
 * @param options - The tag, the default mode and metadata.
 * @returns The table.
 * @throws MdmlError with code `value` for an empty table or a default that is not a mode (as `union` does).
 * @index function Build the table of modes for one slot, with each mode's knobs.
 * @example
 * ```ts
 * import { modes } from 'mdml-engine/trace'
 * import { dict, real } from 'mdml-engine/space'
 * const countdown = (start: number) => ({ name: 'countdown', config: {}, init: () => start, step: (n: number) => n - 1 })
 * const solvers = modes({
 *   fast: { space: dict({}), make: () => countdown(8), params: (n: number) => ({ n }) },
 *   slow: { space: dict({ start: real(1, 100, { default: 50 }) }), make: (_p, k) => countdown(k['start'] as number), params: (n: number) => ({ n }) },
 * })
 * solvers.names // => ['fast', 'slow']
 * solvers.modeOf({ kind: 'slow' }).name // => 'slow'
 * solvers.params({ kind: 'fast' }, 5) // => { n: 5 }
 * ```
 */
export function modes<Problem, P extends Json>(
  // biome-ignore lint/suspicious/noExplicitAny: each mode has a state type of its own; the table erases them
  table: Readonly<Record<string, ModeSpec<Problem, any, P>>>,
  options: ModeTableOptions = {},
): ModeTable<Problem, P> {
  const tag = options.tag ?? 'kind'
  const names = Object.keys(table)
  const space = union(Object.fromEntries(names.map((n) => [n, table[n]?.space as DictSpace])), {
    tag,
    ...(options.default === undefined ? {} : { default: options.default }),
    ...(options.label === undefined ? {} : { label: options.label }),
    ...(options.description === undefined ? {} : { description: options.description }),
  })
  const nameOf = (config: SpaceValue | undefined): string => {
    const fallback = options.default ?? (names[0] as string)
    if (typeof config !== 'object' || config === null || Array.isArray(config)) return fallback
    const name = (config as Readonly<Record<string, SpaceValue>>)[tag]
    return typeof name === 'string' && Object.hasOwn(table, name) ? name : fallback
  }
  const modeOf = (config: SpaceValue | undefined): NamedMode<Problem, P> => {
    const name = nameOf(config)
    return { ...(table[name] as ModeSpec<Problem, unknown, P>), name }
  }
  const knobsOf = (config: SpaceValue | undefined): ModeConfig =>
    typeof config === 'object' && config !== null && !Array.isArray(config)
      ? (config as Readonly<Record<string, SpaceValue>>)
      : {}
  return {
    tag,
    names,
    space,
    modeOf,
    admissible: (others) => names.filter((n) => admits(table[n] as ModeSpec<unknown, unknown, Json>, others)),
    make: (problem, config, key, others) => {
      const mode = modeOf(config)
      if (others !== undefined && !admits(mode as ModeSpec<unknown, unknown, Json>, others))
        throw new MdmlError(
          'value',
          `modes: '${mode.name}' does not admit ${JSON.stringify(others)} (needs ${JSON.stringify(mode.needs)})`,
        )
      return mode.make(problem, knobsOf(config), key)
    },
    params: (config, state) => modeOf(config).params(state),
    extras: (config) => modeOf(config).extras ?? {},
  }
}
