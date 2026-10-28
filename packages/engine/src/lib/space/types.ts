/**
 * The space types and their constructors: real and integer intervals (optionally log-scaled or stepped), choices and
 * booleans for configuration spaces, gymnasium-style `box`, `discrete`, `multiDiscrete` and `multiBinary` spaces for
 * observations and actions, tuples and dicts that compose them, with conditional dimensions declared by `when`, and
 * tagged unions (a mode with the knobs of its own that each option brings).
 */

import { MdmlError } from '../base/index.ts'

/**
 * Descriptive metadata every space can carry (for documentation, controls and reports), and the condition that makes
 * it a conditional dimension. Each space kind also takes a `default` value of its own type.
 */
export interface SpaceMeta {
  readonly label?: string
  readonly description?: string
  readonly unit?: string
  /**
   * A conditional dimension: inside a dict, the child is active only when the sibling `key` is active and has one of
   * the listed values (conditions may chain: the sibling may itself be conditional). Inactive children are absent
   * from values, drawn by no sampler and varied by no search.
   */
  readonly when?: { readonly key: string; readonly in: readonly (string | number | boolean)[] }
}

/** A real interval, optionally log-scaled or stepped. */
export interface RealSpace extends SpaceMeta {
  readonly kind: 'real'
  readonly low: number
  readonly high: number
  readonly log?: boolean
  readonly step?: number
  readonly default?: number
}

/** An integer interval (inclusive), optionally log-scaled. */
export interface IntSpace extends SpaceMeta {
  readonly kind: 'int'
  readonly low: number
  readonly high: number
  readonly log?: boolean
  readonly default?: number
}

/** A choice among listed options. */
export interface ChoiceSpace<T extends string | number | boolean = string | number | boolean> extends SpaceMeta {
  readonly kind: 'choice'
  readonly options: readonly T[]
  readonly default?: T
}

/** A boolean flag. */
export interface BoolSpace extends SpaceMeta {
  readonly kind: 'bool'
  readonly default?: boolean
}

/** A box of real numbers with a shape (gymnasium `Box`); values are flat row-major arrays. */
export interface BoxSpace extends SpaceMeta {
  readonly kind: 'box'
  readonly low: number | readonly number[]
  readonly high: number | readonly number[]
  readonly shape: readonly number[]
  readonly default?: readonly number[]
}

/** The integers `start, ..., start + n - 1` (gymnasium `Discrete`). */
export interface DiscreteSpace extends SpaceMeta {
  readonly kind: 'discrete'
  readonly n: number
  readonly start?: number
  readonly default?: number
}

/** Several discrete coordinates, coordinate `i` in `0 .. nvec[i] - 1` (gymnasium `MultiDiscrete`). */
export interface MultiDiscreteSpace extends SpaceMeta {
  readonly kind: 'multiDiscrete'
  readonly nvec: readonly number[]
  readonly default?: readonly number[]
}

/** Binary vectors of length `n` (gymnasium `MultiBinary`). */
export interface MultiBinarySpace extends SpaceMeta {
  readonly kind: 'multiBinary'
  readonly n: number
  readonly default?: readonly number[]
}

/** A fixed-length tuple of spaces. */
export interface TupleSpace extends SpaceMeta {
  readonly kind: 'tuple'
  readonly spaces: readonly Space[]
}

/** Named sub-spaces, with conditional dimensions (`when` on a child). */
export interface DictSpace extends SpaceMeta {
  readonly kind: 'dict'
  readonly spaces: Readonly<Record<string, Space>>
}

/**
 * A tagged union: named options, each a dict space of its own, so choosing an option (a mode: a solver, an optimiser,
 * a kernel) brings its own knobs and no others. A value names the option under `tag` and carries that option's values
 * beside it: `{ kind: 'gradient', lr: 0.1, batch: 32 }`. Options may nest unions, so a tree of modes is one space.
 */
export interface UnionSpace extends SpaceMeta {
  readonly kind: 'union'
  readonly options: Readonly<Record<string, DictSpace>>
  /** The key of the value that names the option (default `'kind'`); no option may have a knob of that name. */
  readonly tag?: string
  /** The option chosen by default (default the first). */
  readonly default?: string
}

/**
 * A typed set of values with metadata. One type describes hyperparameters, dataset knobs, observations and actions,
 * search spaces and figure controls, so any of these can be sampled, tuned, encoded or turned into controls.
 *
 * @concept spaces
 */
export type Space =
  | RealSpace
  | IntSpace
  | ChoiceSpace
  | BoolSpace
  | BoxSpace
  | DiscreteSpace
  | MultiDiscreteSpace
  | MultiBinarySpace
  | TupleSpace
  | DictSpace
  | UnionSpace

/** A value of a space. */
export type SpaceValue =
  | number
  | string
  | boolean
  | readonly number[]
  | readonly SpaceValue[]
  | { readonly [k: string]: SpaceValue }

function checkInterval(low: number, high: number, log: boolean | undefined, name: string): void {
  if (!(high >= low)) throw new MdmlError('value', `${name}: need high >= low, got [${low}, ${high}]`)
  if (log === true && !(low > 0)) throw new MdmlError('value', `${name}: a log scale needs low > 0`)
}

/**
 * A real interval `[low, high]`.
 *
 * @param low - Lower bound.
 * @param high - Upper bound.
 * @param options - Log scale, step, default and metadata.
 * @returns The space.
 * @concept spaces
 * @example
 * ```ts
 * import { real } from 'mdml-engine/space'
 * real(1e-4, 1, { log: true, label: 'learning rate' }).log // => true
 * ```
 */
export function real(low: number, high: number, options: Omit<RealSpace, 'kind' | 'low' | 'high'> = {}): RealSpace {
  checkInterval(low, high, options.log, 'real')
  return { kind: 'real', low, high, ...options }
}

/**
 * An integer interval `[low, high]` (inclusive).
 *
 * @param low - Lower bound.
 * @param high - Upper bound.
 * @param options - Log scale, default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { int } from 'mdml-engine/space'
 * int(1, 10).high // => 10
 * ```
 */
export function int(low: number, high: number, options: Omit<IntSpace, 'kind' | 'low' | 'high'> = {}): IntSpace {
  checkInterval(low, high, options.log, 'int')
  return { kind: 'int', low: Math.ceil(low), high: Math.floor(high), ...options }
}

/**
 * A choice among options.
 *
 * @param options - The options.
 * @param meta - Default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { choice } from 'mdml-engine/space'
 * choice(['relu', 'tanh']).options.length // => 2
 * ```
 */
export function choice<T extends string | number | boolean>(
  options: readonly T[],
  meta: Omit<ChoiceSpace<T>, 'kind' | 'options'> = {},
): ChoiceSpace<T> {
  if (options.length === 0) throw new MdmlError('value', 'choice: need at least one option')
  return { kind: 'choice', options, ...meta }
}

/**
 * A boolean flag.
 *
 * @param meta - Default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { bool } from 'mdml-engine/space'
 * bool({ default: true }).default // => true
 * ```
 */
export function bool(meta: Omit<BoolSpace, 'kind'> = {}): BoolSpace {
  return { kind: 'bool', ...meta }
}

/**
 * A box of reals with a shape.
 *
 * @param low - Lower bound (scalar or per element).
 * @param high - Upper bound (scalar or per element).
 * @param shape - The shape.
 * @param meta - Default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { box } from 'mdml-engine/space'
 * box(-1, 1, [2]).shape // => [2]
 * ```
 */
export function box(
  low: number | readonly number[],
  high: number | readonly number[],
  shape: readonly number[],
  meta: Omit<BoxSpace, 'kind' | 'low' | 'high' | 'shape'> = {},
): BoxSpace {
  return { kind: 'box', low, high, shape, ...meta }
}

/**
 * The integers `start .. start + n - 1`.
 *
 * @param n - Number of values.
 * @param options - The start (default 0) and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { discrete } from 'mdml-engine/space'
 * discrete(4).n // => 4
 * ```
 */
export function discrete(n: number, options: Omit<DiscreteSpace, 'kind' | 'n'> = {}): DiscreteSpace {
  if (!Number.isInteger(n) || n < 1) throw new MdmlError('value', 'discrete: n must be a positive integer')
  return { kind: 'discrete', n, ...options }
}

/**
 * Several discrete coordinates.
 *
 * @param nvec - Number of values per coordinate.
 * @param meta - Default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { multiDiscrete } from 'mdml-engine/space'
 * multiDiscrete([3, 2]).nvec // => [3, 2]
 * ```
 */
export function multiDiscrete(
  nvec: readonly number[],
  meta: Omit<MultiDiscreteSpace, 'kind' | 'nvec'> = {},
): MultiDiscreteSpace {
  return { kind: 'multiDiscrete', nvec, ...meta }
}

/**
 * Binary vectors.
 *
 * @param n - Length.
 * @param meta - Default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { multiBinary } from 'mdml-engine/space'
 * multiBinary(3).n // => 3
 * ```
 */
export function multiBinary(n: number, meta: Omit<MultiBinarySpace, 'kind' | 'n'> = {}): MultiBinarySpace {
  return { kind: 'multiBinary', n, ...meta }
}

/**
 * A tuple of spaces.
 *
 * @param spaces - The components.
 * @param meta - Metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { tuple, discrete } from 'mdml-engine/space'
 * tuple([discrete(2), discrete(3)]).spaces.length // => 2
 * ```
 */
export function tuple(spaces: readonly Space[], meta: SpaceMeta = {}): TupleSpace {
  return { kind: 'tuple', spaces, ...meta }
}

/**
 * Named sub-spaces (a configuration space).
 *
 * @param spaces - The named components (each may have a `when` condition on a sibling).
 * @param meta - Metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { dict, real, choice } from 'mdml-engine/space'
 * Object.keys(dict({ lr: real(1e-4, 1, { log: true }), act: choice(['relu', 'tanh']) }).spaces) // => ['lr', 'act']
 * ```
 */
export function dict(spaces: Readonly<Record<string, Space>>, meta: SpaceMeta = {}): DictSpace {
  return { kind: 'dict', spaces, ...meta }
}

/**
 * A tagged union of named options, each a dict of the knobs that option brings (a mode and its settings). The value
 * names the option under `tag` (default `'kind'`) and carries the option's values beside it.
 *
 * @param options - The options by name, each a dict space (empty for an option with no knobs).
 * @param meta - The tag key, the default option and metadata.
 * @returns The space.
 * @throws MdmlError with code `value` for no options, a default that is not an option, or an option with a knob named
 *   like the tag.
 * @concept spaces
 * @example
 * ```ts
 * import { defaults, dict, int, real, union } from 'mdml-engine/space'
 * const solver = union({ irls: dict({ tol: real(1e-10, 1e-2, { log: true }) }), sgd: dict({ batch: int(1, 256) }) })
 * defaults(solver) // => { kind: 'irls', tol: 1e-6 }
 * ```
 */
export function union(
  options: Readonly<Record<string, DictSpace>>,
  meta: Omit<UnionSpace, 'kind' | 'options'> = {},
): UnionSpace {
  const names = Object.keys(options)
  if (names.length === 0) throw new MdmlError('value', 'union: need at least one option')
  const tag = meta.tag ?? 'kind'
  if (meta.default !== undefined && !(meta.default in options))
    throw new MdmlError('value', `union: default '${meta.default}' is not an option`)
  for (const [name, option] of Object.entries(options))
    if (tag in option.spaces)
      throw new MdmlError('value', `union: option '${name}' has a knob named like the tag '${tag}'`)
  return { kind: 'union', options, ...meta }
}
