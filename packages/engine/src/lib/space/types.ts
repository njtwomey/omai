import { WhiteboxError } from '../base/index.ts'

/** Descriptive metadata every space can carry (for documentation, controls and reports). */
export interface SpaceMeta {
  readonly label?: string
  readonly description?: string
  readonly unit?: string
  /** Active only when a sibling key of the enclosing dict has one of these values. */
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

/** A box of real numbers with a shape (gymnasium `Box`). */
export interface BoxSpace extends SpaceMeta {
  readonly kind: 'box'
  readonly low: number | readonly number[]
  readonly high: number | readonly number[]
  readonly shape: readonly number[]
}

/** The integers `start, ..., start + n - 1` (gymnasium `Discrete`). */
export interface DiscreteSpace extends SpaceMeta {
  readonly kind: 'discrete'
  readonly n: number
  readonly start?: number
}

/** Several discrete coordinates (gymnasium `MultiDiscrete`). */
export interface MultiDiscreteSpace extends SpaceMeta {
  readonly kind: 'multiDiscrete'
  readonly nvec: readonly number[]
}

/** Binary vectors of length `n` (gymnasium `MultiBinary`). */
export interface MultiBinarySpace extends SpaceMeta {
  readonly kind: 'multiBinary'
  readonly n: number
}

/** A fixed-length tuple of spaces. */
export interface TupleSpace extends SpaceMeta {
  readonly kind: 'tuple'
  readonly spaces: readonly Space[]
}

/** Named sub-spaces, with optional conditional activation (`when` on a child). */
export interface DictSpace extends SpaceMeta {
  readonly kind: 'dict'
  readonly spaces: Readonly<Record<string, Space>>
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

/** A value of a space. */
export type SpaceValue =
  | number
  | string
  | boolean
  | readonly number[]
  | readonly SpaceValue[]
  | { readonly [k: string]: SpaceValue }

function checkInterval(low: number, high: number, log: boolean | undefined, name: string): void {
  if (!(high >= low)) throw new WhiteboxError('value', `${name}: need high >= low, got [${low}, ${high}]`)
  if (log === true && !(low > 0)) throw new WhiteboxError('value', `${name}: a log scale needs low > 0`)
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
 * import { real } from 'whitebox-engine/space'
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
 * import { int } from 'whitebox-engine/space'
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
 * import { choice } from 'whitebox-engine/space'
 * choice(['relu', 'tanh']).options.length // => 2
 * ```
 */
export function choice<T extends string | number | boolean>(
  options: readonly T[],
  meta: Omit<ChoiceSpace<T>, 'kind' | 'options'> = {},
): ChoiceSpace<T> {
  if (options.length === 0) throw new WhiteboxError('value', 'choice: need at least one option')
  return { kind: 'choice', options, ...meta }
}

/**
 * A boolean flag.
 *
 * @param meta - Default and metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { bool } from 'whitebox-engine/space'
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
 * @param meta - Metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { box } from 'whitebox-engine/space'
 * box(-1, 1, [2]).shape // => [2]
 * ```
 */
export function box(
  low: number | readonly number[],
  high: number | readonly number[],
  shape: readonly number[],
  meta: SpaceMeta = {},
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
 * import { discrete } from 'whitebox-engine/space'
 * discrete(4).n // => 4
 * ```
 */
export function discrete(n: number, options: Omit<DiscreteSpace, 'kind' | 'n'> = {}): DiscreteSpace {
  if (!Number.isInteger(n) || n < 1) throw new WhiteboxError('value', 'discrete: n must be a positive integer')
  return { kind: 'discrete', n, ...options }
}

/**
 * Several discrete coordinates.
 *
 * @param nvec - Number of values per coordinate.
 * @param meta - Metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { multiDiscrete } from 'whitebox-engine/space'
 * multiDiscrete([3, 2]).nvec // => [3, 2]
 * ```
 */
export function multiDiscrete(nvec: readonly number[], meta: SpaceMeta = {}): MultiDiscreteSpace {
  return { kind: 'multiDiscrete', nvec, ...meta }
}

/**
 * Binary vectors.
 *
 * @param n - Length.
 * @param meta - Metadata.
 * @returns The space.
 * @example
 * ```ts
 * import { multiBinary } from 'whitebox-engine/space'
 * multiBinary(3).n // => 3
 * ```
 */
export function multiBinary(n: number, meta: SpaceMeta = {}): MultiBinarySpace {
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
 * import { tuple, discrete } from 'whitebox-engine/space'
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
 * import { dict, real, choice } from 'whitebox-engine/space'
 * Object.keys(dict({ lr: real(1e-4, 1, { log: true }), act: choice(['relu', 'tanh']) }).spaces) // => ['lr', 'act']
 * ```
 */
export function dict(spaces: Readonly<Record<string, Space>>, meta: SpaceMeta = {}): DictSpace {
  return { kind: 'dict', spaces, ...meta }
}
