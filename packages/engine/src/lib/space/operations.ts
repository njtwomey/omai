/**
 * Operations on any space: sampling a value (log-scaled dimensions uniformly in log space, conditional keys only when
 * active), membership tests, clipping into the space, defaults, the active dimensions of a value, and flattening to and
 * from a vector of numbers.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { sizeOf } from '../tensor/index.ts'
import { activeKeysOf, resolveDict } from './_active.ts'
import { joinUnion, optionNames, splitUnion, tagOf } from './_union.ts'
import type { DictSpace, Space, SpaceValue } from './types.ts'

type Rec = Readonly<Record<string, SpaceValue>>

function u01(k: Key, n: number): number[] {
  return Array.from(uniform(k, [n]).data)
}

function elementBounds(bound: number | readonly number[], i: number): number {
  return typeof bound === 'number' ? bound : (bound[i] ?? 0)
}

function snap(low: number, high: number, step: number | undefined, x: number): number {
  return step === undefined ? x : Math.min(high, low + Math.round((x - low) / step) * step)
}

/**
 * The active dimensions of a value: the dotted paths (`'optimiser.lr'`, `'layers.0'`) of the leaf spaces that are
 * active for it, in flattening order. A dict child is active when its `when` condition holds; tuple elements always
 * are. A space that is neither a dict nor a tuple is the single dimension `''`; a box or vector space is one
 * dimension however many numbers it holds.
 *
 * @param space - The space.
 * @param value - A value of the space (inactive keys may be absent).
 * @returns The active paths.
 * @concept spaces
 * @example
 * ```ts
 * import { activeDims, choice, dict, int, real } from 'mdml-engine/space'
 * const s = dict({
 *   kind: choice(['linear', 'tree']),
 *   alpha: real(0, 1, { when: { key: 'kind', in: ['linear'] } }),
 *   depth: int(1, 8, { when: { key: 'kind', in: ['tree'] } }),
 * })
 * activeDims(s, { kind: 'tree', depth: 3 }) // => ['kind', 'depth']
 * ```
 */
export function activeDims(space: Space, value: SpaceValue): string[] {
  const walk = (s: Space, v: SpaceValue, prefix: string): string[] => {
    const at = (name: string): string => (prefix === '' ? name : `${prefix}.${name}`)
    if (s.kind === 'dict') {
      const record = (typeof v === 'object' && v !== null && !Array.isArray(v) ? v : {}) as Readonly<
        Record<string, SpaceValue>
      >
      return activeKeysOf(s, record).flatMap((k) => walk(s.spaces[k] as Space, record[k] as SpaceValue, at(k)))
    }
    if (s.kind === 'tuple') {
      const items = (Array.isArray(v) ? v : []) as readonly SpaceValue[]
      return s.spaces.flatMap((c, i) => walk(c, items[i] as SpaceValue, at(String(i))))
    }
    if (s.kind === 'union') {
      // The tag, then the chosen option's dimensions under the option's name (`solver.kind`, `solver.sgd.batch`).
      const { name, option, rest } = splitUnion(s, v)
      return [at(tagOf(s)), ...walk(option, rest, at(name))]
    }
    return [prefix]
  }
  return walk(space, value, '')
}

/**
 * Draw a random value from a space (log-scaled dimensions are sampled uniformly in log space; conditional keys are
 * only drawn when active).
 *
 * @param space - The space.
 * @param k - The random key.
 * @returns A value of the space.
 * @concept spaces
 * @index function Draw a random value from a space.
 * @example
 * ```ts
 * import { sample, real } from 'mdml-engine/space'
 * import { key } from 'mdml-engine/random'
 * const v = sample(real(0, 1), key(0)) as number
 * v >= 0 && v <= 1 // => true
 * ```
 */
export function sample(space: Space, k: Key): SpaceValue {
  switch (space.kind) {
    case 'real': {
      const [u] = u01(k, 1) as [number]
      const x = space.log
        ? Math.exp(Math.log(space.low) + u * (Math.log(space.high) - Math.log(space.low)))
        : space.low + u * (space.high - space.low)
      return snap(space.low, space.high, space.step, x)
    }
    case 'int': {
      const [u] = u01(k, 1) as [number]
      if (space.log) {
        // Uniform in log space over [low, high + 1), then floored: each integer gets the log-width of its cell.
        const x = Math.exp(Math.log(space.low) + u * (Math.log(space.high + 1) - Math.log(space.low)))
        return Math.min(space.high, Math.max(space.low, Math.floor(x)))
      }
      return Math.min(space.high, space.low + Math.floor(u * (space.high - space.low + 1)))
    }
    case 'choice': {
      const [u] = u01(k, 1) as [number]
      return space.options[Math.min(space.options.length - 1, Math.floor(u * space.options.length))] as SpaceValue
    }
    case 'bool':
      return (u01(k, 1)[0] ?? 0) < 0.5
    case 'box': {
      const n = sizeOf(space.shape)
      return u01(k, n).map(
        (u, i) => elementBounds(space.low, i) + u * (elementBounds(space.high, i) - elementBounds(space.low, i)),
      )
    }
    case 'discrete':
      return (space.start ?? 0) + Math.min(space.n - 1, Math.floor((u01(k, 1)[0] ?? 0) * space.n))
    case 'multiDiscrete':
      return u01(k, space.nvec.length).map((u, i) =>
        Math.min((space.nvec[i] ?? 1) - 1, Math.floor(u * (space.nvec[i] ?? 1))),
      )
    case 'multiBinary':
      return u01(k, space.n).map((u) => (u < 0.5 ? 0 : 1))
    case 'tuple':
      return space.spaces.map((s, i) => sample(s, foldIn(k, i)))
    case 'dict':
      // Each key draws from its own key (by position), so adding a conditional key never changes the others.
      return resolveDict(space, (_, s, i) => sample(s, foldIn(k, i)))
    case 'union': {
      // The option from the key itself, its knobs from a key of its own (by position), as for dict keys.
      const names = optionNames(space)
      const [u] = u01(k, 1) as [number]
      const i = Math.min(names.length - 1, Math.floor(u * names.length))
      const name = names[i] as string
      return joinUnion(space, name, sample(space.options[name] as DictSpace, foldIn(k, 1 + i)) as Rec)
    }
  }
}

/**
 * Whether a value belongs to a space (for dicts: every active key is present and belongs to its space).
 *
 * @param space - The space.
 * @param value - The value.
 * @returns `true` when the value is a member.
 * @example
 * ```ts
 * import { contains, discrete } from 'mdml-engine/space'
 * contains(discrete(3), 2) // => true
 * contains(discrete(3), 3) // => false
 * ```
 */
export function contains(space: Space, value: SpaceValue): boolean {
  switch (space.kind) {
    case 'real':
      return typeof value === 'number' && value >= space.low && value <= space.high
    case 'int':
      return typeof value === 'number' && Number.isInteger(value) && value >= space.low && value <= space.high
    case 'choice':
      return space.options.includes(value as never)
    case 'bool':
      return typeof value === 'boolean'
    case 'box':
      return (
        Array.isArray(value) &&
        value.length === sizeOf(space.shape) &&
        (value as number[]).every((v, i) => v >= elementBounds(space.low, i) && v <= elementBounds(space.high, i))
      )
    case 'discrete': {
      const start = space.start ?? 0
      return typeof value === 'number' && Number.isInteger(value) && value >= start && value < start + space.n
    }
    case 'multiDiscrete':
      return (
        Array.isArray(value) &&
        value.length === space.nvec.length &&
        (value as number[]).every((v, i) => Number.isInteger(v) && v >= 0 && v < (space.nvec[i] ?? 0))
      )
    case 'multiBinary':
      return Array.isArray(value) && value.length === space.n && (value as number[]).every((v) => v === 0 || v === 1)
    case 'tuple':
      return (
        Array.isArray(value) &&
        value.length === space.spaces.length &&
        space.spaces.every((s, i) => contains(s, (value as SpaceValue[])[i] as SpaceValue))
      )
    case 'dict': {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
      const record = value as Readonly<Record<string, SpaceValue>>
      return activeKeysOf(space, record).every(
        (k) => k in record && contains(space.spaces[k] as Space, record[k] as SpaceValue),
      )
    }
    case 'union': {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
      const name = (value as Rec)[tagOf(space)]
      if (typeof name !== 'string' || !(name in space.options)) return false
      const { option, rest } = splitUnion(space, value)
      return contains(option, rest)
    }
  }
}

/**
 * Bring a value into a space: clamp numbers (snapping stepped reals to their grid), round integers, discrete and
 * binary coordinates, and drop inactive dict keys.
 *
 * @param space - The space.
 * @param value - The value.
 * @returns The nearest member.
 * @throws MdmlError with code `value` for a choice value that is not an option.
 * @example
 * ```ts
 * import { clip, int } from 'mdml-engine/space'
 * clip(int(0, 5), 7.6) // => 5
 * ```
 */
export function clip(space: Space, value: SpaceValue): SpaceValue {
  switch (space.kind) {
    case 'real':
      return snap(space.low, space.high, space.step, Math.min(space.high, Math.max(space.low, value as number)))
    case 'int':
      return Math.min(space.high, Math.max(space.low, Math.round(value as number)))
    case 'box':
      return (value as number[]).map((v, i) =>
        Math.min(elementBounds(space.high, i), Math.max(elementBounds(space.low, i), v)),
      )
    case 'discrete':
      return Math.min((space.start ?? 0) + space.n - 1, Math.max(space.start ?? 0, Math.round(value as number)))
    case 'multiDiscrete':
      return (value as number[]).map((v, i) => Math.min((space.nvec[i] ?? 1) - 1, Math.max(0, Math.round(v))))
    case 'multiBinary':
      return (value as number[]).map((v) => (v >= 0.5 ? 1 : 0))
    case 'tuple':
      return space.spaces.map((s, i) => clip(s, (value as SpaceValue[])[i] as SpaceValue))
    case 'dict': {
      const record = value as Readonly<Record<string, SpaceValue>>
      const out: Record<string, SpaceValue> = {}
      for (const k of activeKeysOf(space, record))
        if (k in record) out[k] = clip(space.spaces[k] as Space, record[k] as SpaceValue)
      return out
    }
    case 'choice':
      if (!space.options.includes(value as never))
        throw new MdmlError('value', `clip: ${String(value)} is not an option`)
      return value
    case 'bool':
      return value
    case 'union': {
      const name =
        typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Rec)[tagOf(space)] : undefined
      if (typeof name !== 'string' || !(name in space.options))
        throw new MdmlError('value', `clip: ${String(name)} is not an option`)
      const { option, rest } = splitUnion(space, value)
      return joinUnion(space, name, clip(option, rest) as Rec)
    }
  }
}

/**
 * The default value of a space: declared defaults, else the middle of intervals (the geometric middle when
 * log-scaled, snapped to the step), the first option, false, zeros clamped into boxes, and the first value of discrete
 * spaces. Conditional dict keys get defaults when the defaults make them active.
 *
 * @param space - The space.
 * @returns The default value.
 * @example
 * ```ts
 * import { defaults, dict, real, choice } from 'mdml-engine/space'
 * defaults(dict({ lr: real(1e-4, 1, { log: true }), act: choice(['relu', 'tanh']) })) // => { lr: 0.01, act: 'relu' }
 * ```
 */
export function defaults(space: Space): SpaceValue {
  switch (space.kind) {
    case 'real':
      return (
        space.default ??
        snap(
          space.low,
          space.high,
          space.step,
          space.log ? Math.sqrt(space.low * space.high) : (space.low + space.high) / 2,
        )
      )
    case 'int':
      return space.default ?? Math.round(space.log ? Math.sqrt(space.low * space.high) : (space.low + space.high) / 2)
    case 'choice':
      return space.default ?? (space.options[0] as SpaceValue)
    case 'bool':
      return space.default ?? false
    case 'box':
      return (
        space.default ??
        Array.from({ length: sizeOf(space.shape) }, (_, i) =>
          Math.min(Math.max(0, elementBounds(space.low, i)), elementBounds(space.high, i)),
        )
      )
    case 'discrete':
      return space.default ?? space.start ?? 0
    case 'multiDiscrete':
      return space.default ?? space.nvec.map(() => 0)
    case 'multiBinary':
      return space.default ?? new Array<number>(space.n).fill(0)
    case 'tuple':
      return space.spaces.map(defaults)
    case 'dict':
      return resolveDict(space, (_, s) => defaults(s))
    case 'union': {
      const name = space.default ?? (optionNames(space)[0] as string)
      return joinUnion(space, name, defaults(space.options[name] as DictSpace) as Rec)
    }
  }
}

/**
 * The number of numbers a flattened value of the space has (choices and discrete values count as one index).
 *
 * @param space - The space.
 * @returns The flat dimension.
 * @example
 * ```ts
 * import { flatDim, box, tuple, discrete } from 'mdml-engine/space'
 * flatDim(tuple([box(0, 1, [2, 2]), discrete(3)])) // => 5
 * ```
 */
export function flatDim(space: Space): number {
  switch (space.kind) {
    case 'box':
      return sizeOf(space.shape)
    case 'multiDiscrete':
      return space.nvec.length
    case 'multiBinary':
      return space.n
    case 'tuple':
      return space.spaces.reduce((n, s) => n + flatDim(s), 0)
    case 'dict':
      return Object.values(space.spaces).reduce((n, s) => n + flatDim(s), 0)
    case 'union':
      // The tag, then every option's dimensions in turn (the inactive ones NaN or 0.5, as inactive dict keys are).
      return 1 + Object.values(space.options).reduce((n, s) => n + flatDim(s), 0)
    default:
      return 1
  }
}

/**
 * Flatten a value to numbers (choices become their index, booleans 0 or 1; inactive dict keys become NaN).
 *
 * @param space - The space.
 * @param value - The value.
 * @returns The numbers.
 * @example
 * ```ts
 * import { flatten, dict, real, choice } from 'mdml-engine/space'
 * flatten(dict({ a: real(0, 1), b: choice(['x', 'y']) }), { a: 0.5, b: 'y' }) // => [0.5, 1]
 * ```
 */
export function flatten(space: Space, value: SpaceValue): number[] {
  switch (space.kind) {
    case 'real':
    case 'int':
    case 'discrete':
      return [value as number]
    case 'choice':
      return [space.options.indexOf(value as never)]
    case 'bool':
      return [value ? 1 : 0]
    case 'box':
    case 'multiDiscrete':
    case 'multiBinary':
      return [...(value as number[])]
    case 'tuple':
      return space.spaces.flatMap((s, i) => flatten(s, (value as SpaceValue[])[i] as SpaceValue))
    case 'dict': {
      const record = value as Readonly<Record<string, SpaceValue>>
      return Object.entries(space.spaces).flatMap(([k, s]) =>
        k in record ? flatten(s, record[k] as SpaceValue) : new Array<number>(flatDim(s)).fill(Number.NaN),
      )
    }
    case 'union': {
      const { name, rest } = splitUnion(space, value)
      return [
        optionNames(space).indexOf(name),
        ...Object.entries(space.options).flatMap(([n, s]) =>
          n === name ? flatten(s, rest) : new Array<number>(flatDim(s)).fill(Number.NaN),
        ),
      ]
    }
  }
}

/**
 * Rebuild a value from numbers (the inverse of {@link flatten}).
 *
 * @param space - The space.
 * @param numbers - The flat numbers.
 * @returns The value.
 * @example
 * ```ts
 * import { unflatten, dict, real, choice } from 'mdml-engine/space'
 * unflatten(dict({ a: real(0, 1), b: choice(['x', 'y']) }), [0.5, 1]) // => { a: 0.5, b: 'y' }
 * ```
 */
export function unflatten(space: Space, numbers: readonly number[]): SpaceValue {
  let offset = 0
  const take = (n: number): number[] => {
    const out = numbers.slice(offset, offset + n)
    offset += n
    return out
  }
  const build = (s: Space): SpaceValue | undefined => {
    switch (s.kind) {
      case 'real':
      case 'int':
      case 'discrete': {
        const [v] = take(1)
        return Number.isNaN(v) ? undefined : (v as number)
      }
      case 'choice': {
        const [v] = take(1)
        return Number.isNaN(v) ? undefined : (s.options[v as number] as SpaceValue)
      }
      case 'bool': {
        const [v] = take(1)
        return Number.isNaN(v) ? undefined : v === 1
      }
      case 'box':
      case 'multiDiscrete':
      case 'multiBinary': {
        const v = take(flatDim(s))
        return v.some(Number.isNaN) ? undefined : v
      }
      case 'tuple':
        return s.spaces.map((c) => build(c) as SpaceValue)
      case 'dict': {
        const out: Record<string, SpaceValue> = {}
        for (const [k, c] of Object.entries(s.spaces)) {
          const v = build(c)
          if (v !== undefined) out[k] = v
        }
        return out
      }
      case 'union': {
        const [idx] = take(1)
        const names = optionNames(s)
        const name = Number.isNaN(idx)
          ? undefined
          : names[Math.min(names.length - 1, Math.max(0, Math.round(idx as number)))]
        // Every option's numbers are consumed (so offsets line up); the named option's value is kept.
        let kept: SpaceValue | undefined
        for (const [n, c] of Object.entries(s.options)) {
          const v = build(c)
          if (n === name) kept = v
        }
        return name === undefined ? undefined : joinUnion(s, name, (kept ?? {}) as Rec)
      }
    }
  }
  return build(space) as SpaceValue
}
