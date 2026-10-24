import { WhiteboxError } from '../base/index.ts'
import { sizeOf } from '../tensor/index.ts'
import { activeKeysOf, conditionHolds, resolveDict } from './_active.ts'
import { defaults, flatDim } from './operations.ts'
import type { Space, SpaceValue } from './types.ts'

function elementBounds(bound: number | readonly number[], i: number): number {
  return typeof bound === 'number' ? bound : (bound[i] ?? 0)
}

function unit(lo: number, hi: number, x: number, log = false): number {
  if (hi === lo) return 0.5
  return log ? (Math.log(x) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (x - lo) / (hi - lo)
}

function fromUnit(lo: number, hi: number, u: number, log = false): number {
  const c = Math.min(1, Math.max(0, u))
  return log ? Math.exp(Math.log(lo) + c * (Math.log(hi) - Math.log(lo))) : lo + c * (hi - lo)
}

function snap(low: number, high: number, step: number | undefined, x: number): number {
  return step === undefined ? x : Math.min(high, low + Math.round((x - low) / step) * step)
}

/**
 * Map a configuration to the unit cube, the representation Bayesian optimisation and other model-based tuners work in:
 * log dimensions in log space, choices and discrete values by index, booleans and binary coordinates as 0 or 1, boxes
 * element by element. Inactive conditional keys encode as 0.5 per coordinate, so every value of a space has
 * `encodedDim(space)` coordinates.
 *
 * @param space - The space.
 * @param value - The value.
 * @returns Coordinates in [0, 1].
 * @concept spaces
 * @example
 * ```ts
 * import { encode, dict, real } from 'whitebox-engine/space'
 * encode(dict({ lr: real(1e-4, 1, { log: true }) }), { lr: 0.01 }) // => [0.5]
 * ```
 */
export function encode(space: Space, value: SpaceValue): number[] {
  switch (space.kind) {
    case 'real':
    case 'int':
      return [unit(space.low, space.high, value as number, space.log)]
    case 'choice':
      return [space.options.length === 1 ? 0.5 : space.options.indexOf(value as never) / (space.options.length - 1)]
    case 'bool':
      return [value ? 1 : 0]
    case 'discrete':
      return [space.n === 1 ? 0.5 : ((value as number) - (space.start ?? 0)) / (space.n - 1)]
    case 'box':
      return (value as readonly number[]).map((v, i) =>
        unit(elementBounds(space.low, i), elementBounds(space.high, i), v),
      )
    case 'multiDiscrete':
      return (value as readonly number[]).map((v, i) => {
        const n = space.nvec[i] ?? 1
        return n === 1 ? 0.5 : v / (n - 1)
      })
    case 'multiBinary':
      return [...(value as readonly number[])]
    case 'tuple':
      return space.spaces.flatMap((s, i) => encode(s, (value as readonly SpaceValue[])[i] as SpaceValue))
    case 'dict': {
      const record = value as Readonly<Record<string, SpaceValue>>
      return Object.entries(space.spaces).flatMap(([k, s]) =>
        k in record ? encode(s, record[k] as SpaceValue) : new Array<number>(flatDim(s)).fill(0.5),
      )
    }
  }
}

/**
 * Map unit-cube coordinates back to a configuration (rounding integers, indices and steps; conditional keys are
 * dropped when inactive). `decode(space, encode(space, v))` gives `v` back for members of the space.
 *
 * @param space - The space.
 * @param u - Coordinates in [0, 1] (clamped).
 * @returns The value.
 * @example
 * ```ts
 * import { decode, dict, int, choice } from 'whitebox-engine/space'
 * decode(dict({ k: int(1, 9), act: choice(['relu', 'tanh']) }), [0.5, 1]) // => { k: 5, act: 'tanh' }
 * ```
 */
export function decode(space: Space, u: readonly number[]): SpaceValue {
  let offset = 0
  const next = (): number => u[offset++] ?? 0.5
  const build = (s: Space): SpaceValue => {
    switch (s.kind) {
      case 'real':
        return snap(s.low, s.high, s.step, fromUnit(s.low, s.high, next(), s.log))
      case 'int':
        return Math.min(s.high, Math.max(s.low, Math.round(fromUnit(s.low, s.high, next(), s.log))))
      case 'choice':
        return s.options[Math.round(Math.min(1, Math.max(0, next())) * (s.options.length - 1))] as SpaceValue
      case 'bool':
        return next() >= 0.5
      case 'discrete':
        return (s.start ?? 0) + Math.round(Math.min(1, Math.max(0, next())) * (s.n - 1))
      case 'box':
        return Array.from({ length: sizeOf(s.shape) }, (_, i) =>
          fromUnit(elementBounds(s.low, i), elementBounds(s.high, i), next()),
        )
      case 'multiDiscrete':
        return s.nvec.map((n) => Math.round(Math.min(1, Math.max(0, next())) * (n - 1)))
      case 'multiBinary':
        return Array.from({ length: s.n }, () => (next() >= 0.5 ? 1 : 0))
      case 'tuple':
        return s.spaces.map(build)
      case 'dict': {
        // Decode every key in order (so offsets line up), then keep the ones the decoded values make active.
        const all: Record<string, SpaceValue> = {}
        for (const [k, c] of Object.entries(s.spaces)) all[k] = build(c)
        return resolveDict(s, (k) => all[k] as SpaceValue)
      }
    }
  }
  return build(space)
}

/**
 * Every configuration of a grid: `points` values per real or integer dimension (spread evenly in the unit-cube
 * encoding, so log dimensions are log-spaced, and deduplicated for integers), every option of choices, both booleans,
 * every value of discrete spaces, and the product over tuples and dicts. Conditional keys vary only where they are
 * active.
 *
 * @param space - A space without boxes or vector spaces (those have no finite grid).
 * @param points - Points per continuous dimension (default 5).
 * @returns The configurations, in row-major order.
 * @throws WhiteboxError with code `value` for box, multi-discrete and multi-binary spaces.
 * @example
 * ```ts
 * import { grid, dict, real, bool } from 'whitebox-engine/space'
 * grid(dict({ a: real(0, 1), b: bool() }), 3).length // => 6
 * ```
 */
export function grid(space: Space, points = 5): SpaceValue[] {
  const ticks = (s: Space): SpaceValue[] =>
    Array.from({ length: points }, (_, i) => decode(s, [points === 1 ? 0.5 : i / (points - 1)]))
  switch (space.kind) {
    case 'choice':
      return [...space.options]
    case 'bool':
      return [false, true]
    case 'discrete':
      return Array.from({ length: space.n }, (_, i) => (space.start ?? 0) + i)
    case 'int':
    case 'real':
      return [...new Set(ticks(space))]
    case 'tuple': {
      let rows: SpaceValue[][] = [[]]
      for (const s of space.spaces) {
        const axis = grid(s, points)
        rows = rows.flatMap((r) => axis.map((v) => [...r, v]))
      }
      return rows
    }
    case 'dict': {
      let configs: Record<string, SpaceValue>[] = [{}]
      // Repeat passes so that keys conditional on later keys are expanded once their condition is known.
      const keys = Object.keys(space.spaces)
      const done = new Set<string>()
      let progress = true
      while (progress) {
        progress = false
        for (const k of keys) {
          if (done.has(k)) continue
          const s = space.spaces[k] as Space
          if (s.when !== undefined && !done.has(s.when.key)) continue
          done.add(k)
          progress = true
          const axis = grid(s, points)
          configs = configs.flatMap((c) => (conditionHolds(s, c) ? axis.map((v) => ({ ...c, [k]: v })) : [c]))
        }
      }
      // Restore the space's key order.
      return configs.map((c) => resolveDict(space, (k) => c[k] as SpaceValue))
    }
    default:
      throw new WhiteboxError('value', `grid: ${space.kind} spaces have no grid`)
  }
}

/**
 * Distance between two configurations in the unit-cube encoding (Euclidean).
 *
 * @param space - The space.
 * @param a - First value.
 * @param b - Second value.
 * @returns The distance.
 * @example
 * ```ts
 * import { distance, real } from 'whitebox-engine/space'
 * distance(real(0, 10), 0, 10) // => 1
 * ```
 */
export function distance(space: Space, a: SpaceValue, b: SpaceValue): number {
  const x = encode(space, a)
  const y = encode(space, b)
  let s = 0
  for (let i = 0; i < x.length; i++) s += ((x[i] ?? 0) - (y[i] ?? 0)) ** 2
  return Math.sqrt(s)
}

/**
 * The dimension of the unit-cube encoding (the same as the flat dimension).
 *
 * @param space - The space.
 * @returns The number of coordinates.
 * @example
 * ```ts
 * import { encodedDim, dict, real, bool } from 'whitebox-engine/space'
 * encodedDim(dict({ a: real(0, 1), b: bool() })) // => 2
 * ```
 */
export function encodedDim(space: Space): number {
  return flatDim(space)
}

function sameValue(a: SpaceValue, b: SpaceValue): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b)
}

function unique(values: readonly SpaceValue[], current: SpaceValue): SpaceValue[] {
  const out: SpaceValue[] = []
  for (const v of values) if (!sameValue(v, current) && !out.some((o) => sameValue(o, v))) out.push(v)
  return out
}

function realSteps(lo: number, hi: number, x: number, points: number, log = false): number[] {
  const u = unit(lo, hi, x, log)
  const h = 1 / Math.max(1, points - 1)
  return [u - h, u + h].filter((v) => v >= -1e-12 && v <= 1 + 1e-12).map((v) => fromUnit(lo, hi, v, log))
}

function neighboursOf(s: Space, v: SpaceValue, points: number): SpaceValue[] {
  switch (s.kind) {
    case 'real': {
      const x = v as number
      const moves =
        s.step === undefined
          ? realSteps(s.low, s.high, x, points, s.log)
          : [x - s.step, x + s.step].filter((y) => y >= s.low - 1e-12 && y <= s.high + 1e-12)
      return unique(
        moves.map((y) => snap(s.low, s.high, s.step, Math.min(s.high, Math.max(s.low, y)))),
        x,
      )
    }
    case 'int':
      return unique(
        [(v as number) - 1, (v as number) + 1].filter((y) => y >= s.low && y <= s.high),
        v,
      )
    case 'discrete': {
      const start = s.start ?? 0
      return unique(
        [(v as number) - 1, (v as number) + 1].filter((y) => y >= start && y < start + s.n),
        v,
      )
    }
    case 'choice':
      return unique(s.options, v)
    case 'bool':
      return [!v]
    case 'box': {
      const xs = v as readonly number[]
      return xs.flatMap((x, i) =>
        realSteps(elementBounds(s.low, i), elementBounds(s.high, i), x, points)
          .filter((y) => y !== x)
          .map((y) => xs.map((old, j) => (j === i ? y : old))),
      )
    }
    case 'multiDiscrete': {
      const xs = v as readonly number[]
      return xs.flatMap((x, i) =>
        [x - 1, x + 1]
          .filter((y) => y >= 0 && y < (s.nvec[i] ?? 0))
          .map((y) => xs.map((old, j) => (j === i ? y : old))),
      )
    }
    case 'multiBinary': {
      const xs = v as readonly number[]
      return xs.map((_, i) => xs.map((old, j) => (j === i ? 1 - old : old)))
    }
    case 'tuple': {
      const xs = v as readonly SpaceValue[]
      return s.spaces.flatMap((c, i) =>
        neighboursOf(c, xs[i] as SpaceValue, points).map((y) => xs.map((old, j) => (j === i ? y : old))),
      )
    }
    case 'dict': {
      const record = v as Readonly<Record<string, SpaceValue>>
      const out: SpaceValue[] = []
      for (const k of activeKeysOf(s, record)) {
        if (!(k in record)) continue
        for (const y of neighboursOf(s.spaces[k] as Space, record[k] as SpaceValue, points)) {
          // Change one key, then repair: keys that become inactive go, keys that become active get their defaults.
          const moved = { ...record, [k]: y }
          out.push(resolveDict(s, (key, c) => (key in moved ? (moved[key] as SpaceValue) : defaults(c))))
        }
      }
      return out
    }
  }
}

/**
 * The neighbourhood of a configuration for local search (hill climbing, simulated annealing, tabu search): every
 * value that differs in one active dimension by one move. Reals step to the adjacent point of a `points`-point grid in
 * their unit-cube encoding (or by their `step`), integers and discrete values by one, booleans and binary coordinates
 * flip, choices change to each other option, boxes and multi-discrete spaces move one coordinate. Changing a key that
 * controls conditional dimensions repairs the configuration: dimensions that become inactive are dropped and ones that
 * become active take their defaults. Every neighbour belongs to the space.
 *
 * @param space - The space.
 * @param value - A member of the space.
 * @param options - Grid points per real dimension (default 11, so steps of a tenth of the range).
 * @returns The neighbours, dimension by dimension.
 * @throws WhiteboxError with code `value` when `points` is not an integer of at least 2.
 * @concept spaces
 * @example
 * ```ts
 * import { neighbours, dict, int, bool, choice } from 'whitebox-engine/space'
 * neighbours(dict({ k: int(1, 9), flag: bool() }), { k: 1, flag: false }) // => [{ k: 2, flag: false }, { k: 1, flag: true }]
 * const s = dict({ kind: choice(['a', 'b']), depth: int(1, 3, { when: { key: 'kind', in: ['b'] } }) })
 * neighbours(s, { kind: 'a' }) // => [{ kind: 'b', depth: 2 }]
 * ```
 */
export function neighbours(space: Space, value: SpaceValue, options: { readonly points?: number } = {}): SpaceValue[] {
  const points = options.points ?? 11
  if (!(Number.isInteger(points) && points >= 2)) {
    throw new WhiteboxError('value', 'neighbours: points must be an integer of at least 2')
  }
  return neighboursOf(space, value, points)
}
