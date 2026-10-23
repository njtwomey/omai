import { WhiteboxError } from '../base/index.ts'
import { flatDim } from './operations.ts'
import type { Space, SpaceValue } from './types.ts'

function scalarEncode(s: Space, v: SpaceValue): number {
  switch (s.kind) {
    case 'real':
    case 'int': {
      if (s.high === s.low) return 0.5
      const x = v as number
      return s.log
        ? (Math.log(x) - Math.log(s.low)) / (Math.log(s.high) - Math.log(s.low))
        : (x - s.low) / (s.high - s.low)
    }
    case 'choice':
      return s.options.length === 1 ? 0.5 : s.options.indexOf(v as never) / (s.options.length - 1)
    case 'bool':
      return v ? 1 : 0
    case 'discrete':
      return s.n === 1 ? 0.5 : ((v as number) - (s.start ?? 0)) / (s.n - 1)
    default:
      throw new WhiteboxError('value', `encode: ${s.kind} is not a scalar space`)
  }
}

function scalarDecode(s: Space, u: number): SpaceValue {
  const c = Math.min(1, Math.max(0, u))
  switch (s.kind) {
    case 'real': {
      const x = s.log
        ? Math.exp(Math.log(s.low) + c * (Math.log(s.high) - Math.log(s.low)))
        : s.low + c * (s.high - s.low)
      return s.step === undefined ? x : Math.min(s.high, s.low + Math.round((x - s.low) / s.step) * s.step)
    }
    case 'int': {
      const x = s.log
        ? Math.exp(Math.log(s.low) + c * (Math.log(s.high) - Math.log(s.low)))
        : s.low + c * (s.high - s.low)
      return Math.min(s.high, Math.max(s.low, Math.round(x)))
    }
    case 'choice':
      return s.options[Math.round(c * (s.options.length - 1))] as SpaceValue
    case 'bool':
      return c >= 0.5
    case 'discrete':
      return (s.start ?? 0) + Math.round(c * (s.n - 1))
    default:
      throw new WhiteboxError('value', `decode: ${s.kind} is not a scalar space`)
  }
}

/**
 * Map a configuration to the unit cube (log dimensions in log space, choices by index), the representation Bayesian
 * optimisation and other model-based tuners work in. Inactive conditional keys encode as 0.5.
 *
 * @param space - A scalar space, or a dict of scalar spaces.
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
  if (space.kind === 'dict') {
    const record = value as Readonly<Record<string, SpaceValue>>
    return Object.entries(space.spaces).flatMap(([k, s]) => (k in record ? encode(s, record[k] as SpaceValue) : [0.5]))
  }
  if (space.kind === 'tuple') return space.spaces.flatMap((s, i) => encode(s, (value as SpaceValue[])[i] as SpaceValue))
  return [scalarEncode(space, value)]
}

/**
 * Map unit-cube coordinates back to a configuration (rounding integers and choices; conditional keys are dropped when
 * inactive).
 *
 * @param space - A scalar space, or a dict or tuple of scalar spaces.
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
  const build = (s: Space): SpaceValue => {
    if (s.kind === 'dict') {
      const out: Record<string, SpaceValue> = {}
      const entries = Object.entries(s.spaces)
      const values = entries.map(([, c]) => build(c))
      entries.forEach(([k, c], i) => {
        if (c.when === undefined) out[k] = values[i] as SpaceValue
      })
      entries.forEach(([k, c], i) => {
        if (c.when?.in.includes(out[c.when.key] as string | number | boolean)) out[k] = values[i] as SpaceValue
      })
      return out
    }
    if (s.kind === 'tuple') return s.spaces.map(build)
    return scalarDecode(s, u[offset++] ?? 0.5)
  }
  return build(space)
}

/**
 * Every configuration of a grid with `points` values per real or integer dimension (all options of choices).
 *
 * @param space - A scalar space or a dict of scalar spaces.
 * @param points - Points per continuous dimension (default 5).
 * @returns The configurations, in row-major order.
 * @example
 * ```ts
 * import { grid, dict, real, bool } from 'whitebox-engine/space'
 * grid(dict({ a: real(0, 1), b: bool() }), 3).length // => 6
 * ```
 */
export function grid(space: Space, points = 5): SpaceValue[] {
  const axis = (s: Space): SpaceValue[] => {
    switch (s.kind) {
      case 'choice':
        return [...s.options]
      case 'bool':
        return [false, true]
      case 'discrete':
        return Array.from({ length: s.n }, (_, i) => (s.start ?? 0) + i)
      case 'int': {
        const values = Array.from(
          { length: points },
          (_, i) => scalarDecode(s, points === 1 ? 0.5 : i / (points - 1)) as number,
        )
        return [...new Set(values)]
      }
      case 'real':
        return Array.from({ length: points }, (_, i) => scalarDecode(s, points === 1 ? 0.5 : i / (points - 1)))
      default:
        throw new WhiteboxError('value', `grid: ${s.kind} spaces have no grid`)
    }
  }
  if (space.kind !== 'dict') return axis(space)
  let configs: Record<string, SpaceValue>[] = [{}]
  for (const [k, s] of Object.entries(space.spaces)) {
    configs = configs.flatMap((c) =>
      s.when !== undefined && !s.when.in.includes(c[s.when.key] as string | number | boolean)
        ? [c]
        : axis(s).map((v) => ({ ...c, [k]: v })),
    )
  }
  return configs
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
 * The dimension of the unit-cube encoding.
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
  return space.kind === 'dict' || space.kind === 'tuple' ? flatDim(space) : 1
}
