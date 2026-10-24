// The scalar parameters ("leaves") of a search space, shared by the samplers, the surrogates and the analysis. Not
// re-exported. Every leaf is one coordinate of `flatten(space, value)` and of `encode(space, value)`, in that order, so
// a sampler can work leaf by leaf in natural units (TPE), on the unit cube (quasi-random sequences, CMA-ES, Bayesian
// optimisation) or on equal-mass cells (Latin hypercubes), and hand the result back as a value through `decode`.
import { WhiteboxError } from '../base/index.ts'
import { choice, clip, decode, encode, flatten, int, real, type Space, type SpaceValue } from '../space/index.ts'
import { sizeOf } from '../tensor/index.ts'

/** One scalar parameter of a space. */
export interface Leaf {
  /** Dotted path of the parameter (`'lr'`, `'optimiser.momentum'`, `'weights[2]'`). */
  readonly path: string
  /** Real numbers, integers, or unordered options (choices, booleans, binary coordinates). */
  readonly kind: 'float' | 'int' | 'categorical'
  /** Bounds in natural units (for categorical leaves, 0 and n - 1). */
  readonly low: number
  readonly high: number
  readonly log: boolean
  /** The grid step of a stepped real leaf (integers have step 1). */
  readonly step?: number
  /** Number of options of a categorical leaf (0 otherwise). */
  readonly n: number
  /** The scalar space of the leaf, for its unit-cube encoding. */
  readonly space: Space
  /** The value of the leaf for a flat number (an option for an index, a boolean for 0 or 1). */
  readonly value: (x: number) => SpaceValue
}

function scalarLeaf(path: string, s: Space): Leaf {
  switch (s.kind) {
    case 'real':
      return {
        path,
        kind: 'float',
        low: s.low,
        high: s.high,
        log: s.log === true,
        ...(s.step === undefined ? {} : { step: s.step }),
        n: 0,
        space: s,
        value: (x) => x,
      }
    case 'int':
      return {
        path,
        kind: 'int',
        low: s.low,
        high: s.high,
        log: s.log === true,
        step: 1,
        n: 0,
        space: s,
        value: (x) => x,
      }
    case 'discrete': {
      const low = s.start ?? 0
      return { path, kind: 'int', low, high: low + s.n - 1, log: false, step: 1, n: 0, space: s, value: (x) => x }
    }
    case 'choice':
      return {
        path,
        kind: 'categorical',
        low: 0,
        high: s.options.length - 1,
        log: false,
        n: s.options.length,
        space: s,
        value: (x) => s.options[x] as SpaceValue,
      }
    case 'bool':
      return { path, kind: 'categorical', low: 0, high: 1, log: false, n: 2, space: s, value: (x) => x === 1 }
    default:
      throw new WhiteboxError('value', `tune: ${s.kind} is not a scalar space`)
  }
}

/**
 * The leaves of a space in flattening order: dict keys and tuple items recursively, box elements as reals, binary
 * coordinates as two-option leaves and multi-discrete coordinates as integers.
 */
export function leavesOf(space: Space): Leaf[] {
  const out: Leaf[] = []
  const walk = (s: Space, path: string): void => {
    const at = (name: string): string => (path === '' ? name : `${path}.${name}`)
    switch (s.kind) {
      case 'dict':
        for (const [k, c] of Object.entries(s.spaces)) walk(c, at(k))
        return
      case 'tuple':
        for (const [i, c] of s.spaces.entries()) walk(c, at(String(i)))
        return
      case 'box': {
        const size = sizeOf(s.shape)
        for (let i = 0; i < size; i++) {
          const lo = typeof s.low === 'number' ? s.low : (s.low[i] ?? 0)
          const hi = typeof s.high === 'number' ? s.high : (s.high[i] ?? 0)
          out.push(scalarLeaf(`${path}[${i}]`, real(lo, hi)))
        }
        return
      }
      case 'multiDiscrete':
        for (const [i, n] of s.nvec.entries()) out.push(scalarLeaf(`${path}[${i}]`, int(0, n - 1)))
        return
      case 'multiBinary':
        for (let i = 0; i < s.n; i++) out.push(scalarLeaf(`${path}[${i}]`, choice([0, 1])))
        return
      default:
        out.push(scalarLeaf(path, s))
    }
  }
  walk(space, '')
  return out
}

/** The flat numbers of a value, leaf by leaf: natural units, option indices, NaN for inactive leaves. */
export function nativeOf(space: Space, value: SpaceValue): number[] {
  return flatten(space, value)
}

/** The unit-cube coordinate of a leaf at a flat number (0.5 for NaN, an inactive leaf). */
export function leafCoordinate(leaf: Leaf, x: number): number {
  if (Number.isNaN(x)) return 0.5
  return encode(leaf.space, leaf.value(x))[0] ?? 0.5
}

/**
 * `decode`, then `clip`: the exponential of a log-scaled end point can round just outside its interval (exp(log(1e-5))
 * is 9.999999999999999e-6), so decoded values are clipped back into the space.
 */
export function decodeInto(space: Space, u: readonly number[]): SpaceValue {
  return clip(space, decode(space, u))
}

/** The value of a space for flat numbers per leaf (NaN leaves are left to `decode`'s default). */
export function assemble(space: Space, leaves: readonly Leaf[], natives: readonly number[]): SpaceValue {
  return decodeInto(
    space,
    leaves.map((l, i) => leafCoordinate(l, natives[i] ?? Number.NaN)),
  )
}

/**
 * The unit-cube coordinate of an equal-mass cell: integers and options get cells of equal width (log-width for log
 * integers), as `space.sample` draws them, so a uniform point of the cube is a uniform draw of the space.
 */
export function equalMassCoordinate(leaf: Leaf, u: number): number {
  if (leaf.kind === 'float') return u
  if (leaf.kind === 'categorical') return leafCoordinate(leaf, Math.min(leaf.n - 1, Math.floor(u * leaf.n)))
  const v = leaf.log
    ? Math.floor(Math.exp(Math.log(leaf.low) + u * (Math.log(leaf.high + 1) - Math.log(leaf.low))))
    : leaf.low + Math.floor(u * (leaf.high - leaf.low + 1))
  return leafCoordinate(leaf, Math.min(leaf.high, Math.max(leaf.low, v)))
}

/** The value of a space at a point of the unit cube, read through equal-mass cells. */
export function fromCube(space: Space, leaves: readonly Leaf[], u: ArrayLike<number>): SpaceValue {
  return decodeInto(
    space,
    leaves.map((l, i) => equalMassCoordinate(l, u[i] ?? 0.5)),
  )
}
