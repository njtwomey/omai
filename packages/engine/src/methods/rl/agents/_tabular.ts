// Shared plumbing of the tabular agents (private to methods/rl/agents): observations to state indices, step sizes,
// exploration schedules, random tie-breaking and copy-on-write tables.
import { maxOf, WhiteboxError } from '../../../lib/base/index.ts'
import type { Schedule } from '../../../lib/optim/index.ts'
import { foldIn, type Key, uniform } from '../../../lib/random/index.ts'
import type { Space } from '../../../lib/space/index.ts'

/** What a tabular agent needs from an environment: its spaces (and, for type inference, its observations). */
export interface TabularEnvLike<O> {
  readonly observationSpace: Space
  readonly actionSpace: Space
  reset?(key: Key): { readonly observation: O }
}

/** How observations map to rows of a table. */
export interface TabularCoding<O> {
  readonly nStates: number
  readonly nActions: number
  encode(observation: O): number
}

/** Options shared by every tabular agent for mapping observations to table rows. */
export interface TabularCodingOptions<O> {
  /** Map an observation to a state index in `[0, nStates)` (default: discrete, tuple-of-discrete or multi-discrete). */
  readonly encode?: (observation: O) => number
  /** The number of states (default from the observation space). */
  readonly nStates?: number
  /** The number of actions (default from the discrete action space). */
  readonly nActions?: number
}

function radices(space: Space): number[] | undefined {
  if (space.kind === 'discrete') return [space.n]
  if (space.kind === 'multiDiscrete') return [...space.nvec]
  if (space.kind === 'tuple') {
    const out: number[] = []
    for (const s of space.spaces) {
      if (s.kind !== 'discrete') return undefined
      out.push(s.n)
    }
    return out
  }
  return undefined
}

/**
 * The coding of a tabular agent: the given `encode`, or a mixed-radix code of a discrete, tuple-of-discrete or
 * multi-discrete observation space (`[a, b, c]` in spaces of sizes `[na, nb, nc]` is `(a nb + b) nc + c`).
 */
export function tabularCoding<O>(
  env: TabularEnvLike<O>,
  options: TabularCodingOptions<O>,
  name: string,
): TabularCoding<O> {
  const action = env.actionSpace
  const nActions = options.nActions ?? (action.kind === 'discrete' ? action.n : undefined)
  if (nActions === undefined) throw new WhiteboxError('value', `${name}: needs a discrete action space or nActions`)
  const space = env.observationSpace
  const r = radices(space)
  const start = space.kind === 'discrete' ? (space.start ?? 0) : 0
  let nStates = options.nStates
  if (nStates === undefined && r !== undefined) nStates = r.reduce((a, b) => a * b, 1)
  if (nStates === undefined) throw new WhiteboxError('value', `${name}: give nStates for this observation space`)
  if (options.encode !== undefined) return { nStates, nActions, encode: options.encode }
  if (r === undefined) throw new WhiteboxError('value', `${name}: give encode for this observation space`)
  const encode = (o: O): number => {
    if (typeof o === 'number') return o - start
    const xs = o as unknown as readonly number[]
    let code = 0
    for (const [i, n] of r.entries()) code = code * n + (xs[i] ?? 0)
    return code
  }
  return { nStates, nActions, encode }
}

/** A step size: a constant, sample averages (`1 / n`) or any function of the visit count `n`. */
export type StepSize = number | 'sample-average' | ((visits: number) => number)

/** The step size after `visits` visits (counting the current one). */
export function stepSizeAt(alpha: StepSize, visits: number): number {
  if (typeof alpha === 'number') return alpha
  if (alpha === 'sample-average') return 1 / Math.max(visits, 1)
  return alpha(visits)
}

/** A rate (exploration or step size) as a constant or a schedule. */
export type Rate = number | Schedule

/** The value of a rate at a count. */
export function rateAt(rate: Rate, count: number): number {
  return typeof rate === 'number' ? rate : rate(count)
}

/** A uniform number in [0, 1) from a key. */
export function unit(k: Key): number {
  return uniform(k, []).data[0] ?? 0
}

/** A uniformly random index below `n`. */
export function randomIndex(k: Key, n: number): number {
  return Math.min(n - 1, Math.floor(unit(k) * n))
}

/** The maximising index of `values[offset .. offset + n)`, ties broken uniformly at random by the key. */
export function argmaxRandom(values: ArrayLike<number>, offset: number, n: number, k: Key): number {
  let best = -Infinity
  let count = 0
  let choice = 0
  const u = uniform(k, [n]).data
  // Reservoir sampling over the maximisers: one pass, every maximiser equally likely.
  for (let a = 0; a < n; a++) {
    const v = values[offset + a] ?? -Infinity
    if (v > best) {
      best = v
      count = 1
      choice = a
    } else if (v === best) {
      count += 1
      if ((u[a] ?? 0) * count < 1) choice = a
    }
  }
  return choice
}

/** Row `s` of a flat `[nStates * nActions]` table (a view). */
export function row(table: Float64Array, s: number, nA: number): Float64Array {
  return table.subarray(s * nA, (s + 1) * nA)
}

/**
 * The value of row `s` under an epsilon-greedy policy: `(1 - epsilon) max_a Q(s, a) + epsilon mean_a Q(s, a)` (the
 * expectation does not depend on how ties are broken).
 */
export function epsilonGreedyValue(Q: Float64Array, s: number, nA: number, epsilon: number): number {
  let total = 0
  for (let a = 0; a < nA; a++) total += Q[s * nA + a] ?? 0
  return (1 - epsilon) * maxOf(row(Q, s, nA)) + (epsilon * total) / nA
}

/** Epsilon-greedy action of row `s`: uniform with probability epsilon, else a greedy action (random ties). */
export function epsilonGreedyRow(Q: ArrayLike<number>, s: number, nA: number, epsilon: number, k: Key): number {
  if (epsilon > 0 && unit(foldIn(k, 0)) < epsilon) return randomIndex(foldIn(k, 1), nA)
  return argmaxRandom(Q, s * nA, nA, foldIn(k, 2))
}

/** A copy of a table with one entry replaced. */
export function withEntry(table: Float64Array, index: number, value: number): Float64Array {
  const out = table.slice()
  out[index] = value
  return out
}

/** A table filled with a value. */
export function filled(n: number, value: number): Float64Array {
  return new Float64Array(n).fill(value)
}

/** A flat `[rows * cols]` table as nested rows. */
export function nested(table: ArrayLike<number>, rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => table[i * cols + j] ?? 0))
}
