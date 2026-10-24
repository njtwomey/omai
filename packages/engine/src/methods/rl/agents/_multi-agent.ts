// Private helpers of the multi-agent learners (methods/rl/agents/multi-agent.ts): a tabular Q-learner, the solution of
// small zero-sum matrix games, reservoir sampling and observation encoding.
import { argmaxOf, maxOf, minOf, WhiteboxError } from '../../../lib/base/index.ts'
import { solve } from '../../../lib/numerics/linalg/index.ts'
import { linearSchedule } from '../../../lib/optim/index.ts'
import { foldIn, type Key, uniform } from '../../../lib/random/index.ts'
import type { Agent, Transition } from '../../../lib/rl/index.ts'
import { flatDim, type Space } from '../../../lib/space/index.ts'
import { toFlat } from '../../../lib/tensor/index.ts'

const u1 = (k: Key): number => uniform(k, []).data[0] ?? 0

/** A uniformly random index below `n`. */
export const randomIndex = (k: Key, n: number): number => Math.min(n - 1, Math.floor(u1(k) * n))

/** Sample an index from probabilities. */
export function sampleIndex(k: Key, probs: readonly number[]): number {
  let u = u1(k)
  for (const [i, p] of probs.entries()) {
    u -= p
    if (u < 0) return i
  }
  return probs.length - 1
}

/** The table key of an observation: numbers as themselves, anything else as JSON. */
export const observationKey = (o: unknown): string => (typeof o === 'number' ? String(o) : JSON.stringify(o))

// ------------------------------------------------------------------------------------------------- tabular Q

/** Options of the tabular Q-learner used by the independent learners. */
export interface TabularQOptions {
  /** Step size (default 0.1). */
  readonly alpha?: number
  /** Discount (default 0.95). */
  readonly gamma?: number
  /** Exploration: linear from `epsilonStart` (default 1) to `epsilonEnd` (0.05) over `explorationSteps` (1000). */
  readonly epsilonStart?: number
  readonly epsilonEnd?: number
  readonly explorationSteps?: number
  /** The initial action value (default 0; optimistic values encourage exploration). */
  readonly initialValue?: number
}

/** The state of a tabular Q-learner: action values per observation key. */
export interface TabularQState {
  readonly q: Readonly<Record<string, readonly number[]>>
  readonly steps: number
}

/** Watkins' Q-learning on observations used as table keys, with epsilon-greedy exploration (pure updates). */
export function tabularQ(nActions: number, options: TabularQOptions = {}): Agent<TabularQState, unknown, number> {
  const alpha = options.alpha ?? 0.1
  const gamma = options.gamma ?? 0.95
  const init = options.initialValue ?? 0
  const epsilon = linearSchedule(
    options.epsilonStart ?? 1,
    options.epsilonEnd ?? 0.05,
    options.explorationSteps ?? 1000,
  )
  const row = (s: TabularQState, o: unknown): readonly number[] =>
    s.q[observationKey(o)] ?? new Array<number>(nActions).fill(init)
  return {
    name: 'tabularQ',
    config: { alpha, gamma, initialValue: init },
    init: () => ({ q: {}, steps: 0 }),
    act: (s, o, k) => (u1(foldIn(k, 0)) < epsilon(s.steps) ? randomIndex(foldIn(k, 1), nActions) : argmaxOf(row(s, o))),
    greedy: (s, o) => argmaxOf(row(s, o)),
    learn: (s, t: Transition<unknown, number>) => {
      const q = row(s, t.observation)
      const target = t.reward + (t.terminated ? 0 : gamma * maxOf(row(s, t.nextObservation)))
      const updated = q.map((v, a) => (a === t.action ? v + alpha * (target - v) : v))
      return { q: { ...s.q, [observationKey(t.observation)]: updated }, steps: s.steps + 1 }
    },
  }
}

// ----------------------------------------------------------------------------------------- zero-sum matrix games

/** The solution of a zero-sum matrix game. */
export interface ZeroSumSolution {
  /** The maximising row player's mixed strategy. */
  readonly row: readonly number[]
  /** The minimising column player's mixed strategy. */
  readonly column: readonly number[]
  /** The value `row^T M column`. */
  readonly value: number
  /** The duality gap `max_i (M column)_i - min_j (row^T M)_j` (zero at an exact equilibrium). */
  readonly gap: number
  readonly iterations: number
}

const subsets = (n: number, k: number): number[][] => {
  const out: number[][] = []
  const rec = (start: number, acc: number[]): void => {
    if (acc.length === k) {
      out.push(acc)
      return
    }
    for (let i = start; i < n; i++) rec(i + 1, [...acc, i])
  }
  rec(0, [])
  return out
}

/** Solve a small linear system, or `undefined` when it is singular. */
function solveSmall(a: number[][], b: number[]): number[] | undefined {
  try {
    const x = toFlat(solve(a, b))
    return x.every(Number.isFinite) ? x : undefined
  } catch {
    return undefined
  }
}

/**
 * Support enumeration for a zero-sum game: for supports of equal size, solve the indifference conditions of both
 * players and accept the first pair of non-negative strategies that are mutual best responses (exact for
 * non-degenerate games; the cost grows combinatorially, so only small games are enumerated).
 */
function supportEnumeration(at: (i: number, j: number) => number, m: number, n: number): ZeroSumSolution | undefined {
  const eps = 1e-10
  for (let k = 1; k <= Math.min(m, n); k++) {
    for (const rows of subsets(m, k)) {
      for (const cols of subsets(n, k)) {
        // Column strategy y on `cols` making every row of `rows` worth v: [M_IJ, -1; 1, 0] [y; v] = [0; 1].
        const ya = rows.map((i) => [...cols.map((j) => at(i, j)), -1])
        ya.push([...cols.map(() => 1), 0])
        const yv = solveSmall(ya, [...rows.map(() => 0), 1])
        const xa = cols.map((j) => [...rows.map((i) => at(i, j)), -1])
        xa.push([...rows.map(() => 1), 0])
        const xv = solveSmall(xa, [...cols.map(() => 0), 1])
        if (yv === undefined || xv === undefined) continue
        const v = yv[k] ?? 0
        if (yv.slice(0, k).some((p) => p < -eps) || xv.slice(0, k).some((p) => p < -eps)) continue
        const column = new Array<number>(n).fill(0)
        for (const [t, j] of cols.entries()) column[j] = Math.max(0, yv[t] ?? 0)
        const row = new Array<number>(m).fill(0)
        for (const [t, i] of rows.entries()) row[i] = Math.max(0, xv[t] ?? 0)
        let best = -Infinity
        for (let i = 0; i < m; i++) {
          let s = 0
          for (let j = 0; j < n; j++) s += at(i, j) * (column[j] ?? 0)
          best = Math.max(best, s)
        }
        let worst = Infinity
        for (let j = 0; j < n; j++) {
          let s = 0
          for (let i = 0; i < m; i++) s += (row[i] ?? 0) * at(i, j)
          worst = Math.min(worst, s)
        }
        if (best <= v + 1e-9 && worst >= v - 1e-9) return { row, column, value: v, gap: best - worst, iterations: 0 }
      }
    }
  }
  return undefined
}

// TODO: solve by linear programming (optim/linear-programming.ts linprog) once it exists; support enumeration is
// limited to small games and regret matching is exact only to its tolerance.
/**
 * Solve `max_x min_y x^T M y` over mixed strategies: exactly at a pure saddle point or by support enumeration for
 * games up to 6 x 6, otherwise by regret matching+ (Tammelin 2014) with alternating updates and linearly weighted
 * averages, which converges to an equilibrium; it stops when the duality gap of the averages falls below `tol`.
 */
export function solveZeroSum(
  matrix: readonly (readonly number[])[],
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): ZeroSumSolution {
  const m = matrix.length
  const n = matrix[0]?.length ?? 0
  if (m === 0 || n === 0) throw new WhiteboxError('value', 'solveZeroSum: empty matrix')
  const at = (i: number, j: number): number => matrix[i]?.[j] ?? 0
  if (m <= 6 && n <= 6) {
    const exact = supportEnumeration(at, m, n)
    if (exact !== undefined) return exact
  }
  const rowValues = (y: readonly number[]): number[] =>
    Array.from({ length: m }, (_, i) => {
      let v = 0
      for (let j = 0; j < n; j++) v += at(i, j) * (y[j] ?? 0)
      return v
    })
  const columnValues = (x: readonly number[]): number[] =>
    Array.from({ length: n }, (_, j) => {
      let v = 0
      for (let i = 0; i < m; i++) v += (x[i] ?? 0) * at(i, j)
      return v
    })
  // A pure saddle point: the max-min of the rows equals the min-max of the columns.
  const rowMins = Array.from({ length: m }, (_, i) => minOf(Array.from({ length: n }, (_, j) => at(i, j))))
  const colMaxs = Array.from({ length: n }, (_, j) => maxOf(Array.from({ length: m }, (_, i) => at(i, j))))
  const i0 = argmaxOf(rowMins)
  const j0 = colMaxs.indexOf(minOf(colMaxs))
  if ((rowMins[i0] ?? 0) === (colMaxs[j0] ?? 0)) {
    const row = Array.from({ length: m }, (_, i) => (i === i0 ? 1 : 0))
    const column = Array.from({ length: n }, (_, j) => (j === j0 ? 1 : 0))
    return { row, column, value: at(i0, j0), gap: 0, iterations: 0 }
  }
  const tol = options.tol ?? 1e-6
  const maxIter = options.maxIter ?? 20_000
  const normalise = (r: readonly number[]): number[] => {
    let total = 0
    for (const v of r) total += v
    return total > 0 ? r.map((v) => v / total) : r.map(() => 1 / r.length)
  }
  let rx = new Array<number>(m).fill(0)
  let ry = new Array<number>(n).fill(0)
  const sx = new Array<number>(m).fill(0)
  const sy = new Array<number>(n).fill(0)
  let gap = Infinity
  let t = 0
  let ax: number[] = normalise(sx)
  let ay: number[] = normalise(sy)
  while (t < maxIter) {
    t += 1
    const y = normalise(ry)
    const x = normalise(rx)
    const u = rowValues(y)
    let ev = 0
    for (let i = 0; i < m; i++) ev += (x[i] ?? 0) * (u[i] ?? 0)
    rx = rx.map((r, i) => Math.max(0, r + (u[i] ?? 0) - ev))
    const x2 = normalise(rx)
    const w = columnValues(x2)
    let ew = 0
    for (let j = 0; j < n; j++) ew += (y[j] ?? 0) * (w[j] ?? 0)
    ry = ry.map((r, j) => Math.max(0, r + ew - (w[j] ?? 0)))
    for (let i = 0; i < m; i++) sx[i] = (sx[i] ?? 0) + t * (x2[i] ?? 0)
    for (let j = 0; j < n; j++) sy[j] = (sy[j] ?? 0) + t * (y[j] ?? 0)
    if (t % 10 === 0 || t === maxIter) {
      ax = normalise(sx)
      ay = normalise(sy)
      gap = maxOf(rowValues(ay)) - minOf(columnValues(ax))
      if (gap < tol) break
    }
  }
  const cv = columnValues(ax)
  let value = 0
  for (let j = 0; j < n; j++) value += (cv[j] ?? 0) * (ay[j] ?? 0)
  return { row: ax, column: ay, value, gap, iterations: t }
}

// ------------------------------------------------------------------------------------------- reservoir sampling

const CHUNK = 64

/** A persistent reservoir (Vitter's algorithm R): a uniform sample of everything ever added, in fixed memory. */
export interface Reservoir<T> {
  readonly chunks: readonly (readonly T[])[]
  readonly size: number
  readonly seen: number
  readonly capacity: number
}

/** An empty reservoir. */
export const reservoir = <T>(capacity: number): Reservoir<T> => ({ chunks: [], size: 0, seen: 0, capacity })

/** Offer an item: kept while there is room, then with probability `capacity / seen` in place of a random one. */
export function reservoirAdd<T>(r: Reservoir<T>, item: T, k: Key): Reservoir<T> {
  const seen = r.seen + 1
  let slot: number
  if (r.size < r.capacity) slot = r.size
  else {
    slot = Math.floor(u1(k) * seen)
    if (slot >= r.capacity) return { ...r, seen }
  }
  const c = Math.floor(slot / CHUNK)
  const chunk = [...(r.chunks[c] ?? [])]
  chunk[slot % CHUNK] = item
  const chunks = [...r.chunks]
  chunks[c] = chunk
  return { chunks, size: Math.max(r.size, slot + 1), seen, capacity: r.capacity }
}

/** Draw `n` items uniformly with replacement. */
export function reservoirSample<T>(r: Reservoir<T>, k: Key, n: number): T[] {
  if (r.size === 0) throw new WhiteboxError('value', 'reservoirSample: the reservoir is empty')
  const u = uniform(k, [n]).data
  return Array.from({ length: n }, (_, i) => {
    const slot = Math.min(r.size - 1, Math.floor((u[i] ?? 0) * r.size))
    return r.chunks[Math.floor(slot / CHUNK)]?.[slot % CHUNK] as T
  })
}

// ------------------------------------------------------------------------------------------------ observations

/** The vector size of an observation space: a discrete space is one-hot encoded. */
export const encodedSize = (space: Space | undefined): number =>
  space === undefined ? 0 : space.kind === 'discrete' ? space.n : flatDim(space)

/** An observation as a vector: numbers one-hot in a discrete space, arrays as they are. */
export function encodeObservation(space: Space | undefined, o: unknown): number[] {
  if (typeof o === 'number') {
    const n = space?.kind === 'discrete' ? space.n : 1
    if (n === 1 && space?.kind !== 'discrete') return [o]
    const out = new Array<number>(n).fill(0)
    out[o - (space?.kind === 'discrete' ? (space.start ?? 0) : 0)] = 1
    return out
  }
  if (Array.isArray(o)) return o.map((v) => Number(v))
  throw new WhiteboxError('value', 'encodeObservation: observations must be numbers or number arrays')
}
