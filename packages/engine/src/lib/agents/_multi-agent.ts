// Private helpers of the multi-agent learners (methods/rl/agents/multi-agent.ts): a tabular Q-learner, reservoir
// sampling and observation encoding. Zero-sum stage games are solved by games.solveZeroSum.
import { argmaxOf, MdmlError, maxOf } from '../base/index.ts'
import { linearSchedule } from '../optim/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import type { Agent, Transition } from '../rl/index.ts'
import { flatDim, type Space } from '../space/index.ts'

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
  if (r.size === 0) throw new MdmlError('value', 'reservoirSample: the reservoir is empty')
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
  throw new MdmlError('value', 'encodeObservation: observations must be numbers or number arrays')
}
