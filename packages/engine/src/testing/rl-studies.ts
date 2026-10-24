// A tiny tabular learner for study tests (tabular agents proper live in methods/rl/agents). Node only; not published.
import { argmaxOf } from '../lib/base/index.ts'
import { foldIn, uniform } from '../lib/random/index.ts'
import type { Agent } from '../lib/rl/index.ts'

/** The state of {@link tinyQLearning}: action values by observation key. */
export interface TinyQState {
  readonly q: ReadonlyMap<string, readonly number[]>
}

const keyOf = (o: unknown): string => (typeof o === 'number' ? String(o) : JSON.stringify(o))

/**
 * One-step Q-learning with epsilon-greedy exploration (random tie-breaking) over any observations, keyed by their JSON
 * text, for fast study tests.
 */
export function tinyQLearning(
  nActions: number,
  options: { readonly alpha?: number; readonly gamma?: number; readonly epsilon?: number; readonly name?: string } = {},
): Agent<TinyQState, unknown, number> {
  const alpha = options.alpha ?? 0.5
  const gamma = options.gamma ?? 0.95
  const epsilon = options.epsilon ?? 0.1
  const zeros = new Array<number>(nActions).fill(0)
  const values = (s: TinyQState, o: unknown): readonly number[] => s.q.get(keyOf(o)) ?? zeros
  return {
    name: options.name ?? 'tiny-q',
    config: { alpha, gamma, epsilon },
    init: () => ({ q: new Map() }),
    act: (s, o, k) => {
      const u = uniform(foldIn(k, 0), [2]).data
      const q = values(s, o)
      if ((u[0] ?? 1) < epsilon) return Math.floor((u[1] ?? 0) * nActions)
      const best = q[argmaxOf(q)] ?? 0
      const ties = q.flatMap((v, a) => (v === best ? [a] : []))
      return ties[Math.floor((u[1] ?? 0) * ties.length)] ?? 0
    },
    greedy: (s, o) => argmaxOf(values(s, o)),
    learn: (s, t) => {
      const q = [...values(s, t.observation)]
      const next = values(s, t.nextObservation)
      const target = t.reward + (t.terminated ? 0 : gamma * (next[argmaxOf(next)] ?? 0))
      q[t.action] = (q[t.action] ?? 0) + alpha * (target - (q[t.action] ?? 0))
      const map = new Map(s.q)
      map.set(keyOf(t.observation), q)
      return { q: map }
    },
  }
}
