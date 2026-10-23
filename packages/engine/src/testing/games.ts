// Exact analysis of small turn-based games for tests: negamax values by exhaustive search with memoisation, and
// expected returns of explicit mixed strategies by enumerating the game tree. Node only; not published.
import { key } from '../lib/random/index.ts'
import type { AecEnv } from '../lib/rl/index.ts'

const K = key(0)

/** The negamax value of a position for the player to move (deterministic games; rewards to the mover). */
export function negamaxValue<S, O>(env: AecEnv<S, O, number>, state: S, id: (s: S) => string): number {
  const memo = new Map<string, number>()
  const value = (s: S): number => {
    const mover = env.currentAgent(s)
    if (mover === undefined) return 0
    const k = id(s)
    const cached = memo.get(k)
    if (cached !== undefined) return cached
    let best = -Infinity
    for (const a of env.legalActions?.(s, mover) ?? []) {
      const r = env.step(s, a, K)
      const next = env.currentAgent(r.state)
      const future = next === undefined ? 0 : next === mover ? value(r.state) : -value(r.state)
      best = Math.max(best, (r.rewards[mover] ?? 0) + future)
    }
    memo.set(k, best)
    return best
  }
  return value(state)
}

/** A mixed strategy: action probabilities over the legal actions, in their order. */
export type Strategy<S> = (state: S, agent: string, legal: readonly number[]) => readonly number[]

/** The expected total rewards of the two players from a position when both follow `strategy` (exact enumeration). */
export function expectedReturns<S, O>(env: AecEnv<S, O, number>, state: S, strategy: Strategy<S>): [number, number] {
  const mover = env.currentAgent(state)
  if (mover === undefined) return [0, 0]
  const legal = env.legalActions?.(state, mover) ?? []
  const probs = strategy(state, mover, legal)
  const out: [number, number] = [0, 0]
  for (const [i, a] of legal.entries()) {
    const p = probs[i] ?? 0
    if (p === 0) continue
    const r = env.step(state, a, K)
    const future = expectedReturns(env, r.state, strategy)
    out[0] += p * ((r.rewards['player_0'] ?? 0) + future[0])
    out[1] += p * ((r.rewards['player_1'] ?? 0) + future[1])
  }
  return out
}

/** A strategy that plays every legal action with equal probability. */
export function uniformStrategy<S>(_state: S, _agent: string, legal: readonly number[]): number[] {
  return legal.map(() => 1 / legal.length)
}
