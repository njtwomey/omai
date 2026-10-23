// Helpers for reinforcement-learning tests. Node only; not published.
import type { Env } from '../lib/rl/index.ts'

/** Our model's outcomes for the (state, action) pairs of a gymnasium table, in the same row layout. */
export function modelRows(env: Env<number, number, number>, reference: readonly (readonly number[])[]): number[][] {
  const model = env.model
  if (model === undefined) throw new Error('no model')
  const pairs: [number, number][] = []
  for (const row of reference) {
    const s = row[0] ?? 0
    const a = row[1] ?? 0
    const last = pairs[pairs.length - 1]
    if (last === undefined || last[0] !== s || last[1] !== a) pairs.push([s, a])
  }
  return pairs.flatMap(([s, a]) =>
    model.transitions(s, a).map((o) => [s, a, o.probability, o.state, o.reward, o.terminated ? 1 : 0]),
  )
}
