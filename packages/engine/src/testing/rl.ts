// Helpers for reinforcement-learning tests. Node only; not published.
import type { AecEnv, Env, ParallelEnv } from '../lib/rl/index.ts'
import { scene } from '../lib/scene/index.ts'
import { discrete } from '../lib/space/index.ts'

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

// A one-heap subtraction game: take 1 or 2 tokens; whoever takes the last token wins.
export interface Heap {
  readonly tokens: number
  readonly mover: 0 | 1
}
export const AGENTS: readonly string[] = ['player_0', 'player_1']
/** A one-heap subtraction game (take one or two tokens; taking the last wins), for multi-agent tests. */
export const takeAway = (start: number): AecEnv<Heap, number, number> => ({
  spec: { id: 'TakeAway-v0', description: 'take one or two tokens' },
  possibleAgents: AGENTS,
  observationSpaces: { player_0: discrete(start + 1), player_1: discrete(start + 1) },
  actionSpaces: { player_0: discrete(2), player_1: discrete(2) },
  reset: () => ({ tokens: start, mover: 0 }),
  currentAgent: (s) => (s.tokens === 0 ? undefined : AGENTS[s.mover]),
  observe: (s) => s.tokens,
  legalActions: (s) => (s.tokens >= 2 ? [0, 1] : [0]),
  step: (s, a) => {
    const tokens = Math.max(0, s.tokens - (a + 1))
    const mover = s.mover === 0 ? 1 : 0
    const me = AGENTS[s.mover] as string
    const other = AGENTS[mover] as string
    return { state: { tokens, mover }, rewards: tokens === 0 ? { [me]: 1, [other]: -1 } : {} }
  },
  terminations: (s) => ({ player_0: s.tokens === 0, player_1: s.tokens === 0 }),
  truncations: () => ({ player_0: false, player_1: false }),
  render: () => scene(1, 1, []),
})

// A repeated coordination game: both get 1 when they choose the same action.
/** A repeated two-player coordination game, for multi-agent tests. */
export const coordination: ParallelEnv<number, number, number> = {
  spec: { id: 'Coordination-v0', description: 'match actions' },
  possibleAgents: AGENTS,
  observationSpaces: { player_0: discrete(1), player_1: discrete(1) },
  actionSpaces: { player_0: discrete(2), player_1: discrete(2) },
  agents: () => AGENTS,
  reset: () => ({ state: 0, observations: { player_0: 0, player_1: 0 }, infos: {} }),
  step: (t, actions) => {
    const r = actions['player_0'] === actions['player_1'] ? 1 : 0
    return {
      state: t + 1,
      observations: { player_0: 0, player_1: 0 },
      rewards: { player_0: r, player_1: r },
      terminations: { player_0: false, player_1: false },
      truncations: { player_0: false, player_1: false },
      infos: {},
    }
  },
  render: () => scene(1, 1, []),
}
