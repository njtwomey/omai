// Small cooperative games for multi-agent learning tests. Node only; not published.
import type { ParallelEnv } from '../lib/rl/index.ts'
import { scene } from '../lib/scene/index.ts'
import { discrete } from '../lib/space/index.ts'

/** The state of the two-step game: 0 the first step, 1 matrix game 2A, 2 matrix game 2B, 3 over. */
export type TwoStepState = 0 | 1 | 2 | 3

const AGENTS = ['agent_0', 'agent_1']
const GAME_2B = [
  [0, 1],
  [1, 8],
]

const oneHot = (s: TwoStepState): number[] => [0, 1, 2].map((i) => (i === s ? 1 : 0))

/**
 * The two-step cooperative matrix game of Rashid et al. (2018, QMIX, figure 3): agent 0's first action picks game 2A
 * (every joint action pays 7) or game 2B (`[[0, 1], [1, 8]]`); both agents share the reward. The optimum, 8, needs a
 * non-additive joint value: QMIX finds it, VDN settles for 7. Observations are the one-hot state.
 */
export const twoStepGame: ParallelEnv<TwoStepState, readonly number[], number> = {
  spec: { id: 'TwoStepGame-v0', description: 'the QMIX two-step game', maxEpisodeSteps: 2 },
  possibleAgents: AGENTS,
  observationSpaces: { agent_0: discrete(3), agent_1: discrete(3) },
  actionSpaces: { agent_0: discrete(2), agent_1: discrete(2) },
  agents: (s) => (s === 3 ? [] : AGENTS),
  reset: () => ({ state: 0, observations: { agent_0: oneHot(0), agent_1: oneHot(0) }, infos: {} }),
  step: (s, actions) => {
    const a0 = actions['agent_0'] ?? 0
    const a1 = actions['agent_1'] ?? 0
    const next: TwoStepState = s === 0 ? (a0 === 0 ? 1 : 2) : 3
    const r = s === 1 ? 7 : s === 2 ? (GAME_2B[a0]?.[a1] ?? 0) : 0
    const done = next === 3
    return {
      state: next,
      observations: { agent_0: oneHot(next), agent_1: oneHot(next) },
      rewards: { agent_0: r, agent_1: r },
      terminations: { agent_0: done, agent_1: done },
      truncations: { agent_0: false, agent_1: false },
      infos: {},
    }
  },
  render: () => scene(1, 1, []),
}
