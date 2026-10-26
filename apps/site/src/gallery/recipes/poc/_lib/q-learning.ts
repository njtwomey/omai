// A sequential-decision method on the contract: Q-learning on cliff walking through the core's training algorithm,
// whose reward, agent statistics and frames are renamed to paths and whose episode event schedules the rest. The
// Q-table, the greedy policy and greedy rollouts are derived from any state, so a page can click an episode and watch
// that episode's policy act.
import { cliffWalking } from 'mdml-engine/methods/rl'
import { qLearning, type TabularState } from 'mdml-engine/methods/rl/agents'
import { type Key, key as makeKey } from 'mdml-engine/random'
import { type Env, greedyPolicyOf, make, runEpisode, type TrainState, trainingAlgorithm } from 'mdml-engine/rl'
import { dict, real } from 'mdml-engine/space'
import { asTensor, reshape } from 'mdml-engine/tensor'
import { type Derived, defineMethod, derived, event, frame, Log, type Method, scalar } from 'mdml-engine/trace'

export type QConfig = { readonly epsilon: number; readonly alpha: number; readonly gamma: number }
export type QState = TrainState<TabularState, unknown, number>

function built(c: QConfig): { env: Env<unknown, number, number>; agent: ReturnType<typeof qLearning<number>> } {
  const env = make(cliffWalking)
  const agent = qLearning(env, { epsilon: c.epsilon, alpha: c.alpha, gamma: c.gamma })
  return { env, agent }
}

const greedyTable = (g: TabularState): number[] =>
  Array.from({ length: g.nStates }, (_, s) => {
    let best = 0
    for (let a = 1; a < g.nActions; a++)
      if ((g.Q[s * g.nActions + a] ?? 0) > (g.Q[s * g.nActions + best] ?? 0)) best = a
    return best
  })

export const qLearningMethod: Method<QState, QConfig> = defineMethod<QState, QConfig>({
  name: 'poc/q-learning',
  config: dict({
    epsilon: real(0, 1, { default: 0.1 }),
    alpha: real(0.05, 1, { default: 0.5 }),
    gamma: real(0.8, 1, { default: 0.99 }),
  }),
  make: (c, _key: Key) => {
    const { env, agent } = built(c)
    return trainingAlgorithm(env, agent, { checkpoints: false })
  },
  contract: (c) => {
    const { env, agent } = built(c)
    return {
      'rl/step/reward': scalar({ from: 'reward', description: 'the reward of the step' }),
      'rl/episode/return-so-far': scalar({ from: 'return', description: 'the return of the current episode so far' }),
      'rl/episode/end': event({
        context: { episode: 'the episode', return: 'its return', length: 'its length', terminated: 'reached the goal' },
      }),
      'agent/epsilon': scalar({ from: 'epsilon', description: 'the exploration rate' }),
      'agent/td-error': scalar({ from: 'tdError', description: 'the last TD error' }),
      'env/frame': frame({ description: 'the grid world as the agent moves' }),
      'agent/q': derived({
        description: 'the action-value table, states by actions',
        compute: (s) => reshape(asTensor(Array.from(s.agent.Q)), [s.agent.nStates, s.agent.nActions]),
      }),
      'agent/policy': derived({ description: 'the greedy action per state', compute: (s) => greedyTable(s.agent) }),
      'agent/rollout': derived({
        description: 'greedy episodes played with the policy at this step, with frames',
        cost: 'dear',
        compute: (s, args) => {
          const a = (args ?? {}) as { readonly n?: number; readonly seed?: number }
          const policy = greedyPolicyOf(agent, s.agent)
          const episodes = Array.from({ length: a.n ?? 1 }, (_, i) => {
            const e = runEpisode(env, policy, makeKey((a.seed ?? 0) + i), { frames: true, maxSteps: 100 })
            return { return: e.return, length: e.length, frames: e.frames ?? [] }
          })
          return episodes as unknown as Derived
        },
      }),
    }
  },
  defaults: { 'rl/step/reward': Log(), 'rl/episode/end': Log() },
})
