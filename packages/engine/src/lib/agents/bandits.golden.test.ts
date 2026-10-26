import { describe, it, vi } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { key } from '../random/index.ts'
import type { Agent } from '../rl/index.ts'
import { box, discrete, tuple } from '../space/index.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import {
  type ArmFeatures,
  type BanditState,
  epsilonGreedyBandit,
  exp3,
  gaussianPosterior,
  gradientBandit,
  klUcb,
  linUcb,
  thompsonBernoulli,
  ucb1,
} from './bandits.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

const arms = { actionSpace: discrete(4) }
const pull = (action: number, reward: number) => ({
  observation: 0,
  action,
  reward,
  nextObservation: 0,
  terminated: false,
  truncated: false,
})

// Let an agent choose against a reward table (rewards[t][arm]); its actions and final state.
function play<G>(agent: Agent<G, unknown, number>, table: number[][]): { actions: number[]; state: G } {
  let s = agent.init(key(0))
  const actions: number[] = []
  for (const [t, rewards] of table.entries()) {
    const a = agent.act(s, 0, key(1000 + t))
    actions.push(a)
    s = agent.learn(s, pull(a, rewards[a] ?? 0), key(t))
  }
  return { actions, state: s }
}

// Replay fixed pulls; the state after each.
function replayPulls<G>(agent: Agent<G, unknown, number>, actions: number[], rewards: number[]): G[] {
  let s = agent.init(key(0))
  const out: G[] = []
  for (const [t, a] of actions.entries()) {
    s = agent.learn(s, pull(a, rewards[t] ?? 0), key(t))
    out.push(s)
  }
  return out
}

describe('bandit update rules against numpy transcriptions', () => {
  for (const c of loadFixture('methods/rl/agents/bandits').cases) {
    it(c.name, () => {
      const input = (i: number): unknown => toArray(decode(c.inputs[i]) as Tensor)
      let got: number[]
      if (c.fn === 'ucb1' || c.fn === 'klUcb') {
        const run = play(c.fn === 'ucb1' ? ucb1(arms, { c: 2 }) : klUcb(arms), input(0) as number[][])
        got = c.name.endsWith('actions') ? run.actions : Array.from(run.state.means)
      } else if (c.fn === 'linUcb') {
        const features = input(0) as number[][][]
        const table = input(1) as number[][]
        const model = c.params['model'] as 'shared' | 'disjoint'
        const env = { observationSpace: tuple([box(-1, 1, [3])]), actionSpace: discrete(4) }
        const agent = linUcb(env, { alpha: 0.5, lambda: 1, model })
        let s = agent.init(key(0))
        const actions: number[] = []
        for (const [t, x] of features.entries()) {
          const obs: ArmFeatures = x
          const a = agent.act(s, obs, key(t))
          actions.push(a)
          s = agent.learn(
            s,
            {
              observation: obs,
              action: a,
              reward: table[t]?.[a] ?? 0,
              nextObservation: obs,
              terminated: false,
              truncated: false,
            },
            key(t),
          )
        }
        got = c.name.endsWith('actions') ? actions : s.theta.flatMap((th) => Array.from(th))
      } else {
        const actions = input(0) as number[]
        const rewards = input(1) as number[]
        switch (c.fn) {
          case 'exp3':
            got = replayPulls(exp3(arms, { gamma: 0.2 }), actions, rewards).flatMap((s) => Array.from(s.probabilities))
            break
          case 'gradientBandit': {
            const last = replayPulls(gradientBandit(arms, { alpha: 0.1 }), actions, rewards).at(-1)
            got = last === undefined ? [] : [...last.preferences, last.baseline]
            break
          }
          case 'epsilonGreedyBandit':
            got = Array.from(
              replayPulls(epsilonGreedyBandit(arms, { alpha: 0.1, initialValue: 5 }), actions, rewards).at(-1)
                ?.estimates ?? [],
            )
            break
          case 'thompsonBernoulli': {
            const last = replayPulls(thompsonBernoulli(arms), actions, rewards).at(-1)
            got = last === undefined ? [] : [...last.alpha, ...last.beta]
            break
          }
          default: {
            const last = replayPulls(ucb1(arms), actions, rewards).at(-1) as BanditState
            const p = gaussianPosterior(last, c.params as never)
            got = [...p.mean, ...p.precision, ...(p.shape ?? []), ...(p.rate ?? [])]
          }
        }
      }
      expectMatches(tensor(got), c.output, c.rtol, c.atol, c.name)
    })
  }
})
