import { describe, it } from 'vitest'
import { foldIn, key, uniform } from '../../../lib/random/index.ts'
import type { Agent } from '../../../lib/rl/index.ts'
import { discrete } from '../../../lib/space/index.ts'
import { type Tensor, tensor, toArray } from '../../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { replay, transitionsOf } from '../../../testing/rl-tabular.ts'
import {
  type DoubleQState,
  doubleQLearning,
  dynaQ,
  expectedSarsa,
  monteCarloControl,
  nStepSarsa,
  prioritizedSweeping,
  qLearning,
  sarsa,
  sarsaLambda,
  type TabularState,
  type TdState,
  td0,
  tdLambda,
} from './tabular.ts'

const spaces = { observationSpace: discrete(5), actionSpace: discrete(3) }
const gamma = 0.9

type AnyAgent = Agent<TabularState | TdState, number, number>

const agents: Readonly<Record<string, () => AnyAgent>> = {
  'qLearning/constant': () => qLearning(spaces, { alpha: 0.3, gamma, initialValue: 0.5 }) as AnyAgent,
  'qLearning/sample-average': () => qLearning(spaces, { alpha: 'sample-average', gamma }) as AnyAgent,
  sarsa: () => sarsa(spaces, { alpha: 0.4, gamma, epsilon: 0.2 }) as AnyAgent,
  'expectedSarsa/behaviour': () => expectedSarsa(spaces, { alpha: 0.4, gamma, epsilon: 0.2 }) as AnyAgent,
  'expectedSarsa/greedy-target': () =>
    expectedSarsa(spaces, { alpha: 0.4, gamma, epsilon: 0.2, targetEpsilon: 0 }) as AnyAgent,
  'nStepSarsa/n1': () => nStepSarsa(spaces, { n: 1, alpha: 0.4, gamma, epsilon: 0.2 }) as AnyAgent,
  'nStepSarsa/n3': () => nStepSarsa(spaces, { n: 3, alpha: 0.4, gamma, epsilon: 0.2 }) as AnyAgent,
  'nStepSarsa/expected-n2': () =>
    nStepSarsa(spaces, { n: 2, alpha: 0.4, gamma, epsilon: 0.2, expected: true }) as AnyAgent,
  'monteCarloControl/first-visit': () => monteCarloControl(spaces, { gamma, epsilon: 0.2 }) as AnyAgent,
  'monteCarloControl/every-visit': () =>
    monteCarloControl(spaces, { gamma, epsilon: 0.2, alpha: 0.1, everyVisit: true }) as AnyAgent,
  'sarsaLambda/accumulating': () => sarsaLambda(spaces, { alpha: 0.3, gamma, epsilon: 0.2, lambda: 0.7 }) as AnyAgent,
  'sarsaLambda/replacing': () =>
    sarsaLambda(spaces, { alpha: 0.3, gamma, epsilon: 0.2, lambda: 0.7, trace: 'replacing' }) as AnyAgent,
  'dynaQ/no-planning': () => dynaQ(spaces, { alpha: 0.3, gamma, planningSteps: 0 }) as AnyAgent,
  prioritizedSweeping: () =>
    prioritizedSweeping(spaces, { alpha: 0.5, gamma, planningSteps: 3, theta: 1e-3 }) as AnyAgent,
  td0: () => td0(spaces, { alpha: 0.2, gamma }) as AnyAgent,
  'tdLambda/accumulating': () => tdLambda(spaces, { alpha: 0.2, gamma, lambda: 0.6 }) as AnyAgent,
  'tdLambda/replacing': () => tdLambda(spaces, { alpha: 0.2, gamma, lambda: 0.6, trace: 'replacing' }) as AnyAgent,
  'tdLambda/dutch': () => tdLambda(spaces, { alpha: 0.2, gamma, lambda: 0.6, trace: 'dutch' }) as AnyAgent,
}

// The double Q-learning coin is `uniform(foldIn(key, 0)) < 0.5`; find learn keys that flip the fixture's coins.
function keyWithCoin(updateA: boolean, from: number): { readonly k: ReturnType<typeof key>; readonly next: number } {
  for (let i = from; ; i++) {
    const k = key(i)
    if ((uniform(foldIn(k, 0), []).data[0] ?? 0) < 0.5 === updateA) return { k, next: i + 1 }
  }
}

describe('tabular update rules against numpy transcriptions of Sutton and Barto', () => {
  for (const c of loadFixture('methods/rl/agents/tabular').cases) {
    it(c.name, () => {
      const rows = toArray(decode(c.inputs[0]) as Tensor) as number[][]
      const transitions = transitionsOf(rows)
      if (c.name === 'doubleQLearning') {
        const coins = toArray(decode(c.inputs[1]) as Tensor) as number[]
        const agent = doubleQLearning(spaces, { alpha: 0.3, gamma })
        let s: DoubleQState = agent.init(key(0))
        let next = 0
        for (const [i, t] of transitions.entries()) {
          const found = keyWithCoin(coins[i] === 1, next)
          next = found.next
          s = agent.learn(s, t, found.k)
        }
        expectMatches(tensor([...s.QA, ...s.QB, ...s.Q]), c.output, c.rtol, c.atol, c.name)
        return
      }
      if (c.name === 'dynaQ/model') {
        const states = replay(dynaQ(spaces, { planningSteps: 0 }), transitions)
        const m = (states[states.length - 1] as { model: { reward: Float64Array; next: Int32Array } }).model
        expectMatches(tensor([...m.reward, ...m.next]), c.output, c.rtol, c.atol, c.name)
        return
      }
      const make = agents[c.name]
      if (make === undefined) throw new Error(`no agent for ${c.name}`)
      const states = replay(make(), transitions)
      const last = states[states.length - 1] as TabularState | TdState
      expectMatches(tensor(Array.from('V' in last ? last.V : last.Q)), c.output, c.rtol, c.atol, c.name)
    })
  }
})
