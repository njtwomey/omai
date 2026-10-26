import { describe, expect, it } from 'vitest'
import { makeCliffWalking, makeNChain, makeRiverSwim, makeWindyGridWorld } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { scene } from '../scene/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  greedyPolicy,
  mdpFromEnv,
  occupancyMeasure,
  policyEvaluation,
  policyIteration,
  tabularEnv,
  tabularMdp,
  valueIteration,
  valueIterationAlgorithm,
} from './mdp.ts'

// A two-state chain: from state 0, action 1 pays 1 and ends; action 0 stays for nothing.
const chain = (gamma: number) =>
  tabularMdp({
    transitions: [
      [
        [1, 0],
        [0, 1],
      ],
      [
        [0, 1],
        [0, 1],
      ],
    ],
    rewards: [
      [0, 1],
      [0, 0],
    ],
    terminalStates: [1],
    initial: [1, 0],
    gamma,
  })

describe('tabular MDPs', () => {
  it('validate transition rows and expected rewards', () => {
    expect(() => tabularMdp({ transitions: [[[0.5, 0.2]]], rewards: [[0]] })).toThrow(/sums to/)
    const mdp = tabularMdp({
      transitions: [[[0.5, 0.5]], [[0, 1]]],
      rewards: [[[2, 4]], [[0, 0]]],
    })
    expect(Array.from(mdp.rewards)).toEqual([3, 0])
    expect(Array.from(mdp.initial)).toEqual([0.5, 0.5])
  })

  it('solve a chain by value and policy iteration and agree with exact evaluation', () => {
    const mdp = chain(0.9)
    const vi = valueIteration(mdp)
    const pi = policyIteration(mdp)
    expect(vi.V).toEqual([1, 0])
    expect(pi.policy).toEqual([1, 0])
    expect(pi.converged).toBe(true)
    expect(policyEvaluation(mdp, pi.policy)).toEqual([1, 0])
    expect(policyEvaluation(mdp, [0, 0], { method: 'iterative' })).toEqual([0, 0])
    expect(
      policyEvaluation(
        mdp,
        [
          [0.5, 0.5],
          [1, 0],
        ],
        { method: 'iterative' },
      )[0],
    ).toBeCloseTo(0.5 / (1 - 0.45), 10)
    expect(() => policyEvaluation(chain(1), [0, 0])).toThrow(/singular/)
  })

  it('trace value iteration: residuals fall and the start value rises', () => {
    const rec = recorder()
    runAlgorithm(valueIterationAlgorithm(mdpFromEnv(makeCliffWalking(), 1)), { maxSteps: 100, tracer: rec })
    const deltas = rec.run().events.flatMap((e) => (e.type === 'metric' && e.name === 'delta' ? [e.value] : []))
    expect(deltas.at(-1)).toBe(0)
  })

  it('find the known optima: cliff walking, the windy grid world, NChain and RiverSwim', () => {
    // Cliff walking: the shortest safe path from the start takes 13 steps.
    expect(valueIteration(mdpFromEnv(makeCliffWalking(), 1)).V[36]).toBe(-13)
    // The windy grid world takes 15 steps with four moves and 7 with king's moves (Sutton and Barto, 6.5 and 6.9).
    expect(valueIteration(mdpFromEnv(makeWindyGridWorld(), 1)).V[30]).toBe(-15)
    expect(valueIteration(mdpFromEnv(makeWindyGridWorld({ kingsMoves: true }), 1)).V[30]).toBe(-7)
    // NChain: with patience, always move forward; RiverSwim: always swim right.
    expect(valueIteration(mdpFromEnv(makeNChain(), 0.99)).policy).toEqual([0, 0, 0, 0, 0])
    expect(valueIteration(mdpFromEnv(makeRiverSwim(), 0.95)).policy).toEqual([1, 1, 1, 1, 1, 1])
    expect(valueIteration(mdpFromEnv(makeNChain(), 0.5)).policy[0]).toBe(1)
  })

  it('compute occupancy measures that sum to the expected discounted horizon', () => {
    const mdp = chain(0.5)
    expect(occupancyMeasure(mdp, [0, 0]).state).toEqual([2, 0])
    const n = occupancyMeasure(
      mdp,
      [
        [0.5, 0.5],
        [1, 0],
      ],
      { normalize: true },
    )
    expect(n.state.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(n.stateAction[0]?.[0]).toBeCloseTo(0.5, 12)
    expect(
      greedyPolicy([
        [1, 1],
        [0, 2],
      ]),
    ).toEqual([0, 1])
  })

  it('build environments from transition tables that sample outcomes by their probabilities', () => {
    const coin = tabularEnv({
      id: 'Coin-v0',
      description: 'a fair coin',
      nStates: 3,
      nActions: 1,
      initial: [1, 0, 0],
      render: () => scene(1, 1, []),
      transitions: () => [
        { probability: 0.5, state: 1, reward: 1, terminated: true },
        { probability: 0.5, state: 2, reward: 0, terminated: true },
      ],
    })
    let heads = 0
    for (let i = 0; i < 2000; i++) heads += coin.step(0, 0, key(i)).reward
    expect(Math.abs(heads / 2000 - 0.5)).toBeLessThan(0.04)
    expect(coin.step(0, 0, key(1)).info['prob']).toBe(0.5)
    expect(mdpFromEnv(coin).terminal[0]).toBe(1)
    expect(() => mdpFromEnv({ ...coin, model: undefined } as never)).toThrow(/no model/)
  })
})
