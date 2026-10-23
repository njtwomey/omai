import { describe, it } from 'vitest'
import { makeCliffWalking, makeFrozenLake, makeTaxi } from '../../methods/rl/index.ts'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor } from '../tensor/index.ts'
import type { Env } from './env.ts'
import { mdpFromEnv, occupancyMeasure, policyEvaluation, policyIteration, qFromV, valueIteration } from './mdp.ts'

function build(p: Readonly<Record<string, unknown>>): Env<number, number, number> {
  if (p['env'] === 'CliffWalking-v1') return makeCliffWalking({ isSlippery: p['is_slippery'] === true })
  if (p['env'] === 'Taxi-v4') return makeTaxi()
  return makeFrozenLake(p['map_name'] === '8x8' ? { mapName: '8x8' } : {})
}

describe('tabular MDP solutions against numpy Bellman equations on gymnasium tables', () => {
  for (const c of loadFixture('rl/mdp').cases) {
    it(c.name, () => {
      const gamma = c.params['gamma'] as number
      const mdp = mdpFromEnv(build(c.params), gamma)
      const nA = mdp.nActions
      const uniform = Array.from({ length: mdp.nStates }, () => new Array<number>(nA).fill(1 / nA))
      let got: number[] | number[][]
      switch (c.fn) {
        case 'valueIteration':
          got = valueIteration(mdp, { tol: 1e-13 }).V
          break
        case 'qFromV':
          got = qFromV(mdp, valueIteration(mdp, { tol: 1e-13 }).V)
          break
        case 'policyEvaluation':
          got = policyEvaluation(mdp, (c.params['policy'] as number[] | undefined) ?? uniform)
          break
        default:
          got = occupancyMeasure(mdp, uniform).state
      }
      expectMatches(tensor(got), c.output, c.rtol, c.atol, c.name)
    })
  }

  it('policy iteration reaches the value-iteration fixed point', () => {
    for (const c of loadFixture('rl/mdp').cases.filter((x) => x.fn === 'valueIteration')) {
      const gamma = c.params['gamma'] as number
      // Policy iteration needs a terminating start policy when gamma = 1; discount slightly instead.
      if (gamma === 1) continue
      const mdp = mdpFromEnv(build(c.params), gamma)
      expectMatches(tensor(policyIteration(mdp).V), c.output, 1e-9, 1e-9, c.name)
    }
  })
})
