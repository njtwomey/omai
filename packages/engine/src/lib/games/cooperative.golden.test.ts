import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import {
  bankruptcyGame,
  banzhafIndex,
  type CoalitionalGame,
  coalitionalGame,
  corePoint,
  gloveGame,
  isInCore,
  leastCore,
  nucleolus,
  shapleyValue,
  weightedVotingGame,
} from './index.ts'

function gameOf(n: number, spec: Record<string, unknown>, inputs: readonly unknown[]): CoalitionalGame {
  switch (spec['rule']) {
    case 'glove':
      return gloveGame(spec['left'] as number, spec['right'] as number)
    case 'voting':
      return weightedVotingGame(spec['quota'] as number, spec['weights'] as number[])
    case 'bankruptcy':
      return bankruptcyGame(spec['estate'] as number, spec['claims'] as number[])
    default:
      return coalitionalGame(n, toFlat(decode(inputs[0]) as Tensor))
  }
}

describe('Shapley values and Banzhaf indices from their definitions', () => {
  for (const c of loadFixture('games/cooperative').cases) {
    it(c.name, () => {
      const game = gameOf(c.params['n'] as number, c.params['game'] as Record<string, unknown>, c.inputs)
      expectMatches(tensor(shapleyValue(game)), c.output, c.rtol, c.atol, c.name)
      const beta = banzhafIndex(game)
      for (const [i, v] of (c.params['banzhaf'] as number[]).entries()) expect(beta[i]).toBeCloseTo(v, 12)
    })
  }

  it('gives a permanent Security Council member a Shapley-Shubik power of 0.1963', () => {
    const council = weightedVotingGame(39, [7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expect(shapleyValue(council)[0]).toBeCloseTo(0.19627, 5)
    expect(banzhafIndex(council, { normalised: true })[0]).toBeCloseTo(0.1669, 4)
  })
})

describe('the nucleolus and the least core', () => {
  for (const c of loadFixture('games/core').cases) {
    it(c.name, () => {
      const n = c.params['n'] as number
      const game = coalitionalGame(n, toFlat(decode(c.inputs[0]) as Tensor))
      if (c.fn === 'leastCore') {
        expectMatches(leastCore(game).epsilon, c.output, 0, c.atol, c.name)
        return
      }
      expectMatches(tensor(nucleolus(game)), c.output, 0, c.atol, c.name)
      const eps = leastCore(game).epsilon
      expect(eps).toBeCloseTo(c.params['leastCore'] as number, 8)
      const point = corePoint(game)
      if (eps <= 1e-9) expect(point !== null && isInCore(game, point, 1e-7)).toBe(true)
      else expect(point).toBeNull()
    })
  }
})
