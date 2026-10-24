import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { boardGame, type Square } from '../../testing/search.ts'
import { alphaBeta, minimax, negamax } from './index.ts'

describe('game-tree search against aima-python', () => {
  for (const c of loadFixture('search/adversarial').cases) {
    it(c.name, () => {
      const p = c.params
      const game = boardGame(p['h'] as number, p['v'] as number, p['k'] as number, p['game'] === 'connect-four')
      const state = game.play(p['moves'] as Square[])
      const options =
        c.fn === 'alphaBetaCutoff' ? { depth: (p['d'] as number) + 1, evaluate: game.utility } : ({} as const)
      const r = c.fn === 'minimax' ? minimax(game, state, options) : alphaBeta(game, state, options)
      expect(r.action).toEqual(p['action'])
      expect(r.value).toBe(p['value'])
      expect(r.nodes).toBe(p['nodes'])
      // Negamax with alpha-beta finds the same value from the mover's side.
      expect(negamax(game, state, options).value).toBe(p['value'])
    })
  }
})
