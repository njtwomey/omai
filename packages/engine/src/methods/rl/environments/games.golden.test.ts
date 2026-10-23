import { describe, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { tensor } from '../../../lib/tensor/index.ts'
import { expectedReturns, negamaxValue, type Strategy, uniformStrategy } from '../../../testing/games.ts'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type KuhnPokerState, kuhnPoker, kuhnPokerState, leducPoker, leducPokerState, nim, ticTacToe } from './games.ts'

const board = (s: { readonly board: readonly number[] }): string => s.board.join('')
const heaps = (s: { readonly heaps: readonly number[] }): string => s.heaps.join(',')

const kuhnNash =
  (alpha: number): Strategy<KuhnPokerState> =>
  (s, agent) => {
    const h = s.history.join(',')
    const card = s.cards[agent === 'player_0' ? 0 : 1]
    const bet = (p: number): number[] => [1 - p, p]
    if (agent === 'player_0') return bet((h === '' ? [alpha, 0, 3 * alpha] : [0, alpha + 1 / 3, 1])[card] ?? 0)
    return bet((h === '0' ? [1 / 3, 0, 1] : [0, 1 / 3, 1])[card] ?? 0)
  }

describe('games against exhaustive search', () => {
  for (const c of loadFixture('methods/rl/environments/games').cases) {
    it(c.name, () => {
      const p = c.params
      let got: number[]
      if (c.fn.startsWith('ticTacToe')) {
        const env = ticTacToe.create()
        const start = env.reset(key(0))
        const after = (moves: number[]): typeof start => {
          let s = start
          for (const m of moves) s = env.step(s, m, key(0)).state
          return s
        }
        if (c.fn === 'ticTacToeValue') got = [negamaxValue(env, start, board)]
        else if (c.fn === 'ticTacToeFirst')
          got = Array.from({ length: 9 }, (_, i) => -negamaxValue(env, after([i]), board))
        else {
          got = []
          for (let i = 0; i < 9; i++)
            for (let j = 0; j < 9; j++) if (i !== j) got.push(negamaxValue(env, after([i, j]), board))
        }
      } else if (c.fn === 'nimMisere') {
        const env = nim.create({ heaps: [3, 3, 4], misere: true })
        got = (p['heaps'] as number[][]).map((h) => negamaxValue(env, { ...env.reset(key(0)), heaps: h }, heaps))
      } else if (c.fn === 'kuhnNash') {
        const env = kuhnPoker.create()
        let v = 0
        for (const [a, b] of [
          [0, 1],
          [0, 2],
          [1, 0],
          [1, 2],
          [2, 0],
          [2, 1],
        ] as const)
          v += expectedReturns(env, kuhnPokerState([a, b]), kuhnNash(p['alpha'] as number))[0] / 6
        got = [v]
      } else {
        const env = leducPoker.create()
        got = (p['deals'] as [number, number, number][]).map(
          (d) => expectedReturns(env, leducPokerState(d), uniformStrategy)[0],
        )
      }
      expectMatches(tensor(got), c.output, c.rtol, c.atol, c.name)
    })
  }
})
