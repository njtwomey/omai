import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { playGameSearch, randomTree, textbookTree, ticTacToe } from './adversarial.ts'

describe('playGameSearch', () => {
  it('finds the textbook minimax value and prunes with alpha-beta', () => {
    const mm = playGameSearch(textbookTree(), { algorithm: 'minimax' })
    const ab = playGameSearch(textbookTree(), { algorithm: 'alpha-beta' })
    expect([mm.value, mm.action]).toEqual([3, 'a'])
    expect([ab.value, ab.action]).toEqual([3, 'a'])
    expect(mm.visited).toBe(13)
    expect(ab.visited).toBeLessThan(mm.visited)
    const [board] = recordsOf(ab.exposition, 'boardSteps')
    expect(board?.layout.type).toBe('graph')
    const last = board?.steps.at(-1)
    expect(last?.cells.filter((c) => c === 5).length).toBeGreaterThan(0)
    expect(last?.cells.filter((c) => c === 6).length).toBe(3)
  })

  it('agrees between minimax and alpha-beta on random trees and tic-tac-toe', () => {
    const t = randomTree(key(3), 3, 3)
    expect(playGameSearch(t, { algorithm: 'alpha-beta', ordered: true }).value).toBe(
      playGameSearch(t, { algorithm: 'minimax' }).value,
    )
    const g = ticTacToe('XOX.O..X.')
    const ab = playGameSearch(g)
    expect(ab.value).toBe(playGameSearch(g, { algorithm: 'minimax' }).value)
    expect(ab.treeSize).toBeGreaterThan(ab.visited)
  })
})
