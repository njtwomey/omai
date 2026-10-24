import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { isSolution } from '../../lib/search/index.ts'
import { solveMapColouring, solveQueens, solveSudoku } from './csp.ts'
import { AUSTRALIA, mapColouring, nQueens, randomMap, SUDOKUS, sudoku } from './puzzles.ts'

describe('played CSP solvers', () => {
  it('place eight queens by backtracking with forward checking, and by min-conflicts', () => {
    for (const solver of ['backtracking', 'min-conflicts'] as const) {
      const r = solveQueens(8, { solver, key: key(2) })
      expect(r.status).toBe('solved')
      expect(isSolution(nQueens(8), r.assignment ?? {})).toBe(true)
      const [board] = recordsOf(r.exposition, 'boardSteps')
      expect(board?.layout).toEqual({ type: 'grid', rows: 8, columns: 8 })
      expect(board?.steps.at(-1)?.labels?.filter((l) => l === '♛').length).toBe(8)
    }
  })

  it('backtracks more without inference', () => {
    const plain = solveQueens(8, { variableOrder: 'static', inference: 'none' })
    const fc = solveQueens(8, { variableOrder: 'static', inference: 'forward-checking' })
    expect(plain.backtracks).toBeGreaterThan(fc.backtracks)
  })

  it('solves the Sudokus and colours the maps', () => {
    for (const grid of [SUDOKUS.easy, SUDOKUS.mini, SUDOKUS.hard]) {
      const r = solveSudoku(grid)
      expect(r.status).toBe('solved')
      expect(isSolution(sudoku(grid), r.assignment ?? {})).toBe(true)
    }
    const au = solveMapColouring(AUSTRALIA)
    expect(isSolution(mapColouring(AUSTRALIA), au.assignment ?? {})).toBe(true)
    const m = randomMap(key(4), 14)
    const four = solveMapColouring(m, ['red', 'green', 'blue', 'yellow'], { solver: 'min-conflicts', key: key(1) })
    expect(four.status).toBe('solved')
  })
})
