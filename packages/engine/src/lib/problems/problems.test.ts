import { describe, expect, it } from 'vitest'
import { cdcl, satisfies } from '../logic/index.ts'
import { foldIn, key } from '../random/index.ts'
import { breadthFirstSearch, idaStarSearch } from '../search/index.ts'
import {
  circleTour,
  findHamiltonianCycle,
  gridTour,
  heldKarp,
  isHamiltonianCycle,
  isSolvable,
  optimalMoves,
  pigeonhole,
  plantedHamiltonian,
  plantedKSat,
  randomKSat,
  randomTour,
  SAT_THRESHOLDS,
  scrambledBoard,
  slidingPuzzle,
  solvedBoard,
  tourLength,
  towersOfHanoi,
} from './index.ts'

describe('sliding puzzles', () => {
  it('splits the arrangements by parity: a swap of two tiles is unsolvable, every move preserves solvability', () => {
    expect(isSolvable(solvedBoard(3))).toBe(true)
    expect(isSolvable([1, 2, 3, 4, 5, 6, 8, 7, 0])).toBe(false)
    expect(isSolvable([2, 1, 3, 4, 5, 6, 7, 8, 0])).toBe(false)
    // The 15-puzzle: Sam Loyd's 14-15 swap is unsolvable, and the blank's row matters on an even side.
    const loyd = [...solvedBoard(4)]
    ;[loyd[13], loyd[14]] = [loyd[14] as number, loyd[13] as number]
    expect(isSolvable(loyd)).toBe(false)
    expect(isSolvable(solvedBoard(4))).toBe(true)
    const p = slidingPuzzle(solvedBoard(4))
    let board = p.initial
    for (let t = 0; t < 40; t++) {
      const moves = p.actions(board)
      board = p.result(board, moves[(t * 7) % moves.length] as never)
      expect(isSolvable(board)).toBe(true)
    }
    for (let s = 0; s < 20; s++) expect(isSolvable(scrambledBoard(key(s), 3, 25).board)).toBe(true)
  })

  it('has the optimal depths of the 8-puzzle: the whole space of 181,440 boards and a deepest board of 31 moves', () => {
    const goal = solvedBoard(3)
    const seen = new Map<string, number>([[goal.join(), 0]])
    let frontier: (readonly number[])[] = [goal]
    const problem = slidingPuzzle(goal)
    let depth = 0
    while (frontier.length > 0) {
      const next: (readonly number[])[] = []
      for (const b of frontier) {
        for (const m of problem.actions(b)) {
          const c = problem.result(b, m)
          if (!seen.has(c.join())) {
            seen.set(c.join(), depth + 1)
            next.push(c)
          }
        }
      }
      frontier = next
      depth += 1
    }
    expect(seen.size).toBe(181_440)
    expect([...seen.values()].reduce((a, b) => Math.max(a, b), 0)).toBe(31)
    for (const board of [
      [1, 2, 3, 4, 5, 6, 0, 7, 8],
      [8, 6, 7, 2, 5, 4, 3, 0, 1],
      [6, 4, 7, 8, 5, 0, 3, 2, 1],
    ]) {
      const optimal = seen.get(board.join()) as number
      expect(optimalMoves(board)).toBe(optimal)
      expect(slidingPuzzle(board).heuristic?.(board) ?? Number.NaN).toBeLessThanOrEqual(optimal)
    }
    expect(optimalMoves([1, 2, 3, 4, 5, 6, 8, 7, 0])).toBeUndefined()
    expect(seen.get([8, 6, 7, 2, 5, 4, 3, 0, 1].join())).toBe(31)
  })

  it('solves 15-puzzle scrambles within the number of moves that made them', () => {
    for (let s = 0; s < 3; s++) {
      const { board, moves } = scrambledBoard(key(s), 4, 14)
      const r = idaStarSearch(slidingPuzzle(board))
      expect(r.status).toBe('solved')
      expect(r.cost).toBeLessThanOrEqual(moves)
      expect(r.cost % 2).toBe(moves % 2)
    }
    expect(() => slidingPuzzle([1, 1, 2, 3])).toThrow()
  })
})

describe('the Towers of Hanoi', () => {
  it('needs exactly 2^n - 1 moves', () => {
    for (const n of [1, 2, 3, 4, 5]) expect(breadthFirstSearch(towersOfHanoi(n)).cost).toBe(2 ** n - 1)
  })
})

describe('tours', () => {
  it('has Held-Karp agree with brute force and with the closed forms', () => {
    const r = randomTour(key(3), 7)
    const perms = (xs: number[]): number[][] =>
      xs.length <= 1 ? [xs] : xs.flatMap((x, i) => perms([...xs.slice(0, i), ...xs.slice(i + 1)]).map((p) => [x, ...p]))
    const brute = Math.min(...perms([1, 2, 3, 4, 5, 6]).map((p) => tourLength(r.distances, [0, ...p])))
    expect(r.optimalLength).toBeCloseTo(brute, 12)
    expect(tourLength(r.distances, r.optimalTour)).toBeCloseTo(r.optimalLength, 12)
    expect([...r.optimalTour].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6])
    for (const n of [5, 8, 11]) {
      const c = circleTour(n)
      expect(c.optimalLength).toBeCloseTo(2 * n * Math.sin(Math.PI / n), 12)
      expect(heldKarp(c.distances, n).length).toBeCloseTo(c.optimalLength, 10)
    }
    for (const [rows, cols] of [
      [4, 4],
      [2, 5],
      [3, 4],
      [6, 3],
    ] as const) {
      const g = gridTour(rows, cols)
      expect(g.optimalLength).toBeCloseTo(rows * cols, 12)
      expect(new Set(g.optimalTour).size).toBe(rows * cols)
      if (rows * cols <= 15) expect(heldKarp(g.distances, rows * cols).length).toBeCloseTo(rows * cols, 10)
    }
    expect(() => gridTour(3, 5)).toThrow()
  })

  it('plants a Hamiltonian cycle that a backtracking search finds', () => {
    for (let s = 0; s < 6; s++) {
      const p = plantedHamiltonian(key(s), 10, 6)
      expect(isHamiltonianCycle(p.adjacency, p.planted)).toBe(true)
      const found = findHamiltonianCycle(p.adjacency)
      expect(found).toBeDefined()
      expect(isHamiltonianCycle(p.adjacency, found ?? [])).toBe(true)
    }
    expect(findHamiltonianCycle([[1], [0, 2], [1]])).toBeUndefined()
    expect(
      isHamiltonianCycle(
        [
          [1, 2],
          [0, 2],
          [0, 1],
        ],
        [0, 1, 1],
      ),
    ).toBe(false)
  })
})

describe('satisfiability instances', () => {
  it('satisfies a planted instance by its planted assignment, at any ratio', () => {
    for (const ratio of [2, 4.3, 8]) {
      const p = plantedKSat(key(1), 40, ratio)
      expect(p.formula.clauses).toHaveLength(Math.round(40 * ratio))
      expect(satisfies(p.formula, p.assignment)).toBe(true)
      expect(cdcl(p.formula).status).toBe('sat')
    }
    expect(() => plantedKSat(key(0), 2, 3, 3)).toThrow()
  })

  it('shows the phase transition: the satisfiable fraction falls from one to zero across ratio 4.27', () => {
    const n = 36
    const trials = 40
    const fraction = (ratio: number): number => {
      let sat = 0
      for (let t = 0; t < trials; t++) {
        const f = randomKSat(foldIn(key(5), t * 100 + Math.round(ratio * 10)), n, ratio)
        if (cdcl(f).status === 'sat') sat += 1
      }
      return sat / trials
    }
    const low = fraction(3)
    const mid = fraction(SAT_THRESHOLDS[3])
    const high = fraction(6)
    expect(low).toBeGreaterThan(0.9)
    expect(high).toBeLessThan(0.1)
    expect(mid).toBeGreaterThan(high)
    expect(mid).toBeLessThan(low)
  })

  it('has the pigeonhole formulas unsatisfiable exactly when pigeons outnumber holes', () => {
    for (const holes of [1, 2, 3, 4, 5]) {
      expect(cdcl(pigeonhole(holes)).status, `${holes + 1} pigeons`).toBe('unsat')
      expect(cdcl(pigeonhole(holes, holes)).status, `${holes} pigeons`).toBe('sat')
    }
    expect(pigeonhole(3).clauses).toHaveLength(4 + 3 * 6)
  })
})
