import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { mapColouring, queensCsp, sendMoreMoney, sudokuCsp } from '../../testing/search.ts'
import { type Assignment, ac3, backtrackingSearch, type Csp, isSolution } from './index.ts'

const INFERENCES = ['none', 'forward-checking', 'mac'] as const

/** Every solution under each inference, as rows of values in variable order, sorted. */
function allSolutions<D>(problem: Csp<D>, options: Parameters<typeof backtrackingSearch>[1] = {}): D[][][] {
  return INFERENCES.map((inference) =>
    backtrackingSearch(problem, { ...options, inference, all: true })
      .solutions.map((s: Assignment<D>) => problem.variables.map((v) => s[v] as D))
      .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
  )
}

describe('constraint satisfaction against python-constraint and aima-python', () => {
  for (const c of loadFixture('search/csp').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'queens') {
        const problem = queensCsp(p['n'] as number)
        for (const sols of allSolutions(problem)) {
          expect(sols.length).toBe(p['count'])
          if ((p['solutions'] as unknown[]).length > 0) expect(sols).toEqual(p['solutions'])
        }
        // Chronological backtracking in static order finds the lexicographically first solution.
        const first = backtrackingSearch(problem).assignment ?? {}
        expect(problem.variables.map((v) => first[v])).toEqual(p['first'])
        for (const variableOrder of ['mrv', 'degree', 'mrv-degree'] as const) {
          const r = backtrackingSearch(problem, { variableOrder, valueOrder: 'lcv', inference: 'forward-checking' })
          expect(isSolution(problem, r.assignment ?? {})).toBe(true)
        }
      } else if (c.fn === 'australia') {
        const problem = mapColouring(
          p['regions'] as string[],
          p['borders'] as [string, string][],
          p['colours'] as string[],
        )
        for (const sols of allSolutions(problem)) {
          expect(sols).toEqual(
            (p['solutions'] as string[][])
              .map((s) => s)
              .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
          )
        }
      } else if (c.fn === 'cryptarithm') {
        const problem = sendMoreMoney()
        for (const inference of INFERENCES) {
          const r = backtrackingSearch(problem, { inference, all: true, variableOrder: 'mrv' })
          expect(r.solutions).toEqual(p['solutions'])
        }
      } else if (c.fn === 'sudoku4') {
        const problem = sudokuCsp(p['grid'] as string, 4)
        for (const sols of allSolutions(problem, { variableOrder: 'mrv' })) {
          expect(sols.map((s) => s.join(''))).toEqual(p['solutions'])
        }
      } else if (c.fn === 'sudoku9') {
        const problem = sudokuCsp(p['grid'] as string)
        const r = backtrackingSearch(problem, { variableOrder: 'mrv', inference: 'forward-checking' })
        expect(problem.variables.map((v) => r.assignment?.[v]).join('')).toBe(p['solution'])
        const mac = backtrackingSearch(problem, { variableOrder: 'mrv', inference: 'mac' })
        expect(mac.assignment).toEqual(r.assignment)
      } else if (c.fn === 'ac3Australia') {
        const problem = mapColouring(
          p['regions'] as string[],
          p['borders'] as [string, string][],
          p['colours'] as string[],
          p['fixed'] as Record<string, string>,
        )
        const r = ac3(problem)
        expect(r.consistent).toBe(p['consistent'])
        if (r.consistent) expect(r.domains).toEqual(p['domains'])
      } else if (c.fn === 'ac3Queens') {
        const r = ac3(queensCsp(p['n'] as number, p['row'] as number))
        expect(r.consistent).toBe(p['consistent'])
        if (r.consistent) expect(r.domains).toEqual(p['domains'])
      } else {
        const problem = sudokuCsp(p['grid'] as string)
        const r = ac3(problem)
        expect(r.consistent).toBe(p['consistent'])
        expect(problem.variables.map((v) => (r.domains[v]?.length === 1 ? r.domains[v]?.[0] : '.')).join('')).toBe(
          p['solution'],
        )
      }
    })
  }
})
