import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { eightPuzzle, nQueensProblem, pythonOrder, type RoadMap, roadProblem } from '../../testing/search.ts'
import {
  bidirectionalSearch,
  breadthFirstSearch,
  depthFirstSearch,
  depthLimitedSearch,
  iterativeDeepeningSearch,
  type SearchProblem,
  type SearchResult,
  uniformCostSearch,
} from './index.ts'

interface Expected {
  readonly status: string
  readonly actions: readonly unknown[]
  readonly cost: number | null
  readonly depth: number
  readonly expanded: number
  readonly goalTests: number
  readonly generated: number
}

function problemOf(p: Readonly<Record<string, unknown>>): SearchProblem<unknown, unknown> {
  if (p['problem'] === 'romania') {
    return roadProblem(p['neighbours'] as RoadMap, p['start'] as string, p['goal'] as string) as SearchProblem<
      unknown,
      unknown
    >
  }
  if (p['problem'] === 'eight-puzzle') return eightPuzzle(p['tiles'] as number[]) as SearchProblem<unknown, unknown>
  return nQueensProblem(p['n'] as number) as SearchProblem<unknown, unknown>
}

function run(fn: string, problem: SearchProblem<unknown, unknown>, limit: number): SearchResult<unknown, unknown> {
  if (fn === 'breadthFirst') return breadthFirstSearch(problem)
  if (fn === 'depthFirst') return depthFirstSearch(problem)
  if (fn === 'depthLimited') return depthLimitedSearch(problem, limit)
  if (fn === 'iterativeDeepening') return iterativeDeepeningSearch(problem)
  return uniformCostSearch(problem, { tieBreak: pythonOrder })
}

describe('uninformed search against aima-python', () => {
  for (const c of loadFixture('search/uninformed').cases) {
    it(c.name, () => {
      const p = c.params
      const problem = problemOf(p)
      if (c.fn === 'bidirectional') {
        const r = bidirectionalSearch({ ...problem, goal: p['goal'] as string })
        expect(r.cost).toBe(p['cost'])
        expect(r.path[r.path.length - 1]).toBe(p['goal'])
        return
      }
      const e = p['expected'] as Expected
      const r = run(c.fn, problem, (p['limit'] as number | undefined) ?? 0)
      expect(r.status).toBe(e.status)
      expect(r.actions).toEqual(e.actions)
      expect(r.depth).toBe(e.depth)
      if (e.cost !== null) expect(r.cost).toBe(e.cost)
      expect(r.stats.expanded).toBe(e.expanded)
      expect(r.stats.goalTests).toBe(e.goalTests)
      expect(r.stats.generated).toBe(e.generated)
    })
  }
})
