import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { eightPuzzle, pythonOrder, type RoadMap, roadProblem } from '../../testing/search.ts'
import {
  aStarSearch,
  greedyBestFirstSearch,
  idaStarSearch,
  recursiveBestFirstSearch,
  type SearchProblem,
  type SearchResult,
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
    const h = p['h'] as Record<string, number>
    return roadProblem(p['neighbours'] as RoadMap, p['start'] as string, p['goal'] as string, h) as SearchProblem<
      unknown,
      unknown
    >
  }
  return eightPuzzle(p['tiles'] as number[], p['heuristic'] as 'misplaced' | 'manhattan') as SearchProblem<
    unknown,
    unknown
  >
}

function run(fn: string, problem: SearchProblem<unknown, unknown>): SearchResult<unknown, unknown> {
  if (fn === 'aStar' || fn === 'networkx') return aStarSearch(problem, { tieBreak: pythonOrder })
  if (fn === 'greedy') return greedyBestFirstSearch(problem, { tieBreak: pythonOrder })
  return recursiveBestFirstSearch(problem)
}

describe('informed search against aima-python', () => {
  for (const c of loadFixture('search/informed').cases) {
    it(c.name, () => {
      const p = c.params
      const problem = problemOf(p)
      const r = run(c.fn, problem)
      if (c.fn === 'networkx') {
        // NetworkX's A* reopens closed nodes, so it is optimal even where aima-python's graph search is not; the
        // optimum is what recursive best-first search and IDA* (tree searches) reach with this heuristic.
        expect(recursiveBestFirstSearch(problem).cost).toBe(p['cost'])
        expect(idaStarSearch(problem).cost).toBe(p['cost'])
        expect(r.cost).toBeGreaterThanOrEqual(p['cost'] as number)
        return
      }
      const e = p['expected'] as Expected
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
