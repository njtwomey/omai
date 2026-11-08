import { describe, expect, it } from 'vitest'
import { eightPuzzle, nQueensProblem } from '../../testing/search.ts'
import { isOmaiError } from '../base/index.ts'
import { graph } from '../graph/index.ts'
import { messages, policies, recorder, runAlgorithm, validateAgainstSchema } from '../trace/index.ts'
import {
  aStarSearch,
  beamSearch,
  bidirectionalSearch,
  breadthFirstSearch,
  breadthFirstSearchAlgorithm,
  childNode,
  depthFirstSearch,
  depthLimitedSearch,
  effectiveBranchingFactor,
  graphProblem,
  idaStarSearch,
  iterativeDeepeningSearch,
  recursiveBestFirstSearch,
  replay,
  rootNode,
  type SearchProblem,
  searchTree,
  uniformCostSearch,
} from './index.ts'

/** The integers, unbounded: every search must stop on its budget. */
const integers: SearchProblem<number, number> = {
  initial: 0,
  actions: () => [-1, 1],
  result: (s, a) => s + a,
  isGoal: () => false,
}

describe('budgets and termination', () => {
  it('stops every search on an infinite space without a goal', () => {
    const options = { maxExpansions: 50 }
    for (const r of [
      breadthFirstSearch(integers, options),
      depthFirstSearch(integers, options),
      depthFirstSearch(integers, { ...options, graph: false }),
      uniformCostSearch(integers, options),
      aStarSearch(integers, options),
      idaStarSearch(integers, options),
      recursiveBestFirstSearch(integers, options),
      beamSearch(integers, options),
      iterativeDeepeningSearch(integers, options),
      bidirectionalSearch({ ...integers, goal: 1000 }, { maxExpansions: 50 }),
    ]) {
      expect(r.status).toBe('budget')
      expect(r.stats.expanded).toBeLessThanOrEqual(50 + 3)
      expect(r.cost).toBe(Number.POSITIVE_INFINITY)
    }
  })

  it('reports failure on finite spaces without a goal and cutoff at a depth limit', () => {
    const finite: SearchProblem<number, number> = {
      initial: 0,
      actions: (s) => (s < 5 ? [1] : []),
      result: (s, a) => s + a,
      isGoal: () => false,
    }
    expect(breadthFirstSearch(finite).status).toBe('failure')
    expect(depthFirstSearch(finite).status).toBe('failure')
    expect(uniformCostSearch(finite).status).toBe('failure')
    expect(idaStarSearch(finite).status).toBe('failure')
    expect(recursiveBestFirstSearch(finite).status).toBe('failure')
    expect(depthLimitedSearch(finite, 3).status).toBe('cutoff')
    expect(depthLimitedSearch(finite, 9).status).toBe('failure')
    expect(iterativeDeepeningSearch(finite).status).toBe('failure')
    expect(iterativeDeepeningSearch(integers, { maxDepth: 4 }).status).toBe('cutoff')
    expect(bidirectionalSearch({ ...finite, goal: 9 }).status).toBe('failure')
  })

  it('solves the initial state immediately', () => {
    const p = { ...integers, isGoal: (s: number) => s === 0 }
    for (const r of [breadthFirstSearch(p), depthFirstSearch(p), uniformCostSearch(p), idaStarSearch(p)]) {
      expect(r.status).toBe('solved')
      expect(r.path).toEqual([0])
      expect(r.cost).toBe(0)
      expect(Number.isNaN(r.effectiveBranchingFactor)).toBe(true)
    }
    expect(bidirectionalSearch({ ...p, goal: 0 }).path).toEqual([0])
  })

  it('rejects bad arguments', () => {
    const bad = { ...integers, stepCost: () => -1 }
    expect(() => uniformCostSearch(bad)).toThrow()
    try {
      childNode(bad, rootNode(bad), 1)
    } catch (e) {
      expect(isOmaiError(e, 'value')).toBe(true)
    }
    expect(() => depthLimitedSearch(integers, -1)).toThrow()
    expect(() => beamSearch(integers, { width: 0 })).toThrow()
    expect(() => breadthFirstSearch(integers, { maxExpansions: -1 })).toThrow()
    expect(() => replay(integers, [3])).toThrow()
  })
})

describe('textbook problems', () => {
  it('solves n-queens by depth-first search with valid placements', () => {
    for (const n of [4, 6, 8]) {
      const r = depthFirstSearch(nQueensProblem(n))
      expect(r.status).toBe('solved')
      const rows = r.path[r.path.length - 1] ?? []
      expect(new Set(rows).size).toBe(n)
    }
    expect(breadthFirstSearch(nQueensProblem(3)).status).toBe('failure')
  })

  it('agrees on the eight puzzle optimum across searches', () => {
    const tiles = [1, 6, 2, 5, 3, 0, 4, 7, 8]
    const depth = breadthFirstSearch(eightPuzzle(tiles)).depth
    expect(aStarSearch(eightPuzzle(tiles, 'manhattan')).depth).toBe(depth)
    expect(idaStarSearch(eightPuzzle(tiles, 'manhattan')).depth).toBe(depth)
    expect(iterativeDeepeningSearch(eightPuzzle(tiles)).depth).toBe(depth)
    const goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    const reversible = { ...eightPuzzle(tiles), goal }
    expect(bidirectionalSearch(reversible).depth).toBe(depth)
  })

  it('computes the effective branching factor', () => {
    expect(effectiveBranchingFactor(52, 5)).toBeCloseTo(1.92, 2)
    expect(effectiveBranchingFactor(3, 3)).toBe(1)
    expect(effectiveBranchingFactor(1, 3)).toBeLessThan(1)
    expect(Number.isNaN(effectiveBranchingFactor(10, 0))).toBe(true)
  })
})

describe('traces and search trees', () => {
  it('records expansions, generations and the solution as declared messages', () => {
    const g = graph([
      ['a', 'b', 1],
      ['b', 'c', 1],
      ['a', 'c', 5],
    ])
    const rec = recorder()
    const r = uniformCostSearch(graphProblem(g, 'a', 'c'), { tracer: rec, policy: policies.teaching })
    expect(r.path).toEqual([0, 1, 2])
    const run = rec.run()
    expect(messages(run, 'expand').map((m) => (m.payload as { node: string }).node)).toEqual(['a', 'b', 'c'])
    expect(messages(run, 'solution')[0]?.payload).toEqual({ path: ['a', 'b', 'c'], cost: 2 })
    const algorithm = breadthFirstSearchAlgorithm(graphProblem(g, 'a', 'c'))
    const rec2 = recorder()
    runAlgorithm(algorithm, { maxSteps: 10, tracer: rec2, policy: policies.teaching })
    expect(
      validateAgainstSchema(
        rec2.events,
        algorithm.schema ?? { metrics: {}, tensors: {}, messages: {}, episodes: false, frames: false },
      ),
    ).toEqual([])
    expect(messages(rec2.run(), 'generate').length).toBeGreaterThan(0)
  })

  it('draws the search tree with the solution highlighted', () => {
    const p: SearchProblem<number, number> = {
      initial: 1,
      actions: (s) => (s < 8 ? [0, 1] : []),
      result: (s, a) => 2 * s + a,
      isGoal: (s) => s === 6,
    }
    const r = breadthFirstSearch(p, { recordTree: true })
    const s = searchTree(r)
    expect(s.type).toBe('search-tree')
    expect(s.nodes.length).toBe(r.tree?.length)
    expect(s.edges.length).toBe(s.nodes.length - 1)
    expect(s.annotations?.length).toBe(3)
    expect(() => searchTree(breadthFirstSearch(p))).toThrow()
  })
})
