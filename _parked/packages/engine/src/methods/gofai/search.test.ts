import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { mazeWorld, searchMap } from './maps.ts'
import { gridSearch, type PathSearchName } from './search.ts'

const ALL: PathSearchName[] = [
  'breadth-first',
  'depth-first',
  'uniform-cost',
  'greedy',
  'a-star',
  'weighted-a-star',
  'ida-star',
  'bidirectional',
]

describe('gridSearch', () => {
  for (const algorithm of ALL)
    it(`plays ${algorithm} on a maze`, () => {
      const w = mazeWorld(key(1), { rows: 6, cols: 8 })
      const r = gridSearch(w, { algorithm })
      expect(r.status).toBe('solved')
      expect(r.path[0]).toBe(w.starts[0])
      expect(r.path.at(-1)).toBe(w.goals[0])
      const [board] = recordsOf(r.exposition, 'boardSteps')
      expect(board?.layout).toEqual({ type: 'grid', rows: 13, columns: 17 })
      expect(board?.steps.length).toBeGreaterThan(1)
      expect(board?.steps.at(-1)?.cells.filter((c) => c === 4).length).toBeGreaterThan(0)
    })

  it('finds cheapest paths with uniform cost and A*, not with breadth-first search, across a river', () => {
    const w = searchMap('river')
    const ucs = gridSearch(w, { algorithm: 'uniform-cost' })
    const astar = gridSearch(w, { algorithm: 'a-star' })
    const bfs = gridSearch(w, { algorithm: 'breadth-first' })
    expect(astar.cost).toBe(ucs.cost)
    expect(bfs.cost).toBeGreaterThan(ucs.cost)
    expect(astar.expanded).toBeLessThan(ucs.expanded)
  })

  it('lets greedy best-first search into the cup that A* avoids', () => {
    const w = searchMap('detour')
    const greedy = gridSearch(w, { algorithm: 'greedy' })
    const astar = gridSearch(w, { algorithm: 'a-star' })
    expect(greedy.cost).toBeGreaterThanOrEqual(astar.cost)
  })
})
