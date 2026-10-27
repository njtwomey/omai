import { describe, expect, it } from 'vitest'
import { SEARCH_MAPS } from '../../lib/problems/index.ts'
import { metricSeries, type Run, session, tensorSeries } from '../../lib/trace/index.ts'
import { BOARD_CELLS, gridSearch } from './gridSearch.ts'

describe('gridSearch', () => {
  it('plays A* on the first map to the goal with a board per step', () => {
    const map = Object.keys(SEARCH_MAPS)[0] as 'open'
    const s = session(gridSearch, { mode: 'eager', steps: 1000, config: { algorithm: 'a-star', map } })
    expect(s.status).toBe('converged')
    const run: Run = s.run()
    const boards = tensorSeries(run, 'board')
    expect(boards.length).toBe(s.t)
    const rows = SEARCH_MAPS[map].length
    const cols = SEARCH_MAPS[map][0]?.length ?? 0
    const last = boards.at(-1)
    expect(last?.shape).toEqual([rows, cols])
    // The solution is drawn between the start and the goal, and the effort is on the tape.
    expect(last?.values?.filter((v) => v === BOARD_CELLS.path).length).toBeGreaterThan(10)
    expect(last?.values?.filter((v) => v === BOARD_CELLS.goal).length).toBe(1)
    expect(metricSeries(run, 'frontier').values.length).toBe(s.t)
    expect(metricSeries(run, 'pathCost').values.at(-1)).toBe(13)
  })

  it('finds the cheaper way round the water with uniform-cost search', () => {
    const cost = (algorithm: string): number => {
      const s = session(gridSearch, { mode: 'eager', steps: 1000, config: { algorithm, map: 'river' } })
      return metricSeries(s.run(), 'pathCost').values.at(-1) ?? Number.NaN
    }
    expect(cost('uniform-cost')).toBeLessThan(cost('breadth-first'))
  })
})
