import { describe, expect, it } from 'vitest'
import { boardSteps } from './boards.ts'
import { populationSteps } from './populations.ts'
import { isExposition } from './record.ts'

describe('board and population steps', () => {
  const grid = { type: 'grid', rows: 2, columns: 2 } as const

  it('check cells, labels, categories, edges and metrics against the layout', () => {
    const ok = boardSteps({ method: 'bfs', layout: grid, categories: ['a', 'b'], steps: [{ cells: [0, 1, 0, 0] }] })
    expect(isExposition(ok)).toBe(true)
    expect(JSON.parse(JSON.stringify(ok))).toEqual(ok)
    expect(() => boardSteps({ method: 'x', layout: grid, categories: ['a'], steps: [{ cells: [0] }] })).toThrow(
      /steps\[0\]\.cells/,
    )
    expect(() =>
      boardSteps({ method: 'x', layout: grid, categories: ['a'], steps: [{ cells: [0, 0, 0, 0], labels: [''] }] }),
    ).toThrow(/labels/)
    expect(() =>
      boardSteps({ method: 'x', layout: grid, categories: ['a'], steps: [{ cells: [0, 0, 0, 1] }] }),
    ).toThrow(/category 1/)
    expect(() =>
      boardSteps({
        method: 'x',
        layout: { type: 'graph', nodes: [{ x: 0, y: 0 }], edges: [[0, 1]] },
        categories: ['a'],
        steps: [],
      }),
    ).toThrow(/edge/)
    expect(() =>
      boardSteps({
        method: 'x',
        layout: grid,
        categories: ['a'],
        steps: [{ cells: [0, 0, 0, 0] }],
        metrics: [{ name: 'm', values: [] }],
      }),
    ).toThrow(/metrics\.m/)
  })

  it('check a population against its surface, points and curves', () => {
    const surface = {
      x: [0, 1],
      y: [0, 1],
      values: [
        [0, 1],
        [1, 2],
      ],
    }
    const g = {
      fitness: [1, 2],
      points: [
        [0, 0],
        [1, 1],
      ] as [number, number][],
      best: { fitness: 1 },
    }
    expect(populationSteps({ method: 'es', surface, generations: [g] }).kind).toBe('populationSteps')
    expect(() => populationSteps({ method: 'es', surface: { ...surface, y: [0] }, generations: [] })).toThrow(
      /surface\.values/,
    )
    expect(() => populationSteps({ method: 'es', generations: [{ ...g, fitness: [1] }] })).toThrow(/points/)
    expect(() =>
      populationSteps({
        method: 'gp',
        data: { x: [0], y: [0], grid: [0, 1] },
        generations: [{ fitness: [0], best: { fitness: 0, curve: [0] } }],
      }),
    ).toThrow(/curve/)
  })
})
