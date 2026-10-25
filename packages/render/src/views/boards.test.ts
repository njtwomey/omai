import { boardSteps, populationSteps } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { boardColours, boardFrames, populationFrames } from './boards.ts'
import { stillOf, viewFor } from './registry.ts'

const seriesOf = (o: object): { name?: string; type?: string }[] => (o as { series: { name?: string }[] }).series

describe('board and population views', () => {
  it('colours well-known categories alike and the rest by the palette', () => {
    const [frontier, other, again] = boardColours(['frontier', 'a', 'b'])
    expect(frontier).not.toBe(other)
    expect(other).not.toBe(again)
    expect(boardColours(['explored'])).toEqual(boardColours(['visited']))
  })

  it('plays a grid board with labels, a legend of the categories used and metrics', () => {
    const r = boardSteps({
      method: 'bfs',
      layout: { type: 'grid', rows: 2, columns: 3 },
      categories: ['empty', 'wall', 'explored', 'frontier'],
      steps: [
        { cells: [3, 0, 0, 1, 0, 0], caption: 'start' },
        { cells: [2, 3, 0, 1, 3, 0], labels: ['0', '1', '', '', '1', ''], caption: 'expand (0, 0)' },
      ],
      metrics: [{ name: 'expanded', values: [0, 1] }],
      stepName: 'expansion',
    })
    const frames = boardFrames(r)
    expect(frames.count).toBe(2)
    expect(frames.label(1)).toBe('expansion 1/1 · expand (0, 0)')
    const names = seriesOf(frames.option(1)).map((s) => s.name)
    expect(names).toContain('__board')
    expect(names).toContain('__labels')
    expect(names).toContain('frontier')
    expect(names).not.toContain('empty-never-used')
    expect(viewFor('boardSteps').component).toBe('BoardPlayer')
    expect(stillOf(r)).toBeTruthy()
  })

  it('draws a graph board as edges and nodes by category', () => {
    const r = boardSteps({
      method: 'alpha-beta',
      layout: {
        type: 'graph',
        nodes: [
          { x: 0, y: 0, label: 'A' },
          { x: -1, y: 1 },
          { x: 1, y: 1 },
        ],
        edges: [
          [0, 1],
          [0, 2],
        ],
      },
      categories: ['unvisited', 'visited', 'pruned'],
      steps: [{ cells: [1, 1, 2], labels: ['3', '3', ''] }],
    })
    const names = seriesOf(boardFrames(r).option(0)).map((s) => s.name)
    expect(names).toEqual(expect.arrayContaining(['__edges', 'visited', 'pruned', '__labels']))
  })

  it('plays a population over its surface, a program over its data, and plain fitness', () => {
    const surface = {
      x: [-1, 0, 1],
      y: [-1, 0, 1],
      values: [
        [2, 1, 2],
        [1, 0, 1],
        [2, 1, 2],
      ],
      optimum: [0, 0] as const,
    }
    const gen = (f: number) => ({
      fitness: [f, f + 1],
      points: [
        [0.5, 0.5],
        [-0.5, 0.2],
      ] as [number, number][],
      mean: [0, 0] as const,
      covariance: [0.2, 0, 0.2] as const,
      best: { fitness: f, point: [0.5, 0.5] as const },
    })
    const plane = populationSteps({ method: 'cma-es', lowerIsBetter: true, surface, generations: [gen(2), gen(1)] })
    const frames = populationFrames(plane)
    expect(frames.count).toBe(2)
    expect(frames.label(1)).toBe('generation 1/1 · best 1')
    expect(seriesOf(frames.option(1)).map((s) => s.name)).toEqual(
      expect.arrayContaining(['population', 'best', 'optimum']),
    )
    const program = populationSteps({
      method: 'gp',
      data: { x: [0, 1], y: [0, 1], grid: [0, 0.5, 1] },
      generations: [{ fitness: [1], best: { fitness: 1, label: 'x', curve: [0, 0.5, 1] } }],
    })
    expect(seriesOf(populationFrames(program).option(0)).map((s) => s.name)).toContain('best program')
    const plain = populationSteps({ method: 'ga', generations: [{ fitness: [1, 2], best: { fitness: 2 } }] })
    expect(stillOf(plain)).toBeTruthy()
    expect(viewFor('populationSteps').component).toBe('PopulationPlayer')
  })
})
