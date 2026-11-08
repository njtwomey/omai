import { dependence } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { dependenceSurfaceView, dependenceView, iceRows } from './dependence.ts'
import { stillOf, viewFor } from './registry.ts'

type Opt = { grid: unknown[]; series: { name?: string; type?: string }[] }

const pdp = dependence({
  method: 'partial dependence',
  features: ['a', 'b'],
  output: 'y',
  curves: [
    {
      feature: 0,
      grid: [0, 1, 2],
      average: [0, 1, 2],
      individual: [
        [0, 1, 1],
        [0, 1, 3],
      ],
      rug: [0.5, 1.5],
    },
    { feature: 1, grid: [0, 1], average: [1, 0] },
  ],
  surfaces: [
    {
      features: [0, 1],
      gridX: [0, 1, 2],
      gridY: [0, 1],
      values: [
        [0, 1],
        [1, 2],
        [2, 3],
      ],
    },
  ],
})

describe('dependence views', () => {
  it('draws one feature with its ICE curves and rug, or every feature as small multiples', () => {
    const one = dependenceView(pdp, { curve: 0 }) as unknown as Opt
    expect(one.grid).toHaveLength(1)
    expect(one.series.some((s) => s.name === 'average')).toBe(true)
    expect(one.series.some((s) => s.name === '__rug')).toBe(true)
    expect((dependenceView(pdp) as unknown as Opt).grid).toHaveLength(2)
    expect(viewFor('dependence').component).toBe('DependenceView')
    expect(stillOf(pdp)).toBeDefined()
  })

  it('draws ALE with a zero line and a pair as a heatmap', () => {
    const ale = dependence({
      method: 'ale',
      features: ['a'],
      output: 'y',
      curves: [{ feature: 0, grid: [0, 1], average: [-0.5, 0.5], counts: [4] }],
    })
    expect((dependenceView(ale) as unknown as Opt).series.some((s) => s.name === 'ALE')).toBe(true)
    expect((dependenceSurfaceView(pdp) as unknown as Opt).series.length).toBeGreaterThan(0)
    expect(iceRows(1000, 5)).toEqual([0, 250, 500, 749, 999])
  })
})
