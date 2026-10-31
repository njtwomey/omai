import { describe, expect, it } from 'vitest'
import { facetCells, type MultiplesSpec, multiplesColumns, sharedLimits } from './multiples.ts'

describe('the cells of small multiples', () => {
  it('makes a cell per variant, each titled and bound to its variant', () => {
    const spec: MultiplesSpec = {
      kind: 'multiples',
      panel: { kind: 'curves', paths: ['loss'] },
      over: { by: 'variants' },
    }
    const cells = facetCells(spec, ['a', 'b'])
    expect(cells.map((c) => [c.label, c.variant])).toEqual([
      ['a', 0],
      ['b', 1],
    ])
    expect(cells[1]?.panel).toEqual({ kind: 'curves', paths: ['loss'], title: 'b' })
    // A session on its own has one unlabelled variant.
    expect(facetCells(spec, [''])[0]?.label).toBe('run')
  })

  it('makes a chart per path of a curves panel and keeps the rest of its options', () => {
    const spec: MultiplesSpec = {
      kind: 'multiples',
      panel: { kind: 'curves', paths: ['a', 'b'], log: 'y' },
      over: { by: 'paths' },
    }
    const cells = facetCells(spec, [''])
    expect(cells.map((c) => c.panel)).toEqual([
      { kind: 'curves', paths: ['a'], log: 'y', title: 'a' },
      { kind: 'curves', paths: ['b'], log: 'y', title: 'b' },
    ])
    expect(facetCells({ ...spec, panel: { kind: 'tape' } }, [''])).toEqual([])
  })

  it('extends the probe arguments by each value, keeping the arguments the panel has', () => {
    const spec: MultiplesSpec = {
      kind: 'multiples',
      panel: { kind: 'field', path: 'flow', args: { fixed: 1 } },
      over: { by: 'args', key: 'k', values: [1, 'x'], labels: ['one'] },
    }
    const cells = facetCells(spec, [''])
    expect(cells.map((c) => c.label)).toEqual(['one', 'k = x'])
    expect(cells.map((c) => (c.panel as { args: unknown }).args)).toEqual([
      { fixed: 1, k: 1 },
      { fixed: 1, k: 'x' },
    ])
  })

  it('lays out the cells in at most the columns asked for', () => {
    expect(multiplesColumns(2)).toBe(2)
    expect(multiplesColumns(7)).toBe(3)
    expect(multiplesColumns(7, 4)).toBe(4)
    expect(multiplesColumns(0, 0)).toBe(1)
  })

  it('shares no limits when nothing is recorded', () => {
    expect(sharedLimits([], ['loss'])).toBeUndefined()
  })
})
