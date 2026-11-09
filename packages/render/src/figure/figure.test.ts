import { describe, expect, it } from 'vitest'
import { boxForAspect, fitDataAspect } from './aspect.ts'
import { figure, MEDIA, parseAspect, plotArea, scaleFigure, squarePlot } from './figure.ts'
import { bento, gridSpec, layoutGrid, linkedGroups, sharedGroups, withMarginals } from './grid.ts'
import { extentOf, niceRange, padRange, resolveLimits, unionRange } from './limits.ts'
import { brushSelection, linkStore, toggleSelection } from './linking.ts'

describe('figure', () => {
  it('has a default size and aspect per medium', () => {
    for (const medium of Object.keys(MEDIA) as (keyof typeof MEDIA)[]) {
      const f = figure({ medium })
      expect(f.width / f.height).toBeCloseTo(MEDIA[medium].aspect, 1)
    }
    expect(figure({ width: 300, height: 100 }).aspect).toBe(3)
    expect(figure({ title: 't', mode: 'dark' }).mode).toBe('dark')
    expect(() => parseAspect('1:0')).toThrow()
    expect(() => figure({ width: 0 })).toThrow()
  })
  it('scales padding and type with width', () => {
    const f = scaleFigure(figure(), 320)
    expect(f.padding.left).toBe(28)
    expect(plotArea(f).height).toBe(f.height - f.padding.top - f.padding.bottom)
  })
})

describe('grid', () => {
  it('places spans across gaps', () => {
    const g = gridSpec(2, 2, {
      wspace: 10,
      hspace: 0,
      heightRatios: [1, 3],
      cells: [
        { id: 'top', row: 0, col: 0, colSpan: 2 },
        { id: 'a', row: 1, col: 0 },
        { id: 'b', row: 1, col: 1, sharey: 'a' },
      ],
    })
    const r = layoutGrid(g, { x: 5, y: 0, width: 210, height: 400 })
    expect(r['top']).toEqual({ x: 5, y: 0, width: 210, height: 100 })
    expect(r['b']).toEqual({ x: 115, y: 100, width: 100, height: 300 })
    expect(sharedGroups(g, 'y')).toEqual([['a', 'b']])
    expect(linkedGroups(g, 'x')).toEqual([])
  })
  it('rejects bad grids', () => {
    expect(() => gridSpec(0, 1)).toThrow()
    expect(() => gridSpec(1, 1, { widthRatios: [1, 2] })).toThrow()
    expect(() => gridSpec(1, 1, { widthRatios: [-1] })).toThrow()
    expect(() => gridSpec(1, 2, { cells: [{ id: 'a', row: 0, col: 0, colSpan: 3 }] })).toThrow(/outside/)
    expect(() =>
      gridSpec(1, 2, {
        cells: [
          { id: 'a', row: 0, col: 0, colSpan: 2 },
          { id: 'b', row: 0, col: 1 },
        ],
      }),
    ).toThrow(/overlap/)
    expect(() => gridSpec(1, 1, { cells: [{ id: 'a', row: 0, col: 0, sharex: 'z' }] })).toThrow(/unknown/)
    expect(() =>
      gridSpec(1, 2, {
        cells: [
          { id: 'a', row: 0, col: 0 },
          { id: 'a', row: 0, col: 1 },
        ],
      }),
    ).toThrow(/duplicate/)
  })
  it('builds marginal and bento layouts', () => {
    const m = withMarginals({ sides: ['right'], ratio: 0.5 })
    expect(m.cells.map((c) => c.id)).toEqual(['main', 'right'])
    const boxes = layoutGrid(withMarginals(), { x: 0, y: 0, width: 500, height: 500 })
    expect((boxes['main']?.x ?? 0) + (boxes['main']?.width ?? 0)).toBe(boxes['right']?.x)
    const b = bento(['a . b', 'c c b'], { kinds: { c: 'tex' } })
    expect(b.cells.find((c) => c.id === 'c')?.kind).toBe('tex')
    expect(b.cells).toHaveLength(3)
    expect(() => bento(['a b a'])).toThrow(/rectangle/)
    expect(() => bento(['a b', 'c'])).toThrow()
  })
})

describe('aspect', () => {
  it('widens the short range so units are square', () => {
    const { x, y } = fitDataAspect([0, 1], [0, 1], { width: 300, height: 100 }, 'equal')
    expect((x[1] - x[0]) / 300).toBeCloseTo((y[1] - y[0]) / 100)
    const r = fitDataAspect([0, 1], [0, 1], { width: 100, height: 100 }, 2)
    expect(r.x[1] - r.x[0]).toBeCloseTo(2)
    expect(fitDataAspect([0, 1], [0, 1], { width: 0, height: 1 }, 'equal').x).toEqual([0, 1])
    expect(fitDataAspect([0, 0], [0, 1], { width: 1, height: 1 }, 'equal').x).toEqual([0, 0])
    expect(() => fitDataAspect([0, 1], [0, 1], { width: 1, height: 1 }, -1)).toThrow()
    expect(boxForAspect([0, 1], [0, 2], { width: 300, height: 300 }, 'equal')).toEqual({ width: 150, height: 300 })
    expect(boxForAspect([0, 1], [0, 2], { width: 3, height: 3 }, 'auto')).toEqual({ width: 3, height: 3 })
  })
})

describe('limits', () => {
  it('resolves every policy', () => {
    expect(resolveLimits({ kind: 'auto' }, undefined)).toEqual([0, 1])
    expect(resolveLimits({ kind: 'auto', nice: true, includeZero: true }, [3, 9.2])).toEqual([0, 10])
    expect(resolveLimits({ kind: 'expand', pad: 0 }, [1, 3])).toEqual([1, 3])
    expect(resolveLimits({ kind: 'expand', pad: 0, nice: true }, [1, 3], [2, 2.5])).toEqual([1, 3])
    expect(resolveLimits({ kind: 'expand' }, undefined, [2, 3])).toEqual([2, 3])
    expect(resolveLimits({ kind: 'follow', window: 5, lead: 1 }, undefined)).toEqual([1, 6])
    expect(() => resolveLimits([1, 1], undefined)).toThrow()
    expect(() => resolveLimits({ kind: 'follow', window: 0 }, undefined)).toThrow()
    expect(padRange([0, 0])).toEqual([-1, 1])
    expect(niceRange([-0.3, 0.3])[0]).toBeLessThanOrEqual(-0.3)
    expect(extentOf([Number.POSITIVE_INFINITY])).toBeUndefined()
    expect(unionRange([])).toBeUndefined()
  })
})

describe('squarePlot', () => {
  it('makes the plotting area square at the same width, keeping the padding', () => {
    const fig = squarePlot(figure({ width: 400 }))
    const area = plotArea(fig)
    expect(area.width).toBe(area.height)
    expect(fig.width).toBe(400)
    expect(fig.padding).toEqual(figure({ width: 400 }).padding)
    expect(fig.aspect).toBeCloseTo(fig.width / fig.height, 12)
    // Too narrow to hold a plotting area: unchanged.
    const tiny = figure({ width: 60 })
    expect(squarePlot(tiny)).toBe(tiny)
  })
})

describe('linking', () => {
  it('shares hover and selection between subscribers', () => {
    const link = linkStore({ selection: [1] })
    const seen: unknown[] = []
    const stop = link.subscribe((s) => seen.push(s.selection))
    link.set((s) => ({ ...s, selection: toggleSelection(s.selection, 2) }))
    link.set((s) => s)
    stop()
    link.set({ hover: 'a' })
    expect(seen).toEqual([[1, 2]])
    expect(link.get().hover).toBe('a')
    expect(
      brushSelection(
        [
          [0, 0],
          [1, 5],
        ],
        { y: [1, 6] },
      ),
    ).toEqual([1])
  })
})
