import type { Pixels } from 'omai-engine/scene'
import { describe, expect, it } from 'vitest'
import { parseColour } from '../theme/palettes.ts'
import { tokens } from '../theme/tokens.ts'
import { heatmap, heatmapChart, matrix } from './matrices.ts'
import { cartesian, layer } from './plot.ts'
import { colourSelection, DIMMED_OPACITY, dimColour, selectionFilter } from './selection.ts'

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any
const opt = (o: unknown): Opt => o as Opt
const api = {
  coord: (p: readonly number[]) => [...p],
  size: (p: readonly number[]) => p.map((v) => v * 30),
  value: (d: number) => d,
}
// Pixels pass through unencoded, so the tests read the repainted image directly.
const raw = (p: Pixels): Pixels => p
const imageOf = (update: Opt): Pixels => update.series[0].renderItem({}, api).style.image
const rgbAt = (p: Pixels, x: number, y: number): number[] => {
  const i = 4 * (y * p.width + x)
  return [p.rgba[i] ?? 0, p.rgba[i + 1] ?? 0, p.rgba[i + 2] ?? 0]
}

describe('selectionFilter', () => {
  const bar = { min: 0, max: 10 }
  it('keeps the values between the handles and leaves ends at the bar open', () => {
    const keep = selectionFilter(bar, [2, 10])
    expect([keep?.(1), keep?.(2), keep?.(50)]).toEqual([0, 1, 1])
    expect(selectionFilter(bar, [0, 10])).toBeUndefined()
    expect(selectionFilter(bar, [5, 3])?.(4)).toBe(1)
  })
  it('ramps over a spread at the edges', () => {
    const keep = selectionFilter(bar, [2, 5])
    expect([keep?.(1, 2), keep?.(2, 2), keep?.(3, 2)]).toEqual([0, 0.5, 1])
  })
  it('merges adjacent selected pieces, with no seam between them', () => {
    const pieces = {
      pieces: [
        { min: 0, max: 1 },
        { min: 1, max: 2 },
        { min: 2, max: 3 },
      ],
    }
    const keep = selectionFilter(pieces, { 0: true, 1: true, 2: false })
    expect([keep?.(0.5), keep?.(1, 0.5), keep?.(2.5), keep?.(-1)]).toEqual([1, 1, 0, 1])
    expect(selectionFilter(pieces, { 0: true, 1: true, 2: true })).toBeUndefined()
  })
})

describe('dimColour', () => {
  it('blends towards the background', () => {
    expect(dimColour([100, 100, 100], [0, 0, 0], 0)).toEqual([15, 15, 15])
    expect(dimColour([100, 100, 100], [0, 0, 0], 1)).toEqual([100, 100, 100])
  })
})

describe('colour bar selections', () => {
  const z = [
    [0, 1, 2, 3],
    [0, 1, 2, 3],
  ]
  for (const mode of ['light', 'dark'] as const)
    it(`repaints a heatmap's image with the values outside dimmed (${mode})`, () => {
      const option = opt(heatmapChart(z, { smooth: 'cells', ramp: ['#000000', '#ffffff'] }, { mode }))
      const map = option.visualMap[0]
      expect(map.calculable).toBe(true)
      const update = opt(colourSelection(option, { visualMapId: map.id, selected: [1.5, 3] }, raw))
      // The update targets the image series by id.
      expect(update.series[0].id).toBe(option.series[0].id)
      const before = option.series[0].renderItem({}, api).style.image
      expect(typeof before).toBe('string')
      const p = imageOf(update)
      const bg = parseColour(tokens(mode).bg)
      // Cells of value 0 (left) are dimmed towards the background; cells of value 3 (right) keep their colour.
      expect(rgbAt(p, 0, 0)).toEqual(dimColour([0, 0, 0], bg, 0))
      expect(rgbAt(p, p.width - 1, 0)).toEqual([255, 255, 255])
      // Back to the whole range: the colours return.
      const all = imageOf(colourSelection(option, { visualMapId: map.id, selected: [0, 3] }, raw))
      expect(rgbAt(all, 0, 0)).toEqual([0, 0, 0])
    })

  it('dims a smooth heatmap with soft edges', () => {
    const option = opt(heatmapChart(z, { ramp: ['#000000', '#ffffff'] }))
    const p = imageOf(colourSelection(option, { visualMapId: option.visualMap[0].id, selected: [1.5, 3] }, raw))
    const row = Array.from({ length: p.width }, (_, x) => rgbAt(p, x, 0)[0] ?? 0)
    // Dimmed on the left, full on the right, and some pixels between the two at the edge.
    expect(row[0]).toBe(dimColour([0, 0, 0], parseColour(tokens('light').bg), 0)[0])
    expect(row[row.length - 1]).toBe(255)
  })

  it('answers piece clicks on a contour heatmap', () => {
    const option = opt(heatmapChart(z, { smooth: 'contours', levels: [1, 2] }))
    const map = option.visualMap[0]
    expect(map.type).toBe('piecewise')
    expect(map.selectedMode).toBe('multiple')
    const update = colourSelection(option, { visualMapId: map.id, selected: { 0: false, 1: true, 2: true } }, raw)
    expect(update).toBeDefined()
  })

  it('dims matrix cells outside the selection', () => {
    const option = opt(
      cartesian([
        matrix(
          [
            [0, 1],
            [2, 3],
          ],
          { ramp: ['#000000', '#ffffff'] },
        ),
      ]),
    )
    const update = opt(colourSelection(option, { visualMapId: option.visualMap[0].id, selected: [1.5, 3] }))
    const render = update.series[0].renderItem
    const cell = (i: number): Opt => render({ dataIndex: i }, api).children
    expect(cell(0)[0].style.fill).not.toBe('#000000')
    expect(cell(0)[1].style.opacity).toBe(DIMMED_OPACITY)
    expect(cell(3)[0].style.fill).toBe('#ffffff')
  })

  it('gives a decorative colour bar no handles and ignores selections on it', () => {
    const bar = layer([{ type: 'scatter', data: [] }], {}, { colourBar: { min: 0, max: 1, colours: ['#000', '#fff'] } })
    const option = opt(cartesian([bar]))
    expect(option.visualMap[0].calculable).toBe(false)
    expect(colourSelection(option, { visualMapId: option.visualMap[0].id, selected: [0.2, 0.4] })).toBeUndefined()
    const pieces = layer([], {}, { colourBar: { min: 0, max: 1, colours: ['#000', '#fff'], levels: [0, 0.5, 1] } })
    expect(opt(cartesian([pieces])).visualMap[0].selectedMode).toBe(false)
  })

  it('keeps series ids unique across layers', () => {
    const option = opt(cartesian([heatmap(z), heatmap(z, { colourBar: false }), heatmap(z)]))
    const ids = option.series.map((s: Opt) => s.id).filter((id: unknown) => id !== undefined)
    expect(new Set(ids).size).toBe(2)
  })
})
