import { describe, expect, it } from 'vitest'
import { resample } from './_resample.ts'
import { confusionMatrixChart, correlationChart, heatmap, heatmapChart, image, matrix, rampStops } from './matrices.ts'

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any
const opt = (o: unknown): Opt => o as Opt
const api = {
  coord: (p: readonly number[]) => [...p],
  size: (p: readonly number[]) => p.map((v) => v * 30),
  value: (d: number) => d,
}

describe('resample', () => {
  const g = { rows: 2, cols: 2, values: Float64Array.from([0, 1, 2, 3]) }
  it('keeps the corner nodes and interpolates between them', () => {
    for (const m of ['nearest', 'bilinear', 'bicubic'] as const) {
      const out = resample(g, 3, 3, m)
      expect([out[0], out[2], out[6], out[8]]).toEqual([0, 1, 2, 3])
    }
    expect(resample(g, 3, 3, 'bilinear')[4]).toBe(1.5)
  })
  it('never overshoots the input range bicubically', () => {
    const step = { rows: 1, cols: 4, values: Float64Array.from([0, 0, 1, 1]) }
    const out = resample({ ...step, rows: 2, values: Float64Array.from([0, 0, 1, 1, 0, 0, 1, 1]) }, 2, 40, 'bicubic')
    expect(Math.min(...out)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...out)).toBeLessThanOrEqual(1)
  })
})

describe('heatmap', () => {
  it('draws one smooth image with a colour bar', () => {
    const l = heatmap(
      [
        [0, 1, 2],
        [1, 2, 3],
      ],
      { x: [-1, 1], y: [0, 1], label: 'p' },
    )
    expect(l.series).toHaveLength(2)
    const s = opt(l.series[0])
    const el = s.renderItem({}, api)
    expect(el.style.image.startsWith('data:image/png;base64,')).toBe(true)
    expect(el.style).toMatchObject({ x: -1, y: 1, width: 2, height: -1 })
    expect(l.colourBar).toMatchObject({ min: 0, max: 3, target: 1, label: 'p' })
  })
  it('makes blocky cells, filled contours and diverging ramps on request', () => {
    expect(heatmap([[0, 1]], { smooth: 'bilinear' }).extent).toEqual({ x: [-0.5, 1.5], y: [-0.5, 0.5], tight: true })
    const c = heatmap(
      [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, Number.NaN],
      ],
      { smooth: 'contours', levels: [1, 2], centre: 1, colourBar: true },
    )
    expect(c.colourBar?.levels).toEqual([0, 1, 2, 3])
    expect(c.colourBar?.colours).toHaveLength(3)
    expect(heatmap([[1, 2]], { colourBar: false }).colourBar).toBeUndefined()
    expect(() => heatmap([[]])).toThrow()
    expect(() => heatmap([1, 2])).toThrow()
    expect(
      opt(
        heatmapChart(
          [
            [0, 1],
            [1, 0],
          ],
          { smooth: 'bilinear' },
          { mode: 'dark' },
        ),
      ).visualMap,
    ).toHaveLength(1)
  })
  it('places images from pixels', () => {
    expect(image([[[1, 0, 0]]], { x: [0, 2], y: [0, 1] }).extent.x).toEqual([0, 2])
  })
})

describe('matrix', () => {
  it('colours and annotates cells, row 0 at the top', () => {
    const l = matrix(
      [
        [0, 1],
        [2, 3],
      ],
      { colLabels: ['x', 'y'] },
    )
    const s = opt(l.series[0])
    expect(s.data[0]).toEqual([0, 1, 0])
    const el = s.renderItem({ dataIndex: 3 }, api)
    expect(el.children[1].style.text).toBe('3')
    expect(el.children[1].style.fill).toBe('#1f2328')
    expect(s.renderItem({ dataIndex: 9 }, api).children).toEqual([])
    expect(() => matrix([[1]], { rowLabels: ['a', 'b'] })).toThrow()
    expect(rampStops('magma', 'dark')[0]).toBe('#000004')
  })
  it('builds confusion and correlation charts', () => {
    const o = opt(
      confusionMatrixChart(
        [
          [3, 1],
          [0, 0],
        ],
        ['a', 'b'],
        { normalise: true },
      ),
    )
    expect(o.series[0].data.map((d: number[]) => d[2])).toEqual([0.75, 0.25, 0, 0])
    expect(o.yAxis[0].data).toEqual(['b', 'a'])
    expect(
      opt(
        correlationChart(
          [
            [1, -1],
            [-1, 1],
          ],
          ['u', 'v'],
        ),
      ).visualMap[0].max,
    ).toBe(1)
  })
})
