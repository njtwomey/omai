import { describe, expect, it } from 'vitest'
import { resample } from './_resample.ts'
import {
  confusionMatrixChart,
  correlationChart,
  heatmap,
  heatmapChart,
  image,
  matrix,
  rampStops,
  reencodeImages,
} from './matrices.ts'

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

// The 2-D kernel the separable bicubic replaced: four rows of four taps per output pixel.
function bicubic2d(g: { rows: number; cols: number; values: Float64Array }, rows: number, cols: number): Float64Array {
  const cubic = (p0: number, p1: number, p2: number, p3: number, t: number): number =>
    p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))
  const at = (r: number, c: number): number =>
    g.values[Math.min(g.rows - 1, Math.max(0, r)) * g.cols + Math.min(g.cols - 1, Math.max(0, c))] ?? Number.NaN
  const finite = Array.from(g.values).filter(Number.isFinite)
  const lo = Math.min(...finite)
  const hi = Math.max(...finite)
  const out = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const u = rows === 1 ? 0 : (i * (g.rows - 1)) / (rows - 1)
    for (let j = 0; j < cols; j++) {
      const w = cols === 1 ? 0 : (j * (g.cols - 1)) / (cols - 1)
      const [r0, c0] = [Math.floor(u), Math.floor(w)]
      const row = (r: number): number => cubic(at(r, c0 - 1), at(r, c0), at(r, c0 + 1), at(r, c0 + 2), w - c0)
      const v = cubic(row(r0 - 1), row(r0), row(r0 + 1), row(r0 + 2), u - r0)
      out[i * cols + j] = Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : v
    }
  }
  return out
}

describe('separable bicubic', () => {
  const surface = (rows: number, cols: number, f: (i: number, j: number) => number) => ({
    rows,
    cols,
    values: Float64Array.from({ length: rows * cols }, (_, k) => f(Math.floor(k / cols), k % cols)),
  })
  it('equals the 2-D kernel on smooth, wide, one-row and NaN-holding grids, up and down in size', () => {
    const cases: readonly [string, ReturnType<typeof surface>, number, number][] = [
      ['a smooth square surface up to 3x', surface(24, 24, (i, j) => Math.sin(i / 5) * Math.cos(j / 4)), 70, 70],
      ['a wide grid, down-sampled in one axis', surface(9, 31, (i, j) => (i * 7 + j * 3) % 5), 5, 63],
      ['a single row', surface(1, 6, (_, j) => j * j), 4, 20],
      [
        'a grid with a NaN node (it spreads to the pixels it touches)',
        surface(6, 6, (i, j) => (i === 2 && j === 3 ? Number.NaN : i - j)),
        17,
        19,
      ],
    ]
    for (const [name, g, rows, cols] of cases) {
      const got = resample(g, rows, cols, 'bicubic')
      const want = bicubic2d(g, rows, cols)
      for (let k = 0; k < want.length; k++)
        expect(Math.abs((got[k] as number) - (want[k] as number)) < 1e-12 || Object.is(got[k], want[k]), name).toBe(
          true,
        )
    }
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
  it('takes Pixels as they are: RGBA bytes with their alpha, no shape or range to give', () => {
    const rgba = new Uint8ClampedArray([200, 100, 50, 128, 0, 0, 0, 0, 255, 255, 255, 255, 10, 20, 30, 40])
    const pixels = { width: 2, height: 2, rgba }
    const l = image(pixels, { x: [0, 1000], y: [0, 500], opacity: 0.5 })
    expect(l.extent).toEqual({ x: [0, 1000], y: [0, 500], tight: true })
    const drawn = opt(l.series[0]).omaiImage
    expect(drawn.pixels).toBe(pixels)
    expect(Array.from(drawn.pixels.rgba)).toEqual(Array.from(rgba))
    expect(drawn.opacity).toBe(0.5)
    expect(image(pixels).extent).toEqual({ x: [0, 2], y: [0, 2], tight: true })
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
    expect(el.children[1].style.fill).toBe('#0a0a0a')
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

describe('reencodeImages', () => {
  it('draws image series through the encoder, with the same geometry, and leaves the other series alone', () => {
    const layer = heatmap(
      [
        [0, 1],
        [1, 0],
      ],
      { x: [0, 1], y: [0, 1], colourBar: false },
    )
    const plain = { type: 'line', data: [] } as never
    const [first] = layer.series
    const seen: unknown[] = []
    const out = reencodeImages([first as never, plain], (pixels) => {
      seen.push(pixels)
      return 'the-encoded-image'
    })
    expect(out).toHaveLength(2)
    expect(out[1]).toBe(plain)
    // Lazy: nothing is encoded until the item is drawn, and then once however often it is drawn.
    expect(seen).toHaveLength(0)
    const item = opt(out[0]).renderItem({}, api)
    opt(out[0]).renderItem({}, api)
    expect(seen).toHaveLength(1)
    expect(item.style.image).toBe('the-encoded-image')
    // The same rectangle as the data-URL version.
    const original = opt(first).renderItem({}, api)
    expect(item.style.width).toBe(original.style.width)
    expect(item.style.height).toBe(original.style.height)
    expect(original.style.image).toMatch(/^data:image\/png;base64,/)
  })

  it('survives the chart builders: the image series of a finished chart can still be re-encoded', () => {
    const option = opt(
      heatmapChart(
        [
          [0, 1],
          [1, 0],
        ],
        { smooth: 'cells' },
      ),
    )
    const series = option.series as unknown[]
    const out = reencodeImages(series as never, () => 'canvas')
    expect(
      out.filter((s) => opt(s).type === 'custom' && opt(s).renderItem({}, api).style?.image === 'canvas'),
    ).toHaveLength(1)
  })
})
