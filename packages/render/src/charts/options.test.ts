import * as echarts from 'echarts'
import { tensor } from 'omai-engine/tensor'
import { describe, expect, it } from 'vitest'
import { figure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import { annotations } from './annotations.ts'
import { bars } from './bars.ts'
import { subplotRects, subplots } from './compose.ts'
import { fieldArrows, vectorField } from './fields.ts'
import { referenceLines } from './functions.ts'
import { lines } from './lines.ts'
import { maskMatrix, matrix } from './matrices.ts'
import { cartesian, colourBarWidth, layer, panelParts, plotRectOf } from './plot.ts'
import { points } from './points.ts'

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any
const opt = (o: unknown): Opt => o as Opt

function svgOf(option: echarts.EChartsCoreOption, width = 640, height = 400): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width, height })
  chart.setOption(option)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

// The font sizes of the text elements of an SVG whose text is `text`.
function sizesOf(svg: string, text: string): number[] {
  return [...svg.matchAll(/<text[^>]*font-size:(\d+(?:\.\d+)?)px[^>]*>([^<]*)</g)]
    .filter((m) => m[2] === text)
    .map((m) => Number(m[1]))
}

const unit = [layer([{ type: 'scatter', data: [] }], { x: [0, 1], y: [0, 1], tight: true })]

describe('type size follows the figure', () => {
  it("sets the axis type from figure.fontSize and keeps today's sizes at the default", () => {
    const at = (fontSize?: number): Opt =>
      opt(cartesian(unit, { x: { label: 'x' }, figure: figure(fontSize === undefined ? {} : { fontSize }) }))
    expect(at().xAxis[0].axisLabel.fontSize).toBe(10)
    expect(at().xAxis[0].nameTextStyle.fontSize).toBe(11)
    expect(at(20).xAxis[0].axisLabel.fontSize).toBe(19)
    expect(at(20).yAxis[0].nameTextStyle.fontSize).toBe(20)
    expect(at(20).xAxis[0].nameGap).toBeGreaterThan(at().xAxis[0].nameGap)
  })

  it('applies it in subplots too', () => {
    const o = opt(subplots(gridSpec(1, 1), { r0c0: { layers: unit } }, { figure: figure({ fontSize: 16 }) }))
    expect(o.xAxis[0].axisLabel.fontSize).toBe(15)
  })

  it('draws reference line, annotation, point and bar labels at the figure size, unless a layer says otherwise', () => {
    const lay = [
      points([{ x: [0, 1], y: [0, 1], labels: ['pointlabel', 'two'] }]),
      referenceLines({ x: [{ value: 0.5, label: 'refline' }] }),
      annotations({ labels: [{ at: [0.2, 0.2], text: 'note' }] }),
    ]
    const svg = svgOf(cartesian(lay, { figure: figure({ fontSize: 20 }) }))
    expect(sizesOf(svg, 'pointlabel')).toEqual([20])
    expect(sizesOf(svg, 'refline')).toEqual([20])
    expect(sizesOf(svg, 'note')).toEqual([20])
    const own = svgOf(
      cartesian(
        [
          points([{ x: [0, 1], y: [0, 1], labels: ['pointlabel', 'two'], labelSize: 9 }]),
          referenceLines({ x: [{ value: 0.5, label: 'refline' }] }, { fontSize: 14 }),
          annotations({ labels: [{ at: [0.2, 0.2], text: 'note' }] }, { fontSize: 13 }),
        ],
        { figure: figure({ fontSize: 20 }) },
      ),
    )
    expect(sizesOf(own, 'pointlabel')).toEqual([9])
    expect(sizesOf(own, 'refline')).toEqual([14])
    expect(sizesOf(own, 'note')).toEqual([13])
    const barSvg = svgOf(
      cartesian([bars(['a', 'b'], [7.25, 5], { labels: true })], { figure: figure({ fontSize: 18 }) }),
    )
    expect(sizesOf(barSvg, '7.25')).toEqual([18])
    expect(opt(bars(['a'], [1], { labels: true, labelSize: 8 }).series[0]).label.fontSize).toBe(8)
  })

  it('spaces the lines of a colour bar label by more than the type size, so a second line does not touch the value', () => {
    const make = (size: number): string => {
      const l = layer(
        [{ type: 'scatter', data: [[0.5, 0.5]] }],
        { x: [0, 1], y: [0, 1] },
        { colourBar: { min: 0, max: 6.79, colours: ['#fff', '#00f'], label: 'height\n(metres)' } },
      )
      return svgOf(cartesian([l], { figure: figure({ width: 700, height: 460, fontSize: size }) }), 700, 460)
    }
    for (const size of [11, 22]) {
      const svg = make(size)
      const ys = ['height', '(metres)', '6.79'].map((t) => {
        const m = svg.match(
          new RegExp(`<text[^>]* y="(-?[\\d.]+)"[^>]*>${t.replace('(', '\\(').replace(')', '\\)')}</text>`),
        )
        return Number(m?.[1])
      })
      const [a = 0, b = 0, c = 0] = ys
      expect(b - a).toBeGreaterThanOrEqual((size - 1) * 1.25)
      expect(c - b).toBeGreaterThanOrEqual((size - 1) * 1.25)
    }
  })

  it('sets the font family and weight of tick labels, and the weight of point labels', () => {
    const l = points([{ x: [0.5], y: [0.5], labels: ['a'], labelWeight: 'bold', labelFamily: 'monospace' }], {})
    const o = opt(
      cartesian([l], { x: { ticks: { fontFamily: 'ui-monospace', fontWeight: 'bold' } }, figure: figure() }),
    ) as { xAxis: { axisLabel: Record<string, unknown> }[]; series: { label: Record<string, unknown> }[] }
    expect(o.xAxis[0]?.axisLabel['fontFamily']).toBe('ui-monospace')
    expect(o.xAxis[0]?.axisLabel['fontWeight']).toBe('bold')
    expect(o.series[0]?.label['fontWeight']).toBe('bold')
    expect(o.series[0]?.label['fontFamily']).toBe('monospace')
  })

  it('sizes the colour bar text from the figure and reserves room from the label', () => {
    const bar = { min: 0, max: 1, colours: ['#000', '#fff'] }
    const make = (label?: string, extra: object = {}): ReturnType<typeof layer> =>
      layer(
        [{ type: 'scatter', data: [] }],
        { x: [0, 1], y: [0, 1] },
        { colourBar: { ...bar, ...(label ? { label } : {}), ...extra } },
      )
    expect(colourBarWidth([make()], 11)).toBe(64)
    expect(colourBarWidth([make('p')], 11)).toBe(64)
    expect(colourBarWidth([make('a long colour bar label')], 11)).toBeGreaterThan(100)
    expect(colourBarWidth([make('a long colour bar label')], 20)).toBeGreaterThan(
      colourBarWidth([make('a long colour bar label')], 11),
    )
    expect(colourBarWidth([make('a long colour bar label', { width: 70 })], 11)).toBe(70)
    expect(colourBarWidth([], 11)).toBe(0)
    const narrow = opt(cartesian([make()], {}))
    const wide = opt(cartesian([make('a long colour bar label')], {}))
    expect(wide.grid[0].width).toBeLessThan(narrow.grid[0].width)
    expect(narrow.visualMap[0].textStyle.fontSize).toBe(10)
    expect(opt(cartesian([make()], { figure: figure({ fontSize: 16 }) })).visualMap[0].textStyle.fontSize).toBe(15)
    expect(opt(cartesian([make('', { fontSize: 7 })])).visualMap[0].textStyle.fontSize).toBe(7)
    // The subplot grid leaves the same room.
    const rects = subplotRects(gridSpec(1, 1), { r0c0: { layers: [make('a long colour bar label')] } }, figure())
    expect(rects['r0c0']?.width).toBeCloseTo(wide.grid[0].width, 6)
  })
})

describe('adjustable', () => {
  const area = { x: 50, y: 20, width: 400, height: 200 }
  const square = [layer([], { x: [0, 1], y: [0, 1], tight: true })]
  it('widens the ranges by default (datalim) and shrinks the box under box', () => {
    const lim = panelParts(square, { aspect: 'equal' }, area, 0, 0, 0)
    expect(lim.ranges.x).toEqual([-0.5, 1.5])
    expect(lim.rect).toEqual(area)
    const box = panelParts(square, { aspect: 'equal', adjustable: 'box' }, area, 0, 0, 0)
    expect(box.ranges).toEqual({ x: [0, 1], y: [0, 1] })
    expect(box.rect).toEqual({ x: 150, y: 20, width: 200, height: 200 })
    expect(box.grid).toMatchObject({ left: 150, top: 20, width: 200, height: 200 })
  })

  it('leaves the box alone for auto aspect and places the colour bar by the shrunken box', () => {
    const same = panelParts(square, { adjustable: 'box' }, area, 0, 0, 0)
    expect(same.rect).toEqual(area)
    const withBar = [
      layer(
        [{ type: 'scatter', data: [] }],
        { x: [0, 1], y: [0, 1], tight: true },
        {
          colourBar: { min: 0, max: 1, colours: ['#000', '#fff'] },
        },
      ),
    ]
    const box = panelParts(withBar, { aspect: 'equal', adjustable: 'box' }, area, 0, 0, 0)
    expect(box.visualMap[0]).toMatchObject({ left: 150 + 200 + 12, top: 20 })
  })

  it('works through cartesian and plotRectOf, and in subplots', () => {
    const fig = figure({ width: 400, aspect: 2 })
    const o = opt(cartesian(square, { figure: fig, aspect: 'equal', adjustable: 'box' }))
    expect(o.grid[0].width).toBeCloseTo(o.grid[0].height, 9)
    expect(o.xAxis[0].min).toBe(0)
    const where = plotRectOf(square, { figure: fig, aspect: 'equal', adjustable: 'box' })
    expect(where.rect).toEqual({
      x: o.grid[0].left,
      y: o.grid[0].top,
      width: o.grid[0].width,
      height: o.grid[0].height,
    })
    expect(where.ranges).toEqual({ x: [0, 1], y: [0, 1] })
    const s = opt(
      subplots(
        gridSpec(1, 1),
        { r0c0: { layers: square, options: { aspect: 'equal', adjustable: 'box' } } },
        { figure: fig },
      ),
    )
    expect(s.grid[0].width).toBeCloseTo(s.grid[0].height, 9)
    expect(s.xAxis[0].min).toBe(0)
  })
})

describe('plotRectOf', () => {
  it('is the grid cartesian draws, with room for a colour bar and a wrapped legend', () => {
    const fig = figure()
    const bar = layer(
      [{ type: 'scatter', data: [] }],
      { x: [0, 1], y: [0, 1] },
      {
        colourBar: { min: 0, max: 1, colours: ['#000', '#fff'] },
      },
    )
    const o = opt(cartesian([bar], { figure: fig }))
    expect(plotRectOf([bar], { figure: fig }).rect).toEqual({
      x: o.grid[0].left,
      y: o.grid[0].top,
      width: o.grid[0].width,
      height: o.grid[0].height,
    })
    const names = ['gradientDescent', 'conjugateGradient', 'newton', 'bfgs', 'lbfgs', 'coordinateDescent', 'gradient']
    const many = lines(names.map((name) => ({ y: [1, 2], name })))
    const small = figure({ width: 440, height: 320 })
    expect(plotRectOf([many], { figure: small }).rect.y).toBe(opt(cartesian([many], { figure: small })).grid[0].top)
    expect(plotRectOf([many], { figure: small }).rect.y).toBeGreaterThan(28)
  })
})

describe('axis ticks', () => {
  it('merges interval, rotation and size into the axis label', () => {
    const cats = bars(['alpha', 'beta', 'gamma', 'delta'], [1, 2, 3, 4])
    const o = opt(cartesian([cats], { x: { ticks: { interval: 1, rotate: 45, fontSize: 9 } } }))
    expect(o.xAxis[0].axisLabel).toMatchObject({ interval: 1, rotate: 45, fontSize: 9 })
    const auto = opt(cartesian([cats], { x: { ticks: { interval: 'auto' } } }))
    expect(auto.xAxis[0].axisLabel.interval).toBeUndefined()
    const value = opt(cartesian(unit, { y: { ticks: { interval: 0.25, rotate: 30 } } }))
    expect(value.yAxis[0].interval).toBe(0.25)
    expect(value.yAxis[0].axisLabel.rotate).toBe(30)
    expect(opt(cartesian([cats], {})).xAxis[0].axisLabel.rotate).toBeUndefined()
  })

  it('shows in the rendered SVG (a rotated label carries a rotate transform)', () => {
    const svg = svgOf(cartesian([bars(['alpha', 'beta'], [1, 2])], { x: { ticks: { rotate: 40, fontSize: 15 } } }))
    expect(sizesOf(svg, 'alpha')).toEqual([15])
    expect(svg).toMatch(/transform="matrix\([^"]*"[^>]*>alpha/)
  })
})

describe('point labels', () => {
  const base = { x: [0, 1], y: [0, 1], labels: ['one', 'two'] }
  it('takes size, position and colour', () => {
    const s = opt(points([{ ...base, labelSize: 13, labelPosition: 'top', labelColour: '#ff0000' }]).series[0])
    expect(s.label).toMatchObject({ fontSize: 13, color: '#ff0000' })
    expect(s.data[0].label.position).toBe('top')
    expect(opt(points([base]).series[0]).data[0].label.position).toBe('right')
  })

  it('keeps the labels when the symbol is none: a circle of size zero', () => {
    const s = opt(points([{ ...base, symbol: 'none', size: 9, sizes: [1, 2] }]).series[0])
    expect(s.symbol).toBe('circle')
    expect(s.symbolSize).toBe(0)
    expect(s.data[0].symbolSize).toBeUndefined()
    const svg = svgOf(cartesian([points([{ ...base, symbol: 'none', labelPosition: 'inside' }])]))
    expect(sizesOf(svg, 'one')).toHaveLength(1)
    expect(sizesOf(svg, 'two')).toHaveLength(1)
  })
})

describe('regions', () => {
  it('shades rectangles and bands as mark areas', () => {
    const l = annotations({
      spans: [{ x: [0, 1] }],
      regions: [
        { x: [1, 2], y: [3, 4], label: 'box', colour: '#ff0000' },
        { y: [5, 6] },
        { x: [7, 8], label: 'band' },
        {},
      ],
    })
    expect(l.series).toHaveLength(3)
    const data = opt(l.series[2]).markArea.data
    expect(data).toHaveLength(4)
    expect(data[1][0]).toMatchObject({ xAxis: 1, yAxis: 3, name: 'box', itemStyle: { color: '#ff0000' } })
    expect(data[1][1]).toEqual({ xAxis: 2, yAxis: 4 })
    expect(data[2][0].xAxis).toBeUndefined()
    expect(data[2][0].yAxis).toBe(5)
    expect(data[3][0]).toMatchObject({ xAxis: 7, name: 'band' })
    expect(l.extent).toEqual({ x: [0, 8], y: [3, 6] })
  })

  it('renders as filled areas in the SVG', () => {
    const svg = svgOf(
      cartesian([annotations({ regions: [{ x: [0.2, 0.6], y: [0.2, 0.6], label: 'zone', colour: '#ff0000' }] })], {
        x: { limits: [0, 1] },
        y: { limits: [0, 1] },
      }),
    )
    expect(svg).toContain('zone')
    expect(svg).toMatch(/<polygon[^>]*fill="#ff0000"/)
  })
})

describe('vector field colouring', () => {
  const arrows = fieldArrows((x, y) => [x, y], { x: [-1, 1], y: [-1, 1], n: [3, 3], length: 'magnitude' })
  it('takes a fixed range and a ramp, as heatmap does', () => {
    const own = opt(vectorField(arrows).series[0]).data.map((d: Opt) => d.lineStyle.color)
    const fixed = opt(vectorField(arrows, { range: [0, 100] }).series[0]).data.map((d: Opt) => d.lineStyle.color)
    const mags = arrows.map((a) => a.magnitude)
    const top = mags.indexOf(Math.max(...mags))
    expect(own[top]).toBe('#fde725')
    expect(fixed[top]).not.toBe('#fde725')
    const hot = opt(vectorField(arrows, { range: [0, 2], ramp: ['#000000', '#ffffff'] }).series[0]).data
    const brightest = hot[mags.indexOf(Math.max(...mags))].lineStyle.color
    expect(brightest).not.toBe('#ffffff')
    expect(opt(vectorField(arrows, { range: [0, 2], ramp: 'blues' }).series[0]).data[0].lineStyle.color).toMatch(/^#/)
  })
})

describe('matrix label colours and marks', () => {
  const z = [
    [1, 0],
    [0, 1],
  ]
  it('colours row and column labels in the axis order', () => {
    const l = matrix(z, {
      rowLabels: ['a', 'b'],
      rowColours: ['#ff0000', undefined],
      colColours: [undefined, '#00ff00'],
    })
    expect(l.yLabelColours).toEqual(['#ff0000', undefined].reverse())
    expect(l.xLabelColours).toEqual([undefined, '#00ff00'])
    const o = opt(cartesian([l], {}))
    expect(o.yAxis[0].axisLabel.color('b', 1)).toBe('#ff0000')
    expect(o.yAxis[0].axisLabel.color('a', 0)).not.toBe('#ff0000')
    expect(o.xAxis[0].axisLabel.color('1', 1)).toBe('#00ff00')
    expect(typeof opt(cartesian([matrix(z)], {})).xAxis[0].axisLabel.color).toBe('string')
  })

  it('writes marks in cells, over the value or alone', () => {
    const api = { coord: (p: readonly number[]) => [...p], size: () => [30, 30], value: (d: number) => d }
    const text = (l: ReturnType<typeof matrix>, i: number): string =>
      opt(l.series[0]).renderItem({ dataIndex: i }, { ...api, value: (d: number) => [0, 0][d] ?? 0 }).children[1]?.style
        .text
    const marked = matrix(z, { annotate: false, colourBar: false, marks: (v) => (v > 0 ? '✓' : undefined) })
    expect(text(marked, 0)).toBe('✓')
    expect(opt(marked.series[0]).renderItem({ dataIndex: 1 }, api).children).toHaveLength(1)
    const both = matrix(z, { marks: (v) => (v > 0 ? '✓' : undefined) })
    expect(text(both, 0)).toBe('✓')
    expect(text(both, 1)).toBe('0')
  })
})

describe('maskMatrix', () => {
  it('colours the cells that are on by a function of the row and column, one stop per colour', () => {
    const mask = [
      [true, false, true],
      [true, true, true],
    ]
    const l = maskMatrix(mask, { on: (row, col) => (col === 0 ? 'tab0' : row === 0 ? 'tab1' : 'tab2') })
    const data = (l.series[0] as { data: number[][] }).data
    // The first row is drawn at the top, so its y index is the largest.
    const code = (r: number, c: number): number | undefined => data.find((d) => d[0] === c && d[1] === 1 - r)?.[2]
    // Off is 0; the three colours are numbered by first use, scanning rows then columns.
    expect([code(0, 0), code(0, 1), code(0, 2), code(1, 0), code(1, 1), code(1, 2)]).toEqual([1, 0, 2, 1, 3, 3])
    expect(l.colourBar).toBeUndefined()
  })

  const mask = [
    [true, false, false],
    [true, true, false],
    [true, true, true],
  ]
  it('is a two-colour matrix without a bar, with labels, label colours and marks', () => {
    const l = maskMatrix(mask, {
      on: '#ff0000',
      off: '#eeeeee',
      rowLabels: ['a', 'b', 'c'],
      colLabels: ['x', 'y', 'z'],
      rowColours: ['#111111', undefined, undefined],
      marks: (on) => (on ? '✓' : undefined),
    })
    expect(l.colourBar).toBeUndefined()
    expect(l.yCategories).toEqual(['c', 'b', 'a'])
    expect(l.yLabelColours?.[2]).toBe('#111111')
    const cells = opt(l.series[0]).data
    expect(cells).toHaveLength(9)
    const api = { coord: (p: readonly number[]) => [...p], size: () => [30, 30], value: (d: number) => d }
    const drawn = (i: number): Opt => opt(l.series[0]).renderItem({ dataIndex: i }, api).children
    expect(drawn(0)[0].style.fill).toBe('#ff0000')
    expect(drawn(1)[0].style.fill).toBe('#eeeeee')
    expect(drawn(0)[1].style.text).toBe('✓')
    expect(drawn(1)).toHaveLength(1)
  })

  it('reads numbers and tensors as masks (nonzero is on)', () => {
    const l = maskMatrix([[0, 2]])
    expect(opt(l.series[0]).data.map((d: number[]) => d[2])).toEqual([0, 1])
    const t = maskMatrix(tensor([[3, 0]]))
    expect(opt(t.series[0]).data.map((d: number[]) => d[2])).toEqual([1, 0])
  })

  it('renders in SVG with its label colours', () => {
    const svg = svgOf(
      cartesian(
        [maskMatrix(mask, { rowLabels: ['aa', 'bb', 'cc'], rowColours: ['#ff0000', undefined, undefined] })],
        {},
      ),
    )
    expect(svg).toContain('aa')
    expect(svg).toMatch(/<text[^>]*fill="#ff0000"[^>]*>aa/)
  })
})
