import { describe, expect, it } from 'vitest'
import { gridSpec, withMarginals } from '../figure/grid.ts'
import { histogram } from './bars.ts'
import { subplots } from './compose.ts'
import { decimate, lttb, minMaxDecimate } from './downsample.ts'
import {
  calibrationChart,
  intervalChart,
  learningCurveChart,
  metricBarChart,
  precisionRecallChart,
  residualChart,
  rocChart,
} from './evaluation.ts'
import { argmaxMargins, contours, decisionRegions, fieldArrows, streamlines, vectorField } from './fields.ts'
import {
  fn,
  implicit,
  parametric,
  polar,
  referenceLines,
  riemann,
  secantLine,
  tangentLine,
  taylor,
} from './functions.ts'
import { lines } from './lines.ts'
import { points } from './points.ts'

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any
const opt = (o: unknown): Opt => o as Opt
const api = {
  coord: (p: readonly (number | string)[]) => p.map((v) => (typeof v === 'number' ? v : 0)),
  value: (d: number) => d,
}

const bump = Array.from({ length: 21 }, (_, i) =>
  Array.from({ length: 21 }, (_, j) => Math.exp(-((i - 10) ** 2 + (j - 10) ** 2) / 20)),
)

describe('fields', () => {
  it('draws labelled contour lines, coloured along a ramp on request', () => {
    const l = contours(bump, { x: [-1, 1], y: [-1, 1], levels: 3, colour: 'ramp' })
    expect(l.extent).toEqual({ x: [-1, 1], y: [-1, 1], tight: true })
    const labels = opt(l.series[l.series.length - 1])
    expect(labels.data.length).toBe(l.series.length - 1)
    expect(labels.label.formatter({ value: [0, 0, 0.25] })).toBe('0.25')
    expect(contours(bump, { levels: [2] }).series).toHaveLength(1)
  })
  it('draws decision regions with boundaries', () => {
    const p = bump.map((r) => r.map((v) => v))
    const q = bump.map((r) => r.map((v) => 1 - v))
    const l = decisionRegions([p, q], { resolution: 40 })
    expect(l.series.length).toBeGreaterThan(1)
    expect(decisionRegions([p, q, q], { boundaries: false, confidence: false }).series).toHaveLength(1)
    expect(() => decisionRegions([])).toThrow()
    expect(() => argmaxMargins([[[1]], [[1, 2]]])).toThrow()
    expect(argmaxMargins([])).toEqual([])
  })
  it('samples vector fields and draws arrows and streamlines', () => {
    const arrows = fieldArrows((x, y) => [x, y], { x: [-1, 1], y: [-1, 1], n: [3, 3], length: 'magnitude' })
    expect(arrows).toHaveLength(8)
    expect(() => fieldArrows(() => [1, 0], { x: [0, 1], y: [0, 1], n: 1 })).toThrow()
    const v = opt(vectorField(arrows, { colour: 'tab2' }).series[0])
    expect(v.data[0].lineStyle.color).toBe('#2ca02c')
    expect(vectorField([]).extent).toEqual({})
    expect(
      opt(
        streamlines([
          [
            [0, 0],
            [1, 1],
          ],
          [[2, 2]],
        ]).series[1],
      ).data,
    ).toHaveLength(1)
  })
})

describe('functions', () => {
  it('breaks curves at asymptotes', () => {
    const data = opt(fn(Math.tan, [-3, 3], { n: 200 }).series[0]).data
    expect(data.filter((d: number[]) => d[1] === null).length).toBeGreaterThanOrEqual(2)
    const c = opt(
      fn((x) => x, [0, 10], { n: 11, clip: [2, 8], name: 'f', colour: 'tab1', width: 3, dash: 'dotted' }).series[0],
    )
    expect(c.data.filter((d: number[]) => d[1] === null)).toHaveLength(4)
    expect(c.lineStyle).toMatchObject({ width: 3, type: 'dotted' })
  })
  it('draws parametric, polar and implicit curves', () => {
    expect(
      parametric(
        (t) => t,
        (t) => t * t,
        [0, 1],
        { n: 2 },
      ).extent,
    ).toEqual({ x: [0, 1], y: [0, 1] })
    expect(polar((t) => t).series).toHaveLength(1)
    const circle = implicit((x, y) => x * x + y * y - 1, [-2, 2], [-2, 2])
    expect(circle.extent.y?.[0]).toBeCloseTo(-1, 2)
  })
  it('draws tangents, secants, Riemann sums and Taylor polynomials', () => {
    const t = opt(tangentLine(Math.sin, 0).series[0])
    expect(t.data[1][1]).toBeCloseTo(1, 6)
    expect(() => secantLine(Math.sin, 1, 1)).toThrow()
    const r = riemann((x) => x * x, [0, 1], 4, { rule: 'right' })
    expect(r.sum).toBeCloseTo(0.46875)
    expect(riemann((x) => x, [0, 1], 1).sum).toBe(0.5)
    expect(opt(r.series[0]).renderItem({ dataIndex: 0 }, api).shape.points).toHaveLength(4)
    expect(() => riemann((x) => x, [0, 1], 0)).toThrow()
    expect(opt(taylor([0, 1, 0, -1], 0, [-1, 1], { n: 3 }).series[0]).data[2][1]).toBeCloseTo(1 - 1 / 6)
    expect(opt(referenceLines({ x: [{ value: 1, label: 'a' }] }).series[0]).markLine.data[0].label.show).toBe(true)
  })
})

describe('evaluation', () => {
  const roc = { fpr: [0, 0.5, 1], tpr: [0, 0.8, 1], thresholds: [Infinity, 0.5, 0] }
  it('draws ROC, PR and calibration charts on the unit square', () => {
    expect(opt(rocChart(roc)).xAxis[0]).toMatchObject({ min: 0, max: 1 })
    expect(
      opt(precisionRecallChart([{ name: 'm', curve: { precision: [1, 1], recall: [1, 0], thresholds: [0.5] } }]))
        .series,
    ).toHaveLength(1)
    expect(
      opt(calibrationChart([{ name: 'm', curve: { probTrue: [0.2], probPred: [0.3], counts: [5] } }])).series,
    ).toHaveLength(3)
  })
  it('draws learning curves, residuals, intervals and metric bars', () => {
    expect(opt(learningCurveChart([1, 2], { mean: [1, 1] }, { mean: [0, 0], std: [0.1, 0.1] })).series).toHaveLength(3)
    expect(opt(residualChart([1, 2], [0, 1])).series).toHaveLength(2)
    const o = opt(
      intervalChart([{ name: 'a', value: { metric: 'm', estimate: { value: 1 }, n: 3 } }], { reference: 0 }),
    )
    expect(o.series[0].markLine.data[0].xAxis).toBe(0)
    const item = o.series[1].renderItem(
      {},
      { coord: (p: (number | string)[]) => [Number(p[0]), 0], value: (d: number) => [0, 1, 0.5, 1.5][d] ?? 0 },
    )
    expect(item.children[0].shape).toMatchObject({ x1: 0.5, x2: 1.5 })
    expect(o.series[1].tooltip.formatter({ value: [0, 1, 0.5, 1.5] })).toBe('a: 1 [0.5, 1.5]')
    expect(
      opt(metricBarChart([{ name: 'acc', value: { metric: 'acc', estimate: { value: 0.5 }, n: 1 } }])).series,
    ).toHaveLength(2)
  })
})

describe('downsample', () => {
  it('keeps ends and extremes', () => {
    const pts = Array.from({ length: 1000 }, (_, i): [number, number] => [i, i === 500 ? 100 : Math.sin(i)])
    const kept = lttb(pts, 50)
    expect(kept).toHaveLength(50)
    expect(kept.some((p) => p[1] === 100)).toBe(true)
    expect(kept[0]).toEqual(pts[0])
    expect(kept[49]).toEqual(pts[999])
    expect(() => lttb(pts, 2)).toThrow()
    expect(lttb(pts.slice(0, 5), 10)).toHaveLength(5)
    const mm = minMaxDecimate(pts, 10)
    expect(mm.some((p) => p[1] === 100)).toBe(true)
    expect(minMaxDecimate(pts.slice(0, 4), 10)).toHaveLength(4)
    expect(decimate([1, 2, 3], 10, [5, 6, 7]).x).toEqual([5, 6, 7])
  })
})

describe('subplots', () => {
  it('shares ranges, hides inner labels and links hover in marginal layouts', () => {
    const x = [0, 1, 2, 3, 4]
    const y = [2, 1, 4, 3, 5]
    const o = opt(
      subplots(
        withMarginals(),
        {
          main: { layers: [points([{ x, y }])], options: { x: { label: 'x' }, title: 'joint' } },
          top: { layers: [histogram(x, { bins: 3 })] },
          right: { layers: [histogram(y, { bins: 3, horizontal: true })] },
        },
        { title: 'all', zoom: true },
      ),
    )
    expect(o.xAxis[0].min).toBe(o.xAxis[1].min)
    expect(o.yAxis[0].max).toBe(o.yAxis[2].max)
    expect(o.yAxis[2].axisLabel.show).toBe(false)
    expect(o.axisPointer.link).toEqual([{ xAxisIndex: [0, 1] }, { yAxisIndex: [0, 2] }])
    expect(o.title.map((t: Opt) => t.text)).toEqual(['all', 'joint'])
    expect(o.dataZoom[0].xAxisIndex).toEqual([0, 1])
    expect(o.series.map((s: Opt) => s.xAxisIndex)).toEqual([0, 1, 2])
  })
  it('holds text cells and rejects unknown panels', () => {
    const g = gridSpec(1, 2, {
      cells: [
        { id: 'a', row: 0, col: 0 },
        { id: 't', row: 0, col: 1, kind: 'text', content: 'hello' },
      ],
    })
    const o = opt(subplots(g, { a: { layers: [lines([{ y: [1, 2] }])] } }, { link: false }))
    expect(o.graphic[0].style.text).toBe('hello')
    expect(o.axisPointer).toBeUndefined()
    expect(() => subplots(g, { zz: { layers: [] } })).toThrow()
  })
})

describe('subplot details', () => {
  it('gives each panel its own legend and drops touching edge labels', () => {
    const g = gridSpec(2, 2, {
      hspace: 0,
      wspace: 0,
      cells: [
        { id: 'a', row: 0, col: 0 },
        { id: 'b', row: 1, col: 1 },
      ],
    })
    const two = [
      { y: [1, 2], name: 'u' },
      { y: [2, 1], name: 'v' },
    ]
    const o = opt(subplots(g, { a: { layers: [lines(two)] }, b: { layers: [lines(two)] } }))
    expect(o.legend).toHaveLength(2)
    expect(o.xAxis[0].axisLabel.showMaxLabel).toBe(false)
    expect(o.yAxis[1].axisLabel.showMaxLabel).toBe(false)
    expect(opt(subplots(g, { a: { layers: [lines([{ y: [1] }])] } })).legend.show).toBe(false)
  })
})
