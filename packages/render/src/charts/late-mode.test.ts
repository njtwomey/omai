import { bernoulli, normal } from 'omai-engine/dists'
import { dict, real } from 'omai-engine/space'
import { functionObjective, randomSampler, tune } from 'omai-engine/tune'
import { describe, expect, it } from 'vitest'
import { figure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { atMode } from './_mode.ts'
import { annotations } from './annotations.ts'
import { bars, histogram, pareto } from './bars.ts'
import { hexbin } from './binning.ts'
import { subplots } from './compose.ts'
import {
  boxPlot,
  density,
  ecdfCurve,
  kde,
  meanBand,
  pmf,
  priorPosterior,
  qqPlot,
  ridgeline,
  violinPlot,
} from './distributions.ts'
import { contours, decisionRegions, fieldArrows, streamlines, vectorField } from './fields.ts'
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
import { band, lines, polygons } from './lines.ts'
import { heatmap, maskMatrix, matrix } from './matrices.ts'
import { cartesian, type Layer, panelParts, plotRectOf } from './plot.ts'
import { errorBars, points, rug } from './points.ts'
import { stackedAreas } from './stacks.ts'
import { historyLayers } from './tuning.ts'

type Opts = { readonly mode?: ColourMode; readonly colour?: string }
type Builder = (o: Opts) => Layer | readonly Layer[]

const study = tune(
  functionObjective((p) => (p as { lr: number }).lr),
  dict({ lr: real(1e-4, 1, { log: true }) }),
  { sampler: randomSampler(), nTrials: 6, seed: 1 },
).study
const grid3 = [
  [0, 1, 0.5],
  [1, 2, 1],
  [0.5, 1, 0],
]
const xs = [0, 1, 2, 3, 4]
const poly = [
  [0, 0],
  [1, 0],
  [1, 1],
] as const

/** Every layer builder of `charts/`, with data that exercises its colours; `colour: 'fg'` where it takes a paint token. */
const builders: Record<string, Builder> = {
  lines: (o) => lines([{ x: xs, y: [1, 3, 2, 4, 3], colour: 'fg', area: true }, { y: [2, 1, 2, 1, 2] }], o),
  band: (o) =>
    band(
      [
        { x: xs, lower: [0, 1, 0, 1, 0], upper: [2, 3, 2, 3, 2], colour: 'accent', edges: true },
        { x: xs, lower: xs, upper: xs },
      ],
      o,
    ),
  polygons: (o) => polygons([{ points: poly, colour: 'fg' }, { points: poly }], o),
  points: (o) =>
    points(
      [
        { x: xs, y: [1, 3, 2, 4, 3], groups: [0, 1, 0, 1, 0] },
        { x: xs, y: xs, values: xs, colour: 'fg' },
      ],
      o,
    ),
  rug: (o) => rug(xs, o),
  errorBars: (o) =>
    errorBars(
      [
        { x: 1, y: 2, yErr: 0.5 },
        { x: 2, y: 3, yErr: 0.2 },
      ],
      o,
    ),
  stackedAreas: (o) =>
    stackedAreas(
      [0, 1],
      [
        { name: 'a', y: [1, 1] },
        { name: 'b', y: [2, 2] },
      ],
      o,
    ),
  bars: (o) =>
    bars(
      ['a', 'b'],
      [
        { name: 'n', values: [3, 5] },
        { name: 'm', values: [1, 2] },
      ],
      o,
    ),
  histogram: (o) => histogram([0, 1, 1, 2, 3], { bins: 3, ...o }),
  pareto: (o) => pareto(['a', 'b'], [1, 3], o),
  hexbin: (o) => hexbin([0, 0.01, 5], [0, 0.01, 5], { radius: 1, ...o }),
  annotations: (o) =>
    annotations(
      {
        arrows: [{ from: [0, 0], to: [1, 1], label: 'a' }],
        spans: [{ x: [2, 3] }],
        regions: [{ x: [0, 1], y: [2, 3], label: 'z' }],
      },
      o,
    ),
  density: (o) => density(normal(0, 1), { range: [-1, 1], n: 5, ...o }),
  kde: (o) => kde([0, 1, 2, 2.5], { n: 20, ...o }),
  ecdfCurve: (o) => ecdfCurve([1, 3, 3, 4], o),
  pmf: (o) => pmf(bernoulli({ probs: 0.25 }), { range: [0, 1], ...o }),
  boxPlot: (o) =>
    boxPlot(
      [
        { name: 'a', values: [1, 2, 3, 4, 100] },
        { name: 'b', values: [2, 3, 4] },
      ],
      o,
    ),
  violinPlot: (o) => violinPlot([{ name: 'a', values: [0, 1, 1, 2] }], o),
  qqPlot: (o) => qqPlot([3, 1, 2, 5], o),
  priorPosterior: (o) => priorPosterior(normal(0, 2), normal(1, 0.5), o),
  meanBand: (o) => meanBand([0, 1], [0, 1], [-1, 0], [1, 2], o),
  ridgeline: (o) =>
    ridgeline(
      [
        { name: 'a', values: [0, 1, 2] },
        { name: 'b', values: [1, 2, 3] },
      ],
      { n: 5, ...o },
    ),
  contours: (o) => contours(grid3, { levels: [0.5, 1.5], ...o }),
  decisionRegions: (o) =>
    decisionRegions(
      [
        [
          [0.9, 0.1],
          [0.8, 0.3],
        ],
        [
          [0.1, 0.9],
          [0.2, 0.7],
        ],
      ],
      { x: [-1, 1], y: [-1, 1], ...o },
    ),
  vectorField: (o) =>
    vectorField(
      fieldArrows((x, y) => [-y, x], { x: [-1, 1], y: [-1, 1], n: 3 }),
      o,
    ),
  streamlines: (o) =>
    streamlines(
      [
        [
          [0, 0],
          [1, 0],
          [2, 1],
        ],
      ],
      o,
    ),
  fn: (o) => fn((x) => x * x, [-1, 1], { n: 9, colour: 'fg', ...o }),
  parametric: (o) => parametric(Math.cos, Math.sin, [0, 2 * Math.PI], { n: 9, colour: 'fg', ...o }),
  polar: (o) => polar(() => 2, [0, Math.PI], { n: 5, colour: 'fg', ...o }),
  implicit: (o) => implicit((x, y) => x * x + y * y - 1, [-2, 2], [-2, 2], { n: 11, colour: 'fg', ...o }),
  tangentLine: (o) => tangentLine((x) => x * x, 1, { derivative: 2, span: 2, colour: 'fg', ...o }),
  secantLine: (o) => secantLine((x) => x * x, 0, 2, { colour: 'fg', ...o }),
  riemann: (o) => riemann((x) => x, [0, 1], 4, { colour: 'fg', ...o }),
  taylor: (o) => taylor([1, 1, 1], 0, [0, 1], { n: 5, colour: 'fg', ...o }),
  referenceLines: (o) => referenceLines({ x: [0], y: [{ value: 1, label: 'target' }] }, { colour: 'fg', ...o }),
  heatmap: (o) => heatmap(grid3, { x: [0, 2], y: [0, 2], ...o }),
  matrix: (o) =>
    matrix(
      [
        [1, 2],
        [3, 4],
      ],
      { rowLabels: ['a', 'b'], ...o },
    ),
  maskMatrix: (o) =>
    maskMatrix(
      [
        [true, false],
        [true, true],
      ],
      { rowLabels: ['a', 'b'], ...o },
    ),
  historyLayers: (o) => historyLayers(study, o),
}

/** The option with every function replaced by its source (rebuilt closures are different values of the same text). */
function plain(value: unknown): unknown {
  if (typeof value === 'function') return `fn:${value.toString()}`
  if (Array.isArray(value)) return value.map(plain)
  if (value !== null && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype)
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, plain(v)]))
  return value
}
const layersOf = (l: Layer | readonly Layer[]): readonly Layer[] => (Array.isArray(l) ? l : [l as Layer])
const composed = (layers: readonly Layer[], mode: ColourMode): unknown => plain(cartesian(layers, { mode }))

describe('layers built without a mode, drawn in the mode of the chart', () => {
  const modes: ColourMode[] = ['light', 'dark']
  for (const [name, build] of Object.entries(builders)) {
    it(`${name}: cartesian in each mode equals the layer built with that mode`, () => {
      const open = layersOf(build({}))
      for (const mode of modes) {
        const explicit = layersOf(build({ mode }))
        expect(composed(open, mode)).toEqual(composed(explicit, mode))
      }
    })
  }

  it('really changes between the modes (the property above is not vacuous) for the builders with theme colours', () => {
    const same: string[] = []
    for (const [name, build] of Object.entries(builders))
      if (
        JSON.stringify(composed(layersOf(build({})), 'dark')) === JSON.stringify(composed(layersOf(build({})), 'light'))
      )
        same.push(name)
    expect(same).toEqual([])
  })

  it('the layer itself is the light layer as before: the same series as one built with mode light', () => {
    for (const [name, build] of Object.entries(builders)) {
      const open = layersOf(build({}))
      const light = layersOf(build({ mode: 'light' }))
      expect(plain(open.map((l) => l.series)), name).toEqual(plain(light.map((l) => l.series)))
    }
  })

  it('an explicit mode on the layer wins over the chart mode, as before', () => {
    const dark = lines([{ y: [1, 2], colour: 'fg' }], { mode: 'dark' })
    expect(atMode(dark, 'light')).toBe(dark)
    const series = (o: unknown): { lineStyle: { color: string } } =>
      (o as { series: { lineStyle: { color: string } }[] }).series[0] as { lineStyle: { color: string } }
    expect(series(composed([dark], 'light')).lineStyle.color).toBe('#fafafa')
  })

  it('draws the issue case: series colour fg built with no mode is the dark foreground in a dark chart', () => {
    const l = lines([{ x: [0, 1], y: [0, 1], colour: 'fg' }])
    const colour = (o: unknown): string =>
      (o as { series: { lineStyle: { color: string } }[] }).series[0]?.lineStyle.color ?? ''
    expect(colour(composed([l], 'dark'))).toBe('#fafafa')
    expect(colour(composed([l], 'light'))).toBe('#0a0a0a')
    // An explicit colour is what it was.
    const hex = lines([{ x: [0, 1], y: [0, 1], colour: '#123456' }])
    expect(colour(composed([hex], 'dark'))).toBe('#123456')
  })

  it('a figure with a mode stands for the mode: subplots and plotRectOf take it', () => {
    const dark = figure({ mode: 'dark' })
    const l = lines([{ x: [0, 1], y: [0, 1], colour: 'fg' }])
    const explicit = lines([{ x: [0, 1], y: [0, 1], colour: 'fg' }], { mode: 'dark' })
    const panel = (layer: Layer) => ({ r0c0: { layers: [layer] } })
    expect(plain(subplots(gridSpec(1, 1), panel(l), { figure: dark }))).toEqual(
      plain(subplots(gridSpec(1, 1), panel(explicit), { figure: dark })),
    )
    expect(plain(plotRectOf([l], { figure: dark }))).toEqual(plain(plotRectOf([explicit], { figure: dark })))
    expect(plain(panelParts([l], { mode: 'dark' }, { x: 0, y: 0, width: 100, height: 100 }, 0, 0, 0).series)).toEqual(
      plain(panelParts([explicit], { mode: 'dark' }, { x: 0, y: 0, width: 100, height: 100 }, 0, 0, 0).series),
    )
  })

  it('a layer that is copied or edited is plain data: it is drawn as it was, not redrawn from the old recipe', () => {
    const l = lines([{ y: [1, 2], colour: 'fg' }])
    const copy = { ...l, secondary: true }
    expect(atMode(copy, 'dark')).toBe(copy)
    expect(atMode(l, 'dark')).not.toBe(l)
    // The recipe is cached: composing again in a mode gives the same layer, without building it again.
    expect(atMode(l, 'dark')).toBe(atMode(l, 'dark'))
  })
})
