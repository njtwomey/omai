/**
 * Attribution views for other modalities, read from a feature-attribution record's `layout`: an image's input beside
 * its attribution map, a text's tokens as a strip highlighted by attribution, and a series with its attribution along
 * time.
 */
import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf } from 'omai-engine/base'
import type { FeatureAttributions } from 'omai-engine/exposition'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { heatmap } from '../charts/matrices.ts'
import { layer } from '../charts/plot.ts'
import type { Figure } from '../figure/figure.ts'
import { gridSpec } from '../figure/grid.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { short, spanOf, type ViewOptions, viewFigure } from './_view.ts'
import { ATTRIBUTION_DOWN, ATTRIBUTION_UP } from './attributions.ts'

/** Options of the modality views: which explained row. */
export interface ModalityViewOptions extends ViewOptions {
  /** Which explained row (default 0). */
  readonly row?: number
}

function imageShape(record: FeatureAttributions): [number, number] {
  if (record.layout?.kind === 'image') return [record.layout.shape[0], record.layout.shape[1]]
  const d = record.features.length
  const s = Math.round(Math.sqrt(d))
  return s * s === d ? [s, s] : [1, d]
}

function asImage(values: readonly number[], h: number, w: number): number[][] {
  // Heat maps put row 0 at the bottom: reversed, so the image's first row is on top.
  return Array.from({ length: h }, (_, r) => Array.from({ length: w }, (_, c) => values[(h - 1 - r) * w + c] ?? 0))
}

/**
 * An image's attributions (captum's visualize_image_attr, side by side): the input in grey and its attribution map on
 * a diverging ramp centred on zero (red raises the output, blue lowers it), pixel for pixel.
 *
 * @param record - A feature-attribution record with an image layout (or a square number of features).
 * @param options - row, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { featureAttributions } from 'omai-engine/exposition'
 * import { attributionImageView } from 'omai-render/views'
 * const r = featureAttributions({ method: 'occlusion', features: ['a', 'b', 'c', 'd'], output: 'y', baseValue: 0, values: [[1, 0, 0, -1]], data: [[1, 0, 0, 1]], predictions: [1], layout: { kind: 'image', shape: [2, 2] } })
 * attributionImageView(r).grid.length // => 2
 * ```
 */
export function attributionImageView(record: FeatureAttributions, options: ModalityViewOptions = {}): EChartsOption {
  const row = options.row ?? 0
  const [h, w] = imageShape(record)
  const fig = viewFigure(options, { aspect: Math.min(2.6, Math.max(1.6, (2.3 * w) / h)), padding: { right: 84 } })
  const mode = fig.mode
  const values = record.values[row] ?? []
  const data = record.data[row] ?? []
  const m = Math.max(1e-12, maxOf(values.map(Math.abs)))
  const hidden = { show: false, splitLine: false }
  const panels: Record<string, Panel> = {
    input: {
      layers: [heatmap(asImage(data, h, w), { smooth: 'cells', ramp: 'greys', colourBar: false, mode })],
      options: { title: 'input', x: hidden, y: hidden, aspect: 'equal' },
    },
    attribution: {
      layers: [
        heatmap(asImage(values, h, w), {
          smooth: 'cells',
          centre: 0,
          range: [-m, m],
          label: `attribution to ${record.output}`,
          mode,
        }),
      ],
      options: { title: `${record.method} · ${record.output}`, x: hidden, y: hidden, aspect: 'equal' },
    },
  }
  const grid = gridSpec(1, 2, {
    wspace: 24,
    cells: [
      { id: 'input', row: 0, col: 0 },
      { id: 'attribution', row: 0, col: 1 },
    ],
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: false })
}

/** A token's place in a strip: its line, its left edge and width in pixels. */
export interface TokenBox {
  readonly token: number
  readonly line: number
  readonly x: number
  readonly width: number
}

/**
 * Lay tokens out in lines of a given width, as text wraps: each token as wide as its characters at the font size
 * (an average glyph of 0.6 em) plus padding, a gap between tokens.
 *
 * @param texts - The tokens.
 * @param width - The line width in pixels.
 * @param fontSize - The font size in pixels.
 * @returns A box per token.
 * @example
 * ```ts
 * import { tokenLayout } from 'omai-render/views'
 * tokenLayout(['ab', 'cd'], 1000, 10).map((b) => b.line) // => [0, 0]
 * tokenLayout(['ab', 'cd'], 30, 10).map((b) => b.line) // => [0, 1]
 * ```
 */
export function tokenLayout(texts: readonly string[], width: number, fontSize: number): TokenBox[] {
  const out: TokenBox[] = []
  let line = 0
  let x = 0
  const gap = Math.round(fontSize * 0.35)
  texts.forEach((t, i) => {
    const w = Math.ceil(t.length * fontSize * 0.6 + fontSize * 0.7)
    if (x > 0 && x + w > width) {
      line += 1
      x = 0
    }
    out.push({ token: i, line, x, width: w })
    x += w + gap
  })
  return out
}

/**
 * The height a token strip needs at a figure's width and font size: a header line, then one line per wrapped line of
 * tokens.
 *
 * @param texts - The tokens.
 * @param fig - The figure (its width, padding and font size).
 * @returns The height in pixels.
 * @example
 * ```ts
 * import { figure } from 'omai-render/figure'
 * import { tokenStripHeight } from 'omai-render/views'
 * tokenStripHeight(['a', 'b'], figure()) > 0 // => true
 * ```
 */
export function tokenStripHeight(texts: readonly string[], fig: Figure): number {
  const fs = fig.fontSize + 2
  const boxes = tokenLayout(texts, fig.width - fig.padding.left - fig.padding.right, fs)
  const nLines = (boxes.at(-1)?.line ?? 0) + 1
  return fig.padding.top + Math.round(fs * 2.2) + nLines * Math.round(fs * 3) + Math.round(fs * 1.2)
}

function mix(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => Number.parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => Number.parseInt(b.slice(i, i + 2), 16))
  const c = pa.map((v, i) => Math.round(v + ((pb[i] ?? 0) - v) * t))
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`
}

function tokenColour(v: number, max: number, mode: ColourMode): string {
  const bg = tokens(mode).bg
  const base = /^#[0-9a-f]{6}$/i.test(bg) ? bg : mode === 'dark' ? '#121212' : '#ffffff'
  const t = Math.min(1, Math.abs(v) / max)
  return mix(base, v >= 0 ? ATTRIBUTION_UP : ATTRIBUTION_DOWN, 0.85 * t)
}

/**
 * A text's attributions as a token strip (captum's text visualisation, shap's text plot): the tokens in reading
 * order, wrapped into lines, each on a background from blue (lowers the output) through the page colour to red
 * (raises it), with the attribution beneath; the prediction and base value above.
 *
 * @param record - A feature-attribution record whose features are tokens.
 * @param options - row, mode and figure.
 * @returns The ECharts option (drawn with graphic elements).
 * @example
 * ```ts
 * import { featureAttributions } from 'omai-engine/exposition'
 * import { tokenStripView } from 'omai-render/views'
 * const r = featureAttributions({ method: 'ablation', features: ['not', 'bad'], output: 'P(positive)', baseValue: 0.5, values: [[0.2, -0.1]], data: [[1, 1]], predictions: [0.6], layout: { kind: 'tokens' } })
 * tokenStripView(r).graphic.length > 2 // => true
 * ```
 */
export function tokenStripView(record: FeatureAttributions, options: ModalityViewOptions = {}): EChartsOption {
  const row = options.row ?? 0
  const base = viewFigure(options, { height: tokenStripHeight(record.features, viewFigure(options, {})) })
  const fs = base.fontSize + 2
  const width = base.width - base.padding.left - base.padding.right
  const boxes = tokenLayout(record.features, width, fs)
  const lineHeight = Math.round(fs * 3)
  const top = base.padding.top + Math.round(fs * 2.2)
  const t = tokens(base.mode)
  const values = record.values[row] ?? []
  const max = Math.max(1e-12, maxOf(values.map(Math.abs)))
  const prediction = record.predictions[row]
  const header = `${record.output}: ${prediction === undefined ? '' : short(prediction)}  (base ${short(record.baseValues?.[row] ?? record.baseValue)}) · ${record.method}`
  const graphic: Record<string, unknown>[] = [
    {
      type: 'text',
      left: base.padding.left,
      top: base.padding.top,
      style: { text: header, fill: t.fgSecondary, font: `${fs}px ${t.font}` },
    },
  ]
  for (const b of boxes) {
    const v = values[b.token] ?? 0
    const x = base.padding.left + b.x
    const y = top + b.line * lineHeight
    graphic.push(
      {
        type: 'rect',
        shape: { x, y, width: b.width, height: Math.round(fs * 1.6), r: 3 },
        style: { fill: tokenColour(v, max, base.mode), stroke: t.muted, lineWidth: 0.5 },
      },
      {
        type: 'text',
        x: x + b.width / 2,
        y: y + Math.round(fs * 0.8),
        style: {
          text: record.features[b.token] ?? '',
          fill: t.fg,
          font: `${fs}px ${t.font}`,
          align: 'center',
          verticalAlign: 'middle',
        },
      },
      {
        type: 'text',
        x: x + b.width / 2,
        y: y + Math.round(fs * 2.2),
        style: {
          text: short(v),
          fill: v >= 0 ? ATTRIBUTION_UP : ATTRIBUTION_DOWN,
          font: `${Math.max(8, fs - 3)}px ${t.font}`,
          align: 'center',
          verticalAlign: 'middle',
        },
      },
    )
  }
  return {
    backgroundColor: t.bg,
    animation: false,
    graphic,
    series: [],
  } as EChartsOption
}

/**
 * A series' attributions (time-series saliency): the series above, each step's point coloured by its attribution, and
 * the attributions as bars along the same time axis beneath.
 *
 * @param record - A feature-attribution record whose features are time steps.
 * @param options - row, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { featureAttributions } from 'omai-engine/exposition'
 * import { attributionSeriesView } from 'omai-render/views'
 * const r = featureAttributions({ method: 'occlusion', features: ['t0', 't1', 't2'], output: 'y', baseValue: 0, values: [[0, 1, -1]], data: [[1, 2, 3]], predictions: [0], layout: { kind: 'series' } })
 * attributionSeriesView(r).grid.length // => 2
 * ```
 */
export function attributionSeriesView(record: FeatureAttributions, options: ModalityViewOptions = {}): EChartsOption {
  const row = options.row ?? 0
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const t = tokens(mode)
  const d = record.features.length
  const time =
    record.layout?.kind === 'series' && record.layout.time !== undefined
      ? record.layout.time
      : Array.from({ length: d }, (_, i) => i)
  const values = record.values[row] ?? []
  const data = record.data[row] ?? []
  const max = Math.max(1e-12, maxOf(values.map(Math.abs)))
  const dots = layer(
    [
      {
        type: 'scatter',
        name: '__steps',
        data: time.map((x, i) => ({
          value: [x, data[i] ?? 0],
          itemStyle: { color: tokenColour(values[i] ?? 0, max, mode), borderColor: t.muted, borderWidth: 0.5 },
        })),
        symbolSize: 7,
        z: 4,
      } as SeriesOption,
    ],
    { x: spanOf(time), y: spanOf(data) },
  )
  const panels: Record<string, Panel> = {
    series: {
      layers: [lines([{ x: time, y: data, colour: t.fgSecondary, width: 1.5 }], { mode }), dots],
      options: { y: { label: 'value' }, x: { label: '', show: false } },
    },
    attribution: {
      layers: [
        bars(
          time.map((x) => short(x)),
          [{ name: '__attribution', values, colour: 'accent' }],
          { mode },
        ),
        referenceLines({ y: [0] }, { mode, colour: t.muted, dash: 'solid' }),
      ],
      options: { x: { label: 'time' }, y: { label: `attribution to ${record.output}` } },
    },
  }
  const grid = gridSpec(2, 1, {
    heightRatios: [1.3, 1],
    hspace: 36,
    cells: [
      { id: 'series', row: 0, col: 0 },
      { id: 'attribution', row: 1, col: 0 },
    ],
  })
  return subplots(grid, panels, { figure: fig, legend: false, tooltip: 'item' })
}
