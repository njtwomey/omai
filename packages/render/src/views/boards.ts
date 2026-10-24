import type { EChartsOption, SeriesOption } from 'echarts'
import type { BoardSteps, PopulationSteps } from 'whitebox-engine/exposition'
import { covarianceEllipse } from 'whitebox-engine/numerics/geometry'
import type { Pixels } from 'whitebox-engine/scene'
import { toArray } from 'whitebox-engine/tensor'
import { bars } from '../charts/bars.ts'
import { type Panel, subplots } from '../charts/compose.ts'
import { contours } from '../charts/fields.ts'
import { type Curve, lines } from '../charts/lines.ts'
import { heatmap, imageSeries } from '../charts/matrices.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { gridSpec } from '../figure/grid.ts'
import { categoricalColour, mixColours, parseColour, relativeLuminance } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { outlined, short, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

// Well-known category names and the colours they get, so every search, solver and automaton reads alike.
function knownColour(name: string, mode: ColourMode): string | undefined {
  const t = tokens(mode)
  // Tints of a colour towards the background (lighter in light mode, deeper in dark mode).
  const tint = (c: string, k: number): string => mixColours(c, t.bg, mode === 'light' ? k : 0.75 * k)
  switch (name) {
    case 'empty':
    case 'unvisited':
    case 'unseen':
    case 'unassigned':
    case 'dead':
    case 'open':
      return t.surface
    case 'wall':
      return mode === 'light' ? mixColours(t.fg, t.bg, 0.25) : mixColours(t.fg, t.bg, 0.55)
    case 'alive':
    case 'queen':
      return t.fg
    case 'explored':
    case 'visited':
    case 'expanded':
      return tint(t.accent, 0.62)
    case 'frontier':
    case 'candidate':
      return categoricalColour(1)
    case 'current':
      return t.accent
    case 'path':
    case 'solution':
    case 'frequent':
      return categoricalColour(2)
    case 'start':
      return categoricalColour(4)
    case 'goal':
      return t.success
    case 'pruned':
    case 'infrequent':
    case 'ruled out':
      return tint(t.muted, 0.45)
    case 'conflict':
      return t.danger
    case 'given':
      return tint(t.fg, 0.82)
    case 'shade':
      return mixColours(t.surface, t.fg, 0.08)
    case 'assigned':
      return tint(t.accent, 0.72)
    case 'MAX':
      return tint(categoricalColour(0), 0.25)
    case 'MIN':
      return tint(categoricalColour(3), 0.25)
    case 'water':
      return tint(categoricalColour(0), 0.45)
    case 'forest':
      return tint(categoricalColour(5), 0.4)
    case 'grass':
      return tint(categoricalColour(8), 0.45)
    case 'red':
      return categoricalColour(3)
    case 'green':
      return categoricalColour(2)
    case 'blue':
      return categoricalColour(0)
    case 'yellow':
      return categoricalColour(8)
    default:
      return undefined
  }
}

/**
 * The colour of each of a board's categories: well-known names (`wall`, `explored`, `frontier`, `current`, `path`,
 * `goal`, `pruned`, `conflict`, `alive`, colour names, ...) get fixed theme colours, the rest the tab10 palette in
 * order.
 *
 * @param categories - The category names.
 * @param mode - The colour mode.
 * @returns A CSS colour per category.
 * @example
 * ```ts
 * import { boardColours } from 'whitebox-render/views'
 * boardColours(['frontier', 'type A'])[1] // => '#1f77b4'
 * ```
 */
export function boardColours(categories: readonly string[], mode: ColourMode = 'light'): string[] {
  let next = 0
  return categories.map((name) => knownColour(name, mode) ?? categoricalColour(next++))
}

/** Text that reads on a fill: dark on light fills, light on dark ones. */
function inkOn(fill: string, mode: ColourMode): string {
  const t = tokens(mode)
  const light = relativeLuminance(fill) > 0.35
  return light ? (mode === 'light' ? t.fg : '#111111') : '#ffffff'
}

// A numbered metric over steps with the current step marked.
function metricPanel(
  metrics: readonly { readonly name: string; readonly values: readonly number[] }[],
  at: number,
  stepName: string,
  mode: ColourMode,
): Panel {
  const curves: Curve[] = metrics.map((m, i) => ({
    x: m.values.map((_, j) => j),
    y: m.values,
    name: m.name,
    colour: i === 0 ? 'fgSecondary' : categoricalColour(i),
    width: 1.5,
  }))
  const now = metrics.map((m, i) => ({
    x: [at],
    y: [m.values[at] ?? 0],
    name: `__now${i}`,
    colour: 'accent',
    size: 8,
  }))
  return {
    layers: [lines(curves, { mode }), points(now, { mode })],
    options: { x: { label: stepName }, y: { label: metrics.length === 1 ? (metrics[0]?.name ?? '') : 'value' } },
  }
}

// The legend entries of the categories a record uses (empty series carry the names and colours).
function legendSeries(names: readonly string[], colours: readonly string[], used: ReadonlySet<number>): SeriesOption[] {
  return names.flatMap((name, i) =>
    used.has(i)
      ? [
          {
            type: 'scatter',
            name,
            data: [],
            symbol: 'rect',
            itemStyle: { color: colours[i] },
          } as SeriesOption,
        ]
      : [],
  )
}

// Square legend keys, the shape of the cells they name.
function squareKeys(option: EChartsOption): EChartsOption {
  const legend = option.legend
  if (legend === undefined || Array.isArray(legend)) return option
  return { ...option, legend: { ...legend, icon: 'rect', itemWidth: 10, itemHeight: 10 } }
}

/** Options of {@link boardFrames}. */
export type BoardViewOptions = ViewOptions

/**
 * A board's steps as frames for a player: the grid (or graph) with every cell coloured by its category at the step and
 * its text, the step's caption in the stamp, and the record's metrics over the steps with the current step marked.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The frames.
 */
export function boardFrames(record: BoardSteps, options: BoardViewOptions = {}): ViewFrames {
  const fig = viewFigure(options, { aspect: 1.75 })
  const mode = fig.mode
  const t = tokens(mode)
  const colours = boardColours(record.categories, mode)
  const count = Math.max(1, record.steps.length)
  const used = new Set<number>()
  for (const s of record.steps) for (const c of s.cells) used.add(c)
  const metrics = (record.metrics ?? []).slice(0, 2)
  const stepName = record.stepName ?? 'step'
  const withMetrics = metrics.length > 0 && record.steps.length > 1
  const mainWidth = fig.width * (withMetrics ? 0.58 : 0.9)
  const layout = record.layout
  // Graph nodes as large as the closest pair allows at the board's scale (between 6 and 30 pixels).
  let nodeSize = 20
  if (layout.type === 'graph' && layout.nodes.length > 1) {
    const xs = layout.nodes.map((p) => p.x)
    const ys = layout.nodes.map((p) => p.y)
    const span = Math.max(Math.max(...xs) - Math.min(...xs), 1e-9)
    const rise = Math.max(Math.max(...ys) - Math.min(...ys), 1e-9)
    const scale = Math.min(mainWidth / span, (fig.height * 0.75) / rise)
    let closest = Number.POSITIVE_INFINITY
    for (let i = 0; i < layout.nodes.length; i++)
      for (let j = i + 1; j < layout.nodes.length; j++) {
        const d = Math.hypot((xs[i] ?? 0) - (xs[j] ?? 0), (ys[i] ?? 0) - (ys[j] ?? 0))
        if (d > 0 && d < closest) closest = d
      }
    nodeSize = Math.max(6, Math.min(30, Math.round(0.8 * closest * scale)))
  }
  const fillRgba = colours.map(parseColour)
  const gridRgba = parseColour(t.grid)
  const option = (frame: number): EChartsOption => {
    const at = Math.min(Math.max(frame, 0), count - 1)
    const step = record.steps[at]
    const cells = step?.cells ?? []
    const labels = step?.labels
    const main: Layer[] = []
    if (layout.type === 'grid') {
      const { rows, columns } = layout
      // One block of k by k pixels per cell, its last row and column the grid line.
      const k = Math.max(4, Math.min(24, Math.floor(480 / Math.max(rows, columns))))
      const width = columns * k
      const height = rows * k
      const rgba = new Uint8ClampedArray(width * height * 4)
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < columns; c++) {
          const fill = fillRgba[cells[r * columns + c] ?? 0] ?? gridRgba
          for (let i = 0; i < k; i++)
            for (let j = 0; j < k; j++) {
              const edge = i === k - 1 || j === k - 1 || i === 0 || j === 0
              const [cr = 0, cg = 0, cb = 0] = edge && k >= 6 ? gridRgba : fill
              const p = 4 * ((r * k + i) * width + c * k + j)
              rgba[p] = cr
              rgba[p + 1] = cg
              rgba[p + 2] = cb
              rgba[p + 3] = 255
            }
        }
      const pixels: Pixels = { width, height, rgba }
      const series: SeriesOption[] = [imageSeries(pixels, [0, columns], [0, rows], { name: '__board' })]
      if (labels !== undefined) {
        const cellPx = Math.min(mainWidth / columns, (fig.height * 0.8) / rows)
        const short = labels.every((l) => l.length <= 1)
        const size = Math.max(7, Math.min(short ? 24 : 18, Math.round(cellPx * (short ? 0.62 : 0.5))))
        const data = labels.flatMap((text, i) => {
          if (text === '') return []
          const r = Math.floor(i / columns)
          const c = i % columns
          return [
            {
              value: [c + 0.5, rows - r - 0.5],
              name: text,
              label: { color: inkOn(colours[cells[i] ?? 0] ?? t.bg, mode) },
            },
          ]
        })
        series.push({
          type: 'scatter',
          name: '__labels',
          data,
          symbolSize: 0,
          silent: true,
          z: 5,
          label: { show: true, position: 'inside', formatter: '{b}', fontSize: size, fontFamily: t.font },
        } as SeriesOption)
      }
      series.push(...legendSeries(record.categories, colours, used))
      main.push(layer(series, { x: [0, columns], y: [0, rows], tight: true }))
    } else {
      const nodes = layout.nodes
      const xs: number[] = []
      const ys: number[] = []
      for (const [a, b] of layout.edges) {
        const p = nodes[a]
        const q = nodes[b]
        if (p === undefined || q === undefined) continue
        xs.push(p.x, q.x, Number.NaN)
        ys.push(-p.y, -q.y, Number.NaN)
      }
      if (xs.length > 0) main.push(lines([{ x: xs, y: ys, name: '__edges', colour: 'axis', width: 1.2 }], { mode }))
      const size = nodeSize
      const byCategory = new Map<number, number[]>()
      for (const [i, c] of cells.entries()) byCategory.set(c, [...(byCategory.get(c) ?? []), i])
      const sets = [...byCategory.entries()]
        .sort((p, q) => p[0] - q[0])
        .map(([c, idx]) => ({
          x: idx.map((i) => nodes[i]?.x ?? 0),
          y: idx.map((i) => -(nodes[i]?.y ?? 0)),
          name: record.categories[c] ?? `${c}`,
          colour: colours[c] ?? t.muted,
          size,
          outline: false,
        }))
      main.push(outlined(points(sets, { mode }), mode, 1))
      const text = nodes.map((n, i) => {
        const l = labels?.[i]
        return l !== undefined && l !== '' ? l : (n.label ?? '')
      })
      if (text.some((s) => s !== '')) {
        const fontSize = Math.max(7, Math.min(13, Math.round(size * 0.45)))
        const data = text.flatMap((s, i) =>
          s === ''
            ? []
            : [
                {
                  value: [nodes[i]?.x ?? 0, -(nodes[i]?.y ?? 0)],
                  name: s,
                  label: {
                    color: s.length > 3 ? t.fg : inkOn(colours[cells[i] ?? 0] ?? t.bg, mode),
                    position: s.length > 3 ? 'bottom' : 'inside',
                  },
                },
              ],
        )
        main.push(
          layer([
            {
              type: 'scatter',
              name: '__labels',
              data,
              symbolSize: 0,
              silent: true,
              z: 5,
              label: { show: true, formatter: '{b}', fontSize, fontFamily: t.font },
            } as SeriesOption,
          ]),
        )
      }
    }
    const board: Panel = {
      layers: main,
      options: {
        x: { show: false, splitLine: false },
        y: { show: false, splitLine: false },
        aspect: 'equal',
      },
    }
    if (!withMetrics)
      return squareKeys(cartesian(board.layers, { ...board.options, figure: fig, mode, legend: used.size > 1 }))
    const grid = gridSpec(1, 2, {
      widthRatios: [1.6, 1],
      wspace: 64,
      cells: [
        { id: 'board', row: 0, col: 0 },
        { id: 'metrics', row: 0, col: 1 },
      ],
    })
    return squareKeys(
      subplots(
        grid,
        { board, metrics: metricPanel(metrics, at, stepName, mode) },
        { figure: fig, legend: used.size > 1 },
      ),
    )
  }
  return {
    count,
    label: (f) => {
      const caption = record.steps[f]?.caption
      return `${stepName} ${f}/${count - 1}${caption === undefined ? '' : ` · ${caption}`}`
    },
    option,
  }
}

/** Options of {@link populationFrames}. */
export type PopulationViewOptions = ViewOptions

/**
 * A population's generations as frames for a player: the individuals over the objective's contours (with the search
 * distribution's ellipses and the best point), or the best program's curve over its data, beside the best and mean
 * fitness per generation with the current one marked.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The frames.
 */
export function populationFrames(record: PopulationSteps, options: PopulationViewOptions = {}): ViewFrames {
  const fig = viewFigure(options, { aspect: 1.75 })
  const mode = fig.mode
  const gens = record.generations
  const count = Math.max(1, gens.length)
  const lower = record.lowerIsBetter ?? false
  const bestSoFar: number[] = []
  for (const g of gens) {
    const prev = bestSoFar[bestSoFar.length - 1]
    const v = g.best.fitness
    bestSoFar.push(prev === undefined ? v : lower ? Math.min(prev, v) : Math.max(prev, v))
  }
  const means = gens.map((g) => {
    const finite = g.fitness.filter(Number.isFinite)
    return finite.length === 0 ? Number.NaN : finite.reduce((a, b) => a + b, 0) / finite.length
  })
  const surface = record.surface
  const backdrop: Layer[] = []
  if (surface !== undefined) {
    backdrop.push(
      heatmap(surface.values, {
        x: surface.x,
        y: surface.y,
        smooth: 'contours',
        ramp: 'viridis',
        opacity: 0.45,
        colourBar: false,
        mode,
      }),
      contours(surface.values, { x: surface.x, y: surface.y, levels: 9, colour: 'fgSecondary', labels: false, mode }),
    )
  }
  const option = (frame: number): EChartsOption => {
    const at = Math.min(Math.max(frame, 0), count - 1)
    const g = gens[at]
    const main: Layer[] = [...backdrop]
    let mainOptions: Panel['options'] = {}
    if (surface !== undefined) {
      if (g?.points !== undefined)
        main.push(
          outlined(
            points(
              [
                {
                  x: g.points.map((p) => p[0]),
                  y: g.points.map((p) => p[1]),
                  name: 'population',
                  colour: 'tab1',
                  size: 7,
                },
              ],
              { mode },
            ),
            mode,
            0.8,
          ),
        )
      if (g?.mean !== undefined && g.covariance !== undefined) {
        const [xx, xy, yy] = g.covariance
        const rings: Curve[] = [1, 2].map((nStd) => {
          const e = toArray(
            covarianceEllipse(
              [g.mean?.[0] ?? 0, g.mean?.[1] ?? 0],
              [
                [xx, xy],
                [xy, yy],
              ],
              { nStd, points: 72 },
            ).points,
          ) as number[][]
          const closed = [...e, e[0] ?? [0, 0]]
          return {
            x: closed.map((p) => p[0] ?? 0),
            y: closed.map((p) => p[1] ?? 0),
            name: `__ring${nStd}`,
            colour: 'accent',
            width: nStd === 1 ? 2 : 1,
            dash: nStd === 1 ? 'solid' : 'dashed',
          }
        })
        main.push(lines(rings, { mode }))
      }
      const marks = []
      if (surface.optimum !== undefined)
        marks.push({
          x: [surface.optimum[0]],
          y: [surface.optimum[1]],
          name: 'optimum',
          colour: 'fg',
          size: 12,
          symbol: 'path://M0,0L10,10M10,0L0,10',
        })
      if (g?.best.point !== undefined)
        marks.push({
          x: [g.best.point[0]],
          y: [g.best.point[1]],
          name: 'best',
          colour: 'tab3',
          size: 14,
          symbol: 'diamond',
        })
      if (marks.length > 0) main.push(outlined(points(marks, { mode }), mode, 1.2))
      const [a = 'x', b = 'y'] = surface.names ?? []
      mainOptions = {
        x: { label: a, limits: [surface.x[0] ?? 0, surface.x[surface.x.length - 1] ?? 1] },
        y: { label: b, limits: [surface.y[0] ?? 0, surface.y[surface.y.length - 1] ?? 1] },
      }
    } else if (record.data !== undefined) {
      const d = record.data
      main.push(points([{ x: d.x, y: d.y, name: 'data', colour: 'muted', size: 6 }], { mode }))
      if (g?.best.curve !== undefined)
        main.push(lines([{ x: d.grid, y: g.best.curve, name: 'best program', colour: 'accent', width: 2 }], { mode }))
      const lo = Math.min(...d.y)
      const hi = Math.max(...d.y)
      const pad = 0.25 * (hi - lo || 1)
      mainOptions = { x: { label: 'x' }, y: { label: 'y', limits: [lo - pad, hi + pad] } }
    } else {
      const sorted = [...(g?.fitness ?? [])].sort((p, q) => (lower ? p - q : q - p))
      main.push(
        bars(
          sorted.map((_, i) => `${i}`),
          [{ name: 'fitness', values: sorted, colour: 'tab0' }],
          { mode },
        ),
      )
      mainOptions = { x: { label: 'individual (sorted)', show: false }, y: { label: 'fitness' } }
    }
    const label = g?.best.label
    const panelMain: Panel = {
      layers: main,
      options: { ...mainOptions, ...(label === undefined ? {} : { title: label }) },
    }
    if (gens.length < 2) return cartesian(panelMain.layers, { ...panelMain.options, figure: fig, mode, legend: false })
    const x = gens.map((_, i) => i)
    const progress: Panel = {
      layers: [
        lines(
          [
            { x, y: bestSoFar, name: 'best so far', colour: 'accent', width: 1.8 },
            { x, y: means, name: 'mean', colour: 'fgSecondary', width: 1.2, dash: 'dashed' },
          ],
          { mode },
        ),
        points([{ x: [at], y: [bestSoFar[at] ?? 0], name: '__now', colour: 'accent', size: 9 }], { mode }),
      ],
      options: { x: { label: 'generation' }, y: { label: 'fitness' } },
    }
    const grid = gridSpec(1, 2, {
      widthRatios: [1.6, 1],
      wspace: 64,
      cells: [
        { id: 'main', row: 0, col: 0 },
        { id: 'progress', row: 0, col: 1 },
      ],
    })
    return subplots(grid, { main: panelMain, progress }, { figure: fig, legend: false })
  }
  return {
    count,
    label: (f) => {
      const g = gens[f]
      const best = g === undefined ? '' : ` · best ${short(g.best.fitness)}`
      return `generation ${f}/${count - 1}${best}`
    },
    option,
  }
}
