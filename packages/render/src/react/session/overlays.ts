/**
 * The overlays of the rows panel as one mechanism: per-row channels (a value as colour with a colour bar, a size, a
 * marker per category, regions of the plane, a chosen row with what is connected to it) each read from a path of the
 * session at the viewed step, a probe or a tensor of the tape, and drawn as chart layers over the rows. This module is
 * the pure part: reading a channel, coercing what it holds, and building the layers; the panel owns the probing.
 */

import type { SeriesOption } from 'echarts'
import { maxOf, minOf } from 'mdml-engine/base'
import { toFlat } from 'mdml-engine/tensor'
import { type Session, tensorSeries } from 'mdml-engine/trace'
import { lines, polygons } from '../../charts/lines.ts'
import { type Layer, layer } from '../../charts/plot.ts'
import { categoricalColour, colourScale, divergingRamp, sampleRamp, sequentialRamp } from '../../theme/palettes.ts'
import { type ColourMode, tokens } from '../../theme/tokens.ts'
import type { RowSource } from './layout.ts'

/** A region of the plane: a box (`lower` and `upper` corners; a missing or non-finite bound is open) or an ellipse. */
export type Region =
  | {
      readonly lower: readonly (number | null)[]
      readonly upper: readonly (number | null)[]
      readonly label?: string
    }
  | {
      readonly centre: readonly number[]
      /** The half-axes along x and y (before rotation). */
      readonly radii: readonly number[]
      /** The rotation in radians, anticlockwise (default 0). */
      readonly angle?: number
      readonly label?: string
    }

/**
 * The chosen row of a focus overlay and what is drawn from it: a path of rows (an isolation path, the order of
 * queries), the rows connected to it (its neighbours), a radius around it and a point related to it (its
 * reconstruction, its counterfactual).
 */
export interface Focus {
  readonly row: number
  /** Rows joined to the chosen one by thin lines. */
  readonly neighbours?: readonly number[]
  /** Rows visited in order, joined by one bold line that ends at the chosen row when it is the last. */
  readonly path?: readonly number[]
  /** A circle of this radius (in data units) around the chosen row: a neighbourhood, an attack's ε-ball. */
  readonly radius?: number
  /** A point related to the chosen row, joined to it by a dashed line. */
  readonly point?: readonly number[]
}

/**
 * Read a channel at a step: a derived path is probed, anything else is the latest full tensor of the tape at or before
 * the step (as a flat array).
 *
 * @param session - The session.
 * @param t - The step.
 * @param source - The path and the probe's arguments.
 * @returns The value, or `undefined` when the tape holds nothing there yet.
 */
export function readChannel(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  session: Session<any, any>,
  t: number,
  source: RowSource,
): unknown {
  if (session.contract[source.path]?.kind === 'derived') return session.probe(t, source.path, source.args)
  const event = tensorSeries(session.run(), source.path)
    .filter((e) => e.values !== undefined && e.indices === undefined && e.step <= t)
    .at(-1)
  return event?.values === undefined ? undefined : Array.from(event.values)
}

/**
 * Whether a source is read by probing (so it depends on the step only), not from the growing tape.
 *
 * @param session - The session, whose contract says what a path is.
 * @param source - The channel's path and arguments.
 * @returns True for a derived path (or no source): its value depends on the viewed step only.
 */
export function isProbed(
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  session: Session<any, any>,
  source: RowSource | undefined,
): boolean {
  return source === undefined || session.contract[source.path]?.kind === 'derived'
}

/**
 * A number per row from a channel's value: an array (or tensor) of numbers, or a record with `values`.
 *
 * @param value - What the channel holds.
 * @returns The numbers, or `undefined` when it holds none.
 */
export function channelNumbers(value: unknown): number[] | undefined {
  const inner =
    typeof value === 'object' && value !== null && !Array.isArray(value) && 'values' in value
      ? (value as { readonly values: unknown }).values
      : value
  if (inner === undefined || inner === null) return undefined
  try {
    const flat =
      Array.isArray(inner) && inner.every((v) => typeof v === 'number') ? (inner as number[]) : toFlat(inner as never)
    return Array.from(flat, (v) => v)
  } catch {
    return undefined
  }
}

/**
 * A category per row from a channel's value, for the marks overlay: an array with a category (a string, number or
 * `true`) for each marked row and `null`, `false` or `undefined` for the rest, or `{ sets }` naming the rows of each
 * category (`{ sets: { chosen: [3], flagged: [1, 2] } }`; later sets win where they overlap).
 *
 * @param value - What the channel holds.
 * @param n - The number of rows.
 * @returns A category (or `undefined`, unmarked) per row; `undefined` when the value is neither form.
 */
export function channelMarks(value: unknown, n: number): (string | undefined)[] | undefined {
  const category = (v: unknown): string | undefined =>
    v === undefined || v === null || v === false ? undefined : v === true ? 'marked' : String(v)
  if (Array.isArray(value)) return Array.from({ length: n }, (_, i) => category(value[i]))
  if (typeof value === 'object' && value !== null && 'sets' in value) {
    const out = new Array<string | undefined>(n).fill(undefined)
    for (const [name, rows] of Object.entries((value as { readonly sets: Record<string, readonly number[]> }).sets))
      for (const i of rows) if (i >= 0 && i < n) out[i] = name
    return out
  }
  return undefined
}

/**
 * The regions of a channel's value: one region, or an array of them.
 *
 * @param value - What the channel holds.
 * @returns The regions (none for anything else).
 */
export function channelRegions(value: unknown): Region[] {
  const list = Array.isArray(value) ? value : value === undefined || value === null ? [] : [value]
  return list.filter(
    (r): r is Region =>
      typeof r === 'object' && r !== null && (('lower' in r && 'upper' in r) || ('centre' in r && 'radii' in r)),
  )
}

/**
 * The focus of a channel's value.
 *
 * @param value - What the channel holds.
 * @returns The focus, or `undefined` when the value has no chosen row.
 */
export function channelFocus(value: unknown): Focus | undefined {
  if (typeof value !== 'object' || value === null || !('row' in value)) return undefined
  const row = (value as { readonly row: unknown }).row
  return typeof row === 'number' && Number.isInteger(row) && row >= 0 ? (value as Focus) : undefined
}

/**
 * The vertices of a region, with open box bounds filled from `bounds` (`[x0, x1, y0, y1]`).
 *
 * @param region - The region.
 * @param bounds - The plot bounds `[x0, x1, y0, y1]` that open box edges extend to.
 * @returns The vertices as `[x, y]` pairs.
 */
export function regionPoints(region: Region, bounds: readonly [number, number, number, number]): [number, number][] {
  const [bx0, bx1, by0, by1] = bounds
  if ('lower' in region) {
    const pick = (v: number | null | undefined, fallback: number): number =>
      v === undefined || v === null || !Number.isFinite(v) ? fallback : v
    const x0 = pick(region.lower[0], bx0)
    const y0 = pick(region.lower[1], by0)
    const x1 = pick(region.upper[0], bx1)
    const y1 = pick(region.upper[1], by1)
    return [
      [x0, y0],
      [x1, y0],
      [x1, y1],
      [x0, y1],
    ]
  }
  const [cx = 0, cy = 0] = region.centre
  const [rx = 1, ry = 1] = region.radii
  const a = region.angle ?? 0
  const [c, s] = [Math.cos(a), Math.sin(a)]
  return Array.from({ length: 72 }, (_, k): [number, number] => {
    const u = rx * Math.cos((2 * Math.PI * k) / 72)
    const v = ry * Math.sin((2 * Math.PI * k) / 72)
    return [cx + c * u - s * v, cy + s * u + c * v]
  })
}

/** The rows' own scatter and the legend entries it brings. */
export interface RowChannels {
  readonly mode: ColourMode
  /** The rows' coordinates. */
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** A group per row (the class), drawn as a marker shape when `values` colour the rows. */
  readonly groups?: readonly (string | number)[]
  readonly values?: {
    readonly numbers: readonly number[]
    readonly diverging: boolean
    readonly range?: readonly [number, number]
    readonly label?: string
  }
  readonly sizes?: { readonly numbers: readonly number[]; readonly range?: readonly [number, number] }
  readonly marks?: {
    readonly categories: readonly (string | undefined)[]
    readonly names?: Readonly<Record<string, string>>
  }
  readonly regions?: readonly Region[]
  readonly focus?: Focus
}

/** The layers the channels add around the rows' own scatter. */
export interface ChannelLayers {
  /** Beneath the rows: the regions. */
  readonly under: readonly Layer[]
  /** The rows coloured and sized by their channels, replacing the plain scatter (`undefined` when neither is set). */
  readonly rows?: Layer
  /** Over the rows: the focus's lines, the marks, the focus's ring, and the colour bar. */
  readonly over: readonly Layer[]
}

const SYMBOLS = ['circle', 'triangle', 'rect', 'diamond'] as const
const MARK_SYMBOLS = ['circle', 'rect', 'triangle', 'diamond'] as const
// Tab10 slots for the marks, leaving out grey (7), which would read as unmarked.
const MARK_SLOTS = [6, 9, 8, 4, 1] as const
const markColour = (k: number): string => categoricalColour(MARK_SLOTS[k % MARK_SLOTS.length] ?? 6)

// Size of a row in pixels from its channel value, between the range's ends (square-root scale: area is the value).
function sizeScale(numbers: readonly number[], range: readonly [number, number] = [4, 16]): (i: number) => number {
  const finite = numbers.filter(Number.isFinite)
  const lo = minOf(finite)
  const hi = maxOf(finite)
  const [a, b] = range
  return (i) => {
    const f = hi > lo ? ((numbers[i] ?? lo) - lo) / (hi - lo) : 0.5
    return Math.sqrt(a * a + Math.min(1, Math.max(0, f)) * (b * b - a * a))
  }
}

/**
 * The layers of the row channels: regions beneath, the rows coloured by a value and sized by a second, then over them
 * the focus (thin lines to its neighbours, a bold path, a radius, a related point), a marker shape per category of
 * marks, the ring of the chosen row and, for values, the colour bar.
 *
 * @param channels - The rows and what each channel holds.
 * @returns The layers.
 */
export function channelLayers(channels: RowChannels): ChannelLayers {
  const { mode, x, y } = channels
  const t = tokens(mode)
  const n = x.length
  const at = (i: number): [number, number] => [x[i] ?? 0, y[i] ?? 0]
  const finite = (xs: readonly number[]): number[] => xs.filter(Number.isFinite)
  const bounds = ((): [number, number, number, number] => {
    const [x0, x1, y0, y1] = [minOf(finite(x)), maxOf(finite(x)), minOf(finite(y)), maxOf(finite(y))]
    const px = 0.1 * (x1 - x0 || 1)
    const py = 0.1 * (y1 - y0 || 1)
    return [x0 - px, x1 + px, y0 - py, y1 + py]
  })()
  const under: Layer[] = []
  const over: Layer[] = []
  // Regions: a faint fill and an outline each, a colour of its own (named by the label for the legend, if it has one).
  if (channels.regions !== undefined && channels.regions.length > 0)
    under.push(
      polygons(
        channels.regions.map((r, k) => ({
          points: regionPoints(r, bounds),
          colour: categoricalColour(k + 2),
          ...(r.label === undefined ? {} : { name: r.label }),
        })),
        { mode },
      ),
    )
  // The rows themselves, when a value or a size is to be shown.
  let rows: Layer | undefined
  const { values, sizes } = channels
  if (values !== undefined || sizes !== undefined) {
    const stops = values?.diverging === true ? divergingRamp('rdbu') : sequentialRamp('viridis', mode)
    const finiteValues = finite(values?.numbers ?? [])
    const lo = values?.range?.[0] ?? minOf(finiteValues)
    const hi = values?.range?.[1] ?? maxOf(finiteValues)
    const bound = Math.max(Math.abs(lo), Math.abs(hi))
    const scale =
      values === undefined
        ? undefined
        : values.diverging
          ? colourScale(stops, [-bound, 0, bound])
          : colourScale(stops, [lo, hi])
    const sizeOf = sizes === undefined ? undefined : sizeScale(sizes.numbers, sizes.range)
    const order: (string | number)[] = []
    for (const g of channels.groups ?? []) if (!order.includes(g)) order.push(g)
    order.sort((a, b) => (typeof a === 'number' && typeof b === 'number' ? a - b : 0))
    if (order.length === 0) order.push('__all')
    const series: SeriesOption[] = order.map(
      (g, gi) =>
        ({
          type: 'scatter',
          name: g === '__all' ? '__rows' : `class ${g}`,
          symbol: SYMBOLS[gi % SYMBOLS.length],
          symbolSize: 8,
          color: t.fgSecondary,
          data: x.flatMap((_, i) =>
            g === '__all' || channels.groups?.[i] === g
              ? [
                  {
                    value: at(i),
                    index: i,
                    ...(sizeOf === undefined ? {} : { symbolSize: sizeOf(i) }),
                    itemStyle: {
                      color: scale === undefined ? t.fgSecondary : scale(values?.numbers[i] ?? 0),
                      opacity: 0.9,
                      borderColor: t.bg,
                      borderWidth: 0.8,
                    },
                  },
                ]
              : [],
          ),
          emphasis: { scale: 1.3 },
          clip: true,
        }) as SeriesOption,
    )
    rows = layer(series, {
      x: [minOf(finite(x)), maxOf(finite(x))],
      y: [minOf(finite(y)), maxOf(finite(y))],
    })
    if (values !== undefined && finiteValues.length > 0)
      over.push(
        layer(
          [{ type: 'scatter', name: '__colourbar', data: [], silent: true } as SeriesOption],
          {},
          {
            colourBar: {
              min: values.diverging ? -bound : lo,
              max: values.diverging ? bound : hi,
              colours: sampleRamp(stops, 9),
              target: 0,
              ...(values.label === undefined ? {} : { label: values.label }),
            },
          },
        ),
      )
  }
  const { focus } = channels
  if (focus !== undefined && focus.row < n) {
    const from = at(focus.row)
    const neighbours = (focus.neighbours ?? []).filter((j) => j >= 0 && j < n)
    if (neighbours.length > 0)
      over.push(
        lines(
          [
            {
              // One series with gaps between the spokes.
              x: neighbours.flatMap((j) => [from[0], at(j)[0], from[0]]),
              y: neighbours.flatMap((j) => [from[1], at(j)[1], Number.NaN]),
              name: '__neighbours',
              colour: 'muted',
              width: 1.2,
            },
          ],
          { mode },
        ),
      )
    const path = (focus.path ?? []).filter((j) => j >= 0 && j < n)
    if (path.length > 1)
      over.push(
        lines(
          [{ x: path.map((j) => at(j)[0]), y: path.map((j) => at(j)[1]), name: 'path', colour: 'accent', width: 2.5 }],
          { mode },
        ),
      )
    const radius = focus.radius
    if (radius !== undefined && radius > 0)
      over.push(
        lines(
          [
            {
              x: Array.from({ length: 73 }, (_, k) => from[0] + radius * Math.cos((2 * Math.PI * k) / 72)),
              y: Array.from({ length: 73 }, (_, k) => from[1] + radius * Math.sin((2 * Math.PI * k) / 72)),
              name: '__radius',
              colour: 'muted',
              width: 1,
              dash: 'dashed',
            },
          ],
          { mode },
        ),
      )
    const related = focus.point
    if (related !== undefined && related.length >= 2) {
      const [px = 0, py = 0] = related
      over.push(
        lines([{ x: [from[0], px], y: [from[1], py], name: '__related', colour: 'muted', width: 1, dash: 'dashed' }], {
          mode,
        }),
        layer(
          [
            {
              type: 'scatter',
              name: 'related point',
              data: [[px, py]],
              symbol: 'diamond',
              symbolSize: 11,
              color: categoricalColour(3),
              itemStyle: { color: categoricalColour(3), borderColor: t.bg, borderWidth: 1 },
              clip: true,
            } as SeriesOption,
          ],
          { x: [px, px], y: [py, py] },
        ),
      )
    }
  }
  // Marks: a marker shape per category, hollow so the row beneath still shows its colour.
  if (channels.marks !== undefined) {
    const { categories, names } = channels.marks
    const order: string[] = []
    for (const c of categories) if (c !== undefined && !order.includes(c)) order.push(c)
    const series = order.map(
      (c, k) =>
        ({
          type: 'scatter',
          name: names?.[c] ?? c,
          symbol: MARK_SYMBOLS[k % MARK_SYMBOLS.length],
          symbolSize: 13 + (k % MARK_SYMBOLS.length === 2 ? 2 : 0),
          color: markColour(k),
          itemStyle: { color: 'transparent', borderColor: markColour(k), borderWidth: 2 },
          data: categories.flatMap((cat, i) => (cat === c ? [{ value: at(i), index: i }] : [])),
          clip: true,
        }) as SeriesOption,
    )
    if (series.length > 0) over.push(layer(series, {}))
  }
  if (focus !== undefined && focus.row < n)
    over.push(
      layer(
        [
          {
            type: 'scatter',
            name: 'chosen',
            data: [at(focus.row)],
            symbol: 'circle',
            symbolSize: 20,
            color: t.fg,
            itemStyle: { color: 'transparent', borderColor: t.fg, borderWidth: 2.5 },
            clip: true,
          } as SeriesOption,
        ],
        {},
      ),
    )
  return { under, ...(rows === undefined ? {} : { rows }), over }
}
