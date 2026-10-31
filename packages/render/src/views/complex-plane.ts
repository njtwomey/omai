/**
 * The view of a `ComplexPlane` record: poles as crosses and zeros as circles on equal axes, the boundary of the stable
 * region (the imaginary axis, or the unit circle) and the branches of a root locus. A pole outside the stable region is
 * drawn in the danger colour and listed as unstable, so the picture does not rest on colour alone.
 */

import type { EChartsOption } from 'echarts'
import { type Complex, type ComplexPlane, isStablePole } from 'mdml-engine/exposition'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { ariaOf, spanOf, type ViewOptions, viewFigure } from './_view.ts'

// A filled cross (an x), the marker of a pole: a path symbol is a filled shape, so strokes alone would not show.
const CROSS = 'path://M-3,-2L-2,-3L0,-1L2,-3L3,-2L1,0L3,2L2,3L0,1L-2,3L-3,2L-1,0Z'

/**
 * The axis limits of a complex plane: symmetric about the origin's side, wide enough for every point, the locus and
 * the boundary, with the same range on both axes (the view draws them to scale).
 *
 * @param record - The record.
 * @returns The limits of the real and imaginary axes.
 * @example
 * ```ts
 * import { complexPlane } from 'mdml-engine/exposition'
 * import { complexPlaneLimits } from 'mdml-render/views'
 * complexPlaneLimits(complexPlane({ domain: 'z', poles: [[0.5, 0]] })).real // => [-1.5, 1.5]
 * ```
 */
export function complexPlaneLimits(record: ComplexPlane): { real: [number, number]; imaginary: [number, number] } {
  const all: Complex[] = [...record.poles, ...(record.zeros ?? []), ...(record.locus ?? []).flatMap((b) => b.path)]
  const reach = Math.max(record.domain === 'z' ? 1 : 0.5, ...all.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]))
  const r = Math.ceil(reach * 1.25 * 10) / 10
  const [lo, hi] = spanOf(all.map((p) => p[0]))
  // On the s-plane the real axis is lopsided: poles sit mostly on the left.
  return record.domain === 'z'
    ? { real: [-Math.max(1.5, r), Math.max(1.5, r)], imaginary: [-Math.max(1.5, r), Math.max(1.5, r)] }
    : { real: [Math.min(-r, lo - 0.2 * r), Math.max(r / 2, hi + 0.2 * r)], imaginary: [-r, r] }
}

/**
 * The layers of a complex plane: the stable region's boundary, the locus branches, the zeros and the poles (stable
 * and unstable apart, so a legend names them).
 *
 * @param record - The record.
 * @param options - The colour mode.
 * @returns The layers.
 * @example
 * ```ts
 * import { complexPlane } from 'mdml-engine/exposition'
 * import { complexPlaneLayers } from 'mdml-render/views'
 * complexPlaneLayers(complexPlane({ domain: 's', poles: [[-1, 1], [0.5, 0]], zeros: [[-2, 0]] })).length // => 3
 * ```
 */
export function complexPlaneLayers(record: ComplexPlane, options: ViewOptions = {}): Layer[] {
  const mode = options.mode ?? options.figure?.mode ?? 'light'
  const { imaginary } = complexPlaneLimits(record)
  const layers: Layer[] = []
  if (record.domain === 'z') {
    const t = Array.from({ length: 129 }, (_, i) => (2 * Math.PI * i) / 128)
    layers.push(
      lines(
        [
          {
            x: t.map(Math.cos),
            y: t.map(Math.sin),
            name: '__unit circle',
            colour: 'muted',
            width: 1.4,
            dash: 'dashed',
          },
        ],
        {
          mode,
        },
      ),
    )
  } else
    layers.push(
      lines([{ x: [0, 0], y: imaginary, name: '__imaginary axis', colour: 'muted', width: 1.4, dash: 'dashed' }], {
        mode,
      }),
    )
  for (const [k, b] of (record.locus ?? []).entries())
    layers.push(
      lines(
        [
          {
            x: b.path.map((p) => p[0]),
            y: b.path.map((p) => p[1]),
            name: b.name ?? (k === 0 ? 'locus' : `__locus ${k}`),
            colour: `tab${(k + 3) % 10}`,
            width: 1.2,
            opacity: 0.8,
          },
        ],
        { mode },
      ),
    )
  const zeros = record.zeros ?? []
  if (zeros.length > 0)
    layers.push(
      points(
        [
          {
            x: zeros.map((p) => p[0]),
            y: zeros.map((p) => p[1]),
            name: 'zeros',
            colour: 'info',
            symbol: 'emptyCircle',
            size: 12,
          },
        ],
        {
          mode,
        },
      ),
    )
  const named = (stable: boolean): { x: number[]; y: number[]; labels?: string[] } => {
    const keep = record.poles.map((p, i) => ({ p, i })).filter(({ p }) => isStablePole(record.domain, p) === stable)
    return {
      x: keep.map(({ p }) => p[0]),
      y: keep.map(({ p }) => p[1]),
      ...(record.labels === undefined ? {} : { labels: keep.map(({ i }) => record.labels?.[i] ?? '') }),
    }
  }
  const stable = named(true)
  const unstable = named(false)
  const sets = [
    ...(stable.x.length > 0 ? [{ ...stable, name: 'poles (stable)', colour: 'success', symbol: CROSS, size: 13 }] : []),
    ...(unstable.x.length > 0
      ? [{ ...unstable, name: 'poles (unstable)', colour: 'danger', symbol: CROSS, size: 17 }]
      : []),
  ]
  if (sets.length > 0) layers.push(points(sets, { mode }))
  return layers
}

/**
 * A complex plane as a figure on equal axes: poles, zeros, the locus and the stability boundary.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { complexPlane } from 'mdml-engine/exposition'
 * import { complexPlaneView } from 'mdml-render/views'
 * complexPlaneView(complexPlane({ domain: 's', poles: [[-1, 1], [-1, -1]] })).series.length // => 2
 * ```
 */
export function complexPlaneView(record: ComplexPlane, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.3 })
  const { real, imaginary } = complexPlaneLimits(record)
  return {
    ...cartesian(complexPlaneLayers(record, { ...options, figure: fig }), {
      mode: fig.mode,
      figure: fig,
      aspect: 'equal',
      x: { label: 'real', limits: real },
      y: { label: 'imaginary', limits: imaginary },
      legend: true,
    }),
    animation: false,
    ...ariaOf(
      `Complex plane (${record.domain}): ${record.poles.length} poles, ${record.poles.filter((p) => !isStablePole(record.domain, p)).length} of them unstable, and ${(record.zeros ?? []).length} zeros`,
    ),
  }
}
