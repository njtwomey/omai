import type { BayesianGamFit, BayesianGamReference } from 'omai-engine/probprog/models'
import { band, bars, cartesian, type Layer, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { ColourMode } from 'omai-render/theme'
import type { CurveData } from '../../shared/curve-data.ts'

export const WIDE = figure({ width: 640, height: 380 })
export const NARROW = figure({ width: 420, height: 380 })
export const FG = '#e6edf3'
export const ORANGE = '#e8743b'
/** The curves of the posterior drawn with their parts. */
const SHOWN = 4

/** One colour per basis function, the same whatever the draw or the step. */
export const basisColour = (j: number, count: number): string =>
  `hsl(${Math.round((360 * j) / Math.max(1, count))} 65% 58%)`

interface FitChart {
  readonly fit: BayesianGamFit | undefined
  readonly curve: CurveData
  readonly truth: readonly number[] | undefined
  readonly reference: BayesianGamReference | undefined
  readonly bases: boolean
  readonly mode: ColourMode
}

/**
 * The data, the credible band, sampled curves and the posterior mean, each with its basis functions weighted by their
 * coefficients (one colour per basis, thin and translucent for a sampled curve) when `bases` is on, the centre of every basis
 * marked on the axis, the truth and the exact band when there are some.
 */
export function fitChart(o: FitChart): ReturnType<typeof cartesian> {
  const { fit, curve, mode } = o
  const layers: Layer[] = [
    points([{ x: [...curve.x], y: [...curve.y], name: 'data', opacity: 0.5 }], { mode, colourOffset: 7 }),
  ]
  if (o.reference !== undefined) {
    const { grid, lower, upper } = o.reference
    layers.push(
      lines(
        [
          { x: [...grid], y: [...lower], name: 'exact band', dash: 'dotted', width: 1.5, colour: FG },
          { x: [...grid], y: [...upper], name: '', dash: 'dotted', width: 1.5, colour: FG },
        ],
        { mode },
      ),
    )
  }
  const floor = Math.min(...curve.y)
  if (fit !== undefined) {
    const { grid, bases } = fit
    const p = bases.mean.length
    layers.push(
      band(
        [
          {
            x: [...grid],
            lower: [...fit.lower],
            upper: [...fit.upper],
            name: `${Math.round(fit.level * 100)}% band`,
            colour: ORANGE,
            opacity: 0.22,
          },
        ],
        { mode },
      ),
    )
    if (o.bases && bases.local)
      for (const parts of bases.curves.slice(0, SHOWN))
        layers.push(
          lines(
            parts.map((y, j) => ({
              x: [...grid],
              y: [...y],
              name: '',
              width: 1,
              opacity: 0.28,
              colour: basisColour(j, p),
            })),
            { mode },
          ),
        )
    layers.push(
      lines(
        fit.curves.map((c) => ({ x: [...grid], y: [...c], name: '', width: 1, opacity: 0.35, colour: ORANGE })),
        { mode },
      ),
    )
    if (o.bases && bases.local)
      layers.push(
        lines(
          bases.mean.map((y, j) => ({
            x: [...grid],
            y: [...y],
            name: j === 0 ? 'weighted bases' : '',
            width: 2,
            opacity: 0.9,
            colour: basisColour(j, p),
          })),
          { mode },
        ),
      )
    layers.push(lines([{ x: [...grid], y: [...fit.mean], name: 'posterior mean', width: 3, colour: ORANGE }], { mode }))
    layers.push(
      points(
        bases.centres.map((x, j) => ({
          x: [x],
          y: [floor],
          name: j === 0 ? 'basis centres' : '',
          colour: basisColour(j, p),
        })),
        { mode },
      ),
    )
  }
  if (o.truth !== undefined && fit !== undefined)
    layers.push(
      lines([{ x: [...fit.grid], y: [...o.truth], name: 'truth', dash: 'dashed', width: 2, colour: 'muted' }], {
        mode,
      }),
    )
  return cartesian(layers, { mode, figure: WIDE, x: { label: 'x', limits: [curve.lo, curve.hi] }, y: { label: 'y' } })
}

/** The posterior probability that each basis is in the slab, one bar per basis in the basis's colour. */
export function inclusionChart(inclusion: readonly number[], mode: ColourMode): ReturnType<typeof cartesian> {
  const p = inclusion.length
  const names = inclusion.map((_, j) => `${j + 1}`)
  return cartesian(
    [
      bars(
        names,
        inclusion.map((_, j) => ({
          name: '',
          colour: basisColour(j, p),
          values: inclusion.map((v, i) => (i === j ? v : 0)),
        })),
        { mode, stacked: true },
      ),
    ],
    {
      mode,
      figure: NARROW,
      x: { label: 'basis function' },
      y: { label: 'in use (slab probability, or ARD relevance)', limits: [0, 1] },
    },
  )
}

/** The convergence trace up to a step, with a marker at it. */
export function traceChart(
  trace: { label: string; steps: number[]; values: number[] },
  t: number,
  head: number,
  mode: ColourMode,
): ReturnType<typeof cartesian> {
  const upto = trace.steps.map((st, i) => [st, trace.values[i] ?? Number.NaN] as const).filter(([st]) => st <= t)
  const finite = trace.values.filter(Number.isFinite)
  return cartesian(
    [
      lines([{ x: upto.map((q) => q[0]), y: upto.map((q) => q[1]), name: trace.label, width: 2 }], { mode }),
      lines([{ x: [t, t], y: [Math.min(...finite), Math.max(...finite)], colour: FG, dash: 'dashed', width: 1 }], {
        mode,
      }),
    ],
    { mode, figure: NARROW, x: { label: 'step', limits: [0, Math.max(1, head)] }, y: { label: trace.label } },
  )
}
