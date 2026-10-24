import type { EChartsOption } from 'echarts'
import type { DensityCurve, DensityCurves } from 'whitebox-engine/exposition'
import { referenceLines } from '../charts/functions.ts'
import { band, type Curve, lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { categoricalColour } from '../theme/palettes.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { short, spanOf, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link densityFrames}. */
export type DensityViewOptions = ViewOptions

// The style of a density by its role: the prior dashed and muted, the posterior filled in the accent, the rest plain.
function styleOf(c: DensityCurve, i: number): Pick<Curve, 'colour' | 'dash' | 'area' | 'width'> {
  switch (c.role) {
    case 'prior':
      return { colour: 'fgSecondary', dash: 'dashed', width: 1.5 }
    case 'posterior':
      return { colour: 'tab0', area: 0.22, width: 2 }
    case 'likelihood':
      return { colour: 'tab1', dash: 'dotted', width: 1.5 }
    default:
      return { colour: categoricalColour(i + 2), width: 1.5 }
  }
}

// The region under a density between two values, as a band down to zero.
function shaded(
  x: readonly number[],
  density: readonly number[],
  [lo, hi]: readonly [number, number],
  mode: ColourMode,
): Layer {
  const keep = x.flatMap((v, i) => (v >= lo && v <= hi ? [i] : []))
  return band(
    [
      {
        x: keep.map((i) => x[i] ?? 0),
        lower: keep.map(() => 0),
        upper: keep.map((i) => density[i] ?? 0),
        colour: 'tab0',
        opacity: 0.3,
        name: '__interval',
      },
    ],
    { mode },
  )
}

/**
 * Densities over one quantity as frames for a player: with steps, the posterior after each prefix of the data (filled,
 * its interval shaded and its estimate marked) over the prior (dashed), from no data to all of it; without, every
 * curve at once (the prior dashed, the posterior filled with its credible interval shaded) with the record's marks.
 * The axes span every frame, so the posterior visibly narrows.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The frames.
 * @example
 * ```ts
 * import { densityCurves } from 'whitebox-engine/exposition'
 * import { densityFrames } from 'whitebox-render/views'
 * const r = densityCurves({ parameter: 'p', x: [0, 0.5, 1], curves: [{ name: 'prior', density: [1, 1, 1], role: 'prior' }] })
 * densityFrames(r).count // => 1
 * ```
 */
export function densityFrames(record: DensityCurves, options: DensityViewOptions = {}): ViewFrames {
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const steps = record.steps ?? []
  const count = Math.max(1, steps.length)
  const [, top] = spanOf([...record.curves.flatMap((c) => c.density), ...steps.flatMap((s) => s.density)])
  const x = record.x
  const [x0, x1] = spanOf(x)
  const posterior = record.curves.find((c) => c.role === 'posterior')
  const option = (frame: number): EChartsOption => {
    const t = Math.min(Math.max(frame, 0), count - 1)
    const step = steps[t]
    const layers: Layer[] = []
    const curves: Curve[] = []
    for (const [i, c] of record.curves.entries()) {
      // While playing, the record's posterior gives way to the step's.
      if (step !== undefined && c.role === 'posterior') continue
      curves.push({ x, y: c.density, name: c.name, ...styleOf(c, i) })
    }
    if (step !== undefined) {
      curves.push({
        x,
        y: step.density,
        name: posterior?.name ?? 'posterior',
        ...styleOf({ name: '', density: [], role: 'posterior' }, 0),
      })
      if (step.interval !== undefined) layers.push(shaded(x, step.density, step.interval, mode))
    } else if (record.interval !== undefined && posterior !== undefined)
      layers.push(shaded(x, posterior.density, [record.interval.lower, record.interval.upper], mode))
    layers.push(lines(curves, { mode }))
    const marks = (record.marks ?? []).map((m) => ({ value: m.value, label: m.name }))
    if (step === undefined && marks.length > 0)
      layers.push(referenceLines({ x: marks }, { mode, colour: 'fgSecondary' }))
    if (step?.estimate !== undefined)
      layers.push(referenceLines({ x: [step.estimate] }, { mode, colour: 'accent', dash: 'solid' }))
    return cartesian(layers, {
      figure: fig,
      mode,
      x: { label: record.parameter, limits: [x0, x1] },
      y: { label: 'density', limits: [0, top > 0 ? top * 1.05 : 1] },
      tooltip: 'axis',
    })
  }
  return {
    count,
    label: (f) => {
      const s = steps[Math.min(Math.max(f, 0), count - 1)]
      if (s === undefined) return record.parameter
      return `${s.observed} observation${s.observed === 1 ? '' : 's'}${s.estimate === undefined ? '' : ` · mean ${short(s.estimate)}`}`
    },
    option,
  }
}
