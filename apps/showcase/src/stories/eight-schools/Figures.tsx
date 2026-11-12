import { cartesian, errorBars, histogram, lines } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { EFFECTS, ERRORS, NAMES } from './data.ts'

/** How far left of its school's tick the raw estimate is drawn, and how far right the pooled one (so they do not overlap). */
const SHIFT = 0.13

const letter = (v: number): string => (Number.isInteger(v) ? (NAMES[v] ?? '') : '')

const X_AXIS = {
  label: 'school',
  limits: [-1, NAMES.length] as [number, number],
  format: letter,
  ticks: { interval: 1 },
}
const Y_AXIS = { label: 'effect of the coaching, in SAT points', limits: [-30, 50] as [number, number] }

/**
 * The eight estimates with an error bar of one standard error each, on axes that already span where the pooled estimates will go
 * (so the figure does not rescale when the result arrives). With a result, each estimate has a line to its pooled estimate (the
 * dot, with the 90% interval of the school's effect), and the common mean is a dashed line.
 *
 * @param props - The result, once there is one.
 * @returns The chart.
 */
export function SchoolsFigure(props: { readonly result: Result | undefined }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const raw = errorBars(
        EFFECTS.map((y, j) => ({ x: j - SHIFT, y, yErr: ERRORS[j] ?? 0 })),
        { mode, colour: 'tab1', name: 'estimate', cap: 0.06 },
      )
      const common = { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false as const }
      if (result === undefined) return cartesian([raw], common)
      const pooled = errorBars(
        result.schools.map((s, j) => ({ x: j + SHIFT, y: s.pooled.mean, yErr: (s.pooled.hi - s.pooled.lo) / 2 })),
        { mode, colour: 'tab0', name: 'pooled', cap: 0.06 },
      )
      const pulls = lines(
        result.schools.map((s, j) => ({
          x: [j - SHIFT, j + SHIFT],
          y: [s.estimate, s.pooled.mean],
          colour: 'fgSecondary',
          width: 1.6,
          opacity: 0.8,
        })),
        { mode },
      )
      const mean = lines(
        [
          {
            x: [-1, NAMES.length],
            y: [result.mu.mean, result.mu.mean],
            colour: 'muted',
            dash: 'dashed' as const,
            width: 1.5,
          },
        ],
        { mode },
      )
      return cartesian([mean, pulls, raw, pooled], common)
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.7} narrowAspect={0.95} build={build} />
}

/**
 * The posterior of the spread between schools, tau, as a histogram of its draws. The axis stops at `limit`: draws beyond it are
 * not drawn.
 *
 * @param props - The result and where the axis stops.
 * @returns The chart.
 */
export function SpreadFigure(props: { readonly result: Result; readonly limit: number }): ReactNode {
  const { result, limit } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian([histogram(result.tauDraws, { bins: limit, range: [0, limit], density: true, mode })], {
        mode,
        figure,
        x: { label: 'tau, the spread of the true effects between schools', limits: [0, limit] },
        y: { label: 'density', show: false },
        legend: false,
        tooltip: false as const,
      }),
    [result, limit, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.3} build={build} />
}
