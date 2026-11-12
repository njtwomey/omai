import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { DT } from './data.ts'

const X_LABEL = 'time'

/**
 * The readings alone, as dots against time (before anything has run).
 *
 * @param props - The readings.
 * @returns The chart.
 */
export function ReadingsFigure(props: { readonly readings: readonly number[] }): ReactNode {
  const { readings } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          points(
            [{ x: readings.map((_, i) => i * DT), y: readings, name: 'readings', colour: 'fgSecondary', size: 7 }],
            {
              mode,
            },
          ),
        ],
        {
          mode,
          figure,
          x: { label: X_LABEL, limits: [0, (readings.length - 1) * DT] },
          y: { label: 'position read by the sensor' },
          legend: false,
          tooltip: false,
        },
      ),
    [readings, mode],
  )
  return <ResponsiveChart aspect={2} narrowAspect={1.3} build={build} />
}

/**
 * The track after the run: the readings as dots, the 90 percent band and the mean of the position at every step (the band is
 * blue where there are readings and orange for the steps ahead), and the true positions as a thin line.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function TrackFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const n = result.readings.length
      const time = result.beliefs.map((_, i) => i * DT)
      const part = (from: number, to: number): { x: number[]; mean: number[]; lo: number[]; hi: number[] } => {
        const slice = result.beliefs.slice(from, to)
        return {
          x: time.slice(from, to),
          mean: slice.map((b) => b.mean),
          lo: slice.map((b) => b.lo),
          hi: slice.map((b) => b.hi),
        }
      }
      // the forecast starts at the last reading, so the two parts join
      const seen = part(0, n)
      const ahead = part(n - 1, result.beliefs.length)
      return cartesian(
        [
          band([{ x: seen.x, lower: seen.lo, upper: seen.hi, name: '__seen', colour: 'tab0', opacity: 0.22 }], {
            mode,
          }),
          band([{ x: ahead.x, lower: ahead.lo, upper: ahead.hi, name: '__ahead', colour: 'tab1', opacity: 0.22 }], {
            mode,
          }),
          lines([{ x: seen.x, y: seen.mean, name: 'estimated path', colour: 'tab0', width: 2.4 }], { mode }),
          lines([{ x: ahead.x, y: ahead.mean, name: 'prediction', colour: 'tab1', width: 2.4 }], { mode }),
          lines([{ x: time, y: [...result.truth], name: 'true position', colour: 'fg', width: 1.2, dash: 'dotted' }], {
            mode,
          }),
          points(
            [
              {
                x: time.slice(0, n),
                y: [...result.readings],
                name: 'readings',
                colour: 'fgSecondary',
                size: 6,
                opacity: 0.85,
              },
            ],
            {
              mode,
            },
          ),
        ],
        {
          mode,
          figure,
          x: { label: X_LABEL, limits: [0, (result.beliefs.length - 1) * DT] },
          y: { label: 'position' },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}
