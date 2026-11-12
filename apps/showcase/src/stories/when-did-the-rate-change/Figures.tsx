import { cartesian, histogram, lines } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Detection, Result } from './compute.ts'
import { FIRST_YEAR, SIMULATED } from './data.ts'

/** The years the figures span: from the first year to just after the last, so each bar is one year wide. */
const span = (n: number): [number, number] => [FIRST_YEAR, FIRST_YEAR + n]

/** The middle of each year, where a point standing for the whole year is placed. */
const middles = (n: number): number[] => Array.from({ length: n }, (_, i) => FIRST_YEAR + i + 0.5)

/**
 * The yearly counts as bars on a year axis; with a result, the rate the model expects in each year is drawn over them as a line
 * (the old rate before the switch, the new one after, and in between a blend that follows the doubt about the year).
 *
 * @param props - The counts, and the detection once there is one.
 * @returns The chart.
 */
export function SeriesFigure(props: {
  readonly counts: readonly number[]
  readonly detection: Detection | undefined
}): ReactNode {
  const { counts, detection } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const years = middles(counts.length)
      const bars = histogram(years, {
        weights: counts,
        bins: counts.length,
        range: span(counts.length),
        colour: 'tab0',
        opacity: 0.75,
        mode,
      })
      const layers =
        detection === undefined
          ? [bars]
          : [
              bars,
              lines([{ x: years, y: detection.expected, name: 'expected rate', colour: 'tab1', width: 3 }], { mode }),
            ]
      return cartesian(layers, {
        mode,
        figure,
        x: { label: 'year', limits: span(counts.length), format: (v: number) => String(Math.round(v)) },
        y: { label: 'disasters in the year', limits: [0, 7] },
        legend: false,
        tooltip: false,
      })
    },
    [counts, detection, mode],
  )
  return <ResponsiveChart aspect={2.3} narrowAspect={1.4} build={build} />
}

/**
 * The chance that the switch happened in each year, as bars on the same year axis as the series above it.
 *
 * @param props - The detection.
 * @returns The chart.
 */
export function WhenFigure(props: { readonly detection: Detection }): ReactNode {
  const { detection } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const n = detection.when.length
      return cartesian(
        [
          histogram(middles(n), {
            weights: detection.when,
            bins: n,
            range: span(n),
            density: false,
            colour: 'tab1',
            opacity: 0.85,
            mode,
          }),
        ],
        {
          mode,
          figure,
          x: { label: 'year', limits: span(n), format: (v: number) => String(Math.round(v)) },
          y: {
            label: 'chance the switch was this year',
            limits: [0, Math.max(0.1, 1.15 * Math.max(...detection.when))],
            format: 'percent',
          },
          legend: false,
          tooltip: false,
        },
      )
    },
    [detection, mode],
  )
  return <ResponsiveChart aspect={2.9} narrowAspect={1.6} build={build} />
}

/**
 * The same question asked of the two made-up series: the chance of the switch in each year for the abrupt change and for the
 * gentle one, with a dashed line at the year the rate really changed.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CompareFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const n = result.abrupt.when.length
      const x = Array.from({ length: n }, (_, i) => i + 1)
      const top = Math.max(...result.abrupt.when, ...result.gentle.when)
      return cartesian(
        [
          lines(
            [
              { x, y: result.abrupt.when, name: 'abrupt change', colour: 'tab1', width: 2.6, area: 0.2 },
              { x, y: result.gentle.when, name: 'gentle change', colour: 'tab2', width: 2.6, area: 0.2 },
              {
                x: [result.truth + 1, result.truth + 1],
                y: [0, 1.15 * top],
                name: 'when it really changed',
                colour: 'fgSecondary',
                dash: 'dashed' as const,
                width: 1.5,
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: {
            label: `year of the made-up series (the rate really changed in year ${SIMULATED.at + 1})`,
            limits: [25, 95],
          },
          y: { label: 'chance the switch was this year', limits: [0, 1.15 * top], format: 'percent' },
          legend: true,
          tooltip: false,
        },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={2.3} narrowAspect={1.4} build={build} />
}
