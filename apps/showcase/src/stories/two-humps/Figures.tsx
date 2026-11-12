import { cartesian, histogram, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { WAIT_RANGE } from './data.ts'
import { SURE } from './reading.ts'

/** The width of a slot of the histogram, in minutes. */
const SLOT = 2

/** The ticks and limits of the axes, fixed so the figure does not rescale when the result arrives. */
const X_AXIS = { label: 'minutes until the next eruption', limits: [WAIT_RANGE[0], WAIT_RANGE[1]] as [number, number] }
const Y_AXIS = {
  label: 'eruptions in each two-minute slot',
  limits: [-5, 36] as [number, number],
  // the strip of dots sits below zero and has no scale
  format: (v: number): string => (v < 0 ? '' : String(v)),
  splitLine: false,
}

/** A repeatable scatter for the strip of dots: each dot's height is a fixed fraction, so the dots never overlap exactly. */
const strip = (waits: readonly number[]): number[] => waits.map((_, i) => -3.9 + 3.1 * ((i * 0.6180339887) % 1))

/**
 * The histogram of the waits with a strip of one dot per eruption beneath it. Before the run the dots are grey; with a result
 * the two fitted curves and their sum are drawn over the histogram, and each dot takes the colour of the group the eruption most
 * likely belongs to (grey where it is not clear).
 *
 * @param props - The waits, and the result once there is one.
 * @returns The chart.
 */
export function HumpsFigure(props: {
  readonly waits: readonly number[]
  readonly result: Result | undefined
}): ReactNode {
  const { waits, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const bars = histogram(waits, {
        bins: Math.round((100 - 40) / SLOT),
        range: [40, 100],
        colour: 'fgSecondary',
        opacity: 0.4,
        mode,
      })
      const height = strip(waits)
      const dots = (indices: readonly number[], colour: string, name: string): ReturnType<typeof points> =>
        points(
          [
            {
              x: indices.map((i) => waits[i] as number),
              y: indices.map((i) => height[i] as number),
              name,
              colour,
              size: 5,
              opacity: 0.85,
            },
          ],
          {
            mode,
          },
        )
      const all = waits.map((_, i) => i)
      if (result === undefined)
        return cartesian([bars, dots(all, 'fgSecondary', 'eruptions')], {
          mode,
          figure,
          x: X_AXIS,
          y: Y_AXIS,
          legend: false,
          tooltip: false,
        })
      // A curve is drawn as a count: its density times the number of eruptions and the width of a slot.
      const scale = result.rows * SLOT
      const counts = (density: readonly number[]): number[] => density.map((d) => d * scale)
      const lower = all.filter((i) => (result.chance[i] as number) <= 1 - SURE)
      const upper = all.filter((i) => (result.chance[i] as number) >= SURE)
      const unsure = all.filter((i) => (result.chance[i] as number) > 1 - SURE && (result.chance[i] as number) < SURE)
      return cartesian(
        [
          bars,
          lines(
            [{ x: result.grid, y: counts(result.lower), name: 'lower group', colour: 'tab0', width: 2, area: 0.18 }],
            { mode },
          ),
          lines(
            [{ x: result.grid, y: counts(result.upper), name: 'upper group', colour: 'tab1', width: 2, area: 0.18 }],
            { mode },
          ),
          lines([{ x: result.grid, y: counts(result.total), name: 'the two together', colour: 'fg', width: 2.6 }], {
            mode,
          }),
          dots(lower, 'tab0', 'surely lower'),
          dots(upper, 'tab1', 'surely upper'),
          dots(unsure, 'fgSecondary', 'could be either'),
        ],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      )
    },
    [waits, result, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}

/**
 * The two centres after each sweep of notes: both start at the average of all the waits, split almost at once, and then settle.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function SettlingFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          lines(
            [
              {
                x: result.trace.sweep,
                y: result.trace.lower,
                name: 'lower group',
                colour: 'tab0',
                width: 2.4,
                markers: true,
              },
              {
                x: result.trace.sweep,
                y: result.trace.upper,
                name: 'upper group',
                colour: 'tab1',
                width: 2.4,
                markers: true,
              },
            ],
            { mode },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'sweeps of notes', limits: [0, result.sweeps] },
          y: { label: 'centre of the group (minutes)', limits: [50, 85] },
          legend: false,
          tooltip: false,
        },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={2.1} narrowAspect={1.3} build={build} />
}
