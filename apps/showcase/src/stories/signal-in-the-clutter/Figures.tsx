import { cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Case } from './compute.ts'
import { RANGE } from './data.ts'
import { CLUTTER_ABOVE } from './reading.ts'

const X_AXIS = { label: 'the hidden value', limits: [RANGE[0], RANGE[1]] as [number, number] }

/** Heights of the dots in the strip under the curves, spread so that readings with equal values do not overlap exactly. */
const strip = (n: number, low: number, high: number): number[] =>
  Array.from({ length: n }, (_, i) => low + (high - low) * ((i * 0.6180339887) % 1))

/**
 * The readings alone, as dots on a number line (before anything has run).
 *
 * @param props - The readings.
 * @returns The chart.
 */
export function ReadingsFigure(props: { readonly values: readonly number[] }): ReactNode {
  const { values } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          points([{ x: values, y: strip(values.length, 0.1, 0.9), name: 'readings', colour: 'fgSecondary', size: 9 }], {
            mode,
          }),
        ],
        {
          mode,
          figure,
          x: X_AXIS,
          y: { limits: [0, 1], show: false, splitLine: false },
          legend: false,
          tooltip: false,
        },
      ),
    [values, mode],
  )
  return <ResponsiveChart aspect={4} narrowAspect={2.4} build={build} />
}

/**
 * One solved case: the readings as dots (grey where the model calls the reading clutter), the exact curve for the hidden value
 * as a thin line, the bell curve that expectation propagation settled on as a thick line with a light fill, and a dotted line at
 * the value the first twelve readings were made around.
 *
 * @param props - The case and the true value.
 * @returns The chart.
 */
export function SolutionFigure(props: { readonly result: Case; readonly truth: number }): ReactNode {
  const { result, truth } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const grid = result.exact.curve.x
      const bell = grid.map(
        (x) =>
          Math.exp(-0.5 * ((x - result.bell.mean) / result.bell.sd) ** 2) / (result.bell.sd * Math.sqrt(2 * Math.PI)),
      )
      const top = Math.max(...bell, ...result.exact.curve.density) * 1.12
      const height = strip(result.readings.length, -0.16 * top, -0.04 * top)
      const keep = result.readings.map((_, i) => i).filter((i) => (result.clutter[i] as number) <= CLUTTER_ABOVE)
      const drop = result.readings.map((_, i) => i).filter((i) => (result.clutter[i] as number) > CLUTTER_ABOVE)
      const dots = (indices: readonly number[], colour: string, name: string): ReturnType<typeof points> =>
        points(
          [
            {
              x: indices.map((i) => result.readings[i] as number),
              y: indices.map((i) => height[i] as number),
              name,
              colour,
              size: 9,
              opacity: 0.9,
            },
          ],
          { mode },
        )
      return cartesian(
        [
          lines([{ x: grid, y: bell, name: 'the bell curve', colour: 'tab1', width: 2.6, area: 0.14 }], { mode }),
          lines(
            [
              {
                x: grid,
                y: [...result.exact.curve.density],
                name: 'the exact curve',
                colour: 'fg',
                width: 1.4,
                opacity: 0.8,
              },
            ],
            { mode },
          ),
          lines(
            [
              {
                x: [truth, truth],
                y: [0, top],
                name: 'the true value',
                colour: 'fgSecondary',
                width: 1.4,
                dash: 'dotted',
              },
            ],
            {
              mode,
            },
          ),
          dots(keep, 'tab0', 'kept readings'),
          dots(drop, 'fgSecondary', 'readings called clutter'),
        ],
        {
          mode,
          figure,
          x: X_AXIS,
          y: {
            label: 'how likely each value is',
            limits: [-0.2 * top, top],
            format: (v: number) => (v < 0 ? '' : v.toFixed(1)),
            splitLine: false,
          },
          legend: false,
          tooltip: false,
        },
      )
    },
    [result, truth, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1.2} build={build} />
}
