import { cartesian, contours, heatmap, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { categoricalColour, mixColours, tokens } from 'omai-render/theme'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { EXTENT, type Points } from './data.ts'
import { UNSURE } from './reading.ts'

const AXES = {
  x: { label: 'first coordinate', limits: [EXTENT.x[0], EXTENT.x[1]] as [number, number] },
  y: { label: 'second coordinate', limits: [EXTENT.y[0], EXTENT.y[1]] as [number, number] },
}

/** The two classes' points as one layer, in the colours of the two ends of the probability ramp. */
const dots = (p: Points, mode: 'light' | 'dark'): ReturnType<typeof points> =>
  points(
    [
      {
        x: p.x.map((q) => q[0]),
        y: p.x.map((q) => q[1]),
        groups: p.y,
        groupNames: { 0: 'class 1', 1: 'class 2' },
        size: 9,
        outline: true,
      },
    ],
    { mode },
  )

/**
 * The points alone, in the plane, coloured by class (before anything has run).
 *
 * @param props - The points.
 * @returns The chart.
 */
export function PointsFigure(props: { readonly points: Points }): ReactNode {
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => cartesian([dots(props.points, mode)], { mode, figure, ...AXES, legend: true, tooltip: false }),
    [props.points, mode],
  )
  return <ResponsiveChart aspect={1.35} narrowAspect={1.05} build={build} />
}

/**
 * The probability map: the chance of the second class at each place in the plane, from blue (the first class) through a neutral
 * middle (a half) to orange (the second), with a solid line where the chance is a half and dashed lines at the edges of the
 * unsure band, and the points on top.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function MapFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const middle = mixColours(tokens(mode).bg, tokens(mode).fgSecondary, 0.25)
      return cartesian(
        [
          heatmap(result.probability as number[][], {
            x: [...result.xs],
            y: [...result.ys],
            ramp: [categoricalColour(0), middle, categoricalColour(1)],
            range: [0, 1],
            colourBar: false,
            mode,
          }),
          contours(result.probability as number[][], {
            x: [...result.xs],
            y: [...result.ys],
            levels: [0.5],
            colour: 'fg',
            width: 2.4,
            labels: false,
            mode,
          }),
          contours(result.probability as number[][], {
            x: [...result.xs],
            y: [...result.ys],
            levels: [...UNSURE],
            colour: 'fg',
            width: 1,
            dash: 'dashed',
            opacity: 0.7,
            labels: false,
            mode,
          }),
          dots(result.points, mode),
        ],
        { mode, figure, ...AXES, legend: false, tooltip: false },
      )
    },
    [result, mode],
  )
  return <ResponsiveChart aspect={1.2} narrowAspect={0.95} build={build} />
}

/**
 * The doubt map: how unsure the model is about the hidden score at each place (the standard deviation of the score), darker
 * where it is surer, with the points on top.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function DoubtFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          heatmap(result.doubt as number[][], {
            x: [...result.xs],
            y: [...result.ys],
            ramp: 'blues',
            colourBar: false,
            mode,
          }),
          dots(result.points, mode),
        ],
        { mode, figure, ...AXES, legend: false, tooltip: false },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={1.2} narrowAspect={0.95} build={build} />
}
