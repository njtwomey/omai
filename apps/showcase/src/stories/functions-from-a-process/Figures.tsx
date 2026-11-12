import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import { type Crowd, X_RANGE } from './compute.ts'

const X_AXIS = { label: 'x', limits: [X_RANGE[0], X_RANGE[1]] as [number, number] }
const Y_AXIS = { label: 'f(x)', limits: [-3, 3] as [number, number] }

/**
 * Functions on fixed axes: with nothing it is an empty frame (the axes already span what will be drawn, so the figure does not
 * rescale when the result arrives); with a process it draws the 90% band, the random functions drawn from it (thin), and, if
 * there is a mean, the mean (thick); with data, the observed points on top.
 *
 * @param props - The grid, the process to draw (the prior or the posterior), and the observed points, if they are to be shown.
 * @returns The chart.
 */
export function FunctionsFigure(props: {
  readonly grid?: readonly number[]
  readonly crowd?: Crowd
  readonly showMean?: boolean
  readonly data?: { readonly x: readonly number[]; readonly y: readonly number[] }
}): ReactNode {
  const { grid, crowd, showMean = false, data } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const common = { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false as const }
      if (grid === undefined || crowd === undefined) return cartesian([], common)
      const dots =
        data === undefined
          ? []
          : [
              points([{ x: data.x, y: data.y, groups: data.x.map(() => 'observed'), size: 12 }], {
                mode,
                colourOffset: 1,
              }),
            ]
      return cartesian(
        [
          band([{ x: grid, lower: crowd.lower, upper: crowd.upper, opacity: 0.18 }], { mode }),
          lines(
            crowd.draws.map((y) => ({ x: grid, y, width: 1.4, opacity: 0.55, colour: 'tab0' })),
            { mode },
          ),
          ...(showMean ? [lines([{ x: grid, y: crowd.mean, name: 'mean', width: 2.8 }], { mode })] : []),
          ...dots,
        ],
        common,
      )
    },
    [grid, crowd, showMean, data, mode],
  )
  return <ResponsiveChart aspect={1.9} narrowAspect={1} build={build} />
}
