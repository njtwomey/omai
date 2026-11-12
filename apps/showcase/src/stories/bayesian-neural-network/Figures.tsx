import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { DRAW_RANGE, type Points } from './data.ts'

const X_AXIS = { label: 'x', limits: [DRAW_RANGE[0], DRAW_RANGE[1]] as [number, number] }
const Y_AXIS = { label: 'y', limits: [-2, 2] as [number, number] }

/** Which of the sampled networks is drawn on its own, as the one a single training run might have given. */
export const ONE_NETWORK = 7

/**
 * The scatter of the sixteen points on axes that already span the whole range the networks will be drawn over, so the figure does not
 * rescale when the result arrives; with a result, the same axes also show the band, the sampled networks (one of them picked out), their
 * mean and the points again on top.
 *
 * @param props - The points, and the result once there is one.
 * @returns The chart.
 */
export function NetworksFigure(props: { readonly data: Points; readonly result: Result | undefined }): ReactNode {
  const { data, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const dots = points([{ x: data.x, y: data.y, groups: data.x.map(() => 'the data') }], { mode, colourOffset: 1 })
      if (result === undefined)
        return cartesian([dots], { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false })
      const one = result.fan[Math.min(ONE_NETWORK, result.fan.length - 1)] ?? []
      return cartesian(
        [
          band([{ x: result.grid, lower: result.lower, upper: result.upper, opacity: 0.4 }], { mode }),
          lines(
            result.fan.map((y) => ({ x: result.grid, y, width: 1, opacity: 0.25, colour: 'tab0' })),
            { mode },
          ),
          lines([{ x: result.grid, y: result.mean, name: 'mean network', width: 2.6, colour: 'fgSecondary' }], {
            mode,
          }),
          lines([{ x: result.grid, y: one, name: 'one network', width: 2.2, colour: 'tab2' }], { mode }),
          dots,
        ],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      )
    },
    [data, result, mode],
  )
  return <ResponsiveChart aspect={1.5} narrowAspect={1.1} build={build} />
}
