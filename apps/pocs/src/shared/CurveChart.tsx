import { type AxisSpec, type Curve, cartesian, lines, type PointSet, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'

/**
 * Curves (and optionally points) on one pair of axes, drawn to the width of the box they are in: the picture most pages
 * of this app need beside the library's own views. `curves` and `dots` must be stable (memoised) or the chart is rebuilt
 * on every render.
 *
 * @param props - The curves, the points, the axes and the aspect.
 * @returns The chart.
 */
export function CurveChart(props: {
  readonly curves: readonly Curve[]
  readonly dots?: readonly PointSet[]
  readonly x?: AxisSpec
  readonly y?: AxisSpec
  readonly aspect?: number
  readonly narrowAspect?: number
}): ReactNode {
  const { curves, dots, x, y, aspect = 1.7, narrowAspect = 1.3 } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian([lines(curves, { mode }), ...(dots === undefined ? [] : [points(dots, { mode, colourOffset: 4 })])], {
        mode,
        figure,
        tooltip: 'axis',
        ...(x === undefined ? {} : { x }),
        ...(y === undefined ? {} : { y }),
      }),
    [curves, dots, x, y, mode],
  )
  return <ResponsiveChart aspect={aspect} narrowAspect={narrowAspect} build={build} />
}
