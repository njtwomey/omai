import { band, cartesian, lines, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Result } from './compute.ts'
import { DRAW_RANGE, type Points, TRUTH } from './data.ts'

const X_AXIS = { label: 'x', limits: [DRAW_RANGE[0], DRAW_RANGE[1]] as [number, number] }
const Y_AXIS = { label: 'y', limits: [-3, 10] as [number, number] }

/**
 * The scatter of the twelve points on axes that already span the range the lines will cover, so the figure does not rescale when
 * the result arrives; with a result, the same axes also show the band, the sampled lines, their mean and the line the data came from.
 *
 * @param props - The points, and the result once there is one.
 * @returns The chart.
 */
export function LinesFigure(props: { readonly data: Points; readonly result: Result | undefined }): ReactNode {
  const { data, result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const dots = points([{ x: data.x, y: data.y, groups: data.x.map(() => 'the data') }], { mode, colourOffset: 1 })
      if (result === undefined)
        return cartesian([dots], { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false })
      const truth = {
        x: [DRAW_RANGE[0], DRAW_RANGE[1]],
        y: [TRUTH.intercept + TRUTH.slope * DRAW_RANGE[0], TRUTH.intercept + TRUTH.slope * DRAW_RANGE[1]],
        name: 'the line the data came from',
        colour: 'fgSecondary',
        dash: 'dashed' as const,
        width: 1.5,
      }
      return cartesian(
        [
          band([{ x: result.grid, lower: result.lower, upper: result.upper, opacity: 0.3 }], { mode }),
          lines(
            result.fan.map((y) => ({ x: result.grid, y, width: 1, opacity: 0.3, colour: 'tab0' })),
            { mode },
          ),
          lines([{ x: result.grid, y: result.mean, name: 'mean line', width: 2.6 }], { mode }),
          lines([truth], { mode }),
          dots,
        ],
        { mode, figure, x: X_AXIS, y: Y_AXIS, legend: false, tooltip: false },
      )
    },
    [data, result, mode],
  )
  return <ResponsiveChart aspect={1.55} narrowAspect={1.1} build={build} />
}

/**
 * The same crowd of lines as points: each dot is one sampled line, placed by its intercept and slope, with the true pair marked.
 *
 * @param props - The result.
 * @returns The chart.
 */
export function CloudFigure(props: { readonly result: Result }): ReactNode {
  const { result } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) =>
      cartesian(
        [
          points(
            [{ x: result.cloud.intercept, y: result.cloud.slope, groups: result.cloud.slope.map(() => 'a line') }],
            {
              mode,
            },
          ),
          points(
            [
              {
                x: [TRUTH.intercept],
                y: [TRUTH.slope],
                groups: ['the line the data came from'],
                sizes: [1],
                sizeRange: [13, 13],
                labels: ['the truth'],
                labelPosition: 'top',
              },
            ],
            { mode, colourOffset: 2 },
          ),
        ],
        {
          mode,
          figure,
          x: { label: 'intercept' },
          y: { label: 'slope' },
          legend: false,
          tooltip: false,
        },
      ),
    [result, mode],
  )
  return <ResponsiveChart aspect={1.55} narrowAspect={1.1} build={build} />
}
