import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo } from 'react'
import type { GroupEstimates } from '../examples/index.ts'

const FIGURE = figure({ width: 520, height: 320 })

/**
 * The shrinkage of a hierarchical model's groups: each group's raw estimate (a diamond) joined to its posterior mean (a dot,
 * with its interval) at its place on the axis that measures how much data stands behind it, the population mean as a dashed
 * line and, for generated data, the true effect (a cross). A group moves from its raw estimate towards the line, and the
 * ones with the least data move furthest.
 */
export function GroupEstimatesView(props: {
  readonly record: GroupEstimates
  /** Whether the x axis is on a log scale (counts of houses). */
  readonly logX?: boolean
}): ReactNode {
  const { record: r, logX = false } = props
  const mode = useColourMode()
  const option = useMemo(() => {
    const gap = (f: (j: number) => number[]): number[] => r.x.flatMap((_, j) => [...f(j), Number.NaN])
    const x = Math.min(...r.x)
    const x1 = Math.max(...r.x)
    const layers = [
      ...(r.population === null
        ? []
        : [
            lines(
              [
                {
                  x: [x, x1],
                  y: [r.population, r.population],
                  name: 'population mean',
                  dash: 'dashed',
                  width: 1.5,
                  colour: 'muted',
                },
              ],
              { mode },
            ),
          ]),
      lines(
        [
          {
            x: gap((j) => [r.x[j] as number, r.x[j] as number]),
            y: gap((j) => [r.raw[j] as number, r.mean[j] as number]),
            width: 1,
            colour: 'muted',
            name: 'moved',
          },
        ],
        { mode },
      ),
      lines(
        [
          {
            x: gap((j) => [r.x[j] as number, r.x[j] as number]),
            y: gap((j) => [r.lo[j] as number, r.hi[j] as number]),
            width: 3,
            opacity: 0.55,
            colour: 'tab0',
            name: '90% interval',
          },
        ],
        { mode },
      ),
      points(
        [
          { x: r.x, y: r.raw, name: 'raw estimate', symbol: 'diamond', colour: 'tab1', size: 8, outline: true },
          { x: r.x, y: r.mean, name: 'posterior mean', symbol: 'circle', colour: 'tab0', size: 7 },
          ...(r.truth === null
            ? []
            : [{ x: r.x, y: r.truth, name: 'true effect', symbol: 'cross', colour: 'fg', size: 7 }]),
        ],
        { mode },
      ),
    ]
    return cartesian(layers, {
      mode,
      figure: FIGURE,
      x: { label: r.xLabel, ...(logX ? { scale: 'log' as const } : {}) },
      y: { label: r.yLabel },
      legend: true,
    })
  }, [r, logX, mode])
  return <Chart option={option} figure={FIGURE} />
}
