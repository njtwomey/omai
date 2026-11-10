import { cartesian, contours, heatmap, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import { ResponsiveChart, useColourMode } from 'omai-render/react'
import { type ReactNode, useCallback } from 'react'
import type { Problem } from './types.ts'

/**
 * The posterior predictive probability of the positive class over the plane for one method, as a diverging heat map around
 * one half with the 0.5 contour (the decision boundary) and the training points (filled) and the held-out points
 * (open) over it. The grid is the worker's, at the frame the playhead is on.
 *
 * @param props - The problem (the points and the rectangle), the grid `size * size` and a title.
 * @returns The chart.
 */
export function Boundary(props: {
  readonly problem: Problem
  readonly grid: ArrayLike<number>
  readonly size: number
  readonly title: string
}): ReactNode {
  const { problem, grid, size, title } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const { extent } = problem
      const axis = (lo: number, hi: number): number[] =>
        Array.from({ length: size }, (_, i) => lo + ((hi - lo) * i) / (size - 1))
      const z = Array.from({ length: size }, (_, j) =>
        Array.from({ length: size }, (_, i) => grid[j * size + i] ?? Number.NaN),
      )
      const xs = axis(extent.x[0], extent.x[1])
      const ys = axis(extent.y[0], extent.y[1])
      const split = (pts: Problem['train'], cls: number): { x: number[]; y: number[] } => ({
        x: pts.x.flatMap((r, i) => (pts.y[i] === cls ? [r[0] ?? 0] : [])),
        y: pts.x.flatMap((r, i) => (pts.y[i] === cls ? [r[1] ?? 0] : [])),
      })
      const dots = [
        { ...split(problem.train, 0), name: 'train 0', colour: 'tab0', size: 6, opacity: 0.9 },
        { ...split(problem.train, 1), name: 'train 1', colour: 'tab3', size: 6, opacity: 0.9 },
        { ...split(problem.test, 0), name: 'test 0', colour: 'tab0', size: 8, symbol: 'emptyCircle' },
        { ...split(problem.test, 1), name: 'test 1', colour: 'tab3', size: 8, symbol: 'emptyCircle' },
      ]
      return cartesian(
        [
          heatmap(z, { x: xs, y: ys, range: [0, 1], centre: 0.5, mode }),
          contours(z, { x: xs, y: ys, levels: [0.5], labels: false, colour: 'fg', width: 2, mode }),
          points(dots, { mode }),
        ],
        {
          mode,
          figure,
          title,
          legend: false,
          tooltip: false,
          x: { limits: [...extent.x] as [number, number] },
          y: { limits: [...extent.y] as [number, number] },
        },
      )
    },
    [problem, grid, size, title, mode],
  )
  return <ResponsiveChart aspect={1.05} narrowAspect={1} build={build} />
}
