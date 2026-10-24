import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, lines } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import {
  Chart,
  DraggablePoints,
  FigureFrame,
  InteractionOverlay,
  Overlaid,
  type Point2,
  useChartProjection,
  useColourMode,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Drag points',
  question: 'How do I let the reader drag points and have the figure follow?',
  explain:
    'An `InteractionOverlay` lies over the chart with the chart’s own projection (`useChartProjection` reads it from the live `Chart`); `DraggablePoints` draws handles there and calls `onChange` with the moved points. The points are ordinary state, so the curve through them (a quadratic Bézier here) is drawn from the same state. Give the axes fixed limits so they hold still under the hand; handles take focus, and arrow keys nudge them.',
}

const FIGURE = figure()

function bezier([a, b, c]: readonly Point2[]): Point2[] {
  if (a === undefined || b === undefined || c === undefined) return []
  return Array.from({ length: 101 }, (_, i): Point2 => {
    const t = i / 100
    const u = 1 - t
    return [u * u * a[0] + 2 * u * t * b[0] + t * t * c[0], u * u * a[1] + 2 * u * t * b[1] + t * t * c[1]]
  })
}

export default function DragPoints(): ReactNode {
  const mode = useColourMode()
  const { projection, onInit } = useChartProjection()
  const [points, setPoints] = useState<Point2[]>([
    [1, 1],
    [5, 5.5],
    [9, 1.5],
  ])
  const option = useMemo(() => {
    const curve = bezier(points)
    return cartesian(
      [
        lines(
          [
            {
              x: points.map((p) => p[0]),
              y: points.map((p) => p[1]),
              name: 'control polygon',
              dash: 'dashed',
              width: 1,
              colour: 'muted',
            },
            { x: curve.map((p) => p[0]), y: curve.map((p) => p[1]), name: 'Bézier curve' },
          ],
          { mode },
        ),
      ],
      { mode, figure: FIGURE, x: { label: 'x', limits: [0, 10] }, y: { label: 'y', limits: [0, 6] }, tooltip: false },
    )
  }, [points, mode])
  return (
    <FigureFrame figure={FIGURE} filename="bezier">
      <Overlaid
        width={FIGURE.width}
        height={FIGURE.height}
        overlay={
          projection === null ? null : (
            <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
              <DraggablePoints points={points} onChange={setPoints} labels={['start', 'control', 'end']} />
            </InteractionOverlay>
          )
        }
      >
        <Chart option={option} figure={FIGURE} onInit={onInit} />
      </Overlaid>
    </FigureFrame>
  )
}
