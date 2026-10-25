import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  DraggablePoints,
  FigureFrame,
  InteractionOverlay,
  inBox,
  Overlaid,
  type Point2,
  useChartProjection,
  useColourMode,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Click to add points',
  question: 'How do I let the reader place data points, move them and remove them?',
  explain:
    'With `add`, a press on empty space adds a point there (and it follows the pointer until release); with `remove`, a double click or the Delete key removes one; `max` caps the count. Here a least-squares line is refitted from the points as they change.',
}

const FIGURE = figure()

function leastSquares(points: readonly Point2[]): ((x: number) => number) | null {
  const n = points.length
  if (n < 2) return null
  const mx = points.reduce((a, p) => a + p[0], 0) / n
  const my = points.reduce((a, p) => a + p[1], 0) / n
  const sxx = points.reduce((a, p) => a + (p[0] - mx) ** 2, 0)
  const sxy = points.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0)
  if (sxx === 0) return null
  return (x) => my + (sxy / sxx) * (x - mx)
}

export default function ClickToAdd(): ReactNode {
  const mode = useColourMode()
  const { projection, onInit } = useChartProjection()
  const [points, setPoints] = useState<Point2[]>([
    [2, 3],
    [5, 4.5],
    [8, 7],
  ])
  const option = useMemo(() => {
    const fit = leastSquares(points)
    return cartesian(
      [
        lines(fit === null ? [] : [{ x: [0, 10], y: [fit(0), fit(10)], name: 'least squares' }], {
          mode,
          colourOffset: 1,
        }),
      ],
      { mode, figure: FIGURE, x: { label: 'x', limits: [0, 10] }, y: { label: 'y', limits: [0, 10] }, tooltip: false },
    )
  }, [points, mode])
  return (
    <div className="flex flex-col gap-2">
      <FigureFrame
        figure={FIGURE}
        filename="least-squares"
        caption={`${points.length} points. Click to add, drag to move, double-click to remove.`}
      >
        <Overlaid
          width={FIGURE.width}
          height={FIGURE.height}
          overlay={
            projection === null ? null : (
              <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
                <DraggablePoints
                  points={points}
                  onChange={setPoints}
                  constrain={inBox([0, 10], [0, 10])}
                  add
                  remove
                  max={30}
                />
              </InteractionOverlay>
            )
          }
        >
          <Chart option={option} figure={FIGURE} onInit={onInit} />
        </Overlaid>
      </FigureFrame>
    </div>
  )
}
