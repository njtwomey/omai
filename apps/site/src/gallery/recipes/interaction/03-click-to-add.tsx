import { type ReactNode, useState } from 'react'
import { DraggablePoints, InteractionOverlay, inBox, Overlaid, type Point2 } from 'whitebox-render/react'
import { Line, Plot, plotFrame, sampleCurve } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Click to add points',
  question: 'How do I let the reader place data points, move them and remove them?',
  explain:
    'With `add`, a press on empty space adds a point there (and it follows the pointer until release); with `remove`, a double click or the Delete key removes one; `max` caps the count. Here a least-squares line is refitted from the points as they change.',
}

const frame = plotFrame({ x: [0, 10], y: [0, 10] })

function leastSquares(points: readonly Point2[]): ((x: number) => number) | null {
  const n = points.length
  if (n < 2) return null
  const mx = points.reduce((a, p) => a + p[0], 0) / n
  const my = points.reduce((a, p) => a + p[1], 0) / n
  const sxx = points.reduce((a, p) => a + (p[0] - mx) ** 2, 0)
  const sxy = points.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0)
  if (sxx === 0) return null
  const slope = sxy / sxx
  return (x) => my + slope * (x - mx)
}

export default function ClickToAdd(): ReactNode {
  const [points, setPoints] = useState<Point2[]>([
    [2, 3],
    [5, 4.5],
    [8, 7],
  ])
  const fit = leastSquares(points)
  return (
    <div className="flex flex-col gap-2">
      <Overlaid
        width={frame.fig.width}
        height={frame.fig.height}
        overlay={
          <InteractionOverlay projection={frame.projection} width={frame.fig.width} height={frame.fig.height}>
            <DraggablePoints
              points={points}
              onChange={setPoints}
              constrain={inBox([0, 10], [0, 10])}
              add
              remove
              max={30}
            />
          </InteractionOverlay>
        }
      >
        <Plot frame={frame} xLabel="x" yLabel="y">
          {fit === null ? null : <Line frame={frame} points={sampleCurve(fit, 0, 10, 2)} colour="var(--wb-tab1)" />}
        </Plot>
      </Overlaid>
      <p className="m-0 text-xs text-muted">
        {points.length} points. Click to add, drag to move, double-click to remove.
      </p>
    </div>
  )
}
