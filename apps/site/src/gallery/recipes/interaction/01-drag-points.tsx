import { type ReactNode, useState } from 'react'
import { DraggablePoints, InteractionOverlay, Overlaid, type Point2 } from 'whitebox-render/react'
import { Line, Plot, plotFrame } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Drag points',
  question: 'How do I let the reader drag points and have the figure follow?',
  explain:
    'An `InteractionOverlay` lies over the figure with the figure’s projection (data to pixels); `DraggablePoints` draws handles there and calls `onChange` with the moved points. The points are ordinary state, so the curve through them (a quadratic Bézier here) is drawn from the same state. Handles take focus; arrow keys nudge them.',
}

const frame = plotFrame({ x: [0, 10], y: [0, 6] })

function bezier([a, b, c]: readonly Point2[]): Point2[] {
  if (a === undefined || b === undefined || c === undefined) return []
  return Array.from({ length: 101 }, (_, i): Point2 => {
    const t = i / 100
    const u = 1 - t
    return [u * u * a[0] + 2 * u * t * b[0] + t * t * c[0], u * u * a[1] + 2 * u * t * b[1] + t * t * c[1]]
  })
}

export default function DragPoints(): ReactNode {
  const [points, setPoints] = useState<Point2[]>([
    [1, 1],
    [5, 5.5],
    [9, 1.5],
  ])
  return (
    <Overlaid
      width={frame.fig.width}
      height={frame.fig.height}
      overlay={
        <InteractionOverlay projection={frame.projection} width={frame.fig.width} height={frame.fig.height}>
          <DraggablePoints points={points} onChange={setPoints} labels={['start', 'control', 'end']} />
        </InteractionOverlay>
      }
    >
      <Plot frame={frame} xLabel="x" yLabel="y">
        <Line frame={frame} points={points} colour="var(--wb-muted)" width={1} dashed />
        <Line frame={frame} points={bezier(points)} />
      </Plot>
    </Overlaid>
  )
}
