import { type ReactNode, useState } from 'react'
import {
  alongAxis,
  DraggablePoints,
  InteractionOverlay,
  Overlaid,
  onCircle,
  onGraph,
  onGrid,
  type Point2,
} from 'whitebox-render/react'
import { Line, Plot, plotFrame, sampleCurve } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Constrained drags',
  question: 'How do I keep a dragged point on a curve, a circle, a line or a grid?',
  explain:
    'A `Constraint` maps the point under the pointer to the nearest allowed one: `onGraph(f)` keeps it on y = f(x), `onCircle`, `onSegment` and `onPolyline` on shapes, `alongAxis` on a line, `onGrid` on a lattice, `inBox` in a range, and `chain` combines them. `constrainEach` gives each point its own rule.',
}

const frame = plotFrame({ x: [-4, 4], y: [-2.5, 2.5] })
const f = (x: number): number => Math.sin(x)
const rules = [onGraph(f, [-4, 4]), onCircle([0, 0], 2), alongAxis('x', -2), onGrid(1)]
const circle: Point2[] = sampleCurve((t) => t, 0, 2 * Math.PI).map(([t]) => [2 * Math.cos(t), 2 * Math.sin(t)])
const colours = ['var(--wb-tab0)', 'var(--wb-tab1)', 'var(--wb-tab2)', 'var(--wb-tab3)']

export default function ConstrainedDrags(): ReactNode {
  const [points, setPoints] = useState<Point2[]>([
    [1, f(1)],
    [0, 2],
    [-1, -2],
    [2, 1],
  ])
  return (
    <Overlaid
      width={frame.fig.width}
      height={frame.fig.height}
      overlay={
        <InteractionOverlay projection={frame.projection} width={frame.fig.width} height={frame.fig.height}>
          <DraggablePoints
            points={points}
            onChange={setPoints}
            constrainEach={(i) => rules[i]}
            colour={(i) => colours[i] ?? 'var(--wb-accent)'}
            labels={['on the sine curve', 'on the circle', 'along y = −2', 'on the grid']}
          />
        </InteractionOverlay>
      }
    >
      <Plot frame={frame} xLabel="x" yLabel="y">
        <Line frame={frame} points={sampleCurve(f, -4, 4)} colour="var(--wb-tab0)" width={1.5} />
        <Line frame={frame} points={circle} colour="var(--wb-tab1)" width={1.5} />
        <Line
          frame={frame}
          points={[
            [-4, -2],
            [4, -2],
          ]}
          colour="var(--wb-tab2)"
          width={1.5}
        />
      </Plot>
    </Overlaid>
  )
}
