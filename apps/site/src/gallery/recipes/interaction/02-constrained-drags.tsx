import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, lines } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import {
  alongAxis,
  Chart,
  DraggablePoints,
  FigureFrame,
  InteractionOverlay,
  Overlaid,
  onCircle,
  onGraph,
  onGrid,
  type Point2,
  useChartProjection,
  useColourMode,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Constrained drags',
  question: 'How do I keep a dragged point on a curve, a circle, a line or a grid?',
  explain:
    'A `Constraint` maps the point under the pointer to the nearest allowed one: `onGraph(f)` keeps it on y = f(x), `onCircle`, `onSegment` and `onPolyline` on shapes, `alongAxis` on a line, `onGrid` on a lattice, `inBox` in a range, and `chain` combines them. `constrainEach` gives each point its own rule.',
}

const FIGURE = figure()
const f = (x: number): number => Math.sin(x)
const rules = [onGraph(f, [-4, 4]), onCircle([0, 0], 2), alongAxis('x', -2), onGrid(1)]
const colours = ['var(--wb-tab0)', 'var(--wb-tab1)', 'var(--wb-tab2)', 'var(--wb-tab3)']
const xs = Array.from({ length: 201 }, (_, i) => -4 + (8 * i) / 200)
const ts = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)

export default function ConstrainedDrags(): ReactNode {
  const mode = useColourMode()
  const { projection, onInit } = useChartProjection()
  const [points, setPoints] = useState<Point2[]>([
    [1, f(1)],
    [0, 2],
    [-1, -2],
    [2, 1],
  ])
  const option = useMemo(
    () =>
      cartesian(
        [
          lines(
            [
              { x: xs, y: xs.map(f), name: 'on y = sin x' },
              { x: ts.map((t) => 2 * Math.cos(t)), y: ts.map((t) => 2 * Math.sin(t)), name: 'on the circle' },
              { x: [-4, 4], y: [-2, -2], name: 'along y = −2' },
            ],
            { mode },
          ),
        ],
        { mode, figure: FIGURE, aspect: 'equal', x: { limits: [-4, 4] }, y: { limits: [-2.5, 2.5] }, tooltip: false },
      ),
    [mode],
  )
  return (
    <FigureFrame figure={FIGURE} filename="constraints">
      <Overlaid
        width={FIGURE.width}
        height={FIGURE.height}
        overlay={
          projection === null ? null : (
            <InteractionOverlay projection={projection} width={FIGURE.width} height={FIGURE.height}>
              <DraggablePoints
                points={points}
                onChange={setPoints}
                constrainEach={(i) => rules[i]}
                colour={(i) => colours[i] ?? 'var(--wb-accent)'}
                labels={['on the sine curve', 'on the circle', 'along y = −2', 'on the grid']}
              />
            </InteractionOverlay>
          )
        }
      >
        <Chart option={option} figure={FIGURE} onInit={onInit} />
      </Overlaid>
    </FigureFrame>
  )
}
