import { type ReactNode, useMemo } from 'react'
import { scatterChart } from 'whitebox-render/charts'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Scatter plots',
  question: 'How do I scatter points by class, with sizes and marker shapes?',
  explain:
    '`scatterChart` (or the `points` layer) makes one series per group in tab10 order. `shapes: true` varies the marker by group so classes never rely on colour alone, and `sizes` maps values to marker area.',
}

// Three deterministic clusters on circles, so the picture is the same on every visit.
const groups = [0, 1, 2]
const pts = groups.flatMap((g) =>
  Array.from({ length: 30 }, (_, i) => {
    const a = i * 2.39996
    const r = 0.25 + (0.6 * ((i * 37) % 30)) / 30
    return {
      g,
      x: Math.cos((2 * Math.PI * g) / 3) * 2 + r * Math.cos(a),
      y: Math.sin((2 * Math.PI * g) / 3) * 2 + r * Math.sin(a),
      s: r,
    }
  }),
)

export default function ScatterGroups(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      scatterChart(
        [
          {
            x: pts.map((p) => p.x),
            y: pts.map((p) => p.y),
            groups: pts.map((p) => p.g),
            groupNames: { 0: 'setosa-ish', 1: 'versicolor-ish', 2: 'virginica-ish' },
            sizes: pts.map((p) => p.s),
            sizeRange: [5, 14],
            shapes: true,
          },
        ],
        { mode, aspect: 'equal', x: { label: 'x₁' }, y: { label: 'x₂' } },
      ),
    [mode],
  )
  return (
    <FigureFrame filename="scatter">
      <Chart option={option} />
    </FigureFrame>
  )
}
