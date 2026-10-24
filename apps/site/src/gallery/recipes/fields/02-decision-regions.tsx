import { type ReactNode, useMemo } from 'react'
import { cartesian, decisionRegions, points } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Decision regions',
  question: 'How do I show a classifier’s regions, confidence and boundaries?',
  explain:
    '`decisionRegions(probabilities, { x, y })` takes one probability grid per class: each pixel gets the winning class’s colour, faded where the classifier is unsure, and the boundaries are the zero contours of the argmax margins. Here the probabilities are a softmax of distances to three centres.',
}

const fig = figure({ width: 480, aspect: 1.15 })
const centres: readonly (readonly [number, number])[] = [
  [-1, -0.6],
  [1.1, -0.4],
  [0, 1.2],
]
const n = 30
const xs = Array.from({ length: n }, (_, j) => -2.5 + (5 * j) / (n - 1))
const probs = centres.map((_, k) =>
  xs.map((y) =>
    xs.map((x) => {
      const s = centres.map(([cx, cy]) => Math.exp(-2 * ((x - cx) ** 2 + (y - cy) ** 2)))
      const total = s.reduce((a, b) => a + b, 0)
      return (s[k] ?? 0) / total
    }),
  ),
)

export default function DecisionRegions(): ReactNode {
  const mode = useColourMode()
  const option = useMemo(
    () =>
      cartesian(
        [
          decisionRegions(probs, { x: [-2.5, 2.5], y: [-2.5, 2.5], mode }),
          points(
            [{ x: centres.map((c) => c[0]), y: centres.map((c) => c[1]), groups: [0, 1, 2], size: 11, shapes: true }],
            { mode },
          ),
        ],
        { mode, figure: fig, aspect: 'equal', legend: false },
      ),
    [mode],
  )
  return (
    <FigureFrame figure={fig} filename="regions">
      <Chart option={option} />
    </FigureFrame>
  )
}
