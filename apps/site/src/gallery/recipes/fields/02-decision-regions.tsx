import { type ReactNode, useMemo, useState } from 'react'
import { cartesian, decisionRegions, points } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Decision regions',
  question: 'How do I show a classifier’s regions, confidence and boundaries?',
  explain:
    '`decisionRegions(probabilities, { x, y })` takes one probability grid per class: each pixel gets the winning class’s colour, faded where the classifier is unsure, and the boundaries are the zero contours of the argmax margins. Here the probabilities are a softmax of squared distances to three centres; raising the sharpness makes the classifier surer.',
}

const fig = figure({ width: 480, aspect: 1.15 })
const centres: readonly (readonly [number, number])[] = [
  [-1, -0.6],
  [1.1, -0.4],
  [0, 1.2],
]
const n = 30
const xs = Array.from({ length: n }, (_, j) => -2.5 + (5 * j) / (n - 1))
function probabilities(sharpness: number): number[][][] {
  return centres.map((_, k) =>
    xs.map((y) =>
      xs.map((x) => {
        const s = centres.map(([cx, cy]) => Math.exp(-sharpness * ((x - cx) ** 2 + (y - cy) ** 2)))
        const total = s.reduce((a, b) => a + b, 0)
        return (s[k] ?? 0) / total
      }),
    ),
  )
}

export default function DecisionRegions(): ReactNode {
  const mode = useColourMode()
  const [sharpness, setSharpness] = useState(1)
  const [confidence, setConfidence] = useState(true)
  const option = useMemo(
    () =>
      cartesian(
        [
          decisionRegions(probabilities(sharpness), { x: [-2.5, 2.5], y: [-2.5, 2.5], mode, confidence }),
          points(
            [{ x: centres.map((c) => c[0]), y: centres.map((c) => c[1]), groups: [0, 1, 2], size: 11, shapes: true }],
            { mode },
          ),
        ],
        { mode, figure: fig, aspect: 'equal', legend: false },
      ),
    [mode, sharpness, confidence],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="sharpness" value={sharpness} onChange={setSharpness} min={0.1} max={10} log />
        <Toggle label="shade by confidence" value={confidence} onChange={setConfidence} />
      </ControlPanel>
      <FigureFrame figure={fig} filename="regions">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
