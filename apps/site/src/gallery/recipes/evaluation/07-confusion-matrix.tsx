import { confusionMatrix } from 'omai-engine/metrics'
import { key, uniform } from 'omai-engine/random'
import { toArray } from 'omai-engine/tensor'
import { confusionMatrixChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, Toggle, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Confusion matrix',
  question: 'How do I draw a multiclass confusion matrix, as counts or as rates?',
  explain:
    '`confusionMatrix(y, yPred)` counts true against predicted labels and `confusionMatrixChart(counts, labels)` draws them with each cell annotated; `normalise` turns rows into rates (each true class sums to one). The toy classifier mistakes each class for a neighbour more often than for a distant one, which shows as a band beside the diagonal.',
}

const FIGURE = figure({ width: 560, height: 500 })
const CLASSES = ['cat', 'lynx', 'fox', 'wolf', 'dog']

export default function ConfusionMatrixRecipe(): ReactNode {
  const mode = useColourMode()
  const [error, setError] = useState(0.3)
  const [normalise, setNormalise] = useState(false)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const n = 1000
    const u = toArray(uniform(key(seed), [3 * n])) as number[]
    const truth = Array.from({ length: n }, (_, i) => Math.floor((u[3 * i] ?? 0) * CLASSES.length))
    const pred = truth.map((c, i) => {
      if ((u[3 * i + 1] ?? 1) >= error) return c
      // A mistake: usually a neighbouring class, sometimes any.
      const step = (u[3 * i + 2] ?? 0) < 0.8 ? ((u[3 * i + 2] ?? 0) < 0.4 ? -1 : 1) : 2
      return (c + step + CLASSES.length) % CLASSES.length
    })
    const m = confusionMatrix(
      truth.map((c) => CLASSES[c] ?? ''),
      pred.map((c) => CLASSES[c] ?? ''),
      { labels: CLASSES },
    )
    return confusionMatrixChart(m.matrix, CLASSES, { mode, normalise })
  }, [error, normalise, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="error rate" value={error} onChange={setError} min={0} max={0.9} step={0.05} />
        <Toggle label="rates by true class" value={normalise} onChange={setNormalise} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="confusion-matrix">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
