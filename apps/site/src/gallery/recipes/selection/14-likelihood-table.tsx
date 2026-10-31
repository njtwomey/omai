import { likelihoodTable } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Likelihood contributions',
  question: 'How do I show which features pushed a naive Bayes classifier towards one class?',
  explain:
    'A `likelihoodTable` record has, for one row, each class’s log prior and each feature’s log-likelihood contribution under that class. `LikelihoodView` draws a heatmap of the log prior and each feature’s contribution per class, and beside it the posterior probability of each class; the class whose row sums highest wins, and the cells show which features carried it. Here three classes of flower have made-up Gaussian feature models; the sliders set the row’s petal length and width, and moving them moves the winner and, within it, which features are responsible.',
  shows: ['likelihoodTable', 'LikelihoodView'],
}

const CLASSES = ['setosa', 'versicolor', 'virginica']
const FEATURES = ['sepal L', 'sepal W', 'petal L', 'petal W']
// Per class and feature: mean and standard deviation.
const MODEL = [
  [
    [5.0, 0.35],
    [3.4, 0.38],
    [1.5, 0.17],
    [0.25, 0.1],
  ],
  [
    [5.9, 0.5],
    [2.8, 0.3],
    [4.3, 0.47],
    [1.3, 0.2],
  ],
  [
    [6.6, 0.6],
    [3.0, 0.32],
    [5.5, 0.55],
    [2.0, 0.27],
  ],
] as const
const logNormal = (v: number, [mu, sd]: readonly [number, number]): number =>
  -0.5 * Math.log(2 * Math.PI * sd * sd) - (v - mu) ** 2 / (2 * sd * sd)

export default function LikelihoodTableRecipe(): ReactNode {
  const [petalLength, setPetalLength] = useState(4.6)
  const [petalWidth, setPetalWidth] = useState(1.5)
  const record = useMemo(() => {
    const row = [5.9, 3.0, petalLength, petalWidth]
    return likelihoodTable({
      classes: CLASSES,
      features: FEATURES,
      logPriors: CLASSES.map(() => Math.log(1 / 3)),
      contributions: MODEL.map((c) => c.map((p, j) => logNormal(row[j] ?? 0, p as [number, number]))),
      scope: { kind: 'query' },
    })
  }, [petalLength, petalWidth])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="petal length" value={petalLength} onChange={setPetalLength} min={0.5} max={7} step={0.1} />
        <NumberField label="petal width" value={petalWidth} onChange={setPetalWidth} min={0.1} max={3} step={0.1} />
      </ControlPanel>
      <ExpositionView record={record} title="Evidence for each class" />
    </div>
  )
}
