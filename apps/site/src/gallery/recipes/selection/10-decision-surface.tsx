import { decisionSurface } from 'omai-engine/exposition'
import { kNeighboursClassifier } from 'omai-engine/learn'
import { toFlat } from 'omai-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { crescents } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Decision surface',
  question: 'How do I show where in the plane a classifier says each class, with the rows on it?',
  explain:
    'A `decisionSurface` record is a grid of predicted classes (or probabilities, or a value) over two features, with the training rows. `DecisionSurfaceView` paints the regions in class colours, draws the rows on them and rings the misclassified. Here a k-nearest-neighbours classifier is asked for its prediction at every grid point (`predict`), so k is a bias-variance dial: 1 chases every noisy label into islands, a large k draws smooth boundaries that cut across the crescents.',
  shows: ['decisionSurface', 'DecisionSurfaceView'],
}

export default function DecisionSurfaceRecipe(): ReactNode {
  const [k, setK] = useState(3)
  const [flip, setFlip] = useState(0.08)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const data = crescents(seed, 120, flip)
    const fitted = kNeighboursClassifier({ nNeighbors: k }).fit(data)
    const m = 60
    const xs = Array.from({ length: m }, (_, i) => -1.8 + (4.6 * i) / (m - 1))
    const ys = Array.from({ length: m }, (_, i) => -1.4 + (3.2 * i) / (m - 1))
    const grid = ys.flatMap((y) => xs.map((x) => [x, y]))
    const labels = Array.from(toFlat(fitted.predict(grid)))
    return decisionSurface({
      features: ['x', 'y'],
      x: xs,
      y: ys,
      labels: ys.map((_, i) => labels.slice(i * m, (i + 1) * m)),
      classes: ['upper crescent', 'lower crescent'],
      rows: { points: data.x, features: ['x', 'y'], labels: data.y, classes: ['upper crescent', 'lower crescent'] },
    })
  }, [k, flip, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="neighbours k" value={k} onChange={setK} min={1} max={40} integer />
        <NumberField label="labels flipped" value={flip} onChange={setFlip} min={0} max={0.3} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Regions of a nearest-neighbours classifier" />
    </div>
  )
}
