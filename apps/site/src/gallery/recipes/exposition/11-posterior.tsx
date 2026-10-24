import { type ReactNode, useMemo, useState } from 'react'
import { recordsOf } from 'whitebox-engine/exposition'
import { gaussianProcessRegressor } from 'whitebox-engine/methods/gp'
import { key, normal, uniform } from 'whitebox-engine/random'
import { toFlat } from 'whitebox-engine/tensor'
import { ControlPanel, NumberField, PosteriorView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Posterior',
  question: 'How do I show a model’s predictive distribution and its uncertainty?',
  explain:
    'A Gaussian process (or a Bayesian linear model) on one input returns a `posteriorGrid` record: the mean, variance and a few samples on a grid, with the observations. `PosteriorView` draws the mean with bands of one and two standard deviations. Add observations and the bands pinch where the data are; raise the noise and they stop passing through them.',
}

// region
const xs = Array.from(toFlat(uniform(key(4), [40], { low: -4, high: 4 })))
const eps = Array.from(toFlat(normal(key(5), [40])))
const ys = xs.map((x, i) => Math.sin(1.4 * x) + 0.3 * x + 0.15 * (eps[i] ?? 0))

export default function PosteriorRecipe(): ReactNode {
  const [n, setN] = useState(8)
  const [noise, setNoise] = useState(0.01)
  const grid = useMemo(() => {
    const data = { x: xs.slice(0, n).map((x) => [x]), y: ys.slice(0, n) }
    const fitted = gaussianProcessRegressor({ alpha: noise, surfaceResolution: 0 }).fit(data, { key: key(0) })
    return recordsOf(fitted.exposition, 'posteriorGrid')[0]
  }, [n, noise])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="observations" value={n} onChange={setN} min={1} max={40} integer />
        <NumberField label="noise variance" value={noise} onChange={setNoise} min={1e-6} max={1} log />
      </ControlPanel>
      {grid !== undefined && <PosteriorView record={grid} />}
    </div>
  )
}
// endregion
