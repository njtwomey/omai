import { recordsOf } from 'mdml-engine/exposition'
import { gaussianProcessRegressor } from 'mdml-engine/methods/gp'
import { key, normal, uniform } from 'mdml-engine/random'
import { toFlat } from 'mdml-engine/tensor'
import { ControlPanel, HyperparameterSurfaceView, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Hyperparameter surface',
  question: 'How do I show the objective a method tuned its hyperparameters on, and the path it took?',
  explain:
    'A Gaussian process fits its kernel by maximising the log marginal likelihood, and returns that surface over its first two (log) hyperparameters, here the signal scale and the length scale, with the optimiser’s path, as a `hyperparameterSurface` record. More data sharpen the peak; more noise moves it towards longer length scales.',
}

// region
const xs = Array.from(toFlat(uniform(key(4), [60], { low: -4, high: 4 })))
const eps = Array.from(toFlat(normal(key(5), [60])))

export default function HyperparameterSurfaceRecipe(): ReactNode {
  const [n, setN] = useState(20)
  const [noise, setNoise] = useState(0.2)
  const surface = useMemo(() => {
    const data = {
      x: xs.slice(0, n).map((x) => [x]),
      y: xs.slice(0, n).map((x, i) => Math.sin(1.4 * x) + noise * (eps[i] ?? 0)),
    }
    const fitted = gaussianProcessRegressor({ alpha: 0.05, surfaceResolution: 30 }).fit(data, { key: key(0) })
    return recordsOf(fitted.exposition, 'hyperparameterSurface')[0]
  }, [n, noise])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="observations" value={n} onChange={setN} min={3} max={60} integer />
        <NumberField label="noise in the data" value={noise} onChange={setNoise} min={0} max={1} />
      </ControlPanel>
      {surface !== undefined && <HyperparameterSurfaceView record={surface} label="log marginal likelihood" />}
    </div>
  )
}
// endregion
