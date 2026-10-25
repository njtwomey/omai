import { diabetes } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { randomForestRegressor } from 'mdml-engine/methods/ensemble'
import { explainPrediction } from 'mdml-engine/methods/explain'
import { key } from 'mdml-engine/random'
import { ControlPanel, CurvesView, NumberField, WaterfallView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Sampling SHAP converging',
  question: 'How do I show a sampled explanation settling as samples accrue?',
  explain:
    "Permutation sampling estimates Shapley values by averaging each feature’s marginal contribution over random feature orders. With `method: 'sampling'`, `explainPrediction` records the running estimate after each batch of permutations: `WaterfallView` plays it on fixed axes, so the bars settle, and a `curves` record traces the leading features’ estimates with their 95% bands, which narrow as one over the square root of the permutations.",
}

// region
const data = diabetes()
const forest = randomForestRegressor({ nEstimators: 30, maxDepth: 5 }).fit(data, { key: key(3) })

export default function SamplingShapRecipe(): ReactNode {
  const [permutations, setPermutations] = useState(400)
  const [seed, setSeed] = useState(1)
  const result = useMemo(
    () => explainPrediction(forest, data, 12, { method: 'sampling', nPermutations: permutations, key: key(seed) }),
    [permutations, seed],
  )
  const waterfall = recordsOf(result.exposition, 'featureAttributions')[0]
  const curves = recordsOf(result.exposition, 'curves')[0]
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField
          label="permutations"
          value={permutations}
          onChange={setPermutations}
          min={10}
          max={2000}
          integer
          log
        />
        <NumberField label="seed" value={seed} onChange={setSeed} min={0} max={999} integer />
      </ControlPanel>
      {waterfall !== undefined && <WaterfallView record={waterfall} startAtFirst />}
      {curves !== undefined && <CurvesView record={curves} />}
    </div>
  )
}
// endregion
