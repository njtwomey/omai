import { permutationImportance } from 'mdml-engine/explain'
import { importances } from 'mdml-engine/exposition'
import { tensor } from 'mdml-engine/tensor'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { CREDIT_FEATURES, creditRows, creditScore, normals, regressor } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Feature importances',
  question: 'How do I show which features a model relies on, with how sure that is?',
  explain:
    'An `importances` record gives per feature the mean change in a metric when the feature is shuffled, its spread over repeats and every repeat itself; `ImportanceView` draws the means as bars sorted by size with each repeat a dot, so a bar whose dots straddle zero is not distinguishable from noise. The toy credit model is shuffled on held-out rows with a noisy target. Few repeats and the spreads widen; a bigger noise swamps the small features.',
  shows: ['importances', 'ImportanceView'],
}

export default function ImportancesRecipe(): ReactNode {
  const [repeats, setRepeats] = useState(8)
  const [noise, setNoise] = useState(1)
  const [seed, setSeed] = useState(7)
  const record = useMemo(() => {
    const x = creditRows(seed, 200)
    const eps = normals(seed + 100, 200, noise)
    const y = x.map((r, i) => creditScore(r) + (eps[i] ?? 0))
    const r = permutationImportance(
      regressor(creditScore, CREDIT_FEATURES.length),
      { x: tensor(x), y },
      {
        scoring: 'neg_mean_squared_error',
        nRepeats: repeats,
      },
    )
    return importances({
      method: 'permutation',
      features: CREDIT_FEATURES,
      mean: [...r.mean],
      spread: [...r.std],
      repeats: r.importances.map((v) => [...v]),
      metric: 'MSE',
      direction: 'lower',
      baseline: -r.baseline,
      on: 'held out',
    })
  }, [repeats, noise, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="repeats" value={repeats} onChange={setRepeats} min={2} max={30} integer />
        <NumberField label="noise on the target" value={noise} onChange={setNoise} min={0} max={5} step={0.25} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Permutation importance" />
    </div>
  )
}
