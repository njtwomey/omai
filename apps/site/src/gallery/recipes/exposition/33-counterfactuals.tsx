import { type ReactNode, useMemo, useState } from 'react'
import { germanCredit } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { type CounterfactualMethod, explainCounterfactuals } from 'whitebox-engine/methods/explain'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { key } from 'whitebox-engine/random'
import { Choice, ControlPanel, CounterfactualView, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Counterfactual table',
  question: 'How do I show what would have to change for a model to decide otherwise?',
  explain:
    '`explainCounterfactuals` searches for rows near the applicant that the credit model would score the other way: growing spheres samples outward until it crosses the boundary and then undoes what it can, DiCE finds several diverse ones together, the prototype-guided search heads for a typical member of the other class, and CEM gives a pertinent negative (what would have to be added) and a pertinent positive (what must stay). The record is a `counterfactuals` record; `CounterfactualView` lays the original beside each, the changes coloured red (up) or blue (down) and the outcome beneath. The first feature is held fixed here, as an immutable attribute would be, and every feature kept whole, as the data’s codes and counts are.',
}

// region
const data = germanCredit()
const model = logisticRegression({ C: 1 }).fit(data)
const methods: CounterfactualMethod[] = ['growing spheres', 'dice', 'prototype', 'cem', 'wachter', 'face']

export default function CounterfactualRecipe(): ReactNode {
  const [row, setRow] = useState(1)
  const [method, setMethod] = useState<CounterfactualMethod>('growing spheres')
  const record = useMemo(
    () =>
      recordsOf(
        explainCounterfactuals(model, data, row, {
          method,
          count: 3,
          constraints: { immutable: [0], integer: data.featureNames.map((_, j) => j) },
          key: key(2),
        }).exposition,
        'counterfactuals',
      )[0],
    [row, method],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="applicant (row)" value={row} onChange={setRow} min={0} max={999} integer />
        <Choice label="method" value={method} onChange={setMethod} options={methods} />
      </ControlPanel>
      {record !== undefined && <CounterfactualView record={record} />}
    </div>
  )
}
// endregion
