import { type ReactNode, useMemo, useState } from 'react'
import { germanCredit } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { intersectionalAudit } from 'whitebox-engine/methods/fairness'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { toFlat } from 'whitebox-engine/tensor'
import { ControlPanel, NumberField, SubgroupView } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Subgroup report',
  question: 'How do I show a gap that hides in the intersection of two protected attributes?',
  explain:
    'A logistic regression on the German credit data is audited over sex and age band together. `intersectionalAudit(model, data, { attributes })` returns a `subgroupReport`: each subgroup’s table and rates, the largest gap between subgroups at least `minSize` large and the worst subgroup, the generalised entropy index of benefits split between and within subgroups, and a GerryFair audit of the linear subgroup of the attributes with the largest weighted false-positive gap. `SubgroupView` draws one rate as a heatmap over the attributes beside every subgroup’s rate; pick the rate beneath. Raise the age cut and the young subgroups grow; a subgroup below the minimum size is greyed and left out of the gap.',
}

const data = germanCredit()
const x = toFlat(data.x)
const age = Array.from({ length: 1000 }, (_, i) => x[i * 12 + 6] ?? 0)
const model = logisticRegression({ maxIter: 200 }).fit(data)

export default function SubgroupRecipe(): ReactNode {
  const [cut, setCut] = useState(26)
  const [minSize, setMinSize] = useState(30)
  const record = useMemo(() => {
    const band = age.map((a) => (a < cut ? `under ${cut}` : `${cut} and over`))
    return recordsOf(
      intersectionalAudit(model, data, { attributes: ['female', band], names: ['sex', 'age'], minSize }).exposition,
      'subgroupReport',
    )[0]
  }, [cut, minSize])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="age cut" value={cut} onChange={setCut} min={20} max={60} step={1} />
        <NumberField
          label="smallest subgroup counted"
          value={minSize}
          onChange={setMinSize}
          min={1}
          max={200}
          step={1}
        />
      </ControlPanel>
      {record !== undefined && <SubgroupView record={record} />}
    </div>
  )
}
