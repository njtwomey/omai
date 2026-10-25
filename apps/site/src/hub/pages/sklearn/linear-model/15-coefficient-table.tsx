import { diabetes } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { standardScaler } from 'mdml-engine/learn/preprocessing'
import { olsSummary } from 'mdml-engine/methods/linear'
import { CoefficientTableView, ControlPanel, NumberField } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../../../gallery/recipe.ts'

export const recipe: Recipe = {
  title: 'Coefficient table',
  question: 'How do I show estimated coefficients with their uncertainty?',
  explain:
    'A least-squares summary (statsmodels’ OLS) returns a `coefficientTable` record: estimates, standard errors, intervals and p-values. `CoefficientTableView` draws the estimates as a forest plot, zero dashed, above the table. Widen the interval level and more of them cross zero.',
}

// region
const raw = diabetes()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }

export default function CoefficientTableRecipe(): ReactNode {
  const [level, setLevel] = useState(0.95)
  const table = useMemo(() => recordsOf(olsSummary(data, { level }).exposition, 'coefficientTable')[0], [level])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="interval level" value={level} onChange={setLevel} min={0.5} max={0.999} />
      </ControlPanel>
      {table !== undefined && <CoefficientTableView record={table} />}
    </div>
  )
}
// endregion
