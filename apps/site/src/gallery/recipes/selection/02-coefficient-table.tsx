import { recordsOf } from 'omai-engine/exposition'
import { olsSummary } from 'omai-engine/linear'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { sparseRegression } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Coefficient table',
  question: 'How do I show a fitted model’s coefficients with how certain each is?',
  explain:
    'A `coefficientTable` record is an estimate per coefficient with its standard error, confidence interval and p-value. `CoefficientTableView` draws a forest of the estimates with their intervals, the zero line marked, and the p-values beside them; an interval that crosses zero is a coefficient the data cannot tell from nothing. `olsSummary` fits least squares to eight features of which three matter. Few rows or much noise widen the intervals until the true coefficients cross zero.',
  shows: ['coefficientTable', 'CoefficientTableView'],
}

export default function CoefficientTableRecipe(): ReactNode {
  const [noise, setNoise] = useState(2)
  const [n, setN] = useState(60)
  const [seed, setSeed] = useState(1)
  const record = useMemo(
    () => recordsOf(olsSummary(sparseRegression(seed, n, 8, noise)).exposition, 'coefficientTable')[0],
    [noise, n, seed],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="noise" value={noise} onChange={setNoise} min={0} max={8} step={0.5} />
        <NumberField label="rows" value={n} onChange={setN} min={15} max={500} step={5} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ExpositionView record={record} title="Estimates and their intervals" />}
    </div>
  )
}
