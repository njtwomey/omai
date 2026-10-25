import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { heteroscedastic } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { type ConformalRegressionMethod, conformalRegression } from 'whitebox-engine/methods/trust'
import { key } from 'whitebox-engine/random'
import { Choice, ConformalView, ControlPanel, NumberField, SeedInput } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Conformal intervals',
  question: 'How do I show prediction intervals with a coverage guarantee, and whether they keep it?',
  explain:
    '`conformalRegression(estimator, data, { alpha, method })` fits a k-nearest-neighbours regressor on 40% of the rows, scores 30% held out (absolute residuals, or residuals over a second model of their size) and widens every prediction by the scores’ conformal quantile; the guarantee is at least 1 − α coverage of new rows, whatever the model. `ConformalView` draws the remaining 30% with their intervals, rows outside marked, beside the calibration scores and their quantile. The noise here grows with x: split intervals have one width, so they over-cover the quiet left and under-cover the noisy right while averaging 1 − α; normalised scores adapt. Change the seed to see the coverage vary around its target.',
}

const METHODS = [
  { value: 'split', label: 'split' },
  { value: 'normalised', label: 'normalised' },
  { value: 'cv-plus', label: 'CV+' },
] as const

export default function ConformalIntervalsRecipe(): ReactNode {
  const [method, setMethod] = useState<ConformalRegressionMethod>('split')
  const [alpha, setAlpha] = useState(0.1)
  const [seed, setSeed] = useState(1)
  const data = useMemo(
    () => generate(heteroscedastic, { key: key(seed), knobs: { n: 600, base: 0.05, slope: 0.8 } }).data,
    [seed],
  )
  const record = useMemo(
    () =>
      recordsOf(
        conformalRegression(kNeighborsRegressor({ nNeighbors: 15 }), data, { alpha, method, key: key(seed) })
          .exposition,
        'conformal',
      )[0],
    [data, alpha, method, seed],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="method" value={method} onChange={setMethod} options={METHODS} />
        <NumberField label="α (miscoverage)" value={alpha} onChange={setAlpha} min={0.02} max={0.5} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ConformalView record={record} />}
    </div>
  )
}
