import { type ReactNode, useMemo, useState } from 'react'
import { generate } from 'whitebox-engine/data'
import { heteroscedastic } from 'whitebox-engine/datasets'
import { recordsOf } from 'whitebox-engine/exposition'
import { kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { onlineConformal } from 'whitebox-engine/methods/trust'
import { key } from 'whitebox-engine/random'
import { ConformalView, ControlPanel, NumberField } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Coverage as data arrive',
  question: 'How does conformal prediction keep its coverage when the data drift?',
  explain:
    '`onlineConformal(estimator, data, { alpha, gamma, window })` fits once, then streams rows in order of x, where the noise keeps growing. A split conformal interval (its width fixed by the calibration rows) over-covers the quiet start and under-covers the noisy end. Adaptive conformal inference (Gibbs and Candès) moves its working level after every row, α_t+1 = α_t + γ(α − err_t), on a window of recent scores, so its running coverage stays near 1 − α. Press play to watch both running coverages as the rows arrive; a larger γ reacts faster but jitters.',
}

export default function OnlineConformalRecipe(): ReactNode {
  const [alpha, setAlpha] = useState(0.1)
  const [gamma, setGamma] = useState(0.02)
  const [window, setWindow] = useState(60)
  const data = useMemo(
    () => generate(heteroscedastic, { key: key(5), knobs: { n: 1000, base: 0.02, slope: 1 } }).data,
    [],
  )
  const record = useMemo(
    () =>
      recordsOf(
        onlineConformal(kNeighborsRegressor({ nNeighbors: 15 }), data, { alpha, gamma, window, key: key(2) })
          .exposition,
        'conformal',
      )[0],
    [data, alpha, gamma, window],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="α (miscoverage)" value={alpha} onChange={setAlpha} min={0.02} max={0.4} step={0.01} />
        <NumberField label="γ (step)" value={gamma} onChange={setGamma} min={0.001} max={0.2} step={0.001} />
        <NumberField label="window" value={window} onChange={setWindow} min={10} max={400} step={10} />
      </ControlPanel>
      {record !== undefined && <ConformalView record={record} />}
    </div>
  )
}
