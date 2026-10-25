import { diabetes } from 'mdml-engine/datasets'
import { recordsOf } from 'mdml-engine/exposition'
import { standardScaler } from 'mdml-engine/learn/preprocessing'
import { elasticNetCV } from 'mdml-engine/methods/linear'
import { ControlPanel, NumberField, PathView } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Regularisation path',
  question: 'How do I show how a penalised model’s coefficients change with the penalty?',
  explain:
    'An elastic net chosen by cross-validation returns a `regularisationPath` record; `PathView` draws every coefficient against the penalty with the active-set size beneath, and dashes the chosen penalty. Lower the L1 share towards ridge and the coefficients stop dropping to zero; move the penalty to read the coefficients there.',
}

// region
const raw = diabetes()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }

export default function RegularisationPathRecipe(): ReactNode {
  const [l1Ratio, setL1Ratio] = useState(1)
  const [penalty, setPenalty] = useState<number | undefined>(undefined)
  const path = useMemo(() => {
    const fitted = elasticNetCV({ l1Ratio, nAlphas: 40 }).fit(data)
    return recordsOf(fitted.exposition, 'regularisationPath')[0]
  }, [l1Ratio])
  if (path === undefined) return null
  const [low, high] = [Math.min(...path.penalties), Math.max(...path.penalties)]
  const selected = Math.min(high, Math.max(low, penalty ?? path.chosen ?? low))
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="L1 share" value={l1Ratio} onChange={setL1Ratio} min={0.05} max={1} />
        <NumberField label="penalty" value={selected} onChange={setPenalty} min={low} max={high} log />
      </ControlPanel>
      <PathView record={path} selected={selected} />
    </div>
  )
}
// endregion
