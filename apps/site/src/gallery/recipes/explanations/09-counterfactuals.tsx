import { diverseCounterfactuals, medianAbsoluteDeviations, wachterCounterfactual } from 'mdml-engine/explain'
import { changedFeatures, counterfactualSet } from 'mdml-engine/exposition'
import { matmul, sigmoid, tensor, toFlat, type Value } from 'mdml-engine/tensor'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput, Toggle } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { CREDIT_FEATURES, creditRows } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Counterfactuals',
  question: 'How do I show what would have to change for a different decision?',
  explain:
    'A `counterfactuals` record holds the original row and its prediction, and for each counterfactual the new row, its prediction, its distance, whether it reaches the desired outcome and which features changed. `CounterfactualView` lays them out as a table with the original first and the changed cells marked. `wachterCounterfactual` finds the single closest point (with some features frozen); `diverseCounterfactuals` finds several different ones jointly. Freeze age and tenure and the change moves to what the applicant can act on.',
  shows: ['counterfactuals', 'CounterfactualView'],
}

const W = [0.9, -0.7, 0.15, 0.25, -0.8, 0.5]
const proba = (v: Value): Value =>
  sigmoid(
    matmul(
      v,
      W.map((w) => [w]),
    ),
  )
const IMMUTABLE = [2, 3]

export default function CounterfactualsRecipe(): ReactNode {
  const [method, setMethod] = useState<'closest' | 'diverse'>('diverse')
  const [count, setCount] = useState(4)
  const [freeze, setFreeze] = useState(true)
  const [seed, setSeed] = useState(6)
  const record = useMemo(() => {
    const rows = creditRows(seed, 100)
    // Someone the model turns down: the first applicant whose probability is under 0.3.
    const x = rows.find((r) => r.reduce((a, v, j) => a + v * (W[j] ?? 0), 0) < -0.9) ?? rows[0] ?? []
    const original = toFlat(proba(tensor([x])))[0] ?? 0
    const scales = medianAbsoluteDeviations(tensor(rows))
    const points: { x: readonly number[]; prediction: number; valid: boolean }[] = []
    if (method === 'closest') {
      const r = wachterCounterfactual(proba, x, {
        target: 0.65,
        tolerance: 0.03,
        scales,
        ...(freeze ? { immutable: IMMUTABLE } : {}),
      })
      points.push(r)
    } else {
      const r = diverseCounterfactuals(proba, x, { count, desired: 1, scales })
      for (const [i, p] of r.points.entries())
        points.push({ x: p, prediction: r.predictions[i] ?? 0, valid: r.valid[i] ?? false })
    }
    return counterfactualSet({
      method: method === 'closest' ? 'Wachter' : 'DiCE',
      features: CREDIT_FEATURES,
      output: 'P(repays)',
      original: x,
      originalPrediction: original,
      desired: 'approved',
      counterfactuals: points.map((p) => ({
        x: p.x,
        prediction: p.prediction,
        valid: p.valid,
        distance: p.x.reduce((a, v, j) => a + Math.abs(v - (x[j] ?? 0)) / (scales[j] ?? 1), 0),
        changed: changedFeatures(x, p.x),
      })),
      ...(method === 'closest' && freeze ? { immutable: IMMUTABLE } : {}),
      scales,
    })
  }, [method, count, freeze, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <Choice label="method" value={method} options={['closest', 'diverse'] as const} onChange={setMethod} />
        {method === 'diverse' ? (
          <NumberField label="how many" value={count} onChange={setCount} min={1} max={6} integer />
        ) : (
          <Toggle label="age and tenure frozen" value={freeze} onChange={setFreeze} />
        )}
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="What would turn the decision" />
    </div>
  )
}
