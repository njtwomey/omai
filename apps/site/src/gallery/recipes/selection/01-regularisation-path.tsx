import { recordsOf } from 'omai-engine/exposition'
import { lassoPath, ridgePath } from 'omai-engine/linear'
import { Choice, ControlPanel, ExpositionView, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { sparseRegression } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Regularisation path',
  question: 'How do I show every coefficient as the penalty grows, and which features survive?',
  explain:
    'A `regularisationPath` record holds the coefficients at each penalty of a decreasing grid, the features still active (lasso) and the solver’s duality gap. `PathView` draws one line per feature against the penalty: the lasso’s lines hit zero one at a time and stay, so the order they leave is an importance ranking; the ridge’s only shrink. The truth has three non-zero coefficients among eight features. More noise sends the true ones to zero earlier and lets the false ones linger.',
  shows: ['regularisationPath', 'PathView'],
}

export default function RegularisationPathRecipe(): ReactNode {
  const [kind, setKind] = useState<'lasso' | 'ridge'>('lasso')
  const [noise, setNoise] = useState(1)
  const [n, setN] = useState(80)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const data = sparseRegression(seed, n, 8, noise)
    const grid = Array.from({ length: 40 }, (_, i) => 10 ** (2 - (5 * i) / 39))
    const r =
      kind === 'lasso'
        ? lassoPath(data)
        : ridgePath(
            data,
            grid.map((a) => a * 10),
          )
    return recordsOf(r.exposition, 'regularisationPath')[0]
  }, [kind, noise, n, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
        <Choice label="penalty" value={kind} options={['lasso', 'ridge'] as const} onChange={setKind} />
        <NumberField label="noise" value={noise} onChange={setNoise} min={0} max={6} step={0.25} />
        <NumberField label="rows" value={n} onChange={setN} min={20} max={400} step={10} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      {record !== undefined && <ExpositionView record={record} title="Coefficients along the penalty" />}
    </div>
  )
}
