import { choice, dict, int, real } from 'omai-engine/space'
import { correlationImportance, functionObjective, randomSampler, tune } from 'omai-engine/tune'
import { importanceChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Parameter importance',
  question: 'How do I show which hyperparameters mattered most?',
  explain:
    '`importanceChart` draws importances (here `correlationImportance` from omai-engine’s `tune`, by rank or linear correlation) as bars, largest on top, each labelled with its share. The made-up loss depends on the learning rate and dropout and hardly at all on the batch size; correlations see only monotone trends, so with fewer trials the U-shaped effects can trade places.',
}

const FIG = figure({ width: 560, aspect: 2.2 })
const space = dict({
  lr: real(1e-4, 1, { log: true }),
  dropout: real(0, 0.7),
  batch: int(16, 256, { log: true }),
  act: choice(['relu', 'tanh', 'gelu']),
})
const loss = functionObjective((p) => {
  const { lr, dropout, batch, act } = p as { lr: number; dropout: number; batch: number; act: string }
  const tail = 0.002 * Math.log2(batch) + (act === 'tanh' ? 0.05 : 0)
  return 0.3 + 0.25 * (Math.log10(lr) + 2.3) ** 2 + 1.2 * (dropout - 0.25) ** 2 + tail
})

export default function ImportanceRecipe(): ReactNode {
  const mode = useColourMode()
  const [method, setMethod] = useState<'spearman' | 'pearson'>('spearman')
  const [trials, setTrials] = useState(80)
  const study = useMemo(() => tune(loss, space, { sampler: randomSampler(), nTrials: trials, seed: 3 }).study, [trials])
  const option = useMemo(
    () => importanceChart(correlationImportance(study, { method }), { mode, figure: FIG }),
    [study, method, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="correlation" value={method} options={['spearman', 'pearson'] as const} onChange={setMethod} />
        <NumberField label="trials" value={trials} onChange={setTrials} min={10} max={200} step={10} />
      </ControlPanel>
      <FigureFrame figure={FIG} filename="importance">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
