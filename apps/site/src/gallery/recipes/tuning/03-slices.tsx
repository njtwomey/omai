import { type ReactNode, useMemo, useState } from 'react'
import { choice, dict, real } from 'whitebox-engine/space'
import { functionObjective, randomSampler, tpeSampler, tune } from 'whitebox-engine/tune'
import { sliceChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Slice plots',
  question: 'How do I see the objective against each hyperparameter?',
  explain:
    '`sliceChart` puts one panel per parameter, every trial’s value against that parameter, coloured by trial number (later trials yellower) on one shared objective axis. Log-scaled parameters get log axes and choices show their names. Under TPE, the late (yellow) trials crowd into the good region.',
}

const FIG = figure({ width: 720, aspect: 2.6 })
const space = dict({ lr: real(1e-4, 1, { log: true }), dropout: real(0, 0.7), act: choice(['relu', 'tanh', 'gelu']) })
const loss = functionObjective((p) => {
  const { lr, dropout, act } = p as { lr: number; dropout: number; act: string }
  return 0.3 + 0.25 * (Math.log10(lr) + 2.3) ** 2 + 1.2 * (dropout - 0.25) ** 2 + (act === 'tanh' ? 0.08 : 0)
})

export default function SlicesRecipe(): ReactNode {
  const mode = useColourMode()
  const [sampler, setSampler] = useState<'random' | 'TPE'>('TPE')
  const [trials, setTrials] = useState(60)
  const study = useMemo(
    () =>
      tune(loss, space, { sampler: sampler === 'TPE' ? tpeSampler() : randomSampler(), nTrials: trials, seed: 7 })
        .study,
    [sampler, trials],
  )
  const option = useMemo(
    () => sliceChart(study, { mode, figure: FIG, label: 'validation loss', log: true }),
    [study, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice label="sampler" value={sampler} options={['random', 'TPE'] as const} onChange={setSampler} />
        <NumberField label="trials" value={trials} onChange={setTrials} min={10} max={120} step={10} />
      </ControlPanel>
      <FigureFrame figure={FIG} filename="slices">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
